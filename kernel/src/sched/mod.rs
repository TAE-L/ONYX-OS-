//! Scheduler (Phase 2: per-CPU run queues, idle stealing, periodic push balancing).
//!
//! Classes, highest priority first:
//! 1. **Deadline**: a periodic budget (`runtime` every `period`) served
//!    earliest-deadline-first, admission-controlled so the rest can never starve.
//! 2. **Fair (EEVDF)**: weighted fair sharing where each task also has a
//!    *slice*; a task with a short slice gets picked sooner after waking, which
//!    is how latency-sensitive threads (input, audio, the game's render thread)
//!    are served without giving them extra CPU time.
//! 3. **Idle**: `sti; hlt`.
//!
//! Time is tickless: the local APIC timer is armed for exactly the next event
//! (a sleeper waking, the running task's slice ending, a budget running out or
//! being replenished) and is disarmed when there is none.
//!
//! Every task switch happens with interrupts off. A switch triggered by the
//! timer happens inside the interrupt handler, on the interrupted task's own
//! kernel stack; the task resumes exactly there when picked again.

use crate::arch::{apic, idt, time};
use crate::percpu::cpu_index;
use crate::perf::{Histogram, rdtsc};
use crate::sync::IrqMutex;
use alloc::{boxed::Box, collections::BTreeMap, collections::BTreeSet, sync::Arc, vec};
use core::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};
use x86_64::instructions::interrupts;

pub type TaskId = u64;
pub const MAX_CPUS: usize = 16;

const STACK_SIZE: usize = 64 * 1024;
const STACK_CANARY: u64 = 0x5AFE_57AC_C0DE_CAFE;
/// Weight of a nice-0 task; virtual time advances at real time for this weight.
pub const NICE0: u32 = 1024;
pub const DEFAULT_SLICE_NS: u64 = 3_000_000;
/// Never program the timer closer than this; avoids interrupt storms.
const MIN_TIMER_NS: u64 = 20_000;
/// At most this share (parts per million) can be promised to deadline tasks.
const DEADLINE_CAP_PPM: u64 = 800_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Fair { weight: u32, slice_ns: u64 },
    Deadline { runtime_ns: u64, period_ns: u64 },
    Idle,
}

impl Class {
    pub const fn fair() -> Self {
        Class::Fair {
            weight: NICE0,
            slice_ns: DEFAULT_SLICE_NS,
        }
    }
    pub const fn fair_with(weight: u32, slice_ns: u64) -> Self {
        Class::Fair { weight, slice_ns }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpawnError {
    /// Deadline reservations would exceed the allowed share of the CPU.
    OverCommitted,
    InvalidClass,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Ready,
    Running,
    Sleeping,
    Zombie,
}

struct Task {
    id: TaskId,
    name: &'static str,
    rsp: u64,
    stack: Option<Box<[u8]>>,
    state: State,
    class: Class,
    // EEVDF
    vruntime: u64,
    vdeadline: u64,
    lag: i64,
    // Deadline class
    dl_start_ns: u64,
    dl_budget_ns: u64,
    // Wake-latency bookkeeping: the TSC value the task asked to wake at.
    wake_target: Option<u64>,
    stats: Arc<Stats>,
    /// Only this CPU may run the task (`None` = any).
    pinned: Option<usize>,
    /// True from the moment the scheduler decides to switch away from the task
    /// until the CPU that switched to something else has fully saved its
    /// registers. Other CPUs must not steal it in that window.
    switching: AtomicBool,
}

/// Task numbers other CPUs and tests may read without touching a run queue.
struct Stats {
    /// CPU time accounted up to `since_tsc`.
    cpu_ns: AtomicU64,
    /// TSC value `cpu_ns` is current to while the task runs, 0 while it does not.
    since_tsc: AtomicU64,
    exited: AtomicBool,
}

impl Task {
    fn slice_virtual(&self) -> u64 {
        match self.class {
            Class::Fair { weight, slice_ns } => slice_ns * u64::from(NICE0) / u64::from(weight),
            _ => 0,
        }
    }
    fn runnable(&self) -> bool {
        matches!(self.state, State::Ready | State::Running)
    }
}

struct Rq {
    cpu: usize,
    tasks: BTreeMap<TaskId, Box<Task>>,
    current: TaskId,
    idle: TaskId,
    sleepers: BTreeSet<(u64, TaskId)>,
    v_floor: u64,
    last_account: u64,
    dl_ppm: u64,
}

/// What other CPUs may look at without taking this CPU's run-queue lock.
struct CpuSlot {
    rq: IrqMutex<Option<Rq>>,
    online: AtomicBool,
    /// Currently running its idle task (so it needs an IPI to notice new work).
    idle: AtomicBool,
    /// Runnable non-idle tasks, including the one running.
    load: AtomicUsize,
    dl_ppm: AtomicU64,
    lapic_id: AtomicU32,
    /// `switching` flag of the task this CPU just switched away from.
    prev_switch: AtomicPtr<AtomicBool>,
}

impl CpuSlot {
    const fn new() -> Self {
        Self {
            rq: IrqMutex::new(None),
            online: AtomicBool::new(false),
            idle: AtomicBool::new(false),
            load: AtomicUsize::new(0),
            dl_ppm: AtomicU64::new(0),
            lapic_id: AtomicU32::new(0),
            prev_switch: AtomicPtr::new(core::ptr::null_mut()),
        }
    }
}

static CPUS: [CpuSlot; MAX_CPUS] = [const { CpuSlot::new() }; MAX_CPUS];
static ONLINE: AtomicUsize = AtomicUsize::new(0);
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
static REGISTRY: IrqMutex<BTreeMap<TaskId, Arc<Stats>>> = IrqMutex::new(BTreeMap::new());

/// Target-time to the moment the woken task is running again, in ns.
pub static WAKE_LATENCY: Histogram = Histogram::new();
/// Target-time to the timer interrupt entry, in ns (the hardware share).
pub static IRQ_LATENCY: Histogram = Histogram::new();
static SWITCHES: AtomicU64 = AtomicU64::new(0);
static TIMER_IRQS: AtomicU64 = AtomicU64::new(0);

core::arch::global_asm!(
    r#"
    .section .text.sched,"ax"
    .global switch_context
    .hidden switch_context
// switch_context(old_rsp_slot: *mut u64 [rdi], new_rsp: u64 [rsi])
switch_context:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rdi], rsp
    mov rsp, rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

    .global task_trampoline
    .hidden task_trampoline
// First code a new task runs; r12 = entry, r13 = argument, rsp 16-byte aligned.
task_trampoline:
    mov rdi, r12
    mov rsi, r13
    xor ebp, ebp
    call task_bootstrap
    ud2
"#
);

unsafe extern "C" {
    fn switch_context(old_rsp_slot: *mut u64, new_rsp: u64);
    fn task_trampoline();
}

#[unsafe(no_mangle)]
extern "C" fn task_bootstrap(entry: usize, arg: usize) -> ! {
    finish_switch();
    interrupts::enable();
    // SAFETY: `entry` was created from a `fn(usize)` in `spawn`.
    let f: fn(usize) = unsafe { core::mem::transmute::<usize, fn(usize)>(entry) };
    f(arg);
    exit()
}

fn idle_entry(_: usize) {
    loop {
        // `sti` takes effect after the next instruction, so no wakeup is lost.
        // SAFETY: enabling interrupts then halting is the standard idle sequence.
        unsafe { core::arch::asm!("sti; hlt", options(nomem, nostack)) };
    }
}

/// Build a stack whose first `switch_context` lands in `task_trampoline`.
fn new_stack(entry: fn(usize), arg: usize) -> (Box<[u8]>, u64) {
    let mut stack = vec![0u8; STACK_SIZE].into_boxed_slice();
    stack[..8].copy_from_slice(&STACK_CANARY.to_le_bytes());
    let top = (stack.as_ptr() as u64 + STACK_SIZE as u64) & !15;
    let rsp = top - 56;
    let slots = rsp as *mut u64;
    // SAFETY: seven slots inside the freshly allocated stack, below `top`.
    unsafe {
        slots.add(0).write(0); // r15
        slots.add(1).write(0); // r14
        slots.add(2).write(arg as u64); // r13
        slots.add(3).write(entry as usize as u64); // r12
        slots.add(4).write(0); // rbx
        slots.add(5).write(0); // rbp
        slots
            .add(6)
            .write(task_trampoline as *const () as usize as u64); // return address
    }
    (stack, rsp)
}

impl Rq {
    fn task(&mut self, id: TaskId) -> &mut Task {
        self.tasks.get_mut(&id).expect("unknown task id")
    }

    /// Weighted mean virtual runtime of the runnable fair tasks: the EEVDF
    /// "fair share point". A task is eligible when its vruntime is not past it.
    fn avg_vruntime(&self) -> u64 {
        let (mut sum, mut weight) = (0u128, 0u128);
        for t in self.tasks.values() {
            if let Class::Fair { weight: w, .. } = t.class
                && t.runnable()
            {
                sum += u128::from(w) * u128::from(t.vruntime);
                weight += u128::from(w);
            }
        }
        sum.checked_div(weight).map_or(self.v_floor, |v| v as u64)
    }

    fn account(&mut self, now_tsc: u64) {
        let dt_ns = time::tsc_to_ns(now_tsc.wrapping_sub(self.last_account));
        self.last_account = now_tsc;
        let cur = self.current;
        let t = self.task(cur);
        t.stats.cpu_ns.fetch_add(dt_ns, Relaxed);
        t.stats.since_tsc.store(now_tsc, Relaxed);
        match t.class {
            Class::Fair { weight, .. } => {
                t.vruntime += dt_ns * u64::from(NICE0) / u64::from(weight);
                if t.vruntime >= t.vdeadline {
                    // Slice used up: ask for the next one.
                    t.vdeadline = t.vruntime + t.slice_virtual();
                }
            }
            Class::Deadline { .. } => t.dl_budget_ns = t.dl_budget_ns.saturating_sub(dt_ns),
            Class::Idle => {}
        }
    }

    fn replenish(&mut self, now_ns: u64) {
        for t in self.tasks.values_mut() {
            if let Class::Deadline {
                runtime_ns,
                period_ns,
            } = t.class
                && now_ns >= t.dl_start_ns + period_ns
            {
                let periods = (now_ns - t.dl_start_ns) / period_ns;
                t.dl_start_ns += periods * period_ns;
                t.dl_budget_ns = runtime_ns;
            }
        }
    }

    fn pick(&self) -> TaskId {
        // 1. Deadline class: earliest absolute deadline with budget left.
        let dl = self
            .tasks
            .values()
            .filter(|t| t.runnable() && t.dl_budget_ns > 0)
            .filter_map(|t| match t.class {
                Class::Deadline { period_ns, .. } => Some((t.dl_start_ns + period_ns, t.id)),
                _ => None,
            })
            .min();
        if let Some((_, id)) = dl {
            return id;
        }
        // 2. Fair class: earliest virtual deadline among the eligible.
        let v = self.avg_vruntime();
        let fair = || {
            self.tasks
                .values()
                .filter(|t| t.runnable() && matches!(t.class, Class::Fair { .. }))
        };
        if let Some(t) = fair()
            .filter(|t| t.vruntime <= v)
            .min_by_key(|t| (t.vdeadline, t.id))
        {
            return t.id;
        }
        if let Some(t) = fair().min_by_key(|t| (t.vruntime, t.id)) {
            return t.id;
        }
        self.idle
    }

    /// Make a sleeping (or new) task runnable, keeping the lag it left with.
    fn make_ready(&mut self, id: TaskId) {
        let v = self.avg_vruntime();
        let t = self.task(id);
        if matches!(t.class, Class::Fair { .. }) {
            t.vruntime = (i128::from(v) - i128::from(t.lag)).max(0) as u64;
            t.vdeadline = t.vruntime + t.slice_virtual();
        }
        t.state = State::Ready;
    }

    /// Record the lag of a fair task that is about to stop being runnable.
    fn save_lag(&mut self, id: TaskId) {
        let v = self.avg_vruntime();
        let t = self.task(id);
        if matches!(t.class, Class::Fair { .. }) {
            let limit = 2 * t.slice_virtual() as i64;
            t.lag = (v as i64 - t.vruntime as i64).clamp(-limit, limit);
        }
        self.v_floor = self.v_floor.max(v);
    }

    fn wake_due(&mut self, now_tsc: u64, entered_tsc: u64) {
        while let Some(&(target, id)) = self.sleepers.first() {
            if target > now_tsc {
                break;
            }
            self.sleepers.pop_first();
            IRQ_LATENCY.record(time::tsc_to_ns(entered_tsc.saturating_sub(target)));
            self.make_ready(id);
            self.task(id).wake_target = Some(target);
        }
    }

    fn reap(&mut self) {
        let cur = self.current;
        let dead: alloc::vec::Vec<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.state == State::Zombie && t.id != cur)
            .map(|t| t.id)
            .collect();
        for id in dead {
            self.tasks.remove(&id);
            REGISTRY.lock().remove(&id);
        }
    }

    /// Program the timer for the next thing that needs the CPU's attention.
    fn rearm(&self, now_tsc: u64, now_ns: u64) {
        let mut next_tsc: Option<u64> = self.sleepers.first().map(|&(t, _)| t);
        let mut consider = |delta_ns: u64| {
            let t = now_tsc + time::ns_to_tsc(delta_ns.max(MIN_TIMER_NS));
            next_tsc = Some(next_tsc.map_or(t, |n| n.min(t)));
        };
        let cur = &self.tasks[&self.current];
        match cur.class {
            Class::Fair { weight, .. } => {
                let others = self
                    .tasks
                    .values()
                    .filter(|t| t.runnable() && t.id != cur.id && t.id != self.idle)
                    .count();
                if others > 0 {
                    let left = cur.vdeadline.saturating_sub(cur.vruntime);
                    consider(left * u64::from(weight) / u64::from(NICE0));
                }
            }
            Class::Deadline { .. } => consider(cur.dl_budget_ns),
            Class::Idle => {}
        }
        for t in self.tasks.values() {
            if let Class::Deadline { period_ns, .. } = t.class
                && t.dl_budget_ns == 0
                && t.state != State::Zombie
                && t.state != State::Sleeping
            {
                consider((t.dl_start_ns + period_ns).saturating_sub(now_ns));
            }
        }
        match next_tsc {
            Some(t) => apic::timer_arm_tsc(t),
            None => apic::timer_cancel(),
        }
    }

    /// Runnable non-idle tasks, the running one included.
    fn runnable_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|t| t.runnable() && t.id != self.idle)
            .count()
    }

    fn publish(&self) {
        let slot = &CPUS[self.cpu];
        slot.load.store(self.runnable_count(), Relaxed);
        slot.dl_ppm.store(self.dl_ppm, Relaxed);
    }

    /// Choose the next task and return the pointers for `switch_context`,
    /// or `None` if the current task stays.
    fn reschedule(&mut self) -> Option<(*mut u64, u64)> {
        let now_tsc = rdtsc();
        self.account(now_tsc);
        let now_ns = time::now_ns();
        self.replenish(now_ns);
        self.reap();

        let old = self.current;
        {
            let t = self.task(old);
            if let Some(stack) = &t.stack {
                let mut c = [0u8; 8];
                c.copy_from_slice(&stack[..8]);
                assert!(
                    u64::from_le_bytes(c) == STACK_CANARY,
                    "kernel stack overflow in task '{}'",
                    t.name
                );
            }
            if t.state == State::Running {
                t.state = State::Ready;
            }
        }
        let next = self.pick();
        self.rearm_for(next, now_tsc, now_ns);
        CPUS[self.cpu].idle.store(next == self.idle, Relaxed);
        if next == old {
            self.task(old).state = State::Running;
            self.publish();
            return None;
        }
        self.task(old).stats.since_tsc.store(0, Relaxed);
        let new_rsp = {
            let n = self.task(next);
            n.state = State::Running;
            n.stats.since_tsc.store(now_tsc, Relaxed);
            if let Some(target) = n.wake_target.take() {
                WAKE_LATENCY.record(time::tsc_to_ns(rdtsc().saturating_sub(target)));
            }
            n.rsp
        };
        self.current = next;
        SWITCHES.fetch_add(1, Relaxed);
        let cpu = self.cpu;
        let old_task = self.task(old);
        old_task.switching.store(true, Release);
        let flag: *const AtomicBool = &old_task.switching;
        CPUS[cpu].prev_switch.store(flag.cast_mut(), Release);
        let old_slot: *mut u64 = &mut old_task.rsp;
        self.publish();
        Some((old_slot, new_rsp))
    }

    /// `rearm` evaluated as if `next` were already current.
    fn rearm_for(&mut self, next: TaskId, now_tsc: u64, now_ns: u64) {
        let saved = self.current;
        self.current = next;
        self.rearm(now_tsc, now_ns);
        self.current = saved;
    }

    /// Remove one fair task that can run on `dst` and is not running or being
    /// switched out, but only if that leaves this CPU at least one task.
    fn take_movable(&mut self, dst: usize) -> Option<Box<Task>> {
        if self.runnable_count() < 2 {
            return None;
        }
        let cur = self.current;
        let id = self
            .tasks
            .values()
            .filter(|t| {
                t.id != cur
                    && t.state == State::Ready
                    && matches!(t.class, Class::Fair { .. })
                    && !t.switching.load(Acquire)
                    && t.pinned.is_none_or(|p| p == dst)
            })
            .max_by_key(|t| (t.vdeadline, t.id))
            .map(|t| t.id)?;
        self.save_lag(id);
        let mut task = self.tasks.remove(&id)?;
        task.state = State::Sleeping;
        self.publish();
        Some(task)
    }
}

// --- Cross-CPU plumbing --------------------------------------------------------

fn slot() -> &'static CpuSlot {
    &CPUS[cpu_index()]
}

fn with_rq<R>(cpu: usize, f: impl FnOnce(&mut Rq) -> R) -> R {
    let mut guard = CPUS[cpu].rq.lock();
    f(guard.as_mut().expect("run queue not initialised"))
}

/// Mark the switch away from the previous task as complete, on the CPU that
/// just resumed. Until then other CPUs must not touch that task.
fn finish_switch() {
    let flag = slot().prev_switch.swap(core::ptr::null_mut(), Acquire);
    if !flag.is_null() {
        // SAFETY: points into a `Box<Task>` that is not freed before this store.
        unsafe { (*flag).store(false, Release) };
    }
}

/// Tell `cpu` to look at its run queue if it is not already going to.
fn kick(cpu: usize, force: bool) {
    if cpu == cpu_index() {
        return;
    }
    let s = &CPUS[cpu];
    if force || s.idle.load(Relaxed) {
        apic::send_ipi(s.lapic_id.load(Relaxed), idt::RESCHED_VECTOR);
    }
}

fn insert_ready(cpu: usize, mut task: Box<Task>) {
    task.state = State::Sleeping;
    let id = task.id;
    let force = matches!(task.class, Class::Deadline { .. });
    with_rq(cpu, |rq| {
        rq.tasks.insert(id, task);
        rq.make_ready(id);
        rq.publish();
    });
    kick(cpu, force);
}

/// An idle CPU pulls one task from the busiest CPU. Returns true if it got one.
fn steal_into(me: usize) -> bool {
    let mut victims: [(usize, usize); MAX_CPUS] = [(0, 0); MAX_CPUS];
    let mut n = 0;
    for (c, s) in CPUS.iter().enumerate() {
        if c != me && s.online.load(Relaxed) {
            let load = s.load.load(Relaxed);
            if load >= 2 {
                victims[n] = (load, c);
                n += 1;
            }
        }
    }
    victims[..n].sort_unstable_by(|a, b| b.cmp(a));
    for &(_, v) in &victims[..n] {
        if let Some(task) = with_rq(v, |rq| rq.take_movable(me)) {
            insert_ready(me, task);
            return true;
        }
    }
    false
}

/// A CPU with a backlog hands one task to a clearly emptier CPU.
fn balance_push(me: usize) {
    let mine = CPUS[me].load.load(Relaxed);
    if mine < 2 {
        return;
    }
    let target = CPUS
        .iter()
        .enumerate()
        .filter(|(c, s)| *c != me && s.online.load(Relaxed))
        .map(|(c, s)| (s.load.load(Relaxed), c))
        .min();
    if let Some((load, dst)) = target
        && load + 2 <= mine
        && let Some(task) = with_rq(me, |rq| rq.take_movable(dst))
    {
        insert_ready(dst, task);
    }
}

/// Decide and switch on this CPU. Interrupts must be off. `pre` changes the
/// current task's state first (sleep, exit); an empty queue tries to steal.
fn schedule(pre: impl FnOnce(&mut Rq)) {
    let me = cpu_index();
    let sw = {
        let mut guard = CPUS[me].rq.lock();
        let rq = guard.as_mut().expect("run queue not initialised");
        pre(rq);
        if rq.runnable_count() == 0 {
            drop(guard);
            steal_into(me);
            guard = CPUS[me].rq.lock();
        }
        guard
            .as_mut()
            .expect("run queue not initialised")
            .reschedule()
    };
    if let Some((old_slot, new_rsp)) = sw {
        // SAFETY: both stacks are live task stacks; interrupts are off.
        unsafe { switch_context(old_slot, new_rsp) };
        finish_switch();
    }
}

/// Run `f` (which may change task states) then switch if the scheduler wants to.
fn switch_after(f: impl FnOnce(&mut Rq)) {
    let was_enabled = interrupts::are_enabled();
    interrupts::disable();
    schedule(f);
    if was_enabled {
        interrupts::enable();
    }
}

// --- Public API ----------------------------------------------------------------

fn new_task(name: &'static str, class: Class, pinned: Option<usize>) -> Task {
    let now_ns = time::now_ns();
    Task {
        id: NEXT_ID.fetch_add(1, Relaxed),
        name,
        rsp: 0,
        stack: None,
        state: State::Sleeping,
        class,
        vruntime: 0,
        vdeadline: 0,
        lag: 0,
        dl_start_ns: now_ns,
        dl_budget_ns: 0,
        wake_target: None,
        stats: Arc::new(Stats {
            cpu_ns: AtomicU64::new(0),
            since_tsc: AtomicU64::new(0),
            exited: AtomicBool::new(false),
        }),
        pinned,
        switching: AtomicBool::new(false),
    }
}

fn register_cpu(cpu: usize, lapic_id: u32, rq: Rq) {
    let s = &CPUS[cpu];
    s.lapic_id.store(lapic_id, Relaxed);
    s.idle.store(true, Relaxed);
    *s.rq.lock() = Some(rq);
    s.online.store(true, Release);
    ONLINE.fetch_add(1, Relaxed);
}

/// Boot CPU: turn the running boot context into a task and create its idle task.
pub fn init(lapic_id: u32) {
    let mut boot = new_task("boot", Class::fair(), Some(0));
    boot.state = State::Running;
    boot.vdeadline = DEFAULT_SLICE_NS;
    let boot_id = boot.id;
    let mut idle = new_task("idle", Class::Idle, Some(0));
    let (stack, rsp) = new_stack(idle_entry, 0);
    idle.stack = Some(stack);
    idle.rsp = rsp;
    idle.state = State::Ready;
    let idle_id = idle.id;
    let mut rq = Rq {
        cpu: 0,
        tasks: BTreeMap::new(),
        current: boot_id,
        idle: idle_id,
        sleepers: BTreeSet::new(),
        v_floor: 0,
        last_account: rdtsc(),
        dl_ppm: 0,
    };
    rq.tasks.insert(boot_id, Box::new(boot));
    rq.tasks.insert(idle_id, Box::new(idle));
    rq.publish();
    register_cpu(0, lapic_id, rq);
    CPUS[0].idle.store(false, Relaxed);
    kprintln!("[sched] EEVDF + deadline class, tickless, per-CPU run queues");
}

/// Application processor: its boot context *is* its idle task. Never returns.
pub fn run_ap(cpu: usize, lapic_id: u32) -> ! {
    let mut idle = new_task("idle", Class::Idle, Some(cpu));
    idle.state = State::Running;
    let idle_id = idle.id;
    let mut rq = Rq {
        cpu,
        tasks: BTreeMap::new(),
        current: idle_id,
        idle: idle_id,
        sleepers: BTreeSet::new(),
        v_floor: 0,
        last_account: rdtsc(),
        dl_ppm: 0,
    };
    rq.tasks.insert(idle_id, Box::new(idle));
    register_cpu(cpu, lapic_id, rq);
    idle_entry(0);
    unreachable!()
}

pub fn online_cpus() -> usize {
    ONLINE.load(Relaxed)
}

pub fn is_online(cpu: usize) -> bool {
    cpu < MAX_CPUS && CPUS[cpu].online.load(Acquire)
}

/// Pick the CPU for a new task: the pinned one, else the least loaded.
fn choose_cpu(class: Class, pinned: Option<usize>) -> Result<usize, SpawnError> {
    if let Some(c) = pinned {
        return if is_online(c) {
            Ok(c)
        } else {
            Err(SpawnError::InvalidClass)
        };
    }
    let me = cpu_index();
    let online = || {
        CPUS.iter()
            .enumerate()
            .filter(|(_, s)| s.online.load(Acquire))
    };
    match class {
        Class::Deadline { .. } => online()
            .map(|(c, s)| (s.dl_ppm.load(Relaxed), c))
            .min()
            .map(|(_, c)| c)
            .ok_or(SpawnError::InvalidClass),
        _ => online()
            .map(|(c, s)| (s.load.load(Relaxed), usize::from(c != me), c))
            .min()
            .map(|(_, _, c)| c)
            .ok_or(SpawnError::InvalidClass),
    }
}

pub fn spawn(
    name: &'static str,
    class: Class,
    entry: fn(usize),
    arg: usize,
) -> Result<TaskId, SpawnError> {
    spawn_on(None, name, class, entry, arg)
}

/// Like [`spawn`], optionally restricted to one CPU.
pub fn spawn_on(
    pinned: Option<usize>,
    name: &'static str,
    class: Class,
    entry: fn(usize),
    arg: usize,
) -> Result<TaskId, SpawnError> {
    match class {
        Class::Idle => return Err(SpawnError::InvalidClass),
        Class::Fair { weight, slice_ns } if weight == 0 || slice_ns == 0 => {
            return Err(SpawnError::InvalidClass);
        }
        Class::Deadline {
            runtime_ns,
            period_ns,
        } if runtime_ns == 0 || runtime_ns > period_ns => return Err(SpawnError::InvalidClass),
        _ => {}
    }
    let mut task = new_task(name, class, pinned);
    let (stack, rsp) = new_stack(entry, arg);
    task.stack = Some(stack);
    task.rsp = rsp;
    let id = task.id;
    let stats = task.stats.clone();
    REGISTRY.lock().insert(id, stats);

    let placed = interrupts::without_interrupts(|| {
        let cpu = choose_cpu(class, pinned)?;
        if let Class::Deadline {
            runtime_ns,
            period_ns,
        } = class
        {
            let ppm = runtime_ns * 1_000_000 / period_ns;
            let ok = with_rq(cpu, |rq| {
                if rq.dl_ppm + ppm > DEADLINE_CAP_PPM {
                    return false;
                }
                rq.dl_ppm += ppm;
                true
            });
            if !ok {
                return Err(SpawnError::OverCommitted);
            }
            task.dl_budget_ns = runtime_ns;
            task.dl_start_ns = time::now_ns();
        }
        insert_ready(cpu, Box::new(task));
        Ok(cpu)
    });
    match placed {
        Ok(cpu) => {
            // Local placement: let EEVDF decide whether the newcomer runs now
            // (and arm the timer for its turn). Remote CPUs were kicked above.
            if cpu == cpu_index() || pinned.is_none() {
                switch_after(|_| {});
            }
            Ok(id)
        }
        Err(e) => {
            REGISTRY.lock().remove(&id);
            Err(e)
        }
    }
}

pub fn yield_now() {
    switch_after(|_| {});
}

pub fn sleep_until_tsc(target: u64) {
    if rdtsc() >= target {
        return;
    }
    switch_after(|rq| {
        let id = rq.current;
        rq.save_lag(id);
        let t = rq.task(id);
        t.state = State::Sleeping;
        t.wake_target = None;
        rq.sleepers.insert((target, id));
    });
}

pub fn sleep_ns(ns: u64) {
    sleep_until_tsc(rdtsc() + time::ns_to_tsc(ns));
}

pub fn exit() -> ! {
    switch_after(|rq| {
        let id = rq.current;
        let t = rq.task(id);
        t.state = State::Zombie;
        t.stats.exited.store(true, Release);
        if let Class::Deadline {
            runtime_ns,
            period_ns,
        } = t.class
        {
            rq.dl_ppm -= runtime_ns * 1_000_000 / period_ns;
        }
    });
    unreachable!("a zombie task was scheduled again");
}

#[allow(dead_code)] // used by Phase 3 (process ids)
pub fn current_id() -> TaskId {
    interrupts::without_interrupts(|| with_rq(cpu_index(), |rq| rq.current))
}

/// CPU time a task has used so far, or `None` once it has exited and been reaped.
pub fn cpu_ns(id: TaskId) -> Option<u64> {
    let stats = REGISTRY.lock().get(&id).cloned()?;
    let base = stats.cpu_ns.load(Relaxed);
    let since = stats.since_tsc.load(Relaxed);
    // A task that is running right now has run since its last accounting point.
    let running = if since == 0 {
        0
    } else {
        time::tsc_to_ns(rdtsc().saturating_sub(since))
    };
    Some(base + running)
}

/// True once the task has exited (it may not have been reaped yet).
pub fn has_exited(id: TaskId) -> bool {
    REGISTRY
        .lock()
        .get(&id)
        .is_none_or(|s| s.exited.load(Acquire))
}

pub fn task_count() -> usize {
    CPUS.iter()
        .filter(|s| s.online.load(Acquire))
        .map(|s| s.rq.lock().as_ref().map_or(0, |rq| rq.tasks.len()))
        .sum()
}

pub fn timer_interrupts() -> u64 {
    TIMER_IRQS.load(Relaxed)
}

pub fn context_switches() -> u64 {
    SWITCHES.load(Relaxed)
}

pub fn deadline_reserved_ppm() -> u64 {
    CPUS.iter().map(|s| s.dl_ppm.load(Relaxed)).sum()
}

/// Called by the IDT on the timer vector, interrupts off, EOI already sent.
pub fn timer_interrupt(entered_tsc: u64) {
    TIMER_IRQS.fetch_add(1, Relaxed);
    let me = cpu_index();
    if CPUS[me].rq.lock().is_none() {
        return;
    }
    balance_push(me);
    schedule(|rq| rq.wake_due(rdtsc(), entered_tsc));
}

/// Called by the IDT on the reschedule IPI, interrupts off, EOI already sent.
pub fn resched_ipi(entered_tsc: u64) {
    let me = cpu_index();
    if CPUS[me].rq.lock().is_none() {
        return;
    }
    schedule(|rq| rq.wake_due(rdtsc(), entered_tsc));
}
