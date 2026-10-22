//! Scheduler (Phase 2a: one CPU, per-CPU structure ready for SMP).
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

use crate::arch::{apic, time};
use crate::perf::{Histogram, rdtsc};
use crate::sync::IrqMutex;
use alloc::{boxed::Box, collections::BTreeMap, collections::BTreeSet, vec};
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
use x86_64::instructions::interrupts;

pub type TaskId = u64;

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
    cpu_ns: u64,
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
    tasks: BTreeMap<TaskId, Box<Task>>,
    current: TaskId,
    idle: TaskId,
    sleepers: BTreeSet<(u64, TaskId)>,
    v_floor: u64,
    last_account: u64,
    next_id: TaskId,
    dl_ppm: u64,
}

/// Target-time to the moment the woken task is running again, in ns.
pub static WAKE_LATENCY: Histogram = Histogram::new();
/// Target-time to the timer interrupt entry, in ns (the hardware share).
pub static IRQ_LATENCY: Histogram = Histogram::new();
static SWITCHES: AtomicU64 = AtomicU64::new(0);
static TIMER_IRQS: AtomicU64 = AtomicU64::new(0);

static RQ: IrqMutex<Option<Rq>> = IrqMutex::new(None);

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
        t.cpu_ns += dt_ns;
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
                    .filter(|t| t.runnable() && t.id != cur.id)
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
        if next == old {
            self.task(old).state = State::Running;
            return None;
        }
        let new_rsp = {
            let n = self.task(next);
            n.state = State::Running;
            if let Some(target) = n.wake_target.take() {
                WAKE_LATENCY.record(time::tsc_to_ns(rdtsc().saturating_sub(target)));
            }
            n.rsp
        };
        self.current = next;
        SWITCHES.fetch_add(1, Relaxed);
        let old_slot: *mut u64 = &mut self.task(old).rsp;
        Some((old_slot, new_rsp))
    }

    /// `rearm` evaluated as if `next` were already current.
    fn rearm_for(&mut self, next: TaskId, now_tsc: u64, now_ns: u64) {
        let saved = self.current;
        self.current = next;
        self.rearm(now_tsc, now_ns);
        self.current = saved;
    }
}

fn with_rq<R>(f: impl FnOnce(&mut Rq) -> R) -> R {
    f(RQ.lock().as_mut().expect("scheduler not initialised"))
}

/// Run `f` (which may change task states) then switch if the scheduler wants to.
/// Interrupts are off for the whole decision and the switch itself.
fn switch_after(f: impl FnOnce(&mut Rq)) {
    let was_enabled = interrupts::are_enabled();
    interrupts::disable();
    let sw = with_rq(|rq| {
        f(rq);
        rq.reschedule()
    });
    if let Some((old_slot, new_rsp)) = sw {
        // SAFETY: both stacks are live task stacks; interrupts are off.
        unsafe { switch_context(old_slot, new_rsp) };
    }
    if was_enabled {
        interrupts::enable();
    }
}

// --- Public API ----------------------------------------------------------------

/// Turn the running boot context into task 0 and create the idle task.
pub fn init() {
    let now_ns = time::now_ns();
    let mut rq = Rq {
        tasks: BTreeMap::new(),
        current: 0,
        idle: 1,
        sleepers: BTreeSet::new(),
        v_floor: 0,
        last_account: rdtsc(),
        next_id: 2,
        dl_ppm: 0,
    };
    rq.tasks.insert(
        0,
        Box::new(Task {
            id: 0,
            name: "boot",
            rsp: 0,
            stack: None,
            state: State::Running,
            class: Class::fair(),
            vruntime: 0,
            vdeadline: DEFAULT_SLICE_NS,
            lag: 0,
            dl_start_ns: now_ns,
            dl_budget_ns: 0,
            wake_target: None,
            cpu_ns: 0,
        }),
    );
    let (stack, rsp) = new_stack(idle_entry, 0);
    rq.tasks.insert(
        1,
        Box::new(Task {
            id: 1,
            name: "idle",
            rsp,
            stack: Some(stack),
            state: State::Ready,
            class: Class::Idle,
            vruntime: 0,
            vdeadline: 0,
            lag: 0,
            dl_start_ns: now_ns,
            dl_budget_ns: 0,
            wake_target: None,
            cpu_ns: 0,
        }),
    );
    *RQ.lock() = Some(rq);
    kprintln!("[sched] EEVDF + deadline class, tickless, 1 CPU");
}

pub fn spawn(
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
    let (stack, rsp) = new_stack(entry, arg);
    let now_ns = time::now_ns();
    let id = interrupts::without_interrupts(|| {
        with_rq(|rq| {
            let mut dl_budget = 0;
            if let Class::Deadline {
                runtime_ns,
                period_ns,
            } = class
            {
                let ppm = runtime_ns * 1_000_000 / period_ns;
                if rq.dl_ppm + ppm > DEADLINE_CAP_PPM {
                    return Err(SpawnError::OverCommitted);
                }
                rq.dl_ppm += ppm;
                dl_budget = runtime_ns;
            }
            let id = rq.next_id;
            rq.next_id += 1;
            rq.tasks.insert(
                id,
                Box::new(Task {
                    id,
                    name,
                    rsp,
                    stack: Some(stack),
                    state: State::Sleeping,
                    class,
                    vruntime: 0,
                    vdeadline: 0,
                    lag: 0,
                    dl_start_ns: now_ns,
                    dl_budget_ns: dl_budget,
                    wake_target: None,
                    cpu_ns: 0,
                }),
            );
            rq.make_ready(id);
            Ok(id)
        })
    })?;
    // Let the scheduler decide: a deadline task preempts at once, a fair one
    // runs when EEVDF says so (and the timer is armed for that turn).
    switch_after(|_| {});
    Ok(id)
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
    interrupts::without_interrupts(|| with_rq(|rq| rq.current))
}

/// CPU time a task has used so far, or `None` once it has exited and been reaped.
pub fn cpu_ns(id: TaskId) -> Option<u64> {
    interrupts::without_interrupts(|| with_rq(|rq| rq.tasks.get(&id).map(|t| t.cpu_ns)))
}

/// True once the task has exited (it may not have been reaped yet).
pub fn has_exited(id: TaskId) -> bool {
    interrupts::without_interrupts(|| {
        with_rq(|rq| rq.tasks.get(&id).is_none_or(|t| t.state == State::Zombie))
    })
}

pub fn task_count() -> usize {
    interrupts::without_interrupts(|| with_rq(|rq| rq.tasks.len()))
}

pub fn timer_interrupts() -> u64 {
    TIMER_IRQS.load(Relaxed)
}

pub fn context_switches() -> u64 {
    SWITCHES.load(Relaxed)
}

pub fn deadline_reserved_ppm() -> u64 {
    interrupts::without_interrupts(|| with_rq(|rq| rq.dl_ppm))
}

/// Called by the IDT on the timer vector, interrupts off, EOI already sent.
pub fn timer_interrupt(entered_tsc: u64) {
    TIMER_IRQS.fetch_add(1, Relaxed);
    let sw = {
        let mut guard = RQ.lock();
        let Some(rq) = guard.as_mut() else {
            return;
        };
        rq.wake_due(rdtsc(), entered_tsc);
        rq.reschedule()
    };
    if let Some((old_slot, new_rsp)) = sw {
        // SAFETY: both stacks are live task stacks; interrupts are off.
        unsafe { switch_context(old_slot, new_rsp) };
    }
}
