//! In-kernel self-tests, run by `cargo xtask test` (boot with `--features selftest`).
//!
//! Output protocol (parsed by xtask):
//!   `[PASS] name` / `[FAIL] name: reason` per test, then
//!   `SELFTEST RESULT: N passed, M failed`, then exit through QEMU's debug-exit.
//!
//! Tests must check *behaviour observed from outside the code under test*
//! (hardware faults, page-table walks, CPU registers), not just round-trips.

use crate::arch::{apic, cpu, idt};
use crate::mm::paging::{self, AddressSpace, MapError, PageSize, flags};
use crate::mm::{self, PAGE_SIZE, frame, phys_to_virt};
use crate::perf::{self, Histogram};
use crate::{boot, qemu};
use alloc::{boxed::Box, string::String, vec::Vec};
use core::sync::atomic::AtomicU64;
use x86_64::structures::paging::PageTableFlags as F;

type TestResult = Result<(), &'static str>;

macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err($msg);
        }
    };
}

struct Test {
    name: &'static str,
    run: fn() -> TestResult,
}

static TESTS: &[Test] = &[
    Test {
        name: "cpu: NX enabled in EFER",
        run: cpu_nx_enabled,
    },
    Test {
        name: "cpu: SMEP/SMAP active when the CPU has them",
        run: cpu_hardening,
    },
    Test {
        name: "perf: TSC is monotonic",
        run: perf_tsc_monotonic,
    },
    Test {
        name: "perf: histogram percentiles",
        run: perf_histogram,
    },
    Test {
        name: "frame: alloc/free round trip, distinct, above 1 MiB",
        run: frame_roundtrip,
    },
    Test {
        name: "frame: 2 MiB-aligned contiguous allocation",
        run: frame_contiguous,
    },
    Test {
        name: "heap: Box, Vec growth, String",
        run: heap_basic,
    },
    Test {
        name: "idt: breakpoint exception is delivered and returns",
        run: idt_breakpoint,
    },
    Test {
        name: "apic: version register readable",
        run: apic_registers,
    },
    Test {
        name: "apic: self-IPI reaches the IDT handler",
        run: apic_self_ipi,
    },
    Test {
        name: "paging: kernel .text is R-X at the right physical address",
        run: paging_text,
    },
    Test {
        name: "paging: .rodata is R-- and .data is RW-",
        run: paging_data_rodata,
    },
    Test {
        name: "paging: boot stack guard page is unmapped",
        run: paging_stack_guard,
    },
    Test {
        name: "paging: direct map translates to the same physical frame",
        run: paging_hhdm,
    },
    Test {
        name: "paging: kernel root has no lower-half mappings",
        run: paging_lower_half_empty,
    },
    Test {
        name: "paging: W^X audit is clean and RWX mappings are refused",
        run: paging_wx,
    },
    Test {
        name: "paging: map/unmap 4K and 2M pages, errors on misuse",
        run: paging_map_unmap,
    },
    Test {
        name: "ISOLATION: same VA, different data per address space",
        run: isolation_gate,
    },
    Test {
        name: "isolation: address spaces free every table frame on drop",
        run: isolation_no_leak,
    },
    Test {
        name: "time: TSC calibrated and timer kind as configured",
        run: time_calibrated,
    },
    Test {
        name: "sched: sleep_ns wakes on time, not early",
        run: sched_sleep_accuracy,
    },
    Test {
        name: "sched: tickless, a 60 ms sleep costs a handful of timer interrupts",
        run: sched_tickless,
    },
    Test {
        name: "sched: callee-saved registers survive task switches",
        run: sched_registers,
    },
    Test {
        name: "sched: tasks start with interrupts enabled",
        run: sched_interrupts_on,
    },
    Test {
        name: "sched: two busy tasks that never yield both get ~equal CPU (preemption)",
        run: sched_fair_equal,
    },
    Test {
        name: "sched: weight 3:1 gives ~3:1 CPU",
        run: sched_fair_weighted,
    },
    Test {
        name: "sched: a waking short-slice task preempts a hog (p90 wake latency < hog slice)",
        run: sched_wake_latency,
    },
    Test {
        name: "sched: deadline task gets its reserved share, hog gets the rest",
        run: sched_deadline_share,
    },
    Test {
        name: "sched: deadline admission refuses over-commitment and releases on exit",
        run: sched_deadline_admission,
    },
    Test {
        name: "sched: 100 short tasks leave no heap or task leak",
        run: sched_no_leak,
    },
    Test {
        name: "smp: every CPU the bootloader reported is online",
        run: smp_online,
    },
    Test {
        name: "smp: a task pinned to each CPU runs on that CPU (own GS data, own APIC id)",
        run: smp_pinned,
    },
    Test {
        name: "smp: each CPU's own timer wakes its sleepers",
        run: smp_timers,
    },
    Test {
        name: "smp: one busy task per CPU runs in parallel",
        run: smp_parallel,
    },
    Test {
        name: "smp: a task sent to an idle CPU starts promptly (reschedule IPI)",
        run: smp_ipi_wake,
    },
    Test {
        name: "smp: an overloaded CPU's tasks spread out when other CPUs go idle",
        run: smp_rebalance,
    },
    Test {
        name: "smp: heap and IrqMutex stay consistent under all-CPU contention",
        run: smp_lock_stress,
    },
];

pub fn run() -> ! {
    kprintln!("SELFTEST BEGIN ({} tests)", TESTS.len());
    let mut failed = 0;
    for t in TESTS {
        match (t.run)() {
            Ok(()) => kprintln!("[PASS] {}", t.name),
            Err(why) => {
                failed += 1;
                kprintln!("[FAIL] {}: {}", t.name, why);
            }
        }
    }
    kprintln!(
        "SELFTEST RESULT: {} passed, {} failed",
        TESTS.len() - failed,
        failed
    );
    qemu::exit(if failed == 0 { qemu::PASS } else { qemu::FAIL })
}

// --- cpu / perf --------------------------------------------------------------

fn cpu_nx_enabled() -> TestResult {
    use x86_64::registers::model_specific::{Efer, EferFlags};
    ensure!(
        Efer::read().contains(EferFlags::NO_EXECUTE_ENABLE),
        "EFER.NXE is clear"
    );
    Ok(())
}

fn cpu_hardening() -> TestResult {
    use x86_64::registers::control::{Cr4, Cr4Flags};
    let f = cpu::features();
    let cr4 = Cr4::read();
    ensure!(
        !f.smep || cr4.contains(Cr4Flags::SUPERVISOR_MODE_EXECUTION_PROTECTION),
        "CPU has SMEP but it is off"
    );
    ensure!(
        !f.smap || cr4.contains(Cr4Flags::SUPERVISOR_MODE_ACCESS_PREVENTION),
        "CPU has SMAP but it is off"
    );
    Ok(())
}

fn perf_tsc_monotonic() -> TestResult {
    let start = perf::rdtsc();
    let mut last = start;
    for _ in 0..10_000 {
        let now = perf::rdtsc();
        ensure!(now >= last, "TSC went backwards");
        last = now;
    }
    ensure!(last > start, "TSC did not advance");
    Ok(())
}

fn perf_histogram() -> TestResult {
    let h = Histogram::new();
    ensure!(
        h.count() == 0 && h.percentile(50) == 0,
        "empty histogram not zero"
    );
    for _ in 0..1000 {
        h.record(100); // bit length 7 -> bucket 7, upper bound 127
    }
    h.record(100_000); // bit length 17 -> upper bound 131071
    h.record(0);
    ensure!(h.count() == 1002, "count wrong");
    ensure!(h.min() == 0 && h.max() == 100_000, "min/max wrong");
    ensure!(h.percentile(50) == 127, "p50 wrong");
    ensure!(h.percentile(100) == 131_071, "p100 wrong");
    h.record(u64::MAX);
    ensure!(h.percentile(100) == u64::MAX, "top bucket wrong");
    Ok(())
}

// --- frames / heap -------------------------------------------------------------

fn frame_roundtrip() -> TestResult {
    let before = frame::free_frames();
    let mut got = [0u64; 64];
    for slot in got.iter_mut() {
        *slot = frame::alloc_frame().ok_or("out of frames")?;
        ensure!(*slot >= 1 << 20, "frame below 1 MiB handed out");
        ensure!(slot.is_multiple_of(PAGE_SIZE), "unaligned frame");
    }
    ensure!(
        before - frame::free_frames() == 64,
        "free count did not drop by 64"
    );
    for i in 0..got.len() {
        for j in i + 1..got.len() {
            ensure!(got[i] != got[j], "same frame handed out twice");
        }
    }
    for f in got {
        frame::free_frame(f);
    }
    ensure!(frame::free_frames() == before, "free count not restored");
    Ok(())
}

fn frame_contiguous() -> TestResult {
    let before = frame::free_frames();
    let pa = frame::alloc_contiguous(512, 512).ok_or("no 2 MiB run")?;
    ensure!(pa.is_multiple_of(2 << 20), "not 2 MiB aligned");
    ensure!(before - frame::free_frames() == 512, "free count wrong");
    frame::free_contiguous(pa, 512);
    ensure!(frame::free_frames() == before, "free count not restored");
    Ok(())
}

fn heap_basic() -> TestResult {
    let b = Box::new(0xDEAD_BEEF_u64);
    ensure!(*b == 0xDEAD_BEEF, "box value");
    let mut v: Vec<u64> = Vec::new();
    for i in 0..100_000 {
        v.push(i);
    }
    ensure!(v.iter().sum::<u64>() == 99_999 * 100_000 / 2, "vec sum");
    let mut s = String::new();
    for _ in 0..1000 {
        s.push_str("onyx");
    }
    ensure!(s.len() == 4000, "string length");
    drop(v);
    Ok(())
}

// --- interrupts ------------------------------------------------------------------

fn idt_breakpoint() -> TestResult {
    let before = idt::breakpoints();
    // SAFETY: int3 is handled by our IDT and returns.
    unsafe { core::arch::asm!("int3") };
    ensure!(idt::breakpoints() == before + 1, "breakpoint not counted");
    Ok(())
}

fn apic_registers() -> TestResult {
    ensure!(apic::version() != 0, "APIC version reads as zero");
    let _ = apic::id();
    Ok(())
}

fn apic_self_ipi() -> TestResult {
    let before = idt::test_ipis();
    let was_enabled = x86_64::instructions::interrupts::are_enabled();
    apic::send_self_ipi(idt::TEST_IPI_VECTOR);
    x86_64::instructions::interrupts::enable();
    let mut spins = 0u64;
    while idt::test_ipis() == before && spins < 50_000_000 {
        core::hint::spin_loop();
        spins += 1;
    }
    if !was_enabled {
        x86_64::instructions::interrupts::disable();
    }
    ensure!(idt::test_ipis() == before + 1, "self-IPI never arrived");
    Ok(())
}

// --- paging ------------------------------------------------------------------------

static RODATA_MARK: u64 = 0x5EED_0001;
static DATA_MARK: AtomicU64 = AtomicU64::new(1);

fn paging_text() -> TestResult {
    let info = boot::info();
    let va = paging_text as *const () as u64;
    let t = paging::kernel()
        .lock()
        .translate(va)
        .ok_or(".text not mapped")?;
    ensure!(
        t.phys == va - info.kernel_virt + info.kernel_phys,
        ".text maps to the wrong physical address"
    );
    ensure!(!t.flags.contains(F::WRITABLE), ".text is writable");
    ensure!(!t.flags.contains(F::NO_EXECUTE), ".text is not executable");
    Ok(())
}

fn paging_data_rodata() -> TestResult {
    let k = paging::kernel().lock();
    let ro = k
        .translate(&raw const RODATA_MARK as u64)
        .ok_or(".rodata not mapped")?;
    ensure!(!ro.flags.contains(F::WRITABLE), ".rodata is writable");
    ensure!(ro.flags.contains(F::NO_EXECUTE), ".rodata is executable");
    let data = k
        .translate(&raw const DATA_MARK as u64)
        .ok_or(".data not mapped")?;
    ensure!(data.flags.contains(F::WRITABLE), ".data is read-only");
    ensure!(data.flags.contains(F::NO_EXECUTE), ".data is executable");
    Ok(())
}

fn paging_stack_guard() -> TestResult {
    let guard = paging::boot_stack_guard();
    ensure!(
        paging::kernel().lock().translate(guard).is_none(),
        "stack guard page is mapped"
    );
    ensure!(
        idt::try_read_u8(guard).is_none(),
        "stack guard page is readable"
    );
    Ok(())
}

fn paging_hhdm() -> TestResult {
    let pa = frame::alloc_zeroed().ok_or("no frame")?;
    let va = phys_to_virt(pa) as u64;
    let t = paging::kernel()
        .lock()
        .translate(va)
        .ok_or("direct map hole");
    frame::free_frame(pa);
    let t = t?;
    ensure!(t.phys == pa, "direct map points at the wrong frame");
    ensure!(t.flags.contains(F::NO_EXECUTE), "direct map is executable");
    Ok(())
}

fn paging_lower_half_empty() -> TestResult {
    ensure!(
        paging::kernel().lock().user_entries_present() == 0,
        "kernel root still maps the lower half"
    );
    Ok(())
}

fn paging_wx() -> TestResult {
    let (leaves, bad) = paging::kernel().lock().audit_wx();
    ensure!(leaves > 0, "audit saw no mappings");
    ensure!(bad.is_none(), "kernel mappings contain a W+X page");
    let mut space = AddressSpace::new_user().ok_or("no address space")?;
    let rwx = F::PRESENT | F::WRITABLE | F::USER_ACCESSIBLE;
    let r = space.map(0x1000, 0x20_0000, PageSize::Size4K, rwx);
    ensure!(
        r == Err(MapError::WritableExecutable),
        "RWX mapping was accepted"
    );
    Ok(())
}

fn paging_map_unmap() -> TestResult {
    let mut space = AddressSpace::new_user().ok_or("no address space")?;
    let pa = frame::alloc_zeroed().ok_or("no frame")?;
    const VA: u64 = 0x0000_1000_0000_0000;
    space
        .map(VA, pa, PageSize::Size4K, flags::user_rw())
        .map_err(|_| "map failed")?;
    let t = space.translate(VA + 0x123).ok_or("translate failed")?;
    ensure!(
        t.phys == pa + 0x123 && t.size == PageSize::Size4K,
        "bad 4K translation"
    );
    ensure!(
        space.map(VA, pa, PageSize::Size4K, flags::user_rw()) == Err(MapError::AlreadyMapped),
        "double map accepted"
    );
    ensure!(
        space.map(VA + 1, pa, PageSize::Size4K, flags::user_rw()) == Err(MapError::Misaligned),
        "misaligned map accepted"
    );
    ensure!(
        space.map(
            0x0000_9000_0000_0000,
            pa,
            PageSize::Size4K,
            flags::user_rw()
        ) == Err(MapError::NonCanonical),
        "non-canonical map accepted"
    );
    ensure!(
        space.unmap(VA).map_err(|_| "unmap failed")?.phys == pa,
        "unmap returned wrong frame"
    );
    ensure!(space.translate(VA).is_none(), "still mapped after unmap");
    ensure!(
        matches!(space.unmap(VA), Err(MapError::NotMapped)),
        "double unmap accepted"
    );
    frame::free_frame(pa);

    // A 2 MiB page.
    let big = frame::alloc_contiguous(512, 512).ok_or("no 2 MiB run")?;
    const BIG_VA: u64 = 0x0000_2000_0020_0000;
    space
        .map(BIG_VA, big, PageSize::Size2M, flags::user_rw())
        .map_err(|_| "2M map failed")?;
    let t = space
        .translate(BIG_VA + 0x1_2345)
        .ok_or("2M translate failed")?;
    ensure!(
        t.size == PageSize::Size2M && t.phys == big + 0x1_2345,
        "bad 2M translation"
    );
    ensure!(
        space.map(BIG_VA, big, PageSize::Size4K, flags::user_rw()) == Err(MapError::ParentIsHuge),
        "4K map inside a 2M page accepted"
    );
    space.unmap(BIG_VA).map_err(|_| "2M unmap failed")?;
    frame::free_contiguous(big, 512);
    Ok(())
}

// --- isolation: the Phase 1 gate ---------------------------------------------------------

/// Two address spaces map the *same* virtual address to *different* frames.
/// Reading that address with each space's CR3 loaded must return each space's
/// own byte; a third space and the kernel space (where nothing is mapped)
/// must take a real hardware page fault.
fn isolation_gate() -> TestResult {
    const VA: u64 = 0x0000_1000_0000_0000;
    let fa = frame::alloc_zeroed().ok_or("no frame")?;
    let fb = frame::alloc_zeroed().ok_or("no frame")?;
    // SAFETY: both frames are ours and reachable through the direct map.
    unsafe {
        phys_to_virt(fa).write(0xA5);
        phys_to_virt(fb).write(0x5B);
    }
    let mut a = AddressSpace::new_user().ok_or("no address space")?;
    let mut b = AddressSpace::new_user().ok_or("no address space")?;
    let c = AddressSpace::new_user().ok_or("no address space")?;
    a.map(VA, fa, PageSize::Size4K, flags::user_rw())
        .map_err(|_| "map A")?;
    b.map(VA, fb, PageSize::Size4K, flags::user_rw())
        .map_err(|_| "map B")?;

    let faults_before = idt::probe_faults();
    a.activate();
    let in_a = idt::try_read_u8(VA);
    b.activate();
    let in_b = idt::try_read_u8(VA);
    c.activate();
    let in_c = idt::try_read_u8(VA);
    paging::kernel().lock().activate();
    let in_kernel = idt::try_read_u8(VA);
    let faults = idt::probe_faults() - faults_before;

    drop((a, b, c));
    frame::free_frame(fa);
    frame::free_frame(fb);

    ensure!(in_a == Some(0xA5), "space A did not read its own byte");
    ensure!(in_b == Some(0x5B), "space B did not read its own byte");
    ensure!(
        in_c.is_none(),
        "space C could read an address it never mapped"
    );
    ensure!(in_kernel.is_none(), "kernel space can see a user mapping");
    ensure!(faults == 2, "expected exactly two hardware page faults");
    Ok(())
}

fn isolation_no_leak() -> TestResult {
    let before = frame::free_frames();
    for _ in 0..50 {
        let mut s = AddressSpace::new_user().ok_or("no address space")?;
        let pa = frame::alloc_zeroed().ok_or("no frame")?;
        // Spread mappings so several table levels are allocated.
        s.map(0x1000, pa, PageSize::Size4K, flags::user_rw())
            .map_err(|_| "map")?;
        s.map(
            0x0000_4000_0000_0000,
            pa,
            PageSize::Size4K,
            flags::user_rw(),
        )
        .map_err(|_| "map")?;
        drop(s);
        frame::free_frame(pa);
    }
    ensure!(frame::free_frames() == before, "table frames leaked");
    let _ = mm::PAGE_SIZE;
    Ok(())
}

// --- time / scheduler ----------------------------------------------------------------

use crate::arch::time;
use crate::sched::{self, Class, SpawnError};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

static STOP: AtomicBool = AtomicBool::new(false);
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn hog(slot: usize) {
    while !STOP.load(SeqCst) {
        COUNTS[slot].fetch_add(1, SeqCst);
    }
}

fn wait_exit(ids: &[sched::TaskId]) {
    for &id in ids {
        let mut spins = 0;
        while !sched::has_exited(id) && spins < 5000 {
            sched::sleep_ns(1_000_000);
            spins += 1;
        }
    }
}

fn time_calibrated() -> TestResult {
    let hz = time::tsc_hz();
    ensure!(
        (100_000_000..=10_000_000_000).contains(&hz),
        "TSC frequency implausible"
    );
    let kind = apic::timer_kind();
    #[cfg(feature = "timer-oneshot")]
    ensure!(
        matches!(kind, apic::TimerKind::OneShot { .. }),
        "expected the one-shot fallback timer"
    );
    #[cfg(not(feature = "timer-oneshot"))]
    let _ = kind;
    kprintln!("       timer = {:?}, tsc = {} MHz", kind, hz / 1_000_000);
    Ok(())
}

fn sched_sleep_accuracy() -> TestResult {
    let mut worst = 0;
    for _ in 0..5 {
        let t0 = perf::rdtsc();
        sched::sleep_ns(10_000_000);
        let elapsed = time::tsc_to_ns(perf::rdtsc() - t0);
        ensure!(elapsed >= 10_000_000, "woke before the requested time");
        worst = worst.max(elapsed - 10_000_000);
    }
    kprintln!("       worst oversleep over 5 sleeps: {} us", worst / 1000);
    ensure!(worst < 20_000_000, "oversleep above 20 ms");
    Ok(())
}

fn sched_tickless() -> TestResult {
    let before = sched::timer_interrupts();
    sched::sleep_ns(60_000_000);
    let n = sched::timer_interrupts() - before;
    kprintln!("       timer interrupts during a 60 ms sleep: {n}");
    ensure!(n >= 1, "no timer interrupt woke the sleeper");
    ensure!(n <= 4, "timer is ticking while nothing needs it");
    Ok(())
}

core::arch::global_asm!(
    r#"
    .section .text.selftest,"ax"
    .global ctx_regs_check
    .hidden ctx_regs_check
// ctx_regs_check(fn [rdi]) -> eax: 0 if every callee-saved register survived `fn`
ctx_regs_check:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov rbx, 0x1111111111111111
    mov r12, 0x1212121212121212
    mov r13, 0x1313131313131313
    mov r14, 0x1414141414141414
    mov r15, 0x1515151515151515
    mov rbp, 0x1616161616161616
    call rdi
    mov eax, 1
    mov rcx, 0x1111111111111111
    cmp rbx, rcx
    jne 2f
    mov rcx, 0x1212121212121212
    cmp r12, rcx
    jne 2f
    mov rcx, 0x1313131313131313
    cmp r13, rcx
    jne 2f
    mov rcx, 0x1414141414141414
    cmp r14, rcx
    jne 2f
    mov rcx, 0x1515151515151515
    cmp r15, rcx
    jne 2f
    mov rcx, 0x1616161616161616
    cmp rbp, rcx
    jne 2f
    xor eax, eax
2:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
"#
);

unsafe extern "C" {
    fn ctx_regs_check(f: extern "C" fn()) -> u32;
}

extern "C" fn yield_c() {
    sched::yield_now();
}

static REG_BAD: AtomicU64 = AtomicU64::new(0);
static REG_DONE: AtomicU64 = AtomicU64::new(0);

fn reg_checker(_: usize) {
    for _ in 0..300 {
        // SAFETY: the asm only calls `yield_c` and compares registers.
        if unsafe { ctx_regs_check(yield_c) } != 0 {
            REG_BAD.fetch_add(1, SeqCst);
        }
    }
    REG_DONE.fetch_add(1, SeqCst);
}

fn reg_clobberer(_: usize) {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    while REG_DONE.load(SeqCst) < 2 {
        for _ in 0..50 {
            x = x.rotate_left(7) ^ x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            core::hint::black_box(x);
        }
        sched::yield_now();
    }
}

fn sched_registers() -> TestResult {
    REG_BAD.store(0, SeqCst);
    REG_DONE.store(0, SeqCst);
    let ids = [
        sched::spawn("reg-a", Class::fair(), reg_checker, 0).map_err(|_| "spawn")?,
        sched::spawn("reg-b", Class::fair(), reg_checker, 0).map_err(|_| "spawn")?,
        sched::spawn("clobber", Class::fair(), reg_clobberer, 0).map_err(|_| "spawn")?,
    ];
    wait_exit(&ids);
    ensure!(REG_DONE.load(SeqCst) == 2, "checkers did not finish");
    ensure!(
        REG_BAD.load(SeqCst) == 0,
        "a callee-saved register was corrupted"
    );
    Ok(())
}

static IF_SEEN: AtomicUsize = AtomicUsize::new(2);

fn if_probe(_: usize) {
    IF_SEEN.store(
        usize::from(x86_64::instructions::interrupts::are_enabled()),
        SeqCst,
    );
}

fn sched_interrupts_on() -> TestResult {
    IF_SEEN.store(2, SeqCst);
    let id = sched::spawn("if-probe", Class::fair(), if_probe, 0).map_err(|_| "spawn")?;
    wait_exit(&[id]);
    ensure!(
        IF_SEEN.load(SeqCst) == 1,
        "new task ran with interrupts disabled"
    );
    Ok(())
}

fn run_hogs(classes: &[Class], window_ms: u64) -> Result<[u64; 4], &'static str> {
    STOP.store(false, SeqCst);
    for c in &COUNTS {
        c.store(0, SeqCst);
    }
    let mut ids = alloc::vec::Vec::new();
    for (slot, class) in classes.iter().enumerate() {
        ids.push(sched::spawn_on(Some(0), "hog", *class, hog, slot).map_err(|_| "spawn failed")?);
    }
    sched::sleep_ns(window_ms * 1_000_000);
    let mut cpu = [0u64; 4];
    for (slot, id) in ids.iter().enumerate() {
        cpu[slot] = sched::cpu_ns(*id).ok_or("task vanished")?;
    }
    STOP.store(true, SeqCst);
    wait_exit(&ids);
    Ok(cpu)
}

fn sched_fair_equal() -> TestResult {
    let cpu = run_hogs(&[Class::fair(), Class::fair()], 300)?;
    kprintln!(
        "       cpu ms: {} / {}",
        cpu[0] / 1_000_000,
        cpu[1] / 1_000_000
    );
    ensure!(
        COUNTS[0].load(SeqCst) > 0 && COUNTS[1].load(SeqCst) > 0,
        "a busy task never ran: no preemption"
    );
    let (a, b) = (cpu[0].max(1), cpu[1].max(1));
    ensure!(
        a * 10 > b * 7 && b * 10 > a * 7,
        "CPU split is not within 30% of equal"
    );
    Ok(())
}

fn sched_fair_weighted() -> TestResult {
    let cpu = run_hogs(
        &[
            Class::fair_with(1024, 3_000_000),
            Class::fair_with(3072, 3_000_000),
        ],
        400,
    )?;
    kprintln!(
        "       cpu ms: {} / {} (weights 1:3)",
        cpu[0] / 1_000_000,
        cpu[1] / 1_000_000
    );
    let ratio_x100 = cpu[1] * 100 / cpu[0].max(1);
    ensure!(
        (220..=400).contains(&ratio_x100),
        "weight 3 did not get about 3x the CPU"
    );
    Ok(())
}

static LAT: Histogram = Histogram::new();

fn latency_sleeper(_: usize) {
    for _ in 0..50 {
        let target = perf::rdtsc() + time::ns_to_tsc(2_000_000);
        sched::sleep_until_tsc(target);
        LAT.record(time::tsc_to_ns(perf::rdtsc() - target));
    }
}

fn sched_wake_latency() -> TestResult {
    STOP.store(false, SeqCst);
    let hog_id = sched::spawn_on(Some(0), "hog", Class::fair_with(1024, 3_000_000), hog, 0)
        .map_err(|_| "spawn")?;
    let id = sched::spawn_on(
        Some(0),
        "lat",
        Class::fair_with(1024, 300_000),
        latency_sleeper,
        0,
    )
    .map_err(|_| "spawn")?;
    wait_exit(&[id]);
    STOP.store(true, SeqCst);
    wait_exit(&[hog_id]);
    ensure!(LAT.count() == 50, "sleeper did not complete 50 wakeups");
    let (p50, p99, max) = (LAT.percentile(50), LAT.percentile(99), LAT.max());
    kprintln!(
        "       wake latency vs a running hog: p50<={} us p99<={} us max={} us (software emulation: not a hardware number)",
        p50 / 1000,
        p99 / 1000,
        max / 1000
    );
    ensure!(
        LAT.percentile(90) < 3_000_000,
        "p90 wake latency exceeded the hog's slice: no wake preemption"
    );
    ensure!(
        sched::WAKE_LATENCY.count() >= 50,
        "global wake histogram not fed"
    );
    ensure!(
        sched::context_switches() > 100,
        "too few context switches recorded"
    );
    Ok(())
}

fn sched_deadline_share() -> TestResult {
    // 2 ms every 10 ms = 20%.
    let cpu = run_hogs(
        &[
            Class::Deadline {
                runtime_ns: 2_000_000,
                period_ns: 10_000_000,
            },
            Class::fair(),
        ],
        400,
    )?;
    kprintln!(
        "       cpu ms: deadline {} / fair hog {} (of 400, reserved 20%)",
        cpu[0] / 1_000_000,
        cpu[1] / 1_000_000
    );
    ensure!(
        (60_000_000..=100_000_000).contains(&cpu[0]),
        "deadline task did not get ~20% (80 ms of 400)"
    );
    ensure!(cpu[1] > 200_000_000, "the fair task was starved");
    Ok(())
}

fn sched_deadline_admission() -> TestResult {
    STOP.store(false, SeqCst);
    let base = sched::deadline_reserved_ppm();
    let a = sched::spawn_on(
        Some(0),
        "dl-a",
        Class::Deadline {
            runtime_ns: 6_000_000,
            period_ns: 10_000_000,
        },
        hog,
        0,
    )
    .map_err(|_| "60% should be accepted")?;
    let refused = sched::spawn_on(
        Some(0),
        "dl-b",
        Class::Deadline {
            runtime_ns: 3_000_000,
            period_ns: 10_000_000,
        },
        hog,
        1,
    );
    ensure!(
        refused == Err(SpawnError::OverCommitted),
        "90% total was accepted"
    );
    ensure!(
        sched::deadline_reserved_ppm() == base + 600_000,
        "refused task left a reservation behind"
    );
    STOP.store(true, SeqCst);
    wait_exit(&[a]);
    sched::sleep_ns(2_000_000);
    ensure!(
        sched::deadline_reserved_ppm() == base,
        "reservation not released on exit"
    );
    Ok(())
}

fn nop_task(_: usize) {}

fn sched_no_leak() -> TestResult {
    sched::sleep_ns(3_000_000); // reap anything left by earlier tests
    let (tasks0, heap0) = (sched::task_count(), mm::heap::used());
    for _ in 0..100 {
        let id = sched::spawn("nop", Class::fair(), nop_task, 0).map_err(|_| "spawn")?;
        wait_exit(&[id]);
    }
    sched::sleep_ns(3_000_000);
    let (tasks1, heap1) = (sched::task_count(), mm::heap::used());
    ensure!(tasks1 == tasks0, "exited tasks were not reaped");
    ensure!(heap1 <= heap0 + 8192, "heap grew after 100 task lifecycles");
    Ok(())
}

// --- SMP -----------------------------------------------------------------------------

use crate::percpu::cpu_index;

fn online_list() -> alloc::vec::Vec<usize> {
    (0..sched::MAX_CPUS)
        .filter(|&c| sched::is_online(c))
        .collect()
}

fn smp_online() -> TestResult {
    let expected = crate::limine::MP
        .response()
        .map_or(1, |r| r.cpus().len().min(sched::MAX_CPUS));
    kprintln!(
        "       {} CPU(s) online of {} reported",
        sched::online_cpus(),
        expected
    );
    ensure!(
        sched::online_cpus() == expected,
        "not every CPU came online"
    );
    ensure!(sched::is_online(0), "boot CPU not online");
    Ok(())
}

static SEEN_CPU: [AtomicU64; sched::MAX_CPUS] =
    [const { AtomicU64::new(u64::MAX) }; sched::MAX_CPUS];
static SEEN_APIC: [AtomicU64; sched::MAX_CPUS] =
    [const { AtomicU64::new(u64::MAX) }; sched::MAX_CPUS];

fn report_cpu(slot: usize) {
    // Interrupts could in principle move us between the two reads; pin makes it stable.
    SEEN_CPU[slot].store(cpu_index() as u64, SeqCst);
    SEEN_APIC[slot].store(u64::from(apic::id()), SeqCst);
}

fn smp_pinned() -> TestResult {
    let cpus = online_list();
    for c in 0..sched::MAX_CPUS {
        SEEN_CPU[c].store(u64::MAX, SeqCst);
        SEEN_APIC[c].store(u64::MAX, SeqCst);
    }
    let mut ids = alloc::vec::Vec::new();
    for &c in &cpus {
        ids.push(
            sched::spawn_on(Some(c), "pinned", Class::fair(), report_cpu, c)
                .map_err(|_| "spawn")?,
        );
    }
    wait_exit(&ids);
    let mut apics = alloc::vec::Vec::new();
    for &c in &cpus {
        ensure!(
            SEEN_CPU[c].load(SeqCst) == c as u64,
            "task ran on the wrong CPU"
        );
        apics.push(SEEN_APIC[c].load(SeqCst));
    }
    apics.sort_unstable();
    apics.dedup();
    ensure!(
        apics.len() == cpus.len(),
        "two CPUs report the same APIC id"
    );
    Ok(())
}

static TIMER_BAD: AtomicU64 = AtomicU64::new(0);

fn timer_checker(_: usize) {
    for _ in 0..5 {
        let t0 = perf::rdtsc();
        sched::sleep_ns(5_000_000);
        let elapsed = time::tsc_to_ns(perf::rdtsc() - t0);
        if !(5_000_000..=40_000_000).contains(&elapsed) {
            TIMER_BAD.fetch_add(1, SeqCst);
        }
    }
}

fn smp_timers() -> TestResult {
    TIMER_BAD.store(0, SeqCst);
    let mut ids = alloc::vec::Vec::new();
    for c in online_list() {
        ids.push(
            sched::spawn_on(Some(c), "timer", Class::fair(), timer_checker, 0)
                .map_err(|_| "spawn")?,
        );
    }
    wait_exit(&ids);
    ensure!(
        TIMER_BAD.load(SeqCst) == 0,
        "a sleeper woke early or far too late on some CPU"
    );
    Ok(())
}

fn smp_parallel() -> TestResult {
    let cpus = online_list();
    STOP.store(false, SeqCst);
    let mut ids = alloc::vec::Vec::new();
    for (slot, &c) in cpus.iter().enumerate().take(4) {
        ids.push(sched::spawn_on(Some(c), "par", Class::fair(), hog, slot).map_err(|_| "spawn")?);
    }
    let window = 200_000_000u64;
    sched::sleep_ns(window);
    let total: u64 = ids.iter().filter_map(|&i| sched::cpu_ns(i)).sum();
    STOP.store(true, SeqCst);
    wait_exit(&ids);
    kprintln!(
        "       {} busy tasks used {} ms of CPU in a {} ms window",
        ids.len(),
        total / 1_000_000,
        window / 1_000_000
    );
    ensure!(
        total * 100 >= window * ids.len() as u64 * 80,
        "busy tasks on separate CPUs did not run in parallel"
    );
    Ok(())
}

static STARTED_AT: AtomicU64 = AtomicU64::new(0);

fn stamp_start(_: usize) {
    STARTED_AT.store(perf::rdtsc(), SeqCst);
}

fn smp_ipi_wake() -> TestResult {
    let remote: alloc::vec::Vec<usize> = online_list().into_iter().filter(|&c| c != 0).collect();
    if remote.is_empty() {
        return Ok(()); // single-CPU machine: nothing to send an IPI to
    }
    let lat = Histogram::new();
    for i in 0..20 {
        let cpu = remote[i % remote.len()];
        sched::sleep_ns(2_000_000); // let the target CPU go idle and halt
        STARTED_AT.store(0, SeqCst);
        let t0 = perf::rdtsc();
        let id = sched::spawn_on(Some(cpu), "ipi", Class::fair(), stamp_start, 0)
            .map_err(|_| "spawn")?;
        let mut waited = 0;
        while STARTED_AT.load(SeqCst) == 0 && waited < 200 {
            sched::sleep_ns(100_000);
            waited += 1;
        }
        ensure!(
            STARTED_AT.load(SeqCst) != 0,
            "task on an idle CPU never started"
        );
        lat.record(time::tsc_to_ns(STARTED_AT.load(SeqCst) - t0));
        wait_exit(&[id]);
    }
    kprintln!(
        "       spawn-to-run on an idle remote CPU: p50<={} us max={} us (software emulation)",
        lat.percentile(50) / 1000,
        lat.max() / 1000
    );
    ensure!(
        lat.max() < 5_000_000,
        "idle CPU took more than 5 ms to start a task"
    );
    Ok(())
}

fn smp_rebalance() -> TestResult {
    let cpus = online_list();
    if cpus.len() < 2 {
        return Ok(());
    }
    // Per-slot stop flags: fillers keep every other CPU busy while two
    // unpinned tasks pile up on CPU 0 (placement prefers the spawning CPU on ties).
    static FILLER_STOP: AtomicBool = AtomicBool::new(false);
    fn filler(_: usize) {
        while !FILLER_STOP.load(SeqCst) {
            core::hint::spin_loop();
        }
    }
    STOP.store(false, SeqCst);
    FILLER_STOP.store(false, SeqCst);
    let mut fillers = alloc::vec::Vec::new();
    for &c in cpus.iter().filter(|&&c| c != 0).flat_map(|c| [c, c]) {
        fillers.push(
            sched::spawn_on(Some(c), "filler", Class::fair(), filler, 0).map_err(|_| "spawn")?,
        );
    }
    sched::sleep_ns(5_000_000);
    // CPU 0 holds this test task (load 1); every other CPU holds two fillers (load 2),
    // so the next two unpinned tasks both land on CPU 0 (ties go to the spawning CPU).
    let a = sched::spawn("work-a", Class::fair(), hog, 0).map_err(|_| "spawn")?;
    let b = sched::spawn("work-b", Class::fair(), hog, 1).map_err(|_| "spawn")?;
    sched::sleep_ns(20_000_000);
    FILLER_STOP.store(true, SeqCst);
    wait_exit(&fillers);
    let window = 300_000_000u64;
    let (a0, b0) = (
        sched::cpu_ns(a).ok_or("gone")?,
        sched::cpu_ns(b).ok_or("gone")?,
    );
    sched::sleep_ns(window);
    let (da, db) = (
        sched::cpu_ns(a).ok_or("gone")? - a0,
        sched::cpu_ns(b).ok_or("gone")? - b0,
    );
    STOP.store(true, SeqCst);
    wait_exit(&[a, b]);
    kprintln!(
        "       after the other CPUs went idle, the two tasks got {} ms and {} ms of {} ms",
        da / 1_000_000,
        db / 1_000_000,
        window / 1_000_000
    );
    ensure!(
        da * 100 >= window * 70 && db * 100 >= window * 70,
        "tasks stayed stacked on one CPU while others idled"
    );
    Ok(())
}

static SHARED: crate::sync::IrqMutex<u64> = crate::sync::IrqMutex::new(0);
static STRESS_BAD: AtomicU64 = AtomicU64::new(0);
const STRESS_ROUNDS: u64 = 3000;

fn stress(seed: usize) {
    let mut x = 0x1234_5678_9ABC_DEF1u64 ^ (seed as u64).wrapping_mul(0x9E37_79B9);
    for round in 0..STRESS_ROUNDS {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let len = (x % 200 + 1) as usize;
        let mut v: alloc::vec::Vec<u64> = alloc::vec::Vec::with_capacity(len);
        for i in 0..len {
            v.push(round ^ i as u64 ^ seed as u64);
        }
        if (0..len).any(|i| v[i] != round ^ i as u64 ^ seed as u64) {
            STRESS_BAD.fetch_add(1, SeqCst);
        }
        *SHARED.lock() += 1;
    }
}

fn smp_lock_stress() -> TestResult {
    *SHARED.lock() = 0;
    STRESS_BAD.store(0, SeqCst);
    let heap0 = mm::heap::used();
    let cpus = online_list();
    let mut ids = alloc::vec::Vec::new();
    for &c in &cpus {
        ids.push(
            sched::spawn_on(Some(c), "stress", Class::fair(), stress, c).map_err(|_| "spawn")?,
        );
    }
    wait_exit(&ids);
    sched::sleep_ns(3_000_000);
    ensure!(STRESS_BAD.load(SeqCst) == 0, "heap memory was corrupted");
    ensure!(
        *SHARED.lock() == STRESS_ROUNDS * cpus.len() as u64,
        "lost updates under the shared lock"
    );
    ensure!(
        mm::heap::used() <= heap0 + 8192,
        "heap leaked during the stress run"
    );
    Ok(())
}
