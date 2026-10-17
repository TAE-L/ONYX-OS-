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
    apic::send_self_ipi(idt::TEST_IPI_VECTOR);
    x86_64::instructions::interrupts::enable();
    let mut spins = 0u64;
    while idt::test_ipis() == before && spins < 50_000_000 {
        core::hint::spin_loop();
        spins += 1;
    }
    x86_64::instructions::interrupts::disable();
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
