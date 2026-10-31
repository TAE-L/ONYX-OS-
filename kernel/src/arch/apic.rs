//! Legacy PIC shutdown and the local APIC.
//!
//! x2APIC (MSR interface) is used when the CPU has it, which every modern
//! machine does. Otherwise the classic xAPIC memory-mapped interface is used;
//! that is also the only mode QEMU's software CPU emulation offers, so
//! supporting it keeps the OS testable everywhere.

use crate::arch::{cpu, idt::SPURIOUS_VECTOR};
use crate::mm::paging;
use spin::Once;
use x86_64::instructions::port::Port;
use x86_64::registers::model_specific::Msr;

const IA32_APIC_BASE: u32 = 0x1B;
const APIC_ENABLE: u64 = 1 << 11;
const X2APIC_ENABLE: u64 = 1 << 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    X2Apic,
    XApic { base: u64 },
}

static MODE: Once<Mode> = Once::new();

pub fn mode() -> Mode {
    *MODE.get().expect("APIC not initialised")
}

// Register indices as xAPIC byte offsets; x2APIC MSR = 0x800 + offset / 16.
const REG_ID: u32 = 0x20;
const REG_VERSION: u32 = 0x30;
const REG_TPR: u32 = 0x80;
const REG_EOI: u32 = 0xB0;
const REG_SVR: u32 = 0xF0;
const REG_ICR_LOW: u32 = 0x300;
const X2APIC_SELF_IPI: u32 = 0x83F;
const REG_LVT_TIMER: u32 = 0x320;
const REG_TIMER_INITIAL: u32 = 0x380;
const REG_TIMER_CURRENT: u32 = 0x390;
const REG_TIMER_DIVIDE: u32 = 0x3E0;
const IA32_TSC_DEADLINE: u32 = 0x6E0;
const LVT_MASKED: u64 = 1 << 16;
const LVT_TSC_DEADLINE: u64 = 0b10 << 17;

fn read(reg: u32) -> u64 {
    match mode() {
        // SAFETY: valid x2APIC MSR once x2APIC mode is enabled.
        Mode::X2Apic => unsafe { Msr::new(0x800 + (reg >> 4)).read() },
        // SAFETY: `base` is the uncached mapping of the APIC page.
        Mode::XApic { base } => unsafe {
            u64::from(core::ptr::read_volatile(
                (base + u64::from(reg)) as *const u32,
            ))
        },
    }
}

fn write(reg: u32, value: u64) {
    match mode() {
        // SAFETY: as above.
        Mode::X2Apic => unsafe { Msr::new(0x800 + (reg >> 4)).write(value) },
        // SAFETY: as above.
        Mode::XApic { base } => unsafe {
            core::ptr::write_volatile((base + u64::from(reg)) as *mut u32, value as u32);
        },
    }
}

/// Remap the 8259 PICs out of the exception range, then mask every line.
fn disable_pic() {
    // SAFETY: standard 8259 initialisation sequence.
    unsafe {
        let mut cmd1 = Port::<u8>::new(0x20);
        let mut data1 = Port::<u8>::new(0x21);
        let mut cmd2 = Port::<u8>::new(0xA0);
        let mut data2 = Port::<u8>::new(0xA1);
        cmd1.write(0x11);
        cmd2.write(0x11);
        data1.write(0x20); // master offset
        data2.write(0x28); // slave offset
        data1.write(4);
        data2.write(2);
        data1.write(0x01);
        data2.write(0x01);
        data1.write(0xFF);
        data2.write(0xFF);
    }
}

/// Bring this CPU's local APIC up in the mode the boot CPU chose.
pub fn init_ap() {
    // SAFETY: reading the APIC base MSR.
    let base_msr = unsafe { Msr::new(IA32_APIC_BASE).read() };
    // SAFETY: the same documented enable sequence as the boot CPU.
    unsafe {
        Msr::new(IA32_APIC_BASE).write(base_msr | APIC_ENABLE);
        if mode() == Mode::X2Apic {
            Msr::new(IA32_APIC_BASE).write(base_msr | APIC_ENABLE | X2APIC_ENABLE);
        }
    }
    write(REG_TPR, 0);
    write(REG_SVR, 0x100 | u64::from(SPURIOUS_VECTOR));
}

/// Send a fixed, edge-triggered IPI to the CPU with local APIC id `dest`.
pub fn send_ipi(dest: u32, vector: u8) {
    x86_64::instructions::interrupts::without_interrupts(|| match mode() {
        // SAFETY: documented ICR MSR.
        Mode::X2Apic => unsafe {
            Msr::new(0x830).write((u64::from(dest) << 32) | u64::from(vector));
        },
        Mode::XApic { .. } => {
            write(0x310, u64::from(dest) << 24);
            write(REG_ICR_LOW, u64::from(vector));
            // Delivery-status bit: wait until the APIC has accepted it.
            while read(REG_ICR_LOW) & (1 << 12) != 0 {
                core::hint::spin_loop();
            }
        }
    });
}

pub fn init() {
    disable_pic();
    // SAFETY: reading the APIC base MSR.
    let base_msr = unsafe { Msr::new(IA32_APIC_BASE).read() };
    let phys = base_msr & 0x000F_FFFF_FFFF_F000;
    let mode = if cpu::features().x2apic {
        // The architecture forbids jumping from "disabled" straight to x2APIC:
        // enable xAPIC first, then switch the mode.
        // SAFETY: writing the documented enable bits.
        unsafe {
            Msr::new(IA32_APIC_BASE).write(base_msr | APIC_ENABLE);
            Msr::new(IA32_APIC_BASE).write(base_msr | APIC_ENABLE | X2APIC_ENABLE);
        }
        Mode::X2Apic
    } else {
        // SAFETY: as above, without the x2APIC bit.
        unsafe { Msr::new(IA32_APIC_BASE).write(base_msr | APIC_ENABLE) };
        let base = paging::map_mmio(phys, 0x1000).expect("cannot map the APIC page");
        Mode::XApic { base }
    };
    MODE.call_once(|| mode);
    write(REG_TPR, 0);
    write(REG_SVR, 0x100 | u64::from(SPURIOUS_VECTOR));
    kprintln!(
        "[apic] {} enabled: id={} version={:#x}",
        match mode {
            Mode::X2Apic => "x2APIC",
            Mode::XApic { .. } => "xAPIC (MMIO)",
        },
        id(),
        version()
    );
}

pub fn id() -> u32 {
    match mode() {
        Mode::X2Apic => read(REG_ID) as u32,
        Mode::XApic { .. } => (read(REG_ID) >> 24) as u32,
    }
}

pub fn version() -> u32 {
    (read(REG_VERSION) & 0xFF) as u32
}

pub fn eoi() {
    write(REG_EOI, 0);
}

pub fn send_self_ipi(vector: u8) {
    match mode() {
        // SAFETY: documented self-IPI MSR.
        Mode::X2Apic => unsafe { Msr::new(X2APIC_SELF_IPI).write(u64::from(vector)) },
        // Fixed delivery, destination shorthand "self" (bits 19:18 = 01).
        Mode::XApic { .. } => write(REG_ICR_LOW, (1 << 18) | u64::from(vector)),
    }
}

// --- Timer ---------------------------------------------------------------------
//
// Preferred: TSC-deadline mode (one absolute TSC value, no calibration, no
// rounding). Fallback: the LAPIC one-shot countdown, calibrated against the TSC.
// Either way the timer is one-shot: nothing fires unless the scheduler asked.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimerKind {
    TscDeadline,
    OneShot { hz: u64 },
}

static TIMER: Once<TimerKind> = Once::new();

pub fn timer_kind() -> TimerKind {
    *TIMER.get().expect("APIC timer not initialised")
}

fn tsc_deadline_supported() -> bool {
    // CPUID.01H:ECX bit 24.
    let ecx = core::arch::x86_64::__cpuid(1).ecx;
    !cfg!(feature = "timer-oneshot") && ecx & (1 << 24) != 0
}

/// Per-CPU timer register setup for an already chosen kind.
pub fn timer_init_ap() {
    use crate::arch::idt::TIMER_VECTOR;
    match timer_kind() {
        TimerKind::TscDeadline => write(REG_LVT_TIMER, LVT_TSC_DEADLINE | u64::from(TIMER_VECTOR)),
        TimerKind::OneShot { .. } => {
            write(REG_TIMER_DIVIDE, 0b1011);
            write(REG_LVT_TIMER, u64::from(TIMER_VECTOR));
        }
    }
}

pub fn timer_init() {
    use crate::arch::{idt::TIMER_VECTOR, time};
    let kind = if tsc_deadline_supported() {
        write(REG_LVT_TIMER, LVT_TSC_DEADLINE | u64::from(TIMER_VECTOR));
        TimerKind::TscDeadline
    } else {
        // Divide by 1, one-shot, masked while calibrating.
        write(REG_TIMER_DIVIDE, 0b1011);
        write(REG_LVT_TIMER, LVT_MASKED | u64::from(TIMER_VECTOR));
        write(REG_TIMER_INITIAL, u64::from(u32::MAX));
        let t0 = crate::perf::rdtsc();
        let span = time::tsc_hz() / 50; // 20 ms
        while crate::perf::rdtsc() - t0 < span {
            core::hint::spin_loop();
        }
        let elapsed = u64::from(u32::MAX) - read(REG_TIMER_CURRENT);
        write(REG_TIMER_INITIAL, 0);
        let hz = elapsed * 50;
        assert!(hz > 1_000_000, "LAPIC timer calibration failed");
        write(REG_LVT_TIMER, u64::from(TIMER_VECTOR)); // unmasked, one-shot
        TimerKind::OneShot { hz }
    };
    TIMER.call_once(|| kind);
    kprintln!("[apic] timer: {kind:?}");
}

/// Fire the timer interrupt at absolute TSC value `deadline`.
pub fn timer_arm_tsc(deadline: u64) {
    match timer_kind() {
        TimerKind::TscDeadline => {
            if let Mode::XApic { .. } = mode() {
                // The LVT write must be globally visible before the deadline MSR.
                // SAFETY: a plain fence.
                unsafe { core::arch::x86_64::_mm_mfence() };
            }
            // SAFETY: documented MSR; the LVT timer is in TSC-deadline mode.
            unsafe { Msr::new(IA32_TSC_DEADLINE).write(deadline.max(1)) };
        }
        TimerKind::OneShot { hz } => {
            let now = crate::perf::rdtsc();
            let delta = deadline.saturating_sub(now).max(1);
            let ticks = (u128::from(delta) * u128::from(hz)
                / u128::from(crate::arch::time::tsc_hz()))
            .clamp(1, u128::from(u32::MAX)) as u64;
            write(REG_TIMER_INITIAL, ticks);
        }
    }
}

pub fn timer_cancel() {
    match timer_kind() {
        // SAFETY: writing zero disarms TSC-deadline mode.
        TimerKind::TscDeadline => unsafe { Msr::new(IA32_TSC_DEADLINE).write(0) },
        TimerKind::OneShot { .. } => write(REG_TIMER_INITIAL, 0),
    }
}
