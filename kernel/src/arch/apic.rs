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
