//! CPU feature detection and the protections the kernel switches on at boot.

use core::arch::x86_64::{__cpuid, __cpuid_count};
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};
use x86_64::registers::model_specific::{Efer, EferFlags};

pub struct Features {
    pub nx: bool,
    pub gib_pages: bool,
    pub apic: bool,
    pub x2apic: bool,
    pub smep: bool,
    pub smap: bool,
    pub umip: bool,
    pub invariant_tsc: bool,
}

pub fn features() -> Features {
    let (l1, l7, e1, e7) = (
        __cpuid(1),
        __cpuid_count(7, 0),
        __cpuid(0x8000_0001),
        __cpuid(0x8000_0007),
    );
    Features {
        nx: e1.edx & (1 << 20) != 0,
        gib_pages: e1.edx & (1 << 26) != 0,
        apic: l1.edx & (1 << 9) != 0,
        x2apic: l1.ecx & (1 << 21) != 0,
        smep: l7.ebx & (1 << 7) != 0,
        smap: l7.ebx & (1 << 20) != 0,
        umip: l7.ecx & (1 << 2) != 0,
        invariant_tsc: e7.edx & (1 << 8) != 0,
    }
}

pub fn init() {
    init_quiet(false);
}

pub fn init_quiet(quiet: bool) {
    let f = features();
    assert!(f.nx, "CPU lacks NX support; OnyxOS requires it");
    assert!(f.apic, "CPU lacks a local APIC; OnyxOS requires it");

    // SAFETY: enabling write-protect, NX and hardening bits we verified exist.
    unsafe {
        Cr0::update(|c| c.insert(Cr0Flags::WRITE_PROTECT));
        Efer::update(|e| e.insert(EferFlags::NO_EXECUTE_ENABLE));
        Cr4::update(|c| {
            c.insert(Cr4Flags::PAGE_GLOBAL);
            if f.smep {
                c.insert(Cr4Flags::SUPERVISOR_MODE_EXECUTION_PROTECTION);
            }
            if f.smap {
                c.insert(Cr4Flags::SUPERVISOR_MODE_ACCESS_PREVENTION);
            }
            if f.umip {
                c.insert(Cr4Flags::USER_MODE_INSTRUCTION_PREVENTION);
            }
        });
    }
    if quiet {
        return;
    }
    kprintln!(
        "[cpu] nx=on smep={} smap={} umip={} 1gib-pages={} invariant-tsc={}",
        f.smep,
        f.smap,
        f.umip,
        f.gib_pages,
        f.invariant_tsc
    );
    if !f.invariant_tsc {
        kprintln!("[cpu] warning: TSC is not invariant; timing will need another clock");
    }
}

/// Whether SMAP is active, i.e. kernel access to user pages needs `stac`/`clac`.
pub fn smap_enabled() -> bool {
    Cr4::read().contains(Cr4Flags::SUPERVISOR_MODE_ACCESS_PREVENTION)
}

pub fn halt_forever() -> ! {
    loop {
        x86_64::instructions::interrupts::disable();
        x86_64::instructions::hlt();
    }
}
