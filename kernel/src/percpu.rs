//! Per-CPU data, reached through the GS segment base.
//!
//! `cpu_index()` is a single `mov` from `gs:[0]`. It is only stable while the
//! caller cannot be moved to another CPU: with interrupts off, or in code that
//! does not cross a scheduling point. Never cache it across `yield`/`sleep`.

use alloc::boxed::Box;
use x86_64::VirtAddr;
use x86_64::registers::model_specific::GsBase;

#[repr(C)]
pub struct PerCpu {
    /// Must stay first: read by `cpu_index` as `gs:[0]`.
    index: u64,
    pub lapic_id: u32,
}

/// Install this CPU's data block. Call once per CPU, after the heap exists.
pub fn install(index: usize, lapic_id: u32) {
    let block = Box::leak(Box::new(PerCpu {
        index: index as u64,
        lapic_id,
    }));
    GsBase::write(VirtAddr::from_ptr(core::ptr::from_ref(block)));
}

#[inline]
pub fn cpu_index() -> usize {
    let index: u64;
    // SAFETY: `install` set GS base to a leaked `PerCpu` whose first word is the index.
    unsafe {
        core::arch::asm!("mov {}, gs:[0]", out(reg) index, options(nostack, preserves_flags, readonly));
    }
    index as usize
}
