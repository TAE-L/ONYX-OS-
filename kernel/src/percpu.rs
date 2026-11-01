//! Per-CPU data, reached through the GS segment base.
//!
//! In the kernel GS base points at this CPU's [`PerCpu`]; while user code runs
//! the two GS bases are swapped (`swapgs`), so the kernel one sits in
//! `KERNEL_GS_BASE`. `cpu_index()` is a single `mov` from `gs:[0]`. It is only
//! stable while the caller cannot be moved to another CPU: with interrupts
//! off, or in code that does not cross a scheduling point. Never cache it
//! across `yield`/`sleep`.

use alloc::boxed::Box;
use x86_64::VirtAddr;
use x86_64::registers::model_specific::{GsBase, KernelGsBase};
use x86_64::structures::tss::TaskStateSegment;

/// Field offsets are used by the syscall entry assembly: keep them in sync.
#[repr(C)]
pub struct PerCpu {
    /// gs:[0]
    index: u64,
    /// gs:[8] — top of the kernel stack of the task running here (SYSCALL entry).
    kernel_rsp: u64,
    /// gs:[16] — scratch for the user RSP during SYSCALL entry.
    user_rsp: u64,
    /// gs:[24] — address of this block.
    self_ptr: u64,
    pub lapic_id: u32,
    tss: *mut TaskStateSegment,
}

/// Install this CPU's data block. Call once per CPU, after the heap exists.
pub fn install(index: usize, lapic_id: u32, tss: *mut TaskStateSegment) {
    let block = Box::leak(Box::new(PerCpu {
        index: index as u64,
        kernel_rsp: 0,
        user_rsp: 0,
        self_ptr: 0,
        lapic_id,
        tss,
    }));
    block.self_ptr = core::ptr::from_mut(block) as u64;
    GsBase::write(VirtAddr::new(block.self_ptr));
    // The user's GS base (what `swapgs` swaps in) starts out empty.
    KernelGsBase::write(VirtAddr::zero());
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

/// Where interrupts from ring 3 and SYSCALL will land on this CPU. The
/// scheduler calls this with interrupts off whenever it switches tasks.
pub fn set_kernel_stack(top: u64) {
    let block: u64;
    // SAFETY: gs:[24] holds the address of this CPU's leaked `PerCpu`.
    unsafe {
        core::arch::asm!("mov {}, gs:[24]", out(reg) block, options(nostack, preserves_flags, readonly));
        let block = &mut *(block as *mut PerCpu);
        block.kernel_rsp = top;
        (*block.tss).privilege_stack_table[0] = VirtAddr::new(top);
    }
}
