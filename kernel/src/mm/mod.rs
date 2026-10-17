//! Memory management: physical frames, page tables / address spaces, heap.

pub mod frame;
pub mod heap;
pub mod paging;

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub const PAGE_SIZE: u64 = 4096;

static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);

pub fn set_hhdm(offset: u64) {
    HHDM_OFFSET.store(offset, Relaxed);
}

pub fn hhdm() -> u64 {
    HHDM_OFFSET.load(Relaxed)
}

/// Kernel virtual address of a physical address through the direct map.
pub fn phys_to_virt(phys: u64) -> *mut u8 {
    (hhdm() + phys) as *mut u8
}

pub const fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

pub const fn align_down(value: u64, align: u64) -> u64 {
    value & !(align - 1)
}
