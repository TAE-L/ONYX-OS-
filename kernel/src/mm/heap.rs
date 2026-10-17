//! Kernel heap: a linked-list allocator over contiguous frames in the direct
//! map. Good enough for Phase 1; Phase 2 adds per-CPU slab caches in front.

use super::{PAGE_SIZE, frame, phys_to_virt};
use linked_list_allocator::LockedHeap;

#[global_allocator]
static HEAP: LockedHeap = LockedHeap::empty();

pub fn init(bytes: usize) {
    let frames = bytes as u64 / PAGE_SIZE;
    let phys = frame::alloc_contiguous(frames, 512).expect("no memory for the kernel heap");
    // SAFETY: the region is exclusively ours and mapped RW+NX in the direct map.
    unsafe { HEAP.lock().init(phys_to_virt(phys), bytes) };
    kprintln!("[mm] heap: {} MiB at {:#x}", bytes >> 20, phys);
}
