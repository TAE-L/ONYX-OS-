//! Kernel heap: a linked-list allocator over contiguous frames in the direct
//! map, behind an interrupt-safe lock (tasks can be preempted mid-allocation).
//! Per-CPU slab caches in front of it arrive with SMP.

use super::{PAGE_SIZE, frame, phys_to_virt};
use crate::sync::IrqMutex;
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::{self, NonNull};
use linked_list_allocator::Heap;

struct IrqHeap(IrqMutex<Heap>);

// SAFETY: allocations come from one exclusively owned region, serialised by the lock.
unsafe impl GlobalAlloc for IrqHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0
            .lock()
            .allocate_first_fit(layout)
            .map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if let Some(p) = NonNull::new(p) {
            // SAFETY: `p` came from `alloc` with this layout.
            unsafe { self.0.lock().deallocate(p, layout) };
        }
    }
}

#[global_allocator]
static HEAP: IrqHeap = IrqHeap(IrqMutex::new(Heap::empty()));

pub fn init(bytes: usize) {
    let frames = bytes as u64 / PAGE_SIZE;
    let phys = frame::alloc_contiguous(frames, 512).expect("no memory for the kernel heap");
    // SAFETY: the region is exclusively ours and mapped RW+NX in the direct map.
    unsafe { HEAP.0.lock().init(phys_to_virt(phys), bytes) };
    kprintln!("[mm] heap: {} MiB at {:#x}", bytes >> 20, phys);
}

/// Bytes currently allocated.
pub fn used() -> usize {
    HEAP.0.lock().used()
}
