//! Physical frame allocator: a bitmap (1 = in use) over all usable RAM.
//!
//! Deliberately simple for Phase 1: one global lock, next-fit search. The
//! interface (`alloc_frame`, `alloc_contiguous`, `free_*`) is what the rest of
//! the kernel depends on; the implementation becomes a buddy allocator with
//! per-CPU caches in Phase 2 when SMP lands.
//!
//! The first MiB is never handed out: real-mode memory is reserved for the AP
//! startup trampoline.

use super::{PAGE_SIZE, align_down, align_up, phys_to_virt};
use crate::boot::BootInfo;
use crate::limine::MEMMAP_USABLE;
use crate::sync::IrqMutex as Mutex;

const LOW_RESERVED: u64 = 1 << 20;

struct Allocator {
    bitmap: *mut u64,
    words: usize,
    /// Number of frames covered by the bitmap.
    frames: u64,
    free: u64,
    /// Search hint (frame index).
    hint: u64,
}

// SAFETY: the bitmap lives in dedicated physical memory and is only touched
// while holding the global lock.
unsafe impl Send for Allocator {}

impl Allocator {
    fn test(&self, i: u64) -> bool {
        // SAFETY: callers keep `i < self.frames`.
        unsafe { (*self.bitmap.add((i / 64) as usize) >> (i % 64)) & 1 == 1 }
    }

    fn set(&mut self, i: u64) {
        // SAFETY: as above.
        unsafe { *self.bitmap.add((i / 64) as usize) |= 1 << (i % 64) };
    }

    fn clear(&mut self, i: u64) {
        // SAFETY: as above.
        unsafe { *self.bitmap.add((i / 64) as usize) &= !(1 << (i % 64)) };
    }
}

static ALLOC: Mutex<Option<Allocator>> = Mutex::new(None);

pub fn init(info: &BootInfo) {
    let top = info
        .memmap
        .iter()
        .filter(|e| e.kind == MEMMAP_USABLE)
        .map(|e| e.base + e.length)
        .max()
        .expect("no usable memory");
    assert!(top > LOW_RESERVED, "no usable memory above 1 MiB");

    let frames = top / PAGE_SIZE;
    let words = frames.div_ceil(64) as usize;
    let bitmap_frames = (words as u64 * 8).div_ceil(PAGE_SIZE);

    // Home for the bitmap: the first usable region above 1 MiB that fits it.
    let bitmap_phys = info
        .memmap
        .iter()
        .filter(|e| e.kind == MEMMAP_USABLE)
        .find_map(|e| {
            let start = align_up(e.base.max(LOW_RESERVED), PAGE_SIZE);
            let end = e.base + e.length;
            (end > start && end - start >= bitmap_frames * PAGE_SIZE).then_some(start)
        })
        .expect("no region large enough for the frame bitmap");

    let bitmap = phys_to_virt(bitmap_phys).cast::<u64>();
    for w in 0..words {
        // SAFETY: the region is usable RAM reachable through the direct map.
        unsafe { bitmap.add(w).write(u64::MAX) };
    }
    let mut a = Allocator {
        bitmap,
        words,
        frames,
        free: 0,
        hint: LOW_RESERVED / PAGE_SIZE,
    };
    for e in info.memmap.iter().filter(|e| e.kind == MEMMAP_USABLE) {
        let start = align_up(e.base.max(LOW_RESERVED), PAGE_SIZE);
        let end = align_down(e.base + e.length, PAGE_SIZE);
        for f in start / PAGE_SIZE..end.max(start) / PAGE_SIZE {
            a.clear(f);
            a.free += 1;
        }
    }
    for f in bitmap_phys / PAGE_SIZE..bitmap_phys / PAGE_SIZE + bitmap_frames {
        a.set(f);
        a.free -= 1;
    }
    kprintln!(
        "[mm] frame allocator: {} frames tracked, {} free ({} MiB), bitmap {} KiB at {:#x}",
        frames,
        a.free,
        (a.free * PAGE_SIZE) >> 20,
        (bitmap_frames * PAGE_SIZE) >> 10,
        bitmap_phys
    );
    *ALLOC.lock() = Some(a);
}

/// Allocate one 4 KiB frame; returns its physical address.
pub fn alloc_frame() -> Option<u64> {
    let mut guard = ALLOC.lock();
    let a = guard.as_mut()?;
    if a.free == 0 {
        return None;
    }
    let start = (a.hint / 64) as usize;
    for k in 0..a.words {
        let w = (start + k) % a.words;
        // SAFETY: `w < words`.
        let v = unsafe { *a.bitmap.add(w) };
        if v != u64::MAX {
            let idx = w as u64 * 64 + u64::from((!v).trailing_zeros());
            a.set(idx);
            a.free -= 1;
            a.hint = idx;
            return Some(idx * PAGE_SIZE);
        }
    }
    None
}

/// Allocate `count` physically contiguous frames aligned to `align_frames`
/// (a power of two). Used for the heap and huge pages.
pub fn alloc_contiguous(count: u64, align_frames: u64) -> Option<u64> {
    assert!(align_frames.is_power_of_two() && count > 0);
    let mut guard = ALLOC.lock();
    let a = guard.as_mut()?;
    let mut i = align_up(LOW_RESERVED / PAGE_SIZE, align_frames);
    while i + count <= a.frames {
        let mut j = 0;
        while j < count && !a.test(i + j) {
            j += 1;
        }
        if j == count {
            for k in 0..count {
                a.set(i + k);
            }
            a.free -= count;
            return Some(i * PAGE_SIZE);
        }
        i = align_up(i + j + 1, align_frames);
    }
    None
}

/// Like [`alloc_frame`] but zero-filled.
pub fn alloc_zeroed() -> Option<u64> {
    let pa = alloc_frame()?;
    // SAFETY: the frame is ours and mapped in the direct map.
    unsafe { core::ptr::write_bytes(phys_to_virt(pa), 0, PAGE_SIZE as usize) };
    Some(pa)
}

pub fn free_frame(phys: u64) {
    free_contiguous(phys, 1);
}

pub fn free_contiguous(phys: u64, count: u64) {
    assert!(
        phys.is_multiple_of(PAGE_SIZE),
        "free of unaligned frame {phys:#x}"
    );
    let mut guard = ALLOC.lock();
    let a = guard.as_mut().expect("frame allocator not initialised");
    let first = phys / PAGE_SIZE;
    assert!(first + count <= a.frames, "free of frame outside RAM");
    for i in first..first + count {
        assert!(a.test(i), "double free of frame {:#x}", i * PAGE_SIZE);
        a.clear(i);
    }
    a.free += count;
    a.hint = a.hint.min(first);
}

pub fn free_frames() -> u64 {
    ALLOC.lock().as_ref().map_or(0, |a| a.free)
}

pub fn total_frames() -> u64 {
    ALLOC.lock().as_ref().map_or(0, |a| a.frames)
}
