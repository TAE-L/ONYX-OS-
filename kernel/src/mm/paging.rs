//! Page tables and address spaces.
//!
//! * The kernel builds its own tables instead of inheriting the bootloader's.
//! * W^X is enforced by the mapping API itself ([`AddressSpace::map`] refuses
//!   a writable, executable page) and re-checked by [`AddressSpace::audit_wx`].
//! * Every upper-half PML4 entry is allocated once at boot, so user address
//!   spaces share the kernel's upper half by copying 256 entries and never
//!   need re-syncing.
//! * The kernel image is reachable only through its own mapping. It is
//!   deliberately *not* aliased writable in the direct map.

use super::{PAGE_SIZE, frame, phys_to_virt};
use crate::boot::BootInfo;
use crate::limine;
use crate::sync::IrqMutex as Mutex;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Once;
use x86_64::PhysAddr;
use x86_64::VirtAddr;
use x86_64::instructions::tlb;
use x86_64::registers::control::{Cr3, Cr3Flags};
use x86_64::structures::paging::{PageTable, PageTableFlags as F, PhysFrame};

/// First address above the user half.
pub const USER_END: u64 = 1 << 47;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageSize {
    Size4K,
    Size2M,
    Size1G,
}

impl PageSize {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Size4K => 1 << 12,
            Self::Size2M => 1 << 21,
            Self::Size1G => 1 << 30,
        }
    }

    const fn leaf_level(self) -> usize {
        match self {
            Self::Size4K => 0,
            Self::Size2M => 1,
            Self::Size1G => 2,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MapError {
    Misaligned,
    NonCanonical,
    AlreadyMapped,
    ParentIsHuge,
    OutOfMemory,
    WritableExecutable,
    NotMapped,
}

#[derive(Debug, Clone, Copy)]
pub struct Translation {
    pub phys: u64,
    pub flags: F,
    pub size: PageSize,
}

pub mod flags {
    use super::F;
    pub const fn kernel_text() -> F {
        F::PRESENT.union(F::GLOBAL)
    }
    pub const fn kernel_rodata() -> F {
        F::PRESENT.union(F::NO_EXECUTE).union(F::GLOBAL)
    }
    pub const fn kernel_data() -> F {
        F::PRESENT
            .union(F::WRITABLE)
            .union(F::NO_EXECUTE)
            .union(F::GLOBAL)
    }
    /// Device memory: uncached, never executable.
    pub const fn kernel_mmio() -> F {
        F::PRESENT
            .union(F::WRITABLE)
            .union(F::NO_EXECUTE)
            .union(F::GLOBAL)
            .union(F::NO_CACHE)
            .union(F::WRITE_THROUGH)
    }
    pub const fn user_rw() -> F {
        F::PRESENT
            .union(F::WRITABLE)
            .union(F::NO_EXECUTE)
            .union(F::USER_ACCESSIBLE)
    }
}

const fn index(va: u64, level: usize) -> usize {
    ((va >> (12 + 9 * level)) & 0x1ff) as usize
}

const fn canonical(va: u64) -> bool {
    (((va << 16) as i64) >> 16) as u64 == va
}

/// # Safety
/// `phys` must be a page-table frame owned by the caller's address space.
unsafe fn table(phys: u64) -> &'static mut PageTable {
    // SAFETY: forwarded; the direct map covers all allocatable frames.
    unsafe { &mut *phys_to_virt(phys).cast::<PageTable>() }
}

/// A root page table. User address spaces own their lower-half tables and
/// free them on drop; the kernel space is never dropped.
pub struct AddressSpace {
    root: u64,
    owns_user_tables: bool,
}

impl AddressSpace {
    fn new_empty() -> Option<Self> {
        Some(Self {
            root: frame::alloc_zeroed()?,
            owns_user_tables: false,
        })
    }

    /// A fresh user address space: empty lower half, shared kernel upper half.
    pub fn new_user() -> Option<Self> {
        let mut space = Self::new_empty()?;
        space.owns_user_tables = true;
        let kernel_root = kernel().lock().root;
        // SAFETY: both roots are valid, distinct page-table frames.
        unsafe {
            let src = table(kernel_root);
            let dst = table(space.root);
            for i in 256..512 {
                dst[i] = src[i].clone();
            }
        }
        Some(space)
    }

    pub fn is_active(&self) -> bool {
        Cr3::read().0.start_address().as_u64() == self.root
    }

    pub fn activate(&self) {
        let frame = PhysFrame::containing_address(PhysAddr::new(self.root));
        // SAFETY: the root is a valid PML4 whose upper half maps the kernel.
        unsafe { Cr3::write(frame, Cr3Flags::empty()) };
    }

    pub fn map(&mut self, va: u64, pa: u64, size: PageSize, flags: F) -> Result<(), MapError> {
        if flags.contains(F::WRITABLE) && !flags.contains(F::NO_EXECUTE) {
            return Err(MapError::WritableExecutable);
        }
        if !va.is_multiple_of(size.bytes()) || !pa.is_multiple_of(size.bytes()) {
            return Err(MapError::Misaligned);
        }
        if !canonical(va) {
            return Err(MapError::NonCanonical);
        }
        let user = va < USER_END;
        let mut tbl = self.root;
        for level in (size.leaf_level() + 1..=3).rev() {
            // SAFETY: `tbl` is a table frame reachable from our root.
            let entry = unsafe { &mut table(tbl)[index(va, level)] };
            if entry.is_unused() {
                let new = frame::alloc_zeroed().ok_or(MapError::OutOfMemory)?;
                let mut f = F::PRESENT | F::WRITABLE;
                if user {
                    f |= F::USER_ACCESSIBLE;
                }
                entry.set_addr(PhysAddr::new(new), f);
            } else if entry.flags().contains(F::HUGE_PAGE) {
                return Err(MapError::ParentIsHuge);
            }
            tbl = entry.addr().as_u64();
        }
        // SAFETY: as above.
        let entry = unsafe { &mut table(tbl)[index(va, size.leaf_level())] };
        if !entry.is_unused() {
            return Err(MapError::AlreadyMapped);
        }
        let mut f = flags | F::PRESENT;
        if size != PageSize::Size4K {
            f |= F::HUGE_PAGE;
        }
        entry.set_addr(PhysAddr::new(pa), f);
        Ok(())
    }

    /// Map `[va, va+len)` to `[pa, pa+len)` using the largest pages that fit.
    pub fn map_range(
        &mut self,
        mut va: u64,
        mut pa: u64,
        mut len: u64,
        flags: F,
        allow_1g: bool,
    ) -> Result<(), MapError> {
        while len > 0 {
            let size = if allow_1g
                && va.is_multiple_of(PageSize::Size1G.bytes())
                && pa.is_multiple_of(PageSize::Size1G.bytes())
                && len >= PageSize::Size1G.bytes()
            {
                PageSize::Size1G
            } else if va.is_multiple_of(PageSize::Size2M.bytes())
                && pa.is_multiple_of(PageSize::Size2M.bytes())
                && len >= PageSize::Size2M.bytes()
            {
                PageSize::Size2M
            } else {
                PageSize::Size4K
            };
            self.map(va, pa, size, flags)?;
            va += size.bytes();
            pa += size.bytes();
            len -= size.bytes();
        }
        Ok(())
    }

    /// Remove the mapping that starts exactly at `va`. Intermediate tables are
    /// kept (they are freed when the address space is dropped).
    pub fn unmap(&mut self, va: u64) -> Result<Translation, MapError> {
        let mut tbl = self.root;
        for level in (0..=3).rev() {
            // SAFETY: `tbl` is a table frame reachable from our root.
            let entry = unsafe { &mut table(tbl)[index(va, level)] };
            if entry.is_unused() {
                return Err(MapError::NotMapped);
            }
            let f = entry.flags();
            let leaf = level == 0 || (level <= 2 && f.contains(F::HUGE_PAGE));
            if leaf {
                let size = match level {
                    0 => PageSize::Size4K,
                    1 => PageSize::Size2M,
                    _ => PageSize::Size1G,
                };
                if !va.is_multiple_of(size.bytes()) {
                    return Err(MapError::Misaligned);
                }
                let t = Translation {
                    phys: entry.addr().as_u64(),
                    flags: f,
                    size,
                };
                entry.set_unused();
                if self.is_active() {
                    tlb::flush(VirtAddr::new(va));
                }
                return Ok(t);
            }
            tbl = entry.addr().as_u64();
        }
        Err(MapError::NotMapped)
    }

    pub fn translate(&self, va: u64) -> Option<Translation> {
        let mut tbl = self.root;
        for level in (0..=3).rev() {
            // SAFETY: `tbl` is a table frame reachable from our root.
            let entry = unsafe { &table(tbl)[index(va, level)] };
            if entry.is_unused() {
                return None;
            }
            let f = entry.flags();
            if level == 0 || (level <= 2 && f.contains(F::HUGE_PAGE)) {
                let size = match level {
                    0 => PageSize::Size4K,
                    1 => PageSize::Size2M,
                    _ => PageSize::Size1G,
                };
                let offset = va & (size.bytes() - 1);
                return Some(Translation {
                    phys: entry.addr().as_u64() + offset,
                    flags: f,
                    size,
                });
            }
            tbl = entry.addr().as_u64();
        }
        None
    }

    /// Number of present entries in the lower (user) half of the root.
    pub fn user_entries_present(&self) -> usize {
        // SAFETY: root is a valid table.
        let t = unsafe { table(self.root) };
        t.iter().take(256).filter(|e| !e.is_unused()).count()
    }

    /// Walk every mapping and report any page that is both writable and
    /// executable. Returns `(leaf mappings seen, first violation)`.
    pub fn audit_wx(&self) -> (usize, Option<(u64, F)>) {
        fn walk(
            tbl: u64,
            level: usize,
            prefix: u64,
            leaves: &mut usize,
            bad: &mut Option<(u64, F)>,
        ) {
            // SAFETY: tbl comes from walking an address space we hold.
            let t = unsafe { table(tbl) };
            for (i, entry) in t.iter().enumerate() {
                if entry.is_unused() {
                    continue;
                }
                let f = entry.flags();
                let mut va = prefix | ((i as u64) << (12 + 9 * level));
                if level == 3 && i >= 256 {
                    va |= 0xffff_0000_0000_0000;
                }
                if level == 0 || (level <= 2 && f.contains(F::HUGE_PAGE)) {
                    *leaves += 1;
                    if f.contains(F::WRITABLE) && !f.contains(F::NO_EXECUTE) && bad.is_none() {
                        *bad = Some((va, f));
                    }
                } else {
                    walk(entry.addr().as_u64(), level - 1, va, leaves, bad);
                }
            }
        }
        let (mut leaves, mut bad) = (0, None);
        walk(self.root, 3, 0, &mut leaves, &mut bad);
        (leaves, bad)
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        assert!(
            !self.is_active(),
            "dropping the active address space would unmap the running kernel's tables"
        );
        if !self.owns_user_tables {
            return;
        }
        fn free_tables(tbl: u64, level: usize) {
            // SAFETY: tbl is a table frame owned by the dropping space.
            let t = unsafe { table(tbl) };
            if level > 0 {
                let limit = if level == 3 { 256 } else { 512 };
                for entry in t.iter().take(limit) {
                    if !entry.is_unused() && !entry.flags().contains(F::HUGE_PAGE) {
                        free_tables(entry.addr().as_u64(), level - 1);
                    }
                }
            }
            frame::free_frame(tbl);
        }
        free_tables(self.root, 3);
    }
}

static KERNEL: Once<Mutex<AddressSpace>> = Once::new();

pub fn kernel() -> &'static Mutex<AddressSpace> {
    KERNEL.get().expect("kernel address space not initialised")
}

unsafe extern "C" {
    static __text_start: u8;
    static __text_end: u8;
    static __rodata_start: u8;
    static __rodata_end: u8;
    static __data_start: u8;
    static __boot_stack_guard: u8;
    static __boot_stack_bottom: u8;
    static __bss_end: u8;
}

/// Window of kernel virtual address space for device (MMIO) mappings.
pub const MMIO_BASE: u64 = 0xffff_c000_0000_0000;
static MMIO_NEXT: AtomicU64 = AtomicU64::new(MMIO_BASE);

/// Map a physical device region uncached into the kernel's MMIO window.
/// Returns the virtual address corresponding to `phys`. Mappings are never
/// removed; each is followed by an unmapped guard page.
pub fn map_mmio(phys: u64, len: u64) -> Result<u64, MapError> {
    let start = super::align_down(phys, PAGE_SIZE);
    let offset = phys - start;
    let size = super::align_up(offset + len, PAGE_SIZE);
    let va = MMIO_NEXT.fetch_add(size + PAGE_SIZE, Ordering::SeqCst);
    kernel()
        .lock()
        .map_range(va, start, size, flags::kernel_mmio(), false)?;
    Ok(va + offset)
}

pub fn symbol(sym: &u8) -> u64 {
    core::ptr::from_ref(sym) as u64
}

pub fn boot_stack_guard() -> u64 {
    // SAFETY: address of a linker symbol.
    unsafe { symbol(&__boot_stack_guard) }
}

pub fn init_kernel_space(info: &BootInfo) {
    let mut space = AddressSpace::new_empty().expect("no frame for the kernel PML4");

    // Pre-allocate every upper-half PML4 entry so user spaces can share them.
    // SAFETY: freshly allocated root table.
    unsafe {
        let root = table(space.root);
        for i in 256..512 {
            let t = frame::alloc_zeroed().expect("no frame for a kernel PDPT");
            root[i].set_addr(PhysAddr::new(t), F::PRESENT | F::WRITABLE);
        }
    }

    // Kernel image: each section with its own permissions. The boot stack
    // guard page between .bss and the stack is intentionally left unmapped.
    let to_phys = |va: u64| va - info.kernel_virt + info.kernel_phys;
    // SAFETY: addresses of linker symbols.
    let (text, text_end, ro, ro_end, data, guard, stack, bss_end) = unsafe {
        (
            symbol(&__text_start),
            symbol(&__text_end),
            symbol(&__rodata_start),
            symbol(&__rodata_end),
            symbol(&__data_start),
            symbol(&__boot_stack_guard),
            symbol(&__boot_stack_bottom),
            symbol(&__bss_end),
        )
    };
    for (start, end, f) in [
        (text, text_end, flags::kernel_text()),
        (ro, ro_end, flags::kernel_rodata()),
        (data, guard, flags::kernel_data()),
        (stack, bss_end, flags::kernel_data()),
    ] {
        space
            .map_range(start, to_phys(start), end - start, f, false)
            .expect("mapping the kernel image failed");
    }

    // Direct map of physical memory (RW + NX). The kernel image region is
    // excluded on purpose so no writable alias of kernel code exists.
    let gib = crate::arch::cpu::features().gib_pages;
    let mut mapped = 0u64;
    for e in info.memmap {
        let wanted = matches!(
            e.kind,
            limine::MEMMAP_USABLE
                | limine::MEMMAP_ACPI_RECLAIMABLE
                | limine::MEMMAP_ACPI_NVS
                | limine::MEMMAP_BOOTLOADER_RECLAIMABLE
                | limine::MEMMAP_FRAMEBUFFER
        );
        if !wanted {
            continue;
        }
        space
            .map_range(
                info.hhdm + e.base,
                e.base,
                e.length,
                flags::kernel_data(),
                gib,
            )
            .expect("mapping the direct map failed");
        mapped += e.length;
    }

    let (leaves, violation) = space.audit_wx();
    if let Some((va, f)) = violation {
        panic!("W^X violation in the kernel mappings at {va:#x}: {f:?}");
    }
    space.activate();
    kprintln!(
        "[mm] kernel page tables active: {} MiB direct-mapped, {} leaf mappings, W^X audit clean",
        mapped >> 20,
        leaves
    );
    let _ = PAGE_SIZE;
    KERNEL.call_once(|| Mutex::new(space));
}
