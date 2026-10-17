//! Minimal Limine boot-protocol bindings (base revision 3).
//!
//! Written in-tree instead of depending on a crate because the protocol
//! surface we need is tiny and the published crate requires unstable compiler
//! features. The request IDs below were checked against the protocol
//! specification and the reference crate source.

use core::cell::UnsafeCell;

const COMMON_MAGIC: [u64; 2] = [0xc7b1dd30df4c8b88, 0x0a82e883a194f07b];
const BASE_REVISION: u64 = 3;

/// Bootloader handshake: Limine zeroes the third word if it supports the revision.
#[repr(C)]
pub struct BaseRevision(UnsafeCell<[u64; 3]>);

// SAFETY: written only by the bootloader before the kernel runs.
unsafe impl Sync for BaseRevision {}

impl BaseRevision {
    const fn new() -> Self {
        Self(UnsafeCell::new([
            0xf9562b2d5c95a6c8,
            0x6a7b384944536bdc,
            BASE_REVISION,
        ]))
    }

    pub fn is_supported(&self) -> bool {
        // SAFETY: plain read of a word the bootloader may have rewritten.
        unsafe { core::ptr::read_volatile(self.0.get().cast::<u64>().add(2)) == 0 }
    }
}

#[repr(C)]
pub struct Request<R> {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const R>,
}

// SAFETY: the response pointer is written once by the bootloader before entry.
unsafe impl<R> Sync for Request<R> {}

impl<R> Request<R> {
    const fn new(id: [u64; 2]) -> Self {
        Self {
            magic: COMMON_MAGIC,
            id,
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static R> {
        // SAFETY: Limine either leaves null or points at a valid response that
        // lives in bootloader-reclaimable memory, which we never free.
        unsafe { core::ptr::read_volatile(self.response.get()).as_ref() }
    }
}

#[repr(C)]
pub struct HhdmResponse {
    _revision: u64,
    pub offset: u64,
}

#[repr(C)]
pub struct ExecutableAddressResponse {
    _revision: u64,
    pub physical_base: u64,
    pub virtual_base: u64,
}

pub const MEMMAP_USABLE: u64 = 0;
pub const MEMMAP_RESERVED: u64 = 1;
pub const MEMMAP_ACPI_RECLAIMABLE: u64 = 2;
pub const MEMMAP_ACPI_NVS: u64 = 3;
pub const MEMMAP_BAD_MEMORY: u64 = 4;
pub const MEMMAP_BOOTLOADER_RECLAIMABLE: u64 = 5;
pub const MEMMAP_EXECUTABLE_AND_MODULES: u64 = 6;
pub const MEMMAP_FRAMEBUFFER: u64 = 7;

#[repr(C)]
pub struct MemmapEntry {
    pub base: u64,
    pub length: u64,
    pub kind: u64,
}

#[repr(C)]
pub struct MemmapResponse {
    _revision: u64,
    entry_count: u64,
    entries: *const &'static MemmapEntry,
}

impl MemmapResponse {
    pub fn entries(&self) -> &'static [&'static MemmapEntry] {
        // SAFETY: Limine provides `entry_count` valid pointers.
        unsafe { core::slice::from_raw_parts(self.entries, self.entry_count as usize) }
    }
}

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static BASE: BaseRevision = BaseRevision::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static HHDM: Request<HhdmResponse> = Request::new([0x48dcf1cb8ad2b852, 0x63984e959a98244b]);

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static MEMMAP: Request<MemmapResponse> = Request::new([0x67cf3d9d378a806f, 0xe304acdfc50c3c62]);

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static EXECUTABLE_ADDRESS: Request<ExecutableAddressResponse> =
    Request::new([0x71ba76863cc55f63, 0xb2644a48c516a487]);
