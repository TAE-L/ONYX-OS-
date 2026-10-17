//! Boot information handed over by Limine, validated once and kept for the
//! lifetime of the kernel.

use crate::limine::{self, MemmapEntry};
use spin::Once;

pub struct BootInfo {
    /// Virtual offset of the bootloader's direct map of physical memory.
    pub hhdm: u64,
    pub memmap: &'static [&'static MemmapEntry],
    pub kernel_phys: u64,
    pub kernel_virt: u64,
}

static INFO: Once<BootInfo> = Once::new();

pub fn info() -> &'static BootInfo {
    INFO.get().expect("boot info not collected")
}

fn kind_name(kind: u64) -> &'static str {
    match kind {
        limine::MEMMAP_USABLE => "usable",
        limine::MEMMAP_RESERVED => "reserved",
        limine::MEMMAP_ACPI_RECLAIMABLE => "acpi-reclaimable",
        limine::MEMMAP_ACPI_NVS => "acpi-nvs",
        limine::MEMMAP_BAD_MEMORY => "bad",
        limine::MEMMAP_BOOTLOADER_RECLAIMABLE => "bootloader",
        limine::MEMMAP_EXECUTABLE_AND_MODULES => "kernel",
        limine::MEMMAP_FRAMEBUFFER => "framebuffer",
        _ => "unknown",
    }
}

pub fn collect() -> &'static BootInfo {
    assert!(
        limine::BASE.is_supported(),
        "bootloader does not support Limine base revision 3"
    );
    let hhdm = limine::HHDM.response().expect("no HHDM response").offset;
    let memmap = limine::MEMMAP.response().expect("no memory map").entries();
    let exe = limine::EXECUTABLE_ADDRESS
        .response()
        .expect("no executable address");

    kprintln!("[boot] HHDM at {hhdm:#x}");
    kprintln!(
        "[boot] kernel phys {:#x} virt {:#x}",
        exe.physical_base,
        exe.virtual_base
    );
    let mut usable = 0u64;
    for e in memmap {
        kprintln!(
            "[boot]   {:#012x}-{:#012x} {}",
            e.base,
            e.base + e.length,
            kind_name(e.kind)
        );
        if e.kind == limine::MEMMAP_USABLE {
            usable += e.length;
        }
    }
    kprintln!("[boot] usable memory: {} MiB", usable >> 20);

    INFO.call_once(|| BootInfo {
        hhdm,
        memmap,
        kernel_phys: exe.physical_base,
        kernel_virt: exe.virtual_base,
    })
}
