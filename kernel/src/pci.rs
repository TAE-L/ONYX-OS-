//! PCI Express enumeration over ECAM (memory-mapped configuration space).
#![allow(dead_code)] // consumed by the NVMe driver (Phase 4b)

use crate::acpi::{self, EcamRegion};
use crate::mm::paging;
use alloc::vec::Vec;
use spin::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
}

impl core::fmt::Display for Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:02x}:{:02x}.{}", self.bus, self.dev, self.func)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bar {
    pub index: u8,
    pub phys: u64,
    pub size: u64,
    pub is_64: bool,
    pub prefetchable: bool,
}

#[derive(Clone, Debug)]
pub struct Device {
    pub addr: Address,
    pub vendor: u16,
    pub device: u16,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub header_type: u8,
    pub bars: Vec<Bar>,
}

struct Ecam {
    region: EcamRegion,
    /// Mapped virtual base of each bus's 1 MiB window (0 = not mapped yet).
    bus_va: [u64; 256],
}

static ECAM: Mutex<Vec<Ecam>> = Mutex::new(Vec::new());
static DEVICES: Mutex<Vec<Device>> = Mutex::new(Vec::new());

fn cfg_va(a: Address, offset: u16) -> Option<*mut u8> {
    let mut regions = ECAM.lock();
    let e = regions
        .iter_mut()
        .find(|e| a.bus >= e.region.start_bus && a.bus <= e.region.end_bus)?;
    if e.bus_va[a.bus as usize] == 0 {
        let phys = e.region.base + (u64::from(a.bus - e.region.start_bus) << 20);
        e.bus_va[a.bus as usize] = paging::map_mmio(phys, 1 << 20).ok()?;
    }
    let off = (u64::from(a.dev) << 15) | (u64::from(a.func) << 12) | u64::from(offset);
    Some((e.bus_va[a.bus as usize] + off) as *mut u8)
}

pub fn read32(a: Address, offset: u16) -> u32 {
    match cfg_va(a, offset & !3) {
        // SAFETY: inside a mapped ECAM window, 4-byte aligned.
        Some(p) => unsafe { core::ptr::read_volatile(p.cast::<u32>()) },
        None => u32::MAX,
    }
}

pub fn write32(a: Address, offset: u16, value: u32) {
    if let Some(p) = cfg_va(a, offset & !3) {
        // SAFETY: as above.
        unsafe { core::ptr::write_volatile(p.cast::<u32>(), value) };
    }
}

pub fn read16(a: Address, offset: u16) -> u16 {
    (read32(a, offset & !3) >> ((offset & 2) * 8)) as u16
}

pub fn write16(a: Address, offset: u16, value: u16) {
    let shift = (offset & 2) * 8;
    let old = read32(a, offset & !3);
    let new = (old & !(0xFFFF << shift)) | (u32::from(value) << shift);
    write32(a, offset & !3, new);
}

pub const CMD_IO: u16 = 1 << 0;
pub const CMD_MEMORY: u16 = 1 << 1;
pub const CMD_BUS_MASTER: u16 = 1 << 2;
pub const CMD_INTX_DISABLE: u16 = 1 << 10;

/// Turn on memory decoding and bus mastering (DMA) for a device.
pub fn enable(a: Address) {
    let cmd = read16(a, 4);
    write16(a, 4, cmd | CMD_MEMORY | CMD_BUS_MASTER);
}

/// Decode and size the BARs of a type-0 header. Decoding is switched off while
/// sizing so the temporary all-ones value is never seen by the device's decoder.
fn read_bars(a: Address) -> Vec<Bar> {
    let cmd = read16(a, 4);
    write16(a, 4, cmd & !(CMD_IO | CMD_MEMORY));
    let mut bars = Vec::new();
    let mut i = 0u8;
    while i < 6 {
        let off = 0x10 + u16::from(i) * 4;
        let orig = read32(a, off);
        if orig & 1 == 1 {
            i += 1; // I/O BAR: not used by anything modern we drive
            continue;
        }
        let is_64 = (orig >> 1) & 3 == 2;
        let prefetchable = orig & 8 != 0;
        write32(a, off, 0xFFFF_FFFF);
        let lo = read32(a, off);
        write32(a, off, orig);
        let (mut size_mask, mut base) = (u64::from(lo & !0xF), u64::from(orig & !0xF));
        if is_64 && i < 5 {
            let orig_hi = read32(a, off + 4);
            write32(a, off + 4, 0xFFFF_FFFF);
            let hi = read32(a, off + 4);
            write32(a, off + 4, orig_hi);
            size_mask |= u64::from(hi) << 32;
            base |= u64::from(orig_hi) << 32;
        } else {
            size_mask |= 0xFFFF_FFFF_0000_0000; // 32-bit BAR: upper bits read as 1s
        }
        if size_mask & !0xF != 0 && lo != 0 {
            let size = (!size_mask).wrapping_add(1);
            if size != 0 {
                bars.push(Bar {
                    index: i,
                    phys: base,
                    size,
                    is_64,
                    prefetchable,
                });
            }
        }
        i += if is_64 { 2 } else { 1 };
    }
    write16(a, 4, cmd);
    bars
}

fn probe(a: Address) -> Option<Device> {
    let id = read32(a, 0);
    if id & 0xFFFF == 0xFFFF {
        return None;
    }
    let class = read32(a, 8);
    let header_type = (read32(a, 0xC) >> 16) as u8;
    let bars = if header_type & 0x7F == 0 {
        read_bars(a)
    } else {
        Vec::new()
    };
    Some(Device {
        addr: a,
        vendor: id as u16,
        device: (id >> 16) as u16,
        class: (class >> 24) as u8,
        subclass: (class >> 16) as u8,
        prog_if: (class >> 8) as u8,
        header_type,
        bars,
    })
}

/// Scan every bus the MCFG table covers. Returns the number of devices found.
pub fn init() -> usize {
    let regions = acpi::ecam_regions();
    if regions.is_empty() {
        kprintln!("[pci] no MCFG table: PCIe configuration space unavailable");
        return 0;
    }
    {
        let mut e = ECAM.lock();
        e.clear();
        for r in regions.iter().filter(|r| r.segment == 0) {
            e.push(Ecam {
                region: *r,
                bus_va: [0; 256],
            });
        }
    }
    let mut found = Vec::new();
    // Start at each window's first bus and follow PCI-to-PCI bridges downstream,
    // so only buses that exist get their configuration window mapped.
    for r in regions.iter().filter(|r| r.segment == 0) {
        scan_bus(r.start_bus, 0, &mut found);
    }
    let n = found.len();
    for d in &found {
        kprintln!(
            "[pci] {} {:04x}:{:04x} class {:02x}{:02x}{:02x}",
            d.addr,
            d.vendor,
            d.device,
            d.class,
            d.subclass,
            d.prog_if
        );
    }
    *DEVICES.lock() = found;
    n
}

fn scan_bus(bus: u8, depth: u8, out: &mut Vec<Device>) {
    if depth > 8 {
        return;
    }
    for dev in 0..32u8 {
        let mut func = 0u8;
        while func < 8 {
            let a = Address { bus, dev, func };
            let Some(d) = probe(a) else {
                if func == 0 {
                    break;
                }
                func += 1;
                continue;
            };
            let multi = d.header_type & 0x80 != 0;
            let bridge = d.header_type & 0x7F == 1;
            out.push(d);
            if bridge {
                let secondary = (read32(a, 0x18) >> 8) as u8;
                if secondary > bus {
                    scan_bus(secondary, depth + 1, out);
                }
            }
            if func == 0 && !multi {
                break;
            }
            func += 1;
        }
    }
}

pub fn devices() -> Vec<Device> {
    DEVICES.lock().clone()
}

pub fn find_class(class: u8, subclass: u8, prog_if: u8) -> Option<Device> {
    DEVICES
        .lock()
        .iter()
        .find(|d| d.class == class && d.subclass == subclass && d.prog_if == prog_if)
        .cloned()
}
