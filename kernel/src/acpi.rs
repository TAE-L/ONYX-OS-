//! Just enough ACPI to find tables: RSDP, XSDT/RSDT, checksums.

use crate::{limine, mm};
use alloc::vec::Vec;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem: [u8; 6],
    revision: u8,
    rsdt: u32,
    length: u32,
    xsdt: u64,
    ext_checksum: u8,
    _reserved: [u8; 3],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Header {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem: [u8; 6],
    oem_table: [u8; 8],
    oem_rev: u32,
    creator: u32,
    creator_rev: u32,
}

/// A validated table: its physical address and total length.
#[derive(Clone, Copy, Debug)]
pub struct Table {
    pub phys: u64,
    pub len: usize,
}

impl Table {
    pub fn bytes(&self) -> &'static [u8] {
        // SAFETY: ACPI tables live in reclaimable memory mapped by the direct
        // map, which the kernel never frees; length was validated in `read_table`.
        unsafe { core::slice::from_raw_parts(mm::phys_to_virt(self.phys), self.len) }
    }
}

fn sum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |a, &b| a.wrapping_add(b))
}

fn read_unaligned<T: Copy>(phys: u64) -> T {
    // SAFETY: caller passes the address of a firmware table inside the direct map.
    unsafe { mm::phys_to_virt(phys).cast::<T>().read_unaligned() }
}

fn read_table(phys: u64) -> Option<([u8; 4], Table)> {
    let h: Header = read_unaligned(phys);
    let len = h.length as usize;
    if len < core::mem::size_of::<Header>() || len > 16 * 1024 * 1024 {
        return None;
    }
    let t = Table { phys, len };
    if sum(t.bytes()) != 0 {
        return None;
    }
    Some((h.signature, t))
}

/// Every table in the XSDT (or RSDT on ACPI 1.0), with a valid checksum.
pub fn tables() -> Vec<([u8; 4], Table)> {
    let Some(resp) = limine::RSDP.response() else {
        return Vec::new();
    };
    let mut addr = resp.address;
    if addr >= mm::hhdm() {
        addr -= mm::hhdm();
    }
    let rsdp: Rsdp = read_unaligned(addr);
    if &rsdp.signature != b"RSD PTR " {
        return Vec::new();
    }
    // SAFETY: the first 20 bytes are the ACPI 1.0 structure.
    let first20 = unsafe { core::slice::from_raw_parts(mm::phys_to_virt(addr), 20) };
    if sum(first20) != 0 {
        return Vec::new();
    }
    let (root, wide) = if rsdp.revision >= 2 && rsdp.xsdt != 0 {
        (rsdp.xsdt, true)
    } else {
        (u64::from(rsdp.rsdt), false)
    };
    let Some((_, root_table)) = read_table(root) else {
        return Vec::new();
    };
    let body = &root_table.bytes()[core::mem::size_of::<Header>()..];
    let step = if wide { 8 } else { 4 };
    let mut out = Vec::new();
    for chunk in body.chunks_exact(step) {
        let phys = if wide {
            u64::from_le_bytes(chunk.try_into().unwrap())
        } else {
            u64::from(u32::from_le_bytes(chunk.try_into().unwrap()))
        };
        if let Some(entry) = read_table(phys) {
            out.push(entry);
        }
    }
    out
}

pub fn find(signature: &[u8; 4]) -> Option<Table> {
    tables()
        .into_iter()
        .find(|(s, _)| s == signature)
        .map(|(_, t)| t)
}

/// One ECAM window from the MCFG table.
#[derive(Clone, Copy, Debug)]
pub struct EcamRegion {
    pub base: u64,
    pub segment: u16,
    pub start_bus: u8,
    pub end_bus: u8,
}

pub fn ecam_regions() -> Vec<EcamRegion> {
    let Some(t) = find(b"MCFG") else {
        return Vec::new();
    };
    // 36-byte header + 8 reserved bytes, then 16-byte allocation entries.
    let Some(entries) = t.bytes().get(44..) else {
        return Vec::new();
    };
    entries
        .chunks_exact(16)
        .map(|e| EcamRegion {
            base: u64::from_le_bytes(e[0..8].try_into().unwrap()),
            segment: u16::from_le_bytes(e[8..10].try_into().unwrap()),
            start_bus: e[10],
            end_bus: e[11],
        })
        .collect()
}
