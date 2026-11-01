//! A strict loader for static x86_64 ELF executables (`ET_EXEC`).
//!
//! Everything about the file is untrusted: sizes, offsets and addresses are
//! range-checked, segments may not overlap or be writable and executable at
//! once, and nothing may reach into the kernel half. The user stack is added
//! here too. A dynamic loader arrives with the Linux ABI layer (Phase 8).

use crate::mm::paging::{AddressSpace, MapError, USER_END, flags};
use crate::mm::{PAGE_SIZE, align_up, phys_to_virt};
use alloc::vec::Vec;

#[derive(Debug, PartialEq, Eq)]
pub enum ElfError {
    TooShort,
    BadMagic,
    Unsupported(&'static str),
    BadSegment(&'static str),
    OutOfMemory,
}

impl From<MapError> for ElfError {
    fn from(e: MapError) -> Self {
        match e {
            MapError::OutOfMemory => ElfError::OutOfMemory,
            MapError::WritableExecutable => ElfError::BadSegment("writable and executable"),
            _ => ElfError::BadSegment("cannot be mapped"),
        }
    }
}

/// Lowest user address a program may use; the pages below catch null pointers.
pub const USER_MIN: u64 = 0x1_0000;
/// Segments must end below this, leaving a gap before the stack.
pub const SEGMENT_LIMIT: u64 = 0x7000_0000_0000;
pub const STACK_TOP: u64 = 0x7fff_ffff_f000;
pub const STACK_PAGES: u64 = 16;

pub struct Loaded {
    pub entry: u64,
    pub stack_top: u64,
}

fn u16_at(d: &[u8], o: usize) -> Option<u64> {
    Some(u64::from(u16::from_le_bytes(
        d.get(o..o + 2)?.try_into().ok()?,
    )))
}
fn u32_at(d: &[u8], o: usize) -> Option<u64> {
    Some(u64::from(u32::from_le_bytes(
        d.get(o..o + 4)?.try_into().ok()?,
    )))
}
fn u64_at(d: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(d.get(o..o + 8)?.try_into().ok()?))
}

struct Segment {
    vaddr: u64,
    offset: u64,
    filesz: u64,
    memsz: u64,
    flags: u32,
}

const PF_X: u32 = 1;
const PF_W: u32 = 2;

pub fn load(space: &mut AddressSpace, data: &[u8]) -> Result<Loaded, ElfError> {
    let short = || ElfError::TooShort;
    if data.len() < 64 {
        return Err(short());
    }
    if data[..4] != *b"\x7fELF" {
        return Err(ElfError::BadMagic);
    }
    if data[4] != 2 || data[5] != 1 {
        return Err(ElfError::Unsupported("not 64-bit little-endian"));
    }
    if u16_at(data, 16).ok_or_else(short)? != 2 {
        return Err(ElfError::Unsupported("only static ET_EXEC is supported"));
    }
    if u16_at(data, 18).ok_or_else(short)? != 62 {
        return Err(ElfError::Unsupported("not x86_64"));
    }
    let entry = u64_at(data, 24).ok_or_else(short)?;
    let phoff = u64_at(data, 32).ok_or_else(short)?;
    let phentsize = u16_at(data, 54).ok_or_else(short)?;
    let phnum = u16_at(data, 56).ok_or_else(short)?;
    if phentsize != 56 || phnum == 0 || phnum > 32 {
        return Err(ElfError::Unsupported("bad program header table"));
    }

    let mut segments = Vec::new();
    for i in 0..phnum {
        let o = phoff
            .checked_add(i * 56)
            .and_then(|o| usize::try_from(o).ok())
            .ok_or_else(short)?;
        let p_type = u32_at(data, o).ok_or_else(short)?;
        let p_flags = u32_at(data, o + 4).ok_or_else(short)? as u32;
        match p_type {
            1 => segments.push(Segment {
                offset: u64_at(data, o + 8).ok_or_else(short)?,
                vaddr: u64_at(data, o + 16).ok_or_else(short)?,
                filesz: u64_at(data, o + 32).ok_or_else(short)?,
                memsz: u64_at(data, o + 40).ok_or_else(short)?,
                flags: p_flags,
            }),
            2 | 3 => return Err(ElfError::Unsupported("dynamic linking")),
            0x6474_e551 if p_flags & PF_X != 0 => {
                return Err(ElfError::BadSegment("executable stack requested"));
            }
            _ => {}
        }
    }
    if segments.is_empty() {
        return Err(ElfError::Unsupported("no loadable segments"));
    }

    // Validate every segment before mapping anything.
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut entry_ok = false;
    for s in &segments {
        let bad = |why| Err(ElfError::BadSegment(why));
        if s.memsz == 0 || s.filesz > s.memsz {
            return bad("file size larger than memory size");
        }
        let file_end = s.offset.checked_add(s.filesz);
        if file_end.is_none_or(|e| e > data.len() as u64) {
            return bad("file range outside the image");
        }
        let Some(mem_end) = s.vaddr.checked_add(s.memsz) else {
            return bad("address overflow");
        };
        if s.vaddr < USER_MIN || mem_end > SEGMENT_LIMIT || mem_end > USER_END {
            return bad("outside the user address range");
        }
        if s.flags & PF_W != 0 && s.flags & PF_X != 0 {
            return bad("writable and executable");
        }
        let (lo, hi) = (s.vaddr & !(PAGE_SIZE - 1), align_up(mem_end, PAGE_SIZE));
        if ranges.iter().any(|&(a, b)| lo < b && a < hi) {
            return bad("segments share a page");
        }
        ranges.push((lo, hi));
        if s.flags & PF_X != 0 && (s.vaddr..mem_end).contains(&entry) {
            entry_ok = true;
        }
    }
    if !entry_ok {
        return Err(ElfError::BadSegment(
            "entry point is not in executable code",
        ));
    }

    for s in &segments {
        let lo = s.vaddr & !(PAGE_SIZE - 1);
        let hi = align_up(s.vaddr + s.memsz, PAGE_SIZE);
        let page_flags = if s.flags & PF_X != 0 {
            flags::user_rx()
        } else if s.flags & PF_W != 0 {
            flags::user_rw()
        } else {
            flags::user_ro()
        };
        let file = &data[s.offset as usize..(s.offset + s.filesz) as usize];
        let mut va = lo;
        while va < hi {
            let pa = space.map_new(va, page_flags)?;
            // Copy the part of the file image that lands on this page.
            let from = va.max(s.vaddr);
            let to = (va + PAGE_SIZE).min(s.vaddr + s.filesz);
            if from < to {
                let src = &file[(from - s.vaddr) as usize..(to - s.vaddr) as usize];
                // SAFETY: `pa` is a frame we just allocated, reachable through the direct map.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        phys_to_virt(pa).add((from - va) as usize),
                        src.len(),
                    );
                }
            }
            va += PAGE_SIZE;
        }
    }

    for i in 0..STACK_PAGES {
        space.map_new(STACK_TOP - (i + 1) * PAGE_SIZE, flags::user_rw())?;
    }
    Ok(Loaded {
        entry,
        // Look like a `call` just happened: the return-address slot is the top word.
        stack_top: STACK_TOP - 8,
    })
}
