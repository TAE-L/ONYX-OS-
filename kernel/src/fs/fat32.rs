//! FAT32 with long file names, read and write.
//!
//! Everything goes through the page cache. One lock covers each volume, so
//! operations are serialised per filesystem. Timestamps are a fixed date (the
//! kernel has no wall clock yet). The backup boot sector is not rewritten.

#![allow(dead_code)]

use crate::cache::Cache;
use crate::sync::YieldMutex;
use crate::vfs::{DirEntry, FileSystem, FsError, Kind, Node};
use alloc::{collections::BTreeMap, string::String, sync::Arc, vec, vec::Vec};

const ATTR_DIR: u8 = 0x10;
const ATTR_VOLUME: u8 = 0x08;
const ATTR_LFN: u8 = 0x0F;
const EOC_MIN: u32 = 0x0FFF_FFF8;
const EOC: u32 = 0x0FFF_FFFF;
const BAD: u32 = 0x0FFF_FFF7;
const DOS_DATE: u16 = ((2026 - 1980) << 9) | (1 << 5) | 1;

pub struct Fat32 {
    cache: Arc<Cache>,
    bps: u64,
    cluster_bytes: u64,
    fat_off: u64,
    fat_bytes: u64,
    num_fats: u64,
    data_off: u64,
    cluster_count: u32,
    root_cluster: u32,
    fsinfo_off: Option<u64>,
    state: YieldMutex<State>,
}

struct State {
    free: u32,
    next_free: u32,
    chains: BTreeMap<u32, Vec<u32>>,
}

/// One logical directory entry (possibly several 32-byte records).
#[derive(Clone)]
struct Raw {
    name: String,
    short: [u8; 11],
    attr: u8,
    cluster: u32,
    size: u32,
    /// Device byte offset of the short-name record.
    short_off: u64,
    /// Device byte offsets of every record belonging to the entry, LFN first.
    offsets: Vec<u64>,
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn lfn_checksum(short: &[u8; 11]) -> u8 {
    short
        .iter()
        .fold(0u8, |s, &c| s.rotate_right(1).wrapping_add(c))
}

fn upper_eq(a: &str, b: &str) -> bool {
    a.chars()
        .flat_map(char::to_uppercase)
        .eq(b.chars().flat_map(char::to_uppercase))
}

fn short_to_string(short: &[u8; 11], ntres: u8) -> String {
    let lower = |c: u8, flag: bool| -> char {
        let ch = if c == 0x05 { 0xE5 } else { c } as char;
        if flag { ch.to_ascii_lowercase() } else { ch }
    };
    let base: String = short[..8]
        .iter()
        .map(|&c| lower(c, ntres & 0x08 != 0))
        .collect::<String>()
        .trim_end()
        .into();
    let ext: String = short[8..]
        .iter()
        .map(|&c| lower(c, ntres & 0x10 != 0))
        .collect::<String>()
        .trim_end()
        .into();
    if ext.is_empty() {
        base
    } else {
        alloc::format!("{base}.{ext}")
    }
}

fn valid_long_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && !name.ends_with('.')
        && !name.ends_with(' ')
        && !name
            .chars()
            .any(|c| (c as u32) < 0x20 || "\\/:*?\"<>|".contains(c))
}

impl Fat32 {
    pub fn mount(cache: Arc<Cache>) -> Result<Arc<Fat32>, FsError> {
        let b = cache.read_vec(0, 512).map_err(|_| FsError::Io)?;
        if b[510] != 0x55 || b[511] != 0xAA {
            return Err(FsError::Corrupt("no boot signature"));
        }
        let bps = u64::from(le16(&b, 11));
        let spc = u64::from(b[13]);
        let reserved = u64::from(le16(&b, 14));
        let num_fats = u64::from(b[16]);
        let root_entries = le16(&b, 17);
        let total16 = u64::from(le16(&b, 19));
        let fat16 = le16(&b, 22);
        let total = if total16 != 0 {
            total16
        } else {
            u64::from(le32(&b, 32))
        };
        let fat_sectors = u64::from(le32(&b, 36));
        let root_cluster = le32(&b, 44);
        let fsinfo_sector = u64::from(le16(&b, 48));
        if !(512..=4096).contains(&bps) || !bps.is_power_of_two() {
            return Err(FsError::Corrupt("bad sector size"));
        }
        if spc == 0 || !spc.is_power_of_two() || spc > 128 {
            return Err(FsError::Corrupt("bad sectors per cluster"));
        }
        if root_entries != 0 || fat16 != 0 || fat_sectors == 0 || num_fats == 0 || reserved == 0 {
            return Err(FsError::Corrupt("not a FAT32 volume"));
        }
        let data_sector = reserved + num_fats * fat_sectors;
        if total <= data_sector || total * bps > cache.size() {
            return Err(FsError::Corrupt("volume larger than its device"));
        }
        let cluster_count = ((total - data_sector) / spc) as u32;
        if cluster_count < 65525 {
            return Err(FsError::Corrupt("too few clusters for FAT32"));
        }
        if u64::from(cluster_count) + 2 > fat_sectors * bps / 4 {
            return Err(FsError::Corrupt("FAT too small for the cluster count"));
        }
        if root_cluster < 2 || root_cluster >= cluster_count + 2 {
            return Err(FsError::Corrupt("bad root cluster"));
        }
        let fsinfo_off =
            (fsinfo_sector != 0 && fsinfo_sector < reserved).then_some(fsinfo_sector * bps);
        let mut fs = Fat32 {
            cache,
            bps,
            cluster_bytes: bps * spc,
            fat_off: reserved * bps,
            fat_bytes: fat_sectors * bps,
            num_fats,
            data_off: data_sector * bps,
            cluster_count,
            root_cluster,
            fsinfo_off,
            state: YieldMutex::new(State {
                free: 0,
                next_free: 2,
                chains: BTreeMap::new(),
            }),
        };
        // Count free clusters once; FSInfo is a hint that may be stale.
        let free = fs.count_free()?;
        fs.state.get_mut_unchecked().free = free;
        Ok(Arc::new(fs))
    }

    fn count_free(&self) -> Result<u32, FsError> {
        let total = self.cluster_count as usize + 2;
        let bytes = self
            .cache
            .read_vec(self.fat_off, total * 4)
            .map_err(|_| FsError::Io)?;
        Ok(bytes
            .chunks_exact(4)
            .skip(2)
            .filter(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0x0FFF_FFFF == 0)
            .count() as u32)
    }

    fn cluster_off(&self, c: u32) -> u64 {
        self.data_off + u64::from(c - 2) * self.cluster_bytes
    }

    fn valid_cluster(&self, c: u32) -> bool {
        c >= 2 && c < self.cluster_count + 2
    }

    fn fat_get(&self, c: u32) -> Result<u32, FsError> {
        let mut b = [0u8; 4];
        self.cache
            .read(self.fat_off + u64::from(c) * 4, &mut b)
            .map_err(|_| FsError::Io)?;
        Ok(u32::from_le_bytes(b) & 0x0FFF_FFFF)
    }

    fn fat_set(&self, c: u32, v: u32) -> Result<(), FsError> {
        let off = u64::from(c) * 4;
        let mut b = [0u8; 4];
        self.cache
            .read(self.fat_off + off, &mut b)
            .map_err(|_| FsError::Io)?;
        let merged = (u32::from_le_bytes(b) & 0xF000_0000) | (v & 0x0FFF_FFFF);
        for copy in 0..self.num_fats {
            self.cache
                .write(
                    self.fat_off + copy * self.fat_bytes + off,
                    &merged.to_le_bytes(),
                )
                .map_err(|_| FsError::Io)?;
        }
        Ok(())
    }

    /// The cluster chain starting at `first`, cached.
    fn chain(&self, st: &mut State, first: u32) -> Result<Vec<u32>, FsError> {
        if first == 0 {
            return Ok(Vec::new());
        }
        if let Some(c) = st.chains.get(&first) {
            return Ok(c.clone());
        }
        let mut v = Vec::new();
        let mut c = first;
        loop {
            if !self.valid_cluster(c) {
                return Err(FsError::Corrupt("cluster chain leaves the volume"));
            }
            v.push(c);
            if v.len() > self.cluster_count as usize {
                return Err(FsError::Corrupt("cluster chain loops"));
            }
            let next = self.fat_get(c)?;
            if next >= EOC_MIN {
                break;
            }
            if next == BAD || next == 0 {
                return Err(FsError::Corrupt("cluster chain hits a free or bad cluster"));
            }
            c = next;
        }
        if st.chains.len() > 256 {
            st.chains.clear();
        }
        st.chains.insert(first, v.clone());
        Ok(v)
    }

    fn alloc_cluster(&self, st: &mut State, prev: Option<u32>) -> Result<u32, FsError> {
        let total = self.cluster_count + 2;
        let mut c = st.next_free.max(2);
        for _ in 0..self.cluster_count {
            if c >= total {
                c = 2;
            }
            if self.fat_get(c)? == 0 {
                self.fat_set(c, EOC)?;
                if let Some(p) = prev {
                    self.fat_set(p, c)?;
                }
                let zeros = vec![0u8; self.cluster_bytes as usize];
                self.cache
                    .write(self.cluster_off(c), &zeros)
                    .map_err(|_| FsError::Io)?;
                st.next_free = c + 1;
                st.free = st.free.saturating_sub(1);
                return Ok(c);
            }
            c += 1;
        }
        Err(FsError::NoSpace)
    }

    fn free_chain(&self, st: &mut State, first: u32) -> Result<(), FsError> {
        let chain = self.chain(st, first)?;
        for &c in &chain {
            self.fat_set(c, 0)?;
        }
        st.free += chain.len() as u32;
        st.chains.remove(&first);
        Ok(())
    }

    fn write_fsinfo(&self, st: &State) -> Result<(), FsError> {
        let Some(off) = self.fsinfo_off else {
            return Ok(());
        };
        let b = self.cache.read_vec(off, 512).map_err(|_| FsError::Io)?;
        if le32(&b, 0) != 0x4161_5252 || le32(&b, 484) != 0x6141_7272 {
            return Ok(());
        }
        self.cache
            .write(off + 488, &st.free.to_le_bytes())
            .map_err(|_| FsError::Io)?;
        self.cache
            .write(off + 492, &st.next_free.to_le_bytes())
            .map_err(|_| FsError::Io)
    }

    /// Whole directory content and the device offset of each of its clusters.
    fn read_dir_raw(&self, st: &mut State, first: u32) -> Result<(Vec<u8>, Vec<u32>), FsError> {
        let chain = self.chain(st, first)?;
        let mut data = vec![0u8; chain.len() * self.cluster_bytes as usize];
        for (i, &c) in chain.iter().enumerate() {
            let o = i * self.cluster_bytes as usize;
            self.cache
                .read(
                    self.cluster_off(c),
                    &mut data[o..o + self.cluster_bytes as usize],
                )
                .map_err(|_| FsError::Io)?;
        }
        Ok((data, chain))
    }

    fn dev_off(&self, chain: &[u32], index: usize) -> u64 {
        let per = (self.cluster_bytes / 32) as usize;
        self.cluster_off(chain[index / per]) + (index % per) as u64 * 32
    }

    fn parse_dir(&self, data: &[u8], chain: &[u32]) -> Result<Vec<Raw>, FsError> {
        let mut out = Vec::new();
        let mut lfn: Vec<(u8, [u16; 13], u8, u64)> = Vec::new();
        for i in 0..data.len() / 32 {
            let e = &data[i * 32..i * 32 + 32];
            let off = self.dev_off(chain, i);
            if e[0] == 0x00 {
                break;
            }
            if e[0] == 0xE5 {
                lfn.clear();
                continue;
            }
            if e[11] & 0x3F == ATTR_LFN {
                let mut u = [0u16; 13];
                for (k, slot) in u.iter_mut().enumerate() {
                    let o = match k {
                        0..=4 => 1 + 2 * k,
                        5..=10 => 14 + 2 * (k - 5),
                        _ => 28 + 2 * (k - 11),
                    };
                    *slot = le16(e, o);
                }
                if e[0] & 0x40 != 0 {
                    lfn.clear();
                }
                lfn.push((e[0], u, e[13], off));
                continue;
            }
            if e[11] & ATTR_VOLUME != 0 {
                lfn.clear();
                continue;
            }
            let mut short = [0u8; 11];
            short.copy_from_slice(&e[..11]);
            let mut name = short_to_string(&short, e[12]);
            let mut offsets = vec![off];
            // Use the long name only if its sequence and checksum are intact.
            let ok = !lfn.is_empty()
                && lfn.iter().all(|l| l.2 == lfn_checksum(&short))
                && lfn
                    .iter()
                    .enumerate()
                    .all(|(k, l)| usize::from(l.0 & 0x3F) == lfn.len() - k);
            if ok {
                let mut units: Vec<u16> = Vec::new();
                for l in lfn.iter().rev() {
                    units.extend_from_slice(&l.1);
                }
                if let Some(end) = units.iter().position(|&u| u == 0) {
                    units.truncate(end);
                }
                name = char::decode_utf16(units)
                    .map(|r| r.unwrap_or('\u{FFFD}'))
                    .collect();
                offsets = lfn.iter().map(|l| l.3).chain([off]).collect();
            }
            lfn.clear();
            out.push(Raw {
                name,
                short,
                attr: e[11],
                cluster: (u32::from(le16(e, 20)) << 16) | u32::from(le16(e, 26)),
                size: le32(e, 28),
                short_off: off,
                offsets,
            });
        }
        Ok(out)
    }

    fn node_of(&self, r: &Raw) -> Node {
        Node {
            kind: if r.attr & ATTR_DIR != 0 {
                Kind::Dir
            } else {
                Kind::File
            },
            size: if r.attr & ATTR_DIR != 0 {
                0
            } else {
                u64::from(r.size)
            },
            data: [u64::from(r.cluster), r.short_off, 0],
        }
    }

    fn entries_of(&self, st: &mut State, dir: &Node) -> Result<Vec<Raw>, FsError> {
        if dir.kind != Kind::Dir {
            return Err(FsError::NotDir);
        }
        let first = if dir.data[0] == 0 {
            self.root_cluster
        } else {
            dir.data[0] as u32
        };
        let (data, chain) = self.read_dir_raw(st, first)?;
        self.parse_dir(&data, &chain)
    }

    fn update_entry(&self, node: &Node) -> Result<(), FsError> {
        let off = node.data[1];
        if off == 0 {
            return Ok(()); // the root has no entry
        }
        let c = node.data[0] as u32;
        self.cache
            .write(off + 20, &((c >> 16) as u16).to_le_bytes())
            .map_err(|_| FsError::Io)?;
        self.cache
            .write(off + 26, &(c as u16).to_le_bytes())
            .map_err(|_| FsError::Io)?;
        if node.kind == Kind::File {
            self.cache
                .write(off + 28, &(node.size as u32).to_le_bytes())
                .map_err(|_| FsError::Io)?;
        }
        Ok(())
    }

    // --- short names ------------------------------------------------------

    fn make_short(&self, name: &str, existing: &[Raw]) -> ([u8; 11], bool, u8) {
        let up: String = name.chars().flat_map(char::to_uppercase).collect();
        let clean = |s: &str, max: usize| -> (Vec<u8>, bool) {
            let mut lossy = false;
            let mut v = Vec::new();
            for ch in s.chars() {
                if ch == ' ' || ch == '.' {
                    lossy = true;
                    continue;
                }
                if v.len() >= max {
                    lossy = true;
                    continue;
                }
                if ch.is_ascii_alphanumeric() || "$%'-_@~`!(){}^#&".contains(ch) {
                    v.push(ch as u8);
                } else {
                    v.push(b'_');
                    lossy = true;
                }
            }
            (v, lossy)
        };
        let (stem, ext) = match up.rfind('.') {
            Some(i) if i > 0 => (&up[..i], &up[i + 1..]),
            _ => (&up[..], ""),
        };
        let (b, lb) = clean(stem, 8);
        let (e, le) = clean(ext, 3);
        let has_dot_inside = stem.contains('.') || name.starts_with('.');
        let fits =
            !lb && !le && !has_dot_inside && !b.is_empty() && stem.len() <= 8 && ext.len() <= 3;
        let only_lower = name == name.to_lowercase();
        let only_upper = name == up;
        if fits && (only_lower || only_upper) {
            let mut s = [b' '; 11];
            s[..b.len()].copy_from_slice(&b);
            s[8..8 + e.len()].copy_from_slice(&e);
            let taken = existing.iter().any(|r| r.short == s);
            if !taken {
                let mut flags = 0u8;
                if only_lower && name.chars().any(|c| c.is_alphabetic()) {
                    // The case flags apply to the part that has letters.
                    if stem.chars().any(char::is_alphabetic) {
                        flags |= 0x08;
                    }
                    if ext.chars().any(char::is_alphabetic) {
                        flags |= 0x10;
                    }
                }
                return (s, false, flags);
            }
        }
        // Numeric tail.
        let base: Vec<u8> = if b.is_empty() { b"_".to_vec() } else { b };
        for n in 1u32.. {
            let tail = alloc::format!("~{n}");
            let keep = 8 - tail.len();
            let mut s = [b' '; 11];
            let bl = base.len().min(keep);
            s[..bl].copy_from_slice(&base[..bl]);
            s[bl..bl + tail.len()].copy_from_slice(tail.as_bytes());
            s[8..8 + e.len()].copy_from_slice(&e);
            if !existing.iter().any(|r| r.short == s) {
                return (s, true, 0);
            }
        }
        unreachable!()
    }

    /// Write the records for a new entry into `dir`, growing it if needed.
    fn add_entry(
        &self,
        st: &mut State,
        dir: &Node,
        name: &str,
        attr: u8,
        cluster: u32,
    ) -> Result<u64, FsError> {
        let first = if dir.data[0] == 0 {
            self.root_cluster
        } else {
            dir.data[0] as u32
        };
        let (mut data, mut chain) = self.read_dir_raw(st, first)?;
        let existing = self.parse_dir(&data, &chain)?;
        if existing.iter().any(|r| upper_eq(&r.name, name)) {
            return Err(FsError::Exists);
        }
        let (short, needs_lfn, ntres) = self.make_short(name, &existing);
        let units: Vec<u16> = name.encode_utf16().collect();
        let lfn_count = if !needs_lfn && short_to_string(&short, ntres) == name {
            0
        } else {
            units.len().div_ceil(13)
        };
        let need = lfn_count + 1;
        // First run of `need` free records (deleted, or past the end marker).
        let per = data.len() / 32;
        let mut start = None;
        let mut run = 0usize;
        for i in 0..per {
            let free = data[i * 32] == 0x00 || data[i * 32] == 0xE5;
            if free {
                run += 1;
                if run == need {
                    start = Some(i + 1 - need);
                    break;
                }
            } else {
                run = 0;
            }
        }
        let start = match start {
            Some(s) => s,
            None => {
                // Extend with zeroed clusters until the run fits at the end.
                let mut tail = 0usize;
                while tail < per && data[(per - 1 - tail) * 32] == 0x00 {
                    tail += 1;
                }
                let mut start = per - tail;
                let mut total = per;
                while total - start < need {
                    let last = *chain.last().unwrap();
                    let c = self.alloc_cluster(st, Some(last))?;
                    chain.push(c);
                    data.extend(vec![0u8; self.cluster_bytes as usize]);
                    total = data.len() / 32;
                }
                st.chains.remove(&first);
                if start > total {
                    start = total;
                }
                start
            }
        };
        let sum = lfn_checksum(&short);
        for k in 0..lfn_count {
            // Records are stored last-piece-first.
            let piece = lfn_count - 1 - k;
            let mut e = [0u8; 32];
            e[0] = (piece as u8 + 1) | if k == 0 { 0x40 } else { 0 };
            e[11] = ATTR_LFN;
            e[13] = sum;
            for j in 0..13 {
                let idx = piece * 13 + j;
                let u = if idx < units.len() {
                    units[idx]
                } else if idx == units.len() {
                    0
                } else {
                    0xFFFF
                };
                let o = match j {
                    0..=4 => 1 + 2 * j,
                    5..=10 => 14 + 2 * (j - 5),
                    _ => 28 + 2 * (j - 11),
                };
                e[o..o + 2].copy_from_slice(&u.to_le_bytes());
            }
            self.cache
                .write(self.dev_off(&chain, start + k), &e)
                .map_err(|_| FsError::Io)?;
        }
        let mut e = [0u8; 32];
        e[..11].copy_from_slice(&short);
        e[11] = attr;
        e[12] = if lfn_count == 0 { ntres } else { 0 };
        for o in [14usize, 24] {
            e[o..o + 2].copy_from_slice(&DOS_DATE.to_le_bytes());
        }
        e[16..18].copy_from_slice(&DOS_DATE.to_le_bytes());
        e[18..20].copy_from_slice(&DOS_DATE.to_le_bytes());
        e[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
        e[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
        let off = self.dev_off(&chain, start + lfn_count);
        self.cache.write(off, &e).map_err(|_| FsError::Io)?;
        Ok(off)
    }
}

impl FileSystem for Fat32 {
    fn root(&self) -> Node {
        Node {
            kind: Kind::Dir,
            size: 0,
            data: [0, 0, 0],
        }
    }

    fn lookup(&self, dir: &Node, name: &str) -> Result<Node, FsError> {
        let mut st = self.state.lock();
        let entries = self.entries_of(&mut st, dir)?;
        entries
            .iter()
            .find(|r| upper_eq(&r.name, name))
            .map(|r| self.node_of(r))
            .ok_or(FsError::NotFound)
    }

    fn read_dir(&self, dir: &Node) -> Result<Vec<DirEntry>, FsError> {
        let mut st = self.state.lock();
        let entries = self.entries_of(&mut st, dir)?;
        Ok(entries
            .iter()
            .filter(|r| r.name != "." && r.name != "..")
            .map(|r| DirEntry {
                name: r.name.clone(),
                node: self.node_of(r),
            })
            .collect())
    }

    fn read(&self, file: &Node, offset: u64, buf: &mut [u8]) -> Result<usize, FsError> {
        if file.kind != Kind::File {
            return Err(FsError::IsDir);
        }
        let mut st = self.state.lock();
        if offset >= file.size {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(file.size - offset) as usize;
        let chain = self.chain(&mut st, file.data[0] as u32)?;
        let cb = self.cluster_bytes;
        let mut done = 0usize;
        while done < want {
            let pos = offset + done as u64;
            let ci = (pos / cb) as usize;
            let Some(&c) = chain.get(ci) else {
                return Err(FsError::Corrupt("file longer than its cluster chain"));
            };
            // Merge physically contiguous clusters into one cache read.
            let mut run = 1usize;
            while ci + run < chain.len()
                && chain[ci + run] == chain[ci + run - 1] + 1
                && (run as u64 + 1) * cb < (want - done) as u64 + cb
            {
                run += 1;
            }
            let within = pos % cb;
            let n = ((run as u64 * cb - within) as usize).min(want - done);
            self.cache
                .read(self.cluster_off(c) + within, &mut buf[done..done + n])
                .map_err(|_| FsError::Io)?;
            done += n;
        }
        Ok(done)
    }

    fn create(&self, dir: &Node, name: &str, kind: Kind) -> Result<Node, FsError> {
        if !valid_long_name(name) {
            return Err(FsError::Invalid);
        }
        if kind == Kind::Symlink {
            return Err(FsError::Unsupported("symbolic links on FAT"));
        }
        let mut st = self.state.lock();
        if dir.kind != Kind::Dir {
            return Err(FsError::NotDir);
        }
        let (attr, cluster) = if kind == Kind::Dir {
            let c = self.alloc_cluster(&mut st, None)?;
            let parent = if dir.data[0] == 0 || dir.data[0] as u32 == self.root_cluster {
                0
            } else {
                dir.data[0] as u32
            };
            let mut dots = [0u8; 64];
            dots[..11].copy_from_slice(b".          ");
            dots[32..43].copy_from_slice(b"..         ");
            for (i, tgt) in [c, parent].iter().enumerate() {
                let e = &mut dots[i * 32..i * 32 + 32];
                e[11] = ATTR_DIR;
                for o in [14usize, 16, 18, 24] {
                    e[o..o + 2].copy_from_slice(&DOS_DATE.to_le_bytes());
                }
                e[20..22].copy_from_slice(&((tgt >> 16) as u16).to_le_bytes());
                e[26..28].copy_from_slice(&(*tgt as u16).to_le_bytes());
            }
            self.cache
                .write(self.cluster_off(c), &dots)
                .map_err(|_| FsError::Io)?;
            (ATTR_DIR, c)
        } else {
            (0x20, 0)
        };
        let off = match self.add_entry(&mut st, dir, name, attr, cluster) {
            Ok(o) => o,
            Err(e) => {
                if cluster != 0 {
                    let _ = self.free_chain(&mut st, cluster);
                }
                return Err(e);
            }
        };
        self.write_fsinfo(&st)?;
        Ok(Node {
            kind,
            size: 0,
            data: [u64::from(cluster), off, 0],
        })
    }

    fn write(&self, file: &Node, offset: u64, data: &[u8]) -> Result<usize, FsError> {
        if file.kind != Kind::File {
            return Err(FsError::IsDir);
        }
        if data.is_empty() {
            return Ok(0);
        }
        let end = offset
            .checked_add(data.len() as u64)
            .filter(|&e| e <= u64::from(u32::MAX))
            .ok_or(FsError::NoSpace)?;
        let mut st = self.state.lock();
        let mut node = *file;
        let cb = self.cluster_bytes;
        let mut chain = self.chain(&mut st, node.data[0] as u32)?;
        let need = end.div_ceil(cb) as usize;
        while chain.len() < need {
            let c = self.alloc_cluster(&mut st, chain.last().copied())?;
            if chain.is_empty() {
                node.data[0] = u64::from(c);
            }
            chain.push(c);
        }
        st.chains.remove(&(file.data[0] as u32));
        st.chains.insert(node.data[0] as u32, chain.clone());
        // Zero the gap between the old end and a write that starts beyond it.
        if offset > node.size {
            let mut pos = node.size;
            while pos < offset {
                let n = (cb - pos % cb).min(offset - pos);
                let c = chain[(pos / cb) as usize];
                self.cache
                    .write(self.cluster_off(c) + pos % cb, &vec![0u8; n as usize])
                    .map_err(|_| FsError::Io)?;
                pos += n;
            }
        }
        let mut done = 0usize;
        while done < data.len() {
            let pos = offset + done as u64;
            let ci = (pos / cb) as usize;
            let mut run = 1usize;
            while ci + run < chain.len()
                && chain[ci + run] == chain[ci + run - 1] + 1
                && (run as u64 + 1) * cb < (data.len() - done) as u64 + cb
            {
                run += 1;
            }
            let within = pos % cb;
            let n = ((run as u64 * cb - within) as usize).min(data.len() - done);
            self.cache
                .write(self.cluster_off(chain[ci]) + within, &data[done..done + n])
                .map_err(|_| FsError::Io)?;
            done += n;
        }
        if end > node.size {
            node.size = end;
        }
        self.update_entry(&node)?;
        self.write_fsinfo(&st)?;
        Ok(data.len())
    }

    fn truncate(&self, file: &Node, size: u64) -> Result<Node, FsError> {
        if file.kind != Kind::File {
            return Err(FsError::IsDir);
        }
        if size > u64::from(u32::MAX) {
            return Err(FsError::NoSpace);
        }
        if size > file.size {
            // Growing is a write of zeros at the end.
            let zeros = vec![0u8; (size - file.size) as usize];
            self.write(file, file.size, &zeros)?;
            return self.refresh(file);
        }
        let mut st = self.state.lock();
        let mut node = *file;
        let cb = self.cluster_bytes;
        let chain = self.chain(&mut st, node.data[0] as u32)?;
        let keep = size.div_ceil(cb) as usize;
        if keep < chain.len() {
            for &c in &chain[keep..] {
                self.fat_set(c, 0)?;
            }
            st.free += (chain.len() - keep) as u32;
            if keep == 0 {
                node.data[0] = 0;
            } else {
                self.fat_set(chain[keep - 1], EOC)?;
            }
        }
        st.chains.remove(&(file.data[0] as u32));
        node.size = size;
        self.update_entry(&node)?;
        self.write_fsinfo(&st)?;
        Ok(node)
    }

    fn remove(&self, dir: &Node, name: &str) -> Result<(), FsError> {
        let mut st = self.state.lock();
        let entries = self.entries_of(&mut st, dir)?;
        let r = entries
            .iter()
            .find(|r| upper_eq(&r.name, name))
            .ok_or(FsError::NotFound)?
            .clone();
        if r.attr & ATTR_DIR != 0 {
            let sub = Node {
                kind: Kind::Dir,
                size: 0,
                data: [u64::from(r.cluster), r.short_off, 0],
            };
            let inner = self.entries_of(&mut st, &sub)?;
            if inner.iter().any(|e| e.name != "." && e.name != "..") {
                return Err(FsError::NotEmpty);
            }
        }
        for &o in &r.offsets {
            self.cache.write(o, &[0xE5]).map_err(|_| FsError::Io)?;
        }
        if r.cluster != 0 {
            self.free_chain(&mut st, r.cluster)?;
        }
        self.write_fsinfo(&st)
    }

    fn refresh(&self, node: &Node) -> Result<Node, FsError> {
        if node.data[1] == 0 || node.kind != Kind::File {
            return Ok(*node);
        }
        let mut e = [0u8; 32];
        self.cache
            .read(node.data[1], &mut e)
            .map_err(|_| FsError::Io)?;
        let mut n = *node;
        n.size = u64::from(le32(&e, 28));
        n.data[0] = (u64::from(le16(&e, 20)) << 16) | u64::from(le16(&e, 26));
        Ok(n)
    }

    fn sync(&self) -> Result<(), FsError> {
        let st = self.state.lock();
        self.write_fsinfo(&st)?;
        self.cache.flush().map_err(|_| FsError::Io)
    }
}
