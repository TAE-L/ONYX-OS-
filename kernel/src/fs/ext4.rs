//! ext4: extents, 64-bit block numbers, flex_bg, metadata_csum, large directories.
//!
//! Refused on mount (rather than guessed at): a journal that needs recovery,
//! inline data, bigalloc, meta_bg, encryption, casefolding and indirect-block
//! files. Unknown read-only-compatible features make the mount read-only.

#![allow(dead_code)]

use crate::cache::Cache;
use crate::sync::YieldMutex;
use crate::vfs::{DirEntry, FileSystem, FsError, Kind, Node};
use alloc::{string::String, sync::Arc, vec, vec::Vec};

const MAGIC: u16 = 0xEF53;
const ROOT_INO: u32 = 2;

const INCOMPAT_FILETYPE: u32 = 0x2;
const INCOMPAT_RECOVER: u32 = 0x4;
const INCOMPAT_EXTENTS: u32 = 0x40;
const INCOMPAT_64BIT: u32 = 0x80;
const INCOMPAT_FLEX_BG: u32 = 0x200;
const INCOMPAT_CSUM_SEED: u32 = 0x2000;
const INCOMPAT_SUPPORTED: u32 = INCOMPAT_FILETYPE
    | INCOMPAT_EXTENTS
    | INCOMPAT_64BIT
    | INCOMPAT_FLEX_BG
    | INCOMPAT_CSUM_SEED
    | 0x1; // compression flag bit is never set by mke2fs; compat 'dir_prealloc' aside

const RO_SPARSE_SUPER: u32 = 0x1;
const RO_LARGE_FILE: u32 = 0x2;
const RO_HUGE_FILE: u32 = 0x8;
const RO_GDT_CSUM: u32 = 0x10;
const RO_DIR_NLINK: u32 = 0x20;
const RO_EXTRA_ISIZE: u32 = 0x40;
const RO_METADATA_CSUM: u32 = 0x400;
const RO_KNOWN: u32 = RO_SPARSE_SUPER
    | RO_LARGE_FILE
    | RO_HUGE_FILE
    | RO_GDT_CSUM
    | RO_DIR_NLINK
    | RO_EXTRA_ISIZE
    | RO_METADATA_CSUM;

const FL_INDEX: u32 = 0x1000;
const FL_HUGE_FILE: u32 = 0x4_0000;
const FL_EXTENTS: u32 = 0x8_0000;
const FL_INLINE_DATA: u32 = 0x1000_0000;

const S_IFMT: u16 = 0xF000;
const S_IFDIR: u16 = 0x4000;
const S_IFREG: u16 = 0x8000;
const S_IFLNK: u16 = 0xA000;

const EXT_MAGIC: u16 = 0xF30A;
const MAX_SYMLINK: usize = 4095;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x82F6_3B78
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}
static CRC_TABLE: [u32; 256] = crc_table();

/// CRC-32C as ext4 uses it: running value in, running value out, no final inversion.
pub fn crc32c(seed: u32, data: &[u8]) -> u32 {
    data.iter().fold(seed, |c, &b| {
        CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8)
    })
}

pub struct Ext4 {
    cache: Arc<Cache>,
    block_size: u64,
    blocks_count: u64,
    first_data_block: u64,
    blocks_per_group: u64,
    inodes_per_group: u64,
    inodes_count: u32,
    inode_size: u64,
    desc_size: u64,
    groups: u64,
    gdt_block: u64,
    incompat: u32,
    ro_compat: u32,
    pub read_only: bool,
    has_csum: bool,
    csum_seed: u32,
    first_ino: u32,
    lock: YieldMutex<()>,
}

/// The parts of an inode the driver uses.
#[derive(Clone)]
struct Inode {
    mode: u16,
    size: u64,
    flags: u32,
    blocks_lo: u32,
    block: [u8; 60],
}

#[derive(Clone, Copy, Debug)]
struct Extent {
    logical: u32,
    len: u32,
    physical: u64,
    uninit: bool,
}

impl Ext4 {
    pub fn mount(cache: Arc<Cache>) -> Result<Arc<Ext4>, FsError> {
        let sb = cache.read_vec(1024, 1024).map_err(|_| FsError::Io)?;
        if le16(&sb, 56) != MAGIC {
            return Err(FsError::Corrupt("no ext4 magic"));
        }
        let log_bs = le32(&sb, 24);
        if log_bs > 6 {
            return Err(FsError::Corrupt("absurd block size"));
        }
        let block_size = 1024u64 << log_bs;
        let incompat = le32(&sb, 96);
        let ro_compat = le32(&sb, 100);
        if incompat & INCOMPAT_RECOVER != 0 {
            return Err(FsError::Unsupported("journal needs recovery"));
        }
        if incompat & !INCOMPAT_SUPPORTED != 0 {
            return Err(FsError::Unsupported("incompatible feature"));
        }
        if incompat & INCOMPAT_EXTENTS == 0 {
            return Err(FsError::Unsupported("indirect-block filesystems"));
        }
        if ro_compat & 0x200 != 0 {
            return Err(FsError::Unsupported("bigalloc"));
        }
        let is64 = incompat & INCOMPAT_64BIT != 0;
        let desc_size = if is64 { u64::from(le16(&sb, 254)) } else { 32 };
        if !(32..=1024).contains(&desc_size) || (is64 && desc_size < 64) {
            return Err(FsError::Corrupt("bad group descriptor size"));
        }
        let blocks_count = u64::from(le32(&sb, 4))
            | if is64 {
                u64::from(le32(&sb, 336)) << 32
            } else {
                0
            };
        let first_data_block = u64::from(le32(&sb, 20));
        let blocks_per_group = u64::from(le32(&sb, 32));
        let inodes_per_group = u64::from(le32(&sb, 40));
        let inode_size = if le32(&sb, 76) == 0 {
            128
        } else {
            u64::from(le16(&sb, 88))
        };
        let inodes_count = le32(&sb, 0);
        if blocks_per_group == 0 || inodes_per_group == 0 || blocks_count == 0 {
            return Err(FsError::Corrupt("zero geometry"));
        }
        if blocks_count * block_size > cache.size() {
            return Err(FsError::Corrupt("filesystem larger than its device"));
        }
        if inode_size < 128 || inode_size > block_size || !inode_size.is_power_of_two() {
            return Err(FsError::Corrupt("bad inode size"));
        }
        let groups = (blocks_count - first_data_block).div_ceil(blocks_per_group);
        if groups * inodes_per_group < u64::from(inodes_count) {
            return Err(FsError::Corrupt("inode count exceeds group capacity"));
        }
        let read_only = ro_compat & !RO_KNOWN != 0 || le16(&sb, 58) & 2 != 0; // errors recorded
        Ok(Arc::new(Ext4 {
            cache,
            block_size,
            blocks_count,
            first_data_block,
            blocks_per_group,
            inodes_per_group,
            inodes_count,
            inode_size,
            desc_size,
            groups,
            gdt_block: first_data_block + 1,
            incompat,
            ro_compat,
            read_only: read_only
                || (ro_compat & RO_GDT_CSUM != 0 && ro_compat & RO_METADATA_CSUM == 0),
            has_csum: ro_compat & RO_METADATA_CSUM != 0,
            csum_seed: if incompat & INCOMPAT_CSUM_SEED != 0 {
                le32(&sb, 0x270)
            } else {
                crc32c(!0, &sb[104..120])
            },
            first_ino: if le32(&sb, 76) == 0 {
                11
            } else {
                le32(&sb, 84)
            },
            lock: YieldMutex::new(()),
        }))
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), FsError> {
        if block >= self.blocks_count {
            return Err(FsError::Corrupt("block number outside the filesystem"));
        }
        self.cache
            .read(block * self.block_size, buf)
            .map_err(|_| FsError::Io)
    }

    fn group_desc(&self, group: u64) -> Result<Vec<u8>, FsError> {
        if group >= self.groups {
            return Err(FsError::Corrupt("group out of range"));
        }
        self.cache
            .read_vec(
                self.gdt_block * self.block_size + group * self.desc_size,
                self.desc_size as usize,
            )
            .map_err(|_| FsError::Io)
    }

    fn inode_table(&self, group: u64) -> Result<u64, FsError> {
        let d = self.group_desc(group)?;
        let lo = u64::from(le32(&d, 8));
        let hi = if self.desc_size >= 64 {
            u64::from(le32(&d, 40))
        } else {
            0
        };
        Ok(lo | hi << 32)
    }

    fn inode_offset(&self, ino: u32) -> Result<u64, FsError> {
        if ino == 0 || ino > self.inodes_count {
            return Err(FsError::Corrupt("inode number out of range"));
        }
        let idx = u64::from(ino - 1);
        let table = self.inode_table(idx / self.inodes_per_group)?;
        if table == 0 || table >= self.blocks_count {
            return Err(FsError::Corrupt("inode table outside the filesystem"));
        }
        Ok(table * self.block_size + (idx % self.inodes_per_group) * self.inode_size)
    }

    fn read_inode(&self, ino: u32) -> Result<Inode, FsError> {
        let raw = self
            .cache
            .read_vec(self.inode_offset(ino)?, self.inode_size as usize)
            .map_err(|_| FsError::Io)?;
        let mut block = [0u8; 60];
        block.copy_from_slice(&raw[40..100]);
        let size = u64::from(le32(&raw, 4)) | u64::from(le32(&raw, 108)) << 32;
        Ok(Inode {
            mode: le16(&raw, 0),
            size,
            flags: le32(&raw, 32),
            blocks_lo: le32(&raw, 28),
            block,
        })
    }

    fn node_for(&self, ino: u32) -> Result<Node, FsError> {
        let i = self.read_inode(ino)?;
        if i.flags & FL_INLINE_DATA != 0 {
            return Err(FsError::Unsupported("inline data"));
        }
        let kind = match i.mode & S_IFMT {
            S_IFDIR => Kind::Dir,
            S_IFREG => Kind::File,
            S_IFLNK => Kind::Symlink,
            _ => return Err(FsError::Unsupported("special files")),
        };
        Ok(Node {
            kind,
            size: i.size,
            data: [u64::from(ino), 0, 0],
        })
    }

    /// Every extent of the inode in logical order.
    fn extents(&self, inode: &Inode) -> Result<Vec<Extent>, FsError> {
        if inode.flags & FL_EXTENTS == 0 {
            return Err(FsError::Unsupported("file without extents"));
        }
        let mut out = Vec::new();
        self.walk(&inode.block, 0, &mut out, &mut Vec::new())?;
        out.sort_by_key(|e| e.logical);
        Ok(out)
    }

    fn walk(
        &self,
        node: &[u8],
        depth_seen: u32,
        out: &mut Vec<Extent>,
        tree_blocks: &mut Vec<u64>,
    ) -> Result<(), FsError> {
        if depth_seen > 5 || node.len() < 12 || le16(node, 0) != EXT_MAGIC {
            return Err(FsError::Corrupt("bad extent header"));
        }
        let entries = usize::from(le16(node, 2));
        let depth = le16(node, 6);
        if 12 + entries * 12 > node.len() {
            return Err(FsError::Corrupt("extent entries overflow their node"));
        }
        for i in 0..entries {
            let e = &node[12 + i * 12..24 + i * 12];
            if depth == 0 {
                let raw_len = u32::from(le16(e, 4));
                let (len, uninit) = if raw_len > 32768 {
                    (raw_len - 32768, true)
                } else {
                    (raw_len, false)
                };
                let physical = u64::from(le32(e, 8)) | u64::from(le16(e, 6)) << 32;
                if len == 0 || physical + u64::from(len) > self.blocks_count {
                    return Err(FsError::Corrupt("extent outside the filesystem"));
                }
                out.push(Extent {
                    logical: le32(e, 0),
                    len,
                    physical,
                    uninit,
                });
            } else {
                let child = u64::from(le32(e, 4)) | u64::from(le16(e, 8)) << 32;
                let mut buf = vec![0u8; self.block_size as usize];
                self.read_block(child, &mut buf)?;
                tree_blocks.push(child);
                self.walk(&buf, depth_seen + 1, out, tree_blocks)?;
            }
        }
        Ok(())
    }

    fn read_inode_data(
        &self,
        inode: &Inode,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, FsError> {
        if offset >= inode.size {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(inode.size - offset) as usize;
        let buf = &mut buf[..want];
        buf.fill(0); // holes and uninitialised extents read as zeros
        let bs = self.block_size;
        let (first, last) = (offset / bs, (offset + want as u64 - 1) / bs);
        for e in self.extents(inode)? {
            let (e_lo, e_hi) = (
                u64::from(e.logical),
                u64::from(e.logical) + u64::from(e.len) - 1,
            );
            if e_hi < first || e_lo > last || e.uninit {
                continue;
            }
            let (lo, hi) = (e_lo.max(first), e_hi.min(last));
            // Byte range of this extent that the request covers.
            let file_start = (lo * bs).max(offset);
            let file_end = ((hi + 1) * bs).min(offset + want as u64);
            let disk = e.physical * bs + (file_start - e_lo * bs);
            self.cache
                .read(
                    disk,
                    &mut buf[(file_start - offset) as usize..(file_end - offset) as usize],
                )
                .map_err(|_| FsError::Io)?;
        }
        Ok(want)
    }

    fn dir_entries(&self, dir: &Node) -> Result<Vec<(String, u32)>, FsError> {
        if dir.kind != Kind::Dir {
            return Err(FsError::NotDir);
        }
        let inode = self.read_inode(dir.data[0] as u32)?;
        let mut data = vec![0u8; inode.size as usize];
        let n = self.read_inode_data(&inode, 0, &mut data)?;
        data.truncate(n);
        let bs = self.block_size as usize;
        let mut out = Vec::new();
        for block in data.chunks(bs) {
            let mut pos = 0usize;
            while pos + 8 <= block.len() {
                let ino = le32(block, pos);
                let rec_len = usize::from(le16(block, pos + 4));
                let name_len = usize::from(block[pos + 6]);
                if rec_len < 8
                    || rec_len % 4 != 0
                    || pos + rec_len > block.len()
                    || 8 + name_len > rec_len
                {
                    return Err(FsError::Corrupt("bad directory entry"));
                }
                if ino != 0 {
                    let name = core::str::from_utf8(&block[pos + 8..pos + 8 + name_len])
                        .map_err(|_| FsError::Corrupt("directory name is not UTF-8"))?;
                    if name != "." && name != ".." {
                        out.push((String::from(name), ino));
                    }
                }
                pos += rec_len;
            }
        }
        Ok(out)
    }
}

impl FileSystem for Ext4 {
    fn root(&self) -> Node {
        Node {
            kind: Kind::Dir,
            size: 0,
            data: [u64::from(ROOT_INO), 0, 0],
        }
    }

    fn lookup(&self, dir: &Node, name: &str) -> Result<Node, FsError> {
        let _g = self.lock.lock();
        let (_, ino) = self
            .dir_entries(dir)?
            .into_iter()
            .find(|(n, _)| n == name)
            .ok_or(FsError::NotFound)?;
        self.node_for(ino)
    }

    fn read_dir(&self, dir: &Node) -> Result<Vec<DirEntry>, FsError> {
        let _g = self.lock.lock();
        self.dir_entries(dir)?
            .into_iter()
            .map(|(name, ino)| {
                Ok(DirEntry {
                    name,
                    node: self.node_for(ino)?,
                })
            })
            .collect()
    }

    fn read(&self, file: &Node, offset: u64, buf: &mut [u8]) -> Result<usize, FsError> {
        if file.kind == Kind::Dir {
            return Err(FsError::IsDir);
        }
        let _g = self.lock.lock();
        let inode = self.read_inode(file.data[0] as u32)?;
        self.read_inode_data(&inode, offset, buf)
    }

    fn create(&self, dir: &Node, name: &str, kind: Kind) -> Result<Node, FsError> {
        let _g = self.lock.lock();
        self.do_create(dir, name, kind)
    }

    fn write(&self, file: &Node, offset: u64, data: &[u8]) -> Result<usize, FsError> {
        self.writable()?;
        if file.kind != Kind::File {
            return Err(FsError::IsDir);
        }
        if data.is_empty() {
            return Ok(0);
        }
        offset
            .checked_add(data.len() as u64)
            .ok_or(FsError::NoSpace)?;
        let _g = self.lock.lock();
        self.write_data(file.data[0] as u32, offset, data)?;
        Ok(data.len())
    }

    fn truncate(&self, file: &Node, size: u64) -> Result<Node, FsError> {
        self.writable()?;
        if file.kind != Kind::File {
            return Err(FsError::IsDir);
        }
        let _g = self.lock.lock();
        self.truncate_inode(file.data[0] as u32, size)?;
        self.node_for(file.data[0] as u32)
    }

    fn remove(&self, dir: &Node, name: &str) -> Result<(), FsError> {
        let _g = self.lock.lock();
        self.do_remove(dir, name)
    }

    fn refresh(&self, node: &Node) -> Result<Node, FsError> {
        let _g = self.lock.lock();
        self.node_for(node.data[0] as u32)
    }

    fn sync(&self) -> Result<(), FsError> {
        self.cache.flush().map_err(|_| FsError::Io)
    }

    fn read_link(&self, link: &Node) -> Result<String, FsError> {
        if link.kind != Kind::Symlink {
            return Err(FsError::Invalid);
        }
        let _g = self.lock.lock();
        let inode = self.read_inode(link.data[0] as u32)?;
        let len = inode.size as usize;
        if len == 0 || len > MAX_SYMLINK {
            return Err(FsError::Corrupt("bad symlink length"));
        }
        let bytes = if inode.flags & FL_EXTENTS == 0 && len < 60 {
            inode.block[..len].to_vec() // fast symlink: target lives in i_block
        } else {
            let mut v = vec![0u8; len];
            self.read_inode_data(&inode, 0, &mut v)?;
            v
        };
        String::from_utf8(bytes).map_err(|_| FsError::Corrupt("symlink target is not UTF-8"))
    }
}

// =============================================================================
// Write support
//
// Metadata is updated in place with correct checksums. There is no journal
// transaction: the journal is left untouched (and must have been clean at
// mount), so a crash in the middle of an operation can leave the filesystem
// needing e2fsck. Not supported for writing: indexed (htree) directories are
// converted to plain linear ones when modified, groups with uninitialised
// bitmaps are skipped, and files need at most 4 extent-tree leaves.
// =============================================================================

const BG_INODE_UNINIT: u16 = 0x1;
const BG_BLOCK_UNINIT: u16 = 0x2;
/// 2026-01-01 00:00:00 UTC: the kernel has no wall clock yet.
const FIXED_TIME: u32 = 1_767_225_600;

fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn bit(map: &[u8], i: usize) -> bool {
    map[i / 8] & (1 << (i % 8)) != 0
}
fn set_bit(map: &mut [u8], i: usize, on: bool) {
    if on {
        map[i / 8] |= 1 << (i % 8);
    } else {
        map[i / 8] &= !(1 << (i % 8));
    }
}

impl Ext4 {
    fn writable(&self) -> Result<(), FsError> {
        if self.read_only {
            Err(FsError::ReadOnly)
        } else {
            Ok(())
        }
    }

    fn io_write(&self, off: u64, data: &[u8]) -> Result<(), FsError> {
        self.cache.write(off, data).map_err(|_| FsError::Io)
    }

    // --- checksums --------------------------------------------------------

    fn inode_seed(&self, ino: u32, generation: u32) -> u32 {
        crc32c(
            crc32c(self.csum_seed, &ino.to_le_bytes()),
            &generation.to_le_bytes(),
        )
    }

    fn seal_inode(&self, ino: u32, raw: &mut [u8]) {
        if !self.has_csum {
            return;
        }
        let extra = if raw.len() > 128 { le16(raw, 128) } else { 0 };
        let has_hi = raw.len() > 128 && extra >= 4;
        put16(raw, 124, 0);
        if has_hi {
            put16(raw, 130, 0);
        }
        let c = crc32c(self.inode_seed(ino, le32(raw, 100)), raw);
        put16(raw, 124, c as u16);
        if has_hi {
            put16(raw, 130, (c >> 16) as u16);
        }
    }

    fn seal_group_desc(&self, group: u64, d: &mut [u8]) {
        if !self.has_csum {
            return;
        }
        put16(d, 30, 0);
        let mut c = crc32c(self.csum_seed, &(group as u32).to_le_bytes());
        c = crc32c(c, &d[..30]);
        c = crc32c(c, &[0, 0]);
        if d.len() > 32 {
            c = crc32c(c, &d[32..]);
        }
        put16(d, 30, c as u16);
    }

    fn put_group_desc(&self, group: u64, d: &mut [u8]) -> Result<(), FsError> {
        self.seal_group_desc(group, d);
        self.io_write(self.gdt_block * self.block_size + group * self.desc_size, d)
    }

    fn seal_dir_block(&self, ino: u32, generation: u32, block: &mut [u8]) {
        if !self.has_csum {
            return;
        }
        let n = block.len();
        let c = crc32c(self.inode_seed(ino, generation), &block[..n - 12]);
        put32(block, n - 4, c);
    }

    fn seal_extent_block(&self, ino: u32, generation: u32, block: &mut [u8]) {
        if !self.has_csum {
            return;
        }
        let tail = 12 + 12 * usize::from(le16(block, 4));
        let c = crc32c(self.inode_seed(ino, generation), &block[..tail]);
        put32(block, tail, c);
    }

    // --- superblock counters ---------------------------------------------

    fn adjust_sb(&self, d_blocks: i64, d_inodes: i64) -> Result<(), FsError> {
        let mut sb = self.cache.read_vec(1024, 1024).map_err(|_| FsError::Io)?;
        let is64 = self.incompat & INCOMPAT_64BIT != 0;
        let blocks = u64::from(le32(&sb, 12))
            | if is64 {
                u64::from(le32(&sb, 344)) << 32
            } else {
                0
            };
        let blocks = blocks.wrapping_add_signed(d_blocks);
        put32(&mut sb, 12, blocks as u32);
        if is64 {
            put32(&mut sb, 344, (blocks >> 32) as u32);
        }
        let inodes = le32(&sb, 16).wrapping_add_signed(d_inodes as i32);
        put32(&mut sb, 16, inodes);
        put32(&mut sb, 48, FIXED_TIME);
        if self.has_csum {
            let c = crc32c(!0, &sb[..1020]);
            put32(&mut sb, 1020, c);
        }
        self.io_write(1024, &sb)
    }

    // --- bitmaps and allocation ------------------------------------------

    fn bitmap_blocks(&self, d: &[u8]) -> (u64, u64) {
        let hi = self.desc_size >= 64;
        let b = u64::from(le32(d, 0)) | if hi { u64::from(le32(d, 32)) << 32 } else { 0 };
        let i = u64::from(le32(d, 4)) | if hi { u64::from(le32(d, 36)) << 32 } else { 0 };
        (b, i)
    }

    fn gd_free_blocks(&self, d: &[u8]) -> u32 {
        u32::from(le16(d, 12))
            | if self.desc_size >= 64 {
                u32::from(le16(d, 44)) << 16
            } else {
                0
            }
    }
    fn gd_set_free_blocks(&self, d: &mut [u8], v: u32) {
        put16(d, 12, v as u16);
        if self.desc_size >= 64 {
            put16(d, 44, (v >> 16) as u16);
        }
    }
    fn gd_free_inodes(&self, d: &[u8]) -> u32 {
        u32::from(le16(d, 14))
            | if self.desc_size >= 64 {
                u32::from(le16(d, 46)) << 16
            } else {
                0
            }
    }
    fn gd_set_free_inodes(&self, d: &mut [u8], v: u32) {
        put16(d, 14, v as u16);
        if self.desc_size >= 64 {
            put16(d, 46, (v >> 16) as u16);
        }
    }

    fn seal_bitmap_csum(&self, d: &mut [u8], map: &[u8], inode_bitmap: bool) {
        if !self.has_csum {
            return;
        }
        let len = if inode_bitmap {
            self.inodes_per_group
        } else {
            self.blocks_per_group
        } as usize
            / 8;
        let c = crc32c(self.csum_seed, &map[..len]);
        let (lo, hi) = if inode_bitmap { (26, 58) } else { (24, 56) };
        put16(d, lo, c as u16);
        if self.desc_size >= 64 {
            put16(d, hi, (c >> 16) as u16);
        }
    }

    /// Allocate up to `want` contiguous blocks, preferring `goal`. Returns (start, len).
    fn alloc_blocks(&self, goal: u64, want: u64) -> Result<(u64, u64), FsError> {
        let start_group = (goal.saturating_sub(self.first_data_block) / self.blocks_per_group)
            .min(self.groups - 1);
        for k in 0..self.groups {
            let g = (start_group + k) % self.groups;
            let mut d = self.group_desc(g)?;
            if le16(&d, 18) & BG_BLOCK_UNINIT != 0 || self.gd_free_blocks(&d) == 0 {
                continue;
            }
            let (bb, _) = self.bitmap_blocks(&d);
            let mut map = vec![0u8; self.block_size as usize];
            self.read_block(bb, &mut map)?;
            let group_first = self.first_data_block + g * self.blocks_per_group;
            let nbits = (self.blocks_per_group).min(self.blocks_count - group_first) as usize;
            let from = if k == 0 && goal >= group_first && goal < group_first + nbits as u64 {
                (goal - group_first) as usize
            } else {
                0
            };
            // First free bit at or after `from`, wrapping once.
            let found = (from..nbits).chain(0..from).find(|&i| !bit(&map, i));
            let Some(first) = found else { continue };
            let mut run = 1usize;
            while (run as u64) < want && first + run < nbits && !bit(&map, first + run) {
                run += 1;
            }
            for i in first..first + run {
                set_bit(&mut map, i, true);
            }
            self.io_write(bb * self.block_size, &map)?;
            let free = self.gd_free_blocks(&d) - run as u32;
            self.gd_set_free_blocks(&mut d, free);
            self.seal_bitmap_csum(&mut d, &map, false);
            self.put_group_desc(g, &mut d)?;
            self.adjust_sb(-(run as i64), 0)?;
            return Ok((group_first + first as u64, run as u64));
        }
        Err(FsError::NoSpace)
    }

    fn free_blocks(&self, mut start: u64, mut len: u64) -> Result<(), FsError> {
        while len > 0 {
            if start < self.first_data_block || start >= self.blocks_count {
                return Err(FsError::Corrupt("freeing a block outside the filesystem"));
            }
            let g = (start - self.first_data_block) / self.blocks_per_group;
            let group_first = self.first_data_block + g * self.blocks_per_group;
            let n = len.min(group_first + self.blocks_per_group - start);
            let mut d = self.group_desc(g)?;
            let (bb, _) = self.bitmap_blocks(&d);
            let mut map = vec![0u8; self.block_size as usize];
            self.read_block(bb, &mut map)?;
            for i in 0..n {
                let idx = (start - group_first + i) as usize;
                if !bit(&map, idx) {
                    return Err(FsError::Corrupt("freeing a block that is not allocated"));
                }
                set_bit(&mut map, idx, false);
            }
            self.io_write(bb * self.block_size, &map)?;
            let free = self.gd_free_blocks(&d) + n as u32;
            self.gd_set_free_blocks(&mut d, free);
            self.seal_bitmap_csum(&mut d, &map, false);
            self.put_group_desc(g, &mut d)?;
            self.adjust_sb(n as i64, 0)?;
            start += n;
            len -= n;
        }
        Ok(())
    }

    fn alloc_inode(&self, parent_group: u64, is_dir: bool) -> Result<u32, FsError> {
        for k in 0..self.groups {
            let g = (parent_group + k) % self.groups;
            let mut d = self.group_desc(g)?;
            if le16(&d, 18) & BG_INODE_UNINIT != 0 || self.gd_free_inodes(&d) == 0 {
                continue;
            }
            let (_, ib) = self.bitmap_blocks(&d);
            let mut map = vec![0u8; self.block_size as usize];
            self.read_block(ib, &mut map)?;
            let min = if g == 0 {
                self.first_ino as usize - 1
            } else {
                0
            };
            let Some(idx) = (min..self.inodes_per_group as usize).find(|&i| !bit(&map, i)) else {
                continue;
            };
            set_bit(&mut map, idx, true);
            self.io_write(ib * self.block_size, &map)?;
            let free = self.gd_free_inodes(&d) - 1;
            self.gd_set_free_inodes(&mut d, free);
            if is_dir {
                let used = le16(&d, 16).wrapping_add(1);
                put16(&mut d, 16, used);
            }
            // Inodes past `itable_unused` from the end must be declared in use.
            let unused = u32::from(le16(&d, 28))
                | if self.desc_size >= 64 {
                    u32::from(le16(&d, 50)) << 16
                } else {
                    0
                };
            let need_unused = self.inodes_per_group as u32 - (idx as u32 + 1);
            if unused > need_unused {
                put16(&mut d, 28, need_unused as u16);
                if self.desc_size >= 64 {
                    put16(&mut d, 50, (need_unused >> 16) as u16);
                }
            }
            let flags = le16(&d, 18) & !BG_INODE_UNINIT;
            put16(&mut d, 18, flags);
            self.seal_bitmap_csum(&mut d, &map, true);
            self.put_group_desc(g, &mut d)?;
            self.adjust_sb(0, -1)?;
            return Ok((g * self.inodes_per_group) as u32 + idx as u32 + 1);
        }
        Err(FsError::NoSpace)
    }

    fn free_inode(&self, ino: u32, was_dir: bool) -> Result<(), FsError> {
        let idx = u64::from(ino - 1);
        let g = idx / self.inodes_per_group;
        let mut d = self.group_desc(g)?;
        let (_, ib) = self.bitmap_blocks(&d);
        let mut map = vec![0u8; self.block_size as usize];
        self.read_block(ib, &mut map)?;
        let i = (idx % self.inodes_per_group) as usize;
        if !bit(&map, i) {
            return Err(FsError::Corrupt("freeing an inode that is not allocated"));
        }
        set_bit(&mut map, i, false);
        self.io_write(ib * self.block_size, &map)?;
        let free = self.gd_free_inodes(&d) + 1;
        self.gd_set_free_inodes(&mut d, free);
        if was_dir {
            let used = le16(&d, 16).saturating_sub(1);
            put16(&mut d, 16, used);
        }
        self.seal_bitmap_csum(&mut d, &map, true);
        self.put_group_desc(g, &mut d)?;
        self.adjust_sb(0, 1)
    }

    // --- inodes ------------------------------------------------------------

    fn read_inode_raw(&self, ino: u32) -> Result<Vec<u8>, FsError> {
        self.cache
            .read_vec(self.inode_offset(ino)?, self.inode_size as usize)
            .map_err(|_| FsError::Io)
    }

    fn write_inode_raw(&self, ino: u32, raw: &mut [u8]) -> Result<(), FsError> {
        self.seal_inode(ino, raw);
        self.io_write(self.inode_offset(ino)?, raw)
    }

    fn set_size(raw: &mut [u8], size: u64) {
        put32(raw, 4, size as u32);
        put32(raw, 108, (size >> 32) as u32);
    }

    fn set_blocks(&self, raw: &mut [u8], fs_blocks: u64) {
        let sectors = fs_blocks * (self.block_size / 512);
        put32(raw, 28, sectors as u32);
        put16(raw, 116, (sectors >> 32) as u16);
    }

    fn raw_extents(&self, raw: &[u8]) -> Result<(Vec<Extent>, Vec<u64>), FsError> {
        let mut block = [0u8; 60];
        block.copy_from_slice(&raw[40..100]);
        let mut ext = Vec::new();
        let mut tree = Vec::new();
        self.walk(&block, 0, &mut ext, &mut tree)?;
        ext.sort_by_key(|e| e.logical);
        Ok((ext, tree))
    }

    /// Rewrite the inode's extent tree from `extents`, replacing the old tree
    /// blocks, and update i_blocks. The caller writes `raw` afterwards.
    fn store_tree(
        &self,
        ino: u32,
        raw: &mut [u8],
        mut extents: Vec<Extent>,
        old_tree: &[u64],
        goal: u64,
    ) -> Result<(), FsError> {
        extents.sort_by_key(|e| e.logical);
        // Merge neighbours, then split anything too long for one entry.
        let mut merged: Vec<Extent> = Vec::new();
        for e in extents {
            if let Some(last) = merged.last_mut() {
                let max = if e.uninit { 32767 } else { 32768 };
                if last.uninit == e.uninit
                    && last.logical + last.len == e.logical
                    && last.physical + u64::from(last.len) == e.physical
                    && last.len + e.len <= max
                {
                    last.len += e.len;
                    continue;
                }
            }
            merged.push(e);
        }
        let generation = le32(raw, 100);
        let per_leaf = (self.block_size as usize - 12) / 12;
        let mut new_tree = Vec::new();
        let mut block = [0u8; 60];
        put16(&mut block, 0, EXT_MAGIC);
        put16(&mut block, 4, 4);
        let put_entry = |dst: &mut [u8], o: usize, e: &Extent| {
            put32(dst, o, e.logical);
            put16(
                dst,
                o + 4,
                if e.uninit {
                    e.len as u16 + 32768
                } else {
                    e.len as u16
                },
            );
            put16(dst, o + 6, (e.physical >> 32) as u16);
            put32(dst, o + 8, e.physical as u32);
        };
        if merged.len() <= 4 {
            put16(&mut block, 2, merged.len() as u16);
            put16(&mut block, 6, 0);
            for (i, e) in merged.iter().enumerate() {
                put_entry(&mut block, 12 + i * 12, e);
            }
        } else {
            let leaves = merged.len().div_ceil(per_leaf);
            if leaves > 4 {
                return Err(FsError::NoSpace);
            }
            put16(&mut block, 2, leaves as u16);
            put16(&mut block, 6, 1);
            for (li, chunk) in merged.chunks(per_leaf).enumerate() {
                let (pb, got) = self.alloc_blocks(goal, 1)?;
                debug_assert_eq!(got, 1);
                new_tree.push(pb);
                let mut leaf = vec![0u8; self.block_size as usize];
                put16(&mut leaf, 0, EXT_MAGIC);
                put16(&mut leaf, 2, chunk.len() as u16);
                put16(&mut leaf, 4, per_leaf as u16);
                for (i, e) in chunk.iter().enumerate() {
                    put_entry(&mut leaf, 12 + i * 12, e);
                }
                self.seal_extent_block(ino, generation, &mut leaf);
                self.io_write(pb * self.block_size, &leaf)?;
                let o = 12 + li * 12;
                put32(&mut block, o, chunk[0].logical);
                put32(&mut block, o + 4, pb as u32);
                put16(&mut block, o + 8, (pb >> 32) as u16);
            }
        }
        raw[40..100].copy_from_slice(&block);
        for &b in old_tree {
            self.free_blocks(b, 1)?;
        }
        let data: u64 = merged.iter().map(|e| u64::from(e.len)).sum();
        self.set_blocks(raw, data + new_tree.len() as u64);
        Ok(())
    }

    fn phys_of(extents: &[Extent], lb: u64) -> Option<(u64, bool)> {
        extents
            .iter()
            .find(|e| lb >= u64::from(e.logical) && lb < u64::from(e.logical) + u64::from(e.len))
            .map(|e| (e.physical + (lb - u64::from(e.logical)), e.uninit))
    }

    fn goal_for(&self, ino: u32, extents: &[Extent], lb: u64) -> u64 {
        extents
            .iter()
            .rfind(|e| u64::from(e.logical) < lb)
            .map(|e| e.physical + u64::from(e.len))
            .unwrap_or_else(|| {
                self.first_data_block
                    + (u64::from(ino - 1) / self.inodes_per_group) * self.blocks_per_group
            })
    }

    /// Write file bytes, allocating blocks for holes in the range.
    fn write_data(&self, ino: u32, offset: u64, data: &[u8]) -> Result<(), FsError> {
        let mut raw = self.read_inode_raw(ino)?;
        let bs = self.block_size;
        let (mut extents, old_tree) = self.raw_extents(&raw)?;
        let end = offset + data.len() as u64;
        let (first, last) = (offset / bs, (end - 1) / bs);
        let mut changed = false;
        // Turn uninitialised extents the write touches into initialised, zeroed ones.
        for e in extents.iter_mut() {
            let (lo, hi) = (
                u64::from(e.logical),
                u64::from(e.logical) + u64::from(e.len) - 1,
            );
            if e.uninit && hi >= first && lo <= last {
                let zeros = vec![0u8; (e.len as u64 * bs).min(1 << 20) as usize];
                let mut done = 0u64;
                while done < e.len as u64 * bs {
                    let n = (zeros.len() as u64).min(e.len as u64 * bs - done);
                    self.io_write(e.physical * bs + done, &zeros[..n as usize])?;
                    done += n;
                }
                e.uninit = false;
                changed = true;
            }
        }
        // Allocate (and zero) blocks for every hole in [first, last].
        let mut lb = first;
        while lb <= last {
            if Self::phys_of(&extents, lb).is_some() {
                lb += 1;
                continue;
            }
            let mut hole_end = lb;
            while hole_end < last && Self::phys_of(&extents, hole_end + 1).is_none() {
                hole_end += 1;
            }
            let mut at = lb;
            while at <= hole_end {
                let goal = self.goal_for(ino, &extents, at);
                let (pstart, got) = self.alloc_blocks(goal, (hole_end - at + 1).min(32768))?;
                let zeros = vec![0u8; (got * bs).min(1 << 20) as usize];
                let mut done = 0u64;
                while done < got * bs {
                    let n = (zeros.len() as u64).min(got * bs - done);
                    self.io_write(pstart * bs + done, &zeros[..n as usize])?;
                    done += n;
                }
                extents.push(Extent {
                    logical: at as u32,
                    len: got as u32,
                    physical: pstart,
                    uninit: false,
                });
                extents.sort_by_key(|e| e.logical);
                at += got;
                changed = true;
            }
            lb = hole_end + 1;
        }
        if changed {
            let goal = self.goal_for(ino, &extents, last);
            self.store_tree(ino, &mut raw, extents.clone(), &old_tree, goal)?;
        }
        // Every block in range is mapped now: write straight through.
        let mut done = 0usize;
        while done < data.len() {
            let pos = offset + done as u64;
            let lb = pos / bs;
            let (p, _) = Self::phys_of(&extents, lb).ok_or(FsError::Corrupt("block vanished"))?;
            let within = pos % bs;
            let n = ((bs - within) as usize).min(data.len() - done);
            self.io_write(p * bs + within, &data[done..done + n])?;
            done += n;
        }
        let old_size = u64::from(le32(&raw, 4)) | u64::from(le32(&raw, 108)) << 32;
        if end > old_size {
            Self::set_size(&mut raw, end);
            changed = true;
        }
        if changed {
            self.write_inode_raw(ino, &mut raw)?;
        }
        Ok(())
    }

    fn truncate_inode(&self, ino: u32, size: u64) -> Result<(), FsError> {
        let mut raw = self.read_inode_raw(ino)?;
        let bs = self.block_size;
        let old = u64::from(le32(&raw, 4)) | u64::from(le32(&raw, 108)) << 32;
        if size >= old {
            Self::set_size(&mut raw, size);
            return self.write_inode_raw(ino, &mut raw);
        }
        let (extents, old_tree) = self.raw_extents(&raw)?;
        let keep = size.div_ceil(bs); // logical blocks that stay
        let mut kept = Vec::new();
        for e in extents {
            let lo = u64::from(e.logical);
            let hi = lo + u64::from(e.len);
            if hi <= keep {
                kept.push(e);
            } else if lo >= keep {
                self.free_blocks(e.physical, u64::from(e.len))?;
            } else {
                let stay = (keep - lo) as u32;
                self.free_blocks(e.physical + u64::from(stay), u64::from(e.len - stay))?;
                kept.push(Extent { len: stay, ..e });
            }
        }
        // Stale bytes after the new end must not reappear if the file grows again.
        if !size.is_multiple_of(bs)
            && let Some((p, false)) = Self::phys_of(&kept, size / bs)
        {
            let n = (bs - size % bs) as usize;
            self.io_write(p * bs + size % bs, &vec![0u8; n])?;
        }
        let goal = kept.last().map_or(self.first_data_block, |e| e.physical);
        self.store_tree(ino, &mut raw, kept, &old_tree, goal)?;
        Self::set_size(&mut raw, size);
        self.write_inode_raw(ino, &mut raw)
    }

    fn free_inode_contents(&self, ino: u32) -> Result<(), FsError> {
        let raw = self.read_inode_raw(ino)?;
        if le32(&raw, 32) & FL_EXTENTS != 0 {
            let (extents, tree) = self.raw_extents(&raw)?;
            for e in extents {
                self.free_blocks(e.physical, u64::from(e.len))?;
            }
            for b in tree {
                self.free_blocks(b, 1)?;
            }
        }
        Ok(())
    }

    // --- directories --------------------------------------------------------

    fn dir_block_list(&self, raw: &[u8]) -> Result<(Vec<Extent>, Vec<u64>), FsError> {
        self.raw_extents(raw)
    }

    fn entry_size(name_len: usize) -> usize {
        (8 + name_len + 3) & !3
    }

    /// Turn an htree directory into a plain linear one (clears the index flag
    /// and rewrites the index blocks as ordinary, empty directory blocks).
    fn dx_to_linear(&self, ino: u32, raw: &mut [u8]) -> Result<(), FsError> {
        let bs = self.block_size as usize;
        let generation = le32(raw, 100);
        let (extents, _) = self.dir_block_list(raw)?;
        let nblocks = (u64::from(le32(raw, 4)) | u64::from(le32(raw, 108)) << 32) / self.block_size;
        for lb in 0..nblocks {
            let (p, _) =
                Self::phys_of(&extents, lb).ok_or(FsError::Corrupt("hole in a directory"))?;
            let mut blk = vec![0u8; bs];
            self.read_block(p, &mut blk)?;
            let first_rec = usize::from(le16(&blk, 4));
            let is_root = lb == 0;
            let is_dx_node = !is_root && le32(&blk, 0) == 0 && first_rec == bs;
            if !is_root && !is_dx_node {
                continue; // an ordinary leaf block
            }
            if is_root {
                // '.' then '..' whose rec_len spans the index data: shrink it.
                let dot = usize::from(le16(&blk, 4));
                let dotdot_len = usize::from(le16(&blk, dot + 4));
                if dotdot_len < 12 {
                    return Err(FsError::Corrupt("bad dx root"));
                }
                let tail = if self.has_csum { 12 } else { 0 };
                put16(&mut blk, dot + 4, (bs - dot - tail) as u16);
                blk[dot + 12..bs].fill(0);
            } else {
                blk.fill(0);
                put16(
                    &mut blk,
                    4,
                    (bs - if self.has_csum { 12 } else { 0 }) as u16,
                );
            }
            if self.has_csum {
                let n = bs;
                put32(&mut blk, n - 12, 0);
                put16(&mut blk, n - 8, 12);
                blk[n - 6] = 0;
                blk[n - 5] = 0xDE;
                self.seal_dir_block(ino, generation, &mut blk);
            }
            self.io_write(p * self.block_size, &blk)?;
        }
        let flags = le32(raw, 32) & !FL_INDEX;
        put32(raw, 32, flags);
        Ok(())
    }

    fn add_dir_entry(
        &self,
        dir_ino: u32,
        name: &str,
        target: u32,
        ftype: u8,
    ) -> Result<(), FsError> {
        let mut raw = self.read_inode_raw(dir_ino)?;
        if le32(&raw, 32) & FL_INDEX != 0 {
            self.dx_to_linear(dir_ino, &mut raw)?;
            self.write_inode_raw(dir_ino, &mut raw)?;
        }
        let bs = self.block_size as usize;
        let generation = le32(&raw, 100);
        let (extents, old_tree) = self.dir_block_list(&raw)?;
        let size = u64::from(le32(&raw, 4)) | u64::from(le32(&raw, 108)) << 32;
        let nblocks = size / self.block_size;
        let need = Self::entry_size(name.len());
        let tail = if self.has_csum { 12 } else { 0 };
        for lb in 0..nblocks {
            let (p, _) =
                Self::phys_of(&extents, lb).ok_or(FsError::Corrupt("hole in a directory"))?;
            let mut blk = vec![0u8; bs];
            self.read_block(p, &mut blk)?;
            let mut pos = 0usize;
            while pos < bs - tail {
                let rec_len = usize::from(le16(&blk, pos + 4));
                if rec_len < 8 || rec_len % 4 != 0 || pos + rec_len > bs - tail {
                    return Err(FsError::Corrupt("bad directory entry"));
                }
                let ino = le32(&blk, pos);
                let used = if ino == 0 {
                    0
                } else {
                    Self::entry_size(usize::from(blk[pos + 6]))
                };
                if rec_len - used >= need {
                    let at = pos + used;
                    if used > 0 {
                        put16(&mut blk, pos + 4, used as u16);
                    }
                    put32(&mut blk, at, target);
                    put16(&mut blk, at + 4, (rec_len - used) as u16);
                    blk[at + 6] = name.len() as u8;
                    blk[at + 7] = ftype;
                    blk[at + 8..at + 8 + name.len()].copy_from_slice(name.as_bytes());
                    self.seal_dir_block(dir_ino, generation, &mut blk);
                    return self.io_write(p * self.block_size, &blk);
                }
                pos += rec_len;
            }
        }
        // No room: append a block.
        let goal = self.goal_for(dir_ino, &extents, nblocks);
        let (pb, _) = self.alloc_blocks(goal, 1)?;
        let mut blk = vec![0u8; bs];
        put32(&mut blk, 0, target);
        put16(&mut blk, 4, (bs - tail) as u16);
        blk[6] = name.len() as u8;
        blk[7] = ftype;
        blk[8..8 + name.len()].copy_from_slice(name.as_bytes());
        if self.has_csum {
            put16(&mut blk, bs - 8, 12);
            blk[bs - 5] = 0xDE;
        }
        self.seal_dir_block(dir_ino, generation, &mut blk);
        self.io_write(pb * self.block_size, &blk)?;
        let mut ext = extents;
        ext.push(Extent {
            logical: nblocks as u32,
            len: 1,
            physical: pb,
            uninit: false,
        });
        self.store_tree(dir_ino, &mut raw, ext, &old_tree, pb)?;
        Self::set_size(&mut raw, size + self.block_size);
        self.write_inode_raw(dir_ino, &mut raw)
    }

    fn remove_dir_entry(&self, dir_ino: u32, name: &str) -> Result<u32, FsError> {
        let mut raw = self.read_inode_raw(dir_ino)?;
        if le32(&raw, 32) & FL_INDEX != 0 {
            self.dx_to_linear(dir_ino, &mut raw)?;
            self.write_inode_raw(dir_ino, &mut raw)?;
        }
        let bs = self.block_size as usize;
        let generation = le32(&raw, 100);
        let (extents, _) = self.dir_block_list(&raw)?;
        let size = u64::from(le32(&raw, 4)) | u64::from(le32(&raw, 108)) << 32;
        let tail = if self.has_csum { 12 } else { 0 };
        for lb in 0..size / self.block_size {
            let (p, _) =
                Self::phys_of(&extents, lb).ok_or(FsError::Corrupt("hole in a directory"))?;
            let mut blk = vec![0u8; bs];
            self.read_block(p, &mut blk)?;
            let (mut pos, mut prev) = (0usize, None::<usize>);
            while pos < bs - tail {
                let rec_len = usize::from(le16(&blk, pos + 4));
                if rec_len < 8 || rec_len % 4 != 0 || pos + rec_len > bs - tail {
                    return Err(FsError::Corrupt("bad directory entry"));
                }
                let ino = le32(&blk, pos);
                let nl = usize::from(blk[pos + 6]);
                if ino != 0 && &blk[pos + 8..pos + 8 + nl] == name.as_bytes() {
                    match prev {
                        Some(pp) => {
                            let merged = usize::from(le16(&blk, pp + 4)) + rec_len;
                            put16(&mut blk, pp + 4, merged as u16);
                        }
                        None => put32(&mut blk, pos, 0),
                    }
                    self.seal_dir_block(dir_ino, generation, &mut blk);
                    self.io_write(p * self.block_size, &blk)?;
                    return Ok(ino);
                }
                prev = Some(pos);
                pos += rec_len;
            }
        }
        Err(FsError::NotFound)
    }

    fn init_inode(&self, ino: u32, kind: Kind) -> Vec<u8> {
        let mut raw = vec![0u8; self.inode_size as usize];
        let (mode, links) = match kind {
            Kind::Dir => (S_IFDIR | 0o755, 2),
            _ => (S_IFREG | 0o644, 1),
        };
        put16(&mut raw, 0, mode);
        put16(&mut raw, 26, links);
        for o in [8, 12, 16] {
            put32(&mut raw, o, FIXED_TIME);
        }
        put32(&mut raw, 32, FL_EXTENTS);
        put16(&mut raw, 40, EXT_MAGIC);
        put16(&mut raw, 44, 4);
        // A distinct generation per inode keeps the checksum seeds apart.
        put32(&mut raw, 100, ino.wrapping_mul(2_654_435_761));
        if self.inode_size > 128 {
            put16(&mut raw, 128, 32);
            for o in [132, 136, 140, 148] {
                put32(&mut raw, o, 0);
            }
            put32(&mut raw, 144, FIXED_TIME);
        }
        raw
    }

    fn valid_name(name: &str) -> bool {
        !name.is_empty()
            && name.len() <= 255
            && name != "."
            && name != ".."
            && !name.contains(['/', '\0'])
    }

    fn do_create(&self, dir: &Node, name: &str, kind: Kind) -> Result<Node, FsError> {
        self.writable()?;
        if !Self::valid_name(name) {
            return Err(FsError::Invalid);
        }
        if kind == Kind::Symlink {
            return Err(FsError::Unsupported("creating symbolic links"));
        }
        if dir.kind != Kind::Dir {
            return Err(FsError::NotDir);
        }
        let dir_ino = dir.data[0] as u32;
        if self.dir_entries(dir)?.iter().any(|(n, _)| n == name) {
            return Err(FsError::Exists);
        }
        let group = u64::from(dir_ino - 1) / self.inodes_per_group;
        let ino = self.alloc_inode(group, kind == Kind::Dir)?;
        let mut raw = self.init_inode(ino, kind);
        let mut dir_block = None;
        if kind == Kind::Dir {
            let bs = self.block_size as usize;
            let (pb, _) =
                self.alloc_blocks(self.first_data_block + group * self.blocks_per_group, 1)?;
            let tail = if self.has_csum { 12 } else { 0 };
            let mut blk = vec![0u8; bs];
            put32(&mut blk, 0, ino);
            put16(&mut blk, 4, 12);
            blk[6] = 1;
            blk[7] = 2; // EXT4_FT_DIR
            blk[8] = b'.';
            put32(&mut blk, 12, dir_ino);
            put16(&mut blk, 16, (bs - 12 - tail) as u16);
            blk[18] = 2;
            blk[19] = 2;
            blk[20] = b'.';
            blk[21] = b'.';
            if self.has_csum {
                put16(&mut blk, bs - 8, 12);
                blk[bs - 5] = 0xDE;
            }
            self.seal_dir_block(ino, le32(&raw, 100), &mut blk);
            self.io_write(pb * self.block_size, &blk)?;
            self.store_tree(
                ino,
                &mut raw,
                vec![Extent {
                    logical: 0,
                    len: 1,
                    physical: pb,
                    uninit: false,
                }],
                &[],
                pb,
            )?;
            Self::set_size(&mut raw, self.block_size);
            dir_block = Some(pb);
        }
        self.write_inode_raw(ino, &mut raw)?;
        let ftype = if kind == Kind::Dir { 2 } else { 1 };
        if let Err(e) = self.add_dir_entry(dir_ino, name, ino, ftype) {
            if let Some(pb) = dir_block {
                let _ = self.free_blocks(pb, 1);
            }
            let _ = self.free_inode(ino, kind == Kind::Dir);
            return Err(e);
        }
        if kind == Kind::Dir {
            let mut praw = self.read_inode_raw(dir_ino)?;
            let links = le16(&praw, 26).saturating_add(1);
            put16(&mut praw, 26, links);
            self.write_inode_raw(dir_ino, &mut praw)?;
        }
        self.node_for(ino)
    }

    fn do_remove(&self, dir: &Node, name: &str) -> Result<(), FsError> {
        self.writable()?;
        let dir_ino = dir.data[0] as u32;
        let entries = self.dir_entries(dir)?;
        let (_, ino) = entries
            .iter()
            .find(|(n, _)| n == name)
            .ok_or(FsError::NotFound)?
            .clone();
        let node = self.node_for(ino)?;
        if node.kind == Kind::Dir && !self.dir_entries(&node)?.is_empty() {
            return Err(FsError::NotEmpty);
        }
        self.remove_dir_entry(dir_ino, name)?;
        self.free_inode_contents(ino)?;
        let mut raw = self.read_inode_raw(ino)?;
        raw.fill(0);
        self.io_write(self.inode_offset(ino)?, &raw)?;
        self.free_inode(ino, node.kind == Kind::Dir)?;
        if node.kind == Kind::Dir {
            let mut praw = self.read_inode_raw(dir_ino)?;
            let links = le16(&praw, 26).saturating_sub(1).max(2);
            put16(&mut praw, 26, links);
            self.write_inode_raw(dir_ino, &mut praw)?;
        }
        Ok(())
    }
}
