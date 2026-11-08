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
            read_only,
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
        self.walk(&inode.block, 0, &mut out)?;
        out.sort_by_key(|e| e.logical);
        Ok(out)
    }

    fn walk(&self, node: &[u8], depth_seen: u32, out: &mut Vec<Extent>) -> Result<(), FsError> {
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
                self.walk(&buf, depth_seen + 1, out)?;
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
