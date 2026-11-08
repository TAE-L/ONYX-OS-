//! Byte-addressed page cache over a block device.
//!
//! Pages are 4 KiB. Reads fetch missing pages in runs (one device command per
//! contiguous run, so a large sequential read is a few big commands, not many
//! small ones). Writes go through to the device and update any cached copy, so
//! the device always holds what the filesystem believes. Eviction is LRU.

#![allow(dead_code)]

use crate::block::{BlockDevice, BlockError};
use crate::sync::IrqMutex;
use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec, vec::Vec};

pub const PAGE: usize = 4096;
/// Largest run fetched in one device command.
const MAX_RUN_PAGES: u64 = 32;

struct Page {
    data: Box<[u8; PAGE]>,
    tick: u64,
}

struct Inner {
    pages: BTreeMap<u64, Page>,
    lru: BTreeMap<u64, u64>, // tick -> page index
    tick: u64,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl Inner {
    fn touch(&mut self, idx: u64) {
        self.tick += 1;
        let t = self.tick;
        if let Some(p) = self.pages.get_mut(&idx) {
            self.lru.remove(&p.tick);
            p.tick = t;
            self.lru.insert(t, idx);
        }
    }

    fn insert(&mut self, idx: u64, data: Box<[u8; PAGE]>) {
        if let Some(old) = self.pages.remove(&idx) {
            self.lru.remove(&old.tick);
        }
        while self.pages.len() >= self.capacity {
            let Some((&t, &victim)) = self.lru.iter().next() else {
                break;
            };
            self.lru.remove(&t);
            self.pages.remove(&victim);
        }
        self.tick += 1;
        self.lru.insert(self.tick, idx);
        self.pages.insert(
            idx,
            Page {
                data,
                tick: self.tick,
            },
        );
    }
}

pub struct Cache {
    dev: Arc<dyn BlockDevice>,
    inner: IrqMutex<Inner>,
    bytes: u64,
}

impl Cache {
    pub fn new(dev: Arc<dyn BlockDevice>, capacity_pages: usize) -> Arc<Cache> {
        let bytes = dev.block_count() * dev.block_size() as u64;
        Arc::new(Cache {
            dev,
            bytes,
            inner: IrqMutex::new(Inner {
                pages: BTreeMap::new(),
                lru: BTreeMap::new(),
                tick: 0,
                capacity: capacity_pages.max(8),
                hits: 0,
                misses: 0,
            }),
        })
    }

    pub fn size(&self) -> u64 {
        self.bytes
    }

    pub fn stats(&self) -> (u64, u64, usize) {
        let g = self.inner.lock();
        (g.hits, g.misses, g.pages.len())
    }

    fn page_count(&self) -> u64 {
        self.bytes.div_ceil(PAGE as u64)
    }

    /// Read page `first .. first+n` from the device into `out` (n * PAGE bytes).
    fn fetch(&self, first: u64, n: u64, out: &mut [u8]) -> Result<(), BlockError> {
        let bs = self.dev.block_size() as u64;
        let start = first * PAGE as u64;
        let want = (n * PAGE as u64).min(self.bytes - start);
        let blocks = want / bs;
        self.dev
            .read(start / bs, &mut out[..(blocks * bs) as usize])?;
        out[(blocks * bs) as usize..].fill(0);
        Ok(())
    }

    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(BlockError::OutOfRange)?;
        if end > self.bytes {
            return Err(BlockError::OutOfRange);
        }
        if buf.is_empty() {
            return Ok(());
        }
        let (first, last) = (offset / PAGE as u64, (end - 1) / PAGE as u64);
        // Fetch whatever is missing, a contiguous run at a time.
        let mut idx = first;
        while idx <= last {
            if self.inner.lock().pages.contains_key(&idx) {
                idx += 1;
                continue;
            }
            let mut run = 1;
            {
                let g = self.inner.lock();
                while idx + run <= last
                    && run < MAX_RUN_PAGES
                    && !g.pages.contains_key(&(idx + run))
                {
                    run += 1;
                }
            }
            let mut tmp = vec![0u8; run as usize * PAGE];
            self.fetch(idx, run, &mut tmp)?;
            let mut g = self.inner.lock();
            g.misses += run;
            for k in 0..run {
                let mut page = Box::new([0u8; PAGE]);
                page.copy_from_slice(&tmp[k as usize * PAGE..(k as usize + 1) * PAGE]);
                g.insert(idx + k, page);
            }
            idx += run;
        }
        // Copy out. A page can in principle be evicted between fetch and copy
        // when the working set exceeds the capacity; reload it then.
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let (pi, po) = (pos / PAGE as u64, (pos % PAGE as u64) as usize);
            let n = (PAGE - po).min(buf.len() - done);
            let mut g = self.inner.lock();
            if let Some(p) = g.pages.get(&pi) {
                buf[done..done + n].copy_from_slice(&p.data[po..po + n]);
                g.hits += 1;
                g.touch(pi);
                done += n;
                continue;
            }
            drop(g);
            let mut tmp = vec![0u8; PAGE];
            self.fetch(pi, 1, &mut tmp)?;
            buf[done..done + n].copy_from_slice(&tmp[po..po + n]);
            let mut page = Box::new([0u8; PAGE]);
            page.copy_from_slice(&tmp);
            self.inner.lock().insert(pi, page);
            done += n;
        }
        Ok(())
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), BlockError> {
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or(BlockError::OutOfRange)?;
        if end > self.bytes {
            return Err(BlockError::OutOfRange);
        }
        let bs = self.dev.block_size() as u64;
        // Write through at block granularity: read-modify-write the edge blocks.
        let first_blk = offset / bs;
        let last_blk = (end - 1) / bs;
        let mut tmp = vec![0u8; ((last_blk - first_blk + 1) * bs) as usize];
        let head = (offset - first_blk * bs) as usize;
        let tail = (end - last_blk * bs) as usize;
        let n = tmp.len();
        if head != 0 || (tail as u64) != bs {
            // Edge blocks need their existing bytes; read through the cache.
            self.read(
                first_blk * bs,
                &mut tmp[..n.min((self.bytes - first_blk * bs) as usize)],
            )?;
        }
        tmp[head..head + data.len()].copy_from_slice(data);
        let mut g = self.inner.lock();
        self.dev.write(first_blk, &tmp)?;
        // Refresh cached pages the write touched.
        let (p0, p1) = (offset / PAGE as u64, (end - 1) / PAGE as u64);
        for pi in p0..=p1.min(self.page_count() - 1) {
            if g.pages.contains_key(&pi) {
                let pstart = pi * PAGE as u64;
                let lo = offset.max(pstart);
                let hi = end.min(pstart + PAGE as u64);
                let page = g.pages.get_mut(&pi).unwrap();
                page.data[(lo - pstart) as usize..(hi - pstart) as usize]
                    .copy_from_slice(&data[(lo - offset) as usize..(hi - offset) as usize]);
            }
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<(), BlockError> {
        self.dev.flush()
    }

    pub fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>, BlockError> {
        let mut v = vec![0u8; len];
        self.read(offset, &mut v)?;
        Ok(v)
    }
}
