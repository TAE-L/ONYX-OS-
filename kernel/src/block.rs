//! Block devices: a small trait and a registry. Filesystems and the page cache
//! talk to this, never to a driver.

#![allow(dead_code)] // the VFS (Phase 4c) is the main consumer

use alloc::{string::String, sync::Arc, vec::Vec};
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// The request reaches past the end of the device.
    OutOfRange,
    /// Buffer length is not a whole number of blocks, or is empty.
    BadLength,
    /// The device reported an error or stopped answering.
    Device,
}

pub trait BlockDevice: Send + Sync {
    /// Bytes per logical block (512 or 4096 in practice).
    fn block_size(&self) -> usize;
    fn block_count(&self) -> u64;
    /// Read `buf.len() / block_size` blocks starting at `lba`.
    fn read(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
    /// Make completed writes durable.
    fn flush(&self) -> Result<(), BlockError>;
}

struct Entry {
    name: String,
    dev: Arc<dyn BlockDevice>,
}

static DEVICES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

pub fn register(name: &str, dev: Arc<dyn BlockDevice>) {
    DEVICES.lock().push(Entry {
        name: name.into(),
        dev,
    });
}

pub fn get(name: &str) -> Option<Arc<dyn BlockDevice>> {
    DEVICES
        .lock()
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.dev.clone())
}

pub fn names() -> Vec<String> {
    DEVICES.lock().iter().map(|e| e.name.clone()).collect()
}

/// Check a request against the device geometry; returns the block count.
pub fn check(dev: &dyn BlockDevice, lba: u64, len: usize) -> Result<u64, BlockError> {
    let bs = dev.block_size();
    if len == 0 || !len.is_multiple_of(bs) {
        return Err(BlockError::BadLength);
    }
    let n = (len / bs) as u64;
    match lba.checked_add(n) {
        Some(end) if end <= dev.block_count() => Ok(n),
        _ => Err(BlockError::OutOfRange),
    }
}
