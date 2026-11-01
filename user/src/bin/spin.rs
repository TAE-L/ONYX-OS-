//! Never exits: used to test preemption of ring 3 and `kill`.
#![no_std]
#![no_main]

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
use userland as _;

static TICKS: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    loop {
        TICKS.fetch_add(1, Relaxed);
    }
}
