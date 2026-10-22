//! The kernel clock: the TSC, calibrated once against the legacy PIT.
//!
//! Everything else (scheduler, timers, latency histograms) counts in TSC
//! ticks and converts to nanoseconds only at the edges.

use crate::perf::rdtsc;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
use x86_64::instructions::port::Port;

const PIT_HZ: u64 = 1_193_182;
/// PIT count for ~20 ms.
const PIT_COUNT: u16 = 23_864;

static TSC_HZ: AtomicU64 = AtomicU64::new(0);
static TSC_BOOT: AtomicU64 = AtomicU64::new(0);

/// Measure TSC ticks across one PIT channel-2 countdown, return TSC Hz.
fn pit_sample() -> u64 {
    // SAFETY: standard PC speaker/PIT channel 2 programming; the speaker stays off.
    unsafe {
        let mut gate = Port::<u8>::new(0x61);
        let mut cmd = Port::<u8>::new(0x43);
        let mut ch2 = Port::<u8>::new(0x42);
        let v = gate.read();
        gate.write(v & !0x03); // gate low, speaker off
        cmd.write(0xB0); // channel 2, lobyte/hibyte, mode 0
        ch2.write((PIT_COUNT & 0xFF) as u8);
        ch2.write((PIT_COUNT >> 8) as u8);
        gate.write((v & !0x02) | 0x01); // gate high: counting starts
        let t0 = rdtsc();
        while gate.read() & 0x20 == 0 {
            core::hint::spin_loop();
        }
        let t1 = rdtsc();
        gate.write(v & !0x03);
        (t1 - t0) * PIT_HZ / u64::from(PIT_COUNT)
    }
}

pub fn init() {
    let mut s = [pit_sample(), pit_sample(), pit_sample()];
    s.sort_unstable();
    let hz = s[1];
    assert!(
        (100_000_000..=10_000_000_000).contains(&hz),
        "TSC calibration out of range: {hz} Hz"
    );
    TSC_HZ.store(hz, Relaxed);
    TSC_BOOT.store(rdtsc(), Relaxed);
    kprintln!(
        "[time] TSC {}.{:03} MHz (samples {} / {} / {} kHz)",
        hz / 1_000_000,
        hz / 1000 % 1000,
        s[0] / 1000,
        s[1] / 1000,
        s[2] / 1000
    );
}

pub fn tsc_hz() -> u64 {
    TSC_HZ.load(Relaxed)
}

pub fn ns_to_tsc(ns: u64) -> u64 {
    (u128::from(ns) * u128::from(tsc_hz()) / 1_000_000_000) as u64
}

pub fn tsc_to_ns(ticks: u64) -> u64 {
    (u128::from(ticks) * 1_000_000_000 / u128::from(tsc_hz())) as u64
}

/// Nanoseconds since the clock was calibrated.
pub fn now_ns() -> u64 {
    tsc_to_ns(rdtsc().wrapping_sub(TSC_BOOT.load(Relaxed)))
}
