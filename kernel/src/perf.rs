//! Latency instrumentation: TSC reads and a lock-free power-of-two histogram.
//!
//! Rule for this project: a number is only a result if it was measured with
//! this (or equivalent) instrumentation, under hardware acceleration.

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[inline]
pub fn rdtsc() -> u64 {
    // SAFETY: RDTSC is available on every x86_64 CPU and has no memory effects.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// 65 buckets: bucket `b` holds values whose bit length is `b`
/// (bucket 0 = {0}, bucket 1 = {1}, bucket 2 = 2..=3, ... bucket 64 = 2^63..).
pub struct Histogram {
    buckets: [AtomicU64; 65],
    count: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl Histogram {
    pub const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; 65],
            count: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    pub fn record(&self, value: u64) {
        let bucket = (64 - value.leading_zeros()) as usize;
        self.buckets[bucket].fetch_add(1, Relaxed);
        self.count.fetch_add(1, Relaxed);
        self.min.fetch_min(value, Relaxed);
        self.max.fetch_max(value, Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Relaxed)
    }

    pub fn min(&self) -> u64 {
        self.min.load(Relaxed)
    }

    pub fn max(&self) -> u64 {
        self.max.load(Relaxed)
    }

    /// Upper bound of the bucket that contains the `p`-th percentile (1..=100).
    /// Resolution is a factor of two; that is enough to see tail latency.
    pub fn percentile(&self, p: u64) -> u64 {
        let total = self.count();
        if total == 0 {
            return 0;
        }
        let target = (total * p.clamp(1, 100)).div_ceil(100);
        let mut seen = 0;
        for (b, bucket) in self.buckets.iter().enumerate() {
            seen += bucket.load(Relaxed);
            if seen >= target {
                return match b {
                    0 => 0,
                    64 => u64::MAX,
                    _ => (1u64 << b) - 1,
                };
            }
        }
        u64::MAX
    }
}
