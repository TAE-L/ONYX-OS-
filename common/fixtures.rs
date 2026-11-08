// Shared by xtask (which builds the test disk images) and the kernel self-test
// (which checks what it reads back). Plain data and one function, no imports.

/// (path, length, seed): contents are `fixture_byte(seed, i)` for byte `i`.
pub const FIXTURE_FILES: &[(&str, usize, u64)] = &[
    ("HELLO.TXT", 13, 1),
    ("empty", 0, 2),
    ("one", 1, 3),
    ("boundary/b4095.bin", 4095, 4),
    ("boundary/b4096.bin", 4096, 5),
    ("boundary/b4097.bin", 4097, 6),
    ("boundary/b512.bin", 512, 7),
    ("boundary/b513.bin", 513, 8),
    ("big/one_mib.bin", 1 << 20, 9),
    ("big/odd.bin", 3_000_001, 10),
    ("dir with spaces/long file name with spaces and ünïcode.txt", 1234, 11),
    ("deep/a/b/c/d/e/f/g/leaf.txt", 100, 12),
];

/// `many/fNNN.dat` for NNN in 0..FIXTURE_MANY, each `10 + NNN` bytes, seed 1000 + NNN.
pub const FIXTURE_MANY: usize = 300;

/// A file that is zero except for these (offset, length, seed) runs.
pub const FIXTURE_SPARSE: (&str, usize, &[(usize, usize, u64)]) = (
    "sparse.bin",
    8 << 20,
    &[(0, 4096, 20), (4 << 20, 4096, 21), ((8 << 20) - 100, 100, 22)],
);

pub fn fixture_byte(seed: u64, i: usize) -> u8 {
    ((((i as u64).wrapping_mul(2_654_435_761)) ^ seed.wrapping_mul(40_503)) >> 8) as u8
}
