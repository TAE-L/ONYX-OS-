# OnyxOS

A modern, UEFI-only x86_64 operating system in Rust, built for low latency and
gaming, with native access to Linux (ext4) and Windows (FAT32/exFAT/NTFS) files.
Rebuild from scratch; the design and phase plan live in the design doc
("OnyxOS Rebuild: Architecture and Build Order").

## Status

| Phase | State |
|---|---|
| 0 Tooling and harness | done |
| 1 Boot and memory | done (gate passed, see below) |
| 2 Scheduler and timers | next |

Phase 1 gate: *two address spaces stay isolated*. Proven by a hardware test
(`ISOLATION` in the self-test): the same virtual address is mapped to different
frames in two address spaces, each reads its own byte under its own CR3, and a
third space and the kernel space take a real page fault on that address.

## Use

Requirements: Rust >= 1.97 with the `rust-src` component
(`rustup component add rust-src`), QEMU, and UEFI firmware (OVMF/edk2).
Everything goes through one command, identical on Windows and Linux:

```
cargo xtask test      # boot the self-test build in QEMU, check results (about 4 s)
cargo xtask run       # normal boot, serial output on this terminal
cargo xtask check     # rustfmt + clippy -D warnings + test
cargo xtask build [--release]
```

Overrides: `ONYX_QEMU`, `ONYX_OVMF_CODE`, `ONYX_OVMF_VARS`, `--accel kvm|tcg|whpx`.
Only one VM runs at a time (an OS file lock enforces it).

## Layout

| Path | What |
|---|---|
| `kernel/` | the kernel (`arch/` CPU+interrupts, `mm/` memory, `selftest.rs`) |
| `xtask/` | build/run/test driver (Rust, no scripts) |
| `boot/limine.conf`, `third_party/limine/` | bootloader config and the pinned Limine 11.4.1 binary |

## Design rules that came from the first attempt

- Per-process address spaces exist from the start.
- W^X is enforced by the mapping API and re-audited at every boot.
- Tests check behaviour seen from outside the code (hardware faults, page-table
  walks, CPU registers), never only round trips. Break the kernel on purpose to
  confirm a test can fail.
- A number is a result only if measured, and timing is only meaningful under
  hardware acceleration, not QEMU's software emulation.

## Known limits (honest list)

- Verified only on Linux under QEMU software emulation (TCG). The Windows host
  path (QEMU/OVMF locations, WHPX) is untested.
- TCG has no x2APIC, so only the xAPIC (MMIO) code path has actually run. The
  x2APIC path needs a run under KVM/WHPX or on hardware before it is trusted.
- Isolation is proven from ring 0 by switching CR3. Ring-3 execution arrives
  with Phase 3.
- Frame allocator is one bitmap behind one lock; heap is a linked list. Both
  are replaced in Phase 2 (per-CPU caches). Bootloader-reclaimable memory is
  not yet returned to the allocator.
- Toolchain: stable Rust with `RUSTC_BOOTSTRAP=1` and `-Zbuild-std`, because
  the kernel needs `core`/`alloc` built from source for the bare-metal target.
