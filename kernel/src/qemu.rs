//! Test-harness exit through QEMU's `isa-debug-exit` device (iobase 0xf4).
//! QEMU's process exit status is `(value << 1) | 1`.

use x86_64::instructions::port::Port;

pub const PASS: u32 = 0x10; // exit status 33
pub const FAIL: u32 = 0x11; // exit status 35
pub const PANIC: u32 = 0x12; // exit status 37

pub fn exit(code: u32) -> ! {
    // SAFETY: the debug-exit device only exists under QEMU; on real hardware
    // the write goes nowhere and we fall through to the halt loop.
    unsafe { Port::<u32>::new(0xf4).write(code) };
    crate::arch::cpu::halt_forever()
}
