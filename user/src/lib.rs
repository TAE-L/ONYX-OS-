//! Syscall wrappers for OnyxOS test programs. Register ABI: number in rax,
//! arguments in rdi, rsi, rdx, r10, r8; result in rax; rcx and r11 clobbered.
#![no_std]

use core::arch::asm;
use core::panic::PanicInfo;

pub const EXIT: u64 = 0;
pub const WRITE: u64 = 1;
pub const YIELD: u64 = 2;
pub const SLEEP_NS: u64 = 3;
pub const GETPID: u64 = 4;
pub const CLOCK_NS: u64 = 5;
pub const CLOCK_INTO: u64 = 6;

pub const EFAULT: i64 = -14;
pub const EBADF: i64 = -9;
pub const ENOSYS: i64 = -38;

#[inline(always)]
pub fn syscall(nr: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    let ret: i64;
    // SAFETY: the kernel preserves everything except rax, rcx and r11.
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") nr as i64 => ret,
            in("rdi") a0,
            in("rsi") a1,
            in("rdx") a2,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    ret
}

pub fn exit(code: i64) -> ! {
    syscall(EXIT, code as u64, 0, 0);
    loop {}
}

pub fn write(fd: u64, bytes: &[u8]) -> i64 {
    syscall(WRITE, fd, bytes.as_ptr() as u64, bytes.len() as u64)
}

pub fn print(s: &str) {
    write(1, s.as_bytes());
}

pub fn yield_now() {
    syscall(YIELD, 0, 0, 0);
}

pub fn sleep_ns(ns: u64) {
    syscall(SLEEP_NS, ns, 0, 0);
}

pub fn getpid() -> i64 {
    syscall(GETPID, 0, 0, 0)
}

pub fn clock_ns() -> u64 {
    syscall(CLOCK_NS, 0, 0, 0) as u64
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    exit(101)
}
