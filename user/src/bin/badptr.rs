//! The kernel must refuse bad user pointers with an error code and survive.
//! Exit code = number of checks that went wrong.
#![no_std]
#![no_main]

use userland::*;

static RODATA: [u8; 16] = *b"read-only bytes!";

#[unsafe(no_mangle)]
pub extern "C" fn _start(kernel_ptr: u64) -> ! {
    let mut bad = 0;
    let mut check = |ok: bool| {
        if !ok {
            bad += 1;
        }
    };
    // write(): null, kernel half, wrapping range, range running off the stack.
    check(syscall(WRITE, 1, 0, 16) == EFAULT);
    check(syscall(WRITE, 1, 0xffff_8000_0000_1000, 8) == EFAULT);
    check(syscall(WRITE, 1, u64::MAX - 3, 16) == EFAULT);
    check(syscall(WRITE, 1, 0x7fff_ffff_f000 - 4, 16) == EFAULT);
    check(syscall(WRITE, 1, 0x0000_8000_0000_0000, 1) == EFAULT);
    // A kernel address that IS mapped and writable (the kernel passed it in):
    // the syscall layer must refuse it by range, not merely by luck.
    check(syscall(WRITE, 1, kernel_ptr, 8) == EFAULT);
    check(syscall(CLOCK_INTO, kernel_ptr, 0, 0) == EFAULT);
    // Bad file descriptor and bad length.
    check(syscall(WRITE, 9, RODATA.as_ptr() as u64, 4) == EBADF);
    check(syscall(WRITE, 1, RODATA.as_ptr() as u64, u64::MAX) < 0);
    // A valid write still works.
    check(syscall(WRITE, 1, RODATA.as_ptr() as u64, 16) == 16);
    // The kernel writing *to* user memory: null, read-only page, kernel half.
    check(syscall(CLOCK_INTO, 0, 0, 0) == EFAULT);
    check(syscall(CLOCK_INTO, RODATA.as_ptr() as u64, 0, 0) == EFAULT);
    check(syscall(CLOCK_INTO, 0xffff_8000_0000_1000, 0, 0) == EFAULT);
    // ...and into a valid buffer.
    let mut now: u64 = 0;
    check(syscall(CLOCK_INTO, &raw mut now as u64, 0, 0) == 0);
    check(now != 0);
    // Unknown system calls fail cleanly.
    check(syscall(9999, 0, 0, 0) == ENOSYS);
    exit(bad)
}
