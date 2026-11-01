//! Isolation probe. Every instance has this same `MARK` at the same virtual
//! address; each stores its own argument, lets the others run, and checks
//! that nobody else's value ever shows up.
#![no_std]
#![no_main]

use core::ptr::{read_volatile, write_volatile};

static mut MARK: u64 = 0;

#[unsafe(no_mangle)]
pub extern "C" fn _start(arg: u64) -> ! {
    // SAFETY: single-threaded program; volatile so the compiler cannot cache it.
    unsafe {
        write_volatile(&raw mut MARK, arg);
        for _ in 0..40 {
            userland::yield_now();
            for _ in 0..2000 {
                core::hint::spin_loop();
            }
            if read_volatile(&raw const MARK) != arg {
                userland::print("isolation broken\n");
                userland::exit(1);
            }
        }
    }
    userland::print("isolated\n");
    userland::exit(0)
}
