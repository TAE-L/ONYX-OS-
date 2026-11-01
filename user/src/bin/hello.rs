#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    userland::print("hello from ring 3\n");
    userland::exit(7)
}
