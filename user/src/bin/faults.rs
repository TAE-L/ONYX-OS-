//! Does something forbidden, chosen by the argument. The kernel must end this
//! process and carry on. Mode 14 is the one mode that must NOT fault.
#![no_std]
#![no_main]

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

const KERNEL_ADDR: u64 = 0xffff_8000_0000_1000;

#[unsafe(no_mangle)]
pub extern "C" fn _start(mode: u64) -> ! {
    // SAFETY: each arm deliberately violates a protection; that is the test.
    unsafe {
        match mode {
            0 => {
                read_volatile(0 as *const u8);
            }
            1 => {
                read_volatile(KERNEL_ADDR as *const u8);
            }
            2 => asm!("jmp {}", in(reg) 0xffff_ffff_8000_0000u64, options(noreturn)),
            3 => write_volatile(_start as *const () as *mut u8, 0),
            4 => asm!("cli"),
            5 => asm!("hlt"),
            6 => {
                let v: u8;
                asm!("in al, dx", out("al") v, in("dx") 0x3f8u16);
                let _ = v;
            }
            7 => asm!("int 0x30"),
            8 => asm!("ud2"),
            9 => {
                asm!("xor edx, edx", "xor eax, eax", "xor ecx, ecx", "div ecx", out("eax") _, out("edx") _, out("ecx") _)
            }
            10 => write_volatile(KERNEL_ADDR as *mut u8, 1),
            11 => {
                asm!("xor eax, eax", "xor edx, edx", "mov ecx, 0x1b", "wrmsr", out("eax") _, out("edx") _, out("ecx") _)
            }
            12 => {
                let cr3: u64;
                asm!("mov {}, cr3", out(reg) cr3);
                let _ = cr3;
            }
            13 => {
                let mut code = [0xc3u8; 16];
                let f: extern "C" fn() = core::mem::transmute(code.as_mut_ptr());
                f();
            }
            14 => {
                let r = userland::syscall(1234, 0, 0, 0);
                userland::exit(if r == userland::ENOSYS { 0 } else { 1 });
            }
            // Push until the stack runs into the unmapped page below it.
            15 => asm!("2:", "push rax", "jmp 2b", options(noreturn)),
            16 => asm!("sti"),
            17 => asm!("lgdt [{}]", in(reg) 0u64),
            18 => asm!("mov dx, 0x3f8", "mov al, 65", "out dx, al", out("dx") _, out("al") _),
            _ => userland::exit(99),
        }
    }
    userland::print("survived a forbidden operation\n");
    userland::exit(42)
}
