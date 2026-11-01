//! Syscall behaviour: ids, clock, sleeping, yielding, register preservation.
#![no_std]
#![no_main]

use core::arch::asm;
use userland::*;

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    // Bit n of the exit code is set when check n failed.
    let mut bad = 0i64;
    let mut n = 0;
    let mut check = |ok: bool| {
        if !ok {
            bad |= 1 << n;
        }
        n += 1;
    };
    check(getpid() > 0);

    let t0 = clock_ns();
    sleep_ns(5_000_000);
    let t1 = clock_ns();
    check(t1 >= t0 + 5_000_000);
    check(t1 < t0 + 200_000_000);
    for _ in 0..10 {
        yield_now();
    }

    // Everything but rax, rcx, r11 must survive a system call.
    let (mut saved_rbx, mut rbx, mut r12, mut r13, mut r14, mut r15): (
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
    );
    let (mut rsi, mut rdx, mut r8, mut r9, mut r10): (u64, u64, u64, u64, u64);
    // SAFETY: loads known values, performs getpid, reads the values back.
    unsafe {
        asm!(
            "mov {saved}, rbx",
            "mov rbx, 0x1111",
            "mov r12, 0x1212",
            "mov r13, 0x1313",
            "mov r14, 0x1414",
            "mov r15, 0x1515",
            "mov rsi, 0x2222",
            "mov rdx, 0x3333",
            "mov r8, 0x4444",
            "mov r9, 0x5555",
            "mov r10, 0x6666",
            "mov eax, 4",
            "syscall",
            "mov rcx, rbx",
            "mov rbx, {saved}",
            saved = out(reg) saved_rbx,
            out("rcx") rbx, out("r12") r12, out("r13") r13, out("r14") r14, out("r15") r15,
            out("rsi") rsi, out("rdx") rdx, out("r8") r8, out("r9") r9, out("r10") r10,
            out("rax") _, out("r11") _, out("rdi") _,
        );
    }
    let _ = saved_rbx;
    check(rbx == 0x1111);
    check(r12 == 0x1212);
    check(r13 == 0x1313);
    check(r14 == 0x1414);
    check(r15 == 0x1515);
    check(rsi == 0x2222 && rdx == 0x3333 && r8 == 0x4444 && r9 == 0x5555 && r10 == 0x6666);
    exit(bad)
}
