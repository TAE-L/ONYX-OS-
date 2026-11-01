//! System call entry (`SYSCALL`/`SYSRET`) and the native syscall table.
//!
//! ABI (Linux-like register use): number in `rax`; arguments in `rdi, rsi,
//! rdx, r10, r8`; result in `rax` (negative = error). Everything except `rax`,
//! `rcx` and `r11` is preserved. User pointers are never trusted: every access
//! goes through [`copy_from_user`] / [`copy_to_user`], which validate the
//! range and survive a page fault.

use crate::arch::idt;
use crate::mm::paging::USER_END;
use alloc::vec::Vec;
use x86_64::registers::model_specific::{Efer, EferFlags, Msr};

pub const ENOSYS: i64 = -38;
pub const EFAULT: i64 = -14;
pub const EBADF: i64 = -9;
pub const EINVAL: i64 = -22;

pub mod nr {
    pub const EXIT: u64 = 0;
    pub const WRITE: u64 = 1;
    pub const YIELD: u64 = 2;
    pub const SLEEP_NS: u64 = 3;
    pub const GETPID: u64 = 4;
    pub const CLOCK_NS: u64 = 5;
    /// `time_into(ptr)`: store the clock (u64 ns) at a user address.
    pub const CLOCK_INTO: u64 = 6;
}

const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;
/// Cleared on entry: IF, TF, DF, NT, AC. The kernel decides when to re-enable IF.
const SFMASK: u64 = 0x200 | 0x100 | 0x400 | 0x4000 | 0x40000;

core::arch::global_asm!(
    r#"
    .section .text.syscall,"ax"
    .global syscall_entry
    .hidden syscall_entry
// SYSCALL: rcx = user rip, r11 = user rflags, rsp is still the user's.
syscall_entry:
    swapgs
    mov gs:[16], rsp
    mov rsp, gs:[8]
    push qword ptr gs:[16]
    push rcx
    push r11
    push rdi
    push rsi
    push rdx
    push r10
    push r8
    push r9
    sub rsp, 8
    // syscall_dispatch(nr, a0, a1, a2, a3, a4)
    mov r9, r8
    mov r8, r10
    mov rcx, rdx
    mov rdx, rsi
    mov rsi, rdi
    mov rdi, rax
    cld
    sti
    call syscall_dispatch
    cli
    add rsp, 8
    pop r9
    pop r8
    pop r10
    pop rdx
    pop rsi
    pop rdi
    pop r11
    pop rcx
    pop rsp
    swapgs
    sysretq
"#
);

unsafe extern "C" {
    fn syscall_entry();
}

/// Enable SYSCALL on this CPU. Call after the GDT and per-CPU block exist.
pub fn init_cpu() {
    // SAFETY: documented MSRs; selectors match the GDT layout (kernel CS 0x08,
    // SYSRET base 0x10 gives user CS 0x23 and SS 0x1B).
    unsafe {
        Efer::update(|e| e.insert(EferFlags::SYSTEM_CALL_EXTENSIONS));
        Msr::new(IA32_STAR).write((0x0010u64 << 48) | (0x0008u64 << 32));
        Msr::new(IA32_LSTAR).write(syscall_entry as *const () as u64);
        Msr::new(IA32_FMASK).write(SFMASK);
    }
}

// --- User memory access -----------------------------------------------------------

fn user_range_ok(addr: u64, len: usize) -> bool {
    addr.checked_add(len as u64)
        .is_some_and(|end| end <= USER_END)
}

/// Copy `buf.len()` bytes from user address `src`. Fails if any byte is
/// outside the user half or unmapped.
pub fn copy_from_user(src: u64, buf: &mut [u8]) -> Result<(), i64> {
    if !user_range_ok(src, buf.len()) {
        return Err(EFAULT);
    }
    for (i, b) in buf.iter_mut().enumerate() {
        *b = idt::try_read_u8(src + i as u64).ok_or(EFAULT)?;
    }
    Ok(())
}

/// Copy `data` to user address `dst`. Fails if any byte is outside the user
/// half, unmapped or read-only.
pub fn copy_to_user(dst: u64, data: &[u8]) -> Result<(), i64> {
    if !user_range_ok(dst, data.len()) {
        return Err(EFAULT);
    }
    for (i, b) in data.iter().enumerate() {
        if !idt::try_write_u8(dst + i as u64, *b) {
            return Err(EFAULT);
        }
    }
    Ok(())
}

// --- Dispatch -----------------------------------------------------------------------

#[unsafe(no_mangle)]
extern "C" fn syscall_dispatch(number: u64, a0: u64, a1: u64, a2: u64, _a3: u64, _a4: u64) -> i64 {
    let result = match number {
        nr::EXIT => crate::process::exit_current(a0 as i64),
        nr::WRITE => sys_write(a0, a1, a2),
        nr::YIELD => {
            crate::sched::yield_now();
            0
        }
        nr::SLEEP_NS => {
            crate::sched::sleep_ns(a0.min(10_000_000_000));
            0
        }
        nr::GETPID => crate::process::current_pid() as i64,
        nr::CLOCK_NS => crate::arch::time::now_ns() as i64,
        nr::CLOCK_INTO => {
            let now = crate::arch::time::now_ns();
            match copy_to_user(a0, &now.to_le_bytes()) {
                Ok(()) => 0,
                Err(e) => e,
            }
        }
        _ => ENOSYS,
    };
    crate::process::check_kill();
    result
}

fn sys_write(fd: u64, ptr: u64, len: u64) -> i64 {
    if fd != 1 && fd != 2 {
        return EBADF;
    }
    if len > 64 * 1024 {
        return EINVAL;
    }
    let mut buf = Vec::new();
    if buf.try_reserve_exact(len as usize).is_err() {
        return EINVAL;
    }
    buf.resize(len as usize, 0);
    if let Err(e) = copy_from_user(ptr, &mut buf) {
        return e;
    }
    crate::process::write_stdout(&buf);
    len as i64
}
