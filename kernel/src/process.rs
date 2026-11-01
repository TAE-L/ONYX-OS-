//! Processes: a user address space plus a main thread that runs in ring 3.
//!
//! A fault in user mode (bad pointer, privileged instruction, divide by zero,
//! ...) ends that process with a recorded reason and leaves the kernel and
//! every other process untouched.

use crate::arch::idt::TrapFrame;
use crate::elf::{self, ElfError};
use crate::mm::paging::AddressSpace;
use crate::sched::{self, Class};
use crate::sync::IrqMutex;
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitStatus {
    Exited(i64),
    Faulted {
        vector: u64,
        error: u64,
        addr: u64,
        rip: u64,
    },
    Killed,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpawnError {
    Elf(ElfError),
    NoMemory,
    Sched(sched::SpawnError),
}

pub struct Process {
    pub pid: u64,
    pub name: &'static str,
    cr3: u64,
    status: IrqMutex<Option<ExitStatus>>,
    stdout: IrqMutex<Vec<u8>>,
    killed: AtomicBool,
    /// Field order is drop order: the address space (tables and frames) goes
    /// first, then the live counter, so a zero count means the memory is back.
    _space: AddressSpace,
    _live: LiveGuard,
}

struct LiveGuard;

impl Drop for LiveGuard {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Relaxed);
    }
}

static NEXT_PID: AtomicU64 = AtomicU64::new(1);
static LIVE: AtomicU64 = AtomicU64::new(0);
const STDOUT_CAP: usize = 1 << 20;

impl Process {
    pub fn cr3(&self) -> u64 {
        self.cr3
    }

    pub fn status(&self) -> Option<ExitStatus> {
        *self.status.lock()
    }

    pub fn stdout(&self) -> Vec<u8> {
        self.stdout.lock().clone()
    }

    /// Ask the process to die at its next kernel entry. A process spinning in
    /// ring 3 on another CPU is reached with a reschedule IPI.
    pub fn kill(&self) {
        self.killed.store(true, Relaxed);
        sched::kick_other_cpus();
    }

    fn set_status(&self, s: ExitStatus) {
        let mut g = self.status.lock();
        if g.is_none() {
            *g = Some(s);
        }
    }
}

/// Processes whose memory has not been released yet.
pub fn live_processes() -> u64 {
    LIVE.load(Relaxed)
}

struct UserStart {
    rip: u64,
    rsp: u64,
    arg: u64,
}

core::arch::global_asm!(
    r#"
    .section .text.user,"ax"
    .global enter_user
    .hidden enter_user
// enter_user(rip [rdi], rsp [rsi], arg [rdx]): drop to ring 3 with every
// register cleared (nothing of the kernel's leaks) except rdi = arg.
enter_user:
    cli
    push 0x1B
    push rsi
    push 0x202
    push 0x23
    push rdi
    mov rdi, rdx
    xor eax, eax
    xor ebx, ebx
    xor ecx, ecx
    xor edx, edx
    xor esi, esi
    xor ebp, ebp
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor r12d, r12d
    xor r13d, r13d
    xor r14d, r14d
    xor r15d, r15d
    swapgs
    iretq
"#
);

unsafe extern "C" {
    fn enter_user(rip: u64, rsp: u64, arg: u64) -> !;
}

fn user_task(start: usize) {
    // SAFETY: created by `Box::into_raw` in `spawn_elf` and consumed exactly once here.
    let s = unsafe { Box::from_raw(start as *mut UserStart) };
    let (rip, rsp, arg) = (s.rip, s.rsp, s.arg);
    drop(s);
    // SAFETY: the scheduler already switched to this process's page tables and
    // set the kernel stack that interrupts from ring 3 will use.
    unsafe { enter_user(rip, rsp, arg) }
}

/// Load a static ELF into a fresh address space and start its main thread.
pub fn spawn_elf(
    name: &'static str,
    image: &[u8],
    arg: u64,
    pinned: Option<usize>,
) -> Result<Arc<Process>, SpawnError> {
    let mut space = AddressSpace::new_process().ok_or(SpawnError::NoMemory)?;
    let loaded = elf::load(&mut space, image).map_err(SpawnError::Elf)?;
    let process = Arc::new(Process {
        pid: NEXT_PID.fetch_add(1, Relaxed),
        name,
        cr3: space.root_phys(),
        status: IrqMutex::new(None),
        stdout: IrqMutex::new(Vec::new()),
        killed: AtomicBool::new(false),
        _space: space,
        _live: LiveGuard,
    });
    LIVE.fetch_add(1, Relaxed);
    let start = Box::into_raw(Box::new(UserStart {
        rip: loaded.entry,
        rsp: loaded.stack_top,
        arg,
    }));
    match sched::spawn_user(
        pinned,
        name,
        process.clone(),
        Class::fair(),
        user_task,
        start as usize,
    ) {
        Ok(_) => Ok(process),
        Err(e) => {
            // SAFETY: the task never ran, so we still own the start block.
            drop(unsafe { Box::from_raw(start) });
            Err(SpawnError::Sched(e))
        }
    }
}

/// Wait up to `ms` milliseconds for the process to finish.
pub fn wait(p: &Process, ms: u64) -> Option<ExitStatus> {
    for _ in 0..ms {
        if let Some(s) = p.status() {
            return Some(s);
        }
        sched::sleep_ns(1_000_000);
    }
    p.status()
}

// --- Called from kernel entry points while a process's thread is current ------------

pub fn current_pid() -> u64 {
    sched::current_process().map_or(0, |p| p.pid)
}

pub fn write_stdout(bytes: &[u8]) {
    if let Some(p) = sched::current_process() {
        let mut out = p.stdout.lock();
        let room = STDOUT_CAP.saturating_sub(out.len());
        out.extend_from_slice(&bytes[..bytes.len().min(room)]);
        kprint!(
            "[pid {}] {}",
            p.pid,
            core::str::from_utf8(bytes).unwrap_or("<binary>")
        );
    }
}

pub fn exit_current(code: i64) -> ! {
    if let Some(p) = sched::current_process() {
        p.set_status(ExitStatus::Exited(code));
    }
    sched::exit()
}

/// Kill the current process if someone asked for it. Cheap when nobody did.
pub fn check_kill() {
    if let Some(p) = sched::current_process()
        && p.killed.load(Relaxed)
    {
        p.set_status(ExitStatus::Killed);
        drop(p);
        sched::exit();
    }
}

/// A CPU exception raised while running user code.
pub fn user_fault(frame: &TrapFrame, addr: u64) -> ! {
    let status = ExitStatus::Faulted {
        vector: frame.vector,
        error: frame.error,
        addr,
        rip: frame.rip,
    };
    if let Some(p) = sched::current_process() {
        kprintln!(
            "[proc] pid {} ({}) killed by exception {} at rip={:#x} addr={:#x} error={:#x}",
            p.pid,
            p.name,
            frame.vector,
            frame.rip,
            addr,
            frame.error
        );
        p.set_status(status);
    }
    sched::exit()
}
