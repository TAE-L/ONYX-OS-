//! x87/SSE/AVX register state: enabling it, and saving it per task.
//!
//! The kernel itself is compiled soft-float and never touches these registers,
//! so state only has to be switched when the scheduler changes tasks (interrupt
//! handlers cannot corrupt it). The state is saved eagerly with XSAVE (or
//! FXSAVE on a CPU without XSAVE) into a per-task area and restored for the
//! incoming task, so nothing leaks between tasks and a task migrating between
//! CPUs keeps its registers.

use alloc::alloc::{Layout, alloc_zeroed, dealloc};
use core::arch::asm;
use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};

/// XCR0 bits OnyxOS turns on: x87, SSE, AVX (AVX-512 is not enabled yet).
const WANTED_XCR0: u64 = 0b111;
const FXSAVE_SIZE: usize = 512;
const MXCSR_DEFAULT: u32 = 0x1F80;
const FCW_DEFAULT: u16 = 0x037F;

/// XCR0 value in effect (0 = XSAVE not available, FXSAVE is used).
static XCR0: AtomicU64 = AtomicU64::new(0);
/// Bytes one saved state needs.
static AREA_SIZE: AtomicU32 = AtomicU32::new(FXSAVE_SIZE as u32);

fn xsave_supported() -> bool {
    __cpuid(1).ecx & (1 << 26) != 0
}

/// Enable FPU/SSE/AVX on the running CPU. Run on every CPU, before its first task switch.
pub fn init_cpu() {
    // SAFETY: standard x86-64 FPU enabling; SSE is architectural on x86-64.
    unsafe {
        Cr0::update(|c| {
            c.remove(Cr0Flags::EMULATE_COPROCESSOR | Cr0Flags::TASK_SWITCHED);
            c.insert(Cr0Flags::MONITOR_COPROCESSOR);
        });
        Cr4::update(|c| {
            c.insert(Cr4Flags::OSFXSR | Cr4Flags::OSXMMEXCPT_ENABLE);
            if xsave_supported() {
                c.insert(Cr4Flags::OSXSAVE);
            }
        });
        asm!("fninit", options(nomem, nostack));
    }
    if !xsave_supported() {
        return;
    }
    let supported = {
        let l = __cpuid_count(0xD, 0);
        u64::from(l.edx) << 32 | u64::from(l.eax)
    };
    // x87 and SSE are mandatory with XSAVE; AVX only if the CPU has it.
    let xcr0 = WANTED_XCR0 & supported;
    // SAFETY: OSXSAVE is set and the mask is a subset of what CPUID reports.
    unsafe {
        asm!("xsetbv", in("ecx") 0u32, in("eax") xcr0 as u32,
             in("edx") (xcr0 >> 32) as u32, options(nomem, nostack));
    }
    // CPUID.0xD.0:EBX is the size for the features currently enabled in XCR0.
    let size = __cpuid_count(0xD, 0).ebx;
    XCR0.store(xcr0, Relaxed);
    AREA_SIZE.fetch_max(size, Relaxed);
}

pub fn describe() -> (bool, u64, usize) {
    (
        XCR0.load(Relaxed) != 0,
        XCR0.load(Relaxed),
        AREA_SIZE.load(Relaxed) as usize,
    )
}

/// One task's saved FPU/SSE/AVX registers (64-byte aligned, as XSAVE needs).
pub struct FpuState {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the area is plain memory owned by exactly one task.
unsafe impl Send for FpuState {}
unsafe impl Sync for FpuState {}

impl FpuState {
    /// A state equal to a freshly reset CPU: all registers zero, default
    /// rounding/exception masks.
    pub fn clean() -> FpuState {
        let size = AREA_SIZE.load(Relaxed) as usize;
        let layout = Layout::from_size_align(size, 64).expect("fpu layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory for FPU state");
        // FXSAVE layout: FCW at 0, MXCSR at 24. XSTATE_BV (offset 512) stays 0,
        // so XRSTOR puts every component other than MXCSR in its init state.
        // SAFETY: `ptr` is a live allocation of at least 512 bytes.
        unsafe {
            ptr.cast::<u16>().write(FCW_DEFAULT);
            ptr.add(24).cast::<u32>().write(MXCSR_DEFAULT);
        }
        FpuState { ptr, layout }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for FpuState {
    fn drop(&mut self) {
        // SAFETY: allocated in `clean` with this layout.
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

/// Store the running CPU's registers into `area`.
///
/// # Safety
/// `area` must be a live, 64-byte aligned save area; interrupts must be off.
#[inline]
pub unsafe fn save(area: *mut u8) {
    let xcr0 = XCR0.load(Relaxed);
    // SAFETY: caller guarantees the area; XCR0 gates which form is legal.
    unsafe {
        if xcr0 != 0 {
            asm!("xsave [{}]", in(reg) area, in("eax") xcr0 as u32,
                 in("edx") (xcr0 >> 32) as u32, options(nostack));
        } else {
            asm!("fxsave [{}]", in(reg) area, options(nostack));
        }
    }
}

/// Load the running CPU's registers from `area`.
///
/// # Safety
/// `area` must hold a state written by `save` or built by `FpuState::clean`.
#[inline]
pub unsafe fn restore(area: *const u8) {
    let xcr0 = XCR0.load(Relaxed);
    // SAFETY: as above.
    unsafe {
        if xcr0 != 0 {
            asm!("xrstor [{}]", in(reg) area, in("eax") xcr0 as u32,
                 in("edx") (xcr0 >> 32) as u32, options(nostack));
        } else {
            asm!("fxrstor [{}]", in(reg) area, options(nostack));
        }
    }
}
