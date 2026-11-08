//! OnyxOS kernel, Phase 1: boot, CPU setup, memory management.
//!
//! Boot order matters and is deliberate:
//! serial -> boot info -> CPU protections -> frame allocator -> GDT/IDT ->
//! own page tables (W^X audited) -> heap -> x2APIC -> (selftest | idle).

#![no_std]
#![no_main]
#![deny(unsafe_op_in_unsafe_fn)]
// Several accessors exist only for the self-test build until later phases use them.
#![cfg_attr(not(feature = "selftest"), allow(dead_code))]

extern crate alloc;

#[macro_use]
mod log;
mod acpi;
mod arch;
mod block;
mod boot;
mod cache;
mod elf;
mod fs;
mod limine;
mod mm;
mod nvme;
mod panic;
mod pci;
mod percpu;
mod perf;
mod process;
#[cfg(feature = "selftest")]
mod qemu;
mod sched;
#[cfg(feature = "selftest")]
mod selftest;
mod serial;
mod smp;
mod sync;
mod syscall;
mod vfs;

pub const NAME: &str = "OnyxOS";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initial kernel heap. Backed by contiguous physical frames via the direct map.
const HEAP_BYTES: usize = 64 * 1024 * 1024;

// Entry point. Limine enters with interrupts off on its own stack; switch to
// our guarded boot stack before running any Rust code.
core::arch::global_asm!(
    r#"
    .section .text.boot,"ax"
    .global _start
_start:
    cli
    lea rsp, [rip + __boot_stack_top]
    xor ebp, ebp
    call kmain
2:
    hlt
    jmp 2b
"#
);

#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    serial::init();
    kprintln!("{NAME} {VERSION} booting");

    let info = boot::collect();
    mm::set_hhdm(info.hhdm);
    arch::cpu::init();
    mm::frame::init(info);
    arch::gdt::init();
    arch::idt::init();
    mm::paging::init_kernel_space(info);
    mm::heap::init(HEAP_BYTES);
    arch::apic::init();
    arch::time::init();
    arch::apic::timer_init();
    percpu::install(0, arch::apic::id(), arch::gdt::bsp_tss());
    syscall::init_cpu();
    sched::init(arch::apic::id());
    smp::start_aps();
    pci::init();
    x86_64::instructions::interrupts::enable();
    nvme::init();

    kprintln!(
        "boot complete: {} MiB free of {} MiB",
        mm::frame::free_frames() * mm::PAGE_SIZE / (1 << 20),
        mm::frame::total_frames() * mm::PAGE_SIZE / (1 << 20)
    );

    #[cfg(feature = "selftest")]
    selftest::run();

    #[cfg(not(feature = "selftest"))]
    {
        kprintln!("idle");
        loop {
            sched::sleep_ns(1_000_000_000);
        }
    }
}
