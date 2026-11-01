//! Starting the other CPUs (application processors) through Limine's MP
//! interface. Each one gets its own heap-allocated stack, loads the kernel's
//! page tables, then its own GDT/TSS, the shared IDT and its local APIC and
//! timer, and finally becomes the idle task of its own run queue.

use crate::arch::{apic, cpu, gdt, idt};
use crate::limine::{self, MpInfo};
use crate::mm::paging;
use crate::{percpu, perf, sched};
use core::sync::atomic::Ordering::Relaxed;

const AP_STACK_SIZE: usize = 128 * 1024;

#[repr(C)]
struct ApBoot {
    /// Must stay first: loaded by `ap_entry`.
    stack_top: u64,
    index: usize,
}

core::arch::global_asm!(
    r#"
    .section .text.smp,"ax"
    .global ap_entry
    .hidden ap_entry
// Limine enters here with rdi = &MpInfo (extra_argument at offset 24 = &ApBoot).
ap_entry:
    cli
    mov rsi, [rdi + 24]
    mov rsp, [rsi]
    xor ebp, ebp
    call ap_main
    ud2
"#
);

unsafe extern "C" {
    fn ap_entry();
}

#[unsafe(no_mangle)]
extern "C" fn ap_main(info: &MpInfo, boot: &ApBoot) -> ! {
    cpu::init_quiet(true);
    paging::kernel().lock().activate();
    let tss = gdt::init_ap();
    idt::load_on_this_cpu();
    percpu::install(boot.index, info.lapic_id, tss);
    crate::syscall::init_cpu();
    apic::init_ap();
    apic::timer_init_ap();
    kprintln!(
        "[smp] cpu {} online (lapic id {})",
        boot.index,
        info.lapic_id
    );
    sched::run_ap(boot.index, info.lapic_id)
}

/// Start every other CPU and wait (up to one second) for them to come online.
/// Returns how many CPUs are running, the boot CPU included.
pub fn start_aps() -> usize {
    let Some(resp) = limine::MP.response() else {
        kprintln!("[smp] bootloader gave no MP info: single CPU");
        return 1;
    };
    let bsp = resp.bsp_lapic_id;
    let mut next = 1;
    for info in resp.cpus() {
        if info.lapic_id == bsp {
            continue;
        }
        if next >= sched::MAX_CPUS {
            kprintln!(
                "[smp] more than {} CPUs: ignoring the rest",
                sched::MAX_CPUS
            );
            break;
        }
        let stack = alloc::vec![0u8; AP_STACK_SIZE].leak();
        let boot = alloc::boxed::Box::leak(alloc::boxed::Box::new(ApBoot {
            stack_top: (stack.as_ptr() as u64 + AP_STACK_SIZE as u64) & !15,
            index: next,
        }));
        info.extra_argument
            .store(core::ptr::from_ref::<ApBoot>(boot) as u64, Relaxed);
        info.goto_address
            .store(ap_entry as *const () as u64, Relaxed);
        next += 1;
    }
    let expected = next;
    let deadline = perf::rdtsc() + crate::arch::time::ns_to_tsc(1_000_000_000);
    while sched::online_cpus() < expected && perf::rdtsc() < deadline {
        core::hint::spin_loop();
    }
    let up = sched::online_cpus();
    if up < expected {
        kprintln!("[smp] warning: only {up} of {expected} CPUs came online");
    }
    kprintln!("[smp] {up} CPU(s) running");
    up
}
