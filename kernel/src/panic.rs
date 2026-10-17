use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, Ordering};

static PANICKING: AtomicBool = AtomicBool::new(false);

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    x86_64::instructions::interrupts::disable();
    if PANICKING.swap(true, Ordering::SeqCst) {
        // A panic inside the panic path: stop without printing again.
        crate::arch::cpu::halt_forever();
    }
    // SAFETY: we are the panic handler and never return to the lock holder.
    unsafe { crate::serial::force_unlock() };
    kprintln!("\n*** KERNEL PANIC ***\n{info}");
    #[cfg(feature = "selftest")]
    crate::qemu::exit(crate::qemu::PANIC);
    #[cfg(not(feature = "selftest"))]
    crate::arch::cpu::halt_forever()
}
