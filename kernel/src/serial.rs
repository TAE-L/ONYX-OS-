//! COM1 16550 UART, the kernel's only output channel in Phase 1.

use core::fmt::{self, Write};
use spin::Mutex;
use x86_64::instructions::{interrupts, port::Port};

const COM1: u16 = 0x3F8;

pub struct Serial {
    base: u16,
}

impl Serial {
    const fn new(base: u16) -> Self {
        Self { base }
    }

    fn init(&mut self) {
        // SAFETY: standard 16550 programming sequence on the COM1 I/O ports.
        unsafe {
            Port::<u8>::new(self.base + 1).write(0x00); // no UART interrupts
            Port::<u8>::new(self.base + 3).write(0x80); // DLAB on
            Port::<u8>::new(self.base).write(0x01); // divisor 1 = 115200 baud
            Port::<u8>::new(self.base + 1).write(0x00);
            Port::<u8>::new(self.base + 3).write(0x03); // 8N1, DLAB off
            Port::<u8>::new(self.base + 2).write(0xC7); // enable + clear FIFO
            Port::<u8>::new(self.base + 4).write(0x03); // DTR + RTS
        }
    }

    fn write_byte(&mut self, byte: u8) {
        // SAFETY: reads the line status register, writes the data register.
        unsafe {
            let mut status = Port::<u8>::new(self.base + 5);
            while status.read() & 0x20 == 0 {
                core::hint::spin_loop();
            }
            Port::<u8>::new(self.base).write(byte);
        }
    }
}

impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(b);
        }
        Ok(())
    }
}

static SERIAL: Mutex<Serial> = Mutex::new(Serial::new(COM1));

pub fn init() {
    SERIAL.lock().init();
}

pub fn print(args: fmt::Arguments) {
    // Interrupts stay off while the lock is held so a handler that prints
    // cannot deadlock against the code it interrupted.
    interrupts::without_interrupts(|| {
        let _ = SERIAL.lock().write_fmt(args);
    });
}

/// Break a possibly-held lock. Only for the panic path, where the holder will
/// never run again.
///
/// # Safety
/// The caller must be the panic handler (no other code may still use the lock).
pub unsafe fn force_unlock() {
    // SAFETY: forwarded to the caller's contract.
    unsafe { SERIAL.force_unlock() };
}
