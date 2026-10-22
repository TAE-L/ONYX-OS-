//! Interrupt-safe locking.
//!
//! The scheduler can switch tasks from a timer interrupt at any instruction
//! boundary with interrupts on. A plain spin lock held by a task that gets
//! switched out would then deadlock the next task that wants it. `IrqMutex`
//! keeps interrupts off while it is held, which makes it safe to share
//! between tasks and interrupt handlers.

use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use x86_64::instructions::interrupts;

pub struct IrqMutex<T> {
    inner: spin::Mutex<T>,
}

pub struct IrqGuard<'a, T> {
    guard: ManuallyDrop<spin::MutexGuard<'a, T>>,
    was_enabled: bool,
}

impl<T> IrqMutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            inner: spin::Mutex::new(value),
        }
    }

    pub fn lock(&self) -> IrqGuard<'_, T> {
        let was_enabled = interrupts::are_enabled();
        interrupts::disable();
        IrqGuard {
            guard: ManuallyDrop::new(self.inner.lock()),
            was_enabled,
        }
    }

    /// # Safety
    /// Only for the panic path, where the holder will never run again.
    #[allow(dead_code)]
    pub unsafe fn force_unlock(&self) {
        // SAFETY: forwarded to the caller's contract.
        unsafe { self.inner.force_unlock() };
    }
}

impl<T> Deref for IrqGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for IrqGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for IrqGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: dropped exactly once, here, before interrupts are restored.
        unsafe { ManuallyDrop::drop(&mut self.guard) };
        if self.was_enabled {
            interrupts::enable();
        }
    }
}
