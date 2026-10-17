//! `kprint!` / `kprintln!` over the serial port.

#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => { $crate::serial::print(format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! kprintln {
    () => { $crate::kprint!("\n") };
    ($($arg:tt)*) => { $crate::kprint!("{}\n", format_args!($($arg)*)) };
}
