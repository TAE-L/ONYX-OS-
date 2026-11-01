//! Build script: linker arguments and the 256 interrupt entry stubs.
//!
//! The stubs are generated (not hand-written) so that every vector pushes a
//! uniform trap frame: `[vector][error code]` followed by the CPU-pushed
//! frame. Vectors without a hardware error code push a zero.

use std::{env, fmt::Write as _, fs, path::PathBuf};

/// Vectors for which the CPU pushes an error code.
const HAS_ERROR_CODE: [usize; 10] = [8, 10, 11, 12, 13, 14, 17, 21, 29, 30];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!(
        "cargo:rustc-link-arg=-T{}",
        manifest.join("linker.ld").display()
    );
    println!("cargo:rustc-link-arg=-zmax-page-size=0x1000");
    println!("cargo:rerun-if-changed=linker.ld");
    println!("cargo:rerun-if-changed=build.rs");

    let mut asm = String::new();
    asm.push_str(".section .text.isr,\"ax\"\n");
    for v in 0..256usize {
        let _ = writeln!(
            asm,
            ".global isr_stub_{v}\n.hidden isr_stub_{v}\nisr_stub_{v}:"
        );
        if !HAS_ERROR_CODE.contains(&v) {
            asm.push_str("    push 0\n");
        }
        let _ = writeln!(asm, "    push {v}\n    jmp isr_common");
    }
    asm.push_str(
        "isr_common:
    // Entered from ring 3? Then GS still holds the user's base: swap in the kernel's.
    test byte ptr [rsp + 24], 3
    jz 1f
    swapgs
1:
    push rax
    push rcx
    push rdx
    push rbx
    push rbp
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov rdi, rsp
    cld
    call isr_dispatch
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rbp
    pop rbx
    pop rdx
    pop rcx
    pop rax
    add rsp, 16
    test byte ptr [rsp + 8], 3
    jz 2f
    swapgs
2:
    iretq
",
    );
    asm.push_str(".section .rodata\n.balign 8\n.global isr_stub_table\n.hidden isr_stub_table\nisr_stub_table:\n");
    for v in 0..256usize {
        let _ = writeln!(asm, "    .quad isr_stub_{v}");
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("isr_stubs.rs");
    fs::write(out, format!("core::arch::global_asm!(r#\"\n{asm}\"#);\n")).unwrap();
}
