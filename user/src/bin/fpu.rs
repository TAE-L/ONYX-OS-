//! FPU/SSE/AVX state probe. Starts by checking that the registers hold a clean
//! reset state (nothing leaked from another process), then fills every vector
//! register, MXCSR and the x87 stack with values derived from its argument,
//! yields and spins so other copies run (and the scheduler may move it to
//! another CPU), and checks that every bit survived.
//!
//! The kernel and this crate are soft-float, so the vector instructions live
//! in `global_asm!`. Exit codes: 0 ok, 1 vector registers corrupted, 3 MXCSR, 4 x87 control word, 5 x87 stack, 2 not clean at start.
#![no_std]
#![no_main]

core::arch::global_asm!(
    r#"
    .text
// void vec_load(const u8 *buf [rdi], u64 avx [rsi])  16 regs x 32 bytes
    .global vec_load
vec_load:
    test rsi, rsi
    jz 2f
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    vmovdqu ymm\n, [rdi + 32*\n]
    .endr
    ret
2:
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    movdqu xmm\n, [rdi + 32*\n]
    .endr
    ret

// void vec_store(u8 *buf [rdi], u64 avx [rsi])
    .global vec_store
vec_store:
    test rsi, rsi
    jz 2f
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    vmovdqu [rdi + 32*\n], ymm\n
    .endr
    ret
2:
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    movdqu [rdi + 32*\n], xmm\n
    .endr
    ret

// void ctl_set(u32 mxcsr [edi], u16 fcw [esi], f64 bits [rdx]); leaves 1 value on the x87 stack
    .global ctl_set
ctl_set:
    mov [rsp - 8], edi
    ldmxcsr [rsp - 8]
    mov [rsp - 8], si
    fldcw [rsp - 8]
    mov [rsp - 8], rdx
    fld qword ptr [rsp - 8]
    ret

// void ctl_get(u32 *mxcsr [rdi], u16 *fcw [rsi], u64 *top [rdx])
    .global ctl_get
ctl_get:
    stmxcsr [rdi]
    fnstcw [rsi]
    fld st(0)
    fstp qword ptr [rdx]
    ret

// void ctl_pop(): drop the value ctl_set left on the x87 stack
    .global ctl_pop
ctl_pop:
    fstp st(0)
    ret
"#
);

unsafe extern "C" {
    fn vec_load(buf: *const u8, avx: u64);
    fn vec_store(buf: *mut u8, avx: u64);
    fn ctl_set(mxcsr: u32, fcw: u16, bits: u64);
    fn ctl_pop();
    fn ctl_get(mxcsr: *mut u32, fcw: *mut u16, top: *mut u64);
}

fn has_avx() -> bool {
    let c = core::arch::x86_64::__cpuid(1);
    if c.ecx & (1 << 27) == 0 || c.ecx & (1 << 28) == 0 {
        return false;
    }
    let xcr0: u32;
    // SAFETY: OSXSAVE is set, so xgetbv is legal in ring 3.
    unsafe {
        core::arch::asm!("xgetbv", in("ecx") 0, out("eax") xcr0, out("edx") _,
                         options(nomem, nostack));
    }
    xcr0 & 6 == 6
}

fn pattern(seed: u64, round: u64, buf: &mut [u8; 512]) {
    let mut x =
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ round.wrapping_mul(0xD1B5_4A32_D192_ED03);
    for chunk in buf.chunks_exact_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start(arg: u64) -> ! {
    // Low bit of the argument: 1 = stay on SSE even if AVX exists.
    let avx = u64::from(has_avx() && arg & 1 == 0);
    let width = if avx == 1 { 32 } else { 16 };
    let mut got = [0u8; 512];

    // Clean start: every vector register zero, default control words.
    // SAFETY: the buffers are 512 bytes, the asm writes 16*32.
    unsafe { vec_store(got.as_mut_ptr(), avx) };
    let (mut mx, mut cw, mut top) = (0u32, 0u16, 0u64);
    // SAFETY: valid out-pointers.
    unsafe { ctl_get(&mut mx, &mut cw, &mut top) };
    let dirty = (0..16).any(|r| got[r * 32..r * 32 + width].iter().any(|&b| b != 0));
    if dirty || mx != 0x1F80 || cw != 0x037F {
        userland::print("state leaked into a new process\n");
        userland::exit(2);
    }

    for round in 0..50u64 {
        let mut want = [0u8; 512];
        pattern(arg, round, &mut want);
        // A rounding mode and precision that differ per process and round.
        let rc = ((arg >> 1) + round) & 3;
        let mxcsr = 0x1F80 | ((rc as u32) << 13);
        let pc = [0u16, 2, 3][(((arg >> 1) + round) % 3) as usize];
        let fcw = 0x037F & !0x0F00 | pc << 8 | (rc as u16) << 10;
        let fval = f64::from_bits(
            want[0..8].try_into().map(u64::from_le_bytes).unwrap() & 0x3FFF_FFFF_E000_0000,
        );
        // SAFETY: buffers sized as above.
        unsafe {
            vec_load(want.as_ptr(), avx);
            ctl_set(mxcsr, fcw, fval.to_bits());
        }
        userland::yield_now();
        for _ in 0..3000 {
            core::hint::spin_loop();
        }
        // SAFETY: as above.
        unsafe {
            vec_store(got.as_mut_ptr(), avx);
            ctl_get(&mut mx, &mut cw, &mut top);
            ctl_pop();
        }
        let bad_vec = (0..16).any(|r| got[r * 32..r * 32 + width] != want[r * 32..r * 32 + width]);
        let code = if bad_vec {
            1
        } else if mx != mxcsr {
            3
        } else if cw != fcw {
            4
        } else if top != fval.to_bits() {
            5
        } else {
            0
        };
        if code != 0 {
            let mut b = [0u8; 40];
            for (i, v) in [top, fval.to_bits()].iter().enumerate() {
                for d in 0..16 {
                    let n = ((v >> (60 - 4 * d)) & 15) as u8;
                    b[i * 17 + d] = if n < 10 { b'0' + n } else { b'a' + n - 10 };
                }
                b[i * 17 + 16] = b' ';
            }
            b[33] = b'r';
            b[34] = b'0' + round as u8 % 10;
            b[35] = b'\n';
            // SAFETY: only ASCII was written.
            userland::print(unsafe { core::str::from_utf8_unchecked(&b[..36]) });
            userland::print("FPU state corrupted\n");
            userland::exit(code);
        }
    }
    userland::print("fpu ok\n");
    userland::exit(0)
}
