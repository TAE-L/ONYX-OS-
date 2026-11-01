//! Build, run and test driver for OnyxOS: `cargo xtask <command>`.
//!
//! One Rust program instead of a pile of shell scripts, so the same commands
//! work on Windows and Linux, and the test rules (one VM at a time, timeouts,
//! exit-status protocol) are enforced by code rather than by convention.
//!
//! Commands:
//!   build [--release]            build the kernel
//!   run   [--release] [--accel X] boot in QEMU, serial on this terminal
//!   test  [--release] [--accel X] [--timeout S]  boot the self-test build, check results
//!   check                        fmt + clippy + test
//!   clean                        remove build output

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const TARGET: &str = "x86_64-unknown-none";
/// QEMU exit statuses produced by the kernel's debug-exit writes: (value << 1) | 1.
const EXIT_PASS: i32 = 33;
const EXIT_FAIL: i32 = 35;
const EXIT_PANIC: i32 = 37;

type Res<T> = Result<T, String>;

struct Opts {
    release: bool,
    accel: Option<String>,
    timeout: Duration,
    smp: usize,
}

/// Virtual CPUs for the VM: the host's, capped at 4 (emulation gets noisy beyond that).
fn default_smp() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get().clamp(2, 4))
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf()
}

fn parse(args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        release: false,
        accel: None,
        timeout: Duration::from_secs(180),
        smp: default_smp(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--release" => o.release = true,
            "--accel" => o.accel = Some(it.next().ok_or("--accel needs a value")?.clone()),
            "--timeout" => {
                let s: u64 = it
                    .next()
                    .ok_or("--timeout needs seconds")?
                    .parse()
                    .map_err(|_| "--timeout must be a number")?;
                o.timeout = Duration::from_secs(s);
            }
            "--smp" => {
                o.smp = it
                    .next()
                    .ok_or("--smp needs a CPU count")?
                    .parse()
                    .map_err(|_| "--smp must be a number")?;
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(o)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => ("", &[][..]),
    };
    let result = match cmd {
        "build" => parse(rest).and_then(|o| build(&o, false).map(|_| ())),
        "run" => parse(rest).and_then(|o| run(&o)),
        "test" => parse(rest).and_then(|o| test(&o)),
        "check" => check(),
        "clean" => fs::remove_dir_all(root().join("target")).map_err(|e| e.to_string()),
        _ => Err("usage: cargo xtask <build|run|test|check|clean> [--release] [--accel kvm|tcg|whpx] [--timeout SECS] [--smp N]".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// --- building ----------------------------------------------------------------

/// Cargo flags shared by build and clippy. `-Zbuild-std` compiles `core` and
/// `alloc` from the `rust-src` component; `RUSTC_BOOTSTRAP` lets that work on
/// a stable toolchain too.
fn cargo(subcommand: &str, release: bool) -> Command {
    let mut c = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    c.current_dir(root()).env("RUSTC_BOOTSTRAP", "1").args([
        subcommand,
        "-p",
        "kernel",
        "--target",
        TARGET,
        "-Zbuild-std=core,alloc",
        "-Zbuild-std-features=compiler-builtins-mem",
    ]);
    if release {
        c.arg("--release");
    }
    c
}

fn run_checked(mut c: Command) -> Res<()> {
    let status = c.status().map_err(|e| format!("cannot start {c:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed ({status}): {c:?}"))
    }
}

/// The ring-3 test programs, built as static ELFs into `target/user/`. The
/// kernel's self-test embeds them with `include_bytes!`.
fn build_user() -> Res<()> {
    let mut c = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    c.current_dir(root()).env("RUSTC_BOOTSTRAP", "1").args([
        "build",
        "-p",
        "userland",
        "--release",
        "--target",
        TARGET,
        "-Zbuild-std=core",
        "-Zbuild-std-features=compiler-builtins-mem",
    ]);
    run_checked(c)?;
    let out = root().join("target").join("user");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let built = root().join("target").join(TARGET).join("release");
    for name in ["hello", "faults", "spin", "iso", "badptr", "sysc"] {
        let from = built.join(name);
        fs::copy(&from, out.join(format!("{name}.elf")))
            .map_err(|e| format!("copy {}: {e}", from.display()))?;
    }
    Ok(())
}

fn build(o: &Opts, selftest: bool) -> Res<PathBuf> {
    if selftest {
        build_user()?;
    }
    let mut c = cargo("build", o.release);
    if selftest {
        c.args(["--features", "selftest"]);
    }
    run_checked(c)?;
    Ok(root()
        .join("target")
        .join(TARGET)
        .join(if o.release { "release" } else { "debug" })
        .join("kernel"))
}

/// Lay out the EFI System Partition as a plain directory; QEMU serves it to
/// the firmware as a FAT drive, so no disk-image tooling is needed.
fn make_esp(kernel: &Path) -> Res<PathBuf> {
    let esp = root().join("target").join("esp");
    let _ = fs::remove_dir_all(&esp);
    fs::create_dir_all(esp.join("EFI/BOOT")).map_err(|e| e.to_string())?;
    fs::create_dir_all(esp.join("boot")).map_err(|e| e.to_string())?;
    let copy = |from: PathBuf, to: PathBuf| -> Res<()> {
        fs::copy(&from, &to)
            .map(|_| ())
            .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))
    };
    copy(
        root().join("third_party/limine/BOOTX64.EFI"),
        esp.join("EFI/BOOT/BOOTX64.EFI"),
    )?;
    copy(
        root().join("boot/limine.conf"),
        esp.join("EFI/BOOT/limine.conf"),
    )?;
    copy(kernel.to_path_buf(), esp.join("boot/onyxos"))?;
    Ok(esp)
}

// --- QEMU ----------------------------------------------------------------------

fn which(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .flat_map(|d| [d.join(name), d.join(format!("{name}.exe"))])
        .find(|p| p.is_file())
}

fn qemu_binary() -> Res<PathBuf> {
    if let Some(p) = env::var_os("ONYX_QEMU") {
        return Ok(p.into());
    }
    which("qemu-system-x86_64")
        .or_else(|| {
            let p = PathBuf::from(r"C:\Program Files\qemu\qemu-system-x86_64.exe");
            p.is_file().then_some(p)
        })
        .ok_or_else(|| "qemu-system-x86_64 not found (set ONYX_QEMU or add it to PATH)".into())
}

/// Locate UEFI firmware: (code image, variable-store template).
fn find_ovmf() -> Res<(PathBuf, PathBuf)> {
    if let (Some(c), Some(v)) = (env::var_os("ONYX_OVMF_CODE"), env::var_os("ONYX_OVMF_VARS")) {
        return Ok((c.into(), v.into()));
    }
    let pairs = [
        (
            "/usr/share/OVMF/OVMF_CODE_4M.fd",
            "/usr/share/OVMF/OVMF_VARS_4M.fd",
        ),
        (
            "/usr/share/OVMF/OVMF_CODE.fd",
            "/usr/share/OVMF/OVMF_VARS.fd",
        ),
        (
            "/usr/share/edk2/x64/OVMF_CODE.4m.fd",
            "/usr/share/edk2/x64/OVMF_VARS.4m.fd",
        ),
        (
            "/usr/share/edk2/ovmf/OVMF_CODE.fd",
            "/usr/share/edk2/ovmf/OVMF_VARS.fd",
        ),
        (
            "/usr/share/qemu/edk2-x86_64-code.fd",
            "/usr/share/qemu/edk2-i386-vars.fd",
        ),
        (
            r"C:\Program Files\qemu\share\edk2-x86_64-code.fd",
            r"C:\Program Files\qemu\share\edk2-i386-vars.fd",
        ),
    ];
    pairs
        .iter()
        .map(|(c, v)| (PathBuf::from(c), PathBuf::from(v)))
        .find(|(c, v)| c.is_file() && v.is_file())
        .ok_or_else(|| "UEFI firmware not found (set ONYX_OVMF_CODE and ONYX_OVMF_VARS)".into())
}

fn pick_accel(o: &Opts) -> String {
    if let Some(a) = &o.accel {
        return a.clone();
    }
    if Path::new("/dev/kvm").exists() {
        "kvm".into()
    } else {
        "tcg".into()
    }
}

/// One VM at a time, enforced: a second `xtask` run fails fast instead of
/// killing the first run's QEMU or corrupting its logs. This is an OS-level
/// file lock, so it is released automatically even if the holder is killed.
struct VmLock(#[allow(dead_code)] fs::File);

impl VmLock {
    fn acquire() -> Res<Self> {
        let path = root().join("target").join("vm.lock");
        fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Self(file)),
            Err(fs::TryLockError::WouldBlock) => {
                Err("another VM run is active; wait for it to finish".into())
            }
            Err(fs::TryLockError::Error(e)) => Err(format!("cannot lock {}: {e}", path.display())),
        }
    }
}

fn qemu_command(esp: &Path, serial: &str, accel: &str, smp: usize) -> Res<Command> {
    let (code, vars_template) = find_ovmf()?;
    let vars = root().join("target").join("ovmf_vars.fd");
    fs::copy(&vars_template, &vars).map_err(|e| format!("copy OVMF vars: {e}"))?;
    let mut c = Command::new(qemu_binary()?);
    c.args(["-machine", "q35", "-cpu", "max", "-m", "512M"])
        .args(["-smp", &smp.to_string()])
        .args(["-accel", accel])
        .args(["-display", "none", "-no-reboot"])
        .args(["-serial", serial])
        .args(["-device", "isa-debug-exit,iobase=0xf4,iosize=0x04"])
        .arg("-drive")
        .arg(format!(
            "if=pflash,format=raw,unit=0,readonly=on,file={}",
            code.display()
        ))
        .arg("-drive")
        .arg(format!(
            "if=pflash,format=raw,unit=1,file={}",
            vars.display()
        ))
        .arg("-drive")
        .arg(format!("format=raw,file=fat:rw:{}", esp.display()));
    Ok(c)
}

fn run(o: &Opts) -> Res<()> {
    let kernel = build(o, false)?;
    let esp = make_esp(&kernel)?;
    let _lock = VmLock::acquire()?;
    let accel = pick_accel(o);
    eprintln!("booting under QEMU ({accel}); Ctrl-C to stop");
    let mut c = qemu_command(&esp, "stdio", &accel, o.smp)?;
    c.status().map(|_| ()).map_err(|e| e.to_string())
}

fn test(o: &Opts) -> Res<()> {
    let kernel = build(o, true)?;
    let esp = make_esp(&kernel)?;
    let _lock = VmLock::acquire()?;
    let accel = pick_accel(o);
    let log = root().join("target").join("serial.log");
    let _ = fs::remove_file(&log);
    let mut c = qemu_command(&esp, &format!("file:{}", log.display()), &accel, o.smp)?;
    c.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| format!("cannot start QEMU: {e}"))?;

    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(s);
        }
        if start.elapsed() > o.timeout {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(100));
    };
    let serial = fs::read_to_string(&log).unwrap_or_default();

    let Some(status) = status else {
        print_log(&serial, usize::MAX);
        return Err(format!(
            "TIMEOUT after {}s with no exit from the kernel",
            o.timeout.as_secs()
        ));
    };
    let code = status.code().unwrap_or(-1);
    let result_line = serial.lines().find(|l| l.starts_with("SELFTEST RESULT"));
    let kernel_panicked = serial.contains("*** KERNEL PANIC ***");

    match (code, result_line, kernel_panicked) {
        (EXIT_PASS, Some(line), false) if line.contains(" 0 failed") => {
            print_log(&serial, 40);
            eprintln!(
                "\nOK ({accel}, {:.1}s): {line}",
                start.elapsed().as_secs_f32()
            );
            Ok(())
        }
        (EXIT_FAIL, ..) => {
            print_log(&serial, usize::MAX);
            Err("self-test failures (see [FAIL] lines above)".into())
        }
        (EXIT_PANIC, ..) | (_, _, true) => {
            print_log(&serial, usize::MAX);
            Err("the kernel panicked during the self-test".into())
        }
        (0, None, _) => {
            print_log(&serial, usize::MAX);
            Err(
                "the VM exited without reporting a result; this is the signature of a \
                 triple fault or CPU reset (see the last serial lines above)"
                    .into(),
            )
        }
        _ => {
            print_log(&serial, usize::MAX);
            let stderr = child.stderr.take().map(|mut s| {
                let mut buf = String::new();
                let _ = std::io::Read::read_to_string(&mut s, &mut buf);
                buf
            });
            Err(format!(
                "unexpected QEMU exit status {code}; QEMU stderr: {}",
                stderr.unwrap_or_default().trim()
            ))
        }
    }
}

fn print_log(log: &str, tail: usize) {
    let lines: Vec<&str> = log.lines().collect();
    let skip = lines.len().saturating_sub(tail);
    for l in &lines[skip..] {
        eprintln!("{l}");
    }
}

fn check() -> Res<()> {
    let mut fmt = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    fmt.current_dir(root())
        .args(["fmt", "--all", "--", "--check"]);
    run_checked(fmt)?;
    build_user()?;
    let mut clippy = cargo("clippy", false);
    clippy.args(["--features", "selftest", "--", "-D", "warnings"]);
    run_checked(clippy)?;
    test(&Opts {
        release: false,
        accel: None,
        timeout: Duration::from_secs(180),
        smp: default_smp(),
    })
}
