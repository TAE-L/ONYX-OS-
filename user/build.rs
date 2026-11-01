use std::{env, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!(
        "cargo:rustc-link-arg-bins=-T{}",
        manifest.join("user.ld").display()
    );
    println!("cargo:rustc-link-arg-bins=-zmax-page-size=0x1000");
    println!("cargo:rerun-if-changed=user.ld");
    println!("cargo:rerun-if-changed=build.rs");
}
