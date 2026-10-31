// build.rs
//
// Links this crate against Phase 1's pre-built `xenon_inference` shared library
// (inference-engine/build-cuda-app) and copies its runtime DLLs (xenon_inference.dll, rwkv.dll,
// cuBLAS, plus the MSVC/UCRT redistributable DLLs that build already collected next to
// test_inference.exe) next to this crate's own build output, so the Tauri app can load them at
// runtime without editing PATH.
//
// This links the CUDA build so RWKV inference runs on the NVIDIA GPU. On this machine that is
// an RTX 2060 (6 GB). build-cpu-app remains a valid alternate configure of the C++ library.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // manifest_dir = <repo>/app/src-tauri -> repo root is two levels up.
    let repo_root = manifest_dir
        .parent()
        .expect("app/ dir")
        .parent()
        .expect("repo root")
        .to_path_buf();

    let build_dir = repo_root.join("inference-engine").join("build-cuda-app");
    let lib_dir = build_dir.join("Release"); // xenon_inference.lib (import lib)
    let dll_dir = build_dir.join("bin").join("Release"); // xenon_inference.dll, rwkv.dll, cublas, redist DLLs

    if !lib_dir.join("xenon_inference.lib").exists() {
        panic!(
            "xenon_inference.lib not found at {} -- build inference-engine/build-cuda-app first \
             (see inference-engine/README.md, section '2. Build the xenon_inference wrapper').",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xenon_inference");
    println!("cargo:rerun-if-changed={}", lib_dir.display());

    // Copy every DLL next to the eventual .exe so the app can find them without PATH edits.
    // OUT_DIR looks like <target-dir>/<profile>/build/app-<hash>/out -- the target/profile dir
    // that holds the actual executable is three levels up from OUT_DIR.
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let mut target_dir = PathBuf::from(out_dir);
        for _ in 0..3 {
            target_dir.pop();
        }

        if let Ok(entries) = fs::read_dir(&dll_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("dll") {
                    let dest = target_dir.join(path.file_name().unwrap());
                    // Best-effort: don't fail the build if a DLL happens to be locked by a
                    // running instance of the app.
                    let _ = fs::copy(&path, &dest);
                }
            }
        }
    }

    tauri_build::build()
}
