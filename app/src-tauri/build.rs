// build.rs
//
// Links this crate against the pre-built `xenon_inference` shared library and copies that
// library (and librwkv / rwkv.dll, plus cuBLAS on a CUDA build) next to the Tauri executable.
//
// Prefer inference-engine/build-cuda-app so an RTX 2060 (or any CUDA GPU) actually runs the
// model. Fall back to build-cpu-app when the CUDA library has not been built yet. At runtime
// inference.rs offloads every RWKV layer when the linked library was compiled with CUDA, and
// retries on CPU if the GPU init fails, so a CUDA build still launches on a machine with no GPU.
//
// A Visual Studio multi-config build puts the import library in `<build>/Release` and the
// runtime DLLs (xenon_inference.dll, rwkv.dll, cuBLAS, MSVC/UCRT redistributables) in
// `<build>/bin/Release`. Both directories are copied from.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn is_native_lib(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some("dll") | Some("so") | Some("dylib") => true,
        _ => false,
    }
}

fn is_link_lib(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if cfg!(windows) {
        name == "xenon_inference.lib"
    } else if cfg!(target_os = "macos") {
        name == "libxenon_inference.dylib" || name == "libxenon_inference.a"
    } else {
        name == "libxenon_inference.so"
    }
}

fn find_link_lib(build_dir: &Path) -> Option<PathBuf> {
    let mut dirs = vec![
        build_dir.join("bin").join("Release"),
        build_dir.join("bin"),
        build_dir.join("Release"),
        build_dir.to_path_buf(),
    ];
    // Single-config Linux builds drop the .so in bin/ via LIBRARY_OUTPUT_DIRECTORY.
    dirs.retain(|d| d.is_dir());
    for dir in dirs {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if is_link_lib(&path) {
                return Some(path);
            }
        }
    }
    None
}

/// Directories that hold runtime libraries to copy next to the executable.
///
/// The link library's own directory is always included. On a Visual Studio multi-config
/// build the import lib lives in `<build>/<Config>` while the DLLs live in `<build>/bin/<Config>`.
fn runtime_dirs(link_lib: &Path) -> Vec<PathBuf> {
    let lib_dir = link_lib.parent().expect("library parent dir");
    let mut dirs = Vec::new();
    if let (Some(config), Some(build_root)) = (lib_dir.file_name(), lib_dir.parent()) {
        let bin_config = build_root.join("bin").join(config);
        if bin_config.is_dir() && bin_config != lib_dir {
            dirs.push(bin_config);
        }
    }
    dirs.push(lib_dir.to_path_buf());
    dirs
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .expect("app/ dir")
        .parent()
        .expect("repo root")
        .to_path_buf();

    let engine_dir = repo_root.join("inference-engine");
    let cuda_dir = engine_dir.join("build-cuda-app");
    let cpu_dir = engine_dir.join("build-cpu-app");

    let (build_dir, link_lib) = if let Some(lib) = find_link_lib(&cuda_dir) {
        (cuda_dir, lib)
    } else if let Some(lib) = find_link_lib(&cpu_dir) {
        (cpu_dir, lib)
    } else {
        panic!(
            "xenon_inference library not found under {} or {}. \
             On a machine with an RTX 2060, build the CUDA library first: \
             inference-engine/build_cuda.sh (CMAKE_CUDA_ARCHITECTURES=75). \
             Otherwise build inference-engine/build-cpu-app. See inference-engine/README.md.",
            cuda_dir.display(),
            cpu_dir.display()
        );
    };

    let lib_dir = link_lib.parent().expect("library parent dir");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xenon_inference");
    println!("cargo:rerun-if-changed={}", link_lib.display());
    println!("cargo:rerun-if-changed={}", build_dir.display());

    if cfg!(unix) {
        // Look beside the executable for libxenon_inference.so and librwkv.so.
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }

    if let Ok(out_dir) = env::var("OUT_DIR") {
        let mut target_dir = PathBuf::from(out_dir);
        for _ in 0..3 {
            target_dir.pop();
        }

        for dir in runtime_dirs(&link_lib) {
            let Ok(entries) = fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                if is_native_lib(&path) {
                    let dest = target_dir.join(path.file_name().unwrap());
                    let _ = fs::copy(&path, &dest);
                    // Rebuilding the engine rewrites these without touching the folder or the
                    // .lib, so without this the app kept running the old engine DLLs.
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }

    tauri_build::build()
}
