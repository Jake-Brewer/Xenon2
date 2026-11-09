//! Model files kept in git as parts. GitHub's LFS takes files up to 2 GB, and a 2.9B Q5_1 model is
//! 2.75 GB, so `models/<name>.bin` is stored as `<name>.bin.part1`, `.part2`, ... plus
//! `<name>.bin.size` (its byte count). The first run joins them into `<name>.bin`, which git
//! ignores. `split_into_parts` makes them.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

fn part_path(model: &Path, n: usize) -> PathBuf {
    let mut name = model.as_os_str().to_owned();
    name.push(format!(".part{n}"));
    PathBuf::from(name)
}

fn size_path(model: &Path) -> PathBuf {
    let mut name = model.as_os_str().to_owned();
    name.push(".size");
    PathBuf::from(name)
}

fn parts_of(model: &Path) -> Vec<PathBuf> {
    (1..).map(|n| part_path(model, n)).take_while(|p| p.exists()).collect()
}

/// Makes sure `model` exists, joining it from its parts when it doesn't. Does nothing when the
/// file is already there or there are no parts (the caller's load then reports the missing file).
pub fn ensure_joined(model: &Path) -> Result<(), String> {
    if model.exists() {
        return Ok(());
    }
    let parts = parts_of(model);
    if parts.is_empty() {
        return Ok(());
    }
    let expected: u64 = fs::read_to_string(size_path(model))
        .map_err(|e| format!("{} has parts but no .size file: {e}", model.display()))?
        .trim()
        .parse()
        .map_err(|e| format!("bad .size file for {}: {e}", model.display()))?;

    let mut joining = model.as_os_str().to_owned();
    joining.push(".joining");
    let joining = PathBuf::from(joining);
    let result = (|| -> Result<u64, String> {
        let mut out = File::create(&joining).map_err(|e| format!("couldn't create {}: {e}", joining.display()))?;
        let mut total = 0u64;
        for part in &parts {
            let mut input = File::open(part).map_err(|e| format!("couldn't open {}: {e}", part.display()))?;
            let mut head = [0u8; 24];
            let n = input.read(&mut head).map_err(|e| e.to_string())?;
            if head[..n].starts_with(LFS_POINTER_PREFIX) {
                return Err(format!(
                    "{} is a Git LFS pointer, not the model data. Run `git lfs pull` in the repo, then start again.",
                    part.display()
                ));
            }
            out.write_all(&head[..n]).map_err(|e| e.to_string())?;
            total += n as u64 + io::copy(&mut input, &mut out).map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
        Ok(total)
    })();
    match result {
        Ok(total) if total == expected => {
            fs::rename(&joining, model).map_err(|e| format!("couldn't rename {}: {e}", joining.display()))?;
            eprintln!("xenon2: joined {} from {} parts ({total} bytes)", model.display(), parts.len());
            Ok(())
        }
        Ok(total) => {
            let _ = fs::remove_file(&joining);
            Err(format!("joined {} is {total} bytes, expected {expected}; a part is missing or damaged", model.display()))
        }
        Err(e) => {
            let _ = fs::remove_file(&joining);
            Err(e)
        }
    }
}

/// Splits `model` into parts of at most `max_part` bytes and writes its `.size` file.
pub fn split_into_parts(model: &Path, max_part: u64) -> Result<usize, String> {
    let size = fs::metadata(model).map_err(|e| e.to_string())?.len();
    let mut input = File::open(model).map_err(|e| e.to_string())?;
    let count = size.div_ceil(max_part).max(1) as usize;
    let per = size.div_ceil(count as u64);
    for n in 1..=count {
        let mut out = File::create(part_path(model, n)).map_err(|e| e.to_string())?;
        io::copy(&mut (&mut input).take(per), &mut out).map_err(|e| e.to_string())?;
    }
    fs::write(size_path(model), format!("{size}\n")).map_err(|e| e.to_string())?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xenon-parts-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn split_then_join_gives_back_the_same_bytes() {
        let dir = temp_dir("roundtrip");
        let model = dir.join("m.bin");
        let data: Vec<u8> = (0..10_007u32).map(|i| (i * 31 % 251) as u8).collect();
        fs::write(&model, &data).unwrap();
        assert_eq!(split_into_parts(&model, 4000).unwrap(), 3);
        fs::remove_file(&model).unwrap();
        ensure_joined(&model).unwrap();
        assert_eq!(fs::read(&model).unwrap(), data);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_lfs_pointer_part_gives_a_clear_error() {
        let dir = temp_dir("pointer");
        let model = dir.join("m.bin");
        fs::write(part_path(&model, 1), "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5\n").unwrap();
        fs::write(size_path(&model), "5\n").unwrap();
        let err = ensure_joined(&model).unwrap_err();
        assert!(err.contains("git lfs pull"), "{err}");
        assert!(!model.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_size_is_rejected_and_leaves_no_model() {
        let dir = temp_dir("size");
        let model = dir.join("m.bin");
        fs::write(part_path(&model, 1), b"hello").unwrap();
        fs::write(size_path(&model), "6\n").unwrap();
        assert!(ensure_joined(&model).is_err());
        assert!(!model.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_existing_model_or_no_parts_is_left_alone() {
        let dir = temp_dir("noop");
        assert!(ensure_joined(&dir.join("absent.bin")).is_ok());
        fs::remove_dir_all(&dir).ok();
    }

    /// Run by hand to split the real models: `cargo test --lib split_real_models -- --ignored`.
    #[test]
    #[ignore]
    fn split_real_models() {
        let models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
        for name in ["rwkv-7-world-2.9B-Q5_1.bin", "rwkv-7-g1l-2.9B-Q5_1.bin"] {
            let count = split_into_parts(&models.join(name), 1_900_000_000).unwrap();
            println!("{name}: {count} parts");
        }
    }
}
