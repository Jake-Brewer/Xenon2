//! Phase 6 -- external memory export/import, plus the "data directory location" setting.
//!
//! Three related jobs live here:
//! 1. **Export**: copy the real runtime files a working Xenon2 install depends on (quantized
//!    model, tokenizer vocab, piper voice model, all saved conversations + session.json) into a
//!    single portable bundle folder on an external drive, per `EXPORT_FORMAT.md` at the repo root.
//! 2. **Import**: copy a previously-exported bundle's contents back into this machine's real local
//!    paths (copy-in by default -- see `EXPORT_FORMAT.md`'s "Import: copy-in vs. run-in-place").
//! 3. **Data directory setting**: a small `settings.json` (separate from `session.json`) that lets
//!    conversations be saved to/loaded from an external location on an ongoing basis, not just a
//!    one-time export/import.
//!
//! Progress reporting: both export and import copy files large enough (the quantized model is
//! ~450MB) that a naive `fs::copy` would leave the UI looking hung for tens of seconds. Instead,
//! `copy_with_progress` streams the copy in fixed-size chunks and emits a Tauri event
//! (`export-progress` / `import-progress`) after each chunk, so the frontend can render a real
//! per-file byte progress bar instead of a spinner or nothing at all.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

/// Repo root (two levels up from `app/src-tauri`), managed as Tauri state at startup (see
/// `lib.rs`) so this module can find `models/`, `inference-engine/data/`, and
/// `voice-pipeline/models/` without recomputing `CARGO_MANIFEST_DIR` logic itself.
pub struct RepoRoot(pub PathBuf);

/// Chunk size for progress-reporting copies. Small enough that progress updates feel smooth on a
/// ~450MB file, large enough not to spam events (450MB / 1MB = ~450 events per file, well within
/// what a Tauri event listener can handle for a few large files).
const COPY_CHUNK_BYTES: usize = 1024 * 1024;

// ---------------------------------------------------------------------------------------------
// Settings: "data directory location"
// ---------------------------------------------------------------------------------------------

/// Persisted at `<app_data_dir>/settings.json` -- deliberately a separate file from
/// `session.json` (which is restore-on-launch bookkeeping, not user-facing configuration).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// If set, conversations are saved to/loaded from `<dataDir>/conversations/` instead of
    /// `<app_data_dir>/conversations/`. `None` (the default) means "use the local app-data
    /// default", matching every pre-Phase-6 behavior exactly.
    #[serde(default)]
    pub data_dir: Option<String>,
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| format!("Could not resolve the app data directory: {e}"))
}

fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("settings.json"))
}

#[tauri::command]
pub fn load_settings(app: AppHandle) -> Result<Settings, String> {
    let path = settings_path(&app)?;
    if !path.exists() {
        return Ok(Settings::default());
    }
    let raw = fs::read_to_string(&path).map_err(|e| format!("Could not read '{}': {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("Could not parse settings.json: {e}"))
}

#[tauri::command]
pub fn save_settings(app: AppHandle, settings: Settings) -> Result<(), String> {
    let dir = app_data_dir(&app)?;
    fs::create_dir_all(&dir).map_err(|e| format!("Could not create '{}': {e}", dir.display()))?;
    let path = dir.join("settings.json");
    let json = serde_json::to_string_pretty(&settings)
        .map_err(|e| format!("Could not serialize settings: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Could not write '{}': {e}", path.display()))
}

/// The configured data directory, or the local app-data directory when none is set. Files that
/// belong with the user's data rather than with the install, such as `feedback.jsonl`, live here.
pub fn effective_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    match load_settings(app.clone())?.data_dir {
        Some(d) if !d.trim().is_empty() => Ok(PathBuf::from(d)),
        _ => app_data_dir(app),
    }
}

/// Where conversations are currently saved to/loaded from: `<dataDir>/conversations` if the
/// "data directory location" setting is configured, otherwise `<app_data_dir>/conversations` --
/// exactly the Phase 5 default. This is the one function `persistence.rs`'s
/// `default_conversation_path` needs to call to "honor" the setting for ongoing saves/loads.
#[tauri::command]
pub fn effective_conversations_dir(app: AppHandle) -> Result<String, String> {
    let settings = load_settings(app.clone())?;
    let dir = match settings.data_dir {
        Some(d) if !d.trim().is_empty() => PathBuf::from(d).join("conversations"),
        _ => app_data_dir(&app)?.join("conversations"),
    };
    Ok(dir.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------------------------
// Folder picker (mirrors persistence.rs's pick_save_dialog/pick_open_dialog exactly, including
// the timeout hardening -- see that file's doc comments for why: the OS Common Item Dialog can
// hang on this dev machine, confirmed independent of Tauri/rfd).
// ---------------------------------------------------------------------------------------------

const DIALOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn test_override_path(env_var: &str) -> Option<String> {
    std::env::var(env_var).ok().filter(|s| !s.is_empty())
}

#[tauri::command]
pub async fn pick_folder_dialog(_app: AppHandle, test_env_var: Option<String>) -> Result<Option<String>, String> {
    if let Some(var) = &test_env_var {
        if let Some(path) = test_override_path(var) {
            return Ok(Some(path));
        }
    }

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let picked = rfd::FileDialog::new().pick_folder();
        let _ = tx.send(picked.map(|p| p.to_string_lossy().into_owned()));
    });

    tauri::async_runtime::spawn_blocking(move || rx.recv_timeout(DIALOG_TIMEOUT))
        .await
        .map_err(|e| format!("Folder dialog task failed: {e}"))?
        .map_err(|_| {
            "The folder dialog did not respond within 30 seconds. This usually means a Windows \
             shell extension (e.g. a cloud-sync client) is hanging while the dialog loads -- try \
             again, or check Windows for a misbehaving shell extension."
                .to_string()
        })
}

// ---------------------------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct MemoryProgressEvent {
    /// Bundle-relative label, e.g. "models/rwkv-7-world-2.9B-Q5_1.bin".
    file: String,
    #[serde(rename = "fileIndex")]
    file_index: usize,
    #[serde(rename = "totalFiles")]
    total_files: usize,
    #[serde(rename = "bytesDone")]
    bytes_done: u64,
    #[serde(rename = "bytesTotal")]
    bytes_total: u64,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryDoneEvent {
    #[serde(rename = "destination")]
    destination: String,
    #[serde(rename = "filesCopied")]
    files_copied: usize,
    #[serde(rename = "totalBytes")]
    total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryErrorEvent {
    message: String,
}

/// Copies `src` to `dst` in fixed-size chunks, emitting `event_name` after every chunk so the
/// frontend can render real byte progress instead of appearing to hang on a ~450MB file. Creates
/// `dst`'s parent directory if needed. Returns the number of bytes copied.
fn copy_with_progress(
    app: &AppHandle,
    event_name: &str,
    src: &Path,
    dst: &Path,
    label: &str,
    file_index: usize,
    total_files: usize,
) -> Result<u64, String> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create '{}': {e}", parent.display()))?;
    }

    let mut src_file = fs::File::open(src).map_err(|e| format!("Could not open '{}': {e}", src.display()))?;
    let bytes_total = src_file
        .metadata()
        .map_err(|e| format!("Could not stat '{}': {e}", src.display()))?
        .len();
    let mut dst_file = fs::File::create(dst).map_err(|e| format!("Could not create '{}': {e}", dst.display()))?;

    let mut buf = vec![0u8; COPY_CHUNK_BYTES];
    let mut bytes_done: u64 = 0;

    // Emit an initial 0-byte progress event immediately so a slow-to-start large file shows up in
    // the UI right away rather than only after the first chunk completes.
    let _ = app.emit(
        event_name,
        MemoryProgressEvent {
            file: label.to_string(),
            file_index,
            total_files,
            bytes_done: 0,
            bytes_total,
        },
    );

    loop {
        let n = src_file
            .read(&mut buf)
            .map_err(|e| format!("Could not read '{}': {e}", src.display()))?;
        if n == 0 {
            break;
        }
        dst_file
            .write_all(&buf[..n])
            .map_err(|e| format!("Could not write '{}': {e}", dst.display()))?;
        bytes_done += n as u64;

        let _ = app.emit(
            event_name,
            MemoryProgressEvent {
                file: label.to_string(),
                file_index,
                total_files,
                bytes_done,
                bytes_total,
            },
        );
    }

    Ok(bytes_done)
}

/// One entry in the copy plan: an absolute source path, and its label relative to the bundle
/// root (used both for the destination path under the bundle and for progress event display).
struct PlanEntry {
    src: PathBuf,
    rel: String,
}

fn required_model_files(repo_root: &Path) -> Vec<PlanEntry> {
    vec![
        PlanEntry {
            src: repo_root.join("models").join("rwkv-7-world-2.9B-Q5_1.bin"),
            rel: "models/rwkv-7-world-2.9B-Q5_1.bin".to_string(),
        },
        PlanEntry {
            src: repo_root
                .join("inference-engine")
                .join("data")
                .join("world_vocab.bin"),
            rel: "models/world_vocab.bin".to_string(),
        },
        PlanEntry {
            src: repo_root
                .join("voice-pipeline")
                .join("models")
                .join("en_GB-alan-medium.onnx"),
            rel: "voice-models/en_GB-alan-medium.onnx".to_string(),
        },
        PlanEntry {
            src: repo_root
                .join("voice-pipeline")
                .join("models")
                .join("en_GB-alan-medium.onnx.json"),
            rel: "voice-models/en_GB-alan-medium.onnx.json".to_string(),
        },
    ]
}

/// Exports the real runtime files (quantized model + vocab + voice model + all conversations +
/// session.json) into `<destination>/xenon2-backup/`, matching the layout in `EXPORT_FORMAT.md`.
/// Deliberately excludes the `.pth`/FP16 conversion intermediates (~1.8GB, not loaded at runtime)
/// -- see `EXPORT_FORMAT.md` for why.
#[tauri::command]
pub async fn export_memory(
    app: AppHandle,
    repo_root: tauri::State<'_, RepoRoot>,
    destination: String,
) -> Result<(), String> {
    let repo_root = repo_root.0.clone();
    let dest_root = PathBuf::from(&destination).join("xenon2-backup");

    let app_data = app_data_dir(&app)?;
    let conversations_dir = PathBuf::from(effective_conversations_dir(app.clone())?);
    let session_path = app_data.join("session.json");
    let feedback_path = crate::feedback::feedback_path(&app)?;

    let app_clone = app.clone();
    let dest_root_for_closure = dest_root.clone();
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<(usize, u64), String> {
        let dest_root = dest_root_for_closure;
        let mut plan = required_model_files(&repo_root);

        for entry in &plan {
            if !entry.src.exists() {
                return Err(format!(
                    "Required file '{}' is missing. Cannot export a working bundle without it.",
                    entry.src.display()
                ));
            }
        }

        // Conversations: every *.json in the effective conversations dir, plus session.json.
        if conversations_dir.exists() {
            let mut names: Vec<PathBuf> = fs::read_dir(&conversations_dir)
                .map_err(|e| format!("Could not read '{}': {e}", conversations_dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|ext| ext == "json").unwrap_or(false))
                .collect();
            names.sort();
            for path in names {
                let file_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                plan.push(PlanEntry {
                    src: path,
                    rel: format!("conversations/{file_name}"),
                });
            }
        }

        if session_path.exists() {
            plan.push(PlanEntry {
                src: session_path.clone(),
                rel: "conversations/session.json".to_string(),
            });
        }

        // Ratings for learning from feedback. See feedback.rs.
        if feedback_path.exists() {
            plan.push(PlanEntry {
                src: feedback_path.clone(),
                rel: "feedback/feedback.jsonl".to_string(),
            });
        }

        let total_files = plan.len();
        let mut total_bytes: u64 = 0;

        for (i, entry) in plan.iter().enumerate() {
            let dst = dest_root.join(&entry.rel);
            let copied = copy_with_progress(
                &app_clone,
                "export-progress",
                &entry.src,
                &dst,
                &entry.rel,
                i,
                total_files,
            )?;
            total_bytes += copied;
        }

        Ok((total_files, total_bytes))
    })
    .await
    .map_err(|e| format!("Export task failed: {e}"))?;

    match result {
        Ok((files_copied, total_bytes)) => {
            let _ = app.emit(
                "export-done",
                MemoryDoneEvent {
                    destination: dest_root.to_string_lossy().into_owned(),
                    files_copied,
                    total_bytes,
                },
            );
            Ok(())
        }
        Err(msg) => {
            let _ = app.emit("export-error", MemoryErrorEvent { message: msg.clone() });
            Err(msg)
        }
    }
}

/// Imports a previously-exported bundle from `source` (a folder either containing
/// `models/`/`voice-models/`/`conversations/` directly, or an `xenon2-backup/` folder that
/// contains them -- both are accepted so pointing Import at either the drive root or the bundle
/// folder itself works) into this machine's real local paths. Copy-in by default -- see
/// `EXPORT_FORMAT.md`'s "Import: copy-in vs. run-in-place" section for why this is the only mode
/// implemented.
#[tauri::command]
pub async fn import_memory(
    app: AppHandle,
    repo_root: tauri::State<'_, RepoRoot>,
    source: String,
) -> Result<(), String> {
    let repo_root = repo_root.0.clone();
    let source_root = resolve_bundle_root(&PathBuf::from(&source));

    let app_data = app_data_dir(&app)?;
    let conversations_dir = PathBuf::from(effective_conversations_dir(app.clone())?);
    let session_dest = app_data.join("session.json");
    let feedback_dest = crate::feedback::feedback_path(&app)?;

    let app_clone = app.clone();
    let source_root_for_closure = source_root.clone();
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<(usize, u64), String> {
        let source_root = source_root_for_closure;
        let models_dir = source_root.join("models");
        let voice_dir = source_root.join("voice-models");
        let conv_dir = source_root.join("conversations");

        if !models_dir.exists() && !voice_dir.exists() && !conv_dir.exists() {
            return Err(format!(
                "'{}' does not look like a Xenon2 export bundle -- expected a 'models', \
                 'voice-models', or 'conversations' subfolder (see EXPORT_FORMAT.md).",
                source_root.display()
            ));
        }

        let mut plan: Vec<PlanEntry> = Vec::new();

        if models_dir.join("rwkv-7-world-2.9B-Q5_1.bin").exists() {
            plan.push(PlanEntry {
                src: models_dir.join("rwkv-7-world-2.9B-Q5_1.bin"),
                rel: "models/rwkv-7-world-2.9B-Q5_1.bin".to_string(),
            });
        }
        if models_dir.join("world_vocab.bin").exists() {
            plan.push(PlanEntry {
                src: models_dir.join("world_vocab.bin"),
                rel: "inference-engine-data/world_vocab.bin".to_string(),
            });
        }
        if voice_dir.join("en_GB-alan-medium.onnx").exists() {
            plan.push(PlanEntry {
                src: voice_dir.join("en_GB-alan-medium.onnx"),
                rel: "voice-pipeline-models/en_GB-alan-medium.onnx".to_string(),
            });
        }
        if voice_dir.join("en_GB-alan-medium.onnx.json").exists() {
            plan.push(PlanEntry {
                src: voice_dir.join("en_GB-alan-medium.onnx.json"),
                rel: "voice-pipeline-models/en_GB-alan-medium.onnx.json".to_string(),
            });
        }

        if conv_dir.exists() {
            let mut names: Vec<PathBuf> = fs::read_dir(&conv_dir)
                .map_err(|e| format!("Could not read '{}': {e}", conv_dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().map(|ext| ext == "json").unwrap_or(false)
                        && p.file_name().map(|n| n != "session.json").unwrap_or(false)
                })
                .collect();
            names.sort();
            for path in names {
                let file_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                plan.push(PlanEntry {
                    src: path,
                    rel: format!("conversations/{file_name}"),
                });
            }
        }

        let total_files = plan.len();
        let mut total_bytes: u64 = 0;

        for (i, entry) in plan.iter().enumerate() {
            let dst = match entry.rel.split_once('/') {
                Some(("models", rest)) => repo_root.join("models").join(rest),
                Some(("inference-engine-data", rest)) => {
                    repo_root.join("inference-engine").join("data").join(rest)
                }
                Some(("voice-pipeline-models", rest)) => {
                    repo_root.join("voice-pipeline").join("models").join(rest)
                }
                Some(("conversations", rest)) => conversations_dir.join(rest),
                _ => continue,
            };
            let copied = copy_with_progress(
                &app_clone,
                "import-progress",
                &entry.src,
                &dst,
                &entry.rel,
                i,
                total_files,
            )?;
            total_bytes += copied;
        }

        // Ratings are merged, not copied over, so ratings made on this machine are kept.
        let bundle_feedback = source_root.join("feedback").join("feedback.jsonl");
        if bundle_feedback.exists() {
            crate::feedback::merge_into(&feedback_dest, &bundle_feedback)?;
        }

        // session.json is imported last, overwriting the local one -- this is the "restore memory
        // from a bundle" scenario, so the bundle's record of conversation paths (rewritten below
        // to point at this machine's conversations dir) should win.
        let bundle_session = conv_dir.join("session.json");
        if bundle_session.exists() {
            if let Ok(raw) = fs::read_to_string(&bundle_session) {
                if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&raw) {
                    // Rewrite conversationPaths to point at this machine's conversations dir,
                    // keyed by the same conversation ids -- the bundle's original paths are
                    // whatever the *source* machine used and are meaningless here.
                    if let Some(obj) = value.as_object_mut() {
                        if let Some(paths) = obj.get("conversationPaths").and_then(|v| v.as_object()) {
                            let mut rewritten = serde_json::Map::new();
                            for id in paths.keys() {
                                let p = conversations_dir.join(format!("{id}.json"));
                                rewritten.insert(
                                    id.clone(),
                                    serde_json::Value::String(p.to_string_lossy().into_owned()),
                                );
                            }
                            obj.insert("conversationPaths".to_string(), serde_json::Value::Object(rewritten));
                        }
                    }
                    if let Ok(pretty) = serde_json::to_string_pretty(&value) {
                        let _ = fs::write(&session_dest, pretty);
                    }
                }
            }
        }

        Ok((total_files, total_bytes))
    })
    .await
    .map_err(|e| format!("Import task failed: {e}"))?;

    match result {
        Ok((files_copied, total_bytes)) => {
            let _ = app.emit(
                "import-done",
                MemoryDoneEvent {
                    destination: source_root.to_string_lossy().into_owned(),
                    files_copied,
                    total_bytes,
                },
            );
            Ok(())
        }
        Err(msg) => {
            let _ = app.emit("import-error", MemoryErrorEvent { message: msg.clone() });
            Err(msg)
        }
    }
}

/// Accepts either the drive/folder that directly contains `models/`/`voice-models/`/
/// `conversations/`, or its parent if the user picked the drive root and the bundle lives one
/// level down at `xenon2-backup/` (the default folder name `export_memory` creates).
fn resolve_bundle_root(picked: &Path) -> PathBuf {
    let has_expected_children = picked.join("models").exists()
        || picked.join("voice-models").exists()
        || picked.join("conversations").exists();
    if has_expected_children {
        return picked.to_path_buf();
    }
    let nested = picked.join("xenon2-backup");
    if nested.exists() {
        return nested;
    }
    picked.to_path_buf()
}
