//! Phase 5 -- on-disk conversation persistence (Save / Save As / Open / auto-save).
//!
//! This module owns the on-disk JSON shape (`ConversationFile`, matching `SCHEMA.md` at the repo
//! root exactly) plus the small `session.json` file the app uses to remember which conversation
//! was last active and where each known conversation is saved, so a relaunch can restore state
//! without the user manually clicking Open. See `../../../SCHEMA.md` for the authoritative,
//! human-readable schema description; the structs below are its Rust mirror.
//!
//! File I/O here is plain `std::fs` -- no `tauri-plugin-fs` needed, since these are trusted
//! backend commands with paths either chosen by the user via a native dialog
//! (`tauri-plugin-dialog`) or computed by the app itself (the app-data default path), not
//! arbitrary paths handed in from untrusted web content.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// Current on-disk schema version. Bump this and add a migration path (not just a version bump)
/// if the shape ever changes incompatibly -- see SCHEMA.md's "Versioning" section.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FileRole {
    User,
    Assistant,
}

/// One message as written to / read from disk. Deliberately narrower than the frontend's
/// `ChatMessage`: `streaming` and `errored` are transient UI-only state that never makes sense at
/// rest (auto-save only fires after a generation *completes*, so a persisted message is never
/// mid-stream; an errored placeholder is never written either -- see chat.ts's `autoSave`, which
/// only runs from `completeGeneration`, not `failGeneration`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFileMessage {
    pub id: String,
    pub role: FileRole,
    pub content: String,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited: Option<bool>,
    /// Which agent generated this message ("dementia" | "sloth") -- see sloth_memory.rs. Absent
    /// on user messages and on any message saved before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// "good" or "bad" when the user rated this reply. See feedback.rs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rating: Option<String>,
    /// What the user said Xenon should have replied instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction: Option<String>,
}

/// The full on-disk shape of one saved conversation. Mirrors `SCHEMA.md` field-for-field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationFile {
    pub schema_version: u32,
    pub id: String,
    pub title: String,
    /// Name/version of the model that generated this conversation's replies, e.g.
    /// `"rwkv-7-world-2.9B-Q5_1"`. Recorded faithfully; compared against the currently loaded
    /// model on open since Phase 7 (frontend-side, see `stores/chat.ts`'s `modelMismatchWarning`)
    /// -- a mismatch surfaces a dismissable warning rather than being silently ignored.
    pub model: String,
    pub created_at: i64,
    pub saved_at: i64,
    pub messages: Vec<ConversationFileMessage>,
}

/// Small app-level bookkeeping file (NOT part of the conversation schema) that remembers, across
/// restarts, which conversation was last active and where every known conversation currently
/// lives on disk. Lives at `<app_data_dir>/session.json`. This is what lets the app restore the
/// last-active conversation automatically on launch without the user clicking Open.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionFile {
    #[serde(default)]
    pub last_active_conversation_id: Option<String>,
    /// conversationId -> absolute path of its saved .json file (whether via Save/Save As or the
    /// auto-save default path).
    #[serde(default)]
    pub conversation_paths: HashMap<String, String>,
}

fn friendly_read_error(path: &str, e: std::io::Error) -> String {
    format!("Could not read '{}': {}", path, e)
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| format!("Could not resolve the app data directory: {e}"))
}

/// Where auto-save writes a conversation that has never been manually saved: an app-data
/// directory keyed by conversation id, e.g.
/// `<app_data_dir>/conversations/<conversation-id>.json`. See SCHEMA.md / app/README.md's
/// "Auto-save default path policy" section for the reasoning -- short version: never invent a
/// user-visible location silently, but also never block a normal send on a save dialog.
#[tauri::command]
pub fn default_conversation_path(app: AppHandle, conversation_id: String) -> Result<String, String> {
    // Phase 6: honor the "data directory location" setting if one is configured -- this is what
    // makes an external drive usable as ongoing primary storage, not just a one-time export
    // destination. Falls back to the pre-Phase-6 app-data default when unset (see
    // `crate::memory::effective_conversations_dir`).
    let dir = PathBuf::from(crate::memory::effective_conversations_dir(app)?);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("Could not create '{}': {}", dir.display(), e))?;
    let path = dir.join(format!("{conversation_id}.json"));
    Ok(path.to_string_lossy().into_owned())
}

/// Writes a conversation to an exact path (already known -- either user-chosen or the computed
/// default). Pretty-printed, not minified, so a saved file is human-readable in a plain text
/// editor per the phase's acceptance criteria.
#[tauri::command]
pub fn save_conversation_file(path: String, conversation: ConversationFile) -> Result<(), String> {
    let path_buf = PathBuf::from(&path);
    if let Some(parent) = path_buf.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create directory '{}': {}", parent.display(), e))?;
        }
    }

    let json = serde_json::to_string_pretty(&conversation)
        .map_err(|e| format!("Could not serialize conversation: {e}"))?;

    fs::write(&path_buf, json).map_err(|e| format!("Could not write '{}': {}", path, e))
}

/// Reads and validates a conversation file. Returns a clear, specific error (never a panic) for
/// anything that isn't a well-formed Xenon2 conversation -- unreadable file, invalid JSON, missing
/// fields, wrong types (e.g. a `role` that isn't "user"/"assistant"), or an unsupported schema
/// version.
#[tauri::command]
pub fn open_conversation_file(path: String) -> Result<ConversationFile, String> {
    let raw = fs::read_to_string(&path).map_err(|e| friendly_read_error(&path, e))?;

    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "'{}' is not valid JSON ({e}). This does not look like a Xenon2 conversation file.",
            path
        )
    })?;

    let file: ConversationFile = serde_json::from_value(value).map_err(|e| {
        format!(
            "'{}' is valid JSON but not a valid Xenon2 conversation ({e}). \
             Expected fields: schemaVersion, id, title, model, createdAt, savedAt, messages[].",
            path
        )
    })?;

    if file.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "'{}' uses schema version {}, but this build of Xenon2 only supports version {}.",
            path, file.schema_version, SCHEMA_VERSION
        ));
    }

    Ok(file)
}

/// How long to wait for the native file dialog before giving up. Windows' Common Item Dialog
/// (the modern Explorer-style picker `rfd` uses) enumerates the shell namespace -- "This PC",
/// mapped network drives, cloud-sync providers (OneDrive etc), "Quick access" -- while it opens,
/// and a slow or hung shell extension can stall that indefinitely. Verified independently on the
/// dev machine used for Phase 5 testing: a plain `System.Windows.Forms.SaveFileDialog` invoked
/// from an unrelated PowerShell process (nothing to do with Xenon2 or Tauri) hung the same way,
/// while a plain `MessageBox` on the same machine appeared instantly -- confirming this is a
/// machine-level Explorer/shell issue, not a bug in this dialog-invocation code. A bounded
/// timeout turns that failure mode from "the app looks permanently frozen" into a clear, dismissable
/// error.
const DIALOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Test-only escape hatch: if set, `pick_save_dialog`/`pick_open_dialog` return this path
/// immediately instead of invoking the real OS dialog. Exists because the native dialog cannot be
/// driven by UI automation the way the rest of this app can (see `PLAN.md`/phase5 verification
/// notes) -- it lets the *rest* of the Save As / Open pipeline (path bookkeeping, writing,
/// validation, auto-save switching to the new path) be exercised through the real running app and
/// real UI clicks, with only the OS picker step substituted. Unset (the default) in normal use;
/// never read unless explicitly exported before launch.
fn test_override_path(env_var: &str) -> Option<String> {
    std::env::var(env_var).ok().filter(|s| !s.is_empty())
}

/// Opens a native "Save As" dialog defaulting to `default_file_name`, filtered to `.json`.
/// Returns `Ok(None)` (not an error) if the user cancels, or if the dialog times out (see
/// `DIALOG_TIMEOUT`).
///
/// Calls `rfd` directly (not via `tauri-plugin-dialog`'s `DialogExt`): `tauri-plugin-dialog`
/// routes even its "blocking" API through `AppHandle::run_on_main_thread`, and in this app that
/// indirection never actually showed the dialog at all (confirmed via `EnumWindows` -- no dialog
/// window ever appeared, and the frontend's `invoke()` call was left permanently pending).
/// `rfd::FileDialog::save_file()` on Windows just calls the Common Item Dialog COM API directly on
/// the calling thread (see rfd's `win_cid/file_dialog.rs`) with no main-thread requirement, so this
/// bypasses that layer entirely.
#[tauri::command]
pub async fn pick_save_dialog(_app: AppHandle, default_file_name: String) -> Result<Option<String>, String> {
    if let Some(path) = test_override_path("XENON2_TEST_SAVE_DIALOG_PATH") {
        return Ok(Some(path));
    }

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let picked = rfd::FileDialog::new()
            .set_file_name(&default_file_name)
            .add_filter("Xenon2 Conversation", &["json"])
            .save_file();
        let _ = tx.send(picked.map(|p| p.to_string_lossy().into_owned()));
    });

    tauri::async_runtime::spawn_blocking(move || rx.recv_timeout(DIALOG_TIMEOUT))
        .await
        .map_err(|e| format!("Save dialog task failed: {e}"))?
        .map_err(|_| {
            "The save dialog did not respond within 30 seconds. This usually means a Windows \
             shell extension (e.g. a cloud-sync client) is hanging while the dialog loads -- \
             try again, or check Windows for a misbehaving shell extension."
                .to_string()
        })
}

/// Opens a native "Open" dialog filtered to `.json`. Returns `Ok(None)` if the user cancels or the
/// dialog times out. See `pick_save_dialog`'s doc comment for why this calls `rfd` directly and
/// what the timeout is for.
#[tauri::command]
pub async fn pick_open_dialog(_app: AppHandle) -> Result<Option<String>, String> {
    if let Some(path) = test_override_path("XENON2_TEST_OPEN_DIALOG_PATH") {
        return Ok(Some(path));
    }

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let picked = rfd::FileDialog::new()
            .add_filter("Xenon2 Conversation", &["json"])
            .pick_file();
        let _ = tx.send(picked.map(|p| p.to_string_lossy().into_owned()));
    });

    tauri::async_runtime::spawn_blocking(move || rx.recv_timeout(DIALOG_TIMEOUT))
        .await
        .map_err(|e| format!("Open dialog task failed: {e}"))?
        .map_err(|_| {
            "The open dialog did not respond within 30 seconds. This usually means a Windows \
             shell extension (e.g. a cloud-sync client) is hanging while the dialog loads -- \
             try again, or check Windows for a misbehaving shell extension."
                .to_string()
        })
}

/// Loads `session.json` from the app-data directory. A missing file is normal (first ever launch)
/// and returns an empty default session, not an error.
#[tauri::command]
pub fn load_session_file(app: AppHandle) -> Result<SessionFile, String> {
    let path = app_data_dir(&app)?.join("session.json");
    if !path.exists() {
        return Ok(SessionFile::default());
    }
    let raw = fs::read_to_string(&path).map_err(|e| friendly_read_error(&path.to_string_lossy(), e))?;
    serde_json::from_str(&raw).map_err(|e| format!("Could not parse session file: {e}"))
}

/// Writes `session.json` to the app-data directory.
#[tauri::command]
pub fn save_session_file(app: AppHandle, session: SessionFile) -> Result<(), String> {
    let dir = app_data_dir(&app)?;
    fs::create_dir_all(&dir).map_err(|e| format!("Could not create '{}': {}", dir.display(), e))?;
    let path = dir.join("session.json");
    let json = serde_json::to_string_pretty(&session)
        .map_err(|e| format!("Could not serialize session: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Could not write '{}': {}", path.display(), e))
}
