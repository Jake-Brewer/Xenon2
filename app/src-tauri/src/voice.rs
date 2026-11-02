//! Phase 7 -- wires the desktop app's mic button to Phase 2's standalone voice pipeline
//! (`voice-pipeline/`: silero-vad -> faster-whisper STT -> ... -> piper TTS).
//!
//! Chosen integration: run `voice-pipeline/ipc_server.py` as a long-lived sidecar process,
//! communicating over newline-delimited JSON on stdin/stdout (see that file's module doc for
//! the protocol). The alternative considered -- porting VAD/STT/TTS to Rust bindings -- would
//! mean re-verifying faster-whisper/piper/silero-vad's GPU/CPU device placement and CUDA-DLL
//! workarounds (see voice-pipeline/README.md) all over again in a different binding layer, for
//! libraries that only ship first-class Python packages. The sidecar reuses Phase 2's exact,
//! already-verified code (including `IncrementalSpeaker`'s sentence-chunked streaming TTS)
//! unchanged.
//!
//! Deliberately does NOT call `xenon_generate()` from Python -- `inference.rs` owns the one
//! loaded RWKV engine, and per phase7_prompt.md task 1 ("do not build a parallel/separate code
//! path for voice-originated messages"), a voice transcript is fed into the frontend's existing
//! `sendMessage` -> `generate_response` path exactly like typed text. This module only bridges
//! the two edges: mic audio -> transcript (`voice_listen`), and streamed reply text -> speech
//! (`voice_speak_start`/`voice_speak_feed`/`voice_speak_finish`, fed fragment-by-fragment from
//! the frontend's `token-stream` handler -- see App.vue).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// How long a voice-pipeline command may take before giving up and surfacing an error. Listen
/// has its own, larger, caller-supplied budget (see `voice_listen`); this is used for the quick
/// speak_start/speak_feed/speak_finish acks.
const QUICK_TIMEOUT: Duration = Duration::from_secs(10);

/// Guards stdin so each command's ack is matched to that command. `listen` acks immediately
/// and delivers `listen_result` later on `listen_results`, so a capture can overlap TTS.
pub struct VoiceProcess {
    /// `None` if the sidecar failed to spawn at all (e.g. the venv/ipc_server.py aren't present
    /// in this checkout) -- commands then fail fast with `spawn_error` instead of hanging.
    stdin: Mutex<Option<std::process::ChildStdin>>,
    responses: Mutex<Option<Receiver<serde_json::Value>>>,
    /// Async `listen_result` events, demuxed off the ack channel so `speak_feed` cannot consume them.
    listen_results: Mutex<Option<Receiver<serde_json::Value>>>,
    listen_seq: AtomicU64,
    ready: Arc<AtomicBool>,
    spawn_error: Option<String>,
    request_lock: Mutex<()>,
    _child: Mutex<Option<Child>>,
}

#[derive(Clone)]
pub struct VoiceState(pub Arc<VoiceProcess>);

#[derive(Debug, Clone, Serialize)]
pub struct VoiceListenResult {
    pub ok: bool,
    pub text: Option<String>,
    pub reason: Option<String>,
}

fn venv_python(voice_pipeline_dir: &PathBuf) -> PathBuf {
    let win = voice_pipeline_dir.join(".venv").join("Scripts").join("python.exe");
    if win.exists() {
        return win;
    }
    voice_pipeline_dir.join(".venv").join("bin").join("python3")
}

/// Spawns `ipc_server.py` and starts a background reader thread that demultiplexes its stdout:
/// `speak_done` becomes a `voice-speak-done` Tauri event, `listen_result` goes to its own
/// channel, and acks go to `responses`. Never panics -- a spawn failure is recorded in
/// `spawn_error` so the app still starts without voice support.
pub fn spawn_voice_process(repo_root: &PathBuf, app: AppHandle) -> Arc<VoiceProcess> {
    let voice_pipeline_dir = repo_root.join("voice-pipeline");
    let python = venv_python(&voice_pipeline_dir);
    let script = voice_pipeline_dir.join("ipc_server.py");

    let spawn_result = Command::new(&python)
        .arg(&script)
        .current_dir(&voice_pipeline_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // sidecar's own diagnostics -- visible in the dev console, never mixed into the stdout JSON protocol
        .spawn();

    let mut child = match spawn_result {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(
                "Could not start voice pipeline sidecar ('{}' '{}'): {}. Voice input/output is \
                 unavailable this session -- typed chat still works normally.",
                python.display(),
                script.display(),
                e
            );
            eprintln!("[voice] {msg}");
            return Arc::new(VoiceProcess {
                stdin: Mutex::new(None),
                responses: Mutex::new(None),
                listen_results: Mutex::new(None),
                listen_seq: AtomicU64::new(0),
                ready: Arc::new(AtomicBool::new(false)),
                spawn_error: Some(msg),
                request_lock: Mutex::new(()),
                _child: Mutex::new(None),
            });
        }
    };

    let stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

    let (tx, rx) = mpsc::channel::<serde_json::Value>();
    let (listen_tx, listen_rx) = mpsc::channel::<serde_json::Value>();
    let ready = Arc::new(AtomicBool::new(false));
    let ready_writer = ready.clone();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break, // pipe closed -- sidecar exited
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[voice] sidecar emitted non-JSON line, ignoring: {trimmed} ({e})");
                    continue;
                }
            };

            match value.get("type").and_then(|t| t.as_str()) {
                Some("ready") => {
                    ready_writer.store(true, Ordering::SeqCst);
                }
                Some("speak_done") => {
                    let _ = app.emit("voice-speak-done", ());
                }
                Some("listen_result") => {
                    let _ = listen_tx.send(value);
                }
                Some("speech_started") => {
                    let _ = app.emit("voice-user-speaking", ());
                }
                _ => {
                    let _ = tx.send(value);
                }
            }
        }
        ready_writer.store(false, Ordering::SeqCst);
    });

    Arc::new(VoiceProcess {
        stdin: Mutex::new(Some(stdin)),
        responses: Mutex::new(Some(rx)),
        listen_results: Mutex::new(Some(listen_rx)),
        listen_seq: AtomicU64::new(0),
        ready,
        spawn_error: None,
        request_lock: Mutex::new(()),
        _child: Mutex::new(Some(child)),
    })
}

/// Writes one JSON command and waits for its ack. `speak_done` and `listen_result` are not
/// delivered here. The lock is held only until that ack, so the next command can run while a
/// listen capture or TTS playback is still in progress.
fn send_command(
    process: &VoiceProcess,
    cmd: serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    if let Some(err) = &process.spawn_error {
        return Err(err.clone());
    }
    let _guard = process.request_lock.lock().map_err(|_| "voice process lock poisoned".to_string())?;

    {
        let mut stdin_guard = process.stdin.lock().map_err(|_| "voice stdin lock poisoned".to_string())?;
        let stdin = stdin_guard.as_mut().ok_or("voice sidecar is not running")?;
        let mut line = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
        line.push('\n');
        stdin.write_all(line.as_bytes()).map_err(|e| format!("could not write to voice sidecar: {e}"))?;
        stdin.flush().map_err(|e| format!("could not flush voice sidecar stdin: {e}"))?;
    }

    let responses_guard = process.responses.lock().map_err(|_| "voice responses lock poisoned".to_string())?;
    let rx = responses_guard.as_ref().ok_or("voice sidecar is not running")?;
    let value = rx
        .recv_timeout(timeout)
        .map_err(|_| "voice sidecar did not respond in time".to_string())?;

    if value.get("type").and_then(|t| t.as_str()) == Some("error") {
        let msg = value
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown voice pipeline error")
            .to_string();
        return Err(msg);
    }

    Ok(value)
}

#[tauri::command]
pub fn voice_ready(voice: tauri::State<'_, VoiceState>) -> bool {
    voice.0.spawn_error.is_none() && voice.0.ready.load(Ordering::SeqCst)
}

fn drain_listen_results(process: &VoiceProcess) {
    let Ok(guard) = process.listen_results.lock() else {
        return;
    };
    let Some(rx) = guard.as_ref() else {
        return;
    };
    while rx.try_recv().is_ok() {}
}

fn recv_listen_result(
    process: &VoiceProcess,
    listen_id: u64,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("voice sidecar did not return a transcript in time".to_string());
        }
        let guard = process
            .listen_results
            .lock()
            .map_err(|_| "voice listen lock poisoned".to_string())?;
        let rx = guard.as_ref().ok_or("voice sidecar is not running")?;
        match rx.recv_timeout(remaining) {
            Ok(value) => {
                if value.get("id").and_then(|v| v.as_u64()) == Some(listen_id) {
                    return Ok(value);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err("voice sidecar did not return a transcript in time".to_string());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("voice sidecar stopped during listen".to_string());
            }
        }
    }
}

fn listen_result_from_value(value: serde_json::Value) -> VoiceListenResult {
    VoiceListenResult {
        ok: value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
        text: value.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()),
        reason: value.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string()),
    }
}

/// Ends an open listen early, so ending a conversation releases the mic straight away.
#[tauri::command]
pub async fn voice_listen_cancel(voice: tauri::State<'_, VoiceState>) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "listen_cancel" }), QUICK_TIMEOUT).map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Records from the mic, VAD-gated, and transcribes the utterance. The sidecar acks at once and
/// runs capture on a worker, so TTS can keep playing. This waits for that worker's result.
/// Whisper's no-speech filter stays off in the sidecar; Silero already gated the clip.
/// The headset mic is opened on a worker thread, which must initialize COM first.
#[tauri::command]
pub async fn voice_listen(
    voice: tauri::State<'_, VoiceState>,
    max_listen_sec: Option<f64>,
    max_utterance_sec: Option<f64>,
) -> Result<VoiceListenResult, String> {
    let process = voice.0.clone();
    let max_listen = max_listen_sec.unwrap_or(6.0);
    let max_utterance = max_utterance_sec.unwrap_or(10.0);
    // Whisper shares the GPU with RWKV, so leave extra room past the capture window.
    let result_timeout = Duration::from_secs_f64(max_listen + max_utterance + 30.0);

    tauri::async_runtime::spawn_blocking(move || {
        let listen_id = process.listen_seq.fetch_add(1, Ordering::SeqCst);
        drain_listen_results(&process);
        let cmd = serde_json::json!({
            "cmd": "listen",
            "id": listen_id,
            "maxListenSec": max_listen,
            "maxUtteranceSec": max_utterance,
        });
        send_command(&process, cmd, QUICK_TIMEOUT)?;
        let value = recv_listen_result(&process, listen_id, result_timeout)?;
        Ok(listen_result_from_value(value))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Starts a new incremental-speech turn. Call once right before streaming an assistant reply
/// that should be spoken aloud (i.e. one that was triggered by a voice `voice_listen` transcript
/// -- see App.vue's `speakReplies` flag), before any `voice_speak_feed` calls for that turn.
#[tauri::command]
pub async fn voice_speak_start(voice: tauri::State<'_, VoiceState>) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_start" }), QUICK_TIMEOUT).map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Feeds one streamed text fragment (the same fragment forwarded by a `token-stream` event) into
/// the sidecar's `IncrementalSpeaker`, which buffers it and synthesizes+plays each completed
/// sentence as soon as it's available -- see `speech_streamer.py`.
#[tauri::command]
pub async fn voice_speak_feed(voice: tauri::State<'_, VoiceState>, text: String) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_feed", "text": text }), QUICK_TIMEOUT)
            .map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Flushes any remaining buffered text as a final sentence. Returns as soon as the sidecar
/// acknowledges the command (not once playback finishes) -- listen for the `voice-speak-done`
/// Tauri event if the UI needs to know when speech audio has actually finished playing.
#[tauri::command]
pub async fn voice_speak_finish(voice: tauri::State<'_, VoiceState>) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_finish" }), QUICK_TIMEOUT).map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}
