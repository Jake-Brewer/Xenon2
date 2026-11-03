//! Safe(r) wrapper around the Phase 1 `xenon_inference` C API, plus the Tauri IPC command that
//! drives token-by-token streaming into the chat UI.
//!
//! This is the ONLY place in the app that talks to Phase 1's compiled library. It does not
//! reimplement or stub inference -- `generate_reply` calls the real `xenon_generate()` from
//! `inference-engine`, loaded once at app startup via `xenon_load_model()`.

use std::cell::RefCell;
use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::ffi;
use crate::sloth_memory;

/// One turn of conversation, as sent from the Vue frontend. Mirrors the shape of a chat message
/// but deliberately doesn't carry UI-only fields (ids, timestamps, streaming flags, etc) -- this
/// is a minimal payload just for prompt construction.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String, // "user" | "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
struct TokenEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationDoneEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationErrorEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
    message: String,
}

/// Raw engine pointer. `xenon_engine*` is safe to hand off across threads as long as access is
/// serialized, which the Mutex around `EngineInner` guarantees.
pub(crate) struct EnginePtr(*mut ffi::XenonEngine);
unsafe impl Send for EnginePtr {}

/// Raw incremental-state pointer (Phase 8, see `xenon_inference.h`'s `xenon_state` docs). Frees
/// itself via `xenon_state_free` on drop so every place that allocates one (a cache entry, a
/// per-turn scratch copy) can just let it go out of scope instead of remembering to free it.
pub(crate) struct XenonStatePtr(*mut ffi::XenonState);
unsafe impl Send for XenonStatePtr {}

impl Drop for XenonStatePtr {
    fn drop(&mut self) {
        unsafe { ffi::xenon_state_free(self.0) };
    }
}

/// One conversation's cached incremental state, plus the exact turns it has already consumed --
/// used to diff against an incoming `history` array and decide "extend" vs. "rebuild" (see
/// `ensure_cache_extended`). `consumed` never includes the newest, not-yet-answered user turn,
/// and is only ever advanced from caller-confirmed history text, never from this turn's own
/// generation output -- see `generate_blocking`'s doc comment for why that matters.
struct CachedConversation {
    consumed: Vec<ChatTurn>,
    state: XenonStatePtr,
}

/// Bound on how many conversations' incremental state is kept alive at once. Each entry is one
/// `xenon_get_state_len()`-sized buffer (measured at ~20.6 MB for the RWKV-7 2.9B model -- see
/// `inference-engine/README.md`), so this caps the feature's steady-state memory cost at
/// roughly 4 x that, plus the two engine-level static states -- trivial next to the multi-GB
/// model weights already resident. Least-recently-used entries are evicted first (see
/// `ensure_cache_extended`).
const MAX_CACHED_CONVERSATIONS: usize = 4;

/// Everything needed to run generation, guarded by one mutex so only one generation (or cache
/// mutation) runs at a time -- rwkv.cpp's per-engine state is not designed for concurrent calls,
/// and Phase 8's incremental states are per-conversation data guarded alongside it, not a
/// separate lock (see this struct's fields).
pub(crate) struct EngineInner {
    engine: EnginePtr,
    /// The conversation preamble (instruction + demo turns) -- byte-identical on every call ever
    /// made, so it's prefilled once here instead of every turn. See `inference.rs`'s
    /// `STATIC_PREFIX` and `build_prompt`'s Phase 8 doc comment for why this text can be first
    /// and still cacheable (unlike the volatile header, which can't).
    static_prefix: XenonStatePtr,
    /// The Sloth fact-extraction few-shot preamble, prefilled once the same way. Entirely
    /// separate from `conversation_cache` below and never touched by conversation generation --
    /// extraction must never be able to corrupt a conversation's cached state (see
    /// `run_fact_extraction`).
    extraction_prefix: XenonStatePtr,
    /// LRU cache of per-conversation incremental state, keyed by conversation id. A `Vec` (not a
    /// `HashMap`) because eviction order is exactly insertion/access order and there are at most
    /// `MAX_CACHED_CONVERSATIONS` entries -- a linear scan over a handful of entries is simpler
    /// and just as fast as a HashMap + separate LRU list here.
    conversation_cache: Vec<(String, CachedConversation)>,
}

/// `Arc` (not a bare `Mutex`) because the IPC command below needs to clone a handle to it into a
/// `spawn_blocking` closure and hold the lock for the *entire* generate() call, not just long
/// enough to read the pointer out -- rwkv.cpp's per-engine state isn't safe for concurrent eval
/// calls, so the lock must stay held across the whole blocking generation, not be dropped early.
#[derive(Clone)]
pub struct EngineState(pub Arc<Mutex<EngineInner>>);

/// The loaded model's name/version (its filename, minus extension, e.g.
/// `"rwkv-5-world-0.4B-Q4_0"`), recorded so Phase 5's saved conversation files can note which
/// model generated them (see `persistence::ConversationFile::model`). Computed once from the
/// actual path `load_engine` loads, so it can never drift from what's really running.
pub struct ModelInfo(pub String);

/// Short label for the menu bar. "CUDA — all layers on GPU" when offload succeeded.
pub struct DeviceInfo(pub String);

#[tauri::command]
pub fn get_model_name(model_info: tauri::State<'_, ModelInfo>) -> String {
    model_info.0.clone()
}

#[tauri::command]
pub fn get_inference_device(device_info: tauri::State<'_, DeviceInfo>) -> String {
    device_info.0.clone()
}

/// One ggml worker per physical core. Hyperthreads share a core's execution units, and this
/// model gets slower when two workers are pinned to the same core. Dementia and Sloth both use
/// the single engine loaded below, so they both get this count.
fn inference_threads() -> u32 {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1);
    let physical = physical_cpus().unwrap_or(logical);
    physical.clamp(1, logical)
}

fn logical_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1)
}

#[cfg(windows)]
fn physical_cpus() -> Option<u32> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformation(buffer: *mut u8, returned_length: *mut u32) -> i32;
    }
    let mut bytes = 0u32;
    unsafe {
        GetLogicalProcessorInformation(std::ptr::null_mut(), &mut bytes);
    }
    if bytes == 0 {
        return None;
    }
    // The struct is 8-byte aligned. A byte Vec is not, and a misaligned buffer
    // makes the call fail, which would fall back to one worker per hyperthread.
    let words = (bytes as usize).div_ceil(8);
    let mut buf = vec![0u64; words];
    let ok = unsafe {
        GetLogicalProcessorInformation(buf.as_mut_ptr() as *mut u8, &mut bytes)
    };
    if ok == 0 {
        return None;
    }
    let reported = bytes as usize;
    let raw = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, reported.min(buf.len() * 8)) };
    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION on x64 is 32 bytes. Relationship is at offset 8.
    // RelationProcessorCore == 0.
    const STRIDE: usize = 32;
    let mut cores = 0u32;
    let mut off = 0;
    while off + STRIDE <= raw.len() {
        let relationship = u32::from_ne_bytes(raw[off + 8..off + 12].try_into().ok()?);
        if relationship == 0 {
            cores += 1;
        }
        off += STRIDE;
    }
    if cores == 0 {
        None
    } else {
        Some(cores)
    }
}

#[cfg(not(windows))]
fn physical_cpus() -> Option<u32> {
    // "0,6" or "0-1" lists cpu0's hyperthread siblings. One entry means no hyperthreading.
    let raw = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list").ok()?;
    let mut siblings = 0u32;
    for part in raw.trim().split(',').filter(|p| !p.is_empty()) {
        siblings += match part.split_once('-') {
            Some((start, end)) => end.trim().parse::<u32>().ok()? - start.trim().parse::<u32>().ok()? + 1,
            None => 1,
        };
    }
    Some((logical_cpus() / siblings.max(1)).max(1))
}

/// Free VRAM in MiB, when `nvidia-smi` can answer. Used to decide how many RWKV layers fit
/// on the GPU without pushing Whisper off the card.
fn gpu_memory_free_mib() -> Option<u32> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().next()?.trim().parse().ok()
}

/// Q5_1 weights are about 2.6 GB. Full offload needs that plus activations, with room left
/// for Whisper on the same GPU. A tighter card keeps some blocks on the CPU instead of
/// dropping all the way to a CPU-only load.
fn preferred_gpu_layers(free_mib: Option<u32>) -> u32 {
    match free_mib {
        Some(free) if free < 1200 => 0,
        Some(free) if free < 3400 => 16,
        _ => 999,
    }
}

/// Loads the Phase 1 model once at app startup. Panics (fails app startup) rather than silently
/// falling back to a stub -- per the phase spec, this app must call the real inference engine,
/// not reimplement or fake it. Also returns a `ModelInfo` derived from the same path, so the name
/// recorded in saved conversation files (Phase 5) can never drift from what's actually loaded.
pub fn load_engine(repo_root: &PathBuf) -> (EngineState, ModelInfo, DeviceInfo) {
    // Phase 7 follow-up (model upgrade): swapped from RWKV-5 World 0.4B to RWKV-7 World 2.9B v3
    // -- the 0.4B base model was found to collapse onto generic canned replies for short/
    // greeting-style prompts and got basic arithmetic wrong (see conversation history around
    // 2026-08-01). RWKV-7 ("Goose") is a materially more capable architecture per-parameter --
    // the paper reports RWKV7-World3-2.9B nearly matching Qwen2.5-3B on English benchmarks
    // despite far less training data. Same World tokenizer, same rwkv.cpp conversion pipeline,
    // GPU offload is preferred when this binary was built with CUDA (see build.rs, which links
    // build-cuda-app ahead of build-cpu-app). An RTX 2060 has 6GB; this Q5_1 file is ~2.75GB
    // and the RWKV state is ~21MB, so every layer fits. If CUDA init fails, load retries on CPU
    // so a CUDA build still starts on a machine with no NVIDIA GPU.
    let model_path = repo_root.join("models").join("rwkv-7-world-2.9B-Q5_1.bin");
    let vocab_path = repo_root
        .join("inference-engine")
        .join("data")
        .join("world_vocab.bin");

    let model_name = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown-model".to_string());

    let model_path_c = CString::new(model_path.to_string_lossy().as_bytes())
        .expect("model path contains NUL byte");
    let vocab_path_c = CString::new(vocab_path.to_string_lossy().as_bytes())
        .expect("vocab path contains NUL byte");

    // Dementia and Sloth share this one engine. Thread count follows the physical cores.
    // rwkv.cpp clamps n_gpu_layers to n_layer+1, and that clamped value offloads every block
    // plus the output head. A CPU-only library ignores the layer count.
    let threads = inference_threads();
    let gpu_compiled = unsafe { ffi::xenon_has_gpu_support() } != 0;
    let free_mib = gpu_memory_free_mib();
    let mut layer_plan = Vec::new();
    if gpu_compiled {
        let preferred = preferred_gpu_layers(free_mib);
        layer_plan.push(preferred);
        if preferred > 16 {
            layer_plan.push(16);
        }
        if preferred != 0 {
            layer_plan.push(0);
        }
    } else {
        layer_plan.push(0);
    }

    let mut engine = std::ptr::null_mut();
    let mut used_layers = 0u32;
    for (attempt, &layers) in layer_plan.iter().enumerate() {
        if attempt > 0 {
            eprintln!(
                "xenon2: model load with the previous GPU plan failed ({}). Retrying with {layers} GPU layers.",
                last_error()
            );
        }
        engine = unsafe {
            ffi::xenon_load_model(
                model_path_c.as_ptr(),
                vocab_path_c.as_ptr(),
                threads,
                layers,
            )
        };
        if !engine.is_null() {
            used_layers = layers;
            break;
        }
    }

    if engine.is_null() {
        let err = last_error();
        panic!(
            "Failed to load Phase 1 model from '{}' (vocab '{}'): {}",
            model_path.display(),
            vocab_path.display(),
            err
        );
    }

    let n_layer = unsafe { ffi::xenon_get_n_layer(engine) } as u32;
    let device_label = if used_layers == 0 {
        format!("CPU · {threads} threads")
    } else if used_layers >= n_layer.saturating_add(1) {
        format!("CUDA — all layers on GPU · {threads} threads")
    } else {
        format!("CUDA — {used_layers} layers on GPU, rest on CPU · {threads} threads")
    };
    eprintln!(
        "xenon2: inference device: {device_label} (logical cpus {}, free vram {})",
        logical_cpus(),
        free_mib
            .map(|m| format!("{m} MiB"))
            .unwrap_or_else(|| "unknown".to_string())
    );

    // Phase 8: prefill the two engine-level static preambles once, up front, instead of
    // re-evaluating them on every single generate/extraction call -- see `EngineInner`'s doc
    // comment and `STATIC_PREFIX` / `build_extraction_prompt`'s static half below. A failure
    // here means the tokenizer/engine can't even process plain text, which would fail every real
    // generation anyway, so panicking (matching this function's existing "fail app startup
    // loudly" policy) rather than falling back to an unprimed state is intentional.
    let static_prefix = unsafe {
        let state = ffi::xenon_state_new(engine);
        ffi::xenon_state_reset(engine, state);
        let text = CString::new(STATIC_PREFIX).expect("STATIC_PREFIX contains no NUL bytes");
        let status = ffi::xenon_prefill(engine, state, text.as_ptr());
        if status != ffi::XenonStatus::OK {
            panic!("Failed to prefill Phase 8 static prefix: {}", last_error());
        }
        XenonStatePtr(state)
    };

    let extraction_prefix = unsafe {
        let state = ffi::xenon_state_new(engine);
        ffi::xenon_state_reset(engine, state);
        let text = CString::new(EXTRACTION_STATIC_PREFIX)
            .expect("EXTRACTION_STATIC_PREFIX contains no NUL bytes");
        let status = ffi::xenon_prefill(engine, state, text.as_ptr());
        if status != ffi::XenonStatus::OK {
            panic!("Failed to prefill Phase 8 extraction prefix: {}", last_error());
        }
        XenonStatePtr(state)
    };

    (
        EngineState(Arc::new(Mutex::new(EngineInner {
            engine: EnginePtr(engine),
            static_prefix,
            extraction_prefix,
            conversation_cache: Vec::new(),
        }))),
        ModelInfo(model_name),
        DeviceInfo(device_label),
    )
}

fn last_error() -> String {
    unsafe {
        let ptr = ffi::xenon_get_last_error();
        if ptr.is_null() {
            "(no error message)".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// Phase 8 follow-up: this used to be one `build_prompt()` function producing one string in the
// order [volatile date/time header] -> [static few-shot] -> [history] -> "Xenon:". That ordering
// made incremental state caching *impossible*: RWKV state is strictly sequential, so a value that
// changes every call (the header) sitting at position 0 invalidates everything fed after it on
// every single call, no matter how it's cached. Reordered to [static prefix] -> [history] ->
// [volatile header] -> [new turn] -> "Xenon:" -- same wording, same facts injection, same time
// example, just moved so the two truly-cacheable pieces (the static prefix, prefilled once ever
// at engine load; the growing history, prefilled incrementally per conversation) both come before
// the one piece that has to be re-fed every turn regardless (the header). See `EngineInner`'s doc
// comment and `ensure_cache_extended` for how the three pieces below are actually assembled and
// cached. Moving the header to sit immediately before the new turn is plausibly *better* for
// grounding (closer to the generation point = more salient to an autoregressive model) rather
// than worse, and was sanity-checked against the old ordering on real prompts (see Phase 8 commit
// notes) rather than assumed safe.

/// Byte-identical on every single call this app will ever make -- prefilled once into
/// `EngineInner::static_prefix` at `load_engine` time instead of being re-evaluated per turn.
/// Phase 7 follow-up note preserved: the seed example used to end with "How can I help you
/// today?" -- real usage showed the model imitating that closing-question *style* on nearly every
/// reply regardless of content, because a small base model leans heavily on the literal style of
/// whatever example it's shown, not just the instruction-like framing around it. These two
/// example turns both end declaratively, no trailing question, to actually change the imitated
/// style rather than just ask for it in prose (which this model has already shown it doesn't
/// reliably follow -- see sloth_memory's extraction prompt fix for the same lesson).
const STATIC_PREFIX: &str =
    "The following is a coherent, friendly conversation between a user and Xenon, a \
     helpful voice assistant. Xenon answers naturally and doesn't end every reply by asking \
     what else it can help with.\n\n\
     User: Hello Xenon, how are you doing?\n\n\
     Xenon: Hi! I'm doing well, thanks for asking.\n\n\
     User: What's a fun fact about space?\n\n\
     Xenon: A day on Venus is longer than its year -- it rotates so slowly that one spin \
     takes longer than one full trip around the sun.\n\n";

/// Formats a slice of turns exactly as the old `build_prompt`'s history loop did: this is the
/// piece that gets prefilled incrementally into a conversation's cached state (see
/// `ensure_cache_extended`), so its exact text must stay stable across calls -- changing this
/// function's output for turns already in a cache would silently desync the cache from what a
/// fresh rebuild would produce.
fn history_text(turns: &[ChatTurn]) -> String {
    let mut out = String::new();
    for turn in turns {
        if turn.role == "user" {
            out.push_str("User: ");
        } else {
            out.push_str("Xenon: ");
        }
        out.push_str(turn.content.trim());
        out.push_str("\n\n");
    }
    out
}

/// The one part of the prompt that must be re-fed every single turn regardless of caching --
/// see this section's Phase 8 doc comment above for why it can't sit at the front any more.
/// Preserves both Phase 7 follow-up fixes verbatim, just relocated: real-clock grounding (the
/// model was found to confidently make up a specific time with no actual clock access at all)
/// and Sloth's persistent facts injection (skipped entirely for Dementia turns, which always
/// pass an empty `facts` slice, so Dementia's volatile header is byte-for-byte what it always
/// was pre-Phase-8, just at a different position in the overall prompt).
fn volatile_header_text(facts: &[sloth_memory::SlothFact]) -> String {
    let mut out = String::new();

    let now = chrono::Local::now();
    let now_full = now.format("%A, %B %d, %Y, %I:%M %p").to_string();
    let now_time_only = now.format("%I:%M %p").to_string();
    out.push_str("The current date and time is ");
    out.push_str(&now_full);
    out.push_str(".\n\n");

    if !facts.is_empty() {
        out.push_str("Known facts about the user, remembered from past conversations:\n");
        for fact in facts {
            out.push_str("- ");
            out.push_str(&fact.text);
            out.push('\n');
        }
        out.push('\n');
    }

    // A dedicated example for time questions, using the *real* current time computed above (not
    // a fixed fake value) -- a plain instruction to "use the date/time given above" wasn't
    // reliably followed for short phrasings like "What time is it?" (it worked for a more
    // elaborate phrasing but not this one, tested side by side), so this demonstrates the exact
    // pattern instead. Using the real time here means the example is never factually wrong
    // regardless of when it's generated, so there's no risk of the model anchoring on a stale or
    // made-up value the way a fixed example would.
    out.push_str(&format!(
        "User: What time is it?\n\n\
         Xenon: It's currently {now_time_only}.\n\n"
    ));

    out
}

const STOP_SEQUENCES: &[&str] = &["\n\nUser:", "\n\nuser:"];

/// How many trailing bytes of `text` are a proper prefix of a stop sequence. Those bytes stay
/// buffered so the chat never shows the next "User:" prompt, or a cursor sitting on it.
fn stop_hold_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut best = 0;
    for stop in STOP_SEQUENCES {
        let stop_b = stop.as_bytes();
        let max = bytes.len().min(stop_b.len().saturating_sub(1));
        for n in (1..=max).rev() {
            if &bytes[bytes.len() - n..] == &stop_b[..n] {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

fn emit_visible(ctx: &CallbackCtx, text: &str) {
    if text.is_empty() {
        return;
    }
    ctx.full_text.borrow_mut().push_str(text);
    let _ = ctx.app.emit(
        "token-stream",
        TokenEvent {
            conversation_id: ctx.conversation_id.clone(),
            text: text.to_string(),
        },
    );
}

/// Appends one generated fragment. Returns false when the model has started the next user turn
/// and generation should stop. The "User:" text itself is not emitted.
fn absorb_token(ctx: &CallbackCtx, text: &str) -> bool {
    if text.is_empty() {
        return true;
    }
    let mut pending = ctx.pending.borrow_mut();
    pending.push_str(text);
    for stop in STOP_SEQUENCES {
        if pending.ends_with(stop) {
            let keep = pending.len() - stop.len();
            let emit = pending[..keep].to_string();
            pending.clear();
            drop(pending);
            emit_visible(ctx, &emit);
            return false;
        }
    }
    let hold = stop_hold_len(&pending);
    let emit_end = pending.len() - hold;
    let emit = pending[..emit_end].to_string();
    pending.drain(..emit_end);
    drop(pending);
    emit_visible(ctx, &emit);
    true
}

fn flush_pending(ctx: &CallbackCtx) {
    let rest = std::mem::take(&mut *ctx.pending.borrow_mut());
    if rest.is_empty() {
        return;
    }
    let hold = stop_hold_len(&rest);
    emit_visible(ctx, &rest[..rest.len() - hold]);
}

/// Context handed to the C callback via the `user_data` void*. The callback runs synchronously
/// on the same thread that calls `xenon_generate`, so `RefCell` (not a Mutex) is fine here.
struct CallbackCtx {
    app: AppHandle,
    conversation_id: String,
    /// Tail that might still turn into "\n\nUser:". Not shown until it can't be that prompt.
    pending: RefCell<String>,
    /// Reply text actually shown (stop sequence excluded). Used by Sloth fact extraction.
    full_text: RefCell<String>,
}

extern "C" fn on_token(
    text: *const std::os::raw::c_char,
    _token_id: u32,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return true;
    }
    let ctx = unsafe { &*(user_data as *const CallbackCtx) };
    let text_str = if text.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("")
    };
    absorb_token(ctx, text_str)
}

/// Minimal callback for the Sloth fact-extraction call -- just accumulates text into `user_data`
/// (a bare `RefCell<String>`, not a full `CallbackCtx`), since the extraction reply is never
/// shown in the UI and doesn't need `token-stream` events or stop-sequence checking beyond a
/// simple length cap (extraction prompts ask for one short sentence; a runaway generation would
/// mean something went wrong, not that there's more useful fact text coming).
extern "C" fn on_extraction_token(
    text: *const std::os::raw::c_char,
    _token_id: u32,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return true;
    }
    let buf = unsafe { &*(user_data as *const RefCell<String>) };
    let text_str = if text.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("")
    };
    buf.borrow_mut().push_str(text_str);
    buf.borrow().len() < 300 // safety cap, well beyond one short sentence
}

/// The static half of the extraction prompt -- everything except the just-completed exchange
/// itself. Byte-identical on every extraction call, so it's prefilled once into
/// `EngineInner::extraction_prefix` at `load_engine` time (see that field's doc comment) instead
/// of being re-evaluated on every Sloth turn (measured before Phase 8: 205 tokens, every time).
/// Kept deliberately strict ("reply with exactly: NONE") since this is a small,
/// non-instruction-tuned base model -- without a strong steer it tends to invent something
/// rather than correctly recognize "nothing new here". Few-shot, not just an abstract
/// instruction -- a bare "extract a fact or say NONE" instruction was found to make this small
/// base model *invent* a plausible-sounding fact (e.g. "named John, loves pizza") even when the
/// user never said any such thing, rather than correctly recognizing nothing new was revealed.
/// Real bug found in usage (2026-08-02): the model once misattributed Xenon's *own* stated name
/// as a fact about the user ("The user's name is Xenon") after being asked "What is your name?"
/// -- added a dedicated example so it's explicit that facts about Xenon itself never count as
/// facts about the user, the same conflation risk as the NONE examples below.
const EXTRACTION_STATIC_PREFIX: &str =
    "Xenon only records a fact if the user directly stated it about *themselves*. Facts \
     Xenon states about itself (its own name, its own capabilities) are never facts about \
     the user. If the user did not clearly state something new and durable about \
     themselves, Xenon replies with exactly: NONE\n\n\
     User: My favorite color is blue and I have a dog named Rex.\n\n\
     Xenon: That's lovely! Rex sounds like a great dog.\n\n\
     Fact: The user's favorite color is blue and they have a dog named Rex.\n\n\
     User: What's the weather like today?\n\n\
     Xenon: I don't have access to real-time weather data.\n\n\
     Fact: NONE\n\n\
     User: What is my name?\n\n\
     Xenon: I'm sorry, I don't have that information. Can you tell me your name?\n\n\
     Fact: NONE\n\n\
     User: What is your name?\n\n\
     Xenon: My name is Xenon.\n\n\
     Fact: NONE\n\n";

/// The volatile (per-call) half of the extraction prompt: just this turn's exchange.
fn build_extraction_suffix(user_text: &str, reply_text: &str) -> String {
    format!("User: {user_text}\n\nXenon: {reply_text}\n\nFact:")
}

/// Runs the Sloth fact-extraction step: a second, small, low-temperature generate call using the
/// just-completed turn, under the *same* already-held engine lock as the main reply (kept
/// sequential rather than concurrent -- rwkv.cpp's per-engine state isn't safe for concurrent
/// eval calls, see `EngineInner`'s doc comment, and this call is short at `max_tokens=40`).
/// Always starts from a fresh copy of `extraction_prefix` and discards it afterwards -- this
/// call must never be able to read or write a conversation's cached state in
/// `EngineInner::conversation_cache` (Phase 8 requirement: extraction is a side channel, not a
/// second turn of the actual conversation). On success, persists any real fact via
/// `sloth_memory::append_fact` and emits `sloth-facts-updated` so an open Memory panel (see
/// App.vue) can refresh live. Never surfaces its own errors as a `generation-error` -- a
/// failed/garbled extraction just means no new fact gets remembered this turn, which shouldn't
/// interrupt or error out the visible reply that already completed successfully.
fn run_fact_extraction(
    engine: *mut ffi::XenonEngine,
    extraction_prefix: &XenonStatePtr,
    app: &AppHandle,
    user_text: &str,
    reply_text: &str,
) {
    let suffix = match CString::new(build_extraction_suffix(user_text, reply_text)) {
        Ok(c) => c,
        Err(_) => return,
    };

    let scratch = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(scratch, extraction_prefix.0) };

    let buf = RefCell::new(String::new());
    let buf_ptr = &buf as *const RefCell<String> as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate_with_state(
            engine,
            scratch,
            suffix.as_ptr(),
            40,   // max_tokens -- one short sentence, not a full reply
            0.2,  // temperature -- extraction should be far more deterministic than a chat reply
            0.2,  // top_p
            1.1,  // repeat_penalty -- mild; extraction is already short/constrained by the few-shot prompt
            on_extraction_token,
            buf_ptr,
        )
    };
    unsafe { ffi::xenon_state_free(scratch) };
    if status != ffi::XenonStatus::OK {
        return;
    }

    let extracted = buf.into_inner();
    let trimmed = extracted
        .split("\n\n") // guard against the model drifting into a fake next turn, same as chat replies do
        .next()
        .unwrap_or("")
        .trim();

    // Safety net beyond the prompt's own few-shot guidance: an extraction that just echoes the
    // assistant's own reply (observed failure mode -- e.g. "I don't know your name" verbatim as
    // the "fact") is never a real fact about the user, regardless of what the model said.
    let looks_like_echo = trimmed.eq_ignore_ascii_case(reply_text.trim())
        || reply_text.trim().contains(trimmed);

    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("none")
        || trimmed.len() > 300
        || looks_like_echo
    {
        return;
    }

    let embedding = crate::voice::embed_texts(app, &[trimmed.to_string()]).and_then(|mut v| v.pop());
    match sloth_memory::append_fact(app, trimmed.to_string(), now_millis(), embedding) {
        Ok(facts) => {
            let _ = app.emit("sloth-facts-updated", facts);
        }
        Err(e) => eprintln!("[sloth_memory] could not persist extracted fact: {e}"),
    }
}

/// How many facts without an embedding get one per Sloth turn, so facts saved while the embedder
/// was unavailable (or imported from the old JSON store) catch up without one long pause.
const EMBED_BACKFILL_PER_TURN: usize = 64;

/// The facts sent with one Sloth turn: found by meaning when the voice sidecar can embed the
/// message, by keywords otherwise. Also embeds a batch of facts that have no embedding yet.
fn sloth_facts_for(app: &AppHandle, message: &str) -> Vec<sloth_memory::SlothFact> {
    let stored = match sloth_memory::load_facts(app) {
        Ok(facts) => facts,
        Err(e) => {
            eprintln!("[sloth_memory] could not load facts: {e}");
            return Vec::new();
        }
    };
    let missing = sloth_memory::missing_embeddings(&stored, EMBED_BACKFILL_PER_TURN);
    let mut texts = vec![message.to_string()];
    texts.extend(missing.iter().map(|f| f.text.clone()));

    let (query, stored) = match crate::voice::embed_texts(app, &texts) {
        Some(mut vectors) => {
            let query = vectors.remove(0);
            let items: Vec<(String, Vec<f32>)> = missing.iter().map(|f| f.id.clone()).zip(vectors).collect();
            if let Err(e) = sloth_memory::store_embeddings(app, &items) {
                eprintln!("[sloth_memory] could not save embeddings: {e}");
            }
            (Some(query), sloth_memory::load_facts(app).unwrap_or(stored))
        }
        None => (None, stored),
    };
    sloth_memory::relevant_facts(&stored, message, query.as_deref(), sloth_memory::MAX_PROMPT_FACTS)
}

/// Deterministic answer for time/date questions, bypassing the model entirely. Added
/// 2026-08-04 after prompt-grounding (real clock text injected into the prompt, plus a
/// demonstrated example) still wasn't reliable enough: the model would get *close* to the real
/// time but not exact (e.g. reporting "8:05 PM" when it was actually 8:10), and sometimes dropped
/// AM/PM -- a known small-model weakness (precisely reproducing a specific number from context
/// isn't guaranteed even when the right value is right there in the prompt). Time/date has a
/// single objectively correct answer, so there's no reason to leave it up to a small model's
/// approximation once heuristic detection is reliable enough to catch the question.
/// Returns `None` for anything that doesn't clearly ask for the time/date, falling through to
/// real generation as normal.
fn try_answer_time_date_deterministically(user_text: &str) -> Option<String> {
    let lower = user_text.to_lowercase();

    let asks_time = ["what time", "current time", "time is it", "know the time", "tell me the time"]
        .iter()
        .any(|p| lower.contains(p));
    let asks_date = ["what day", "today's date", "what's the date", "what is the date", "what date", "current date"]
        .iter()
        .any(|p| lower.contains(p));

    if !asks_time && !asks_date {
        return None;
    }

    let now = chrono::Local::now();
    // "%-I"/"%-d" (non-padded) so TTS reads "8:10 PM" naturally instead of piper sounding out
    // "zero eight ten" for a zero-padded "08:10".
    let time_str = now.format("%-I:%M %p").to_string();
    let date_str = now.format("%A, %B %-d, %Y").to_string();

    Some(match (asks_time, asks_date) {
        (true, true) => format!("It's {time_str} on {date_str}."),
        (true, false) => format!("It's currently {time_str}."),
        (false, true) => format!("Today is {date_str}."),
        (false, false) => unreachable!(),
    })
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Feeds `text` into `state` via `xenon_prefill`, mapping the C error into a `Result` instead of
/// silently ignoring it. `text` == "" is a cheap no-op (see `xenon_prefill`'s doc comment).
fn prefill_text(engine: *mut ffi::XenonEngine, state: *mut ffi::XenonState, text: &str) -> Result<(), String> {
    let c_text = CString::new(text).map_err(|e| format!("prefill text contained an embedded NUL byte: {e}"))?;
    let status = unsafe { ffi::xenon_prefill(engine, state, c_text.as_ptr()) };
    if status == ffi::XenonStatus::OK {
        Ok(())
    } else {
        Err(last_error())
    }
}

/// The heart of Phase 8: given the turns already known to have happened (`prior_turns` --
/// everything in this call's `history` except the newest, not-yet-answered user turn), returns a
/// pointer to a state that has consumed exactly `STATIC_PREFIX` + `history_text(prior_turns)`,
/// reusing as much of `inner.conversation_cache` as possible instead of rebuilding from scratch.
///
/// Diff logic: if this conversation has a cache entry whose `consumed` turns are a byte-equal
/// prefix of `prior_turns`, this is a plain continuation (or a regenerate of the last reply, which
/// sends the identical `prior_turns` as before) -- "extend" by prefilling only the new tail turns.
/// Otherwise (no entry yet, an evicted entry, or `prior_turns` diverges anywhere -- an edit to an
/// earlier message, a delete, or a shorter history) -- "rebuild" from a fresh copy of
/// `static_prefix`, prefilling all of `prior_turns`. Either way, `consumed` is set to
/// `prior_turns` *before* this turn generates anything, from `history` text the frontend already
/// considers canonical -- this cache is never advanced from this turn's own (possibly still to be
/// trimmed, see `chat.ts`'s `completeGeneration`) generation output. See `EngineInner`'s doc
/// comment and this phase's design notes for why that ordering is what makes the cache safe
/// against edits/regenerates/the frontend's post-hoc trimming without any special-casing here.
fn ensure_cache_extended(
    inner: &mut EngineInner,
    conversation_id: &str,
    prior_turns: &[ChatTurn],
) -> Result<*mut ffi::XenonState, String> {
    let engine = inner.engine.0;

    if let Some(idx) = inner
        .conversation_cache
        .iter()
        .position(|(id, _)| id == conversation_id)
    {
        let is_extendable = {
            let cached = &inner.conversation_cache[idx].1;
            cached.consumed.len() <= prior_turns.len() && cached.consumed == prior_turns[..cached.consumed.len()]
        };

        if is_extendable {
            let cached = &mut inner.conversation_cache[idx].1;
            let delta = &prior_turns[cached.consumed.len()..];
            if !delta.is_empty() {
                prefill_text(engine, cached.state.0, &history_text(delta))?;
            }
            cached.consumed = prior_turns.to_vec();

            // Move to the back (most-recently-used) so LRU eviction below doesn't pick this one.
            let entry = inner.conversation_cache.remove(idx);
            inner.conversation_cache.push(entry);
            return Ok(inner.conversation_cache.last().unwrap().1.state.0);
        }

        // Diverges somewhere -- stale relative to what the frontend just sent (an edit, a
        // delete, or otherwise). Drop it and rebuild below.
        inner.conversation_cache.remove(idx);
    }

    let new_state = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(new_state, inner.static_prefix.0) };
    if let Err(e) = prefill_text(engine, new_state, &history_text(prior_turns)) {
        unsafe { ffi::xenon_state_free(new_state) };
        return Err(e);
    }

    if inner.conversation_cache.len() >= MAX_CACHED_CONVERSATIONS {
        inner.conversation_cache.remove(0); // least-recently-used is always at the front
    }
    inner.conversation_cache.push((
        conversation_id.to_string(),
        CachedConversation { consumed: prior_turns.to_vec(), state: XenonStatePtr(new_state) },
    ));
    Ok(inner.conversation_cache.last().unwrap().1.state.0)
}

/// Runs generation to completion on the calling thread (blocking) and emits Tauri events as
/// tokens stream in. Meant to be called from inside `spawn_blocking` by the IPC command below,
/// never directly on the async/event-loop thread. Takes the already-locked `EngineInner` so the
/// mutex stays held for the full duration of the call (see `EngineInner` doc comment). `agent`
/// ("dementia" | "sloth") controls whether Sloth's persistent facts are injected into the prompt
/// and whether a fact-extraction pass runs after a successful reply -- Dementia leaves both
/// skipped, matching the app's pre-Sloth behavior exactly.
fn generate_blocking(
    inner: &mut EngineInner,
    app: AppHandle,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) {
    let engine = inner.engine.0;

    // `prior_turns` is everything already-happened; the newest user turn is what this call
    // answers. Both the deterministic short-circuit below and the real generation path share
    // the same cache-extend step so the cache never falls behind regardless of which path a
    // given turn takes.
    let (prior_turns, new_user_turn) = match history.split_last() {
        Some((last, rest)) => (rest, last),
        None => {
            let _ = app.emit(
                "generation-error",
                GenerationErrorEvent { conversation_id, message: "generate_response called with empty history".to_string() },
            );
            return;
        }
    };

    let base_state = match ensure_cache_extended(inner, &conversation_id, prior_turns) {
        Ok(s) => s,
        Err(message) => {
            let _ = app.emit("generation-error", GenerationErrorEvent { conversation_id, message });
            return;
        }
    };

    if let Some(answer) = try_answer_time_date_deterministically(&new_user_turn.content) {
        let _ = app.emit(
            "token-stream",
            TokenEvent { conversation_id: conversation_id.clone(), text: answer },
        );
        let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
        return;
    }

    let is_sloth = agent == "sloth";
    let facts = if is_sloth { sloth_facts_for(&app, &new_user_turn.content) } else { Vec::new() };

    // Never generate directly into the authoritative cached state (`base_state`) -- see
    // `ensure_cache_extended`'s doc comment on why the cache only ever advances from
    // caller-confirmed history text, never from this turn's own raw generation output. Copy it
    // into a scratch state, feed the volatile header + new turn, generate there, then discard.
    let turn_state = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(turn_state, base_state) };

    let suffix = format!(
        "{}User: {}\n\nXenon:",
        volatile_header_text(&facts),
        new_user_turn.content.trim()
    );
    let suffix_c = match CString::new(suffix) {
        Ok(c) => c,
        Err(e) => {
            unsafe { ffi::xenon_state_free(turn_state) };
            let _ = app.emit(
                "generation-error",
                GenerationErrorEvent {
                    conversation_id,
                    message: format!("prompt contained an embedded NUL byte: {e}"),
                },
            );
            return;
        }
    };

    let ctx = CallbackCtx {
        app: app.clone(),
        conversation_id: conversation_id.clone(),
        pending: RefCell::new(String::new()),
        full_text: RefCell::new(String::new()),
    };
    let ctx_ptr = &ctx as *const CallbackCtx as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate_with_state(
            engine,
            turn_state,
            suffix_c.as_ptr(),
            200,  // max_tokens
            0.8,  // temperature (matches Phase 1 CLI harness default)
            0.5,  // top_p (matches Phase 1 CLI harness default)
            // repeat_penalty (Phase 7 follow-up, 2026-08-02): added after real usage showed this
            // model imitating its own earlier canned reply for unrelated follow-up questions once
            // that phrase was sitting in the resent conversation history -- see xenon_inference.h's
            // doc comment. Empirically tuned, not guessed: replaying the actual failure
            // conversation (a phrase repeated 3x already in history) showed 1.15 (llama.cpp's
            // typical low end) was NOT enough to break the loop -- the model repeated the trap
            // phrase a 4th time regardless. 1.3 (llama.cpp's typical high end) did break it,
            // producing a real relevant answer, and re-verified to cause no quality loss on normal
            // short exchanges (correct arithmetic, coherent fun facts, normal greetings).
            1.3,
            on_token,
            ctx_ptr,
        )
    };

    unsafe { ffi::xenon_state_free(turn_state) };

    // Drop a held "\n\nUser:" prefix. Anything else still buffered is real reply text.
    flush_pending(&ctx);

    if status == ffi::XenonStatus::OK {
        let reply_text = ctx.full_text.borrow().clone();
        // Done before Sloth's extraction pass. A mic reply speaks its last
        // sentence from this event, and extraction is a second generate on the
        // same engine. Waiting would hold that audio until the extra pass ends.
        let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id: conversation_id.clone() });
        if is_sloth {
            run_fact_extraction(engine, &inner.extraction_prefix, &app, &new_user_turn.content, &reply_text);
        }
    } else {
        let _ = app.emit(
            "generation-error",
            GenerationErrorEvent {
                conversation_id,
                message: last_error(),
            },
        );
    }
}

/// Tauri IPC command: frontend sends the full turn history (ending with the new user message)
/// plus which agent this turn should use, backend streams the assistant's reply back via
/// `token-stream` / `generation-done` / `generation-error` events tagged with `conversation_id`.
#[tauri::command]
pub async fn generate_response(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) -> Result<(), String> {
    let engine_arc = engine.0.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Held for the entire blocking generate() call (cache lookups/extends, the generate
        // call itself, and, for Sloth, the extraction call right after it) -- see `EngineInner`'s
        // doc comment.
        let mut guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        generate_blocking(&mut guard, app, conversation_id, history, agent);
    })
    .await
    .map_err(|e| e.to_string())
}
