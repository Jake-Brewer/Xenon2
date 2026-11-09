//! Raw FFI declarations mirroring `inference-engine/src/xenon_inference.h` exactly.
//!
//! This is a hand-written binding (not bindgen-generated) because the C header is small and
//! deliberately stable/plain-C -- see the header's own top comment. Keep this in sync with
//! `xenon_inference.h` if that header ever changes.
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

/// Opaque handle -- never dereferenced on the Rust side, only passed back into the C API.
#[repr(C)]
pub struct XenonEngine {
    _private: [u8; 0],
}

/// Opaque, caller-owned RWKV state (Phase 8). Never dereferenced on the Rust side -- see
/// `xenon_inference.h`'s docs on `xenon_state`.
#[repr(C)]
pub struct XenonState {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XenonStatus(pub c_int);

impl XenonStatus {
    pub const OK: XenonStatus = XenonStatus(0);
}

/// Matches `xenon_token_callback` in xenon_inference.h: `bool (*)(const char*, uint32_t, void*)`.
/// token_id of the final, empty callback when the model ends its reply itself (see the header).
pub const XENON_END_OF_TEXT: u32 = 0;

pub type XenonTokenCallback =
    extern "C" fn(text: *const c_char, token_id: u32, user_data: *mut c_void) -> bool;

#[link(name = "xenon_inference")]
extern "C" {
    pub fn xenon_load_model(
        model_path: *const c_char,
        vocab_path: *const c_char,
        n_threads: u32,
        n_gpu_layers: u32,
    ) -> *mut XenonEngine;

    pub fn xenon_free_engine(engine: *mut XenonEngine);

    pub fn xenon_reset_state(engine: *mut XenonEngine);

    pub fn xenon_generate(
        engine: *mut XenonEngine,
        prompt: *const c_char,
        max_tokens: c_int,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
        callback: XenonTokenCallback,
        user_data: *mut c_void,
    ) -> XenonStatus;

    // --- Phase 8: caller-owned incremental state -- see xenon_inference.h's docs -------------

    pub fn xenon_state_new(engine: *mut XenonEngine) -> *mut XenonState;

    pub fn xenon_state_free(state: *mut XenonState);

    pub fn xenon_state_copy(dst: *mut XenonState, src: *const XenonState);

    pub fn xenon_state_reset(engine: *mut XenonEngine, state: *mut XenonState);

    pub fn xenon_prefill(engine: *mut XenonEngine, state: *mut XenonState, text: *const c_char) -> XenonStatus;

    pub fn xenon_generate_with_state(
        engine: *mut XenonEngine,
        state: *mut XenonState,
        prompt: *const c_char,
        max_tokens: c_int,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
        callback: XenonTokenCallback,
        user_data: *mut c_void,
    ) -> XenonStatus;

    pub fn xenon_get_state_len(engine: *mut XenonEngine) -> usize;

    pub fn xenon_get_n_layer(engine: *mut XenonEngine) -> usize;

    pub fn xenon_has_gpu_support() -> c_int;

    pub fn xenon_get_last_error() -> *const c_char;
}
