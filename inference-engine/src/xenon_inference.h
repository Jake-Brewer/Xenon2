// xenon_inference.h
//
// Small, stable C API wrapping rwkv.cpp for the Xenon2 project. This boundary is deliberately
// plain C (no C++ types in the public surface) so it can be called via FFI from Rust, Python
// (ctypes/cffi), or any other language later in the project, not just from the bundled CLI
// harness (test_inference.exe). Do not add C++-only types to this header.
#ifndef XENON_INFERENCE_H
#define XENON_INFERENCE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
#   if defined(XENON_INFERENCE_SHARED)
#       if defined(XENON_INFERENCE_BUILD)
#           define XENON_API __declspec(dllexport)
#       else
#           define XENON_API __declspec(dllimport)
#       endif
#   else
#       define XENON_API
#   endif
#else
#   define XENON_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded model + tokenizer, ready for generation. Does NOT itself hold
// conversation state any more (Phase 8) -- state now lives in caller-owned `xenon_state` objects
// below, so a caller can keep multiple independent conversations alive against one loaded model.
typedef struct xenon_engine xenon_engine;

// Opaque, caller-owned RWKV state (recurrent state + logits from the last token evaluated into
// it, plus internal bookkeeping for the repeat-penalty window -- see xenon_prefill). Sized from
// xenon_get_state_len()/xenon_get_logits_len() internally; the caller never touches its layout,
// only passes the pointer around. One xenon_state can represent "this conversation so far",
// letting a caller feed only the *new* text on each turn (xenon_prefill / xenon_generate_with_state)
// instead of re-processing the whole conversation every time -- see those functions' docs.
typedef struct xenon_state xenon_state;

typedef enum xenon_status {
    XENON_OK = 0,
    XENON_ERROR_ARGS = 1,
    XENON_ERROR_MODEL_LOAD = 2,
    XENON_ERROR_VOCAB_LOAD = 3,
    XENON_ERROR_EVAL = 4,
} xenon_status;

// Invoked once per generated token, as soon as it is available (streaming, not batched).
// - text: pointer to a UTF-8, null-terminated buffer containing the newly available decoded
//   text for this step. May be empty ("") if the token only completed a partial multi-byte
//   UTF-8 sequence that is still being buffered internally (in that case a later callback
//   invocation will include the completed characters).
// - token_id: raw vocab id of the token that was generated, in case the caller wants it.
// - user_data: whatever pointer was passed into xenon_generate.
// Return `false` to stop generation early (e.g. caller wants to cancel); return `true` to keep
// going until max_tokens is reached.
// When the model emits its end-of-text token, generation ends with one last call whose text is
// "" and whose token_id is XENON_END_OF_TEXT: the model finished its reply on its own.
typedef bool (*xenon_token_callback)(const char * text, uint32_t token_id, void * user_data);

#define XENON_END_OF_TEXT 0u

// Loads a quantized rwkv.cpp ggml model file and the matching World-tokenizer vocab file
// (see inference-engine/tools/convert_world_vocab.py and inference-engine/data/world_vocab.bin).
// - model_path: path to a .bin ggml model file (FP16/FP32/quantized) produced by rwkv.cpp.
// - vocab_path: path to the binary World-tokenizer vocab file.
// - n_threads: CPU threads used for evaluation; must be >= 1.
// - n_gpu_layers: number of RWKV layers to offload to GPU. 0 = CPU-only. Only has effect if this
//   library was built and linked against a CUDA-enabled rwkv.cpp (see README for the two build
//   configurations); otherwise this value is accepted but has no effect (falls back to CPU).
// Returns NULL on failure -- call xenon_get_last_error() for details.
XENON_API xenon_engine * xenon_load_model(
    const char * model_path,
    const char * vocab_path,
    uint32_t n_threads,
    uint32_t n_gpu_layers
);

// Frees all resources associated with an engine. Safe to call with NULL.
XENON_API void xenon_free_engine(xenon_engine * engine);

// Resets the engine's RWKV state (equivalent to starting a brand new conversation with no
// prior context). Kept for backwards compatibility; equivalent to xenon_reset_state_v2(engine,
// state) on whatever scratch state xenon_generate() itself uses internally. Exposed separately
// in case a caller wants to manage state across multiple generate() calls itself.
XENON_API void xenon_reset_state(xenon_engine * engine);

// --- Phase 8: caller-owned incremental state ------------------------------------------------
//
// A `xenon_state` lets a caller keep a conversation's RWKV state alive across calls instead of
// re-processing the whole conversation as text every turn (which is what plain xenon_generate()
// below still does, for backwards compatibility -- it allocates a scratch xenon_state, resets
// it, and is otherwise implemented in terms of these functions). Typical incremental usage:
//
//   xenon_state * s = xenon_state_new(engine);
//   xenon_state_reset(engine, s);
//   xenon_prefill(engine, s, "<static preamble text, never changes>");
//   // ... later, once per turn:
//   xenon_state * scratch = xenon_state_new(engine);
//   xenon_state_copy(scratch, s);              // don't generate directly into the authoritative state
//   xenon_generate_with_state(engine, scratch, "User: hi\n\nXenon:", ..., callback, user_data);
//   xenon_state_free(scratch);
//   // once the caller knows the canonical (possibly trimmed) text for this turn, commit it:
//   xenon_prefill(engine, s, "User: hi\n\nXenon: <trimmed reply>\n\n");
//
// This split (generate into a scratch copy, only ever advance the authoritative state via
// xenon_prefill of caller-confirmed text) is deliberate: it means the authoritative state can
// never drift from whatever text the caller considers canonical, even if the caller trims or
// otherwise post-processes what a model actually generated before treating it as final.

// Allocates a new state sized for `engine` (from xenon_get_state_len()/xenon_get_logits_len()).
// The state is uninitialized (garbage) until xenon_state_reset() or a copy is applied to it.
// Returns NULL on allocation failure or a NULL engine.
XENON_API xenon_state * xenon_state_new(xenon_engine * engine);

// Frees a state allocated by xenon_state_new(). Safe to call with NULL.
XENON_API void xenon_state_free(xenon_state * state);

// Copies all of `src`'s contents (RWKV state, cached logits, repeat-penalty token window) into
// `dst`. Both must have been allocated for the same engine. This is how callers take a cheap
// snapshot to generate from without mutating the authoritative copy (see usage above).
XENON_API void xenon_state_copy(xenon_state * dst, const xenon_state * src);

// Resets `state` to a fresh/empty conversation (equivalent to what xenon_reset_state() does for
// xenon_generate()'s own internal scratch state): RWKV state zeroed via rwkv_init_state, cached
// logits cleared, repeat-penalty token window cleared.
XENON_API void xenon_state_reset(xenon_engine * engine, xenon_state * state);

// Evaluates `text` into `state` WITHOUT generating anything -- pure prefill. Advances `state`'s
// RWKV state, cached logits (from the last token of `text`), and repeat-penalty token window in
// place, exactly as if `text` had been the tail of a prompt fed to xenon_generate() from a fresh
// state. `text` == "" is a no-op (returns XENON_OK immediately, state untouched) -- convenient
// for callers with an optional block (e.g. "no extra facts this turn") that may be empty.
XENON_API xenon_status xenon_prefill(
    xenon_engine * engine,
    xenon_state * state,
    const char * text
);

// Same contract as xenon_generate() below, except it starts from (and advances in place) a
// caller-supplied `state` instead of an internal scratch one, and `prompt` is just the NEW text
// to evaluate before generating (e.g. the newest turn), not the whole conversation -- whatever
// was already fed into `state` (via xenon_prefill or a previous xenon_generate_with_state call)
// is not re-evaluated. Like xenon_generate(), `prompt` encoding to zero tokens is an error
// (XENON_ERROR_ARGS) -- use xenon_prefill first (which tolerates "") if there's nothing new to
// feed before generating.
XENON_API xenon_status xenon_generate_with_state(
    xenon_engine * engine,
    xenon_state * state,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
);

// Streams a generated continuation of `prompt`, invoking `callback` once per generated token
// as it is produced (not batched). Stops after `max_tokens` tokens, when the callback returns
// false, or when the model emits its end-of-text token (id 0).
// - temperature / top_p: standard sampling parameters (temperature 0 = greedy/argmax).
// - repeat_penalty: discourages resampling tokens that recently appeared in the prompt's tail
//   (up to the last 256 prompt tokens) or earlier in this same generation, by dividing/scaling
//   their logits before sampling (the standard llama.cpp-style repetition penalty). 1.0 = no
//   penalty (identical to the old behavior); values around 1.1-1.3 are the usual useful range.
//   Added after real usage showed this small model imitating a canned phrase that was still
//   sitting in the conversation history from an earlier turn -- with no penalty at all, once a
//   phrase like "I'm sorry, I can't access my memories" entered the resent history, the model
//   would keep reproducing it for unrelated follow-up questions.
// Resets RWKV state at the start of the call (this function has no memory of previous calls --
// as of Phase 8 it is implemented as a thin wrapper: allocate a scratch xenon_state, reset it,
// call xenon_generate_with_state, free it. Kept for the two callers that bind this exact
// signature -- voice-pipeline/xenon_engine.py's ctypes bindings and test_inference.cpp -- see
// xenon_generate_with_state above for the incremental-state version this app actually uses now).
// Returns XENON_OK on success (including early stop via callback), otherwise an error code.
XENON_API xenon_status xenon_generate(
    xenon_engine * engine,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
);

// Number of float32 elements in this model's fixed-size RWKV state. This does NOT grow with
// conversation length -- exposed so callers/tests can verify that architectural property
// directly rather than just assuming it.
XENON_API size_t xenon_get_state_len(xenon_engine * engine);

// Number of transformer-analogous "layers" in the loaded model (used e.g. to validate
// --gpu-layers arguments against the actual model).
XENON_API size_t xenon_get_n_layer(xenon_engine * engine);

// Returns 1 if this build of the engine was compiled against a CUDA/cuBLAS-enabled rwkv.cpp,
// 0 otherwise. Lets callers know whether --gpu-layers will actually do anything.
XENON_API int xenon_has_gpu_support(void);

// Called between chunks while a prompt is read (xenon_prefill, and the prompt part of
// xenon_generate_with_state), on the calling thread. A host can wait in it, e.g. to let a hot GPU
// cool: reading runs the GPU flat out and isn't otherwise paused. NULL (the default) disables it.
typedef void (*xenon_prefill_hook)(void * user_data);
XENON_API void xenon_set_prefill_hook(xenon_prefill_hook hook, void * user_data);

// Human-readable string for the last error that occurred on this thread. May be empty.
XENON_API const char * xenon_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // XENON_INFERENCE_H
