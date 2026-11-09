#include "xenon_inference.h"
#include "world_tokenizer.h"

#include "rwkv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef XENON_HAS_CUDA
#define XENON_HAS_CUDA 0
#endif

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string & msg) {
    g_last_error = msg;
}

// How many trailing tokens (prompt tail + tokens generated so far this call) count toward the
// repetition penalty. Large enough to cover a canned phrase sitting a turn or two back in the
// resent conversation history (the actual failure mode this was added for), small enough that
// scanning it every step is negligible next to the O(n_vocab) softmax pass below.
constexpr size_t REPEAT_PENALTY_WINDOW = 256;

// Softmax + temperature + top-p (nucleus) sampling, mirroring rwkv.cpp's python/sampling.py
// reference implementation so behavior matches the rest of the rwkv.cpp ecosystem. `repeat_penalty`
// (1.0 = disabled) applies the standard llama.cpp-style penalty to any token present in
// `recent_tokens` before the softmax: divide positive logits, scale up negative ones, so recently
// -seen tokens become less likely without being hard-banned.
uint32_t sample_logits(
    const float * raw_logits,
    size_t n_vocab,
    float temperature,
    float top_p,
    float repeat_penalty,
    const std::vector<uint32_t> & recent_tokens,
    std::mt19937 & rng
) {
    std::vector<float> logits_buf;
    const float * logits = raw_logits;

    if (repeat_penalty != 1.0f && repeat_penalty > 0.0f && !recent_tokens.empty()) {
        logits_buf.assign(raw_logits, raw_logits + n_vocab);
        std::vector<bool> penalized(n_vocab, false);
        for (uint32_t t : recent_tokens) {
            if (t < n_vocab && !penalized[t]) {
                penalized[t] = true;
                float & l = logits_buf[t];
                l = (l > 0.0f) ? (l / repeat_penalty) : (l * repeat_penalty);
            }
        }
        logits = logits_buf.data();
    }

    std::vector<float> probs(n_vocab);

    float max_logit = *std::max_element(logits, logits + n_vocab);

    double sum = 0.0;
    for (size_t i = 0; i < n_vocab; i++) {
        probs[i] = std::exp(static_cast<double>(logits[i] - max_logit));
        sum += probs[i];
    }
    for (size_t i = 0; i < n_vocab; i++) {
        probs[i] = static_cast<float>(probs[i] / sum);
    }

    if (temperature <= 0.0f) {
        return static_cast<uint32_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<float> sorted_probs = probs;
        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());

        double cumulative = 0.0;
        float cutoff = sorted_probs.back();
        for (float p : sorted_probs) {
            cumulative += p;
            if (cumulative > top_p) {
                cutoff = p;
                break;
            }
        }

        for (size_t i = 0; i < n_vocab; i++) {
            if (probs[i] < cutoff) probs[i] = 0.0f;
        }
    }

    if (temperature != 1.0f) {
        double s = 0.0;
        for (size_t i = 0; i < n_vocab; i++) {
            probs[i] = std::pow(probs[i], 1.0f / temperature);
            s += probs[i];
        }
        for (size_t i = 0; i < n_vocab; i++) {
            probs[i] = static_cast<float>(probs[i] / s);
        }
    } else {
        double s = std::accumulate(probs.begin(), probs.end(), 0.0);
        for (size_t i = 0; i < n_vocab; i++) probs[i] = static_cast<float>(probs[i] / s);
    }

    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Returns the length of the prefix of `buf` that consists of complete, valid UTF-8 sequences,
// holding back a possibly-incomplete trailing multi-byte sequence for the next call. This lets
// us stream World-tokenizer output (which is byte-level, so a single token can be half of a
// multi-byte UTF-8 character) as valid text without corrupting split characters.
size_t utf8_safe_prefix_len(const std::string & buf) {
    if (buf.empty()) return 0;

    size_t n = buf.size();
    // Look back up to 3 bytes from the end to find the start of a possibly-incomplete sequence.
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char b = static_cast<unsigned char>(buf[n - back]);

        size_t expected_len;
        if ((b & 0x80) == 0x00) expected_len = 1;       // ASCII
        else if ((b & 0xE0) == 0xC0) expected_len = 2;  // 110xxxxx
        else if ((b & 0xF0) == 0xE0) expected_len = 3;  // 1110xxxx
        else if ((b & 0xF8) == 0xF0) expected_len = 4;  // 11110xxx
        else continue; // continuation byte (10xxxxxx), keep looking back

        if (expected_len > back) {
            // Incomplete multi-byte sequence at the tail; hold it back.
            return n - back;
        }
        break;
    }

    return n;
}

// How many trailing tokens xenon_state keeps for the repeat-penalty window -- must match
// REPEAT_PENALTY_WINDOW above (it's the same window, just persisted across calls instead of
// recomputed from a full prompt every time; see xenon_inference.h's xenon_state docs).
constexpr size_t STATE_RECENT_TOKENS_CAP = REPEAT_PENALTY_WINDOW;

} // namespace

// Everything that used to live directly on xenon_engine (Phase 1-7) now lives here instead, so a
// caller can hold several independent conversations against one loaded engine (Phase 8). Defined
// before xenon_engine below because xenon_engine's legacy_state member needs a complete type.
struct xenon_state {
    std::vector<float> state;
    std::vector<float> logits;
    // Rolling window of the last STATE_RECENT_TOKENS_CAP tokens actually fed into or generated
    // from this state, in feed order. Kept persistently (rather than recomputed from a full
    // prompt like the old per-call `recent_tokens` local) so repeat-penalty behavior under
    // incremental prefill is identical to a full re-feed every time -- see xenon_prefill.
    std::vector<uint32_t> recent_tokens;
};

struct xenon_engine {
    rwkv_context * ctx = nullptr;
    std::unique_ptr<WorldTokenizer> tokenizer;

    size_t n_vocab = 0;
    size_t n_layer = 0;
    size_t state_len = 0;
    size_t logits_len = 0;

    std::mt19937 rng{std::random_device{}()};

    // Lazily allocated on first use by the legacy xenon_reset_state(engine)/xenon_generate() path
    // only -- see those functions. Nothing in this codebase actually calls xenon_reset_state
    // directly any more (Phase 8 callers use xenon_state_reset on their own xenon_state), but it
    // stays part of the public C API for anyone binding against this header.
    std::unique_ptr<xenon_state> legacy_state;

    ~xenon_engine() {
        if (ctx) rwkv_free(ctx);
    }
};

namespace {

void push_recent_tokens(xenon_state * state, const uint32_t * tokens, size_t count) {
    state->recent_tokens.insert(state->recent_tokens.end(), tokens, tokens + count);
    size_t n = state->recent_tokens.size();
    if (n > STATE_RECENT_TOKENS_CAP) {
        state->recent_tokens.erase(state->recent_tokens.begin(), state->recent_tokens.begin() + (n - STATE_RECENT_TOKENS_CAP));
    }
}

// Tokens per rwkv_eval_sequence call during prefill. XENON_PREFILL_CHUNK overrides it for
// measurement. With RWKV-7's WKV and L2-norm steps native (inference-engine/patches/), bigger
// chunks keep the GPU busy: on the RTX 2060, ~1500 tokens took 7.2 s at 16, 1.7 s at 128 and
// 1.4 s at 256, with ~20 MB more VRAM at 128.
size_t prefill_chunk_size() {
    static const size_t size = [] {
        const char * env = std::getenv("XENON_PREFILL_CHUNK");
        long v = env ? std::strtol(env, nullptr, 10) : 0;
        return v > 0 ? static_cast<size_t>(v) : static_cast<size_t>(128);
    }();
    return size;
}

// Shared by xenon_prefill and xenon_generate_with_state's initial-prompt step: encode `text` and
// evaluate it into `state` (state_in == state_out == state->state.data()), updating cached
// logits and the repeat-penalty window. Returns the encoded token count via `out_token_count`
// (0 if `text` is empty -- caller decides whether that's an error or a no-op).
bool eval_text_into_state(
    xenon_engine * engine,
    xenon_state * state,
    const char * text,
    size_t * out_token_count
) {
    *out_token_count = 0;
    if (!text || text[0] == '\0') {
        return true; // nothing to do; not an error at this layer
    }

    std::vector<uint32_t> tokens;
    try {
        tokens = engine->tokenizer->encode(text);
    } catch (const std::exception & e) {
        set_last_error(std::string("xenon_inference: tokenizer encode failed: ") + e.what());
        return false;
    }

    if (tokens.empty()) {
        return true;
    }

    bool ok = rwkv_eval_sequence_in_chunks(
        engine->ctx,
        tokens.data(),
        tokens.size(),
        /* chunk_size */ prefill_chunk_size(),
        /* state_in */ state->state.data(),
        /* state_out */ state->state.data(),
        /* logits_out */ state->logits.data()
    );
    if (!ok) {
        set_last_error("xenon_inference: rwkv_eval_sequence_in_chunks failed");
        return false;
    }

    push_recent_tokens(state, tokens.data(), tokens.size());
    *out_token_count = tokens.size();

    // Phase 8 measurement instrumentation: prints the actual number of tokens evaluated on this
    // call when XENON_DEBUG_PREFILL_COUNT is set, so before/after prefill-cost claims can be
    // measured directly instead of estimated from prompt character counts. Off by default (zero
    // cost in the hot path otherwise) -- getenv is only called once per eval, not once per token.
    if (std::getenv("XENON_DEBUG_PREFILL_COUNT")) {
        fprintf(stderr, "[xenon_inference] evaluated %zu token(s)\n", tokens.size());
    }

    return true;
}

// Shared generation loop, used by both xenon_generate_with_state and (via a scratch state)
// xenon_generate. Assumes `state->logits` already holds valid logits for the next token to
// sample (i.e. something -- prefill or a previous generate call -- has evaluated at least one
// token into `state`).
xenon_status run_generation_loop(
    xenon_engine * engine,
    xenon_state * state,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
) {
    std::string pending_utf8; // holds back incomplete multi-byte UTF-8 sequences

    for (int i = 0; i < max_tokens; i++) {
        uint32_t token = sample_logits(
            state->logits.data(), engine->n_vocab, temperature, top_p,
            repeat_penalty, state->recent_tokens, engine->rng
        );

        // Token 0 is RWKV World's end-of-text: the model has finished. It decodes to nothing, so
        // sampling past it starts a new, unrelated "document" (often an invented next user turn).
        // Flush held-back bytes, then send an empty token-0 call as the end marker.
        if (token == 0) {
            if (!pending_utf8.empty()) {
                callback(pending_utf8.c_str(), XENON_END_OF_TEXT, user_data);
            }
            callback("", XENON_END_OF_TEXT, user_data);
            return XENON_OK;
        }

        push_recent_tokens(state, &token, 1);

        const std::string & token_bytes = engine->tokenizer->decode_token(token);
        pending_utf8 += token_bytes;

        size_t safe_len = utf8_safe_prefix_len(pending_utf8);
        std::string to_emit = pending_utf8.substr(0, safe_len);
        pending_utf8.erase(0, safe_len);

        bool keep_going = callback(to_emit.c_str(), token, user_data);
        if (!keep_going) {
            return XENON_OK;
        }

        if (i + 1 < max_tokens) {
            bool ok = rwkv_eval(
                engine->ctx,
                token,
                state->state.data(),
                state->state.data(),
                state->logits.data()
            );

            if (!ok) {
                set_last_error("xenon_inference: rwkv_eval failed at token " + std::to_string(i));
                return XENON_ERROR_EVAL;
            }
        }
    }

    // Flush any leftover (possibly-invalid) tail bytes so callers don't silently lose them.
    if (!pending_utf8.empty()) {
        callback(pending_utf8.c_str(), 0, user_data);
    }

    return XENON_OK;
}

} // namespace

extern "C" {

XENON_API xenon_engine * xenon_load_model(
    const char * model_path,
    const char * vocab_path,
    uint32_t n_threads,
    uint32_t n_gpu_layers
) {
    if (!model_path || !vocab_path || n_threads == 0) {
        set_last_error("xenon_load_model: invalid arguments (model_path/vocab_path must be non-null, n_threads >= 1)");
        return nullptr;
    }

    rwkv_set_print_errors(nullptr, true);

    rwkv_context * ctx = rwkv_init_from_file(model_path, n_threads, n_gpu_layers);
    if (!ctx) {
        set_last_error(std::string("xenon_load_model: rwkv_init_from_file failed for ") + model_path);
        return nullptr;
    }

    auto engine = std::make_unique<xenon_engine>();
    engine->ctx = ctx;

    try {
        engine->tokenizer = std::make_unique<WorldTokenizer>(vocab_path);
    } catch (const std::exception & e) {
        set_last_error(std::string("xenon_load_model: ") + e.what());
        return nullptr;
    }

    engine->n_vocab    = rwkv_get_n_vocab(ctx);
    engine->n_layer    = rwkv_get_n_layer(ctx);
    engine->state_len  = rwkv_get_state_len(ctx);
    engine->logits_len = rwkv_get_logits_len(ctx);

    // The World vocab file only assigns ids 1..65529 (id 0 and a handful of trailing ids in the
    // 65536-sized embedding table are reserved/unused), so a small gap between vocab_size() and
    // n_vocab is normal. A LARGE gap (e.g. World's 65536 vs. the 20B/Pile tokenizer's 50277)
    // means the wrong tokenizer was paired with this model -- reject that loudly instead of
    // silently producing garbage output.
    long long vocab_gap = static_cast<long long>(engine->n_vocab) - static_cast<long long>(engine->tokenizer->vocab_size());
    if (vocab_gap > 1000 || vocab_gap < -1000) {
        set_last_error(
            "xenon_load_model: vocab file entry count (" + std::to_string(engine->tokenizer->vocab_size()) +
            ") is wildly different from the model's n_vocab (" + std::to_string(engine->n_vocab) +
            "). This model almost certainly needs a different tokenizer (e.g. a Pile/20B-tokenizer model " +
            "loaded with the World tokenizer, or vice versa) -- output would be garbage, refusing to proceed."
        );
        return nullptr;
    }

    return engine.release();
}

XENON_API void xenon_free_engine(xenon_engine * engine) {
    delete engine;
}

// Phase 8 follow-up: xenon_engine no longer owns a state at all (see the struct above), so this
// legacy entry point now manages a single lazily-allocated scratch xenon_state internally --
// good enough for the two remaining direct callers of the old signature (voice-pipeline's
// ctypes bindings, test_inference.cpp), neither of which ever called this concurrently on the
// same engine from multiple threads (same discipline as before: this whole API isn't
// thread-safe per engine, see xenon_inference.h).
XENON_API void xenon_reset_state(xenon_engine * engine) {
    if (!engine) return;
    if (!engine->legacy_state) {
        engine->legacy_state.reset(xenon_state_new(engine));
    }
    xenon_state_reset(engine, engine->legacy_state.get());
}

XENON_API xenon_state * xenon_state_new(xenon_engine * engine) {
    if (!engine) return nullptr;
    auto state = std::make_unique<xenon_state>();
    state->state.resize(engine->state_len);
    state->logits.resize(engine->logits_len);
    return state.release();
}

XENON_API void xenon_state_free(xenon_state * state) {
    delete state;
}

XENON_API void xenon_state_copy(xenon_state * dst, const xenon_state * src) {
    if (!dst || !src) return;
    dst->state = src->state;
    dst->logits = src->logits;
    dst->recent_tokens = src->recent_tokens;
}

XENON_API void xenon_state_reset(xenon_engine * engine, xenon_state * state) {
    if (!engine || !state) return;
    rwkv_init_state(engine->ctx, state->state.data());
    std::fill(state->logits.begin(), state->logits.end(), 0.0f);
    state->recent_tokens.clear();
}

XENON_API xenon_status xenon_prefill(
    xenon_engine * engine,
    xenon_state * state,
    const char * text
) {
    if (!engine || !state) {
        set_last_error("xenon_prefill: invalid arguments");
        return XENON_ERROR_ARGS;
    }
    size_t n_tokens = 0;
    if (!eval_text_into_state(engine, state, text, &n_tokens)) {
        return XENON_ERROR_EVAL;
    }
    return XENON_OK;
}

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
) {
    if (!engine || !state || !prompt || !callback || max_tokens <= 0) {
        set_last_error("xenon_generate_with_state: invalid arguments");
        return XENON_ERROR_ARGS;
    }

    size_t n_tokens = 0;
    if (!eval_text_into_state(engine, state, prompt, &n_tokens)) {
        return XENON_ERROR_EVAL;
    }
    if (n_tokens == 0) {
        // Matches xenon_generate()'s long-standing behavior: an empty/zero-token prompt is
        // rejected rather than silently generating from whatever logits happen to already be in
        // `state` (which may be all-zero garbage if this is a freshly-reset state that was never
        // primed with xenon_prefill first).
        set_last_error("xenon_generate_with_state: prompt encoded to zero tokens");
        return XENON_ERROR_ARGS;
    }

    return run_generation_loop(engine, state, max_tokens, temperature, top_p, repeat_penalty, callback, user_data);
}

XENON_API xenon_status xenon_generate(
    xenon_engine * engine,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
) {
    if (!engine) {
        set_last_error("xenon_generate: invalid arguments");
        return XENON_ERROR_ARGS;
    }

    // Byte-identical to the pre-Phase-8 body of this function, just reimplemented on top of a
    // scratch xenon_state instead of fields that used to live directly on xenon_engine -- see
    // xenon_inference.h's docs on xenon_generate vs. xenon_generate_with_state.
    xenon_state * scratch = xenon_state_new(engine);
    if (!scratch) {
        set_last_error("xenon_generate: failed to allocate scratch state");
        return XENON_ERROR_ARGS;
    }
    xenon_state_reset(engine, scratch);

    xenon_status status = xenon_generate_with_state(
        engine, scratch, prompt, max_tokens, temperature, top_p, repeat_penalty, callback, user_data
    );

    xenon_state_free(scratch);
    return status;
}

XENON_API size_t xenon_get_state_len(xenon_engine * engine) {
    return engine ? engine->state_len : 0;
}

XENON_API size_t xenon_get_n_layer(xenon_engine * engine) {
    return engine ? engine->n_layer : 0;
}

XENON_API int xenon_has_gpu_support(void) {
    return XENON_HAS_CUDA ? 1 : 0;
}

XENON_API const char * xenon_get_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
