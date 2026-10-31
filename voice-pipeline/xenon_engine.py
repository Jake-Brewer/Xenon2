"""xenon_engine.py

Thin ctypes bridge to Phase 1's xenon_inference DLL (see
inference-engine/src/xenon_inference.h). This is the *only* place Phase 2 talks to the
RWKV engine -- everything else in the voice pipeline (VAD/STT/TTS) feeds plain text
into `XenonEngine.generate()`, exactly like typed input would (per phase2_prompt.md's
scope note: voice is additive, it converges on the same generate() entry point).

Loads xenon_inference.dll from the CUDA-enabled build (inference-engine/build-cuda-app)
so RWKV inference runs GPU-offloaded, matching Phase 1's validated GPU path and this
phase's GPU/CPU split (GPU: RWKV + Whisper STT; CPU: VAD + TTS).
"""
from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- locate repo paths -------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

def _default_lib_path() -> str:
    """CUDA build, Windows DLL layout or Linux libxenon_inference.so layout."""
    base = os.path.join(REPO_ROOT, "inference-engine", "build-cuda-app")
    candidates = [
        os.path.join(base, "bin", "Release", "xenon_inference.dll"),
        os.path.join(base, "bin", "libxenon_inference.so"),
        os.path.join(base, "libxenon_inference.so"),
        os.path.join(base, "bin", "Release", "libxenon_inference.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


DEFAULT_DLL_PATH = _default_lib_path()
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "rwkv-5-world-0.4B-Q4_0.bin")
DEFAULT_VOCAB_PATH = os.path.join(REPO_ROOT, "inference-engine", "data", "world_vocab.bin")

# --- ctypes signatures, mirroring xenon_inference.h ---------------------------------
XENON_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p
)


class _XenonStatus:
    OK = 0
    ERROR_ARGS = 1
    ERROR_MODEL_LOAD = 2
    ERROR_VOCAB_LOAD = 3
    ERROR_EVAL = 4


def _load_lib(dll_path: str) -> ctypes.CDLL:
    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"xenon_inference.dll not found at {dll_path!r}. Build inference-engine's "
            f"build-cuda-app target first (see inference-engine/README.md)."
        )
    # rwkv.dll / ggml*.dll / cublas64_*.dll live alongside xenon_inference.dll -- add that
    # directory to the DLL search path so LoadLibrary can resolve them (Windows does not
    # search the loading DLL's own directory for its dependencies by default in all configs).
    dll_dir = os.path.dirname(dll_path)
    try:
        os.add_dll_directory(dll_dir)  # py3.8+
    except (AttributeError, FileNotFoundError, OSError):
        pass
    # Linux: load librwkv.so from the same directory before libxenon_inference.so needs it.
    rwkv_so = os.path.join(dll_dir, "librwkv.so")
    if os.path.exists(rwkv_so):
        ctypes.CDLL(rwkv_so, mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(dll_path)


def _configure_signatures(lib: ctypes.CDLL) -> None:
    lib.xenon_load_model.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32
    ]
    lib.xenon_load_model.restype = ctypes.c_void_p

    lib.xenon_free_engine.argtypes = [ctypes.c_void_p]
    lib.xenon_free_engine.restype = None

    lib.xenon_reset_state.argtypes = [ctypes.c_void_p]
    lib.xenon_reset_state.restype = None

    lib.xenon_generate.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, XENON_TOKEN_CALLBACK, ctypes.c_void_p,
    ]
    lib.xenon_generate.restype = ctypes.c_int

    lib.xenon_get_state_len.argtypes = [ctypes.c_void_p]
    lib.xenon_get_state_len.restype = ctypes.c_size_t

    lib.xenon_get_n_layer.argtypes = [ctypes.c_void_p]
    lib.xenon_get_n_layer.restype = ctypes.c_size_t

    lib.xenon_has_gpu_support.argtypes = []
    lib.xenon_has_gpu_support.restype = ctypes.c_int

    lib.xenon_get_last_error.argtypes = []
    lib.xenon_get_last_error.restype = ctypes.c_char_p


CHAT_PRIMER = (
    "The following is a coherent, friendly conversation between a user and Xenon, a "
    "helpful voice assistant.\n\n"
    "User: Hello Xenon, how are you doing?\n\n"
    "Xenon: Hi! I'm doing well, thanks for asking. How can I help you today?\n\n"
)


def build_chat_prompt(user_message: str) -> str:
    """Matches inference-engine/src/test_inference.cpp's build_chat_prompt so Phase 2's
    voice path produces the same style of response Phase 1's CLI harness validated."""
    return CHAT_PRIMER + "User: " + user_message + "\n\nXenon:"


STOP_SEQUENCES = ("\n\nUser:", "\n\nuser:")


@dataclass
class GenerateResult:
    text: str
    tokens_emitted: int
    ttft_sec: float
    total_sec: float
    gpu_layers: int
    has_gpu_support: bool


class XenonEngine:
    """Wraps one loaded xenon_inference engine handle. GPU-offloaded by default (n_gpu_layers>0)
    to match this phase's GPU/CPU split: RWKV inference + Whisper STT share the GPU."""

    def __init__(
        self,
        dll_path: str = DEFAULT_DLL_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        vocab_path: str = DEFAULT_VOCAB_PATH,
        n_threads: int = 6,
        # rwkv.cpp clamps this to n_layer+1. 999 offloads every layer onto the GPU.
        # An RTX 2060 (6GB) holds the Q5_1 2.9B weights (~2.75GB) plus the ~21MB state.
        n_gpu_layers: int = 999,
    ):
        self.lib = _load_lib(dll_path)
        _configure_signatures(self.lib)

        self.has_gpu_support = bool(self.lib.xenon_has_gpu_support())
        if n_gpu_layers > 0 and not self.has_gpu_support:
            print(
                "This xenon_inference build has no CUDA support. Loading on CPU.",
                file=__import__("sys").stderr,
            )
            n_gpu_layers = 0

        self.n_gpu_layers = n_gpu_layers

        t0 = time.perf_counter()
        self._engine = self.lib.xenon_load_model(
            model_path.encode("utf-8"), vocab_path.encode("utf-8"), n_threads, n_gpu_layers
        )
        if not self._engine and n_gpu_layers > 0:
            err = self.lib.xenon_get_last_error()
            err_text = err.decode("utf-8", errors="replace") if err else ""
            print(
                f"CUDA offload failed ({err_text}). Loading on CPU instead.",
                file=__import__("sys").stderr,
            )
            self.n_gpu_layers = 0
            self._engine = self.lib.xenon_load_model(
                model_path.encode("utf-8"), vocab_path.encode("utf-8"), n_threads, 0
            )
        self.load_sec = time.perf_counter() - t0

        if not self._engine:
            err = self.lib.xenon_get_last_error().decode("utf-8", errors="replace")
            raise RuntimeError(f"xenon_load_model failed: {err}")

        self.n_layer = self.lib.xenon_get_n_layer(self._engine)
        self.state_len = self.lib.xenon_get_state_len(self._engine)

    def generate(
        self,
        user_message: str,
        max_tokens: int = 80,
        temperature: float = 0.8,
        top_p: float = 0.5,
        repeat_penalty: float = 1.3,  # matches app/src-tauri/src/inference.rs's tuned value
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> GenerateResult:
        """Streams a response to user_message via the same generate() entry point typed
        input would use (build_chat_prompt formatting matches Phase 1's CLI harness).
        Calls on_partial(text_fragment) as each streamed fragment arrives -- callers (e.g.
        the voice pipeline's incremental TTS feeder) can consume text before generation
        finishes, matching the "do not wait for a full batched response" requirement."""
        prompt = build_chat_prompt(user_message)

        state = {
            "chunks": [],
            "tail": "",
            "tokens": 0,
            "t_first": None,
        }

        def _cb(text_ptr, token_id, user_data):
            text = text_ptr.decode("utf-8", errors="ignore") if text_ptr else ""
            if text:
                if state["t_first"] is None:
                    state["t_first"] = time.perf_counter()
                state["chunks"].append(text)
                state["tail"] = (state["tail"] + text)[-64:]
                if on_partial is not None:
                    on_partial(text)
            state["tokens"] += 1
            for stop in STOP_SEQUENCES:
                if state["tail"].endswith(stop):
                    return False
            return True

        cb = XENON_TOKEN_CALLBACK(_cb)
        t_call_start = time.perf_counter()
        status = self.lib.xenon_generate(
            self._engine, prompt.encode("utf-8"), max_tokens, temperature, top_p,
            repeat_penalty, cb, None
        )
        t_call_end = time.perf_counter()

        if status != _XenonStatus.OK:
            err = self.lib.xenon_get_last_error().decode("utf-8", errors="replace")
            raise RuntimeError(f"xenon_generate failed (status={status}): {err}")

        full_text = "".join(state["chunks"])
        for stop in STOP_SEQUENCES:
            if full_text.endswith(stop):
                full_text = full_text[: -len(stop)]

        t_first = state["t_first"] or t_call_end
        return GenerateResult(
            text=full_text.strip(),
            tokens_emitted=state["tokens"],
            ttft_sec=t_first - t_call_start,
            total_sec=t_call_end - t_call_start,
            gpu_layers=self.n_gpu_layers,
            has_gpu_support=self.has_gpu_support,
        )

    def close(self):
        if getattr(self, "_engine", None):
            self.lib.xenon_free_engine(self._engine)
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick standalone smoke test: python xenon_engine.py "hello, how are you?"
    import sys

    msg = sys.argv[1] if len(sys.argv) > 1 else "hello, how are you?"
    with XenonEngine() as engine:
        print(f"GPU support compiled in: {engine.has_gpu_support}", file=__import__("sys").stderr)
        print(f"n_layer={engine.n_layer} state_len={engine.state_len} load={engine.load_sec:.3f}s",
              file=__import__("sys").stderr)
        result = engine.generate(msg, on_partial=lambda t: print(t, end="", flush=True))
        print()
        print(f"ttft={result.ttft_sec:.3f}s total={result.total_sec:.3f}s "
              f"tokens={result.tokens_emitted} gpu_layers={result.gpu_layers}", file=__import__("sys").stderr)
