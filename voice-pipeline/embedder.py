"""Sentence embeddings for Sloth's memory, so facts are found by meaning, not shared words.

Runs all-MiniLM-L6-v2 (384 dimensions) on the CPU with ONNX Runtime and the `tokenizers`
package, both already installed for Piper and faster-whisper, so this adds no dependencies. The
model is downloaded once from Hugging Face (about 90 MB) and works offline after that.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from hardware import physical_cpus

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_TOKENS = 256


class Embedder:
    def __init__(self) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(MODEL_REPO, "onnx/model.onnx")
        tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, physical_cpus() // 2)
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Unit-length vectors, one row per text, so a dot product is the cosine similarity."""
        if not texts:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self._lock:
            hidden = self._session.run(None, feed)[0]
        # Mean of the token vectors, ignoring padding, then normalized: the model's own pooling.
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)


_shared: Optional[Embedder] = None
_shared_error: Optional[str] = None
_ready = threading.Event()


def load_in_background(log) -> None:
    """Starts loading at sidecar startup, so the first Sloth turn does not wait on a download."""

    def _load() -> None:
        global _shared, _shared_error
        try:
            _shared = Embedder()
            log("embeddings ready")
        except Exception as exc:  # noqa: BLE001 -- Sloth falls back to keyword search
            _shared_error = str(exc)
            log(f"embeddings unavailable ({exc})")
        finally:
            _ready.set()

    threading.Thread(target=_load, daemon=True, name="embedder-load").start()


def embed(texts: List[str], timeout: float = 120.0) -> np.ndarray:
    if not _ready.wait(timeout):
        raise RuntimeError("embedding model is still loading")
    if _shared is None:
        raise RuntimeError(_shared_error or "embedding model unavailable")
    return _shared.embed(texts)
