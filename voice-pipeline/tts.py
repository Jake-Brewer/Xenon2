"""tts.py

Text-to-speech via piper-tts, CPU-only -- there's no meaningful GPU acceleration path
for piper per phase2_prompt.md (it does expose a --cuda flag for its onnxruntime
session, but the spec is explicit that TTS belongs on CPU in this phase's split, and
piper's ONNX models are small enough that CPU synthesis for short responses is cheap).
The session uses one worker per physical core so Piper does not start a hyperthread
pool while RWKV is generating on the same machine.
"""
from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

from hardware import physical_cpus
from piper import PiperVoice

DEFAULT_VOICE_MODEL = "models/en_GB-alan-medium.onnx"
DEFAULT_VOICE_CONFIG = "models/en_GB-alan-medium.onnx.json"


def _limit_onnx_threads(threads: int) -> None:
    import onnxruntime

    if getattr(onnxruntime, "_xenon_threads_patched", False):
        return
    original = onnxruntime.InferenceSession

    def session_with_threads(*args, **kwargs):
        opts = kwargs.get("sess_options")
        if opts is None:
            opts = onnxruntime.SessionOptions()
            kwargs["sess_options"] = opts
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        return original(*args, **kwargs)

    onnxruntime.InferenceSession = session_with_threads
    onnxruntime._xenon_threads_patched = True


@dataclass
class TtsResult:
    audio: np.ndarray  # float32 mono, [-1, 1]
    sample_rate: int
    device: str
    synth_sec: float


class TextToSpeech:
    def __init__(self, model_path: str = DEFAULT_VOICE_MODEL, config_path: str = DEFAULT_VOICE_CONFIG,
                 use_cuda: bool = False):
        self.device = "cpu" if not use_cuda else "cuda"
        # Piper has no thread argument. Its session uses every logical processor
        # unless this is set first. intra-op on the physical cores, one inter-op
        # thread, so synthesis during a reply does not oversubscribe the CPU.
        self.cpu_threads = physical_cpus()
        _limit_onnx_threads(self.cpu_threads)
        t0 = time.perf_counter()
        self.voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
        self.load_sec = time.perf_counter() - t0
        # piper warms up its onnxruntime session lazily on first synthesis call (observed
        # ~5-6s on this machine for the first call vs ~0.06-0.1s steady-state) -- warm it up
        # here so callers measuring round-trip latency aren't penalized by one-time session
        # init the same way model *loading* already isn't penalized by first-run costs.
        self._warm_up()

    def _warm_up(self):
        t0 = time.perf_counter()
        list(self.voice.synthesize("Hello."))
        self.warmup_sec = time.perf_counter() - t0

    def synthesize(self, text: str) -> TtsResult:
        if not text.strip():
            return TtsResult(audio=np.zeros(0, dtype=np.float32), sample_rate=self.voice.config.sample_rate,
                              device=self.device, synth_sec=0.0)

        t0 = time.perf_counter()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize_wav(text, wf)
        synth_sec = time.perf_counter() - t0

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()

        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unexpected piper output sample width: {sampwidth}")

        return TtsResult(audio=audio, sample_rate=sr, device=self.device, synth_sec=synth_sec)
