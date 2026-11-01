"""vad.py

Voice activity detection via silero-vad, forced onto CPU per phase2_prompt.md's
deliberate GPU/CPU split (VAD is cheap/lightweight -- GPU is reserved for RWKV +
Whisper). silero-vad ships as a torch model; the torch install in this venv is the
CPU-only build (torch==2.13.0+cpu, verified via `torch.cuda.is_available()` returning
False), so there is no accidental GPU fallback possible here even if requested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from hardware import physical_cpus
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000


@dataclass
class VadResult:
    speech_detected: bool
    segments: List[dict]  # [{'start': sec, 'end': sec}, ...] in seconds
    device: str
    detect_sec: float


class VoiceActivityDetector:
    def __init__(self):
        # CPU-only torch on purpose (torch==...+cpu). One worker per physical core
        # so VAD does not start a hyperthread pool beside Whisper and Piper.
        self.cpu_threads = physical_cpus()
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        t0 = time.perf_counter()
        self.model = load_silero_vad()
        self.model.to("cpu")
        self.load_sec = time.perf_counter() - t0
        self.device = "cpu"

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        min_speech_prob: float = 0.5,
    ) -> VadResult:
        """audio: float32 mono numpy array in [-1, 1]. Returns speech segment timestamps
        (seconds) plus whether any speech was found at all -- callers use this to gate
        whether STT should run, and to bound the utterance end so STT knows when to stop
        waiting for more audio."""
        t0 = time.perf_counter()
        wav_t = torch.from_numpy(audio.astype(np.float32))
        segments = get_speech_timestamps(
            wav_t,
            self.model,
            sampling_rate=sample_rate,
            return_seconds=True,
            threshold=min_speech_prob,
        )
        detect_sec = time.perf_counter() - t0
        return VadResult(
            speech_detected=len(segments) > 0,
            segments=segments,
            device=self.device,
            detect_sec=detect_sec,
        )
