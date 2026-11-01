"""stt.py

Speech-to-text via faster-whisper (CTranslate2 backend), forced onto GPU per
phase2_prompt.md's explicit split (STT shares the GPU with RWKV inference; do not fall
back to CPU "to keep it simple").

Compatibility note (discovered on this machine, documented rather than worked around
silently): the CTranslate2 4.8.1 wheel on PyPI dynamically links against CUDA 12's
cublas64_12.dll specifically. This dev machine's CUDA *Toolkit* is 13.3 (ships
cublas64_13.dll only), so a bare `pip install ctranslate2` + `device="cuda"` fails at
model-load time with "Library cublas64_12.dll is not found or cannot be loaded" --
CTranslate2 does not bundle its own CUDA runtime for Windows wheels, and does not yet
support linking against CUDA 13's renamed cublas DLL. Fix: install the redistributable
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip packages (these ship the actual
cublas64_12.dll / cudnn64_9.dll files, matching what NVIDIA's own CUDA 12.x installer
would have put on disk) and add their package directory to the DLL search path before
importing ctranslate2/faster_whisper. This does NOT touch or downgrade the system CUDA
13.3 Toolkit used by Phase 1's rwkv.cpp CUDA build -- both coexist independently:
rwkv.dll (Phase 1) is compiled against and loads cublas64_13.dll from the Toolkit;
ctranslate2 (Phase 2) loads cublas64_12.dll from the redistributable pip package. See
`_ensure_cuda12_dlls()` below.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from hardware import physical_cpus

SAMPLE_RATE = 16000

_cuda12_dlls_added = False


def _ensure_cuda12_dlls() -> None:
    """Add nvidia-cublas-cu12 / nvidia-cudnn-cu12's bundled DLL directories to the
    process DLL search path, if those pip packages are installed. Must run before the
    first `import ctranslate2` (or before constructing a WhisperModel, since ctranslate2
    resolves its CUDA deps lazily at model-load time)."""
    global _cuda12_dlls_added
    if _cuda12_dlls_added:
        return

    def _pkg_bin_dir(pkg) -> Optional[str]:
        # nvidia.cublas / nvidia.cudnn are PEP 420 namespace packages (no __init__.py),
        # so they have __file__ = None -- __path__ (a list of directories) is what's
        # actually populated and must be used instead.
        for root in list(getattr(pkg, "__path__", []) or []):
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                return bin_dir
        return None

    found_dirs = []
    try:
        import nvidia.cublas as _cublas_pkg
        cublas_dir = _pkg_bin_dir(_cublas_pkg)
        if cublas_dir:
            found_dirs.append(cublas_dir)
    except ImportError:
        pass
    try:
        import nvidia.cudnn as _cudnn_pkg
        cudnn_dir = _pkg_bin_dir(_cudnn_pkg)
        if cudnn_dir:
            found_dirs.append(cudnn_dir)
    except ImportError:
        pass

    for d in found_dirs:
        # os.add_dll_directory() alone was observed to be insufficient here -- ctranslate2's
        # native extension still failed with "Library cublas64_12.dll is not found or cannot
        # be loaded" even with the directory registered via AddDllDirectory, likely because
        # its delay-loaded CUDA dependencies resolve via the plain DLL search order (which
        # honors PATH but not necessarily AddDllDirectory-registered dirs for every loader).
        # Prepending to PATH as well is what actually made it load, so do both -- belt and
        # suspenders, and harmless either way.
        os.add_dll_directory(d)
    if found_dirs:
        os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")

    _cuda12_dlls_added = True


_ensure_cuda12_dlls()

from faster_whisper import WhisperModel  # noqa: E402  (import after DLL path setup)


@dataclass
class SttResult:
    text: str
    device: str  # what we asked for
    actual_device: str  # what CTranslate2 reports it's actually using
    compute_type: str
    transcribe_sec: float
    language: str
    language_probability: float


class SpeechToText:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.requested_device = device
        self.model_size = model_size
        self.compute_type = compute_type
        # The GPU path still runs the feature extractor on the CPU. One worker
        # per physical core; hyperthreads make that extractor and RWKV slower.
        self.cpu_threads = physical_cpus()

        t0 = time.perf_counter()
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=1,
        )
        self.load_sec = time.perf_counter() - t0

        # Confirm the actual execution device CTranslate2 is using -- do not just trust
        # that device="cuda" silently succeeded, per the acceptance criteria. CTranslate2's
        # Whisper wrapper exposes this via model.model.device (the underlying ctranslate2
        # Whisper/ctc model object), which reflects where it actually ended up, not just
        # the constructor arg.
        try:
            self.actual_device = self.model.model.device
        except AttributeError:
            self.actual_device = "unknown"

        if device == "cuda" and self.actual_device != "cuda":
            raise RuntimeError(
                f"Requested device='cuda' for faster-whisper but CTranslate2 reports it "
                f"actually loaded on device={self.actual_device!r} -- this would silently "
                f"defeat the GPU/CPU split the spec requires. Failing loudly instead of "
                f"continuing on the wrong device."
            )

        # IMPORTANT: `self.model.model.device` above only reflects the *requested* config at
        # construction time -- it does NOT prove CUDA execution actually works. CTranslate2
        # resolves its CUDA library dependencies (cublas64_12.dll etc.) lazily, on the first
        # real inference call, not at model-load time. Discovered the hard way on this
        # machine: construction succeeded and reported actual_device="cuda" even while the
        # *first transcribe() call* was about to fail with "Library cublas64_12.dll is not
        # found or cannot be loaded" (see _ensure_cuda12_dlls() above for why/how that's
        # fixed). So: force that lazy load to happen right here with a real (short, silent)
        # inference call, and let any CUDA failure surface immediately and loudly at
        # construction time rather than on the first real utterance.
        t0 = time.perf_counter()
        _warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        list(self.model.transcribe(_warm_audio, language="en", beam_size=1, vad_filter=False)[0])
        self.warmup_sec = time.perf_counter() - t0

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SttResult:
        """audio: float32 mono numpy array in [-1, 1] at `sample_rate` Hz (faster-whisper's
        feature extractor expects 16kHz; resample before calling if audio came from
        elsewhere at a different rate)."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"faster-whisper expects {SAMPLE_RATE} Hz audio, got {sample_rate}")

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            # Silero already decided this clip contains speech. Whisper's own
            # no-speech filter was discarding quiet headset audio.
            no_speech_threshold=None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        transcribe_sec = time.perf_counter() - t0

        return SttResult(
            text=text,
            device=self.requested_device,
            actual_device=self.actual_device,
            compute_type=self.compute_type,
            transcribe_sec=transcribe_sec,
            language=info.language,
            language_probability=info.language_probability,
        )
