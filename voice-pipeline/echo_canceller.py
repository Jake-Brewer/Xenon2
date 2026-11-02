"""WebRTC audio processing for full-duplex voice chat.

The mic stays open while Xenon speaks, so it hears Xenon. Playback feeds what is being played
into `render`, and capture runs through `capture`, which subtracts it. The same pass suppresses
noise and applies automatic gain, so speech reaches Silero and Whisper at a steady level.

WebRTC processes exactly 10 ms at a time, so both directions buffer into 160-sample frames.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160  # 10 ms
SPEAKING_HOLD_SEC = 1.0  # room echo outlasts the last rendered frame


def _to_pcm(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


class EchoCanceller:
    def __init__(self) -> None:
        from livekit import rtc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True,
        )
        self._lock = threading.Lock()
        self._render_tail = np.zeros(0, dtype=np.int16)
        self._capture_tail = np.zeros(0, dtype=np.int16)
        self._delay_ms = 100
        self._last_render = 0.0

    def _frame(self, pcm: np.ndarray):
        return self._rtc.AudioFrame(
            data=pcm.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=FRAME_SAMPLES
        )

    def set_delay(self, seconds: float) -> None:
        """Time between a frame being rendered and its echo reaching `capture`."""
        self._delay_ms = max(0, int(seconds * 1000))

    @property
    def speaking(self) -> bool:
        return time.monotonic() - self._last_render < SPEAKING_HOLD_SEC

    def render(self, samples: np.ndarray) -> None:
        """Float 16 kHz audio about to be played. Call at the pace it plays, not all at once."""
        with self._lock:
            self._last_render = time.monotonic()
            pcm = np.concatenate((self._render_tail, _to_pcm(samples)))
            usable = len(pcm) - len(pcm) % FRAME_SAMPLES
            for start in range(0, usable, FRAME_SAMPLES):
                self._apm.process_reverse_stream(self._frame(pcm[start : start + FRAME_SAMPLES]))
            self._render_tail = pcm[usable:]

    def capture(self, samples: np.ndarray) -> np.ndarray:
        """Float 16 kHz mic audio in, cleaned float audio out. Holds back any partial frame."""
        with self._lock:
            pcm = np.concatenate((self._capture_tail, _to_pcm(samples)))
            usable = len(pcm) - len(pcm) % FRAME_SAMPLES
            out = []
            for start in range(0, usable, FRAME_SAMPLES):
                frame = self._frame(pcm[start : start + FRAME_SAMPLES])
                self._apm.set_stream_delay_ms(self._delay_ms)
                self._apm.process_stream(frame)
                out.append(np.frombuffer(frame.data, dtype=np.int16).copy())
            self._capture_tail = pcm[usable:]
        if not out:
            return np.zeros(0, dtype=np.float32)
        return (np.concatenate(out).astype(np.float32) / 32768.0).astype(np.float32)


_shared: Optional[EchoCanceller] = None
_shared_failed = False
_shared_lock = threading.Lock()


def shared() -> Optional[EchoCanceller]:
    """The one canceller for both directions, or None if WebRTC is unavailable."""
    global _shared, _shared_failed
    with _shared_lock:
        if _shared is None and not _shared_failed:
            try:
                _shared = EchoCanceller()
            except Exception as exc:  # noqa: BLE001 -- voice still works without echo cancellation
                _shared_failed = True
                print(f"[audio] echo cancellation unavailable: {exc}", file=sys.stderr, flush=True)
        return _shared
