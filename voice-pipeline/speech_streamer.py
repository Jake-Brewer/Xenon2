"""speech_streamer.py

Bridges Phase 1's streamed generate() output to piper TTS incrementally, instead of
waiting for the full response text before starting synthesis -- this is a direct
requirement from phase2_prompt.md's background section ("your voice pipeline should
feed text into generate() incrementally and consume its streamed output... not wait for
a full batched response before starting TTS").

Design: `xenon_generate()` is one blocking FFI call from Python's side (the whole
generation loop runs inside the C library; on_partial fires synchronously on that same
call stack once per token). So true token-level TTS isn't practical -- synthesizing
audio for a half-formed word makes no sense anyway. Instead this buffers streamed text
and flushes it to TTS at each *sentence* boundary as soon as one completes, which is the
smallest unit that both (a) makes sense to speak and (b) lets synthesis of sentence N
start while the model is still generating sentence N+1 -- satisfying "don't wait for the
full response" without trying to synthesize sub-word fragments.

TTS synthesis for the completed sentence happens synchronously inside the streaming
callback (CPU, typically <0.1-0.3s for a short sentence -- see README timings), briefly
pausing token generation. Playback of that sentence's audio happens on a separate
background thread, so it does NOT block generation of the next sentence -- GPU token
generation for sentence N+1 proceeds while sentence N is still being spoken through the
speakers. This is what makes the pipeline's user-facing "time to first spoken audio"
latency independent of the response's total spoken length.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from audio_io import play_audio
from tts import TextToSpeech

# Split on sentence-ending punctuation followed by whitespace (or end of string). Simple
# and not abbreviation-aware (matches Phase 1's own CLI harness's philosophy of using
# straightforward heuristics rather than a full NLP sentence splitter for a small
# assistant model's short replies).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SpokenChunk:
    text: str
    synth_sec: float
    audio_sec: float
    queued_at: float


@dataclass
class IncrementalSpeechResult:
    chunks: List[SpokenChunk] = field(default_factory=list)
    time_to_first_audio_sec: Optional[float] = None  # from stream start to 1st chunk's playback start
    total_tts_synth_sec: float = 0.0
    all_playback_done_sec: Optional[float] = None  # from stream start to last chunk finishing playback
    # Absolute time.perf_counter() values (not relative to this speaker's own construction),
    # so callers can measure latency from an earlier reference point of their own (e.g.
    # pipeline.py wants latency from "utterance received", which predates generate()/this
    # speaker's construction).
    first_audio_abs_time: Optional[float] = None
    all_playback_done_abs_time: Optional[float] = None


class IncrementalSpeaker:
    """Accepts streamed text fragments (call `.feed(fragment)` per generate() callback
    invocation), synthesizes completed sentences via piper as soon as they're available,
    and plays them back in order on a background thread so playback of one sentence
    overlaps with synthesis/generation of the next."""

    def __init__(self, tts: TextToSpeech, speak: bool = True):
        self.tts = tts
        self.speak = speak
        self._buf = ""
        self._t_start = time.perf_counter()
        self._first_audio_started_at: Optional[float] = None
        self._first_audio_lock = threading.Lock()
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._sample_rate = getattr(tts.voice.config, "sample_rate", 22050)
        self._chunks: List[SpokenChunk] = []
        self._total_synth_sec = 0.0
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()
        # Sentence text is queued here so synthesis does not block the sidecar
        # command loop. Mic capture can run while a reply is still being spoken.
        self._text_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._text_worker = threading.Thread(target=self._text_loop, daemon=True)
        self._text_worker.start()
        self._last_chunk_done_event = threading.Event()
        self._pending = 0
        self._pending_lock = threading.Lock()

    def _playback_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            audio = item
            with self._first_audio_lock:
                if self._first_audio_started_at is None:
                    self._first_audio_started_at = time.perf_counter()
            if self.speak and audio.size > 0:
                play_audio(audio, self._sample_rate, block=True)
            with self._pending_lock:
                self._pending -= 1
                if self._pending == 0:
                    self._last_chunk_done_event.set()

    def _flush_sentence(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return
        tts_result = self.tts.synthesize(sentence)
        self._total_synth_sec += tts_result.synth_sec
        self._chunks.append(SpokenChunk(
            text=sentence, synth_sec=tts_result.synth_sec,
            audio_sec=len(tts_result.audio) / max(tts_result.sample_rate, 1),
            queued_at=time.perf_counter() - self._t_start,
        ))
        with self._pending_lock:
            self._pending += 1
            self._last_chunk_done_event.clear()
        self._queue.put(tts_result.audio)

    def feed(self, fragment: str):
        """Queue one streamed text fragment. Synthesis happens off the caller thread."""
        if fragment:
            self._text_q.put(fragment)

    def _text_loop(self):
        while True:
            item = self._text_q.get()
            if item is None:
                if self._buf.strip():
                    self._flush_sentence(self._buf)
                    self._buf = ""
                self._queue.put(None)
                return
            self._buf += item
            parts = _SENTENCE_END_RE.split(self._buf)
            if len(parts) > 1:
                for complete in parts[:-1]:
                    self._flush_sentence(complete)
                self._buf = parts[-1]

    def finish(self, wait_for_playback: bool = True) -> IncrementalSpeechResult:
        """Call after generate() returns. Flushes any remaining buffered text (even
        without terminal punctuation) as a final chunk, then optionally blocks until all
        queued audio has finished playing (useful for a CLI harness that shouldn't exit
        mid-speech; NOT required for the "time to first audio" latency metric, which is
        already fixed by the time this runs)."""
        self._text_q.put(None)
        self._text_worker.join(timeout=30)

        if wait_for_playback and self._chunks:
            # Long enough for a full 200-token reply. Hands-free listening resumes after this.
            self._last_chunk_done_event.wait(timeout=300)

        self._worker.join(timeout=5)

        t_now = time.perf_counter()
        ttfa = (self._first_audio_started_at - self._t_start) if self._first_audio_started_at else None
        all_done = (t_now - self._t_start) if wait_for_playback else None
        return IncrementalSpeechResult(
            chunks=self._chunks,
            time_to_first_audio_sec=ttfa,
            total_tts_synth_sec=self._total_synth_sec,
            all_playback_done_sec=all_done,
            first_audio_abs_time=self._first_audio_started_at,
            all_playback_done_abs_time=t_now if wait_for_playback else None,
        )
