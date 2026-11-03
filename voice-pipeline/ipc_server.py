"""ipc_server.py -- Phase 7 addition.

Long-lived sidecar process the Tauri app spawns once at startup (see
`app/src-tauri/src/voice.rs`) so the desktop app's mic button can drive this
standalone Phase 2 pipeline without reimplementing VAD/STT/TTS in Rust.

Deliberately does NOT import `xenon_engine.py` / call `generate()` -- the app's
Rust side already owns the one loaded RWKV engine (see `inference.rs`) and the
existing `sendMessage` code path both typed and voice input funnel through
(per phase7_prompt.md task 1: "do not build a parallel/separate code path for
voice-originated messages"). This process's only job is speech <-> text at the
two edges: mic -> VAD -> STT -> transcript (for `listen`), and streamed reply
text -> sentence-chunked TTS -> speakers (for `speak_*`), reusing
`speech_streamer.py`'s `IncrementalSpeaker` exactly as Phase 2 built it -- the
Rust side just forwards each streamed token fragment to `speak_feed` instead
of that fragment reaching an in-process `on_partial` callback the way Phase
2's own CLI harness fed it.

Protocol: newline-delimited JSON on stdin (commands in) / stdout (responses
and async events out). One command per line in, zero or more JSON lines out
per command. stderr is left free for Python tracebacks/library warnings so
they never corrupt the stdout protocol stream.

Commands:
  {"cmd": "ping"}
    -> {"type": "pong"}
  {"cmd": "listen", "id": 1, "maxListenSec": 6.0, "maxUtteranceSec": 10.0}
    -> {"type": "ack", "cmd": "listen"} immediately
       {"type": "listen_result", "id": 1, "ok": true, "text": "..."} later
       {"type": "listen_result", "id": 1, "ok": false, "reason": "..."} later
       {"type": "speech_started", "id": 1} when speech begins, before listen_result
    Capture and transcription run on a worker so speak_* can proceed while the
    mic is open. A second listen while one is running acks, then returns
    reason "busy".
  {"cmd": "listen_cancel"}
    -> {"type": "ack", "cmd": "listen_cancel"}, and the open listen ends with reason "cancelled"
  {"cmd": "speak_start"}
    -> {"type": "ack", "cmd": "speak_start"}
  {"cmd": "speak_feed", "text": "..."}
    -> {"type": "ack", "cmd": "speak_feed"}
    The text is queued; synthesis does not run on this thread.
  {"cmd": "speak_finish"}
    -> {"type": "ack", "cmd": "speak_finish"} (immediately)
       {"type": "speak_done"} (later, once playback actually finishes)
  {"cmd": "shutdown"}
    -> process exits
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback

from hardware import logical_cpus, physical_cpus

# Numeric libraries read these at import. Set them before VAD/STT/TTS load.
_CPU_THREADS = physical_cpus()
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))

import numpy as np  # noqa: E402 -- after the thread limits above

import echo_canceller
import mic_level
from vad import VoiceActivityDetector
from stt import SpeechToText
from tts import TextToSpeech
from audio_io import SAMPLE_RATE, record_from_mic, select_input_device
from speech_streamer import IncrementalSpeaker

PHANTOM_TRANSCRIPTS = {"you", "thank you", "thanks for watching", "thank you for watching"}
ECHO_WORD_SHARE = 0.75  # a transcript this much made of Xenon's own recent words is echo
ECHO_MIN_WORDS = 4
COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "can", "do", "for", "i", "if", "in", "is",
    "it", "its", "it's", "me", "my", "no", "not", "of", "on", "or", "so", "that", "the", "this",
    "to", "was", "we", "what", "would", "yes", "you", "your",
}
# A headset mic barely hears Xenon. On laptop speakers, echo cancellation leaves enough behind
# that Whisper invents phrases and misses the user, so speech captured over a reply is dropped.
HEADSET_WORDS = ("headset", "headphone", "earphone", "openrun")

_stdout_lock = threading.Lock()


def send(obj: dict) -> None:
    with _stdout_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    # Diagnostics go to stderr only -- never mixed into the stdout JSON stream.
    print(f"[ipc_server] {msg}", file=sys.stderr, flush=True)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class VoiceServer:
    def __init__(self):
        log(f"voice threads: {_CPU_THREADS} physical / {logical_cpus()} logical")
        log("loading VAD...")
        self.vad = VoiceActivityDetector()
        log("loading STT (faster-whisper)...")
        self.stt = self._load_stt()
        log("loading TTS (piper)...")
        self.tts = TextToSpeech()
        try:
            index, rate, name = select_input_device()
            log(f"input device: {name} #{index} {rate} Hz")
        except Exception as exc:  # noqa: BLE001 -- device choice is diagnostic, not fatal
            log(f"input device selection failed: {exc}")
        log("ready")

        self._speaker: IncrementalSpeaker | None = None
        self._speaker_lock = threading.Lock()
        self._listen_lock = threading.Lock()
        self._listen_thread: threading.Thread | None = None
        self._listen_cancel = threading.Event()
        self._canceller = echo_canceller.shared()
        log(f"echo cancellation: {'on' if self._canceller else 'off'}")
        self._spoken_words: set[str] = set()

    def _load_stt(self) -> SpeechToText:
        """turbo uses about 1.8 GB on the GPU and fits beside RWKV on a 6 GB card. A tighter
        GPU falls back to small, and a machine with no CUDA device falls back to the CPU."""
        for model_size, device, compute_type in (
            ("large-v3-turbo", "cuda", "float16"),
            ("small", "cuda", "float16"),
            ("small", "cpu", "int8"),
        ):
            try:
                stt = SpeechToText(model_size=model_size, device=device, compute_type=compute_type)
                log(f"STT {model_size} on {device}")
                return stt
            except Exception as exc:  # noqa: BLE001 -- try the next option
                log(f"STT {model_size} on {device} unavailable ({exc})")
        raise RuntimeError("no speech-to-text model could be loaded")

    def handle_listen(self, cmd: dict) -> None:
        """Ack immediately and capture on a worker so TTS commands stay responsive."""
        listen_id = cmd.get("id")
        with self._listen_lock:
            busy = self._listen_thread is not None and self._listen_thread.is_alive()
            if not busy:
                self._listen_cancel.clear()
                self._listen_thread = threading.Thread(
                    target=self._listen_worker,
                    args=(cmd,),
                    daemon=True,
                    name="voice-listen",
                )
                self._listen_thread.start()
        send({"type": "ack", "cmd": "listen"})
        if busy:
            send({"type": "listen_result", "id": listen_id, "ok": False, "reason": "busy"})

    def _listen_worker(self, cmd: dict) -> None:
        listen_id = cmd.get("id")
        try:
            payload = self._capture_and_transcribe(cmd)
        except Exception as exc:  # noqa: BLE001 -- the sidecar command loop must stay up
            log(traceback.format_exc())
            payload = {"ok": False, "reason": "error", "message": str(exc)}
        payload["type"] = "listen_result"
        payload["id"] = listen_id
        send(payload)

    def _capture_and_transcribe(self, cmd: dict) -> dict:
        max_listen_sec = float(cmd.get("maxListenSec", 6.0))
        max_utterance_sec = float(cmd.get("maxUtteranceSec", 10.0))
        listen_id = cmd.get("id")
        # Whether Xenon was speaking while the user spoke: checked when their speech starts and
        # when it ends. A window opened during a reply often holds a turn spoken after it.
        spoke_during_capture = False

        def on_speech_start() -> None:
            nonlocal spoke_during_capture
            spoke_during_capture = self._canceller is not None and self._canceller.speaking
            send({"type": "speech_started", "id": listen_id})

        mic_result = record_from_mic(
            self.vad, max_listen_sec=max_listen_sec, max_utterance_sec=max_utterance_sec,
            on_speech_start=on_speech_start, cancel=self._listen_cancel,
        )
        spoke_during_capture = spoke_during_capture or (
            self._canceller is not None and self._canceller.speaking
        )
        detail = (
            f"device={mic_result.device_name!r} reason={mic_result.reason} "
            f"sec={mic_result.listen_sec:.2f} peak={mic_result.peak:.4f} rms={mic_result.rms:.4f}"
        )
        if not mic_result.speech_detected or mic_result.audio is None or mic_result.audio.size == 0:
            log(f"listen {detail} text=''")
            return {"ok": False, "reason": mic_result.reason or "no_speech"}

        audio = np.asarray(mic_result.audio, dtype=np.float32)
        peak = mic_result.peak if mic_result.peak > 0 else float(np.max(np.abs(audio)))
        # A quiet headset still trips VAD, then Whisper drops it as silence.
        if 0.001 < peak < 0.25:
            audio = np.clip(audio * (0.25 / peak), -1.0, 1.0).astype(np.float32)

        # Streaming VAD already accepted this clip. The batch pass only trims it.
        # An empty batch result must not throw the utterance away.
        vad_result = self.vad.detect(audio, sample_rate=SAMPLE_RATE)
        speech_audio = audio
        if vad_result.speech_detected and vad_result.segments:
            first_start = vad_result.segments[0]["start"]
            last_end = vad_result.segments[-1]["end"]
            pad = 0.2
            start_sample = max(0, int((first_start - pad) * SAMPLE_RATE))
            end_sample = min(len(audio), int((last_end + pad) * SAMPLE_RATE))
            trimmed = audio[start_sample:end_sample]
            if trimmed.size > 0:
                speech_audio = trimmed

        stt_result = self.stt.transcribe(speech_audio, sample_rate=SAMPLE_RATE)
        text = stt_result.text.strip()
        log(f"listen {detail} raw_peak={mic_result.raw_peak:.3f} text={text!r}")
        if not spoke_during_capture:
            mic_level.adjust_for_utterance(mic_result.device_name, mic_result.raw_peak)
        # Whisper writes these for noise and near-silence. The hands-free loop would answer them.
        if not text or text.rstrip(".!?").lower() in PHANTOM_TRANSCRIPTS:
            return {"ok": False, "reason": "no_speech"}
        on_headset = any(word in mic_result.device_name.lower() for word in HEADSET_WORDS)
        if spoke_during_capture and (not on_headset or self._is_own_echo(text)):
            log("dropped as echo of Xenon's reply")
            return {"ok": False, "reason": "echo"}
        return {"ok": True, "text": text}

    def _is_own_echo(self, text: str) -> bool:
        """Backstop for echo the canceller leaves behind, which Whisper transcribes accurately."""
        words = [w for w in _words(text) if w not in COMMON_WORDS]
        # A short answer such as "yes I would" reuses Xenon's words by nature.
        if len(words) < ECHO_MIN_WORDS:
            return False
        return sum(word in self._spoken_words for word in words) / len(words) >= ECHO_WORD_SHARE

    def handle_speak_start(self) -> None:
        with self._speaker_lock:
            if self._speaker is not None:
                # A previous turn's speaker was never finished (e.g. an error mid-stream) --
                # tear it down without waiting on playback rather than leaking threads.
                threading.Thread(
                    target=self._speaker.finish, kwargs={"wait_for_playback": False}, daemon=True
                ).start()
            self._speaker = IncrementalSpeaker(self.tts, speak=True)
            self._spoken_words = set()
        send({"type": "ack", "cmd": "speak_start"})

    def handle_speak_feed(self, cmd: dict) -> None:
        text = cmd.get("text", "")
        with self._speaker_lock:
            speaker = self._speaker
        if speaker is not None and text:
            speaker.feed(text)
            self._spoken_words.update(w for w in _words(text) if w not in COMMON_WORDS)
        send({"type": "ack", "cmd": "speak_feed"})

    def handle_speak_finish(self) -> None:
        with self._speaker_lock:
            speaker = self._speaker
            self._speaker = None
        send({"type": "ack", "cmd": "speak_finish"})

        def _finish_and_notify():
            if speaker is not None:
                speaker.finish(wait_for_playback=True)
            send({"type": "speak_done"})

        threading.Thread(target=_finish_and_notify, daemon=True).start()

    def run(self) -> None:
        send({"type": "ready"})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                send({"type": "error", "message": f"invalid JSON command: {e}"})
                continue

            name = cmd.get("cmd")
            try:
                if name == "ping":
                    send({"type": "pong"})
                elif name == "listen":
                    self.handle_listen(cmd)
                elif name == "listen_cancel":
                    self._listen_cancel.set()
                    send({"type": "ack", "cmd": "listen_cancel"})
                elif name == "speak_start":
                    self.handle_speak_start()
                elif name == "speak_feed":
                    self.handle_speak_feed(cmd)
                elif name == "speak_finish":
                    self.handle_speak_finish()
                elif name == "shutdown":
                    send({"type": "ack", "cmd": "shutdown"})
                    return
                else:
                    send({"type": "error", "message": f"unknown command: {name!r}"})
            except Exception as e:  # noqa: BLE001 -- must never crash the sidecar on a bad command
                log(traceback.format_exc())
                send({"type": "error", "message": f"{name} failed: {e}"})


if __name__ == "__main__":
    VoiceServer().run()
