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
  {"cmd": "listen", "maxListenSec": 6.0, "maxUtteranceSec": 10.0}
    -> {"type": "listen_result", "ok": true, "text": "..."}
       {"type": "listen_result", "ok": false, "reason": "no_speech_timeout" | "no_speech"}
  {"cmd": "speak_start"}
    -> {"type": "ack", "cmd": "speak_start"}
  {"cmd": "speak_feed", "text": "..."}
    -> {"type": "ack", "cmd": "speak_feed"}
  {"cmd": "speak_finish"}
    -> {"type": "ack", "cmd": "speak_finish"} (immediately)
       {"type": "speak_done"} (later, once playback actually finishes)
  {"cmd": "shutdown"}
    -> process exits
"""
from __future__ import annotations

import json
import sys
import threading
import traceback

from vad import VoiceActivityDetector
from stt import SpeechToText
from tts import TextToSpeech
from audio_io import record_from_mic, SAMPLE_RATE
from speech_streamer import IncrementalSpeaker

_stdout_lock = threading.Lock()


def send(obj: dict) -> None:
    with _stdout_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    # Diagnostics go to stderr only -- never mixed into the stdout JSON stream.
    print(f"[ipc_server] {msg}", file=sys.stderr, flush=True)


class VoiceServer:
    def __init__(self):
        log("loading VAD...")
        self.vad = VoiceActivityDetector()
        log("loading STT (faster-whisper, small, cuda)...")
        # device="cuda" shares the GPU with RWKV. The desktop app offloads every RWKV layer
        # when build-cuda-app is what it linked (see inference.rs). Whisper small fp16 is
        # about 1GB; together with the 2.75GB Q5_1 model that fits a 6GB RTX 2060.
        self.stt = SpeechToText(model_size="small", device="cuda", compute_type="float16")
        log("loading TTS (piper)...")
        self.tts = TextToSpeech()
        log("ready")

        self._speaker: IncrementalSpeaker | None = None
        self._speaker_lock = threading.Lock()

    def handle_listen(self, cmd: dict) -> None:
        max_listen_sec = float(cmd.get("maxListenSec", 6.0))
        max_utterance_sec = float(cmd.get("maxUtteranceSec", 10.0))

        mic_result = record_from_mic(
            self.vad, max_listen_sec=max_listen_sec, max_utterance_sec=max_utterance_sec
        )
        if not mic_result.speech_detected or mic_result.audio is None or mic_result.audio.size == 0:
            send({"type": "listen_result", "ok": False, "reason": mic_result.reason})
            return

        # Trim to the detected speech span before STT, same as pipeline.py's batch path.
        vad_result = self.vad.detect(mic_result.audio, sample_rate=SAMPLE_RATE)
        if not vad_result.speech_detected:
            send({"type": "listen_result", "ok": False, "reason": "no_speech"})
            return

        first_start = vad_result.segments[0]["start"]
        last_end = vad_result.segments[-1]["end"]
        pad = 0.2
        start_sample = max(0, int((first_start - pad) * SAMPLE_RATE))
        end_sample = min(len(mic_result.audio), int((last_end + pad) * SAMPLE_RATE))
        speech_audio = mic_result.audio[start_sample:end_sample]

        stt_result = self.stt.transcribe(speech_audio, sample_rate=SAMPLE_RATE)
        if not stt_result.text.strip():
            send({"type": "listen_result", "ok": False, "reason": "no_speech"})
            return

        send({"type": "listen_result", "ok": True, "text": stt_result.text})

    def handle_speak_start(self) -> None:
        with self._speaker_lock:
            if self._speaker is not None:
                # A previous turn's speaker was never finished (e.g. an error mid-stream) --
                # tear it down without waiting on playback rather than leaking threads.
                self._speaker.finish(wait_for_playback=False)
            self._speaker = IncrementalSpeaker(self.tts, speak=True)
        send({"type": "ack", "cmd": "speak_start"})

    def handle_speak_feed(self, cmd: dict) -> None:
        text = cmd.get("text", "")
        with self._speaker_lock:
            speaker = self._speaker
        if speaker is not None and text:
            speaker.feed(text)
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
