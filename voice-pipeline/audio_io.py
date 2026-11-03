"""audio_io.py

Audio input/output helpers: loading a WAV file (resampled to 16kHz mono, whatever its
source rate), live microphone capture gated by streaming VAD, and playback through
speakers. Kept separate from vad.py/stt.py/tts.py so pipeline.py can swap the input
source (mic vs. a pre-recorded fixture WAV) without touching the VAD/STT/TTS modules.
"""
from __future__ import annotations

import ctypes
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.signal as sps
import soundfile as sf
import sounddevice as sd

import echo_canceller
from vad import SAMPLE_RATE, VoiceActivityDetector
from silero_vad import VADIterator

FRAME_SAMPLES = 512  # silero-vad's recommended chunk size at 16kHz
INPUT_LATENCY_SEC = 0.04  # typical WASAPI shared-mode capture delay, added to the output's own


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Loads any WAV file and resamples/downmixes to float32 mono @ 16kHz, matching what
    a live mic capture in this pipeline produces -- so a pre-recorded fixture WAV and a
    live mic recording are interchangeable inputs to the rest of the pipeline."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_samples = int(round(len(data) * SAMPLE_RATE / sr))
        data = sps.resample(data, n_samples).astype(np.float32)
    return data


def play_audio(audio: np.ndarray, sample_rate: int, block: bool = True) -> None:
    """Plays float32 mono audio through the default speaker output device."""
    if audio.size == 0:
        return
    _ensure_com()
    canceller = echo_canceller.shared()
    if canceller is None or not block:
        sd.play(audio, sample_rate)
        if block:
            sd.wait()
        return

    # Written in short blocks so the canceller sees each block as it plays. sd.play would hand
    # over the whole sentence at once, and the echo reference would arrive seconds early.
    reference = _to_16k(np.asarray(audio, dtype=np.float32), sample_rate)
    block_frames = max(1, sample_rate // 50)
    reference_frames = echo_canceller.FRAME_SAMPLES * 2
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", latency="low") as stream:
        canceller.set_delay(float(stream.latency) + INPUT_LATENCY_SEC)
        for index, start in enumerate(range(0, len(audio), block_frames)):
            canceller.render(reference[index * reference_frames : (index + 1) * reference_frames])
            stream.write(np.asarray(audio[start : start + block_frames], dtype=np.float32).reshape(-1, 1))


def _log_audio(msg: str) -> None:
    print(f"[audio] {msg}", file=sys.stderr, flush=True)


_com_state = threading.local()


def _ensure_com() -> None:
    """WASAPI capture uses COM. PortAudio is initialized on the sidecar's main
    thread, but listen/playback run on workers. Opening the headset mic from a
    worker that has not called CoInitializeEx fails with WDM-KS 0x490 and the
    mic button reports that voice input is unavailable.
    """
    if getattr(_com_state, "ready", False) or sys.platform != "win32":
        _com_state.ready = True
        return
    # 0x2 is COINIT_APARTMENTTHREADED, which comtypes in mic_level.py also expects on this
    # thread. 0 is S_OK, 1 is S_FALSE (already initialized).
    hr = ctypes.windll.ole32.CoInitializeEx(None, 0x2)
    if hr not in (0, 1) and (hr & 0xFFFFFFFF) != 0x80010106:
        _log_audio(f"CoInitializeEx returned {hr:#010x}")
    _com_state.ready = True


def _wasapi_index() -> Optional[int]:
    for index, host in enumerate(sd.query_hostapis()):
        if "WASAPI" in host["name"].upper():
            return index
    return None


def _skip_input_name(name: str) -> bool:
    lowered = name.lower()
    return any(bad in lowered for bad in ("hands-free", "voicemeeter", "steam", "mapper"))


def input_candidates() -> list[tuple[int, int, str]]:
    """Mics to try, best first.

    The default output is the OpenRun headset, so its WASAPI input comes first.
    A forced 512-frame period on that Bluetooth mic fails to start (or starts
    silent), so callers open these with blocksize 0. WDM-KS entries are skipped;
    several of them are not real capture pins. MME and DirectSound copies of the
    same headset are later fallbacks.
    """
    devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    wasapi = _wasapi_index()

    def usable(index: int) -> bool:
        if index is None or not isinstance(index, int) or index < 0 or index >= len(devices):
            return False
        info = devices[index]
        if info["max_input_channels"] < 1:
            return False
        if "WDM-KS" in hostapis[info["hostapi"]]["name"].upper():
            return False
        return not _skip_input_name(info["name"])

    out_index = hostapis[wasapi]["default_output_device"] if wasapi is not None else int(sd.default.device[1])
    out_name = devices[out_index]["name"].lower() if usable_output(out_index, devices) else ""
    token = "openrun" if "openrun" in out_name else ("headset" if "headset" in out_name else None)
    wasapi_default = hostapis[wasapi]["default_input_device"] if wasapi is not None else -1
    mme_default = int(sd.default.device[0])

    ranked: list[tuple[int, int, int, str]] = []
    for index, info in enumerate(devices):
        if not usable(index):
            continue
        host = hostapis[info["hostapi"]]["name"].upper()
        is_wasapi = "WASAPI" in host
        matches = bool(token and token in info["name"].lower())
        if not matches and index not in (wasapi_default, mme_default):
            continue
        score = 20
        if matches and is_wasapi:
            score = 100
        elif matches:
            score = 80
        elif is_wasapi:
            score = 40
        if abs(float(info["default_samplerate"]) - SAMPLE_RATE) < 1:
            score += 5
        ranked.append((score, index, int(info["default_samplerate"]), info["name"]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    seen: set[int] = set()
    chosen: list[tuple[int, int, str]] = []
    for _score, index, rate, name in ranked:
        if index in seen:
            continue
        seen.add(index)
        chosen.append((index, rate, name))
    if not chosen and devices:
        info = devices[mme_default]
        chosen.append((mme_default, int(info["default_samplerate"]), info["name"]))
    return chosen


def select_input_device() -> tuple[int, int, str]:
    """The preferred mic. `record_from_mic` tries the rest of `input_candidates` if it will not start."""
    found = input_candidates()
    if not found:
        raise RuntimeError("no microphone input device")
    return found[0]


def usable_output(index: int, devices) -> bool:
    return index is not None and 0 <= index < len(devices) and devices[index]["max_output_channels"] > 0


def _to_mono(indata: np.ndarray) -> np.ndarray:
    samples = np.asarray(indata, dtype=np.float32)
    if samples.ndim == 1 or samples.shape[1] == 1:
        return samples.reshape(-1)
    rms = np.sqrt(np.mean(samples * samples, axis=0))
    return samples[:, int(np.argmax(rms))]


def _to_16k(samples: np.ndarray, native_sr: int) -> np.ndarray:
    if native_sr == SAMPLE_RATE or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    divisor = math.gcd(native_sr, SAMPLE_RATE)
    converted = sps.resample_poly(samples, SAMPLE_RATE // divisor, native_sr // divisor)
    return np.ascontiguousarray(converted, dtype=np.float32)


# Silero is level-sensitive. A laptop array mic can peak near 0.004 on speech, which scores
# under the trigger threshold and never starts an utterance. Silero sees a leveled copy of
# each frame. Whisper still gets the raw capture, and the sidecar normalizes that clip itself.
# Only frames well above the room's noise floor are boosted. Boosting everything makes steady
# room noise trip Silero, and Whisper then invents a word ("You") from an empty room.
VAD_TARGET_PEAK = 0.25
VAD_MAX_GAIN = 40.0
VAD_LEVEL_FRAMES = 32  # about one second at 512 samples per frame
VAD_FLOOR_FRAMES = 94  # about three seconds
VAD_GATE_RATIO = 3.0


class _VadLeveler:
    def __init__(self) -> None:
        self._peaks: deque[float] = deque(maxlen=VAD_LEVEL_FRAMES)
        self._levels: deque[float] = deque(maxlen=VAD_FLOOR_FRAMES)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        level = float(np.sqrt(np.mean(chunk * chunk)))
        self._levels.append(level)
        self._peaks.append(float(np.max(np.abs(chunk))))
        if level < VAD_GATE_RATIO * min(self._levels) or level <= 0.0:
            return chunk
        gain = min(VAD_MAX_GAIN, VAD_TARGET_PEAK / max(self._peaks))
        return np.clip(chunk * gain, -1.0, 1.0).astype(np.float32)


@dataclass
class MicCaptureResult:
    audio: Optional[np.ndarray]  # None if no speech was ever detected
    speech_detected: bool
    listen_sec: float
    reason: str  # "speech_end" | "max_utterance" | "no_speech_timeout" | "too_quiet"
    device_name: str = ""
    peak: float = 0.0
    rms: float = 0.0
    raw_peak: float = 0.0  # loudest mic sample during speech, before any processing


def record_from_mic(
    vad: VoiceActivityDetector,
    max_listen_sec: float = 6.0,
    max_utterance_sec: float = 10.0,
    silence_end_ms: int = 900,
    speech_threshold: float = 0.35,
    device: Optional[int] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> MicCaptureResult:
    """Stream the selected mic until an utterance ends.

    The device is opened at its native rate and converted to 16 kHz for Silero.
    A half-second of audio from before the voice-activity trigger is kept, so the
    start of the word is not cut off. A quiet headset mic is boosted later, at
    transcription time.
    """
    _ensure_com()
    if device is None:
        candidates = input_candidates()
        # Bluetooth headsets sometimes refuse the first open. Try the preferred mic twice.
        if candidates:
            candidates = [candidates[0], *candidates]
    else:
        info = sd.query_devices(device)
        candidates = [(device, int(info["default_samplerate"]), info["name"])]

    vad_iterator = VADIterator(
        vad.model, threshold=speech_threshold, sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=silence_end_ms,
    )

    frames: list[np.ndarray] = []
    # 16 frames * 512 samples is about half a second of pre-roll.
    pre_roll: deque[np.ndarray] = deque(maxlen=16)
    speech_started = False
    t_listen_start = time.perf_counter()
    t_speech_start = t_listen_start
    raw_queue: queue.Queue[np.ndarray] = queue.Queue()
    heard = threading.Event()
    pending = np.zeros(0, dtype=np.float32)
    leveler = _VadLeveler()
    canceller = echo_canceller.shared()
    raw_peak = 0.0
    heard_peak = 0.0
    heard_energy = 0.0
    heard_samples = 0

    def _callback(indata, _frame_count, _time_info, status):
        if status:
            _log_audio(f"input status: {status}")
        raw_queue.put(_to_mono(indata))
        heard.set()

    def _finished(reason: str, detected: bool) -> MicCaptureResult:
        audio = np.concatenate(frames) if frames else None
        # Levels cover everything the mic delivered, so a timeout logs what was actually heard.
        rms = math.sqrt(heard_energy / heard_samples) if heard_samples else 0.0
        peak = heard_peak
        if detected and audio is not None and audio.size:
            peak = float(np.max(np.abs(audio)))
        if detected and peak < 0.001:
            return MicCaptureResult(
                audio=None, speech_detected=False,
                listen_sec=time.perf_counter() - t_listen_start,
                reason="too_quiet", device_name=device_name, peak=peak, rms=rms, raw_peak=raw_peak,
            )
        return MicCaptureResult(
            audio=audio, speech_detected=detected,
            listen_sec=time.perf_counter() - t_listen_start,
            reason=reason, device_name=device_name, peak=peak, rms=rms, raw_peak=raw_peak,
        )

    def _start_input(device_index: int, native_sr: int, channels: int) -> sd.InputStream:
        # A fixed 512-frame block makes this OpenRun WASAPI mic fail to start
        # (WDM-KS 0x490) or come up silent. PortAudio's own period delivers audio.
        # WasapiSettings(auto_convert=True) starts the same mic and then never
        # calls back, so leave the host defaults alone.
        stream = sd.InputStream(
            samplerate=native_sr, channels=channels, dtype="float32",
            blocksize=0, device=device_index, callback=_callback,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        return stream

    stream: Optional[sd.InputStream] = None
    last_error: Optional[BaseException] = None
    native_sr = SAMPLE_RATE
    device_name = ""
    try:
        for device_index, rate, name in candidates:
            info = sd.query_devices(device_index)
            channels = max(1, min(int(info["max_input_channels"]), 2))
            _log_audio(f"mic {name!r} #{device_index} {rate} Hz {channels} ch")
            try:
                heard.clear()
                while True:
                    try:
                        raw_queue.get_nowait()
                    except queue.Empty:
                        break
                stream = _start_input(device_index, rate, channels)
                # A Bluetooth mic can report "started" and then never deliver a
                # buffer. Treat that as a failed open and try the next device.
                # The OpenRun mic can take over a second to deliver the first
                # buffer after the capture profile switches. Waiting this long
                # only applies when nothing arrives; the first buffer ends it.
                if not heard.wait(timeout=2.5):
                    stream.stop()
                    stream.close()
                    stream = None
                    raise RuntimeError("microphone produced no audio")
                native_sr = rate
                device_name = name
                break
            except Exception as exc:  # noqa: BLE001 -- try the next host API for this headset
                last_error = exc
                _log_audio(f"mic open failed {name!r} #{device_index}: {exc}")
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
                    stream = None
        if stream is None:
            raise last_error or RuntimeError("no microphone could be opened")
        while True:
            now = time.perf_counter()
            elapsed = now - t_listen_start
            if cancel is not None and cancel.is_set():
                return _finished("cancelled", False)
            if not speech_started and elapsed > max_listen_sec:
                return _finished("no_speech_timeout", False)
            if speech_started and (now - t_speech_start) > max_utterance_sec:
                return _finished("max_utterance", True)

            try:
                raw = raw_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if speech_started and raw.size:
                raw_peak = max(raw_peak, float(np.max(np.abs(raw))))
            converted = _to_16k(raw, native_sr)
            if canceller is not None:
                converted = canceller.capture(converted)
            pending = np.concatenate((pending, converted))
            while pending.size >= FRAME_SAMPLES:
                chunk = pending[:FRAME_SAMPLES]
                pending = pending[FRAME_SAMPLES:]
                heard_peak = max(heard_peak, float(np.max(np.abs(chunk))))
                heard_energy += float(np.sum(chunk * chunk))
                heard_samples += chunk.size
                if speech_started:
                    frames.append(chunk)
                else:
                    pre_roll.append(chunk)
                event = vad_iterator(np.ascontiguousarray(leveler(chunk)), return_seconds=True)
                if event is None:
                    continue
                if "start" in event and not speech_started:
                    speech_started = True
                    t_speech_start = time.perf_counter()
                    frames.extend(pre_roll)
                    pre_roll.clear()
                    if on_speech_start is not None:
                        on_speech_start()
                elif "end" in event and speech_started:
                    return _finished("speech_end", True)
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 -- closing a failed stream must not hide the capture result
                pass
        vad_iterator.reset_states()
