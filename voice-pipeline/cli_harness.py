"""cli_harness.py

Phase 2 CLI test harness. Loads RWKV (GPU) + Whisper (GPU) + VAD (CPU) + TTS (CPU)
once, then runs one full voice round-trip -- either from a live microphone (default,
matching what Phase 3's UI will actually use) or from a pre-recorded WAV file (for
automated verification in this environment, where no human is available to speak into
a mic; see phase2_prompt.md's testing note). Prints per-stage timing, the actual device
each stage ran on, and peak VRAM usage with both RWKV and Whisper loaded.

Usage:
    python cli_harness.py --wav fixtures/sample_greeting.wav
    python cli_harness.py --mic
    python cli_harness.py --wav fixtures/silence.wav   # exercise the no-speech path
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from vad import VoiceActivityDetector
from stt import SpeechToText
from tts import TextToSpeech
from xenon_engine import XenonEngine
import pipeline


def query_vram_mib() -> tuple[int, int]:
    """Returns (used_mib, total_mib) for GPU 0 via nvidia-smi, or (-1, -1) if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        used, total = out.splitlines()[0].split(",")
        return int(used.strip()), int(total.strip())
    except Exception:
        return -1, -1


def main():
    ap = argparse.ArgumentParser(description="Xenon2 Phase 2 voice pipeline test harness")
    ap.add_argument("--wav", type=str, default=None,
                     help="Path to a WAV file to use as input instead of the live mic.")
    ap.add_argument("--mic", action="store_true",
                     help="Capture from the default microphone (default if --wav not given).")
    ap.add_argument("--whisper-model", type=str, default="small", choices=["tiny", "base", "small"],
                     help="faster-whisper model size (default: small -- both 'base' (~529 MiB "
                          "peak VRAM alongside RWKV) and 'small' (~945 MiB peak) were measured to "
                          "fit comfortably in the ~4GB budget; 'small' is chosen for better "
                          "transcription accuracy since there's ample headroom. See README.)")
    ap.add_argument(
        "--gpu-layers",
        type=int,
        default=999,
        help="RWKV layers to offload to GPU. 999 means every layer (clamped to the model).",
    )
    ap.add_argument("--max-response-tokens", type=int, default=60)
    ap.add_argument("--no-speak", action="store_true", help="Skip audio playback (still synthesizes).")
    ap.add_argument("--max-listen-sec", type=float, default=6.0)
    args = ap.parse_args()

    if not args.wav and not args.mic:
        args.mic = True  # mic is the default/primary path per spec

    print("=" * 70)
    print("Xenon2 Phase 2 -- voice pipeline CLI harness")
    print("=" * 70)

    vram_used_before, vram_total = query_vram_mib()
    print(f"VRAM before loading models: {vram_used_before} MiB used / {vram_total} MiB total")

    # --- Load all components once -------------------------------------------------
    print("\nLoading VAD (silero-vad, CPU)...")
    t0 = time.perf_counter()
    vad = VoiceActivityDetector()
    print(f"  loaded in {time.perf_counter() - t0:.3f}s, device={vad.device}")

    print(f"\nLoading STT (faster-whisper '{args.whisper_model}', GPU)...")
    t0 = time.perf_counter()
    stt = SpeechToText(model_size=args.whisper_model, device="cuda", compute_type="float16")
    print(f"  loaded+warmed up in {time.perf_counter() - t0:.3f}s (incl. {stt.warmup_sec:.3f}s CUDA "
          f"lazy-init warm-up transcription), "
          f"requested_device=cuda actual_device={stt.actual_device} compute_type={stt.compute_type}")

    vram_after_whisper, _ = query_vram_mib()
    print(f"  VRAM after Whisper load: {vram_after_whisper} MiB "
          f"(delta {vram_after_whisper - vram_used_before} MiB)")

    print(f"\nLoading RWKV engine (Phase 1 xenon_inference, GPU, gpu_layers={args.gpu_layers})...")
    t0 = time.perf_counter()
    engine = XenonEngine(n_gpu_layers=args.gpu_layers)
    print(f"  loaded in {time.perf_counter() - t0:.3f}s, "
          f"has_gpu_support={engine.has_gpu_support} n_layer={engine.n_layer}")

    vram_after_rwkv, _ = query_vram_mib()
    print(f"  VRAM after RWKV load (both models resident): {vram_after_rwkv} MiB "
          f"(delta from Whisper-only: {vram_after_rwkv - vram_after_whisper} MiB)")

    print("\nLoading TTS (piper-tts, CPU)...")
    t0 = time.perf_counter()
    tts = TextToSpeech()
    print(f"  loaded in {time.perf_counter() - t0:.3f}s (incl. {tts.warmup_sec:.3f}s onnxruntime "
          f"session warm-up), device={tts.device}")

    peak_vram = vram_after_rwkv
    print(f"\n--- Peak VRAM with RWKV + Whisper both resident: {peak_vram} MiB "
          f"/ {vram_total} MiB total ({100.0 * peak_vram / vram_total:.1f}%) ---")

    # --- Run one round-trip ---------------------------------------------------------
    print("\n" + "=" * 70)
    if args.wav:
        print(f"Running one round-trip from WAV file: {args.wav}")
        result = pipeline.run_once_from_wav(
            args.wav, vad, stt, engine, tts, speak=not args.no_speak,
            max_response_tokens=args.max_response_tokens,
        )
    else:
        print(f"Running one round-trip from live microphone "
              f"(listening up to {args.max_listen_sec}s for speech to start)...")
        result = pipeline.run_once_from_mic(
            vad, stt, engine, tts, speak=not args.no_speak,
            max_response_tokens=args.max_response_tokens, max_listen_sec=args.max_listen_sec,
        )
    print("=" * 70)

    # --- Report -----------------------------------------------------------------
    print(f"\nResult: {'OK' if result.ok else 'NOT OK'} (reason={result.reason})")
    if result.reason == "no_speech":
        print("No speech was detected -- pipeline returned gracefully instead of hanging. "
              "This is the expected/correct behavior for silent input, not a failure of the harness.")

    if result.transcript:
        print(f"Transcript (STT output): {result.transcript!r}")
    if result.response_text:
        print(f"Response (generate() output): {result.response_text!r}")

    print("\nPer-stage timing:")
    print(f"  {'stage':<20} {'device':<8} {'seconds':>10}   detail")
    for s in result.stages:
        print(f"  {s.name:<20} {s.device:<8} {s.seconds:>10.4f}   {s.detail}")

    print(f"\nRound-trip time (utterance received -> first spoken audio starts): "
          f"{result.total_sec:.4f}s", end="")
    if result.ok:
        verdict = "PASS (< 2s)" if result.total_sec < 2.0 else "FAIL (>= 2s)"
        print(f"  [{verdict}]")
    else:
        print()
    if result.full_playback_sec is not None:
        print(f"(For reference only, NOT the round-trip metric: full response took "
              f"{result.full_playback_sec:.4f}s until all spoken audio finished playing -- "
              f"this scales with response length/sentence count, not pipeline latency.)")

    vram_final, _ = query_vram_mib()
    print(f"\nVRAM at end of run: {vram_final} MiB / {vram_total} MiB total")

    engine.close()
    return 0 if (result.ok or result.reason == "no_speech") else 1


if __name__ == "__main__":
    sys.exit(main())
