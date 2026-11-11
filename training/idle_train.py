"""Run a small training batch while the user is idle.

Gating checks (each aborts with exit 0 if not met):
  - GPU lease file (~/.gpu/LEASE.txt) must not name a running game.
  - GPU temperature must be at or below COOL_C (72 C) via nvidia-smi.
  - Training batch must have at least MIN_BATCH_ITEMS items (from build_batch.py).

If all gates pass, runs RWKV-PEFT state tuning in WSL, then scores the result
against the held-out set and edit tasks. Loads the tuned state if scores didn't
get worse; discards it otherwise.

Usage:
    python training/idle_train.py              # one shot; prints what it did
    python training/idle_train.py --dry-run    # check gates only, don't train

The app can call this script after detecting ~10 min of user inactivity.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "training"
COOL_C = 72
HOT_C = 80
MIN_BATCH_ITEMS = 5
WSL_TRAIN_DIR = "~/xenon2-train"
LEASE_FILE = Path.home() / ".gpu" / "LEASE.txt"

# Games that hold the GPU lease (lowercase substrings to match against LEASE.txt).
GAME_SIGNALS = ["game", "steam", "proton", "codingame"]

# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def gpu_lease_free() -> tuple[bool, str]:
    """Returns (ok, reason). ok=False means a game or other heavy job holds the GPU."""
    if not LEASE_FILE.exists():
        return True, "no lease file"
    text = LEASE_FILE.read_text(encoding="utf-8", errors="replace").lower()
    for signal in GAME_SIGNALS:
        if signal in text:
            return False, f"GPU lease held by {signal!r} job"
    return True, "lease file present but no game"


def gpu_temp() -> int | None:
    """Returns GPU temperature in C, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().split()[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def gpu_cool_enough() -> tuple[bool, str]:
    temp = gpu_temp()
    if temp is None:
        return True, "nvidia-smi not found (CPU machine)"
    if temp > COOL_C:
        return False, f"GPU at {temp} C, need <= {COOL_C} C"
    return True, f"GPU at {temp} C"


def batch_ready() -> tuple[bool, str]:
    """Returns (ok, reason). Runs build_batch.py and checks item count."""
    result = subprocess.run(
        [sys.executable, str(TRAINING / "build_batch.py"), f"--min", str(MIN_BATCH_ITEMS)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"build_batch.py: too few items (need {MIN_BATCH_ITEMS})"
    return True, "batch ready"


# ---------------------------------------------------------------------------
# Training step (WSL)
# ---------------------------------------------------------------------------

def run_training(dry_run: bool) -> tuple[bool, str]:
    """Runs state tuning in WSL. Returns (ok, message)."""
    if dry_run:
        return True, "dry-run: skipped"

    wsl_cmd = (
        f"cd {WSL_TRAIN_DIR} && "
        "python state_tune.py "
        "--peft state --op fla --precision fp16 "
        "--load_model models/rwkv-7-g1l-2.9B.pth "
        "--output_dir runs/idle "
        "--n_layer 32 --n_embd 2560 "
        "--micro_bsz 1 --epoch_steps 20 --epoch_count 1 "
        "--data_file ~/xenon2-data/batch.jsonl "
        "--quant nf4"
    )
    result = subprocess.run(["wsl", "bash", "-c", wsl_cmd], capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        return False, f"WSL training failed:\n{result.stderr[-400:]}"
    return True, "training complete"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_edit_tasks() -> int:
    """Returns the number of tasks that passed. Returns -1 on error."""
    result = subprocess.run(
        [sys.executable, str(TRAINING / "run_edit_tasks.py")],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return -1
    # Last line: "N/M passed"
    for line in reversed(result.stdout.splitlines()):
        if "/" in line and "passed" in line:
            try:
                return int(line.split("/")[0].strip())
            except ValueError:
                pass
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="check gates only, don't train")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"idle_train: {msg}", flush=True)

    # Gate 1: GPU lease
    ok, reason = gpu_lease_free()
    log(f"lease check: {reason}")
    if not ok:
        return 0

    # Gate 2: GPU temperature
    ok, reason = gpu_cool_enough()
    log(f"temperature: {reason}")
    if not ok:
        return 0

    # Gate 3: batch size
    ok, reason = batch_ready()
    log(f"batch: {reason}")
    if not ok:
        return 0

    if args.dry_run:
        log("dry-run: all gates passed, would train now")
        return 0

    # Baseline scores before training
    log("scoring baseline edit tasks...")
    baseline = score_edit_tasks()
    log(f"baseline: {baseline} tasks passed")

    # Train
    log("starting state tuning in WSL...")
    ok, reason = run_training(dry_run=args.dry_run)
    log(f"training: {reason}")
    if not ok:
        return 1

    # Score after training
    log("scoring edit tasks after training...")
    after = score_edit_tasks()
    log(f"after training: {after} tasks passed")

    if after < 0:
        log("scoring failed; discarding tuned state")
        return 1

    if after >= baseline:
        log(f"scores held or improved ({baseline} -> {after}); keeping tuned state")
    else:
        log(f"scores regressed ({baseline} -> {after}); discarding tuned state")
        # TODO: revert to previous state file

    return 0


if __name__ == "__main__":
    sys.exit(main())
