"""Says whether a state-tuning run on the mistake batch may start now, and prints its command.

Training needs the whole 6 GB GPU (training/README.md), so it can't run inside the app or beside a
game or another project's job. Every gate must pass:

- the Xenon2 app is closed (its code API doesn't answer);
- no job holds a line in the GPU lease file (~/.gpu/LEASE.txt; lines starting with # are notes);
- nvidia-smi shows the GPU cool (<= 72 C) and nearly empty (a game or another job holds memory);
- the batch built by build_batch.py has at least --min items. build_batch.py itself needs the app
  running, so run it before closing the app; this script only counts what it wrote.

    python training/idle_train.py            # exit 0 and print the WSL command when ready, else 2

The run itself, the scores before and after, and keeping or dropping the state are still done by
hand as in training/README.md: a tuned state can't be loaded into rwkv.cpp yet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

COOL_C = 72
# The desktop alone holds a few hundred MiB; a game or a loaded model holds far more.
BUSY_MIB = 800
DEFAULT_MIN = 20
LEASE_FILE = Path.home() / ".gpu" / "LEASE.txt"
DATA = Path(os.environ.get("APPDATA", Path.home() / ".local" / "share")) / "com.xenon2.app"
BATCH = DATA / "training" / "batch.jsonl"
DISCOVERY = DATA / "code-api.json"


def lease_holders(text: str) -> list[str]:
    """The jobs holding the GPU: each non-empty line of the lease file that isn't a # note."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def gpu_reading() -> tuple[int, int] | None:
    """(temperature C, memory used MiB) from nvidia-smi, or None without an NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        temp, used = (int(v) for v in out.stdout.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return temp, used


def gpu_problem(reading: tuple[int, int] | None) -> str | None:
    if reading is None:
        return "no NVIDIA GPU (nvidia-smi not found); training needs one"
    temp, used = reading
    if temp > COOL_C:
        return f"GPU at {temp} C; wait for {COOL_C} C"
    if used > BUSY_MIB:
        return f"GPU has {used} MiB in use (a game, the app, or another job)"
    return None


def app_running() -> bool:
    try:
        url = json.loads(DISCOVERY.read_text(encoding="utf-8"))["url"]
        urllib.request.urlopen(f"{url}/health", timeout=2)
        return True
    except Exception:
        return False


def batch_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def command(batch: Path) -> str:
    """The README's state-tuning command, pointed at the batch (read through /mnt/c)."""
    drive, rest = batch.as_posix().split(":", 1)
    data = f"/mnt/{drive.lower()}{rest}"
    script = "/mnt/c/Users/jake/Documents/src/github/mccrea/Xenon2/training/state_tune.py"
    return (
        f"WANDB_MODE=disabled python {script} \\\n"
        "  --load_model ~/xenon2-train/rwkv7-2.9b.pth \\\n"
        f"  --proj_dir ~/xenon2-train/batch/out --data_file {data} --data_type jsonl \\\n"
        "  --vocab_size 65536 --n_layer 32 --n_embd 2560 \\\n"
        "  --ctx_len 384 --micro_bsz 2 --num_workers 1 \\\n"
        "  --epoch_steps 10 --epoch_count 3 --epoch_save 1 \\\n"
        "  --lr_init 1 --lr_final 0.01 --warmup_steps 5 \\\n"
        "  --accelerator gpu --precision fp16 --devices 1 --strategy auto --grad_cp 1 \\\n"
        "  --my_testing x070 --peft state --op fla --merge 0"
    )


def problems(min_items: int) -> list[str]:
    found = []
    if app_running():
        found.append("the Xenon2 app is running; close it first (training needs the whole GPU)")
    if LEASE_FILE.exists():
        for holder in lease_holders(LEASE_FILE.read_text(encoding="utf-8", errors="replace")):
            found.append(f"lease held: {holder}")
    problem = gpu_problem(gpu_reading())
    if problem:
        found.append(problem)
    items = batch_size(BATCH)
    if items < min_items:
        found.append(f"batch has {items} items; need {min_items} (run build_batch.py with the app open)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=DEFAULT_MIN, help="items the batch needs")
    args = parser.parse_args()
    found = problems(args.min)
    if found:
        for p in found:
            print(f"not yet: {p}")
        return 2
    print("Ready. Add a lease line, then run inside the WSL env from ~/xenon2-train/RWKV-PEFT:\n")
    print(command(BATCH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
