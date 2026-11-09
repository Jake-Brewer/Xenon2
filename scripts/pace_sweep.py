"""Sweeps the app-side heat-pacing settings to find the best sustained tokens-per-minute.

Writing a token is limited by GPU memory bandwidth, not core clock, so slowing the core (or, here,
pacing generation) sheds heat cheaply. This is the app-side alternative to an admin clock cap: it
needs no system settings and survives reboots. For each (start temperature, slope) it restarts the
app with XENON_PACE_FROM_C / XENON_PACE_MS_PER_C set, waits for the GPU to cool, and measures a
long reply with scripts/gpu_speed.py. Results go to training/out/pace-sweep.txt.

    python scripts/pace_sweep.py

Needs a built release app (scripts/run-release.sh). Each setting takes a rebuild-free restart plus
one measurement, a few minutes apiece.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "out" / "pace-sweep.txt"
# (start temperature C, ms per degree). "off" leaves pacing at its defaults is covered by a high
# start that never triggers below the 80 C hard pause.
SETTINGS = [
    ("off", 79, 0),
    ("warm", 76, 8),
    ("default", 74, 12),
    ("cool", 72, 14),
    ("cooler", 70, 16),
]


def health_ok():
    try:
        with urllib.request.urlopen("http://127.0.0.1:47811/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def restart(from_c, ms):
    env = {**os.environ, "XENON_PACE_FROM_C": str(from_c), "XENON_PACE_MS_PER_C": str(ms),
           "XENON_MODEL": "rwkv-7-g1l-2.9B-Q5_1.bin"}
    # run-release.sh already rebuilds (a no-op here) and restarts; it reads these from the env.
    subprocess.run(["bash", "scripts/run-release.sh"], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    for _ in range(60):
        if health_ok():
            return True
        time.sleep(3)
    return False


def measure(label):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gpu_speed.py"), "--label", label, "--tokens", "500"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "(no output)"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Pace sweep (500-token replies, G1l 2.9B Q5_1)"]
    OUT.write_text(lines[0] + "\n", encoding="utf-8")
    best = None
    for label, from_c, ms in SETTINGS:
        if not restart(from_c, ms):
            line = f"{label:>8} (from {from_c} C, {ms} ms/C): app did not come up"
        else:
            result = measure(label)
            line = f"{label:>8} (from {from_c} C, {ms} ms/C): {result}"
            m = re.search(r"sustained\s+([\d.]+)", result)
            if m and (best is None or float(m.group(1)) > best[0]):
                best = (float(m.group(1)), label)
        print(line, flush=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    if best:
        tail = f"best sustained: {best[1]} at {best[0]} tok/s"
        print(tail)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(tail + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
