"""Measures Xenon's speed through the running app, the way a user sees it.

Asks for a 300-token story (streamed) and reports: writing speed (tokens per second while it is
writing), sustained speed (including the app's pauses to let a hot GPU cool), the pauses, and the
GPU's peak temperature and average power draw from nvidia-smi.

    python scripts/gpu_speed.py                # one line of results
    python scripts/gpu_speed.py --label 1200   # tag the line (used by gpu-clock-sweep.ps1)

Needs the Xenon2 app running. Waits for the GPU to cool to 70 C first, so runs compare fairly.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from eval_repo_map import connection  # noqa: E402


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.gr",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    temp, power, clock = (float(x) for x in out.strip().split(", "))
    return temp, power, clock


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="")
    parser.add_argument("--tokens", type=int, default=300)
    args = parser.parse_args()

    while gpu()[0] > 70:
        time.sleep(5)

    samples, done = [], threading.Event()

    def sample():
        while not done.is_set():
            samples.append(gpu())
            time.sleep(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    url, token = connection()
    body = {"stream": True, "max_tokens": args.tokens, "temperature": 0.3,
            "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + token})
    start, stamps = time.time(), []
    with urllib.request.urlopen(request, timeout=900) as reply:
        for line in reply:
            if line.startswith(b"data: {") and b'"content":"' in line and b'"content":""' not in line:
                stamps.append(time.time() - start)
    done.set()
    sampler.join()

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    pauses = [g for g in gaps if g > 1]
    writing = sum(g for g in gaps if g <= 1)
    total = stamps[-1] - stamps[0]
    print(f"{args.label:>8}  writing {len(gaps) / writing:5.1f} tok/s  sustained {len(gaps) / total:5.1f} tok/s  "
          f"pauses {len(pauses)} ({sum(pauses):.0f} s)  peak {max(s[0] for s in samples):.0f} C  "
          f"power {sum(s[1] for s in samples) / len(samples):.0f} W avg  clock {max(s[2] for s in samples):.0f} MHz max",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
