"""Measures how hot the GPU gets while Xenon reads long prompts back to back.

Prompt reading runs at ~900 tok/s on the RTX 2060 and, unlike writing, wasn't paced, so the folder
agent's file-laden prompts pushed the GPU past the 80 C limit (84-85 C) during the edit benchmark.
This sends six requests in a row, each with ~8k tokens of this repo's own code and a 20-token
answer, and reports the peak temperature, time over 80 C, and the total time.

    python scripts/prefill_heat.py --label before

Needs the Xenon2 app running.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from eval_repo_map import connection  # noqa: E402

FILES = ["app/src-tauri/src/folder_agent.rs", "app/src-tauri/src/code_api.rs", "app/src-tauri/src/inference.rs",
         "app/src-tauri/src/code_search.rs", "vscode-xenon/src/extension.ts", "training/make_self_data.py"]


def temperature():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    while temperature() > 68:
        time.sleep(5)

    temps, done = [], threading.Event()

    def sample():
        while not done.is_set():
            temps.append(temperature())
            time.sleep(0.5)

    threading.Thread(target=sample, daemon=True).start()
    url, token = connection()
    started = time.time()
    for i, rel in enumerate(FILES):
        text = (ROOT / rel).read_text(encoding="utf-8")[:32000]
        # A different question each time, so no cached prompt is reused.
        question = f"(request {i}, {time.time()})\n```\n{text}\n```\nIn one sentence, what does this file do?"
        body = {"max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": question}]}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + token})
        urllib.request.urlopen(req, timeout=900).read()
    total = time.time() - started
    done.set()
    over = sum(0.5 for t in temps if t >= 80)
    print(f"{args.label:>8}  6 long prompts in {total:.0f} s  peak {max(temps)} C  {over:.0f} s at or over 80 C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
