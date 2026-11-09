"""Asks Xenon to describe each file listed in .xenon/repo-map.md and puts its answer beside the map's.

The map is checked by hand, so it is the reference. Review the output, then send each wrong answer
back as a correction (the app's example bank uses it on the next similar question, no training):

    python training/eval_repo_map.py                 # writes training/out/repo_map_eval.jsonl
    python training/eval_repo_map.py --correct PATH "what PATH really does"

Needs the Xenon2 app running (its coding API). Uses only the standard library.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / ".xenon" / "repo-map.md"
OUT = ROOT / "training" / "out" / "repo_map_eval.jsonl"
MAX_FILE_CHARS = 8000


def connection():
    appdata = Path(os.environ.get("APPDATA", Path.home() / ".local" / "share"))
    info = json.loads((appdata / "com.xenon2.app" / "code-api.json").read_text())
    return info["url"], info["token"]


def post(route, body):
    url, token = connection()
    req = urllib.request.Request(url + route, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def map_entries():
    section = ""
    for line in MAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            m = re.search(r"(\w[\w-]*)/", line)
            section = m.group(1) if m else ""
        m = re.match(r"^- ([\w./-]+\.\w+): (.+)$", line)
        if not m:
            continue
        rel = m.group(1)
        if not (ROOT / rel).exists() and section:
            rel = f"{section}/{rel}"
        if (ROOT / rel).exists():
            yield rel, m.group(2)


def question(rel):
    text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
    return f"This is `{rel}` from the Xenon2 repo:\n```\n{text}\n```\nIn one sentence, what does this file do in the project?"


def ask(q):
    reply = post("/v1/chat/completions", {"max_tokens": 200, "temperature": 0.2, "messages": [{"role": "user", "content": q}]})
    return reply["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--correct", nargs=2, metavar=("PATH", "DESCRIPTION"), help="send a correction for one file")
    args = parser.parse_args()

    if args.correct:
        rel, fix = args.correct
        q = question(rel)
        before = ask(q)
        post("/v1/feedback", {"message_id": "repo-map-" + rel, "rating": "bad", "correction": fix,
                              "history": [{"role": "user", "content": q}], "reply": before})
        print(f"before: {before}\nafter:  {ask(q)}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for rel, mine in map_entries():
            started = time.time()
            xenon = ask(question(rel))
            out.write(json.dumps({"path": rel, "map": mine, "xenon": xenon}) + "\n")
            print(f"{rel} ({time.time() - started:.0f}s)\n  map:   {mine}\n  xenon: {xenon[:240]}\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
