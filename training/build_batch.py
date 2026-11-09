"""Collects Xenon's corrected mistakes into a training batch, and some into the held-out set.

Training runs on a batch of mistakes, not after each one. Every item pairs a prompt exactly as
Xenon saw it with an answer known to be right:

- exam misses (training/out/exam_misses.jsonl, from `make_self_data.py --score where --split
  train`), answered with the reference path;
- corrections from ratings (feedback.jsonl in the app's data folder), answered with the correction;
- the folder agent's fixed mistakes (mistakes.jsonl in the data folder), answered with the change
  that later passed every check. Never a failed one: Xenon learns the repair, not the mistake.

One in five new items, chosen by a hash of the item, goes to the held-out set instead, so the
score after training measures whether Xenon learned the kind of mistake, not the instances.
Items already in a batch or the held-out set are skipped, so nothing is shown twice.

    python training/build_batch.py              # add new items; prints the counts
    python training/build_batch.py --min 200    # exit 1 unless the batch has 200 items

Needs the Xenon2 app running (prompts are built by its /v1/prompt).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repo_map import ROOT, post  # noqa: E402

DATA = Path(os.environ.get("APPDATA", Path.home() / ".local" / "share")) / "com.xenon2.app"
TRAINING = DATA / "training"
BATCH = TRAINING / "batch.jsonl"
HOLDOUT = TRAINING / "holdout.jsonl"
SEEN = TRAINING / "seen.txt"
EXAM_MISSES = ROOT / "training" / "out" / "exam_misses.jsonl"


def key(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def held_out(prompt):
    # The same rule as the exam's split (make_self_data.held_out), applied to the prompt.
    return int(key(prompt), 16) % 5 == 0


def lines(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a damaged line, as the app also skips
    return out


def prompt_for(messages, workspace=None):
    body = {"messages": messages}
    if workspace:
        body["workspace"] = workspace
    built = post("/v1/prompt", body)
    return built["prior"] + built["last"]


def from_exam():
    for miss in lines(EXAM_MISSES):
        if miss.get("split") != "train":
            continue  # never train on held-out exam items
        prompt = prompt_for([{"role": "user", "content": miss["question"]}], miss.get("workspace"))
        yield {"source": "exam", "prompt": prompt, "answer": miss["answer"].strip()}


def from_feedback():
    latest = {}
    for entry in lines(DATA / "feedback.jsonl"):
        latest[(entry.get("messageId", entry.get("message_id", "")), entry.get("reply", ""))] = entry
    for entry in latest.values():
        if entry.get("agent") != "code" or entry.get("rating") != "bad" or not entry.get("correction"):
            continue  # chat ratings use the chat prompt, built by export_feedback.py
        history = entry.get("history") or []
        if not history or history[-1].get("role") != "user":
            continue
        yield {"source": "feedback", "prompt": prompt_for(history), "answer": entry["correction"].strip()}


def from_agent():
    # Written by the folder agent: the prompt of a failed attempt and the change that later passed.
    for item in lines(DATA / "mistakes.jsonl"):
        if item.get("prompt") and item.get("answer") and item.get("passed"):
            yield {"source": "agent", "prompt": item["prompt"], "answer": item["answer"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min", type=int, default=0, help="exit 1 unless the batch has at least this many items")
    args = parser.parse_args()

    TRAINING.mkdir(parents=True, exist_ok=True)
    seen = set(SEEN.read_text(encoding="utf-8").split()) if SEEN.exists() else set()
    added = {"batch": 0, "holdout": 0}
    with BATCH.open("a", encoding="utf-8") as batch, HOLDOUT.open("a", encoding="utf-8") as holdout, \
            SEEN.open("a", encoding="utf-8") as seen_file:
        for item in [*from_exam(), *from_feedback(), *from_agent()]:
            k = key(item["prompt"])
            if k in seen:
                continue
            seen.add(k)
            seen_file.write(k + "\n")
            target = "holdout" if held_out(item["prompt"]) else "batch"
            # No space before the answer: replies start right after the prompt's opener.
            record = {**item, "text": item["prompt"] + item["answer"] + "\n\n"}
            (holdout if target == "holdout" else batch).write(json.dumps(record) + "\n")
            added[target] += 1

    size = len(lines(BATCH))
    print(f"added {added['batch']} to the batch ({size} now), {added['holdout']} to the held-out set "
          f"({len(lines(HOLDOUT))} now)")
    return 0 if size >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
