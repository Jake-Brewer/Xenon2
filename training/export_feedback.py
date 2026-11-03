"""Turns the app's reply ratings into a state-tuning set.

Reads feedback.jsonl from Xenon's data folder. For each rated reply, the latest entry wins, as in
the app. A good reply is kept as written; a bad reply is used only when the user wrote a
correction, which replaces it. Up to MAX_HISTORY earlier turns go in front so the state learns the
reply in context. Unrated and uncorrected bad replies are skipped.

    python export_feedback.py PATH/TO/feedback.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_HISTORY = 4


def latest_by_reply(path: Path) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a damaged line, as the app also skips
            latest[(entry.get("messageId", ""), entry.get("reply", ""))] = entry
    return list(latest.values())


def clean(text: str) -> str:
    return text.strip().replace("\n\n", "\n")


def to_text(entry: dict) -> str | None:
    if entry.get("rating") == "good":
        reply = entry.get("reply", "")
    elif entry.get("rating") == "bad" and entry.get("correction"):
        reply = entry["correction"]
    else:
        return None
    history = entry.get("history", [])[-(MAX_HISTORY + 1):]
    if not history or history[-1].get("role") != "user":
        return None
    parts = []
    for turn in history:
        speaker = "User" if turn.get("role") == "user" else "Xenon"
        parts.append(f"{speaker}: {clean(turn.get('content', ''))}\n\n")
    parts.append(f"Xenon: {clean(reply)}\n\n")
    return "".join(parts)


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    texts = [t for t in map(to_text, latest_by_reply(source)) if t]
    with open(out, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")
    print(f"{len(texts)} examples -> {out}")


if __name__ == "__main__":
    main()
