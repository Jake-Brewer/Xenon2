"""Builds questions about Xenon2's own code, with answers taken from the code and its history.

Xenon's job is now to maintain and improve this repo, so this is its home exam and its training
set. Three kinds of question, each with a reference answer that needs no human checking:

- where:  "Which file defines `threads_for`?" -> the path, from the source itself.
- commit: a commit's diff -> its message, from git history.
- file:   a file's contents -> its one-line description in .xenon/repo-map.md.

Every 5th item of each kind is held out for scoring (the split depends only on the item's text,
so it stays put as the repo grows). Writes JSONL with {kind, question, answer, split}:

    python training/make_self_data.py                  # training/out/self_data.jsonl
    python training/make_self_data.py --score where    # ask the running app the held-out "where" items
    python training/make_self_data.py --score where --no-search   # same, without searching the repo
    python training/make_self_data.py --sft training/out/self_sft.jsonl   # training text (needs the app)
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repo_map import ROOT, map_entries, post, question as file_question  # noqa: E402

OUT = ROOT / "training" / "out" / "self_data.jsonl"
SOURCE_DIRS = ["app/src", "app/src-tauri/src", "vscode-xenon/src", "voice-pipeline", "training", "scripts",
               "inference-engine/src"]
DEFINITIONS = {
    ".rs": re.compile(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:fn|struct|enum|const|static|trait)\s+([A-Za-z_]\w{3,})"),
    ".ts": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|interface|type)\s+([A-Za-z_]\w{3,})"),
    ".vue": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w{3,})"),
    ".py": re.compile(r"^(?:def|class)\s+([A-Za-z_]\w{3,})"),
    ".cpp": re.compile(r"^[A-Za-z_][\w\s\*:<>]*?\b(xenon_\w+|[a-z_]{4,})\s*\([^;]*$"),
}
SKIP_PARTS = {"node_modules", "target", ".venv", "__pycache__", "out", "dist", "test"}
MAX_DIFF_CHARS = 6000


def held_out(text):
    return int(hashlib.sha1(text.encode()).hexdigest(), 16) % 5 == 0


def source_files():
    for folder in SOURCE_DIRS:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in DEFINITIONS and path.is_file() and not SKIP_PARTS & set(path.parts):
                yield path


def where_items():
    seen = {}
    for path in source_files():
        rel = path.relative_to(ROOT).as_posix()
        pattern = DEFINITIONS[path.suffix]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pattern.match(line)
            if m:
                seen.setdefault(m.group(1), set()).add(rel)
    for name, paths in sorted(seen.items()):
        if len(paths) == 1 and name not in {"main", "test", "tests", "self", "None"}:
            yield {"kind": "where", "question": f"In the Xenon2 repo, which file defines `{name}`? Answer with the path only.",
                   "answer": paths.pop()}


def commit_items():
    log = subprocess.run(["git", "log", "--no-merges", "--format=%H%x00%B%x01"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    for entry in log.split("\x01"):
        if "\x00" not in entry:
            continue
        sha, message = entry.strip().split("\x00", 1)
        message = "\n".join(line for line in message.strip().splitlines() if not line.startswith("Co-Authored-By"))
        diff = subprocess.run(["git", "show", "--format=", "--stat", "--patch", "--", ".", ":!models", ":!*.lock",
                               ":!*package-lock.json", sha], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
        if not diff.strip() or not message:
            continue
        if len(diff) > MAX_DIFF_CHARS:
            diff = diff[:MAX_DIFF_CHARS] + "\n... (diff cut)"
        yield {"kind": "commit", "question": f"Write the commit message for this change to Xenon2:\n```diff\n{diff}\n```",
               "answer": message}


def file_items():
    for rel, description in map_entries():
        yield {"kind": "file", "question": file_question(rel), "answer": description}


def build():
    items = [*where_items(), *commit_items(), *file_items()]
    for item in items:
        item["split"] = "eval" if held_out(item["question"]) else "train"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item) + "\n")
    counts = {}
    for item in items:
        key = f"{item['kind']}/{item['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(items)} items to {OUT}: {dict(sorted(counts.items()))}")


def ask(question, search=True):
    body = {"max_tokens": 120, "temperature": 0, "messages": [{"role": "user", "content": question}]}
    if search:
        body["workspace"] = str(ROOT)
    reply = post("/v1/chat/completions", body)
    return reply["choices"][0]["message"]["content"].strip()


def score(kind, search):
    items = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]
    items = [i for i in items if i["kind"] == kind and i["split"] == "eval"]
    right = right_file = 0
    for item in items:
        reply = ask(item["question"], search)
        ok = item["answer"] in reply if kind == "where" else None
        # Right file, path shortened (e.g. "src/inference.rs"): useful, but not the full answer.
        near = kind == "where" and not ok and re.search(r"\b" + re.escape(item["answer"].rsplit("/", 1)[-1]) + r"\b", reply)
        right += bool(ok)
        right_file += bool(ok or near)
        mark = "ok " if ok else "~  " if near else "-- " if ok is False else "   "
        print(f"{mark}{item['answer'][:60]!r} <- {reply[:100]!r}")
    if kind == "where":
        print(f"{right}/{len(items)} exact path, {right_file}/{len(items)} right file")


def write_sft(path):
    """Training text for RWKV-PEFT ({"text": ...} per line): each train item's prompt exactly as
    the running app builds it (through /v1/prompt, with code search for "where" items), then the
    reference answer."""
    items = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]
    items = [i for i in items if i["split"] == "train"]
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for item in items:
            body = {"messages": [{"role": "user", "content": item["question"]}]}
            if item["kind"] == "where":
                body["workspace"] = str(ROOT)
            prompt = post("/v1/prompt", body)
            out.write(json.dumps({"text": prompt["prior"] + prompt["last"] + " " + item["answer"].strip() + "\n\n"}) + "\n")
    print(f"Wrote {len(items)} training texts to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", choices=["where", "commit", "file"], help="ask the running app the held-out items")
    parser.add_argument("--no-search", action="store_true", help="don't let the app search the repo (baseline)")
    parser.add_argument("--sft", metavar="PATH", help="write the train items as RWKV-PEFT training text")
    args = parser.parse_args()
    if args.sft:
        write_sft(args.sft)
    elif args.score:
        score(args.score, not args.no_search)
    else:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
