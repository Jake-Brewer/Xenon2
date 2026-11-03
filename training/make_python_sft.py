"""Builds a Python coding training set sized for the RTX 2060, plus held-out test problems.

Training data: bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-By). 50k Python instruction /
answer pairs whose code was executed and passed its tests before being kept. `--count` of them are
sampled (default 4000). There is no length cap by default: RWKV has no context window, and
RWKV-PEFT's infctx training (`--train_type infctx --chunk_ctx 512`) runs long examples in
fixed-size chunks with the state carried between them, so VRAM stays flat. `--max-chars` can
still cap length for plain (non-infctx) training, where 1600 chars (~512 tokens) fits 6 GB.

Test data, never trained on: MBPP's sanitized test split (CC-BY-4.0, 257 problems) and HumanEval
(MIT, 164 problems), both with unit tests, so `eval_code.py` can measure how often Xenon's code
actually passes. Training pairs that share a 10-word run with any test problem are dropped.

Run in the WSL training env (needs pandas and pyarrow):
    python make_python_sft.py OUT_DIR [--count 4000] [--max-chars 0] [--assistant Xenon]
Downloads about 90 MB into OUT_DIR/raw on first use.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import pandas as pd

SOURCES = {
    "oss.parquet": "https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/resolve/main/data/train-00000-of-00001.parquet",
    "mbpp_test.parquet": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet",
    "humaneval.parquet": "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet",
}
SHINGLE_WORDS = 10


def fetch(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (raw / name).exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, raw / f"{name}.part")
            (raw / f"{name}.part").rename(raw / name)


def shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


def clean(text: str) -> str:
    """Blank lines would read as the end of a turn in the chat format."""
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"\n{2,}", "\n", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--max-chars", type=int, default=0, help="0 = no cap (for infctx training)")
    parser.add_argument("--assistant", default="Xenon", help="reply label: Xenon (app prompts) or Assistant (G1's own format)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = args.out / "raw"
    fetch(raw)
    pairs = pd.read_parquet(raw / "oss.parquet")
    mbpp = pd.read_parquet(raw / "mbpp_test.parquet")
    humaneval = pd.read_parquet(raw / "humaneval.parquet")

    test_shingles: set[str] = set()
    for text in list(mbpp.prompt) + list(mbpp.code) + list(humaneval.prompt):
        test_shingles |= shingles(text)
    test_names = set(humaneval.entry_point)
    test_names |= {m.group(1) for code in mbpp.code if (m := re.search(r"def\s+(\w+)", code))}

    lengths = pairs.instruction.str.len() + pairs.response.str.len()
    fits = pairs[lengths <= args.max_chars] if args.max_chars > 0 else pairs
    kept = []
    dropped = 0
    for row in fits.itertuples():
        names = set(re.findall(r"def\s+(\w+)", row.response))
        if names & test_names or shingles(row.instruction + " " + row.response) & test_shingles:
            dropped += 1
            continue
        kept.append(row)
    random.Random(args.seed).shuffle(kept)
    chosen = kept[: args.count]

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "python_sft_train.jsonl", "w", encoding="utf-8") as handle:
        for row in chosen:
            text = f"User: {clean(row.instruction)}\n\n{args.assistant}: {clean(row.response)}\n\n"
            handle.write(json.dumps({"text": text}) + "\n")

    with open(args.out / "mbpp_test.jsonl", "w", encoding="utf-8") as handle:
        for row in mbpp.itertuples():
            handle.write(json.dumps({
                "task_id": f"mbpp/{row.task_id}",
                "prompt": row.prompt,
                "tests": list(row.test_list),
                "imports": list(row.test_imports),
            }) + "\n")
    with open(args.out / "humaneval.jsonl", "w", encoding="utf-8") as handle:
        for row in humaneval.itertuples():
            handle.write(json.dumps({
                "task_id": row.task_id,
                "prompt": row.prompt,
                "test": row.test,
                "entry_point": row.entry_point,
            }) + "\n")

    chars = sum(len(r.instruction) + len(r.response) for r in chosen)
    cap = f"fit {args.max_chars} chars" if args.max_chars > 0 else "kept (no length cap)"
    longest = max((len(r.instruction) + len(r.response) for r in chosen), default=0)
    print(f"{len(pairs)} pairs, {len(fits)} {cap}, {dropped} dropped as too close to a test problem")
    print(f"longest chosen pair: {longest} chars (~{longest // 3.2:.0f} tokens); set --ctx_len above that")
    print(f"wrote {len(chosen)} training pairs ({chars / 1e6:.1f}M chars, ~{chars // 3.2 / 1e6:.1f}M tokens), "
          f"{len(mbpp)} MBPP and {len(humaneval)} HumanEval test problems -> {args.out}")


if __name__ == "__main__":
    main()
