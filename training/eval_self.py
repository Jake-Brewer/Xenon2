"""Exam: can Xenon explain how it is built and how its model works?

Each question lists the facts a good answer must contain. A fact is a regex, matched without
case, and `|` gives accepted phrasings. An answer scores the share of its question's facts it
contains; the exam score is the mean. The source of truth is `.xenon/self.md`; when a fact there
changes, change it here too.

    python training/eval_self.py                 # ask the running app, print the score
    python training/eval_self.py --workspace     # send the repo as the workspace (code search on)
    python training/eval_self.py --show          # also print each answer and its missing facts

These questions are the exam only. Training items about Xenon must be written separately, or the
score stops meaning anything. Results go to training/out/eval_self.jsonl. Needs the app running.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repo_map import ROOT, post  # noqa: E402

OUT = ROOT / "training" / "out" / "eval_self.jsonl"

# (question, [fact regexes])
QUESTIONS = [
    ("What language model do you run on?", [r"rwkv", r"rwkv.?7", r"2\.9 ?b"]),
    ("Who made the model you run on?", [r"blinkdl|bo peng|rwkv project"]),
    ("Is your model a transformer?", [r"\bno\b|not a transformer|isn't a transformer|is not", r"recurrent|rnn"]),
    ("How is RWKV different from a transformer?", [r"recurrent|rnn", r"state", r"constant|same (cost|time)|fixed.?size|linear", r"context window|attention"]),
    ("Do you have a context window limit?", [r"\bno\b|no (hard |fixed )?context window|not have a", r"state"]),
    ("What is the downside of RWKV's fixed-size state compared with attention?", [r"compress|summary|forget", r"recall|look back|exact|precise"]),
    ("How big is your model's state?", [r"20\.6 ?mb|about 20 ?mb|~?21 ?mb"]),
    ("How many layers does your model have?", [r"\b32\b"]),
    ("What tokenizer do you use?", [r"world", r"65,?536|65k"]),
    ("What quantization do your weights use, and why?", [r"q5_1|q5|5.?bit", r"6 ?gb|fit|vram|memory"]),
    ("How large is your model file?", [r"2\.6 ?gb"]),
    ("Why are the model files split into parts in the repo?", [r"lfs", r"2 ?gb"]),
    ("What is the difference between the World and G1l models?", [r"g1l", r"code|coding", r"world"]),
    ("Which model should be used for coding, and how is it chosen?", [r"g1l", r"xenon_model"]),
    ("What runs your model? Name the library.", [r"rwkv\.cpp", r"ggml"]),
    ("Can you run without a GPU?", [r"\byes\b|cpu", r"fall ?back|cpu build|build_cpu"]),
    ("What did the rwkv.cpp patches change, and why did it matter?", [r"wkv", r"gpu|cuda", r"cpu", r"96|1\.7|faster|first token"]),
    ("What GPU do you run on?", [r"rtx ?2060", r"6 ?gb"]),
    ("How fast do you generate text?", [r"2[0-3](\.\d)? ?tok|20 tokens|23\.1"]),
    ("Why does the app use only one CPU thread when the model is on the GPU?", [r"spin|wait|idle|busy", r"lag|keyboard|responsive|faster"]),
    ("What happens when the GPU gets hot?", [r"80", r"pause|wait|cool|throttl"]),
    ("How is a chat prompt ordered, and why?", [r"static prefix|static", r"history|conversation", r"header|date|time", r"cache|cached|re-?read|forward"]),
    ("Why is the date and time put near the end of the prompt instead of the start?", [r"cache|cached", r"state|re-?read|re-?fed|everything after"]),
    ("How many conversations' states are cached?", [r"\b4\b|four"]),
    ("What are the four main parts of Xenon?", [r"tauri|desktop app", r"engine|rwkv\.cpp|c\+\+", r"voice|python sidecar", r"vs ?code|extension"]),
    ("What is the app built with?", [r"tauri", r"rust", r"vue"]),
    ("How does the voice pipeline work?", [r"vad|silero|voice activity", r"whisper", r"piper", r"echo"]),
    ("Which speech recognition model do you use?", [r"whisper", r"large.?v3.?turbo|large-v3"]),
    ("How do ratings change your later answers without training?", [r"example", r"similar|meaning|cosine|0\.55", r"prompt"]),
    ("What is the difference between the Sloth and Dementia agents?", [r"sloth", r"fact|memory|remember", r"dementia", r"nothing|forget|no memory|doesn't keep|keeps nothing"]),
    ("How does Sloth find relevant memories?", [r"embedding|meaning|similar", r"minilm|384"]),
    ("Have you been fine-tuned or trained by this project?", [r"\bno\b|not (yet|been)|hasn't|have not|unchanged"]),
    ("How will you be trained?", [r"state tun", r"rwkv-?peft", r"frozen|weights stay"]),
    ("What is state tuning?", [r"state", r"frozen|weights (stay|don't change|unchanged)|only the (initial |starting )?state"]),
    ("Why does training use fp16 and not bf16?", [r"2060|turing|card|gpu", r"no bf16|lacks bf16|doesn't support|not support|no bfloat"]),
    ("How does training fit on a 6 GB card?", [r"8.?bit|quantiz", r"frozen|state"]),
    ("Where does your training data come from?", [r"mistake|correct", r"exam|rating|folder agent|edit"]),
    ("When can a training run start?", [r"app (is )?closed|close the app|app.*not running", r"gpu", r"lease", r"20"]),
    ("What is the coding API?", [r"openai", r"47811|127\.0\.0\.1|localhost", r"token|bearer"]),
    ("How does code search choose what to show the model?", [r"rare|rarity|weight", r"definition", r"line"]),
    ("How does the folder agent make a change?", [r"locat|find|pick", r"edit|rewrite", r"test|check", r"undo|undone"]),
    ("What happens when the folder agent's edit fails the tests?", [r"undo|undone|revert", r"again|retry|try|plan"]),
    ("How does the folder agent handle a goal that touches two files?", [r"step|plan|split", r"round|together|whole|check once"]),
    ("Can the folder agent change test files?", [r"\bno\b|not|protected|unless", r"goal|ask"]),
    ("How well does the folder agent do on its practice tasks?", [r"15", r"3 (of|in) 5|flaky|unreliable|two places"]),
    ("How well do you know where things are defined in this repo?", [r"8[39]", r"100"]),
]


def score(answer: str, facts: list[str]) -> tuple[float, list[str]]:
    missing = [f for f in facts if not re.search(f, answer, re.IGNORECASE)]
    return 1 - len(missing) / len(facts), missing


def ask(question: str, workspace: bool) -> str:
    body = {"max_tokens": 250, "temperature": 0.0, "messages": [{"role": "user", "content": question}]}
    if workspace:
        body["workspace"] = str(ROOT)
    reply = post("/v1/chat/completions", body)
    return reply["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", action="store_true", help="send the repo as the workspace")
    parser.add_argument("--show", action="store_true", help="print each answer and what it missed")
    parser.add_argument("--limit", type=int, default=0, help="ask only the first N questions")
    args = parser.parse_args()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with OUT.open("w", encoding="utf-8") as out:
        for question, facts in questions:
            answer = ask(question, args.workspace)
            got, missing = score(answer, facts)
            total += got
            out.write(json.dumps({"question": question, "answer": answer, "score": got, "missing": missing}) + "\n")
            print(f"{got:4.0%}  {question}", flush=True)
            if args.show:
                print("      " + answer.strip().replace("\n", "\n      "))
                if missing:
                    print(f"      missing: {missing}")
    print(f"score {total / len(questions):.1%} over {len(questions)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
