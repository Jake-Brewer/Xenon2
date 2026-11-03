"""Scores Xenon on Python test problems: the share whose code passes their unit tests (pass@1).

Asks the running Xenon2 app through its coding API (see app/src-tauri/src/code_api.rs), takes the
first code block of each reply, and runs it with the problem's tests in a separate Python process
with a time limit, in a temporary folder. The problems come from `make_python_sft.py`. Uses only
the standard library, so it runs from Windows Python while the app is up.

    python eval_code.py DATA_DIR/mbpp_test.jsonl [--limit 50] [--out results.jsonl]

The code that runs is what the model wrote for standard benchmark problems. It runs with a
timeout and no extra privileges, but it is still code you didn't review.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TIMEOUT_SECONDS = 10


def connection() -> tuple[str, str]:
    base = os.environ.get("APPDATA") or str(Path.home() / ".local" / "share")
    info = json.loads((Path(base) / "com.xenon2.app" / "code-api.json").read_text(encoding="utf-8"))
    return info["url"], info["token"]


def ask(url: str, token: str, question: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "xenon",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def first_code_block(reply: str) -> str:
    match = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", reply, re.S)
    if match:
        return match.group(1)
    # An unfinished block at the length limit: take what's there.
    match = re.search(r"```(?:python|py)?[^\n]*\n(.*)", reply, re.S)
    return match.group(1) if match else reply


def question_for(problem: dict) -> tuple[str, str]:
    """The question to ask and the test code to run after the answer."""
    if "tests" in problem:  # MBPP: one example test shows the expected function name
        question = f"{problem['prompt']} Your code should pass this test:\n{problem['tests'][0]}"
        tests = "\n".join(problem["imports"] + problem["tests"])
    else:  # HumanEval: complete the signature and docstring
        question = f"Complete this Python function:\n```python\n{problem['prompt']}```"
        tests = f"{problem['test']}\ncheck({problem['entry_point']})"
    return question, tests


def passes(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as folder:
        script = Path(folder) / "solution.py"
        script.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            run = subprocess.run([sys.executable, "-I", str(script)], cwd=folder, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return run.returncode == 0, run.stderr.strip().splitlines()[-1:][0] if run.stderr.strip() else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="only the first N problems")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    problems = [json.loads(line) for line in args.problems.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        problems = problems[: args.limit]
    url, token = connection()
    results = []
    started = time.time()
    for i, problem in enumerate(problems, 1):
        question, tests = question_for(problem)
        reply = ask(url, token, question, args.max_tokens)
        code = first_code_block(reply)
        if "tests" not in problem and f"def {problem['entry_point']}" not in code:
            code = problem["prompt"] + code  # the reply continued the body only
        ok, error = passes(code, tests)
        results.append({"task_id": problem["task_id"], "passed": ok, "error": error, "reply": reply})
        passed = sum(r["passed"] for r in results)
        print(f"[{i}/{len(problems)}] {problem['task_id']}: {'pass' if ok else 'FAIL ' + error[:60]}  ({passed}/{i} = {passed / i:.0%})", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result) + "\n")
    passed = sum(r["passed"] for r in results)
    print(f"pass@1 {passed}/{len(results)} = {passed / max(len(results), 1):.1%} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
