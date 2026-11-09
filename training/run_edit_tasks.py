"""Scores the folder agent on the small edit tasks in training/edit_tasks/.

Each task is a tiny project (`repo/`), an English goal (`goal.txt`), and a hidden check
(`check.py`) that tests the goal and that nothing else broke. A task's tests passing isn't
enough: one run fixed `average` and deleted the line above it, and its tests still passed.

For each task this copies `repo/` into a fresh git repo, runs the agent through the running app
with no file list (it finds the files itself), then copies in `check.py` and runs it:

    python training/run_edit_tasks.py              # all tasks
    python training/run_edit_tasks.py off-by-one   # some

Results go to training/out/edit_tasks.jsonl. Needs the Xenon2 app running.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repo_map import ROOT, connection, post  # noqa: E402

TASKS = ROOT / "training" / "edit_tasks"
OUT = ROOT / "training" / "out" / "edit_tasks.jsonl"


def git(folder, *args):
    subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True)


def run_agent(folder, goal):
    job = post("/v1/folder-agent", {"folder": str(folder), "goal": goal, "files": []})
    url, token = connection()
    while True:
        req = urllib.request.Request(f"{url}/v1/folder-agent/{job['id']}", headers={"Authorization": "Bearer " + token})
        status = json.load(urllib.request.urlopen(req))
        if status["state"] not in ("running", "queued"):
            return status
        time.sleep(2)


def run_task(task):
    goal = (task / "goal.txt").read_text(encoding="utf-8").strip()
    with tempfile.TemporaryDirectory(prefix=f"xenon-task-{task.name}-") as tmp:
        folder = Path(tmp) / "repo"
        shutil.copytree(task / "repo", folder)
        git(folder, "init", "-q")
        git(folder, "-c", "user.email=t@x.local", "-c", "user.name=t", "add", ".")
        git(folder, "-c", "user.email=t@x.local", "-c", "user.name=t", "commit", "-q", "-m", "task")
        git(folder, "config", "user.email", "t@x.local")
        git(folder, "config", "user.name", "t")
        started = time.time()
        status = run_agent(folder, goal)
        seconds = round(time.time() - started)
        report = status.get("report") or {}
        shutil.copy(task / "check.py", Path(tmp) / "check.py")
        # Python looks for imports beside the script, not in the working folder.
        env = {**os.environ, "PYTHONPATH": str(folder)}
        check = subprocess.run([sys.executable, str(Path(tmp) / "check.py")], cwd=folder, capture_output=True, text=True, env=env)
        diff = subprocess.run(["git", "diff"], cwd=folder, capture_output=True, text=True).stdout
    return {
        "task": task.name,
        "passed": check.returncode == 0,
        "seconds": seconds,
        "stopped": report.get("stopped_reason") or status.get("error"),
        "narration": report.get("narration", []),
        "check_output": (check.stdout + check.stderr)[-600:],
        "diff": diff,
    }


def main():
    names = sys.argv[1:]
    tasks = [t for t in sorted(TASKS.iterdir()) if t.is_dir() and (not names or t.name in names)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with OUT.open("w", encoding="utf-8") as out:
        for task in tasks:
            result = run_task(task)
            results.append(result)
            out.write(json.dumps(result) + "\n")
            print(f"{'PASS' if result['passed'] else 'fail'} {task.name} ({result['seconds']}s): {result['stopped']}", flush=True)
    print(f"{sum(r['passed'] for r in results)}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
