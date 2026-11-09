#!/usr/bin/env bash
# First-run setup for a fresh checkout of Xenon2. Safe to run again at any time.
# On Windows, run it from Git Bash. It:
#   1. turns on this repo's git hooks (.githooks/: Git LFS, plus a guard against committing
#      joined model files or other big binaries outside LFS),
#   2. fetches the rwkv.cpp submodule and the model weights (Git LFS),
#   3. joins each model from its parts (GitHub's LFS caps files at 2 GB),
#   4. says what to build next. It builds nothing itself; see AGENTS.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Install it from https://git-lfs.com (Git for Windows includes it), then run this again." >&2
  exit 1
fi

# LFS's clean/smudge filters for this repo only; its hooks come from .githooks instead.
git lfs install --local --skip-repo >/dev/null
git config core.hooksPath .githooks
echo "Git hooks: on (.githooks)."

git submodule update --init --recursive
echo "Submodules: up to date."

# Pulls only what this checkout is missing. About 5.5 GB the first time.
git lfs pull
echo "Model weights: downloaded."

python scripts/join_models.py || python3 scripts/join_models.py

cat <<'EOF'

Repo is set up. Next, build what you need (details in AGENTS.md):
  NVIDIA GPU:  cd inference-engine && ./build_cuda.sh
  CPU only:    cd inference-engine && ./build_cpu.sh
  Desktop app: cd app && npm install && npm run tauri dev
  Voice:       see voice-pipeline/README.md (Python venv + a piper voice)
  VS Code:     cd vscode-xenon && npm install && npm run package, then install the .vsix
EOF
