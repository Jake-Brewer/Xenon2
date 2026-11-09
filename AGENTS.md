# Working in Xenon2

Xenon2 is an offline voice and text assistant: a Tauri desktop app (Rust + Vue) running an RWKV-7
model locally, a Python voice sidecar, and a VS Code extension that uses the app's coding API.
`.xenon/repo-map.md` describes every folder and key file; `PLAN.md` has the history and decisions.

## First run on a new checkout

1. Install: Git with Git LFS, Python 3.11, Node.js 20+, Rust (rustup), CMake, and a C++ compiler
   (Visual Studio Build Tools on Windows). For GPU use, also the CUDA toolkit.
2. From the repo root (Git Bash on Windows): `bash scripts/setup-repo.sh`. It turns on the git
   hooks in `.githooks/`, fetches the `rwkv.cpp` submodule, downloads the model weights from
   Git LFS (about 5.5 GB), and joins each model from its parts. Run it again any time; it only
   does what is missing.
3. Build the inference engine, one of:
   - NVIDIA GPU: `cd inference-engine && ./build_cuda.sh`
   - CPU only: `cd inference-engine && ./build_cpu.sh`
   Machines without a GPU are supported. The app uses the CPU build when there is no CUDA build
   and falls back to the CPU if CUDA fails. Only training needs a GPU.
4. Desktop app: `cd app && npm install && npm run tauri dev`.
5. Voice (optional): follow `voice-pipeline/README.md` (a venv in `voice-pipeline/.venv` and a
   piper voice in `voice-pipeline/models/`). Without it, typed chat still works.
6. VS Code extension (optional): `cd vscode-xenon && npm install && npm run package`, then
   `code --install-extension vscode-xenon-0.1.0.vsix --force`.

## Model weights

- Each model is stored in LFS as `models/<name>.bin.part1`, `.part2` plus `<name>.bin.size`,
  because GitHub's LFS caps a file at 2 GB. The joined `models/<name>.bin` is ignored by git.
  `scripts/join_models.py` joins them; the app also joins them at startup if needed.
- The app loads `models/rwkv-7-world-2.9B-Q5_1.bin`. Set `XENON_MODEL` to another file name in
  `models/` to load a different one (e.g. `rwkv-7-g1l-2.9B-Q5_1.bin`, better at code).
- To add a model, put the `.bin` in `models/`, split it with
  `cargo test --lib split_real_models -- --ignored` (edit the list in `model_parts.rs`), and
  commit the parts and `.size` file. Weights we train go in `models/tuned/` (LFS, under 2 GB each).
- The pre-commit hook refuses a joined `models/*.bin` and any file over 100 MB that LFS isn't
  tracking.

## Tests

- Rust: `cd app/src-tauri && cargo test --lib`. On Windows, cargo may not be on PATH (it is under
  `~/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin`), and the tests need
  `app/src-tauri/target/debug` on PATH to find the engine DLLs.
- Frontend types: `cd app && npx vue-tsc --noEmit`.
- VS Code extension: `cd vscode-xenon && npm test`.
- While `npm run tauri dev` is running, every Rust edit rebuilds and restarts the app, which cuts
  off any request in flight. Batch Rust edits when someone is using the app.

## Rules

- Push only to `origin`. Never push to `upstream`.
- Commit small and often, with a message that says why.
- The GPU (an RTX 2060 in a laptop) may be shared with other projects and with games. Before long
  GPU work, check `nvidia-smi`; pause at 80°C until it is back to 72°C or below; stay off the GPU
  while a game is running. On the dev laptop, GPU jobs are announced in `~/.gpu/LEASE.txt`.
