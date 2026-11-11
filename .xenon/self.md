# How Xenon is built

Xenon's own description of itself, for answering questions about how it works. Each section
stands alone, so the app can put just the sections a question needs in front of it. The file map
is `.xenon/repo-map.md`; history and measurements are in `PLAN.md`. Keep this file true: when a
fact here changes, change it here in the same commit.

## What Xenon is

Xenon is an offline voice and text assistant that runs entirely on this laptop, with no cloud
service. Its job now is to maintain and improve its own code base, the Xenon2 repository:
answering questions about it, editing it, and running its checks. It has four parts:
- a Tauri desktop app (Rust backend in `app/src-tauri`, Vue frontend in `app/src`),
- a C++ inference engine (`inference-engine/`) built on rwkv.cpp, which runs the language model,
- a Python voice sidecar (`voice-pipeline/`) for listening and speaking,
- a VS Code extension (`vscode-xenon/`) that uses the app's local coding API.

## The language model

Xenon's language model is RWKV-7 at 2.9 billion parameters, made by BlinkDL (Bo Peng) and the
RWKV project. Two versions are on disk: RWKV-7 World v3 2.9B, which the app loads by default, and
RWKV-7 G1l 2.9B, which is better at code (67.5% of 40 MBPP problems solved on the first try, against World v3's 35%) and is
chosen with `XENON_MODEL=rwkv-7-g1l-2.9B-Q5_1.bin`. G1l is the model used for coding and the
folder agent. Both use the RWKV World tokenizer, a 65,536-token vocabulary. The 2.9B model has 32
layers and an embedding width of 2560. Xenon has not trained its own model yet: the weights are
BlinkDL's, unchanged except for quantization.

## How RWKV differs from a transformer

RWKV is a recurrent network, not a transformer. A transformer attends to every earlier token, so
its memory and time per token grow with the length of the context, and it has a fixed context
window. RWKV instead carries a fixed-size state from token to token: each layer keeps a matrix
memory that every new token updates (RWKV-7's "generalized delta rule", which can both add to and
erase from that memory). So each new token costs the same no matter how long the text is, memory
use is constant, and there is no hard context window. The trade-off is that the state is a
compressed summary: the model cannot look back at an exact earlier token the way attention can,
so long or precise recall is weaker. For the 2.9B model the whole state is about 20.6 MB.

## Quantization and files

The weights run in Q5_1 quantization (about 5 bits per weight, 2.6 GB), made from the FP16
conversion (5.5 GB) of BlinkDL's `.pth` checkpoint. Q5_1 is what fits on the 6 GB GPU with room
left for speech recognition. The direction is speed from the engine, not from lower precision:
no further quantization, and FP16 where it fits. The models live in `models/`, stored in Git LFS
as two parts plus a `.size` file because GitHub caps one LFS file at 2 GB;
`scripts/join_models.py` (or the app at startup) joins them into the `.bin` the engine loads.

## The inference engine

`inference-engine/` is a small C API (`xenon_inference.h/.cpp`) over rwkv.cpp, the RWKV runtime
built on ggml, included as a git submodule. It loads a model, evaluates tokens, samples the next
token, and can save, copy and restore model states. It builds for CPU (`build_cpu.sh`) or NVIDIA
CUDA (`build_cuda.sh`); the app uses CUDA when it can and falls back to the CPU. Xenon patches
rwkv.cpp (`inference-engine/patches/`, applied at build time, no fork) to compute RWKV-7's WKV
step and an L2 norm with native ggml operations on the GPU. Before the patch those ran on the
CPU and the data went back and forth every layer: a 1500-token prompt took 96 s to first token,
and after it 1.7 s, with the same output (KL divergence 0.002). Reading prompts runs in chunks of
128 tokens.

## Hardware and speed

Xenon runs on a laptop with an NVIDIA RTX 2060 (6 GB, compute capability 7.5, no bfloat16) and a
12-thread CPU. With every layer on the GPU, G1l 2.9B Q5_1 writes about 20 tokens per second
(23.1 tok/s measured with one CPU worker) and uses about 3 GB of VRAM. The app uses one CPU
worker when the whole model is on the GPU, because more workers only spin waiting for it and
made the keyboard lag. The app runs at below-normal priority and pauses work when the GPU
reaches 80°C until it cools.

## How a chat prompt is built and cached

Because RWKV's state only moves forward, anything that changes at the start of a prompt forces
everything after it to be read again. So the prompt is ordered by how often it changes:
1. a static prefix (a short description of Xenon and two example turns), read once when the
   model loads and kept as a saved state;
2. the conversation so far, kept per conversation as a cached state and extended by only the new
   turns (up to 4 conversations cached, each about 20.6 MB);
3. a header read fresh every turn: the real date and time, plus any memory facts or rated
   examples chosen for this message;
4. the new user turn and "Xenon:".
The coding API caches its prompts the same way. Turns are written "User: ..." and "Xenon: ...".

## Memory and learning without training

Xenon has two ways to change its answers without changing its weights.
- Ratings and the example bank (`feedback.rs`, `example_bank.rs`): a reply rated good, or a bad
  reply with a correction, is saved to `feedback.jsonl`. Later, when a new question means about
  the same (cosine similarity 0.55 or more), up to 2 of those exchanges are put in the prompt as
  examples. The base model imitates examples far more reliably than instructions: one
  correction nudged the answer and two changed it.
- Sloth memory (`sloth_memory.rs`): the "Sloth" agent saves short facts about the user in SQLite
  and puts the ones related to each message in the prompt. Facts are found by meaning, using
  384-dimension MiniLM sentence embeddings from the voice sidecar. The default agent, "Dementia",
  keeps nothing between conversations.

## Training

Training is planned and partly built, but no tuned model exists yet. The method is state tuning
with RWKV-PEFT: the weights stay frozen and only the starting state (one 64x64 matrix per head per
layer) is trained, which fits the 6 GB card when the frozen weights are loaded in 8-bit. It runs
in WSL with `--op fla --precision fp16`, because the RTX 2060 has no bfloat16. A trained state can
be loaded into rwkv.cpp without changing the model, but that loading has not been built. LoRA
(small trained weight matrices merged into the model) is the option for teaching new knowledge;
it is tighter on memory. Training data comes from Xenon's own corrected mistakes:
`training/build_batch.py` collects exam misses, corrected ratings, and edits the folder agent got
wrong and then right, always as the prompt plus a known-right answer, never a failed attempt.
`training/idle_train.py` says when a run can start: the app closed, the GPU cool and free, no other
job in `~/.gpu/LEASE.txt`, and at least 20 items in the batch.

## The coding API

The app serves an OpenAI-compatible HTTP API on 127.0.0.1:47811, guarded by a bearer token
written to `code-api.json` in the app's data folder. It offers chat completions (streaming),
feedback, code search, and the folder agent. When a request names a workspace folder,
`code_search.rs` finds the lines of the project that best match the question (rarer words count
more, definitions count more, dependencies are skipped) and puts them before the question. The
VS Code extension sends its open folder.

## The folder agent

`folder_agent.rs` makes code changes from an English goal. A small model can't plan a big change
or rewrite a whole file reliably, so the agent splits the work into small jobs:
- Locator: picks the file, then the few lines to rewrite (a short function is shown whole).
- Editor: the model rewrites just those lines.
- Checker: duplicate definitions are rejected, then the project's own tests run (pytest,
  cargo test, npm test, or a typecheck), compared with a baseline taken before any edit.
- Planner: a goal that names several places or files is split into steps; a round of steps is
  checked as a whole, undone together if it fails, and planned again with the failure shown.
- Test-writer: when a goal says a test should change too, it updates the failing assertion.
A failed edit is always undone, test files can't be loosened unless the goal asks, and the run
stops on success, on a budget of edits, or after 10 minutes. On 15 small practice projects it
passed 15 of 15 once (G1l, 2026-11-11), but a change in two places of one file passes only about
3 runs in 5. It has made one real change to this repo from a goal alone (commit `b15c9c1`).

## The voice pipeline

The Python sidecar talks to the app over stdin and stdout. It listens with a microphone, detects
speech with Silero VAD (on the CPU), turns speech into text with faster-whisper large-v3-turbo (on
the GPU), and speaks with Piper (on the CPU), starting to speak while the reply is still being
written. It cancels the echo of its own voice so it can be interrupted.

## How well Xenon knows this repo

Measured on held-out questions of the form "which file defines X": with code search, 89 of 100
answers name the right file and 83 of 100 give the exact path (the misses usually drop the leading
folders). Without search it got 0 of 100 exact. Its explanations of how the parts work have not
been scored yet; `training/eval_self.py` is the exam for that.
