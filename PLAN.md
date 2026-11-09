# Xenon2 — Non-Transformer Offline Voice AI Assistant

Agent-facing project plan. Each task below is scoped to be handed to a coding
agent as a standalone prompt. Work through phases in order — later phases
depend on earlier ones being functionally complete (see "Depends on").

## Project summary

Portable, offline-first desktop AI assistant. Voice + text chat interface
similar to ChatGPT/Claude Desktop. Uses a non-transformer language model
(RWKV) for inference instead of a standard Transformer, for lower memory
overhead and better CPU-only streaming performance. Runs from an installed
location or directly off a USB/external drive with no install step.

## Target hardware (dev machine)

- CPU: Intel i7-12850HX, 16 cores / 24 threads
- RAM: 32GB
- GPU: NVIDIA RTX A1000 Laptop GPU, ~4GB VRAM (CUDA-capable, driver
  installed). Integrated Intel UHD Graphics also present but not relevant
  for inference offload. Build CPU-only first, but validate cuBLAS/GPU
  offload in Phase 1 rather than deferring it — the toolchain to build it
  (CMake, VS Build Tools, CUDA Toolkit) is not yet installed on the dev
  machine as of 2026-07-29 and will need setup before that part of Phase 1
  can run.

## Architecture decision (locked)

- **Model**: RWKV (via `rwkv.cpp`), INT4 quantized. Do not substitute Mamba
  or Liquid networks without re-opening this decision — see "Future work".
- **Desktop shell**: Tauri (Rust backend + Vue 3 frontend). Not Electron.
- **STT**: faster-whisper
- **TTS**: piper-tts
- **VAD**: silero-vad

---

## Phase 1 — Inference Engine Core [x] (vocab-file build fix applied 2026-07-31)

**Depends on**: nothing (start here)

**Goal**: A working, testable RWKV inference pipeline callable from the
command line, with no UI yet.

Tasks:
1. Set up `inference-engine/` as a Rust or C++ project wrapping `rwkv.cpp`
   as a git submodule.
2. Implement model loading for a quantized `.ggml` RWKV model file.
3. Implement a `generate(prompt: str, max_tokens: int) -> str` function
   with streaming token callback (yield tokens one at a time, not batched).
4. Download and quantize one small RWKV model (start with 430M params) to
   INT4 using the GGML quantize tool. Store under `models/`.
5. Write a CLI test harness: `test_inference.exe "hello, how are you?"`
   that prints streamed tokens to stdout.

**Acceptance criteria**:
- CLI harness loads the model and streams a coherent response for a basic
  greeting prompt in under 2 seconds on the dev machine.
- Memory use stays flat regardless of conversation length (verify state is
  fixed-size, not growing).

---

## Phase 2 — Voice I/O Pipeline [x]

**Depends on**: Phase 1 (needs `generate()` to pipe text through)

Tasks:
1. Integrate faster-whisper for STT: microphone audio in, text out.
2. Integrate silero-vad to detect speech start/stop so STT only runs on
   actual speech, not silence.
3. Integrate piper-tts for TTS: text in, audio out through speakers.
4. Wire together: mic → VAD → STT → Phase 1 `generate()` → TTS → speakers.
5. Build a CLI harness that runs a full voice round-trip and logs timing
   for each stage (VAD, STT, inference, TTS).

**Acceptance criteria**:
- Full voice round-trip for a short greeting completes in under 2 seconds
  on the dev machine.
- Pipeline recovers gracefully if no speech is detected (doesn't hang).

---

## Phase 3 — Desktop Shell & Chat UI [x]

**Depends on**: Phase 1 (text works standalone even before Phase 2 is done —
UI work can start in parallel with Phase 2)

**Goal**: ChatGPT/Claude-style desktop window layout.

Tasks:
1. Scaffold Tauri + Vue 3 project under `app/`.
2. Build `Sidebar.vue`: list of past conversations grouped by date
   (Today / Yesterday / Older), "+ New Chat" button, click to load a
   conversation into the main panel.
3. Build `ChatMessage.vue`: renders one message bubble (user or assistant),
   shows edit/delete icons on hover for user messages, shows regenerate
   icon for assistant messages.
4. Build `MessageInput.vue`: text input box with send button, plus a mic
   toggle button. Typed text and voice transcripts both feed into the same
   send-message code path.
5. Build `MenuBar.vue`: File menu with New / Open / Save / Save As /
   Export Memory.
6. Implement Tauri IPC commands connecting the UI to Phase 1's `generate()`
   (text-only first; voice wiring comes after Phase 2 is ready).

**Acceptance criteria**:
- App launches, shows empty sidebar + chat panel.
- User can type a message, see it appear as a bubble, and see a streamed
  response appear below it.
- Layout visually resembles ChatGPT/Claude Desktop (sidebar + main panel).

---

## Phase 4 — Message Editing & Regeneration [x]

**Depends on**: Phase 3

Tasks:
1. Define conversation data model: ordered array of
   `{role, content, edited, timestamp}` objects held in app state (Pinia).
2. Implement "edit user message": clicking edit turns the bubble into an
   editable text field; on submit, truncate the conversation array after
   that message's index, update its content, and re-run `generate()` from
   that point (discard old downstream messages).
3. Implement "regenerate assistant message": re-run `generate()` using the
   same conversation history up to (not including) that message.
4. Persist edits to the in-memory conversation state immediately (actual
   file save happens in Phase 5).

**Acceptance criteria**:
- Editing an earlier message and resubmitting removes all messages after
  it and produces a new assistant response based on the edited text.
- Regenerate replaces only the target assistant message, nothing else.

---

## Phase 5 — Project File Save/Load [x]

**Depends on**: Phase 4 (needs the conversation data model finalized)

Tasks:
1. Define the on-disk conversation JSON schema (see `SCHEMA.md` — create
   this file as part of this phase).
2. Implement File > Save / Save As: write current conversation to a
   user-chosen `.json` file via Tauri's file dialog + filesystem APIs.
3. Implement File > Open: load a `.json` file, validate against the
   schema, populate the sidebar + chat panel.
4. Implement auto-save: write to the last-used path after every completed
   message exchange, so no crash loses history.
5. Store the project's model name/version inside the saved file so a
   reopened project knows which quantized model to reload.

**Acceptance criteria**:
- Closing and reopening the app restores the last conversation.
- A saved `.json` file can be manually opened in a text editor and is
  human-readable.

---

## Phase 6 — External Memory Export/Import [x]

**Depends on**: Phase 5

**Goal**: Back up and migrate everything (models + conversation history) to
a USB drive, external SSD, or another internal drive, and reload it on a
different machine.

Tasks:
1. Define the on-disk layout for a portable data directory, e.g.
   `<root>/xenon2-data/models/` and `<root>/xenon2-data/projects/`.
2. Implement Export: Tauri file/folder picker to choose a destination
   drive/folder, then copy the local `models/` and `projects/` directories
   there.
3. Implement Import: pick a source folder (e.g. a plugged-in USB drive),
   copy its `models/` and `projects/` into the local data directory, or
   optionally run directly from the external path without copying.
4. Add a settings option for "data directory location" so the app can be
   pointed at an external drive as its primary storage instead of the
   local disk.

**Acceptance criteria**:
- Export produces a self-contained folder that includes both the
  quantized model file(s) and all saved conversations.
- That folder, copied to a different machine with Xenon2 installed, can be
  imported and the app resumes with full model + conversation history.

---

## Phase 7 — Hardening & Verification Follow-Through [x] (task 5 needs a human at the mic, see note)

**Depends on**: Phase 6 (all 6 original phases complete)

**Goal**: close specific gaps a project review surfaced after Phase 6 (see
`prompts/phase7_prompt.md`) -- not new user-facing scope, a follow-up pass.

Tasks:
1. Wire the mic button (`MessageInput.vue`) to Phase 2's standalone voice
   pipeline via a long-lived `ipc_server.py` sidecar process
   (`voice-pipeline/ipc_server.py`), bridged into Tauri by
   `app/src-tauri/src/voice.rs`. A transcript feeds into the exact same
   `sendMessage` path typed text uses; the assistant's reply is spoken back
   via the sidecar's `IncrementalSpeaker` (Phase 2's sentence-chunked
   streaming TTS), fed fragment-by-fragment from the existing `token-stream`
   event handler.
2. Implement real delete-message (`useChatStore().deleteMessage`) -- a real
   gap Phase 4 explicitly left unowned. Deleting a user message also removes
   its paired assistant reply; deleting an assistant message removes just
   that message.
3. Warn (dismissable banner, non-blocking) when an opened/active
   conversation's recorded `model` differs from the currently loaded model.
4. Full GUI click-through (real clicks, not just IPC-level calls) of Export
   Memory / Import Memory / Data Directory Settings, using the same UI
   Automation approach Phases 3-5 used.
5. Live human speech test through the real mic, once available.
6. Clean up stray git state (uncommitted prompt doc fix, submodule dirty
   state from untracked build dirs).
7. Replace the default Tauri icon with the project's alien/UFO icon.

**Acceptance criteria**: see `prompts/phase7_prompt.md`'s "Acceptance
criteria" section -- all verified against the real running dev build, not
just code review (see `app/README.md`'s Phase 7 section for specifics).

**Task 5 status**: verified live by the user speaking into the real mic --
confirmed working.

## Phase 7 follow-up — Model upgrade + Dementia/Sloth memory (2026-08-01)

After Phase 7's original 7 tasks shipped, real usage surfaced two further
issues, addressed in the same phase rather than opening a new one:

1. **Model quality**: RWKV-5 World 0.4B (Phase 1's original choice) was
   found to collapse onto generic canned replies for short/greeting-style
   prompts and got basic arithmetic wrong. Upgraded to **RWKV-7 World v3
   2.9B** (`rwkv-7-world-2.9B-Q5_1.bin`) -- same World tokenizer, same
   rwkv.cpp conversion pipeline (rwkv.cpp already supports the v7
   architecture). GPU offload is faster at this size (11.18 vs 8.95 tok/s).
   The app now uses it when `build-cuda-app` is present (RTX 2060 / sm_75)
   and falls back to CPU if CUDA init fails. See `app/README.md`.
2. **Persistent memory**: added a mid-conversation toggle between two
   agents -- **Dementia** (the pre-existing behavior: memory only within
   the current chat, nothing survives outside it) and **Sloth** (reads and
   writes a persistent cross-conversation fact store, auto-extracted by the
   model itself after each Sloth turn, injected into every subsequent Sloth
   prompt regardless of which conversation it's in). New backend module
   `app/src-tauri/src/sloth_memory.rs`; UI: a Dementia/Sloth toggle in
   `MessageInput.vue`, a "Sloth" badge on messages that used it
   (`ChatMessage.vue`), and a "Sloth Memory..." management panel
   (`App.vue`) to inspect/delete/clear what's been remembered, since
   automatic extraction from a small model can be wrong.

### Further follow-up (2026-08-02 to 2026-08-04)

Continued real usage surfaced three more issues, fixed in the same phase:

3. **Closing-question habit**: the model ended nearly every reply with "Is
   there anything else I can help you with?" -- root-caused to the seed
   prompt's own example turn ending the same way (the model imitates the
   literal style of its example, not just the instruction around it). Fixed
   by rewriting the seed example to close declaratively, plus a display-time
   backstop that strips any that still slip through.
4. **Repetition collapse**: `xenon_inference.cpp` had no repetition penalty
   at all -- once a canned phrase entered the resent conversation history,
   the model would imitate it for unrelated follow-up questions instead of
   answering them. Added a standard llama.cpp-style penalty, empirically
   tuned against the actual failure conversation (1.15 wasn't enough; 1.3
   broke it with no quality loss on normal exchanges) -- see
   `inference-engine/README.md`'s "Repetition penalty" section.
5. **No real clock**: "what time is it" was pure hallucination. Fixed by
   grounding every prompt in the real system time (`chrono`), plus a
   demonstrated example turn (a bare instruction wasn't reliably followed
   for short phrasings) -- see `app/README.md`'s "Real system clock
   grounding" section.

Also: a British male Piper voice (`en_GB-alan-medium`) replaced the original
US female voice per user request, and a real bug was caught in the same
pass -- Phase 6's export/import (`memory.rs`) still hardcoded the pre-upgrade
0.4B model filename, so Export Memory had been silently backing up the wrong
model file since the Phase 7 model upgrade. Fixed.

---

## Phase 8 — Persistent RWKV state [x]

Performance phase, not new user-facing scope, closing a gap a performance
audit found: `xenon_generate()` reset RWKV state on every call, so every turn
re-serialized and re-evaluated the entire conversation as text -- a cost that
grew without bound the longer a conversation went on, defeating the whole
point of an architecture (RWKV) chosen specifically for its fixed-size
recurrent state.

Added a caller-owned `xenon_state` type to `inference-engine` (`xenon_prefill`
/ `xenon_generate_with_state`, `xenon_generate()` itself unchanged in
signature/behavior) and an LRU of per-conversation cached state in
`app/src-tauri/src/inference.rs`, diffed against each incoming request's
history to decide "extend" (prefill just the new tail turns) vs. "rebuild"
(an edit or delete happened). Reordered the prompt so the truly-static
few-shot block and the growing history both come before the one part that
has to be re-fed every turn regardless (the real-clock/facts header) --
without that reorder, caching the history was impossible at all (RWKV state
is strictly sequential, so a changing value at the front invalidates
everything cached after it). See `app/README.md`'s "Phase 8" section and
`inference-engine/README.md`'s "Persistent state across calls" section for
the full design, the correctness proof (`test_state_cache.cpp`, byte-identical
output at turn 1/5/edit/regenerate), and measured before/after numbers.

---

## This computer vs the 2060 laptop (2026-10-31)

Two machines. The RTX 2060 is not attached here. This box is an Intel i5-1235U with Iris Xe
only (no `nvidia-smi`, no `nvcc`). Do not install a CUDA toolkit on it.

What is already true on this checkout, and is gitignored so it will not be on a fresh clone:

- `inference-engine/build-cpu-app` was built with `./build_cpu.sh` (AVX2, `XENON_HAS_CUDA=OFF`).
  The desktop app links that when `build-cuda-app` is absent.
- `models/rwkv-7-world-2.9B-Q5_1.bin` (2.6 GB) is the file `load_engine` opens. The 5.5 GB
  source `models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth` is still on disk; the FP16
  intermediate was deleted after quantize. Hugging Face GGUF files are llama.cpp format and
  will not load. Rebuild with `rwkv.cpp/python/convert_pytorch_to_ggml.py` then `quantize.py`
  Q5_1, run from `rwkv.cpp/build-cpu` so `quantize.py` finds `librwkv.so`. Source URL:
  `https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth`.
- CLI check: `test_inference "Say hello in one short sentence."` on CPU, 6 threads, answered
  `Hello!` in 3.6 s (1.38 tok/s, time to first token 3.2 s). `n_layer=32`, state 20.6 MB.
  The desktop app then printed `xenon2: inference device: CPU` and stayed up.
- Host packages the window needed: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`,
  `libayatana-appindicator3-dev`, `librsvg2-dev`, `patchelf`.

Window: `tauri.conf.json` has `"maximized": true`, and `fit_window_to_screen` asks again
after the window is mapped (model load blocks setup, so an earlier geometry read is 0×0).
Wayland reports position `0,0` and ignores `set_position`, so the app cannot choose the
monitor. Verified both placements: built-in panel (LGD `0x0709`) maximized to 1920×1080,
which is that work area; primary HP `w2338h` on HDMI at x=1920 maximized to 1854×1048,
the same size as a maximized window there (top bar and dock). Launch with
`npm run tauri dev` from `app/`. A VS Code snap terminal exports `GSETTINGS_SCHEMA_DIR`
and `GTK_PATH`; system WebKit then aborts
(`org.gnome.settings-daemon.plugins.xsettings` has no key `antialiasing`).
`app/scripts/run_desktop.sh` unsets those. A normal desktop launch does not have them.

Git push: `Jake-Brewer` accepted the write invite to `mrmccrea-coder/Xenon2` on this date
and is the account that can push. `jakez-gh` is pull-only. Do not change the remote.

Open, in priority order:

1. Voice is not installed. Startup logs that `voice-pipeline/.venv/bin/python3` is missing,
   and typed chat keeps working. Next step is `python -m venv .venv` and
   `pip install -r requirements.txt` in `voice-pipeline/`, plus the Piper voice. `ipc_server.py`
   tries CUDA STT first, then CPU int8, so the mic can work here without a GPU.
   `voice-pipeline/README.md` still names the old 0.4B model and `build-cuda-app`.
2. The 5.5 GB `.pth` can be deleted once nobody needs to re-quantize without re-downloading.

## Phase 9 — Semantic Sloth memory [x] (2026-11-02)

**Depends on**: Phase 8

Sloth stores 10,000 facts in JSON and picks 8 by keyword, so "what is my pet called" misses "my
dog is Rex". Replace that with vector search and remove the cap.

Tasks:
1. Store facts in SQLite (`rusqlite`, bundled) with `sqlite-vec`, and import `sloth_memory.json`
   once, keeping the original file.
2. Embed facts with a small local model (`all-MiniLM-L6-v2`, 384 dimensions, CPU, offline after
   the first download). Embed in the voice sidecar, and fall back to keyword search when it is
   not running, so typed chat keeps working.
3. Rank by vector similarity plus keyword match, and page the Sloth Memory panel.
4. Keep the store in the data directory and include it in Export Memory.

`sqlite-vec` searches by brute force, which suits about 100,000 vectors. Above that, add an HNSW
index. At one fact per Sloth turn at most, that size is years away.

**Acceptance criteria**: an old fact is found by meaning with no shared keywords, and a Sloth turn
with 100,000 facts adds under 100 ms.

**Done, with two changes from the tasks above:**
- Plain SQLite (`rusqlite`, bundled) with the search in Rust, not `sqlite-vec`. The crate is still
  an alpha release and searches by brute force anyway.
- Embeddings use ONNX Runtime and `tokenizers`, already installed for Piper and faster-whisper, not
  `fastembed`, which would have added about a dozen packages.

**Verified:** in the real app, "What is my pet called?" in a fresh Sloth chat answered "Your pet is
named Rex." from the fact "The user's dog is named Rex". The 4 facts in the old JSON store were
imported and embedded. Searching 100,000 embedded facts takes 69 ms in a release build (about 1.5
s in the debug `tauri dev` build), after caching each fact's keywords at load. The Memory panel
lists the newest 1,000 facts.

## Phase 10 — Learning from feedback [ ] (in progress)

**Depends on**: Phase 9 (the example bank reuses its search)

See `ONLINE_LEARNING.md` for the research, the plan, and why training does not run between turns.

Done: ratings with corrections, `feedback.jsonl` (exported and merged on import), and the example
bank, which changes later answers from ratings with no training. Next: the state-tuning spike.

## Phase 11 — VS Code integration [x] (2026-11-03), model comparison [ ] (in progress)

**Depends on**: Phase 1 (the engine), nothing else — no memory/example-bank reuse here.

Xenon now helps code from inside VS Code, fully offline, using the model the app already has
loaded. Two new pieces, both on `feature/vscode`:

- **`app/src-tauri/src/code_api.rs`**: an OpenAI-compatible `/v1/chat/completions` server on
  `127.0.0.1` (default port 47811, next free port after that), token-protected (random token
  written with the port to `<app data dir>/code-api.json` for clients to find), streamed over
  server-sent events. It shares the chat window's engine lock and keeps its own prefix-matched
  state cache (`EngineInner::code_cache`) so a follow-up question only prefills the new turns,
  not the whole conversation again. `Style::for_model` switches the prompt's speaker labels
  (`Xenon:` vs. G1's `Assistant:` + `<think></think>`) based on the loaded model's filename.
  `XENON_MODEL` env var picks a different `models/*.bin` file, for comparing models without a
  rebuild. 28 Rust unit tests.
- **`vscode-xenon/`**: a VS Code extension — sidebar chat (streams, Copy/Insert/Replace buttons
  on code blocks), editor commands (Ask, Explain, Fix, Write Tests, Add Docs), a status bar item,
  and an `@xenon` participant for VS Code's own Chat view. Code already sent in a chat isn't
  resent on a follow-up. 8 Node unit tests. Packaged as a `.vsix` and installed locally.

No context limit: RWKV has no fixed context window, so `code_api.rs` keeps system/file text whole
rather than truncating it, and the extension's `xenon.maxContextChars` setting accepts `0` for "no
cap" (still capped in practice by read speed — prefill runs ~35 tok/s on the RTX 2060, since the
vendored ggml has no GPU kernel for RWKV-7's WKV op, only a CPU one; see `rwkv_operators_wkv_v7.inc`).

**Model comparison, baseline done (2026-11-03)**: downloaded and converted `RWKV-7 G1l 2.9B`
(2026-10-31, `BlinkDL/rwkv7-g1`) to the same Q5_1 quantization as the current `RWKV-7 World v3
2.9B`, to see whether G1 (trained with "thinking" and tool-use data) codes better.
`training/make_python_sft.py` builds a 4000-pair training set from bigcode's execution-tested
`self-oss-instruct-sc2-exec-filter-50k` (ODC-By), with MBPP/HumanEval held out, for an infctx
LoRA tune. `training/eval_code.py` scores pass@1 against the running app (first code block of the
reply, run against the problem's unit tests, 10s timeout, no network/extra privileges).

Baseline pass@1 on the same 40 MBPP problems, zero-shot, no fine-tuning, `--temperature 0`:

| Model | pass@1 | Time (40 problems) |
|---|---|---|
| RWKV-7 World v3 2.9B (current default) | 35.0% (14/40) | 352s |
| RWKV-7 G1l 2.9B | **67.5% (27/40)** | 1113s |

**G1l is the clear win even untuned — almost double World v3's accuracy** (G1 was trained with
code/reasoning-heavy data that World v3 lacks). G1l is also noticeably slower per problem here
(longer replies, since it emits an (empty) `<think></think>` block and tends toward more verbose
explanations before code) — worth watching once an infctx LoRA tune is layered on, but the
accuracy gap is large enough that this is the new default regardless. `rwkv-7-g1l-2.9B-Q5_1.bin`
is already in `models/`.

**Not yet done, needs a decision**: switching the *app-wide* default model (`load_engine`'s
hardcoded `rwkv-7-world-2.9B-Q5_1.bin`) to G1l is more than a one-line change. `STATIC_PREFIX` and
`EXTRACTION_STATIC_PREFIX` in `inference.rs` are hand-written for World's `Xenon:`-label format;
G1 expects `Assistant:` turns and a `<think></think>` (or real `<think>`) block (see
`code_api.rs`'s `Style`, which already does this correctly for the coding API only). Swapping the
model without updating those two prefixes would run every voice/chat reply and fact-extraction
call against a prompt shape the model wasn't trained on. There's also no chat-quality benchmark
here yet, only a coding one (MBPP) -- G1l winning at code doesn't by itself say it's as good or
better as "Xenon" the general voice assistant. Left for a deliberate follow-up: update the two
chat prefixes to G1's format, spot-check general conversation quality, *then* flip the default.
In the meantime, G1l is fully usable for coding today via `XENON_MODEL=rwkv-7-g1l-2.9B-Q5_1.bin`
(which `code_api.rs` already formats correctly) -- the VS Code extension and coding API benefit
from the pass@1 win right now without touching the voice/chat path at all.

Also still open: run the infctx LoRA tune on `training/python_sft_train.jsonl` to see how much
further G1l's pass@1 improves, and re-score both HumanEval and the full MBPP set, not just 40
problems.

**Instant feedback loop for code (2026-11-04) [x]**: the chat window's example bank
(`example_bank.rs`) now works for the coding API too, reusing the same mechanism rather than a
new one. `FeedbackEntry.agent` tags which agent a rating belongs to (`"dementia"` for chat,
`"code"` for the coding API — `AGENT` in `code_api.rs`), and `examples_from`/`examples_for` take
an `agent` filter so chat and code ratings never mix. `code_api.rs` adds `POST /v1/feedback`
(records a rating/correction, shared `feedback::record_entry` with the chat window's Tauri
command) and splices the closest-matching rated examples into the prompt (`splice_examples`,
after the cached prefix so the prefix cache still hits) before every `/v1/chat/completions` call.
No training involved — a correction changes the very next similar answer, by embedding-similarity
retrieval into context, the same way chat ratings always have. Verified live: rated a wrong
"reverse a list" answer with a correction, then asked a similarly-worded question — the log line
`code API using 1 rated example(s) in this prompt` confirmed the retrieval fired, and the answer
changed from the wrong pattern to the corrected one.

The VS Code sidebar shows 👍/👎 under each reply; 👎 opens a one-line box for the fix, which is
what gets stored and reused (an uncorrected 👎 is recorded but can't be used as a future example,
same as chat). `vscode-xenon/src/client.ts` adds `sendFeedback`; `chatView.ts` tracks each
rateable reply's id, history, and text so a rating can be sent with the full context the model
saw. 9 Node tests (was 8), 30 Rust tests (was 29).

**Whole-file edits with diff preview (2026-11-04) [x]**: `xenon.editFile` command (editor context
menu and command palette) asks for an instruction, sends Xenon the whole current file plus the
instruction, and asks for the complete updated file back in one fenced block (not a patch — RWKV
has no context limit, so resending the whole file costs only read time, and "the whole file back"
is far more reliable than getting a model to emit a correct unified diff). The result opens as a
real VS Code diff (`vscode.diff`) against the original, never touching disk until the user picks
Apply or Discard from a notification (or the diff tab's title-bar buttons, `xenon.applyEdit` /
`xenon.discardEdit`, shown only while a diff is pending via the `xenon.editPending` context key).
New `vscode-xenon/src/edit.ts`; nothing here touches the chat sidebar's own insert/replace actions
on code blocks, which still work as before for smaller snippets.

**Project-folder agent (2026-11-08) [x]** — "Xenon: Improve Files in This Folder…" in the Command
Palette. You give it a goal in English and a list of Python files in the open folder. It edits one
file per turn, runs the tests (`python -m pytest` for `test_*.py`/`*_test.py`, plain `python`
otherwise), feeds failures back into its next turn, and stops when the tests pass, when it says
`DONE:`, or when it hits its budget. The report opens beside the editor: what it did and why,
before/after for each file, test output, and an **Undo all of this** button. New
`app/src-tauri/src/folder_agent.rs` and `vscode-xenon/src/folderAgent.ts`.

- *Containment:* every path is resolved and must stay inside the folder (no `../` escape), and
  only files on the given list can be edited. Python runs with its working directory set to the
  folder. This is not an OS sandbox (a script could still open an absolute path), so the git
  rollback point is the real safety net.
- *Rollback:* if the folder's git tree is clean, a `xenon-agent: snapshot before run` commit is
  made first, and Undo resets to it. A tree with uncommitted changes gets no snapshot, so Undo can
  never throw away the user's own unsaved work. Undo is then hidden.
- *Budget:* 5 edits, 3 test runs, 10 minutes.
- *Background job, not one long request:* `POST /v1/folder-agent` returns a run id right away
  (202). `GET /v1/folder-agent/<id>` returns progress and the final report.
  `POST /v1/folder-agent/<id>/cancel` stops the run before its next step. Only one run at a time
  (409 otherwise). The first version held one HTTP request open for minutes. Clients timed out
  and lost the report while the run kept the model busy.
- *GPU care:* this laptop's 2060 is shared with another project and with games. Before each model
  turn the agent checks `nvidia-smi` and, at 80°C or above, waits until the GPU is at 72°C or
  below (up to 5 minutes). Generation stops as soon as the `WHY:`/`DONE:` line ends. The model
  used to run on into an invented next turn, which wasted GPU time. A run also stops once the
  tests pass, and treats an "edit" that leaves the file unchanged as done. Sharing with the other
  project goes through a lease file outside the repo (`~/.gpu/LEASE.txt`).
- *Measured (G1l 2.9B, RTX 2060):* the demo's buggy `add` was fixed in one edit and the test
  passed. The run took 21 s, with a peak of 73°C. Before the fixes, the same run used its whole
  3-run budget on no-op edits and reached 82°C. The app uses ~2.9 GB of VRAM resident and
  ~4.8 GB at peak while generating.
- Tests: 43 Rust (containment, snapshot/undo, dirty-tree skip, budget, reply parsing, stopping
  generation early, unknown runs), 10 Node (including the start → poll → report client).

**No reply cap, and Stop instead (2026-11-08) [x]** — reply length is meant to be unbounded like
the context. Chat replies used to be cut at 200 tokens and coding replies at 800 (2048 max).
Now every path uses `MAX_REPLY_TOKENS` = 262,144 in `inference.rs`, a backstop only; the plan is
to raise it to 1M once long replies prove trustworthy. Replies end on their own at a stop
sequence. A long one is ended with **Stop**: a button under the chat (or Esc) calls the
`stop_generation` command, and the token callback stops at the next token, keeping the text so
far. VS Code's Stop hangs up, which already stopped generation. The folder agent's replies also
stop at its time budget or on cancel. Stop sequences now include "User:" after a *single*
newline: G1 sometimes writes that, and without a cap it would have run on. A coding reply that
comes back empty logs what the model actually wrote.

**VS Code chats know the open folder (2026-11-08) [x]** — asked "tell me about this repo", Xenon
described one of Claude Code's read-only command panels, because the extension only sent the
code around the cursor in whatever tab was active. Now code context comes only from real files
(`file:`/`untitled:` tabs), and each chat starts with a system message describing the workspace:
its path, its top two levels of files and folders (skipping `node_modules`, `target`, `.git`
and the like), and `.xenon/repo-map.md` if present, else the README. It is built once per chat,
so the server's prefix cache reads it only once. `.xenon/repo-map.md` is a hand-checked
one-line description of each folder and key file in this repo, also the reference for scoring
Xenon's own descriptions. Measured: "tell me about this repo" now gets a mostly right answer
(it misstated the model as 7B and `main.ts` as the Tauri entry), but took 246 s, almost all of
it reading the 6.7k-character overview cold. A follow-up took 3 s (cached), though its answer
was wrong (it named `persistence.ts` for the Stop button). Feedback examples now go in the
uncached tail of coding prompts, so they no longer break the cache for the next follow-up.

**Next: GPU prefill for RWKV-7.** Reading (prefill) runs at ~35 tok/s at best because the
vendored ggml has no GPU kernel for RWKV-7's WKV op (rwkv.cpp runs its own CPU version through
`ggml_map_custom1`). Upstream ggml has `GGML_OP_RWKV_WKV7` with a CUDA kernel. Plan: carry it as
a patch applied at build time (not a fork of the submodule), build into a separate folder,
check that logits match the current engine, measure, and only then switch the app.

## Future work (not yet scheduled)

- Mamba backend as an alternative to RWKV, once `mamba.cpp` tooling matures.
- GPU acceleration is wired: `inference-engine/build_cuda.sh` targets sm_75 (RTX 2060) and the
  app offloads every layer when that build is present, with a CPU retry if CUDA init fails.
  This Iris Xe machine uses `./build_cpu.sh` instead.
- Wake-word detection for always-listening mode.
- Listening during replies on laptop speakers. Headsets work: speech heard during a reply builds
  the next prompt. On speakers, WebRTC echo cancellation left phantom phrases and lost a second
  voice in testing, so speech captured over a reply is dropped there.
- Interrupting Xenon by talking over it. This also needs a way to cancel generation.
- Conversation branching (currently editing discards the old branch instead
  of preserving it).

---

## Notes for the agent picking up individual phase tasks

- Each phase's tasks should be given to the agent as its own prompt, in
  order. Do not skip ahead to Phase 3 UI wiring for voice before Phase 2's
  voice pipeline harness passes its acceptance criteria.
- Phases 2 and 3 can be worked on in parallel by two different agent runs
  since neither blocks the other until the final IPC wiring step in
  Phase 3, task 6.
- When a phase is complete, update this file's checklist (add `[x]` next to
  completed phase headers) so future agents/prompts know current status.
