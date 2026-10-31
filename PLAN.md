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

## Future work (not yet scheduled)

- Mamba backend as an alternative to RWKV, once `mamba.cpp` tooling matures.
- GPU acceleration is wired: `inference-engine/build_cuda.sh` targets sm_75 (RTX 2060) and the
  app offloads every layer when that build is present, with a CPU retry if CUDA init fails.
- Wake-word detection for always-listening mode.
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
