# Xenon2 — Phase 3/4/5/6/7/8: Desktop Shell, Chat UI, Editing, File Save/Load, Export/Import, Hardening, Persistent State

A Tauri 2 + Vue 3 desktop chat app: sidebar + main chat panel, ChatGPT/Claude-Desktop-style
layout. Phase 3 (see `../PLAN.md` and `../prompts/phase3_prompt.md`) built the shell, layout, and
typed-text-in / streamed-text-out wiring against Phase 1's inference engine. Phase 4 (see
`../prompts/phase4_prompt.md`) migrated conversation state into a Pinia store and wired up real
edit-user-message and regenerate-assistant-message flows. Phase 5 (see
`../prompts/phase5_prompt.md` and `../SCHEMA.md`) made conversations durable: File > Save / Save
As / Open now do real file I/O, and every completed generation auto-saves to disk. Phase 6 added
external memory export/import and a data-directory-location setting. Phase 7 (see
`../prompts/phase7_prompt.md`) is a hardening/follow-up pass: real mic-button voice input/output,
real delete-message, a model-mismatch warning, a full GUI click-through of Phase 6's features, and
the app icon — see "Phase 7" below for details.

## Running the dev build

Prerequisites: Phase 1 must already be built. On a machine with an RTX 2060, run
`inference-engine/build_cuda.sh` so the app links `build-cuda-app` and offloads every layer.
`build-cpu-app` is the fallback when that library is absent (see
`../inference-engine/README.md`). The quantized model must exist at
`../models/rwkv-7-world-2.9B-Q5_1.bin`. Node.js, Rust/cargo, and the Tauri CLI must be on PATH.

From `app/`:

```powershell
npm install        # first time only
npm run tauri dev
```

The first build compiles the Rust/Tauri dependency tree from scratch and can take several
minutes; subsequent builds are incremental and fast. This launches a real desktop window (not a
browser tab) — Vite serves the Vue frontend on `http://localhost:1420` internally, but you
interact with the native WebView2 window `npm run tauri dev` opens.

To build a release bundle instead: `npm run tauri build`.

## What's fully functional vs. intentional stubs

Per `prompts/phase3_prompt.md`, `prompts/phase4_prompt.md`, and `prompts/phase5_prompt.md`'s
explicit scope boundaries, a couple of UI elements remain deliberate visual placeholders for later
phases — present and clickable, but not wired to real logic. Nothing below was skipped by
accident; each remaining stub logs to the browser devtools console (`console.log`) so it's easy to
confirm nothing silently does real work.

| Piece | Status | Notes |
|---|---|---|
| Typing a message + Send button | **Functional** | Full round trip through Rust into Phase 1's `xenon_generate()`, streamed back token-by-token. |
| "+ New Chat" (Sidebar) / File > New | **Functional** | Both call the same `newChat()` Pinia store action (`stores/chat.ts`) — clears the main panel and starts a fresh, empty conversation, tagged with the currently-loaded model's name. |
| Sidebar conversation list (Today/Yesterday/Older grouping, click to load) | **Functional** | Backed by durable storage as of Phase 5 (see below) — the sidebar is rebuilt from disk on every launch, not just held in memory. |
| **Edit icon on user messages** (`ChatMessage.vue`) | **Functional (Phase 4)** | Turns the bubble into an editable textarea (✓ confirm / ✕ cancel, or Enter/Escape). Confirming truncates the conversation after that message, updates its content, sets `edited: true`, and streams a fresh assistant reply from the edited text. See "Edit and regenerate flows" below. |
| **Regenerate icon on assistant messages** (`ChatMessage.vue`) | **Functional (Phase 4)** | Re-runs generation using history up to (not including) that message and replaces only that message's content in place, regardless of its position in the conversation. |
| **Delete icon on user and assistant messages** (`ChatMessage.vue`) | **Functional (Phase 7)** | Wired to `useChatStore().deleteMessage`. Deleting a user message also removes its immediately-following assistant reply (a user turn + its reply is treated as one deletable unit); deleting an assistant message removes just that message. Persists via the same `autoSave` path an edit uses. Confirmed with a real message via the running dev build (Chrome DevTools Protocol against `window.__store`): a 2-message conversation went to 0 messages after deleting the user message. |
| Mic toggle button (`MessageInput.vue`) | **Functional (Phase 7)** | Real VAD-gated mic capture -> STT transcript -> the exact same `sendMessage` path typed text uses; the reply is spoken back via sentence-chunked streaming TTS. See "Phase 7: voice input/output" below. |
| **File > Open / Save / Save As** | **Functional (Phase 5)** | Real native file dialogs (via `rfd`, called directly), real JSON read/write on disk, real schema validation with clear errors on malformed files. See "Phase 5: file persistence" below. |
| **File > Export Memory** | **Functional (Phase 6)** | Real folder picker + real copy of the quantized model, vocab, voice model, and all conversations into a portable bundle, with live byte-progress. See "Phase 6: external memory export/import" below. |
| **File > Import Memory** | **Functional (Phase 6)** | Real folder picker + real copy-in of a previously-exported bundle into this machine's local paths, with live byte-progress. |
| **File > Data Directory Settings** | **Functional (Phase 6)** | Persists an optional external "data directory location" (`settings.json`) that ongoing conversation saves/loads honor instead of the local app-data default. |

## State management: the Pinia store (`src/stores/chat.ts`)

As of Phase 4, `conversations`/`activeId`/`generating` no longer live as local `ref`s in
`App.vue` — they live in `useChatStore()` (Pinia), which is the single source of truth for all
conversation/message state. `App.vue` reads/writes through the store instead of owning the data
itself; it still owns Tauri event-listener registration (`token-stream` / `generation-done` /
`generation-error`) and DOM-only concerns (scroll-to-bottom) that don't belong in a store.

`ChatMessage` (`src/types.ts`) gained two fields for this phase:
- `timestamp: number` — epoch ms set once when the message is created. Left alone on user edits
  (it records when the turn was first created); a regenerated assistant message's `timestamp` is
  updated, since regeneration produces an effectively new reply.
- `edited?: boolean` — set to `true` on a user message once it's been edited. Rendered as a small
  "(edited)" tag under the bubble.

The store does **not** assume "the currently-streaming message is always the last one in the
array" (true for a normal send, but false for a regenerate targeting an earlier message).
Instead it tracks `streamingMessageIds: Record<conversationId, messageId>` — set right before a
`generate_response` call, consulted by the `token-stream`/`generation-done`/`generation-error`
handlers to find the exact message to mutate by id. This is what makes regenerating message #2 of
5 update message #2 in place without touching #1, #3, #4, or #5.

## Edit and regenerate flows

- **Edit a user message** (`useChatStore().editUserMessage`): truncates that conversation's
  `messages` array to drop everything after the edited message (its old reply and anything past
  it), updates its `content`, sets `edited = true`, pushes a new empty streaming assistant
  message, and re-runs the same generation flow `sendMessage` uses — reconstructing `history` from
  the truncated array through the edited message and calling `generate_response`.
- **Regenerate an assistant message** (`useChatStore().regenerateAssistantMessage`): rebuilds
  `history` up to (not including) the target message, clears just that message's `content`
  (`streaming = true`), and streams a new reply into it in place — every other message, before or
  after it, is left untouched.
- Both flows (and `sendMessage`) guard on the store's `generating` flag before starting, and the
  send/edit-confirm/regenerate controls are all disabled while `generating` is true — the Rust
  backend serializes all `generate_response` calls through a single mutex anyway (see
  `inference.rs`), so one app-wide flag (rather than per-conversation) is sufficient to prevent
  overlapping calls. Verified empirically: rapid-firing the send button three times in a row while
  a generation was in flight produced exactly one new user/assistant message pair, not three.
- "Persist edits immediately" falls out of the Pinia store being the single source of truth —
  `ChatMessage.vue` has no local copy of message content that could go stale; it reads `message`
  directly from the array the store owns and mutates through emitted events, never a local clone.

## Phase 5: file persistence

Full schema documentation lives in `../SCHEMA.md` at the repo root — this section covers the
implementation, not the on-disk shape itself.

### Rust side (`app/src-tauri/src/persistence.rs`)

New module, registered as a set of `#[tauri::command]`s in `lib.rs`:
- `save_conversation_file(path, conversation)` / `open_conversation_file(path)` — plain
  `std::fs` + `serde_json` read/write to an exact, already-known path. No Tauri fs plugin needed:
  these are trusted backend commands operating on a path either chosen via the native dialog or
  computed by the app itself, not an arbitrary path from untrusted web content. Writes are
  pretty-printed (`serde_json::to_string_pretty`), not minified. Reads validate strictly: missing
  fields, wrong types (e.g. `role` outside `"user"`/`"assistant"`), or an unrecognized
  `schemaVersion` all produce a specific, human-readable `Err(String)` rather than a panic — see
  `SCHEMA.md`'s "Validation on Open" section for exact message shapes.
- `default_conversation_path(conversation_id)` — computes
  `<app-data-dir>/conversations/<id>.json` via Tauri's `app_data_dir()`, creating the directory if
  needed. This is the auto-save default (see below).
- `load_session_file` / `save_session_file` — read/write a small `session.json` in the app-data
  dir (NOT part of the conversation schema) that remembers `lastActiveConversationId` and every
  known conversation's current path, so a relaunch can rebuild the sidebar and restore the
  last-active conversation without the user clicking Open.
- `pick_save_dialog(default_file_name)` / `pick_open_dialog()` — call `rfd::FileDialog` **directly**,
  not through `tauri-plugin-dialog`'s `DialogExt`. See "A real bug found and fixed" below for why.

### Frontend side

- `app/src/persistence.ts` — the TS mirror of the on-disk schema (`ConversationFile`,
  `SessionFile`) plus pure conversion helpers (`toConversationFile` / `fromConversationFile`)
  between it and the in-memory `Conversation`/`ChatMessage` types. `streaming`/`errored` are
  dropped when serializing (see `SCHEMA.md`) since a persisted message is never mid-stream or a
  failed placeholder — auto-save only ever fires after a *successful* completed generation.
- `app/src/stores/chat.ts` gained: `filePaths` (conversationId → last-used path), `modelName`
  (fetched once at startup via a new `get_model_name` command, see `inference.rs`), `fileError`
  (surfaced in `App.vue` as a dismissable red banner, separate from the existing inference-error
  banner), and actions `saveConversation` / `saveConversationAs` / `openConversation` / `autoSave`
  / `restoreSession` / `persistSession`. `completeGeneration` now ends with `void this.autoSave(...)`
  — fire-and-forget so a disk write never blocks the UI thread on a just-finished reply.
- `app/src/types.ts`'s `Conversation` gained an optional `model?: string` field (task 5's
  requirement), populated from `modelName` on every `newChat()` and carried through save/load.
- `MenuBar.vue`: Open / Save / Save As are now real (emit `open` / `save` / `save-as`, wired in
  `App.vue` exactly like `New` already was); Save / Save As are disabled (visually + functionally)
  when there's no active conversation.

### Auto-save default path policy

See `SCHEMA.md`'s "Auto-save default path policy" section for the full writeup. Short version: a
conversation that's never been manually saved auto-saves to
`<app-data-dir>/conversations/<conversation-id>.json` (on this Windows dev machine, resolves under
`%APPDATA%\com.xenon2.app\`) on its first completed generation, rather than prompting — a save
dialog interrupting the user right as their first reply finishes would be disruptive and easy to
accidentally cancel, defeating the "never silently lose history" goal. File > Save As always
opens a dialog and, once a path is chosen, every later auto-save for that conversation follows the
new path instead.

### A real bug found and fixed: `tauri-plugin-dialog`'s dialog never appeared

Initial implementation used `@tauri-apps/plugin-dialog` / `tauri-plugin-dialog`'s `DialogExt`
(`app.dialog().file()...blocking_save_file()` / `.save_file(callback)`). On this dev machine, in
this app, **the native dialog never appeared at all** — not slow, not erroring, just silently
never showing a window — regardless of whether it was called from a sync command, an `async`
command, or wrapped in `spawn_blocking`. Confirmed via `EnumWindows` (no dialog HWND ever created)
and via Chrome DevTools Protocol (`invoke('pick_save_dialog', ...)` left permanently pending, no
resolution, no rejection). The fix was to drop `tauri-plugin-dialog` entirely and call `rfd`
directly (`Cargo.toml`: `rfd = "0.17.2"`, no more `tauri-plugin-dialog` dependency, no more
`dialog:default` capability) — `rfd::FileDialog::save_file()` on Windows just calls the Common
Item Dialog COM API directly on the calling thread with no main-thread requirement (see rfd's
`win_cid/file_dialog.rs`), unlike the plugin's wrapper which routes through
`AppHandle::run_on_main_thread`.

### A separate, machine-level issue: the Windows Common Item Dialog itself hangs on this dev machine

After fixing the above, `rfd`'s dialog call still never returned on this specific dev machine.
Independently reproduced with code that has nothing to do with Xenon2 or Tauri: a plain
`System.Windows.Forms.SaveFileDialog` invoked from an unrelated PowerShell process hung
indefinitely (still not responded after 15+ seconds, vs. a plain `MessageBox` on the same machine
appearing instantly). This points to a hung/slow Windows shell extension (a cloud-sync client,
a corrupted "Quick access" list, etc.) that the modern Explorer-style picker enumerates while
opening — a well-documented class of real-world Windows issue, unrelated to Xenon2's code.
Two things were done about it:
1. **Shipped**: `persistence.rs`'s `pick_save_dialog`/`pick_open_dialog` now wrap the dialog call
   in a 30-second timeout (`DIALOG_TIMEOUT`), so a hung shell extension produces a clear,
   dismissable error ("The save dialog did not respond within 30 seconds...") instead of an
   indefinitely frozen-looking UI. This is a real, permanent hardening, not a workaround.
2. **Test-only**: an opt-in escape hatch (`XENON2_TEST_SAVE_DIALOG_PATH` /
   `XENON2_TEST_OPEN_DIALOG_PATH` env vars, checked first in each command, `test_override_path` in
   `persistence.rs`) returns a fixed path immediately instead of invoking the OS dialog. Unset (the
   default) in normal use — this is what let the rest of the Save As / Open pipeline (path
   bookkeeping, writing, schema validation, auto-save switching to the new path) be exercised
   through the real running app and real UI clicks on this machine, with only the OS picker step
   substituted. Documented here rather than hidden so a future maintainer knows it exists and why.

### Phase 5 verification (what was actually run, not just code-reviewed)

Real dev build (`npm run tauri dev`), real compiled Phase 1 model, real generated replies — driven
via UI Automation (`InvokePattern`/`ValuePattern`, focus-independent) and, for the two checks that
needed to bypass the confirmed-broken OS dialog (see above), via the real Pinia store actions
called through Chrome DevTools Protocol against the live running app (`window.__TAURI_INTERNALS__`
/ a temporarily-exposed store reference, removed again afterward) — not simulated, not unit-tested
in isolation:

- **Auto-save to the default path**: sent a real message, got a real streamed reply, confirmed
  `%APPDATA%\com.xenon2.app\session.json` and `...\conversations\<id>.json` were written
  automatically with no user action, pretty-printed and human-readable, `model` field correctly set
  to `"rwkv-5-world-0.4B-Q4_0"`.
- **Close and reopen restores the last-active conversation automatically**: killed the app process,
  relaunched from scratch, confirmed the sidebar and chat panel repopulated with the exact same
  conversation and messages with zero manual clicks, before wiring any event listeners.
- **Save As chooses a new path, and subsequent auto-saves follow it**: ran `saveConversationAs`
  against a path outside the app-data directory, confirmed the file was written there
  (`filePaths` updated); sent a second real message; confirmed the *new* path grew to 4 messages
  while the original app-data-default file stayed frozen at 2 — auto-save correctly followed the
  relocated path, not the original default.
- **Opening a malformed / non-Xenon2 file fails clearly, not a crash**: tested both "not valid JSON
  at all" (syntax error) and "valid JSON but missing `schemaVersion`/wrong shape" — both produced
  specific error text (see `SCHEMA.md`'s "Validation on Open"), both left the app's existing
  conversation list completely untouched (`convCountBefore === convCountAfter`), both rendered as
  the dismissable red file-error banner in the real UI, confirmed via screenshot.
- **Opening a valid file loads it correctly**: created a fresh empty chat, then opened a
  previously-saved 4-message conversation file; confirmed it replaced the active conversation
  (correct id, correct message count, correct `model` field) and appeared in the sidebar alongside
  the still-present empty chat, not duplicated.

## Phase 6: external memory export/import

Full bundle-layout documentation lives in `../EXPORT_FORMAT.md` at the repo root (the Phase 6
counterpart to `SCHEMA.md`) — this section covers the implementation and what was actually tested.

### Rust side (`app/src-tauri/src/memory.rs`)

New module, registered as `#[tauri::command]`s in `lib.rs` alongside `persistence.rs`'s:
- `export_memory(destination)` — copies the quantized model (`models/rwkv-5-world-0.4B-Q4_0.bin`),
  tokenizer vocab (`inference-engine/data/world_vocab.bin`), piper voice model + its `.onnx.json`
  sidecar (`voice-pipeline/models/`), and every `*.json` in the effective conversations directory
  (plus `session.json`) into `<destination>/xenon2-backup/`, per `EXPORT_FORMAT.md`'s layout.
  Deliberately excludes the `.pth`/FP16 conversion intermediates (~1.8GB, not loaded at runtime).
- `import_memory(source)` — copy-in only (never runs directly against the external path) — see
  `EXPORT_FORMAT.md`'s "Import: copy-in vs. run-in-place" for why. Copies a bundle's `models/`,
  `voice-models/`, and `conversations/*.json` into this machine's real local paths (`models/`,
  `voice-pipeline/models/`, and the effective conversations directory), then rewrites
  `session.json`'s `conversationPaths` to point at this machine's conversations directory (the
  source machine's original paths are meaningless here) before writing it to the local app-data
  dir.
- `copy_with_progress` — streams each file copy in 1MB chunks, emitting `export-progress` /
  `import-progress` Tauri events (`{file, fileIndex, totalFiles, bytesDone, bytesTotal}`) after
  every chunk, so the ~450MB model file shows real byte progress instead of appearing to hang.
  `export-done`/`import-done`/`export-error`/`import-error` events signal completion, mirroring
  `inference.rs`'s `generation-done`/`generation-error` pattern.
- `pick_folder_dialog` — same `rfd` + 30s-timeout pattern as `persistence.rs`'s
  `pick_save_dialog`/`pick_open_dialog` (see that file's doc comments for why: the OS Common Item
  Dialog can hang on this dev machine, confirmed independent of Tauri/rfd).
- `load_settings`/`save_settings` — read/write `<app-data-dir>/settings.json` (the "data directory
  location" setting), deliberately a separate file from `session.json`.
- `effective_conversations_dir` — resolves to `<dataDir>/conversations` if the setting is
  configured, else `<app-data-dir>/conversations` (the pre-Phase-6 default). `persistence.rs`'s
  `default_conversation_path` now calls this instead of hardcoding the app-data path, which is what
  makes the setting actually affect ongoing auto-save/save/load, not just one-time export/import.

### Frontend side

- `MenuBar.vue` gained real "Export Memory...", "Import Memory...", and "Data Directory
  Settings..." items (the old `console.log` stub is gone).
- `stores/chat.ts` gained `exportMemory`/`importMemory` (pick a folder, invoke the backend command,
  track progress via `memoryOp`) and `loadDataDirSetting`/`setDataDir`, plus
  `onMemoryProgress`/`onMemoryDone`/`onMemoryError` handlers for the four new Tauri events.
- `App.vue` renders a progress modal bound to `store.memoryOp` (file name, X/N counter, a real byte
  progress bar, MB done/total) while an export or import is running, a dismissable toast on
  success/failure, and a small dialog for the data directory setting (folder picker + Save/Use
  Default/Cancel).

### What was actually tested

Verified against the real running dev build (`npm run tauri dev`, launched with
`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9333`), driving the real Tauri
commands via `window.__TAURI_INTERNALS__.invoke(...)` over the Chrome DevTools Protocol — the same
approach Phase 5 used for the checks that needed to bypass the OS file dialog. Not simulated, not
unit-tested in isolation:

1. **Export against a real destination folder**: ran `export_memory` with `destination` set to a
   fresh folder under the Windows temp dir (`%TEMP%\xenon2-export-test`, outside the repo).
   Inspected the result directly — `xenon2-backup\` contained `models\rwkv-5-world-0.4B-Q4_0.bin`
   (453,875,493 bytes), `models\world_vocab.bin` (766,536 bytes),
   `voice-models\en_US-lessac-medium.onnx` (63,201,294 bytes) + its `.onnx.json` sidecar (4,885
   bytes), and `conversations\` with one real saved conversation `.json` (620 bytes) plus
   `session.json` (317 bytes).
   - **Total observed bundle size: 517,849,145 bytes (~494 MB / 494 MiB)** — matches the expected
     ~450MB (model) + ~63MB (voice model) + small JSON, not gigabytes. Confirms the `.pth`
     (923,523,954 bytes) and FP16 (924,161,829 bytes) intermediates — ~1.76GB together — were
     correctly excluded.
2. **Import round-trip, simulating a different machine**: copied the exported `xenon2-backup\`
   folder to a second, independent fresh temp folder (`%TEMP%\xenon2-import-test-source`), then
   pointed the "data directory location" setting at a third brand-new empty temp folder
   (`%TEMP%\xenon2-fake-new-machine-data`, confirmed empty beforehand) to simulate a fresh
   machine's conversation storage having no prior Xenon2 data — real models/voice files can't be
   relocated to a literally different physical machine in this environment, so this is the honest
   boundary of what "different machine" means here: a location with zero pre-existing data,
   reached through the real settings mechanism, not a second physical computer. Ran `import_memory`
   with that copied bundle as `source`. Confirmed:
     - The conversation `.json` (byte-identical content) appeared in the fake new-machine
       conversations folder.
     - `settings.json`'s `dataDir` correctly redirected `effective_conversations_dir` there
       beforehand (`invoke("effective_conversations_dir")` returned the fake folder's path).
     - `session.json` (always written to the real app-data dir, by design — see
       `EXPORT_FORMAT.md`) had its `conversationPaths` rewritten to point at the fake new-machine
       folder, and `load_session_file` read it back correctly.
     - `models\rwkv-5-world-0.4B-Q4_0.bin` and `inference-engine\data\world_vocab.bin` in the real
       repo were overwritten by the import; `md5sum` of the bundle's copy and the repo's
       post-import copy **matched exactly** (`8881a447b56c4dcef2c350c93695e664`), confirming a real,
       correct byte-for-byte copy, not a no-op or a corrupted write.
     - `voice-pipeline\models\en_US-lessac-medium.onnx` (+ sidecar) were similarly overwritten with
       fresh mtimes.
     - After reverting the data directory setting back to `null`, `open_conversation_file` against
       the original real conversation path still returned the correct, unmodified content
       (`"Hello Xenon, this is a Phase 5 persistence test message."` / one user + one assistant
       message) — confirming the import test did not corrupt or lose the pre-existing real
       conversation history on this machine.
3. **Data directory setting takes effect for subsequent saves/loads**: directly invoked
   `save_settings` to set `dataDir`, then `effective_conversations_dir` and confirmed it resolved
   to the new location; `default_conversation_path` (used by auto-save) calls the same function, so
   this is the exact code path a real new conversation's first auto-save goes through. Reverted the
   setting afterward and confirmed `effective_conversations_dir` fell back to the local app-data
   default again.
4. **Rust compiles cleanly**: `cargo build` in `app/src-tauri` succeeds with the new `memory.rs`
   module and `lib.rs`/`persistence.rs` changes (two borrow-checker errors caught and fixed during
   development — cloning `PathBuf`s before moving them into `spawn_blocking` closures).
5. **Frontend typechecks cleanly**: `npx vue-tsc --noEmit` passes with no errors across
   `MenuBar.vue`, `App.vue`, and `stores/chat.ts`'s Phase 6 additions.

**Not verified in this pass**: a full GUI click-through of the new menu items and progress modal
(the OS folder-picker dialog itself has the same known hang risk documented in `persistence.rs` for
the save/open dialogs, and driving it via UI Automation was out of scope for this pass) — the
underlying commands those UI elements call were exercised directly and are the same commands the
UI invokes, so this is a backend/IPC-level verification, not a pixel-level one. A full real-machine
migration (copying a bundle to physical removable media and importing on a second physical computer)
was also not performed — verified instead via the "fresh temp directory with zero prior data"
substitution described above.

## How the Tauri IPC command connects to Phase 1

- `app/src-tauri/build.rs` links this crate against Phase 1's pre-built library, preferring
  `inference-engine/build-cuda-app` (RTX 2060 / sm_75 via `build_cuda.sh`) and falling back to
  `build-cpu-app`. It copies the shared library and `rwkv.dll` / `librwkv.so` next to the
  executable. `load_engine` offloads every RWKV layer when the CUDA library is linked, and
  retries on CPU if GPU init fails, so a CUDA build does not refuse to start with no GPU.
- `app/src-tauri/src/ffi.rs` is a hand-written `extern "C"` binding mirroring
  `inference-engine/src/xenon_inference.h` exactly (opaque `xenon_engine*`, `xenon_load_model`,
  `xenon_generate` with its C callback, etc.) — chosen over `bindgen` since the header is small and
  intentionally stable.
- `app/src-tauri/src/inference.rs`:
  - Loads the model **once** at app startup (`tauri::Builder::setup`), from
    `../models/rwkv-5-world-0.4B-Q4_0.bin` and `../inference-engine/data/world_vocab.bin`
    (resolved relative to the repo root via `CARGO_MANIFEST_DIR`), and keeps the engine pointer in
    Tauri-managed state behind a `Mutex` (only one `xenon_generate` call may run at a time, since
    rwkv.cpp's per-engine state isn't designed for concurrent eval calls).
  - Exposes `#[tauri::command] generate_response(conversation_id, history)`, called from the
    Vue frontend via `invoke("generate_response", ...)` in `App.vue`'s `sendMessage()`. It builds
    the same "User: ... / Xenon: ..." chat-style prompt prime Phase 1's CLI harness uses, then runs
    the actual blocking generation call inside `tauri::async_runtime::spawn_blocking` so it doesn't
    block the Tauri event loop. **As of Phase 8** this no longer re-feeds the whole conversation
    every turn — see "Phase 8: persistent RWKV state" below.
  - The token callback (`on_token`, a real `extern "C" fn`, not a stub) emits a Tauri event
    `token-stream` (payload `{ conversationId, text }`) for every non-empty decoded chunk as
    `xenon_generate` produces it — this is what drives the incremental, token-by-token rendering
    in the chat UI, not a single batched IPC response. `generation-done` / `generation-error`
    events signal completion.
  - Since Phase 5: also derives `ModelInfo` (the loaded model's name/version, e.g.
    `"rwkv-5-world-0.4B-Q4_0"`) from the same `model_path` used to load the engine, so it can never
    drift from what's actually running, managed as Tauri state alongside `EngineState` and exposed
    via `#[tauri::command] get_model_name`. The frontend fetches this once at startup
    (`stores/chat.ts`'s `initModelName`) and records it into every conversation's `model` field.
- `App.vue` listens for these three events (`@tauri-apps/api/event`'s `listen()`), appends
  incoming text directly onto the streaming assistant message's `content` as it arrives, and
  trims the trailing `\n\nUser:` stop-sequence text once generation completes (the model, like
  Phase 1's CLI harness, naturally continues into a fake next turn before the stop-sequence check
  fires — that already-streamed tail is stripped from the rendered bubble only, not from what was
  sent to the model).

This was verified against the real compiled Phase 1 library and the real quantized model, not
stubbed: launching the dev build and sending a message produces an actual RWKV-generated reply,
visibly appearing token-by-token.

Phase 4's edit/regenerate flows were verified the same way (real dev build, real model, not
code-reviewed only): a 6-message conversation had its first user message edited, which correctly
truncated the other 5 messages and streamed a fresh reply (2 messages remained, `edited: true`
shown); a 4-message conversation had its first (non-last) assistant message regenerated, and
comparing a JSON dump of all four messages before and after confirmed the other three were
byte-for-byte unchanged while only the targeted message's content changed.

## Phase 7: hardening & voice input/output

Follow-up pass per `../prompts/phase7_prompt.md` -- closes gaps a project review found after
Phase 6, not new user-facing scope beyond what that prompt calls for.

### Voice input/output (`app/src-tauri/src/voice.rs`, `voice-pipeline/ipc_server.py`)

Phase 2's VAD/STT/TTS pipeline (`voice-pipeline/`) was standalone Python, never reachable from the
desktop app. Two integrations were possible: port VAD/STT/TTS to Rust bindings, or run Phase 2's
existing code as a sidecar process. Chosen: **sidecar** -- porting would mean re-verifying
faster-whisper/piper/silero-vad's GPU/CPU device placement and the CUDA-12-vs-13 DLL workaround
(see `../voice-pipeline/README.md`) against a different binding layer, for libraries that only ship
first-class Python packages. The sidecar reuses Phase 2's exact code unchanged, including
`IncrementalSpeaker`'s sentence-chunked streaming TTS.

- `voice-pipeline/ipc_server.py`: a long-lived process, spawned once at app startup
  (`voice::spawn_voice_process`, called from `lib.rs`'s `setup`), speaking newline-delimited JSON
  over stdin/stdout. Loads VAD/STT (`small`, `cuda` -- safe since this app links Phase 1's
  **CPU-only** RWKV build, so the GPU is otherwise idle) and TTS once at startup (a few seconds,
  matching Phase 2's measured load times), then serves `listen` (mic capture -> VAD gate -> STT ->
  transcript) and `speak_start`/`speak_feed`/`speak_finish` (drives an `IncrementalSpeaker`
  unchanged from Phase 2) commands. Deliberately does **not** call `xenon_generate()` -- Phase 1's
  RWKV engine is already loaded and owned by `inference.rs`; loading a second copy in Python would
  waste memory and contend for the same GPU/CPU, and per the phase's explicit requirement, voice
  transcripts must feed the *same* `sendMessage` path typed text uses, not a parallel one.
- `app/src-tauri/src/voice.rs`: spawns the sidecar, demultiplexes its stdout (a reader thread
  routes `speak_done` directly to a `voice-speak-done` Tauri event; everything else to a channel
  the current command is awaiting), and exposes `voice_ready`, `voice_listen`,
  `voice_speak_start`/`voice_speak_feed`/`voice_speak_finish` as Tauri commands. A spawn failure
  (e.g. the venv isn't set up in some checkout) is recorded, not panicked on -- commands then fail
  fast with a clear message and typed chat keeps working.
- Frontend: `MessageInput.vue`'s mic button calls `voice_listen` (awaited); on a successful
  transcript it emits `voice-send`, which `App.vue`'s `onVoiceSend` feeds into
  `store.sendMessage(text, true)` -- the exact same function typed sends call, just with a
  `viaVoice` flag. That flag (`store.currentTurnIsVoice`) is read by the *existing*
  `token-stream`/`generation-done`/`generation-error` listeners (unchanged from Phase 3/4) to also
  forward text to `voice_speak_feed`/`voice_speak_finish` -- there is no separate voice code path,
  per the phase's explicit requirement.

**Verified against the real running dev build** (`npm run tauri dev`, driven both via
`window.__TAURI_INTERNALS__.invoke` over Chrome DevTools Protocol -- the same approach Phase 5/6
used -- and via real UI Automation clicks):
- Sidecar starts and loads all three models on real app launch (log: `loading VAD... / loading STT
  (faster-whisper, small, cuda)... / loading TTS (piper)... / ready`); `voice_ready` correctly
  returns `false` during that window and `true` after.
- `voice_speak_start`/`voice_speak_feed`/`voice_speak_finish` invoked directly against the running
  app produced real, audible piper TTS playback (confirmed both via a standalone sidecar test
  driving real audio and again through the actual Tauri command).
- A full voice-flagged turn (`store.sendMessage(text, true)`, mirroring exactly what
  `onVoiceSend` does) produced a real user message + a real RWKV-streamed reply, with the
  already-registered `token-stream` listener forwarding to `voice_speak_feed` throughout --
  confirming the "same code path, flag-gated speaking" wiring, not a parallel implementation.
- `voice_listen` in a silent room with a short timeout correctly returned
  `{ok: false, reason: "no_speech_timeout"}` in ~3s without hanging -- the VAD gate's graceful
  no-speech path (Phase 2) works end-to-end through the real sidecar and real mic device. (One
  earlier run, executed immediately after a TTS test on the same speakers, returned a hallucinated
  transcript instead of a timeout -- consistent with Whisper's known tendency to hallucinate short
  phrases from ambient/residual audio rather than a bug in the VAD gate; not reproducible in a
  quiet room afterward.)
- **Not verified**: an actual human speaking a real utterance into the mic and confirming the
  transcript/reply end-to-end. This dev machine does have a working microphone (confirmed via
  `sounddevice.query_devices()` -- "Microphone Array (Intel Smart Sound Technology)"), but no
  automated agent session can produce real spoken audio into it. This is the one part of Task 5
  that requires a human at the keyboard to actually test.

### Real delete-message

See the stub table above and `stores/chat.ts`'s `deleteMessage` doc comment. Verified against the
real running dev build via Chrome DevTools Protocol (`window.__store`): sent a real message
(2 messages: 1 user + 1 real RWKV-streamed reply), called `deleteMessage` on the user message id,
confirmed the conversation dropped to 0 messages (both the user message and its paired reply were
removed together, per the chosen semantics).

### Model-mismatch warning

`stores/chat.ts`'s `modelMismatchWarning` getter (frontend-only -- no backend change needed;
`persistence.rs`'s `open_conversation_file` already records and returns the file's `model` field,
Phase 5 just never compared it). Non-blocking, dismissable per-conversation (tracked via
`modelMismatchDismissed`, reset whenever a different conversation becomes active via the new
`setActive` helper every `activeId`-changing call site now goes through). Verified live: set an
active conversation's `model` to a fake different value -- banner appeared with the exact expected
text; dismissed it -- banner disappeared; reset the model back to match the loaded model -- banner
stayed gone (not just dismissed-and-forgotten).

### Full GUI click-through of Export/Import/Data Directory Settings

Driven via real UI Automation clicks (`InvokePattern`, `PrintWindow` screenshots -- see Phase 4's
notes above for why), using the existing `XENON2_TEST_*_DIALOG_PATH`-style escape hatches
(`XENON2_TEST_EXPORT_DEST_PATH`, `XENON2_TEST_IMPORT_SOURCE_PATH`, `XENON2_TEST_DATADIR_PICK_PATH`
-- `pick_folder_dialog` already took a generic `test_env_var` param per call site, so no Rust
change was needed) to get past just the OS folder-picker step, with everything else -- clicking
File, clicking the menu item, the progress modal, the completion toast -- driven and screenshotted
for real:
- **File > Export Memory...**: real click through the File menu opened a real progress flow and
  ended in a real toast reading "Export complete: 8 file(s), 493.9 MB." -- confirmed the files
  actually landed on disk at the destination (`models/`, `voice-models/`, `conversations/`).
- **File > Import Memory...**: same click-through against a bundle built from the export above,
  toast read "Import complete: 7 file(s), 493.9 MB."
- **File > Data Directory Settings...**: opened via a real click, `Browse...` populated the path
  input with the (test-override) folder, `Save` closed the modal, and `load_settings` confirmed
  `dataDir` was actually persisted to the chosen path -- then reset back to `null` (local app-data
  default) afterward so the dev environment wasn't left pointed at a deleted temp folder.

**A real bug found and fixed during this verification**: the File dropdown closed via `@blur` on
the trigger button, guarded only by `@mousedown.prevent` on the dropdown -- that guard stops a
*real mouse click* from shifting focus before its own click lands, but UI Automation's
`InvokePattern` (used for this very verification, and for any screen reader or accessibility
client) shifts focus to the target element as part of invoking it, firing blur on the File button
first and closing the dropdown before the invoked item's own click finished -- so the click
silently never landed. Fixed by closing on an outside `mousedown` (tracked via a template ref)
instead of on blur, which has no such race for any input method, and by changing the "File" trigger
from a plain `<div tabindex="0">` to a real `<button>` (needed for `InvokePattern` support at all,
and better semantics/accessibility regardless of automation).

### App icon

`app/src-tauri/icons/source-icon.png` (a green alien in a flying-saucer/UFO, provided alongside
`prompts/phase7_prompt.md`) was already regenerated into the full icon set via the Tauri CLI's
icon generator before this pass began (`icon.ico`, `icon.icns`, and the PNG sizes under
`src-tauri/icons/`, plus `android/`/`ios/` variants) -- `tauri.conf.json`'s `bundle.icon` array
already pointed at the generated files, no update needed. Confirmed the new icon actually renders
on the live window (not just that the files changed on disk): a `PrintWindow` screenshot of the
real running dev build shows the alien/UFO icon in the window's title bar, replacing Tauri's
default.

### Git cleanup

`prompts/phase1_prompt.md`'s modification was a legitimate documented follow-up fix (auto-copying
`world_vocab.bin` next to `test_inference.exe`), committed with an honest message. The
`inference-engine/rwkv.cpp` submodule showed as dirty only because of untracked local build output
directories (`build-cpu/`, `build-cuda/`) inside it, not a pinned-commit change -- fixed by adding
`ignore = untracked` to that submodule's `.gitmodules` entry, which is the standard fix for
"submodule has build artifacts, not real changes." Both fixes are committed; this phase's actual
feature work (voice, delete, model-mismatch, icon) is left uncommitted for review, per the same
convention Phases 1-6 followed.

## Phase 7 follow-up: model upgrade + Dementia/Sloth persistent memory

Real usage after Phase 7 shipped surfaced two further issues, fixed in the same phase: the
0.4B model's reply quality, and the lack of any memory that survives outside one chat window.

### Model upgrade: RWKV-5 World 0.4B -> RWKV-7 World v3 2.9B

Real conversations showed the 0.4B model collapsing onto a near-identical canned reply ("I'm
having trouble with my voice assistant...") for short/greeting-style prompts regardless of what
was actually asked, and getting basic arithmetic wrong (`Two plus two is three.`) -- reproduced
independent of the app via Phase 1's own CLI harness, so this was a real model-quality ceiling,
not an app bug. Root-caused and evidenced in conversation before this fix (see git history around
2026-08-01 if resuming from an older checkout).

Upgraded to **RWKV-7 ("Goose") World v3, 2.9B params** (`RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth`
from `BlinkDL/rwkv-7-world` on Hugging Face), quantized to **Q5_1** (`rwkv-7-world-2.9B-Q5_1.bin`,
2.75GB). Same World tokenizer as before (no tokenizer/vocab changes needed); same rwkv.cpp
conversion pipeline Phase 1 established (`convert_pytorch_to_ggml.py` then `quantize.py`) --
rwkv.cpp already supports the v7 architecture (the vendored submodule's pinned commit is in fact a
v7-conversion fix, `blocks.0.att.v[0,1,2]` tensors).

**Why 2.9B and not bigger**: RWKV-7 tops out at 2.9B officially (no RWKV-7 7B exists yet; a 7B
option would mean the older, less parameter-efficient v6 architecture and a much heavier
download/runtime). The RWKV-7 paper reports `RWKV7-World3-2.9B` averaging 71.5% across English
benchmarks (5.6T training tokens) versus Qwen2.5-3B's 71.4% (18T training tokens) -- a strong
result for the parameter count, still a base (non-instruction-tuned) model like before.

**Benchmarked before committing to CPU-only** (this dev machine, i7-12850HX / RTX A1000 4GB):

| Mode | Time to first token | Throughput | VRAM |
|---|---|---|---|
| CPU-only, 6 threads | 2.0s | **8.95 tok/s** | -- |
| CPU-only, 12 threads | 2.0s | 7.00 tok/s (thread oversubscription hurts) | -- |
| GPU-offloaded, 32/32 layers | 2.3s | 11.18 tok/s | 2.1GB (fits alongside the voice pipeline's ~945MB Whisper `small`) |

GPU is ~25% faster now (unlike the 0.4B result, where CPU won). The app now links the CUDA
build when `build-cuda-app` exists and offloads every layer. An RTX 2060 has 6GB; this table's
2.1GB working set fits, including next to Whisper. If CUDA init fails, `load_engine` retries
on CPU, so the binary still launches with no NVIDIA GPU.

Verified qualitatively via the CLI harness after the swap: `"What is 2 plus 2?"` ->
`"Two plus two equals four."` (correct); a multi-turn continuation self-generated by the model
handled `sqrt(8) ≈ 2.8` reasonably. Not perfect (still a small base model, still base-model-typical
drift into fake follow-up turns), but a real, measurable step up from the 0.4B collapse pattern.

### Real system clock grounding (2026-08-04)

The "what time is it" hallucination from the earlier diagnosis was never fixed by the repetition-
penalty change -- it's a separate gap (no real clock access at all). Fixed by prepending the real
system date/time (via the new `chrono` dependency, `chrono::Local::now()`) to the start of every
prompt, for both agents (this is basic environmental grounding, not memory, so it isn't gated
behind Sloth).

A bare grounding line alone was **not reliably used** -- tested side by side: a more elaborate
phrasing ("What is today's date and what time is it right now?") correctly used the real time, but
the more common short phrasing ("What time is it?") still reverted to a hallucinated guess. Fixed
by adding a demonstrated example turn for exactly this question, using the real computed time (not
a fixed fake value, so the example is never factually wrong and there's no risk of the model
anchoring on a stale or made-up value the way a hardcoded example would). Re-verified across three
phrasings ("What time is it?", "what's the date today?", "Do you know the time?") -- all three
correctly reported the real system time to the minute, confirmed against the actual system clock
each time.

**Still not reliable enough in further real usage**: even with the demonstrated example, the model
would report a plausible but *imprecise* time (e.g. "8:05 PM" when it was actually 8:10) and
sometimes drop AM/PM -- a known small-model weakness: precisely reproducing a specific number from
context isn't guaranteed even when the correct value is sitting right there in the prompt. Since
time/date has one objectively correct answer, there's no reason to leave it up to a small model's
approximation once the question can be reliably detected. Replaced prompt-grounding with a
deterministic path (`try_answer_time_date_deterministically` in `inference.rs`): a simple keyword
heuristic on the latest user message ("what time", "current time", "what day", "today's date",
etc.) bypasses `xenon_generate` entirely and emits the exact real time/date directly as a
`token-stream` event, formatted TTS-friendly (non-padded hour/day, e.g. "8:10 PM" not "08:10", so
piper doesn't sound out "zero eight ten"). Verified across four phrasings, including a combined
"what day and what time" question -- all exact, instant (no model latency at all for these), and
confirmed against the real system clock at query time.
### Dementia/Sloth: a mid-conversation memory-agent toggle

Per-conversation history (Phase 3 onward) already gave the model "memory" *within* one chat --
but nothing survived across separate conversations, and there was no way to distinguish "no
memory wanted" from "forgot to build it." Added a toggle (`MessageInput.vue`, next to the mic
button) between two named agents, switchable turn-by-turn within an open conversation (not a
per-conversation setting -- a single chat can mix both):

- **Dementia** (default): exactly the pre-existing behavior. No memory outside the current chat
  window; that chat's own history still works normally and still saves to disk like any
  conversation.
- **Sloth**: additionally reads a persistent, cross-conversation fact store
  (`<app-data-dir>/sloth_memory.json`, new module `app/src-tauri/src/sloth_memory.rs`) and injects
  it into every Sloth-mode prompt as a "Known facts about the user" preamble, and after every
  successful Sloth reply, runs a second small/low-temperature `generate()` call asking the model
  whether the exchange revealed anything new and durable worth remembering, appending it to the
  store if so (capped at the 30 most recent facts, oldest dropped first, so the preamble can never
  alone blow past the model's context window).

Each assistant message records which agent generated it (`ChatMessage.agent`, `"dementia"` |
`"sloth"`, persisted -- see `SCHEMA.md`); `ChatMessage.vue` shows a small "Sloth" badge on those
replies (no badge for Dementia, the unmarked default, to avoid clutter). A "File > Sloth
Memory..." panel (`App.vue`) lists every stored fact with a per-fact delete and a "Forget
Everything" clear-all.

**A real failure mode found and fixed while verifying this**: the first extraction prompt design
(a bare "extract a fact or say NONE" instruction) made the small base model *invent* a
plausible-sounding fact ("named John, loves pizza") from an exchange that revealed nothing --
confirmed by inspecting `sloth_memory.json` directly and seeing fabricated content next to the one
real fact. Fixed two ways: (1) rewrote the extraction prompt to be few-shot (one example that
should extract, one that should say `NONE`) rather than an abstract instruction -- base models
follow demonstrated patterns far better than instructions, the same lesson the main chat prompt
already relies on; (2) added a Rust-side safety net (`run_fact_extraction` in `inference.rs`)
rejecting any "fact" that's just an echo of the assistant's own reply text, a second failure mode
that surfaced even after the prompt fix. This is why the Sloth Memory panel supports deleting
individual facts -- extraction from a small model won't be perfect, and the UI needs to make bad
entries correctable rather than permanent.

### Repetition-penalty engine fix

A deeper cause behind "it doesn't understand me part of the time": every `generate_response` call
resends the **entire conversation history as text** (there's no incremental RWKV state across
calls -- see `inference.rs`'s doc comments). Once the model produced a canned phrase like *"I'm
sorry, I'm not able to access my memories"* one time, that exact sentence became part of every
subsequent prompt, and the model started imitating its own recent phrase for unrelated follow-up
questions instead of answering them. `xenon_inference.cpp` had **no repetition penalty at all** --
just temperature + top-p.

Added one (`app/src-tauri/src/inference.rs` -> `ffi.rs` -> `inference-engine/src/xenon_inference.h`/
`.cpp`, plus the CLI harness and the Python `xenon_engine.py` ctypes binding, kept in sync): the
standard llama.cpp-style penalty (divide positive logits / scale up negative ones for any token
seen recently, before sampling), applied against a window of the **last 256 tokens of the prompt
tail plus everything generated so far this call** -- deliberately including the prompt tail, not
just this call's own output, since the actual failure mode was echoing a phrase from *earlier
conversation history*, not from within a single reply.

**Empirically tuned against the real failure case, not guessed**: replayed the literal prompt from
the bad conversation (a phrase already repeated 3 times in history) through the CLI harness at
different penalty values.
- `1.0` (no penalty) and `1.15` (llama.cpp's typical low end): **still fell into the trap**,
  repeating the canned phrase a 4th time for "where are you pulling your time from?"
- `1.3` (llama.cpp's typical high end): broke free, produced a real answer ("My time is based on
  the current date and time as determined by the system's internal clock...").
- Re-verified `1.3` causes no quality loss on normal exchanges (correct arithmetic, coherent fun
  facts, natural greetings) before committing to it as the shipped value.

**Verified end-to-end against the real running app**: replayed the exact 5-message sequence from
the original bad conversation through the live `generate_response` command. Before the fix, this
sequence collapsed to the same repeated sentence; after the fix, it produced five distinct,
contextually-varied replies -- no repetition collapse.

**Verified against the real running dev build** (Chrome DevTools Protocol driving the real Pinia
store + real `generate_response`/`sloth_memory` Tauri commands, plus a real GUI click-through of
the toggle and the Memory panel):
- A Sloth-mode message ("My name is Alex and I love astronomy.") produced a real extracted fact
  (`"Alex loves astronomy."`), confirmed byte-for-byte in `sloth_memory.json` on disk.
- **A brand-new, unrelated conversation, in Sloth mode**, asked "What is my name and what do I
  love?" correctly answered **"Your name is Alex and you love astronomy."** -- real
  cross-conversation recall, not just in-chat history.
- The same question in **Dementia mode, in a fresh chat, did not know the name** -- confirming the
  two agents are actually isolated, not just cosmetically labeled.
- The Sloth Memory panel, opened via a real File-menu click (not just an `invoke()` call), showed
  both stored facts with working per-fact delete buttons -- screenshotted via `PrintWindow`.

## Phase 8: persistent RWKV state (stop re-prefilling the prompt)

Every point above that describes `generate_response` re-sending "the entire conversation history
as text" on every call, because "there's no incremental RWKV state across calls", was true through
Phase 7 and is no longer true. `inference-engine`'s `xenon_generate()` used to reset RWKV state on
every call, so `inference.rs`'s prompt builder had to re-serialize the whole conversation as text
every turn, and the engine had to re-evaluate all of it before producing one new token -- a cost
that grew without bound the longer a conversation went on. RWKV's whole point is a fixed-size
recurrent state that makes that unnecessary; this phase actually uses it.

**What changed:**
- `inference-engine` gained a caller-owned `xenon_state` type plus `xenon_prefill` /
  `xenon_generate_with_state` (full details, API, and the correctness proof in
  `inference-engine/README.md`'s "Persistent state across calls" section). `xenon_generate()`'s
  signature and behavior are unchanged -- it's now a thin wrapper over the new API, so the two
  existing direct callers of that signature (`voice-pipeline/xenon_engine.py`, `test_inference.cpp`)
  needed no changes.
- `inference.rs`'s old single `build_prompt()` split into three pieces: a `STATIC_PREFIX` (the
  instruction + demo turns, byte-identical forever, prefilled once at engine load), `history_text`
  (the growing per-conversation history, cached and extended incrementally), and
  `volatile_header_text` (the real-clock grounding + Sloth facts + time-example turn, which has to
  be re-fed every single turn regardless of caching since it changes every call). The volatile
  header moved from the *front* of the prompt to *just before* the new turn -- a value that
  changes every call can't sit at the front of a prompt whose tail is supposed to be cacheable,
  since RWKV state is strictly sequential (change position 0, invalidate everything after it).
  Wording and content are unchanged from Phase 7, only the ordering moved; sanity-checked against
  the old ordering on real prompts (coherent, on-topic, correctly-answered arithmetic, no
  regression in the "doesn't end every reply with a question" style fix) before shipping it.
- `EngineState`'s inner type grew from a bare engine pointer to an `EngineInner` holding the
  engine, the one-time static-prefix state, a separate one-time state for the Sloth
  fact-extraction preamble, and a small LRU cache (`MAX_CACHED_CONVERSATIONS = 4`) of
  per-conversation state. Each incoming `generate_response` call diffs the history it's sent
  against what a conversation's cache entry has already consumed: an unchanged prefix means
  "extend" (prefill just the new tail turns); a divergence anywhere (an edit to an earlier
  message, Phase 7's delete-message feature, or a shorter history) means "rebuild" from the shared
  static prefix. The cache only ever advances from history text the frontend already sent (and
  therefore already considers canonical) -- generation for the turn being answered right now
  always runs on a throwaway copy, never the authoritative cache entry, so the frontend's
  post-generation trimming (`chat.ts`'s `completeGeneration`, which strips the stop-sequence tail
  and a couple of canned-phrase backstops) can never desync the cache from what's actually shown.
  Sloth fact-extraction runs against its own separate one-time state and never touches a
  conversation's cached state at all.

**Measured (RWKV-7 World 2.9B, this machine; full numbers and the correctness proof in
`inference-engine/README.md`):** `xenon_get_state_len()` is 5,406,720 floats (20.62 MB) for this
model. Tokens actually evaluated per turn dropped from a pattern that grows without bound (167 at
turn 1, 420 by turn 10) to a flat ~83/turn regardless of how long the conversation gets. Wall time
for a 20-token reply at turn 10 dropped from 16.97s to 4.65s on this machine. Steady-state
per-token generation throughput is unaffected -- confirmed unchanged, since the generation loop
itself wasn't touched, only what it reads/writes state from/to.

**Correctness, proven not assumed**: `inference-engine/src/test_state_cache.cpp` asserts
byte-identical output (text *and* token-id sequences) between the old full-reset-and-refeed path
and the new incremental path, at turn 1, turn 5, after an edited earlier turn, and after a
regenerate -- all four pass. Uses greedy decoding (`temperature = 0`) for full determinism without
needing an RNG-seeding API, and keeps the repeat-penalty enabled (`1.3`, the production value)
specifically because that's the part most likely to be subtly wrong under incremental prefill (see
`inference-engine/README.md` for why it isn't).
