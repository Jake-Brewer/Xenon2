# Xenon2 — Portable Export Bundle Format (Phase 6)

This document defines the on-disk layout Xenon2 writes when exporting "memory" (models + voice
model + conversation history) to an external destination (USB drive, external SSD, another internal
drive), and what Import expects to find. It's the Phase 6 counterpart to `SCHEMA.md` (which
documents the shape of one conversation file) — this doc is about the *bundle* the export/import
commands actually copy files into/out of, implemented in `app/src-tauri/src/memory.rs`.

## Why the real layout differs from `PLAN.md`'s original sketch

`PLAN.md`'s Phase 6 section sketches `<root>/xenon2-data/models/` and
`<root>/xenon2-data/projects/` — a placeholder written before Phase 5 decided where things actually
live. The real, verified-on-disk locations (see `app/README.md`'s "Phase 5: file persistence"
section) are:

- Conversations: `<Tauri app-data-dir>/conversations/<conversation-id>.json` +
  `<Tauri app-data-dir>/session.json` (or, if the Phase 6 "data directory location" setting is
  configured, `<configured-dir>/conversations/...` instead — see "Data directory location setting"
  below).
- The quantized model actually loaded at runtime: `models/rwkv-7-world-2.9B-Q5_1.bin` (repo root;
  upgraded from the original `rwkv-5-world-0.4B-Q4_0.bin` in a Phase 7 follow-up -- see
  `app/README.md`'s "Model upgrade" section).
- The tokenizer vocab required to load it: `inference-engine/data/world_vocab.bin`.
- The piper TTS voice model used by the standalone voice pipeline:
  `voice-pipeline/models/en_GB-alan-medium.onnx` + its `.onnx.json` sidecar.

This document (and the export/import commands) are built against those real paths.

## Bundle layout

```
<destination>/xenon2-backup/
  models/
    rwkv-7-world-2.9B-Q5_1.bin       # the quantized RWKV model actually loaded (~2.75GB)
    world_vocab.bin                  # tokenizer vocab, required to load the model above
  voice-models/
    en_GB-alan-medium.onnx         # piper TTS voice model (~63MB)
    en_GB-alan-medium.onnx.json    # its required sidecar (piper needs both files)
  conversations/
    <conversation-id>.json           # one file per saved conversation, per SCHEMA.md
    session.json                     # last-active conversation + path bookkeeping (see below)
  memory/
    sloth_memory.sqlite              # Sloth's facts and their embeddings (sloth_memory.rs)
  feedback/
    feedback.jsonl                   # reply ratings for learning from feedback (feedback.rs)
```

Notes:

- `world_vocab.bin` is placed under `models/`, not mirroring its real `inference-engine/data/`
  location — the bundle's job is to be a single self-contained, portable thing a non-technical
  destination folder can hold, not a mirror of the repo's internal directory structure. Import maps
  it back to `inference-engine/data/world_vocab.bin` on the destination machine.
- Export always creates the bundle inside a fixed `xenon2-backup/` subfolder of whatever
  destination folder the user picks (so picking, say, the root of a USB drive `E:\` doesn't scatter
  loose files across it) — the resulting path is `E:\xenon2-backup\...`.
- Import accepts either the bundle folder itself (containing `models/`/`voice-models/`/
  `conversations/` directly) or its parent (in which case it looks for `xenon2-backup/` one level
  down) — so pointing Import at either the drive root or the bundle folder both work.

## What's excluded from a default export, and why

`models/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth` and `models/rwkv-5-world-0.4B-FP16.bin` are
conversion intermediates — the original downloaded checkpoint and an unquantized intermediate used
to produce the quantized `.bin` Xenon2 actually loads. Neither is read by the running app. Together
they're **~1.8GB** — including them by default would nearly triple the export size for zero
functional benefit (a destination machine can already run everything it needs from the quantized
model alone). They are not included by default, and this phase does not add an opt-in "full/dev
export" mode for them (no current need for one) — a future phase could add one if reproducing the
quantization step on another machine ever becomes a real requirement.

## Import: copy-in vs. run-in-place

Two designs were possible for Import: (a) copy the bundle's contents into this machine's real local
paths, or (b) run directly against files still sitting on the external drive/folder.

**Copy-in is the only mode implemented, and it's the default (there is no run-in-place option).**
Reasons:

- The app's model is loaded once at startup from a fixed path derived from the repo root
  (`inference.rs`'s `load_engine`, called from `lib.rs`'s `setup` hook) — there's no live "point the
  running engine at a different model file" mechanism, so "run in place" would only ever help
  *before* the app starts, not after Import runs inside a live session. Building a hot-swap path
  for the inference engine is out of scope for this phase.
- USB/external drives are the exact case where "run in place" is riskiest: they can be unplugged,
  sleep/spin down, or simply not be present the next time the app launches, which would turn "my
  external drive holding my conversation history" into "my app that can no longer find its files."
  Copying in means the destination machine has a fully working, disconnected-drive-safe local copy
  after Import completes, matching the acceptance criterion ("the app resumes with full model +
  conversation history" — not "as long as the drive stays plugged in").
- It matches the existing precedent in this codebase: Phase 5's auto-save already prefers a
  guaranteed-present local default over anything that could later go missing (see `SCHEMA.md`'s
  "Auto-save default path policy").

Concretely, Import copies:
- `<bundle>/models/rwkv-5-world-0.4B-Q4_0.bin` → `<repo-root>/models/rwkv-5-world-0.4B-Q4_0.bin`
- `<bundle>/models/world_vocab.bin` → `<repo-root>/inference-engine/data/world_vocab.bin`
- `<bundle>/voice-models/*` → `<repo-root>/voice-pipeline/models/*`
- `<bundle>/conversations/*.json` (excluding `session.json`) → the machine's effective
  conversations directory (the app-data default, or the configured data directory — see below)
- `<bundle>/conversations/session.json` is not copied byte-for-byte: its `conversationPaths` are
  **rewritten** to point at this machine's effective conversations directory (the source machine's
  original paths are meaningless on a different machine), then written to this machine's
  `session.json`. `lastActiveConversationId` is carried over unchanged.

Sloth's facts and the ratings are merged, not overwritten: import adds the facts and ratings this
machine does not already have, so neither machine's are lost.

Existing local files with the same name are overwritten — Import is meant for "restore/replace this
machine's memory from a bundle," not a non-destructive merge. If a bundle is imported onto a machine
that already has differently-named conversations, those are left untouched (only same-named/same-id
files are overwritten); a destructive full-directory wipe-then-copy was deliberately not implemented
to avoid silently deleting conversations the bundle never mentions.

## Data directory location setting

Export/Import are one-time operations. The separate "data directory location" setting
(`app/src-tauri/src/memory.rs`'s `Settings`, persisted at `<app-data-dir>/settings.json` — a small
file, deliberately separate from `session.json`, which is restore-on-launch bookkeeping rather than
user configuration) is for *ongoing* storage: once set, every new conversation's auto-save default
and every default-path lookup (`persistence::default_conversation_path`) resolves against
`<configured-dir>/conversations/` instead of `<app-data-dir>/conversations/`. This is what lets an
external drive serve as primary storage, not just an export target. Model files are not currently
relocated by this setting — see `app/README.md` for the reasoning and scope boundary.

## What was actually tested

Full account in `app/README.md`'s "Phase 6: external memory export/import" section. Short version:
a real `export_memory` run against a fresh temp folder produced a 517,849,145-byte (~494MB)
`xenon2-backup/` bundle (model 453,875,493 bytes + vocab 766,536 bytes + voice model 63,201,294
bytes + sidecar 4,885 bytes + one real conversation + session.json) — confirming the `.pth`/FP16
intermediates (~1.76GB combined) were excluded. That bundle was copied to a second independent temp
folder and imported after pointing the "data directory location" setting at a third, empty temp
folder to simulate a fresh machine's conversation storage; the imported model file's md5sum matched
the exported copy's exactly, the conversation JSON and a rewritten `session.json` appeared at the
simulated new location, and the real pre-existing local conversation was confirmed unaffected
afterward. All verified by directly invoking the real Tauri commands
(`window.__TAURI_INTERNALS__.invoke(...)`) against the real running dev build over the Chrome
DevTools Protocol, not simulated or unit-tested in isolation.
