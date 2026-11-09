# Xenon2 repo map

Xenon2 is an offline voice and text assistant. A Tauri desktop app (Rust + Vue) runs an RWKV-7
language model locally (no cloud), talks and listens through a Python voice sidecar, and serves
a coding API that a VS Code extension uses. RWKV is a recurrent model with no context window.

## Top-level docs
- PLAN.md: the build plan, phase by phase, with what was built, measured, and decided.
- SCHEMA.md: the JSON format of saved conversations.
- EXPORT_FORMAT.md: the layout of a memory export (models, voice models, conversations).
- ONLINE_LEARNING.md: research notes on learning from good/bad ratings of replies.
- prompts/phase1..8_prompt.md: the original task brief for each build phase.
- models/ (not in git): the RWKV model files the app loads.

## app/ (the desktop app)
- src/App.vue: main window; wires chat, voice, menus, and the Stop button to the backend.
- src/stores/chat.ts: Pinia store, the source of truth for conversations, saving, generation state.
- src/components/: ChatMessage.vue (one bubble, edit/regenerate/rate), MessageInput.vue (text box
  and mic), Sidebar.vue (conversation list), MenuBar.vue (File menu).
- src/persistence.ts, src/types.ts: save/load helpers and the shared TypeScript types.
- scripts/run_desktop.mjs: launcher for `npm run tauri` (finds cargo, fixes Linux GTK env).
- src-tauri/src/lib.rs: app entry; loads the model, starts the code API and voice sidecar,
  registers the Tauri commands.
- src-tauri/src/inference.rs: runs the RWKV engine: prompt building, state caching, streaming
  tokens, stop sequences, Stop, chat and coding generation.
- src-tauri/src/ffi.rs: Rust declarations for the C engine in inference-engine/.
- src-tauri/src/code_api.rs: local OpenAI-compatible HTTP API on 127.0.0.1:47811 with a bearer
  token (chat completions, feedback, folder agent).
- src-tauri/src/folder_agent.rs: autonomous loop that edits Python files in a folder, runs their
  tests, and stops on success or budget; git snapshot for undo; GPU heat throttle.
- src-tauri/src/feedback.rs, example_bank.rs: reply ratings saved to feedback.jsonl, and rated or
  corrected replies reused as examples in later prompts (learning without training).
- src-tauri/src/sloth_memory.rs: long-term facts for the "Sloth" agent, found by meaning.
- src-tauri/src/memory.rs: memory export/import and the data folder setting.
- src-tauri/src/persistence.rs: saving and opening conversation files.
- src-tauri/src/voice.rs: starts and talks to the Python voice sidecar.

## inference-engine/ (the C++ model runtime)
- src/xenon_inference.h/.cpp: C API over rwkv.cpp: load a model, keep/copy states, generate
  tokens with a callback.
- src/world_tokenizer.*: the RWKV World tokenizer; data/world_vocab.bin is its vocabulary.
- src/test_inference.cpp, test_state_cache.cpp: command-line test programs.
- rwkv.cpp/: git submodule, the upstream RWKV inference library (ggml, CPU and CUDA).
- build_cpu.sh, build_cuda.sh: builds; CUDA targets the RTX 2060.

## voice-pipeline/ (Python sidecar)
- ipc_server.py: the long-running process the app talks to over stdin/stdout.
- pipeline.py: mic -> VAD -> speech-to-text -> model -> text-to-speech chain.
- vad.py (silero-vad, CPU), stt.py (faster-whisper, GPU), tts.py (piper, CPU).
- audio_io.py, echo_canceller.py, mic_level.py: capture, playback, echo cancellation, mic level.
- speech_streamer.py: speaks a reply while it is still being generated.
- embedder.py: sentence embeddings (MiniLM) for memory and example search.
- xenon_engine.py: Python bridge to the C inference engine; cli_harness.py: command-line test.

## vscode-xenon/ (VS Code extension)
- src/extension.ts: commands, status bar, @xenon chat participant, workspace overview.
- src/chatView.ts + media/chat.js, chat.css: the sidebar chat with ratings.
- src/client.ts: talks to the app's coding API (streaming, feedback, folder agent).
- src/context.ts: chooses which code and workspace info go with a question.
- src/edit.ts: "Edit This File" whole-file rewrite shown as a diff.
- src/folderAgent.ts: "Improve Files in This Folder" command and its report.

## training/ (offline model improvement)
- make_python_sft.py, make_python_data.py: build Python coding training sets.
- eval_code.py: scores Xenon's Python answers by running their tests (pass@1).
- export_feedback.py: turns app ratings into a training set.
- state_tune.py: state-tuning RWKV on the 6 GB RTX 2060; make_spike_data.py: a test set for it.
