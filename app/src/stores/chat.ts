// stores/chat.ts -- Phase 4's Pinia store, extended in Phase 5 with on-disk persistence. Single
// source of truth for conversation/message state, replacing the plain refs App.vue used to own
// directly (Phase 3). App.vue still owns Tauri event-listener registration and DOM concerns
// (scrolling) -- it just calls into this store instead of mutating local refs.
//
// Key design point for edit/regenerate: rather than assuming "the streaming message is always the
// last message in the array" (true for a brand-new send, but NOT true for a regenerate targeting
// an earlier assistant message), this store tracks which message id is currently streaming per
// conversation (`streamingMessageIds`). Token/done/error handlers look the target message up by
// id, so regenerating message #2 in a 5-message conversation updates message #2 in place no
// matter what -- it never touches messages #1, #3, #4, #5.
//
// Phase 7: delete is now implemented (see `deleteMessage` below) -- Phase 4 left it an intentional
// stub (a real gap in PLAN.md's task list, not an oversight); this phase closes it. Deleting a
// user message also removes its immediately-following assistant reply, if any, since a lone reply
// with no prompt above it would be a dangling, confusing artifact; deleting an assistant message
// removes just that message. Persisted the same way an edit persists (autoSave after mutation).
//
// Phase 5 persistence model (see SCHEMA.md and app/README.md's "Auto-save default path policy"
// for the full writeup):
// - `filePaths[conversationId]` records the last path a conversation was written to, whether that
//   came from an explicit Save/Save As or from auto-save picking a default location. Once a path
//   exists, both "Save" and auto-save reuse it silently -- only "Save As" (or a brand-new
//   conversation's first auto-save) needs a dialog or default-path decision.
// - Auto-save runs from `completeGeneration` only (a *successful* completed generation), never
//   from `failGeneration` -- an errored placeholder isn't real generated content worth persisting
//   as the conversation's saved state.
// - `session.json` (app-data dir) remembers `lastActiveConversationId` + `conversationPaths` so
//   relaunching the app can silently reload the last-active conversation (and every other known
//   conversation into the sidebar) without the user clicking Open.

import { defineStore } from "pinia";
import { invoke } from "@tauri-apps/api/core";
import type { Conversation, ChatMessage } from "../types";
import { newId } from "../types";
import {
  toConversationFile,
  fromConversationFile,
  suggestFileName,
  type ConversationFile,
  type SessionFile,
} from "../persistence";

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  /** True whenever a generate_response call is in flight (send, edit, or regenerate). Backend
   * serializes all generation through a single mutex anyway (see inference.rs), so one app-wide
   * flag -- rather than one per conversation -- is sufficient to prevent overlapping calls and
   * matches what MessageInput already disabled on during Phase 3. */
  generating: boolean;
  /** conversationId -> id of the message currently being streamed into. Populated right before a
   * generate_response call, consulted by the token/done/error event handlers, cleared once that
   * generation finishes or errors. */
  streamingMessageIds: Record<string, string>;
  backendError: string | null;
  /** conversationId -> absolute path last written for that conversation (see file header). */
  filePaths: Record<string, string>;
  /** Name/version of the model actually loaded by the backend (fetched once at startup via the
   * `get_model_name` Tauri command) -- recorded into every conversation's `model` field so a
   * saved file always says which model generated it. */
  modelName: string;
  /** "CUDA — all layers on GPU", "CPU", or a fallback note. Empty until startup fetch. */
  deviceLabel: string;
  /** Message from the most recent failed Save/Save As/Open, or null. Separate from
   * `backendError` (inference errors) since these are a different failure domain (file I/O /
   * validation), surfaced in their own banner. */
  fileError: string | null;
  /** True once `restoreSession` has run (successfully or not) so App.vue knows startup restore
   * is done and it's safe to render the normal empty-state UI instead of a loading placeholder. */
  sessionRestored: boolean;
  /** Phase 7: true if the currently-active conversation's model-mismatch warning has been
   * dismissed by the user. Reset to false every time a different conversation becomes active
   * (see `setActive`) so switching conversations doesn't carry a stale dismissal over. */
  modelMismatchDismissed: boolean;

  // --- Phase 6: export/import/settings ---
  /** Non-null while an export or import is running; drives the progress modal in App.vue. */
  memoryOp: MemoryOpState | null;
  /** Message from the most recently finished export/import, shown briefly in the modal before it
   * closes, or surfaced as an error. Separate from `fileError` -- a different failure domain. */
  memoryMessage: string | null;
  memoryError: string | null;
  /** The persisted "data directory location" setting (`null` = using the local app-data default). */
  dataDir: string | null;

  // --- Phase 7: voice ---
  /** True once the voice-pipeline sidecar (VAD/STT/TTS) has finished loading its models and is
   * ready to accept `voice_listen`/`voice_speak_*` calls. Polled from App.vue via
   * `refreshVoiceReady` since there's no push event for this. */
  voiceReady: boolean;
  /** True while the current generation was triggered by a voice transcript (as opposed to typed
   * text) -- App.vue's token-stream/generation-done/generation-error handlers check this to
   * decide whether to also forward text to the voice sidecar's TTS via `voice_speak_feed`/
   * `voice_speak_finish`. Typed replies are never spoken aloud. */
  currentTurnIsVoice: boolean;

  // --- Phase 7 follow-up: Dementia/Sloth agent toggle + Sloth's persistent memory ---
  /** Which agent the *next* generation call uses. A mid-conversation toggle (see
   * MessageInput.vue), not a per-conversation setting -- a single conversation can mix turns
   * generated by each agent. Defaults to "dementia": no persistent memory outside this chat,
   * i.e. exactly the app's pre-existing behavior; Sloth is opt-in. */
  activeAgent: "dementia" | "sloth";
  /** Sloth's persistent cross-conversation fact store, mirrored from the backend
   * (`sloth_memory.rs`) for the "Sloth Memory..." management panel (see App.vue). Loaded on
   * demand when that panel opens and refreshed live via the `sloth-facts-updated` event fired
   * after every Sloth turn's fact-extraction step. */
  slothFacts: SlothFact[];
}

export interface SlothFact {
  id: string;
  text: string;
  createdAt: number;
}

export interface MemoryOpState {
  kind: "export" | "import";
  file: string;
  fileIndex: number;
  totalFiles: number;
  bytesDone: number;
  bytesTotal: number;
}

export const useChatStore = defineStore("chat", {
  state: (): ChatState => ({
    conversations: [],
    activeId: null,
    generating: false,
    streamingMessageIds: {},
    backendError: null,
    filePaths: {},
    modelName: "unknown-model",
    deviceLabel: "",
    fileError: null,
    sessionRestored: false,
    modelMismatchDismissed: false,
    memoryOp: null,
    memoryMessage: null,
    memoryError: null,
    dataDir: null,
    voiceReady: false,
    currentTurnIsVoice: false,
    activeAgent: "dementia",
    slothFacts: [],
  }),

  getters: {
    activeConversation(state): Conversation | null {
      return state.conversations.find((c) => c.id === state.activeId) ?? null;
    },
    /** Phase 7: non-null (a human-readable warning) whenever the active conversation records a
     * `model` different from the backend's currently-loaded model, and the user hasn't dismissed
     * it for this conversation yet. `null` (nothing shown) if there's no active conversation, no
     * recorded model (a conversation from before this field existed), or the models match. Does
     * not block generation -- RWKV can usually still continue a conversation started under a
     * different model, it's just not guaranteed to make sense (per phase7_prompt.md task 3). */
    modelMismatchWarning(state): string | null {
      const convo = state.conversations.find((c) => c.id === state.activeId) ?? null;
      if (!convo || !convo.model || state.modelMismatchDismissed) return null;
      if (convo.model === state.modelName) return null;
      return `This conversation was generated with model "${convo.model}", but "${state.modelName}" is currently loaded. Replies may not be consistent with the earlier ones.`;
    },
  },

  actions: {
    findConversation(conversationId: string): Conversation | null {
      return this.conversations.find((c) => c.id === conversationId) ?? null;
    },

    /** Sets the active conversation and resets the model-mismatch dismissal, so switching to a
     * (possibly different) conversation always re-evaluates the warning fresh. Every place that
     * changes `activeId` should go through this rather than assigning it directly. */
    setActive(id: string | null): void {
      this.activeId = id;
      this.modelMismatchDismissed = false;
    },

    /** Starts a fresh, empty conversation and makes it active. Used by both "+ New Chat"
     * (Sidebar) and File > New (MenuBar). */
    newChat(): void {
      const convo: Conversation = {
        id: newId(),
        title: "New Chat",
        createdAt: Date.now(),
        messages: [],
        model: this.modelName,
      };
      this.conversations.unshift(convo);
      this.setActive(convo.id);
      this.backendError = null;
      this.fileError = null;
      void this.persistSession();
    },

    selectConversation(id: string): void {
      this.setActive(id);
      this.backendError = null;
      this.fileError = null;
      void this.persistSession();
    },

    /** Fetches the backend's actually-loaded model name once at startup. Called from App.vue
     * before `restoreSession` so new conversations created afterwards (or ones loaded without a
     * `model` field) have a real value to fall back on. */
    async initModelName(): Promise<void> {
      try {
        this.modelName = await invoke<string>("get_model_name");
      } catch (err) {
        console.warn("[chat store] could not fetch model name:", err);
      }
      try {
        this.deviceLabel = await invoke<string>("get_inference_device");
      } catch (err) {
        console.warn("[chat store] could not fetch inference device:", err);
      }
    },

    /** Writes session.json (lastActiveConversationId + filePaths) so a relaunch can restore
     * state. Fire-and-forget from the caller's perspective -- a failure here (e.g. disk full)
     * only affects "restore on next launch", not the user's current session, so it's logged, not
     * surfaced as a blocking error. */
    async persistSession(): Promise<void> {
      const session: SessionFile = {
        lastActiveConversationId: this.activeId,
        conversationPaths: { ...this.filePaths },
      };
      try {
        await invoke("save_session_file", { session });
      } catch (err) {
        console.warn("[chat store] could not persist session.json:", err);
      }
    },

    /** Runs once at app startup (App.vue's onMounted). Loads session.json, reads every
     * conversation it points to back off disk, populates the sidebar, and restores whichever
     * conversation was last active -- all without the user clicking Open. A conversation file
     * that fails to load (deleted/corrupted since last run) is skipped with a console warning
     * rather than blocking the rest of startup. */
    async restoreSession(): Promise<void> {
      try {
        const session = await invoke<SessionFile>("load_session_file");
        const loaded: Conversation[] = [];
        const paths: Record<string, string> = {};

        for (const [conversationId, path] of Object.entries(session.conversationPaths ?? {})) {
          try {
            const file = await invoke<ConversationFile>("open_conversation_file", { path });
            loaded.push(fromConversationFile(file));
            paths[conversationId] = path;
          } catch (err) {
            console.warn(`[chat store] skipping conversation at '${path}' (failed to load):`, err);
          }
        }

        loaded.sort((a, b) => b.createdAt - a.createdAt);
        this.conversations = loaded;
        this.filePaths = paths;

        if (session.lastActiveConversationId && paths[session.lastActiveConversationId]) {
          this.setActive(session.lastActiveConversationId);
        } else {
          this.setActive(null);
        }
      } catch (err) {
        // First-ever launch (no session.json yet) or a genuine read failure -- either way, start
        // with an empty sidebar rather than blocking the app from opening.
        console.warn("[chat store] no session to restore (or failed to load):", err);
      } finally {
        this.sessionRestored = true;
      }
    },

    /** Shared by both auto-save and explicit Save: writes `convo` to `path` and records that path
     * as the conversation's current file. Throws on failure so callers can decide how to surface
     * it (auto-save logs; explicit Save/Save As sets `fileError`). */
    async writeConversationTo(convo: Conversation, path: string): Promise<void> {
      const file = toConversationFile(convo, this.modelName);
      await invoke("save_conversation_file", { path, conversation: file });
      this.filePaths[convo.id] = path;
      await this.persistSession();
    },

    /** Auto-save: called after every successfully completed generation (send, edit-regeneration,
     * or manual regenerate -- see `completeGeneration` below). If this conversation has never
     * been saved anywhere yet, writes to the default app-data path
     * (`<app-data-dir>/conversations/<id>.json`) instead of prompting -- see SCHEMA.md /
     * app/README.md's "Auto-save default path policy" for why. Never surfaces a dialog and never
     * blocks the UI on failure; a write error is logged and left for the user to notice via a
     * manual Save attempt, since interrupting a just-finished reply with an error banner over a
     * background auto-save would be worse than a quiet retry-on-next-turn. */
    async autoSave(conversationId: string): Promise<void> {
      const convo = this.findConversation(conversationId);
      if (!convo) return;

      try {
        let path = this.filePaths[conversationId];
        if (!path) {
          path = await invoke<string>("default_conversation_path", { conversationId });
        }
        await this.writeConversationTo(convo, path);
      } catch (err) {
        console.warn(`[chat store] auto-save failed for conversation ${conversationId}:`, err);
      }
    },

    /** File > Save. Reuses the conversation's known path if it has one (from a prior Save As or
     * auto-save); otherwise behaves like Save As (no path to silently reuse yet). */
    async saveConversation(conversationId: string): Promise<void> {
      const convo = this.findConversation(conversationId);
      if (!convo) return;
      this.fileError = null;

      const existingPath = this.filePaths[conversationId];
      if (existingPath) {
        try {
          await this.writeConversationTo(convo, existingPath);
        } catch (err) {
          this.fileError = `Could not save: ${String(err)}`;
        }
        return;
      }
      await this.saveConversationAs(conversationId);
    },

    /** File > Save As. Always opens a native save dialog, even if a path already exists, and
     * subsequent auto-saves for this conversation follow the newly chosen path afterwards. */
    async saveConversationAs(conversationId: string): Promise<void> {
      const convo = this.findConversation(conversationId);
      if (!convo) return;
      this.fileError = null;

      try {
        const path = await invoke<string | null>("pick_save_dialog", {
          defaultFileName: suggestFileName(convo),
        });
        if (!path) return; // user cancelled
        await this.writeConversationTo(convo, path);
      } catch (err) {
        this.fileError = `Could not save: ${String(err)}`;
      }
    },

    /** File > Open. Opens a native file dialog, validates the chosen file against the schema
     * (rejecting malformed/non-Xenon2 JSON with a clear message rather than crashing), and loads
     * it into the store -- replacing any existing in-memory copy of the same conversation id,
     * otherwise adding it to the sidebar. */
    async openConversation(): Promise<void> {
      this.fileError = null;
      try {
        const path = await invoke<string | null>("pick_open_dialog");
        if (!path) return; // user cancelled

        const file = await invoke<ConversationFile>("open_conversation_file", { path });
        const convo = fromConversationFile(file);

        const existingIdx = this.conversations.findIndex((c) => c.id === convo.id);
        if (existingIdx !== -1) {
          this.conversations[existingIdx] = convo;
        } else {
          this.conversations.unshift(convo);
        }
        this.filePaths[convo.id] = path;
        this.setActive(convo.id);
        await this.persistSession();
      } catch (err) {
        // Covers both a cancelled/failed dialog and a malformed file -- either way, the user sees
        // a clear message, the store's existing state is untouched (nothing was replaced before
        // this catch), and the app keeps running normally.
        this.fileError = `Could not open file: ${String(err)}`;
      }
    },

    /** Kicks off a generate_response call against `history` (everything up through the newly
     * pushed/edited user turn), streaming its reply into `assistantMsg`. Shared by sendMessage
     * and editUserMessage -- the only difference between them is how the history/target message
     * were set up beforehand. */
    async runGeneration(
      conversationId: string,
      assistantMsg: ChatMessage,
      history: { role: string; content: string }[]
    ): Promise<void> {
      this.streamingMessageIds[conversationId] = assistantMsg.id;
      this.generating = true;
      this.backendError = null;

      try {
        await invoke("generate_response", { conversationId, history, agent: assistantMsg.agent ?? "dementia" });
      } catch (err) {
        this.generating = false;
        delete this.streamingMessageIds[conversationId];
        assistantMsg.streaming = false;
        assistantMsg.errored = true;
        assistantMsg.content = `(failed to reach inference engine: ${String(err)})`;
        this.backendError = String(err);
      }
    },

    /** Polls the voice sidecar's readiness once. Called on a short interval from App.vue's
     * onMounted until it reports ready (model loading in the sidecar takes a few seconds, see
     * voice-pipeline/README.md's load times) -- there's no push event for "sidecar finished
     * loading", so polling is the simplest option for what's a one-time, short-lived check. */
    async refreshVoiceReady(): Promise<void> {
      try {
        this.voiceReady = await invoke<boolean>("voice_ready");
      } catch (err) {
        console.warn("[chat store] voice_ready check failed:", err);
      }
    },

    /** Normal send flow: append a user message + a new empty streaming assistant message, then
     * generate a reply from the full history. `viaVoice` marks this turn's reply for spoken
     * playback (see `currentTurnIsVoice`) -- set when the text came from the mic button's
     * transcript rather than typed input. */
    async sendMessage(text: string, viaVoice = false): Promise<void> {
      if (this.generating) return;
      this.currentTurnIsVoice = viaVoice;
      if (!this.activeConversation) {
        this.newChat();
      }
      const convo = this.activeConversation!;
      if (convo.title === "New Chat" || !convo.title) {
        convo.title = text.length > 40 ? text.slice(0, 40) + "..." : text;
      }

      const userMsg: ChatMessage = {
        id: newId(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      const assistantMsg: ChatMessage = {
        id: newId(),
        role: "assistant",
        content: "",
        streaming: true,
        timestamp: Date.now(),
        agent: this.activeAgent,
      };
      convo.messages.push(userMsg);
      convo.messages.push(assistantMsg);

      const history = convo.messages
        .filter((m) => m.id !== assistantMsg.id)
        .map((m) => ({ role: m.role, content: m.content }));

      await this.runGeneration(convo.id, assistantMsg, history);
    },

    /** Edit an earlier user message: truncate everything after it, update its content, and
     * regenerate a fresh assistant reply from the edited text. */
    async editUserMessage(conversationId: string, messageId: string, newText: string): Promise<void> {
      if (this.generating) return;
      const convo = this.findConversation(conversationId);
      if (!convo) return;

      const idx = convo.messages.findIndex((m) => m.id === messageId);
      if (idx === -1 || convo.messages[idx].role !== "user") return;

      // Drop everything after the edited message (its old reply + anything past it).
      convo.messages.splice(idx + 1);

      const editedMsg = convo.messages[idx];
      editedMsg.content = newText;
      editedMsg.edited = true;

      const assistantMsg: ChatMessage = {
        id: newId(),
        role: "assistant",
        content: "",
        streaming: true,
        timestamp: Date.now(),
        agent: this.activeAgent,
      };
      convo.messages.push(assistantMsg);

      const history = convo.messages
        .filter((m) => m.id !== assistantMsg.id)
        .map((m) => ({ role: m.role, content: m.content }));

      await this.runGeneration(convo.id, assistantMsg, history);
    },

    /** Regenerate one assistant message in place: rebuild history up to (not including) it, clear
     * just that message's content, and stream a new reply into it -- every other message in the
     * conversation, before or after, is left untouched. */
    async regenerateAssistantMessage(conversationId: string, messageId: string): Promise<void> {
      if (this.generating) return;
      const convo = this.findConversation(conversationId);
      if (!convo) return;

      const idx = convo.messages.findIndex((m) => m.id === messageId);
      if (idx === -1 || convo.messages[idx].role !== "assistant") return;

      const history = convo.messages
        .slice(0, idx)
        .map((m) => ({ role: m.role, content: m.content }));

      const targetMsg = convo.messages[idx];
      targetMsg.content = "";
      targetMsg.streaming = true;
      targetMsg.errored = false;
      targetMsg.timestamp = Date.now();
      targetMsg.agent = this.activeAgent;

      await this.runGeneration(convo.id, targetMsg, history);
    },

    /** Delete a message. Deleting a user message also removes its immediately-following assistant
     * reply (if present) so the conversation never keeps an orphaned reply with no prompt above
     * it; deleting an assistant message removes just that message. Refuses to delete the message
     * currently being streamed into (mirrors the `generating` guard editUserMessage/regenerate use)
     * since mutating the array mid-stream would race the token/done event handlers' by-id lookup.
     * Persists immediately afterward via the same autoSave path an edit's completed generation
     * uses. */
    async deleteMessage(conversationId: string, messageId: string): Promise<void> {
      const convo = this.findConversation(conversationId);
      if (!convo) return;
      if (this.streamingMessageIds[conversationId] === messageId) return;

      const idx = convo.messages.findIndex((m) => m.id === messageId);
      if (idx === -1) return;

      const target = convo.messages[idx];
      if (target.role === "user") {
        const next = convo.messages[idx + 1];
        const removeCount = next && next.role === "assistant" ? 2 : 1;
        convo.messages.splice(idx, removeCount);
      } else {
        convo.messages.splice(idx, 1);
      }

      await this.autoSave(conversationId);
    },

    // --- Tauri event handlers (called from App.vue's listen() callbacks) ---

    appendToken(conversationId: string, text: string): void {
      const messageId = this.streamingMessageIds[conversationId];
      if (!messageId) return;
      const msg = this.findConversation(conversationId)?.messages.find((m) => m.id === messageId);
      if (msg) msg.content += text;
    },

    completeGeneration(conversationId: string): void {
      const messageId = this.streamingMessageIds[conversationId];
      const msg = this.findConversation(conversationId)?.messages.find((m) => m.id === messageId);
      if (msg) {
        msg.streaming = false;
        // The Rust backend's stop-sequence check stops generation as soon as the model starts a
        // new "User:" turn, but that stop-sequence text has already been streamed and appended --
        // strip it here so the bubble shows only Xenon's actual reply.
        msg.content = msg.content.replace(/\n+\s*[Uu]ser:\s*$/, "").trimEnd();
        // Phase 7 follow-up (2026-08-02): a small base model leans heavily on whatever style its
        // prompt demonstrates -- the seed example used to end with "How can I help you today?"
        // and the model picked that up as a closing habit for nearly every reply regardless of
        // content (see inference.rs's build_prompt for the actual prompt-level fix). This is a
        // display/save-time backstop for whatever still slips through despite that fix -- strips
        // a trailing "is there anything else..."/"how can I help..." sentence if the reply ends
        // with one, without touching legitimate content earlier in the same message.
        msg.content = msg.content
          .replace(/\s*is there anything else[^.?!]*[.?!]?\s*$/i, "")
          .replace(/\s*how can i help you(?: today)?\??\s*$/i, "")
          .trimEnd();
      }
      delete this.streamingMessageIds[conversationId];
      this.generating = false;

      // Phase 5 auto-save: every completed exchange (send, edit-regeneration, or manual
      // regenerate all funnel through here) gets written to disk so a crash can't lose history.
      // Deliberately NOT awaited -- completeGeneration is called synchronously from App.vue's
      // event listener and shouldn't block the UI on a disk write; autoSave handles its own
      // errors internally (see its doc comment).
      void this.autoSave(conversationId);
    },

    failGeneration(conversationId: string, message: string): void {
      const messageId = this.streamingMessageIds[conversationId];
      const msg = this.findConversation(conversationId)?.messages.find((m) => m.id === messageId);
      if (msg) {
        msg.streaming = false;
        msg.errored = true;
        if (!msg.content) msg.content = `(inference error: ${message})`;
      }
      delete this.streamingMessageIds[conversationId];
      this.backendError = message;
      this.generating = false;
    },

    // --- Phase 6: export / import / settings ---

    /** Loads the persisted "data directory location" setting into `dataDir`. Called once at
     * startup (App.vue's onMounted) alongside initModelName/restoreSession. */
    async loadDataDirSetting(): Promise<void> {
      try {
        const settings = await invoke<{ dataDir: string | null }>("load_settings");
        this.dataDir = settings.dataDir ?? null;
      } catch (err) {
        console.warn("[chat store] could not load settings.json:", err);
      }
    },

    /** Persists a new "data directory location". Passing `null` reverts to the local app-data
     * default. Takes effect immediately for the next auto-save/default-path lookup -- the Rust
     * side (`persistence::default_conversation_path`) reads settings.json fresh every call, it
     * does not cache the value from startup. */
    async setDataDir(dir: string | null): Promise<void> {
      try {
        await invoke("save_settings", { settings: { dataDir: dir } });
        this.dataDir = dir;
      } catch (err) {
        this.fileError = `Could not save data directory setting: ${String(err)}`;
      }
    },

    /** File > Export Memory. Opens a folder picker for the destination, then runs the backend
     * export command, which emits `export-progress` events (listened to in App.vue) while it
     * copies the quantized model, vocab, voice model, and all conversations into
     * `<destination>/xenon2-backup/`. See EXPORT_FORMAT.md for the bundle layout. */
    async exportMemory(): Promise<void> {
      this.memoryError = null;
      this.memoryMessage = null;
      try {
        const dest = await invoke<string | null>("pick_folder_dialog", {
          testEnvVar: "XENON2_TEST_EXPORT_DEST_PATH",
        });
        if (!dest) return; // user cancelled
        this.memoryOp = { kind: "export", file: "", fileIndex: 0, totalFiles: 0, bytesDone: 0, bytesTotal: 0 };
        await invoke("export_memory", { destination: dest });
      } catch (err) {
        this.memoryError = `Export failed: ${String(err)}`;
        this.memoryOp = null;
      }
    },

    /** File > Import Memory. Opens a folder picker for the source bundle (or a folder containing
     * an `xenon2-backup/` bundle), then runs the backend import command, which copies files
     * in (does not run directly against the external path -- see EXPORT_FORMAT.md) and rewrites
     * `session.json` to point at this machine's conversations directory. Reloads the session
     * afterwards so the imported conversations show up in the sidebar immediately. */
    async importMemory(): Promise<void> {
      this.memoryError = null;
      this.memoryMessage = null;
      try {
        const source = await invoke<string | null>("pick_folder_dialog", {
          testEnvVar: "XENON2_TEST_IMPORT_SOURCE_PATH",
        });
        if (!source) return; // user cancelled
        this.memoryOp = { kind: "import", file: "", fileIndex: 0, totalFiles: 0, bytesDone: 0, bytesTotal: 0 };
        await invoke("import_memory", { source });
        await this.restoreSession();
      } catch (err) {
        this.memoryError = `Import failed: ${String(err)}`;
        this.memoryOp = null;
      }
    },

    /** Handler for the `export-progress`/`import-progress` Tauri events (wired in App.vue). */
    onMemoryProgress(
      kind: "export" | "import",
      payload: { file: string; fileIndex: number; totalFiles: number; bytesDone: number; bytesTotal: number }
    ): void {
      this.memoryOp = { kind, ...payload };
    },

    /** Handler for the `export-done`/`import-done` Tauri events. */
    onMemoryDone(kind: "export" | "import", payload: { filesCopied: number; totalBytes: number }): void {
      this.memoryOp = null;
      const mb = (payload.totalBytes / (1024 * 1024)).toFixed(1);
      this.memoryMessage = `${kind === "export" ? "Export" : "Import"} complete: ${payload.filesCopied} file(s), ${mb} MB.`;
    },

    /** Handler for the `export-error`/`import-error` Tauri events. */
    onMemoryError(kind: "export" | "import", message: string): void {
      this.memoryOp = null;
      this.memoryError = `${kind === "export" ? "Export" : "Import"} failed: ${message}`;
    },

    // --- Phase 7 follow-up: Dementia/Sloth agent toggle + Sloth's persistent memory ---

    /** Flips which agent the *next* generation call uses (see `activeAgent`'s doc comment). */
    setActiveAgent(agent: "dementia" | "sloth"): void {
      this.activeAgent = agent;
    },

    /** Loads Sloth's persistent fact store for the "Sloth Memory..." management panel. */
    async loadSlothFacts(): Promise<void> {
      try {
        this.slothFacts = await invoke<SlothFact[]>("list_sloth_facts");
      } catch (err) {
        console.warn("[chat store] could not load Sloth facts:", err);
      }
    },

    /** Deletes one fact -- lets a bad automatic extraction (the underlying model is small and
     * sometimes gets this wrong) be removed without losing everything else Sloth remembers. */
    async deleteSlothFact(id: string): Promise<void> {
      try {
        this.slothFacts = await invoke<SlothFact[]>("delete_sloth_fact", { id });
      } catch (err) {
        console.warn("[chat store] could not delete Sloth fact:", err);
      }
    },

    /** Wipes Sloth's entire persistent memory. */
    async clearSlothFacts(): Promise<void> {
      try {
        await invoke("clear_sloth_facts");
        this.slothFacts = [];
      } catch (err) {
        console.warn("[chat store] could not clear Sloth facts:", err);
      }
    },

    /** Handler for the `sloth-facts-updated` Tauri event, fired after every Sloth turn's
     * fact-extraction step actually adds something new -- keeps an open Memory panel live. */
    onSlothFactsUpdated(facts: SlothFact[]): void {
      this.slothFacts = facts;
    },
  },
});
