<script setup lang="ts">
// App.vue -- top-level layout: MenuBar on top, Sidebar + chat panel below (Phase 3 shell).
//
// As of Phase 4, conversation/message state lives in the Pinia store (`stores/chat.ts`), not in
// local refs -- this component reads/writes through `useChatStore()`. It still owns Tauri
// event-listener registration and DOM-only concerns (scroll-to-bottom) that don't belong in a
// store. Token/done/error events from the `generate_response` Tauri command are forwarded
// straight into the matching store actions, which know how to find the right message (by id, not
// by array position) to update.

import { onMounted, onUnmounted, ref, watch } from "vue";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

import MenuBar from "./components/MenuBar.vue";
import Sidebar from "./components/Sidebar.vue";
import ChatMessageVue from "./components/ChatMessage.vue";
import MessageInput from "./components/MessageInput.vue";
import { useChatStore } from "./stores/chat";

const store = useChatStore();

const messagesContainer = ref<HTMLElement | null>(null);
const sidebarOpen = ref(true);
const narrow = ref(false);
let wasNarrow = false;

function applyViewport() {
  const isNarrow = window.innerWidth < 860;
  if (isNarrow && !wasNarrow) sidebarOpen.value = false;
  wasNarrow = isNarrow;
  narrow.value = isNarrow;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    const el = messagesContainer.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

function newChat() {
  store.newChat();
  scrollToBottom();
}

function selectConversation(id: string) {
  store.selectConversation(id);
  scrollToBottom();
}

// Phase 5: File > Open / Save / Save As. MenuBar just emits; the store owns the actual Tauri
// dialog + filesystem calls (see stores/chat.ts). Errors surface via store.fileError, rendered in
// the banner below rather than a native alert() so it matches the existing backendError banner.
async function openConversation() {
  await store.openConversation();
  scrollToBottom();
}

async function saveConversation() {
  const id = store.activeConversation?.id;
  if (!id) return;
  await store.saveConversation(id);
}

async function saveConversationAs() {
  const id = store.activeConversation?.id;
  if (!id) return;
  await store.saveConversationAs(id);
}

// Phase 6: Export / Import Memory + data directory settings. MenuBar just emits; the store owns
// the Tauri dialog + backend calls. Progress is rendered from store.memoryOp, populated by the
// export-progress/import-progress event listeners registered below.
async function exportMemory() {
  await store.exportMemory();
}

async function importMemory() {
  await store.importMemory();
}

const settingsOpen = ref(false);
const settingsInput = ref("");

function openSettings() {
  settingsInput.value = store.dataDir ?? "";
  settingsOpen.value = true;
}

async function pickDataDir() {
  try {
    const dir = await import("@tauri-apps/api/core").then((m) =>
      m.invoke<string | null>("pick_folder_dialog", { testEnvVar: "XENON2_TEST_DATADIR_PICK_PATH" })
    );
    if (dir) settingsInput.value = dir;
  } catch (err) {
    console.warn("[App] folder pick failed:", err);
  }
}

async function saveDataDirSetting() {
  await store.setDataDir(settingsInput.value.trim() || null);
  settingsOpen.value = false;
}

async function clearDataDirSetting() {
  await store.setDataDir(null);
  settingsInput.value = "";
  settingsOpen.value = false;
}

// Phase 7 follow-up: Sloth Memory management panel.
const slothMemoryOpen = ref(false);

async function openSlothMemory() {
  await store.loadSlothFacts();
  slothMemoryOpen.value = true;
}

async function deleteSlothFact(id: string) {
  await store.deleteSlothFact(id);
}

async function clearSlothFacts() {
  if (!window.confirm("Delete everything Sloth remembers? This can't be undone.")) return;
  await store.clearSlothFacts();
}

async function sendMessage(text: string) {
  scrollToBottom();
  await store.sendMessage(text);
  scrollToBottom();
}

// Voice transcripts share sendMessage with typed text. The mic stays open while Xenon
// generates and speaks, so everything heard in that time builds up here and goes out as one
// prompt once Xenon has finished and the user has stopped talking.
const pendingVoice = ref<string[]>([]);
let pumpingVoice = false;
const speechOutActive = ref(false);
let speakStartOk = false;
let awaitingSpeakDone = false;

function finishVoicePlayback() {
  if (!store.currentTurnIsVoice) return;
  store.currentTurnIsVoice = false;
  requestSpeakFinish();
}

function requestSpeakFinish() {
  if (!speakStartOk) {
    speechOutActive.value = false;
    return;
  }
  awaitingSpeakDone = true;
  invoke("voice_speak_finish").catch((err) => {
    console.warn("[App] voice_speak_finish failed:", err);
    awaitingSpeakDone = false;
    speechOutActive.value = false;
    void pumpVoice();
  });
}

function enqueueVoice(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return;
  pendingVoice.value.push(trimmed);
  void pumpVoice();
}

function voiceCanSend(): boolean {
  return (
    pendingVoice.value.length > 0 &&
    !store.generating &&
    !speechOutActive.value &&
    !store.voiceUserSpeaking
  );
}

async function pumpVoice() {
  if (pumpingVoice) return;
  pumpingVoice = true;
  try {
    while (voiceCanSend()) {
      await startVoiceTurn(pendingVoice.value.splice(0).join(" "));
    }
  } finally {
    pumpingVoice = false;
    if (voiceCanSend()) void pumpVoice();
  }
}

watch(
  () => store.voiceUserSpeaking,
  (speaking) => {
    if (!speaking) void pumpVoice();
  }
);

async function startVoiceTurn(text: string) {
  speechOutActive.value = true;
  speakStartOk = false;
  scrollToBottom();
  try {
    await invoke("voice_speak_start");
    speakStartOk = true;
  } catch (err) {
    speakStartOk = false;
    speechOutActive.value = false;
    console.warn("[App] voice_speak_start failed (continuing without spoken reply):", err);
  }
  const started = await store.sendMessage(text, true);
  if (!started) {
    pendingVoice.value.unshift(text);
    if (speakStartOk) requestSpeakFinish();
    else speechOutActive.value = false;
    return;
  }
  // generation-done usually lands while sendMessage is still awaiting the backend,
  // and that handler flushes speech. If the backend returned without the event,
  // the flag is still set and the last sentence would never be spoken.
  if (!store.generating) finishVoicePlayback();
  scrollToBottom();
}

function onVoiceSend(text: string) {
  enqueueVoice(text);
}

async function editMessage(messageId: string, newText: string) {
  const conversationId = store.activeConversation?.id;
  if (!conversationId) return;
  scrollToBottom();
  await store.editUserMessage(conversationId, messageId, newText);
  scrollToBottom();
}

async function regenerateMessage(messageId: string) {
  const conversationId = store.activeConversation?.id;
  if (!conversationId) return;
  await store.regenerateAssistantMessage(conversationId, messageId);
  scrollToBottom();
}

async function deleteMessage(messageId: string) {
  const conversationId = store.activeConversation?.id;
  if (!conversationId) return;
  await store.deleteMessage(conversationId, messageId);
}

let unlistenToken: UnlistenFn | null = null;
let unlistenDone: UnlistenFn | null = null;
let unlistenError: UnlistenFn | null = null;
let unlistenExportProgress: UnlistenFn | null = null;
let unlistenExportDone: UnlistenFn | null = null;
let unlistenExportError: UnlistenFn | null = null;
let unlistenImportProgress: UnlistenFn | null = null;
let unlistenImportDone: UnlistenFn | null = null;
let unlistenImportError: UnlistenFn | null = null;
let unlistenSlothFactsUpdated: UnlistenFn | null = null;
let unlistenSpeakDone: UnlistenFn | null = null;

type MemoryProgressPayload = {
  file: string;
  fileIndex: number;
  totalFiles: number;
  bytesDone: number;
  bytesTotal: number;
};
type MemoryDonePayload = { filesCopied: number; totalBytes: number };
type MemoryErrorPayload = { message: string };

onMounted(async () => {
  applyViewport();
  window.addEventListener("resize", applyViewport);

  // Phase 5: fetch the backend's actually-loaded model name first (so any conversation created
  // or loaded afterwards has a real fallback value -- see store.newChat/autoSave), then restore
  // the last session (sidebar + last-active conversation) before wiring up generation events.
  await store.initModelName();
  await store.loadDataDirSetting();
  await store.restoreSession();
  scrollToBottom();

  // Phase 6: export/import progress events (see memory.rs's copy_with_progress).
  unlistenExportProgress = await listen<MemoryProgressPayload>("export-progress", (event) => {
    store.onMemoryProgress("export", event.payload);
  });
  unlistenExportDone = await listen<MemoryDonePayload>("export-done", (event) => {
    store.onMemoryDone("export", event.payload);
  });
  unlistenExportError = await listen<MemoryErrorPayload>("export-error", (event) => {
    store.onMemoryError("export", event.payload.message);
  });
  unlistenImportProgress = await listen<MemoryProgressPayload>("import-progress", (event) => {
    store.onMemoryProgress("import", event.payload);
  });
  unlistenImportDone = await listen<MemoryDonePayload>("import-done", (event) => {
    store.onMemoryDone("import", event.payload);
  });
  unlistenImportError = await listen<MemoryErrorPayload>("import-error", (event) => {
    store.onMemoryError("import", event.payload.message);
  });

  // Phase 7 follow-up: fired after a Sloth turn's fact-extraction step adds something new.
  unlistenSlothFactsUpdated = await listen<{ id: string; text: string; createdAt: number }[]>(
    "sloth-facts-updated",
    (event) => {
      store.onSlothFactsUpdated(event.payload);
    }
  );

  unlistenToken = await listen<{ conversationId: string; text: string }>("token-stream", (event) => {
    store.appendToken(event.payload.conversationId, event.payload.text);
    if (store.currentTurnIsVoice) {
      invoke("voice_speak_feed", { text: event.payload.text }).catch((err) =>
        console.warn("[App] voice_speak_feed failed:", err)
      );
    }
    scrollToBottom();
  });

  unlistenDone = await listen<{ conversationId: string }>("generation-done", (event) => {
    // Clear the flag before completeGeneration so a late token cannot start a second speak.
    const speakThisTurn = store.currentTurnIsVoice;
    store.currentTurnIsVoice = false;
    store.completeGeneration(event.payload.conversationId);
    if (speakThisTurn) requestSpeakFinish();
    void pumpVoice();
  });

  unlistenError = await listen<{ conversationId: string; message: string }>(
    "generation-error",
    (event) => {
      const speakThisTurn = store.currentTurnIsVoice;
      store.currentTurnIsVoice = false;
      store.failGeneration(event.payload.conversationId, event.payload.message);
      // Flush whatever was already fed. finish() speaks a short reply that never
      // hit sentence punctuation, and the next speak_start also drops a leftover speaker.
      if (speakThisTurn) requestSpeakFinish();
      void pumpVoice();
    }
  );

  unlistenSpeakDone = await listen("voice-speak-done", () => {
    if (!awaitingSpeakDone) return;
    awaitingSpeakDone = false;
    speechOutActive.value = false;
    void pumpVoice();
  });

  // Phase 7: poll the voice sidecar's readiness (model loading takes a few seconds -- see
  // voice-pipeline/README.md's load times) until it reports ready, then stop polling.
  await store.refreshVoiceReady();
  const voiceReadyPoll = window.setInterval(async () => {
    await store.refreshVoiceReady();
    if (store.voiceReady) window.clearInterval(voiceReadyPoll);
  }, 1000);
});

onUnmounted(() => {
  window.removeEventListener("resize", applyViewport);
  unlistenToken?.();
  unlistenDone?.();
  unlistenError?.();
  unlistenExportProgress?.();
  unlistenExportDone?.();
  unlistenExportError?.();
  unlistenImportProgress?.();
  unlistenImportDone?.();
  unlistenImportError?.();
  unlistenSlothFactsUpdated?.();
  unlistenSpeakDone?.();
});
</script>

<template>
  <div class="app-shell">
    <MenuBar
      :has-active-conversation="!!store.activeConversation"
      :device-label="store.deviceLabel"
      :sidebar-open="sidebarOpen"
      @toggle-sidebar="sidebarOpen = !sidebarOpen"
      @new="newChat"
      @open="openConversation"
      @save="saveConversation"
      @save-as="saveConversationAs"
      @export-memory="exportMemory"
      @import-memory="importMemory"
      @settings="openSettings"
      @sloth-memory="openSlothMemory"
    />

    <div class="body" :class="{ 'sidebar-closed': !sidebarOpen, narrow }">
      <div v-if="sidebarOpen && narrow" class="sidebar-backdrop" @click="sidebarOpen = false"></div>
      <Sidebar
        :conversations="store.conversations"
        :active-id="store.activeId"
        @new-chat="newChat"
        @select="selectConversation"
      />

      <main class="chat-panel">
        <div v-if="!store.activeConversation" class="empty-state">
          <p>No conversation selected.</p>
          <p class="hint">Type a message below, or click "+ New Chat" to get started.</p>
        </div>

        <template v-else>
          <div ref="messagesContainer" class="messages">
            <ChatMessageVue
              v-for="m in store.activeConversation.messages"
              :key="m.id"
              :message="m"
              :disabled="store.generating"
              @edit="(newText) => editMessage(m.id, newText)"
              @regenerate="() => regenerateMessage(m.id)"
              @delete="() => deleteMessage(m.id)"
            />
          </div>
        </template>

        <div v-if="store.backendError" class="error-banner">{{ store.backendError }}</div>
        <div v-if="store.fileError" class="error-banner file-error">
          {{ store.fileError }}
          <button class="dismiss-btn" title="Dismiss" @click="store.fileError = null">✕</button>
        </div>
        <div v-if="store.modelMismatchWarning" class="error-banner model-mismatch-banner">
          {{ store.modelMismatchWarning }}
          <button class="dismiss-btn" title="Dismiss" @click="store.modelMismatchDismissed = true">✕</button>
        </div>

        <MessageInput
          :disabled="store.generating"
          :voice-ready="store.voiceReady"
          :voice-busy="store.generating || speechOutActive"
          :pending-voice="pendingVoice.join(' ')"
          @send="sendMessage"
          @voice-send="onVoiceSend"
        />
      </main>
    </div>

    <!-- Phase 6: export/import progress -->
    <div v-if="store.memoryOp" class="modal-backdrop">
      <div class="modal memory-modal">
        <h3>{{ store.memoryOp.kind === "export" ? "Exporting Memory..." : "Importing Memory..." }}</h3>
        <p class="memory-file">
          File {{ store.memoryOp.fileIndex + 1 }} / {{ store.memoryOp.totalFiles || "?" }}:
          {{ store.memoryOp.file }}
        </p>
        <div class="progress-bar">
          <div
            class="progress-fill"
            :style="{
              width:
                store.memoryOp.bytesTotal > 0
                  ? (store.memoryOp.bytesDone / store.memoryOp.bytesTotal) * 100 + '%'
                  : '0%',
            }"
          ></div>
        </div>
        <p class="memory-bytes">
          {{ (store.memoryOp.bytesDone / (1024 * 1024)).toFixed(1) }} MB /
          {{ (store.memoryOp.bytesTotal / (1024 * 1024)).toFixed(1) }} MB
        </p>
      </div>
    </div>

    <div v-if="store.memoryMessage && !store.memoryOp" class="toast">
      {{ store.memoryMessage }}
      <button class="dismiss-btn" @click="store.memoryMessage = null">✕</button>
    </div>
    <div v-if="store.memoryError" class="toast toast-error">
      {{ store.memoryError }}
      <button class="dismiss-btn" @click="store.memoryError = null">✕</button>
    </div>

    <!-- Phase 6: data directory location settings -->
    <div v-if="settingsOpen" class="modal-backdrop" @click.self="settingsOpen = false">
      <div class="modal">
        <h3>Data Directory Location</h3>
        <p class="modal-hint">
          When set, new conversations auto-save to and load from this folder's
          <code>conversations/</code> subfolder instead of the local app-data default. Leave blank
          to use the default location.
        </p>
        <div class="modal-row">
          <input v-model="settingsInput" class="modal-input" placeholder="(local app-data default)" />
          <button class="modal-btn" @click="pickDataDir">Browse...</button>
        </div>
        <p v-if="store.dataDir" class="modal-hint">Currently: {{ store.dataDir }}</p>
        <p v-else class="modal-hint">Currently: local app-data default.</p>
        <div class="modal-actions">
          <button class="modal-btn" @click="clearDataDirSetting">Use Default</button>
          <button class="modal-btn" @click="settingsOpen = false">Cancel</button>
          <button class="modal-btn primary" @click="saveDataDirSetting">Save</button>
        </div>
      </div>
    </div>

    <!-- Phase 7 follow-up: Sloth's persistent cross-conversation memory -->
    <div v-if="slothMemoryOpen" class="modal-backdrop" @click.self="slothMemoryOpen = false">
      <div class="modal sloth-modal">
        <h3>Sloth Memory</h3>
        <p class="modal-hint">
          Facts Sloth has automatically remembered across conversations, injected into every
          Sloth-mode reply. Extraction uses the loaded model itself, so it can sometimes be wrong
          -- delete anything that doesn't belong.
        </p>
        <ul v-if="store.slothFacts.length" class="fact-list">
          <li v-for="fact in store.slothFacts" :key="fact.id" class="fact-item">
            <span class="fact-text">{{ fact.text }}</span>
            <button class="dismiss-btn" title="Forget this" @click="deleteSlothFact(fact.id)">✕</button>
          </li>
        </ul>
        <p v-else class="modal-hint">Sloth doesn't remember anything yet.</p>
        <div class="modal-actions">
          <button
            class="modal-btn"
            :disabled="!store.slothFacts.length"
            @click="clearSlothFacts"
          >
            Forget Everything
          </button>
          <button class="modal-btn primary" @click="slothMemoryOpen = false">Close</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  position: fixed;
  inset: 0;
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.body {
  display: flex;
  flex: 1;
  min-height: 0;
  min-width: 0;
  position: relative;
  overflow: hidden;
}

.body.sidebar-closed :deep(.sidebar) {
  display: none;
}

.sidebar-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  z-index: 25;
}

.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: #131314;
}

.messages {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 1.5rem 1.5rem 0.5rem;
  display: flex;
  flex-direction: column;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #888;
  text-align: center;
  gap: 0.3rem;
}

.empty-state .hint {
  font-size: 0.85rem;
  color: #666;
}

.error-banner {
  flex-shrink: 0;
  margin: 0 1rem;
  padding: 0.4rem 0.7rem;
  background: #5a2323;
  color: #ffd6d6;
  border-radius: 6px;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.error-banner.file-error {
  margin-top: 0.4rem;
}

.error-banner.model-mismatch-banner {
  margin-top: 0.4rem;
  background: #4a3a1a;
  color: #ffe4b0;
}

.dismiss-btn {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  font-size: 0.8rem;
  flex-shrink: 0;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: #232325;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  width: min(460px, calc(100vw - 1.5rem));
  max-height: calc(100vh - 1.5rem);
  max-height: calc(100dvh - 1.5rem);
  overflow: auto;
  color: #eee;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
}

.modal h3 {
  margin: 0 0 0.6rem;
  font-size: 1rem;
}

.memory-file {
  font-size: 0.8rem;
  color: #aaa;
  word-break: break-all;
  margin: 0 0 0.6rem;
}

.progress-bar {
  height: 8px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #5b8def;
  transition: width 0.1s linear;
}

.memory-bytes {
  font-size: 0.75rem;
  color: #888;
  margin: 0.5rem 0 0;
  text-align: right;
}

.modal-hint {
  font-size: 0.78rem;
  color: #999;
  margin: 0.3rem 0;
}

.modal-hint code {
  color: #ccc;
}

.modal-row {
  display: flex;
  gap: 0.5rem;
  margin: 0.6rem 0;
}

.modal-input {
  flex: 1;
  background: #1a1a1b;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  color: #eee;
  padding: 0.4rem 0.5rem;
  font-size: 0.85rem;
}

.fact-list {
  list-style: none;
  margin: 0.6rem 0;
  padding: 0;
  max-height: 260px;
  overflow-y: auto;
}

.fact-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.4rem 0.5rem;
  border-radius: 6px;
  background: #1a1a1b;
  margin-bottom: 0.35rem;
}

.fact-text {
  font-size: 0.82rem;
  line-height: 1.35;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1rem;
}

.modal-btn {
  background: #33333a;
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #eee;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  font-size: 0.82rem;
  cursor: pointer;
}

.modal-btn:hover {
  background: #3d3d44;
}

.modal-btn.primary {
  background: #5b8def;
  border-color: #5b8def;
  color: #fff;
}

.toast {
  position: fixed;
  top: 2.75rem;
  left: 50%;
  transform: translateX(-50%);
  max-width: calc(100vw - 1.5rem);
  background: #234a2e;
  color: #d6ffe0;
  padding: 0.6rem 0.9rem;
  border-radius: 6px;
  font-size: 0.82rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  z-index: 110;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.toast-error {
  background: #5a2323;
  color: #ffd6d6;
}
</style>

<style>
:root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Avenir, Helvetica, Arial,
    sans-serif;
  font-size: 16px;
  color: #eee;
  background-color: #131314;
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
}
</style>
