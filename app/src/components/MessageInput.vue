<script setup lang="ts">
// MessageInput.vue -- text box + send button + mic button, all fully functional (Phase 7).
//
// Clicking the mic starts real VAD-gated mic capture via the voice-pipeline sidecar
// (`voice_listen`, see app/src-tauri/src/voice.rs). The button stays usable while a reply
// is generating. A transcript is emitted as `voice-send`. App.vue sends it through the same
// store.sendMessage path as typed text, or holds it until the engine and playback are free.

import { ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { useChatStore } from "../stores/chat";

// Phase 7 follow-up: Dementia/Sloth agent toggle. Reads/writes the store directly (like the mic
// button's own invoke() calls above) rather than round-tripping through App.vue -- this
// component already isn't purely presentational.
const store = useChatStore();

const props = defineProps<{
  disabled?: boolean;
  /** Whether the voice sidecar has finished loading its models (VAD/STT/TTS) and can actually
   * service a `voice_listen` call yet -- see stores/chat.ts's voiceReady/refreshVoiceReady. The
   * mic button stays clickable either way (a call made before this is true just queues behind
   * the sidecar's own startup instead of failing), but is visually dimmed with an explanatory
   * title so it doesn't look broken during the few seconds of startup. */
  voiceReady?: boolean;
  /** True while a reply is generating or its speech is still playing. A transcript captured
   * then is kept and sent later instead of being dropped. */
  voiceBusy?: boolean;
}>();

const emit = defineEmits<{
  (e: "send", text: string): void;
  (e: "voice-send", text: string): void;
}>();

const text = ref("");
const listening = ref(false);
const micStatus = ref("");

function submit() {
  const trimmed = text.value.trim();
  if (!trimmed || props.disabled) return;
  emit("send", trimmed);
  text.value = "";
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
}

function listenFailure(reason?: string): string {
  if (reason === "no_speech_timeout") return "No speech detected";
  if (reason === "too_quiet") return "Mic was too quiet";
  if (reason === "busy") return "Still listening";
  if (reason === "error") return "Voice input unavailable";
  return "Didn't catch that";
}

async function toggleMic() {
  if (listening.value) return;
  listening.value = true;
  micStatus.value = "Listening...";
  try {
    const result = await invoke<{ ok: boolean; text?: string; reason?: string }>("voice_listen", {
      maxListenSec: 6,
      maxUtteranceSec: 10,
    });
    if (result.ok && result.text) {
      emit("voice-send", result.text);
      micStatus.value = props.voiceBusy ? "Queued until Xenon is free" : "";
      if (props.voiceBusy) {
        window.setTimeout(() => {
          if (micStatus.value === "Queued until Xenon is free") micStatus.value = "";
        }, 2500);
      }
    } else {
      micStatus.value = listenFailure(result.reason);
      window.setTimeout(() => {
        if (!listening.value) micStatus.value = "";
      }, 2000);
    }
  } catch (err) {
    console.warn("[MessageInput] voice_listen failed:", err);
    micStatus.value = "Voice input unavailable";
    window.setTimeout(() => {
      if (!listening.value) micStatus.value = "";
    }, 3000);
  } finally {
    listening.value = false;
  }
}
</script>

<template>
  <div class="input-bar">
    <div class="agent-toggle" title="Which agent handles the next message">
      <button
        class="agent-btn"
        :class="{ active: store.activeAgent === 'dementia' }"
        :disabled="disabled"
        title="Dementia: no memory outside this chat window"
        @click="store.setActiveAgent('dementia')"
      >
        Dementia
      </button>
      <button
        class="agent-btn"
        :class="{ active: store.activeAgent === 'sloth' }"
        :disabled="disabled"
        title="Sloth: remembers facts across every conversation"
        @click="store.setActiveAgent('sloth')"
      >
        Sloth
      </button>
    </div>

    <div class="mic-wrap">
      <button
        class="mic-btn"
        :class="{ on: listening }"
        :title="voiceReady === false ? 'Voice input is still starting up...' : 'Voice input'"
        @click="toggleMic"
      >
        <!-- Inline SVG, not an emoji glyph -- the 🎤 emoji was found to render as an unrecognizable
             garbled fallback glyph on this environment's WebView2 (missing color-emoji font
             support), making the mic button impossible to identify visually despite being fully
             wired up. SVG renders identically regardless of what emoji fonts are installed. -->
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      </button>
      <div v-if="micStatus" class="mic-status">{{ micStatus }}</div>
    </div>

    <textarea
      v-model="text"
      class="text-box"
      rows="1"
      placeholder="Type a message..."
      :disabled="disabled"
      @keydown="onKeydown"
    ></textarea>

    <button
      class="send-btn"
      :disabled="disabled || !text.trim()"
      title="Send"
      @click="submit"
    >
      ➤
    </button>
  </div>
</template>

<style scoped>
.input-bar {
  display: flex;
  align-items: flex-end;
  flex-wrap: wrap;
  flex-shrink: 0;
  gap: 0.5rem;
  padding: 0.6rem 0.8rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  background: var(--input-bg, #191919);
  max-width: 100%;
}

.text-box {
  flex: 1 1 140px;
  min-width: 0;
  resize: none;
  max-height: 8rem;
  padding: 0.55rem 0.7rem;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: #232324;
  color: #eee;
  font-size: 0.92rem;
  font-family: inherit;
}

.text-box:focus {
  outline: none;
  border-color: #2f6feb;
}

.agent-toggle {
  display: flex;
  flex-shrink: 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.15);
}

.agent-btn {
  background: #232324;
  color: #999;
  border: none;
  padding: 0 0.6rem;
  height: 2.3rem;
  font-size: 0.75rem;
  cursor: pointer;
}

.agent-btn.active {
  background: #3a3a6b;
  color: #fff;
}

.agent-btn:disabled {
  cursor: default;
  opacity: 0.6;
}

.mic-wrap {
  position: relative;
  flex-shrink: 0;
}

.mic-btn,
.send-btn {
  flex-shrink: 0;
  width: 2.3rem;
  height: 2.3rem;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.mic-btn {
  background: #2a2a2c;
  color: #ccc;
}

.mic-btn.on {
  background: #b8402f;
  color: #fff;
  animation: mic-pulse 1s ease-in-out infinite;
}

@keyframes mic-pulse {
  50% {
    box-shadow: 0 0 0 6px rgba(184, 64, 47, 0.25);
  }
}

.mic-status {
  position: absolute;
  bottom: 2.7rem;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  background: #232325;
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #ccc;
  font-size: 0.72rem;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.send-btn {
  background: #2f6feb;
  color: #fff;
}

.send-btn:disabled {
  background: #2a2a2c;
  color: #666;
  cursor: default;
}
</style>
