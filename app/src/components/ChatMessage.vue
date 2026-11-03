<script setup lang="ts">
// ChatMessage.vue -- one message bubble (user or assistant).
//
// Phase 4: edit (user messages) and regenerate (assistant messages) are now real -- clicking edit
// swaps the bubble into an editable textarea; confirming emits `edit` with the new text, and
// App.vue/the Pinia store handle the truncate + re-generate flow. Regenerate just emits
// `regenerate` and the store replaces that one message's content in place.
//
// Phase 7: delete is now real too (see stores/chat.ts's deleteMessage). Deleting a user message
// also removes its immediately-following assistant reply, if any -- a user turn and its reply are
// treated as one deletable unit rather than leaving an orphaned reply with no prompt above it.
// Confirmed via a native confirm() before emitting, since this is destructive and there's no undo.
//
// Assistant replies can be rated good or bad for learning from feedback (see ONLINE_LEARNING.md).
// A bad rating opens an optional box for what Xenon should have said. Clicking the active rating
// again clears it.

import { ref } from "vue";
import type { ChatMessage, Rating } from "../types";

const props = defineProps<{
  message: ChatMessage;
  /** True while any generation is in flight app-wide -- disables edit-confirm/regenerate so a
   * second generate_response call can't race the one already running. */
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: "edit", newText: string): void;
  (e: "regenerate"): void;
  (e: "delete"): void;
  (e: "rate", rating: Rating | null, correction?: string): void;
}>();

const isEditing = ref(false);
const draft = ref("");
const correcting = ref(false);
const correctionDraft = ref("");

function rate(rating: Rating) {
  if (props.message.streaming || props.message.errored) return;
  if (props.message.rating === rating) {
    correcting.value = false;
    emit("rate", null);
    return;
  }
  emit("rate", rating);
  correcting.value = rating === "bad";
  correctionDraft.value = props.message.correction ?? "";
}

function saveCorrection() {
  emit("rate", "bad", correctionDraft.value.trim() || undefined);
  correcting.value = false;
}

function onCorrectionKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    saveCorrection();
  } else if (e.key === "Escape") {
    e.preventDefault();
    correcting.value = false;
  }
}

function startEdit() {
  if (props.disabled) return;
  draft.value = props.message.content;
  isEditing.value = true;
}

function confirmEdit() {
  const trimmed = draft.value.trim();
  if (!trimmed || props.disabled) return;
  emit("edit", trimmed);
  isEditing.value = false;
}

function cancelEdit() {
  isEditing.value = false;
  draft.value = props.message.content;
}

function onEditKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    confirmEdit();
  } else if (e.key === "Escape") {
    e.preventDefault();
    cancelEdit();
  }
}

function onRegenerate() {
  if (props.disabled) return;
  emit("regenerate");
}

function onDelete() {
  if (props.disabled) return;
  const msg =
    "Delete this message" +
    (props.message.role === "user" ? " and its reply" : "") +
    "? This can't be undone.";
  if (!window.confirm(msg)) return;
  emit("delete");
}
</script>

<template>
  <div class="row" :class="message.role">
    <div class="bubble" :class="{ errored: message.errored, editing: isEditing }">
      <template v-if="isEditing">
        <textarea
          v-model="draft"
          class="edit-box"
          rows="2"
          autofocus
          @keydown="onEditKeydown"
        ></textarea>
        <div class="edit-actions">
          <button class="edit-btn confirm" title="Confirm (Enter)" @click="confirmEdit">✓</button>
          <button class="edit-btn cancel" title="Cancel (Esc)" @click="cancelEdit">✕</button>
        </div>
      </template>

      <template v-else>
        <div class="content">
          {{ message.content }}<span v-if="message.streaming && message.content" class="cursor">▍</span>
        </div>
        <div v-if="message.edited" class="edited-tag">(edited)</div>
        <div v-if="message.agent === 'sloth'" class="agent-badge sloth">Sloth</div>
        <div v-if="message.correction && !correcting" class="correction-note">
          Should have said: {{ message.correction }}
        </div>
        <div v-if="correcting" class="correction">
          <textarea
            v-model="correctionDraft"
            class="edit-box"
            rows="2"
            placeholder="What should Xenon have said? (optional)"
            autofocus
            @keydown="onCorrectionKeydown"
          ></textarea>
          <div class="edit-actions">
            <button class="edit-btn confirm" title="Save (Enter)" @click="saveCorrection">✓</button>
            <button class="edit-btn cancel" title="Skip (Esc)" @click="correcting = false">✕</button>
          </div>
        </div>

        <div class="icons">
          <template v-if="message.role === 'user'">
            <button class="icon-btn" title="Edit" @click="startEdit">✎</button>
            <button class="icon-btn" title="Delete" @click="onDelete">
              <!-- Inline SVG -- 🗑 was found to render as a garbled fallback glyph in this
                   environment's WebView2 (missing color-emoji font support). -->
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
            </button>
          </template>
          <template v-else>
            <template v-if="!message.streaming && !message.errored">
              <button
                class="icon-btn rate"
                :class="{ active: message.rating === 'good' }"
                :title="message.rating === 'good' ? 'Clear rating' : 'Good reply'"
                @click="rate('good')"
              >
                <!-- SVG, not 👍/👎: emoji render as garbled glyphs in this WebView2 (see below). -->
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M7 10v12" />
                  <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
                </svg>
              </button>
              <button
                class="icon-btn rate"
                :class="{ active: message.rating === 'bad' }"
                :title="message.rating === 'bad' ? 'Clear rating' : 'Bad reply'"
                @click="rate('bad')"
              >
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M17 14V2" />
                  <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
                </svg>
              </button>
            </template>
            <button class="icon-btn" title="Regenerate" @click="onRegenerate">⟳</button>
            <button class="icon-btn" title="Delete" @click="onDelete">
              <!-- Inline SVG -- 🗑 was found to render as a garbled fallback glyph in this
                   environment's WebView2 (missing color-emoji font support). -->
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
            </button>
          </template>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.row {
  display: flex;
  width: 100%;
  margin: 0.35rem 0;
}

.row.user {
  justify-content: flex-end;
}

.row.assistant {
  justify-content: flex-start;
}

.bubble {
  position: relative;
  max-width: 70%;
  padding: 0.6rem 0.85rem;
  border-radius: 12px;
  font-size: 0.92rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

.bubble.editing {
  max-width: 85%;
  width: 70%;
}

.row.user .bubble {
  background: #2f6feb;
  color: #fff;
  border-bottom-right-radius: 3px;
}

.row.assistant .bubble {
  background: #2a2a2c;
  color: #eee;
  border-bottom-left-radius: 3px;
}

.bubble.errored {
  background: #5a2323;
  color: #ffd6d6;
}

.cursor {
  display: inline-block;
  animation: blink 1s steps(1) infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.edited-tag {
  margin-top: 0.15rem;
  font-size: 0.7rem;
  opacity: 0.7;
  font-style: italic;
}

.agent-badge {
  display: inline-block;
  margin-top: 0.3rem;
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
}

.agent-badge.sloth {
  background: #3a3a6b;
  color: #cfcfff;
}

.edit-box {
  width: 100%;
  resize: vertical;
  min-height: 2.6rem;
  padding: 0.4rem 0.5rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(0, 0, 0, 0.2);
  color: inherit;
  font: inherit;
  box-sizing: border-box;
}

.edit-box:focus {
  outline: none;
  border-color: rgba(255, 255, 255, 0.55);
}

.edit-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.35rem;
  margin-top: 0.35rem;
}

.edit-btn {
  border: none;
  border-radius: 5px;
  width: 1.6rem;
  height: 1.6rem;
  font-size: 0.8rem;
  cursor: pointer;
  line-height: 1;
  background: rgba(0, 0, 0, 0.35);
  color: #fff;
}

.edit-btn.confirm:hover {
  background: #2f9e57;
}

.edit-btn.cancel:hover {
  background: #b8402f;
}

.icons {
  position: absolute;
  top: -1.3rem;
  display: flex;
  gap: 0.25rem;
  opacity: 0;
  transition: opacity 0.12s ease;
}

.row.user .icons {
  right: 0;
}

.row.assistant .icons {
  left: 0;
}

.bubble:hover .icons {
  opacity: 1;
}

.icon-btn {
  border: none;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  border-radius: 5px;
  width: 1.5rem;
  height: 1.5rem;
  font-size: 0.75rem;
  cursor: pointer;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}

.icon-btn.rate.active {
  background: #2f6feb;
}

/* A rated reply keeps its rating visible without hovering. */
.bubble:has(.icon-btn.rate.active) .icons {
  opacity: 1;
}

.correction {
  margin-top: 0.45rem;
}

.correction-note {
  margin-top: 0.3rem;
  font-size: 0.75rem;
  opacity: 0.75;
  font-style: italic;
}
</style>
