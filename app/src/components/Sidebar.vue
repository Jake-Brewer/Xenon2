<script setup lang="ts">
// Sidebar.vue -- conversation list grouped by date (Today / Yesterday / Older), "+ New Chat",
// click-to-load. Conversations are an in-memory prop from App.vue; real persistence is Phase 5's
// job (see prompts/phase3_prompt.md).

import { computed } from "vue";
import type { Conversation } from "../types";

const props = defineProps<{
  conversations: Conversation[];
  activeId: string | null;
}>();

const emit = defineEmits<{
  (e: "new-chat"): void;
  (e: "select", id: string): void;
}>();

function startOfDay(ts: number): number {
  const d = new Date(ts);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

const groups = computed(() => {
  const today = startOfDay(Date.now());
  const yesterday = today - 24 * 60 * 60 * 1000;

  const todayList: Conversation[] = [];
  const yesterdayList: Conversation[] = [];
  const olderList: Conversation[] = [];

  // Newest first within each group.
  const sorted = [...props.conversations].sort((a, b) => b.createdAt - a.createdAt);

  for (const c of sorted) {
    const day = startOfDay(c.createdAt);
    if (day === today) todayList.push(c);
    else if (day === yesterday) yesterdayList.push(c);
    else olderList.push(c);
  }

  return [
    { label: "Today", items: todayList },
    { label: "Yesterday", items: yesterdayList },
    { label: "Older", items: olderList },
  ].filter((g) => g.items.length > 0);
});
</script>

<template>
  <aside class="sidebar">
    <button class="new-chat-btn" @click="emit('new-chat')">+ New Chat</button>

    <div class="conversation-list">
      <div v-if="conversations.length === 0" class="empty-hint">No conversations yet</div>

      <div v-for="group in groups" :key="group.label" class="group">
        <div class="group-label">{{ group.label }}</div>
        <button
          v-for="c in group.items"
          :key="c.id"
          class="conversation-item"
          :class="{ active: c.id === activeId }"
          @click="emit('select', c.id)"
        >
          {{ c.title || "New Chat" }}
        </button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  width: clamp(168px, 22vw, 240px);
  min-width: 0;
  flex: 0 1 240px;
  background: var(--sidebar-bg, #191919);
  color: #e6e6e6;
  height: 100%;
  box-sizing: border-box;
  padding: 0.6rem;
  gap: 0.6rem;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

@media (max-width: 859px) {
  .sidebar {
    position: absolute;
    z-index: 30;
    top: 0;
    bottom: 0;
    left: 0;
    width: min(260px, calc(100vw - 3rem));
    flex: none;
    box-shadow: 8px 0 24px rgba(0, 0, 0, 0.45);
  }
}

.new-chat-btn {
  width: 100%;
  padding: 0.55rem 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: rgba(255, 255, 255, 0.05);
  color: #fff;
  font-size: 0.9rem;
  cursor: pointer;
  text-align: left;
}

.new-chat-btn:hover {
  background: rgba(255, 255, 255, 0.12);
}

.conversation-list {
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.empty-hint {
  color: #777;
  font-size: 0.82rem;
  padding: 0.4rem 0.2rem;
}

.group-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #888;
  padding: 0.2rem 0.3rem;
}

.conversation-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  color: #ddd;
  padding: 0.45rem 0.5rem;
  font-size: 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conversation-item:hover {
  background: rgba(255, 255, 255, 0.08);
}

.conversation-item.active {
  background: rgba(255, 255, 255, 0.14);
  color: #fff;
}
</style>
