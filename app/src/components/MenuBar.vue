<script setup lang="ts">
// MenuBar.vue -- File menu.
//
// Phase 5: Open / Save / Save As are now real (see stores/chat.ts's openConversation /
// saveConversation / saveConversationAs), wired the same way "New" already was -- this component
// just emits, App.vue calls the store. Save / Save As are disabled when there's no active
// conversation to save. Export Memory remains an intentional visual stub -- its real logic is
// Phase 6's job, not this phase's.
//
// Phase 7 fix: the dropdown used to close via @blur on the "File" trigger button, protected only
// by @mousedown.prevent on the dropdown container to stop a *real* mouse click from shifting
// focus (and blurring) before its own click landed. That protection doesn't cover every path that
// can programmatically focus a dropdown item -- discovered via UI Automation's InvokePattern
// (used for this app's own automated verification, see app/README.md's Phase 4 notes), which
// shifts focus to the target element as part of invoking it, firing blur on the "File" button
// first and unmounting the dropdown (and the very button being invoked) before its click finished
// -- so the click silently never landed. Closing on an outside click/mousedown instead (tracked
// via a template ref, not focus-based) doesn't have this race for any input method.

import { ref, onMounted, onUnmounted } from "vue";

const props = defineProps<{
  /** Disables Save / Save As when there's no active conversation. */
  hasActiveConversation?: boolean;
  /** Shown on the right of the menu bar, e.g. "CUDA — all layers on GPU". */
  deviceLabel?: string;
  sidebarOpen?: boolean;
}>();

const emit = defineEmits<{
  (e: "new"): void;
  (e: "open"): void;
  (e: "save"): void;
  (e: "save-as"): void;
  (e: "export-memory"): void;
  (e: "import-memory"): void;
  (e: "settings"): void;
  (e: "sloth-memory"): void;
  (e: "toggle-sidebar"): void;
}>();

const fileMenuOpen = ref(false);
const menuItemEl = ref<HTMLElement | null>(null);

function toggleFileMenu() {
  fileMenuOpen.value = !fileMenuOpen.value;
}

function closeFileMenu() {
  fileMenuOpen.value = false;
}

function onDocumentMousedown(e: MouseEvent) {
  if (!fileMenuOpen.value) return;
  if (menuItemEl.value && !menuItemEl.value.contains(e.target as Node)) {
    closeFileMenu();
  }
}

onMounted(() => document.addEventListener("mousedown", onDocumentMousedown));
onUnmounted(() => document.removeEventListener("mousedown", onDocumentMousedown));

function onNew() {
  emit("new");
  closeFileMenu();
}

function onOpen() {
  emit("open");
  closeFileMenu();
}

function onSave() {
  if (!props.hasActiveConversation) return;
  emit("save");
  closeFileMenu();
}

function onSaveAs() {
  if (!props.hasActiveConversation) return;
  emit("save-as");
  closeFileMenu();
}

function onExportMemory() {
  emit("export-memory");
  closeFileMenu();
}

function onImportMemory() {
  emit("import-memory");
  closeFileMenu();
}

function onSettings() {
  emit("settings");
  closeFileMenu();
}

function onSlothMemory() {
  emit("sloth-memory");
  closeFileMenu();
}
</script>

<template>
  <div class="menu-bar" @keydown.escape="closeFileMenu">
    <button
      class="menu-item-btn"
      :title="sidebarOpen ? 'Hide conversation list' : 'Show conversation list'"
      @click="emit('toggle-sidebar')"
    >
      {{ sidebarOpen ? "Hide chats" : "Chats" }}
    </button>
    <div class="menu-item" ref="menuItemEl">
      <button class="menu-item-btn" @click="toggleFileMenu">File</button>
      <div v-if="fileMenuOpen" class="dropdown">
        <button class="dropdown-item" @click="onNew">New</button>
        <button class="dropdown-item" @click="onOpen">Open...</button>
        <button
          class="dropdown-item"
          :class="{ disabled: !hasActiveConversation }"
          @click="onSave"
        >
          Save
        </button>
        <button
          class="dropdown-item"
          :class="{ disabled: !hasActiveConversation }"
          @click="onSaveAs"
        >
          Save As...
        </button>
        <div class="dropdown-sep"></div>
        <button class="dropdown-item" @click="onExportMemory">Export Memory...</button>
        <button class="dropdown-item" @click="onImportMemory">Import Memory...</button>
        <div class="dropdown-sep"></div>
        <button class="dropdown-item" @click="onSettings">Data Directory Settings...</button>
        <button class="dropdown-item" @click="onSlothMemory">Sloth Memory...</button>
      </div>
    </div>
    <span v-if="deviceLabel" class="device-label" :title="deviceLabel">{{ deviceLabel }}</span>
  </div>
</template>

<style scoped>
.menu-bar {
  display: flex;
  align-items: stretch;
  background: var(--menu-bg, #1e1e1f);
  color: var(--menu-fg, #ddd);
  font-size: 0.85rem;
  user-select: none;
  -webkit-app-region: drag;
  min-width: 0;
}

.device-label {
  margin-left: auto;
  align-self: center;
  padding: 0 0.75rem;
  color: #9ab;
  font-size: 0.75rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 46vw;
  -webkit-app-region: no-drag;
}

.menu-item {
  position: relative;
  -webkit-app-region: no-drag;
}

.menu-item-btn {
  padding: 0.4rem 0.75rem;
  cursor: default;
  outline: none;
  background: none;
  border: none;
  color: inherit;
  font: inherit;
}

.menu-item-btn:hover,
.menu-item-btn:focus {
  background: rgba(255, 255, 255, 0.08);
}

.dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 170px;
  background: #2a2a2c;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  padding: 4px;
  z-index: 50;
}

.dropdown-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  color: #eee;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
  border-radius: 4px;
  cursor: pointer;
}

.dropdown-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.dropdown-item.stub {
  color: #aaa;
}

.dropdown-item.disabled {
  color: #666;
  cursor: default;
}

.dropdown-item.disabled:hover {
  background: none;
}

.dropdown-sep {
  height: 1px;
  margin: 4px 2px;
  background: rgba(255, 255, 255, 0.1);
}
</style>
