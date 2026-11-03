// Shared frontend types for the chat UI.
//
// As of Phase 5, conversations are durable: the Pinia store (stores/chat.ts) still owns all
// in-memory state, but it now serializes to/from disk via Tauri commands (src-tauri/src/
// persistence.rs). The on-disk JSON shape is documented in SCHEMA.md at the repo root -- these
// types are the in-memory counterpart, not identical field-for-field (see `model` below, and note
// `streaming`/`errored` are UI-only and never persisted).

export type Role = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  /** True while an assistant reply is still streaming in token-by-token. */
  streaming?: boolean;
  /** True if generation ended in an error (message.content holds the error text). */
  errored?: boolean;
  /** True once a user message has been edited via the Phase 4 edit flow. Assistant messages
   * don't use this -- a regenerated assistant message just gets new content, not an "edited"
   * badge (there's no original-vs-edited distinction to show for a replaced reply). */
  edited?: boolean;
  /** Epoch ms when this message was created. Set once at creation and left alone afterwards
   * (including on edit) -- it records when the turn was first created, not last touched.
   * Regenerating an assistant message's content does update this, since regeneration produces an
   * effectively new reply at a new point in time. */
  timestamp: number;
  /** Which agent generated this assistant reply -- "dementia" (default: no memory outside this
   * chat, exactly the pre-existing behavior) or "sloth" (reads/writes the persistent
   * cross-conversation fact store, see stores/chat.ts's slothFacts). Set on assistant messages
   * only; undefined on user messages and on any message created before this feature existed.
   * The active agent is a mid-conversation toggle (store.activeAgent), not a per-conversation
   * setting, so a single conversation can contain a mix of both. */
  agent?: "dementia" | "sloth";
  /** The user's rating of an assistant reply, collected for learning from feedback (see
   * ONLINE_LEARNING.md). Cleared when the reply is regenerated, since that is a new reply. */
  rating?: Rating;
  /** What the user said Xenon should have replied instead. */
  correction?: string;
}

export type Rating = "good" | "bad";

export interface Conversation {
  id: string;
  title: string;
  createdAt: number; // epoch ms, used for Today/Yesterday/Older grouping
  messages: ChatMessage[];
  /** Name/version of the model this conversation was generated with, e.g.
   * "rwkv-5-world-0.4B-Q4_0" (see SCHEMA.md). Set from the backend's actually-loaded model as
   * soon as the conversation is created; only absent for conversations from a version of Xenon2
   * that predates this field (loading such a file would be unusual since Phase 5 is the first
   * phase with any on-disk format at all). Phase 5 does not validate this against the currently
   * loaded model -- see phase5_prompt.md task 5. */
  model?: string;
}

export function newId(): string {
  return crypto.randomUUID();
}
