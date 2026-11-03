// persistence.ts -- Phase 5's frontend counterpart to src-tauri/src/persistence.rs.
//
// Defines the on-disk JSON shape (`ConversationFile`, matching SCHEMA.md at the repo root) and
// the small session-bookkeeping shape (`SessionFile`), plus pure conversion helpers between them
// and the in-memory `Conversation`/`ChatMessage` types (types.ts). Kept separate from
// stores/chat.ts so the store's actions stay focused on *when* to save/load, not the shape
// conversion itself.
//
// `streaming`/`errored` are intentionally dropped when converting to `ConversationFile` -- they're
// transient UI state that never makes sense at rest. This is safe because auto-save only ever
// runs after a generation *completes* successfully (see chat.ts's `completeGeneration`), so a
// conversation being serialized never has a message still mid-stream or holding an error
// placeholder as its "real" content.

import type { Conversation, ChatMessage, Rating, Role } from "./types";

export interface ConversationFileMessage {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  edited?: boolean;
  agent?: "dementia" | "sloth";
  rating?: Rating;
  correction?: string;
}

export interface ConversationFile {
  schemaVersion: number;
  id: string;
  title: string;
  model: string;
  createdAt: number;
  savedAt: number;
  messages: ConversationFileMessage[];
}

export interface SessionFile {
  lastActiveConversationId: string | null;
  conversationPaths: Record<string, string>;
}

export const SCHEMA_VERSION = 1;

/** Converts an in-memory conversation to the on-disk shape. `fallbackModel` is used if the
 * conversation somehow has no `model` set yet (e.g. an empty brand-new chat that gets saved
 * before its first generation records one). */
export function toConversationFile(convo: Conversation, fallbackModel: string): ConversationFile {
  return {
    schemaVersion: SCHEMA_VERSION,
    id: convo.id,
    title: convo.title,
    model: convo.model ?? fallbackModel,
    createdAt: convo.createdAt,
    savedAt: Date.now(),
    messages: convo.messages.map((m) => {
      const out: ConversationFileMessage = {
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
      };
      if (m.edited) out.edited = true;
      if (m.agent) out.agent = m.agent;
      if (m.rating) out.rating = m.rating;
      if (m.correction) out.correction = m.correction;
      return out;
    }),
  };
}

/** Converts a file loaded from disk back into an in-memory conversation. `streaming`/`errored`
 * are left undefined (a loaded message is always "at rest", never mid-generation or a failed
 * placeholder). */
export function fromConversationFile(file: ConversationFile): Conversation {
  return {
    id: file.id,
    title: file.title,
    createdAt: file.createdAt,
    model: file.model,
    messages: file.messages.map(
      (m): ChatMessage => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        edited: m.edited,
        agent: m.agent,
        rating: m.rating,
        correction: m.correction,
      })
    ),
  };
}

/** Suggests a filesystem-safe default file name for Save As, derived from the conversation title. */
export function suggestFileName(convo: Conversation): string {
  const safeTitle = (convo.title || "New Chat")
    .replace(/[\\/:*?"<>|]/g, "_")
    .trim()
    .slice(0, 60);
  return `${safeTitle || "conversation"}.json`;
}
