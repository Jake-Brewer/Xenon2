# Xenon2 — On-Disk Conversation Schema (Phase 5)

This document defines the JSON shape Xenon2 writes when it saves a conversation to disk (File >
Save / Save As, auto-save, and Phase 6's future export), and what File > Open validates against.
It's based directly on the existing in-memory `Conversation`/`ChatMessage` TypeScript types
(`app/src/types.ts`), not a redesign — see "Differences from the in-memory type" below for the
handful of deliberate deltas.

The authoritative Rust mirror of this schema lives in `app/src-tauri/src/persistence.rs`
(`ConversationFile`/`ConversationFileMessage`); the authoritative TypeScript mirror lives in
`app/src/persistence.ts` (`ConversationFile`/`ConversationFileMessage`). Both are kept in sync by
hand — if you change one, change the other and this doc.

## File extension / MIME

`.json`, UTF-8, pretty-printed (2-space indent via `serde_json::to_string_pretty`) — deliberately
not minified, so a saved file is human-readable when opened in a plain text editor (this is one of
Phase 5's acceptance criteria).

## Top-level shape

```jsonc
{
  "schemaVersion": 1,
  "id": "b6e6c8c2-2222-4b3e-9b1a-2f7a6a7b9a11",
  "title": "Help me plan a trip to Kyoto",
  "model": "rwkv-5-world-0.4B-Q4_0",
  "createdAt": 1753747200000,
  "savedAt": 1753747265123,
  "messages": [
    {
      "id": "1a2b3c4d-...",
      "role": "user",
      "content": "Help me plan a trip to Kyoto",
      "timestamp": 1753747200000
    },
    {
      "id": "5e6f7a8b-...",
      "role": "assistant",
      "content": "Sure! Kyoto is beautiful in spring...",
      "timestamp": 1753747210456
    }
  ]
}
```

### Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `schemaVersion` | integer | yes | Currently always `1`. See "Versioning" below. |
| `id` | string (UUID) | yes | Matches the in-memory `Conversation.id`; stable across saves so re-saving the same conversation overwrites rather than duplicates. |
| `title` | string | yes | Matches `Conversation.title`. |
| `model` | string | yes | Name/version of the model that generated this conversation's replies, e.g. `"rwkv-5-world-0.4B-Q4_0"` — the loaded model file's name minus its `.bin` extension. Recorded faithfully; **not validated** against the currently-loaded model this phase (a future non-RWKV backend would need to check this before naively reloading — see `PLAN.md`'s "Future work"). |
| `createdAt` | integer (epoch ms) | yes | Matches `Conversation.createdAt`. Used for Today/Yesterday/Older sidebar grouping on reload. |
| `savedAt` | integer (epoch ms) | yes | New field, not on the in-memory type — timestamp of this particular write (auto-save or manual). Informational only (e.g. could support a future "last saved 2 minutes ago" indicator); not required to round-trip back into the in-memory `Conversation`. |
| `messages` | array of message objects | yes | May be empty (a brand-new, never-messaged conversation). |

### Message object fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string (UUID) | yes | Matches `ChatMessage.id`. |
| `role` | `"user"` \| `"assistant"` | yes | Any other value fails validation on load (see "Validation on Open"). |
| `content` | string | yes | The message text. |
| `timestamp` | integer (epoch ms) | yes | Matches `ChatMessage.timestamp`. |
| `edited` | boolean | no | Omitted entirely when `false`/unset (not written as `"edited": false`) — present and `true` only once a user message has been edited. Matches `ChatMessage.edited`. |
| `rating` | `"good"` \| `"bad"` | no | The user's rating of an assistant reply. Omitted when unrated, and cleared when the reply is regenerated. Every change is also appended to `feedback.jsonl` (see `ONLINE_LEARNING.md` and `app/src-tauri/src/feedback.rs`). |
| `correction` | string | no | What the user said Xenon should have replied instead. Only set with a `"bad"` rating. |
| `agent` | `"dementia"` \| `"sloth"` | no | Phase 7 follow-up. Which agent generated this reply — omitted on user messages and on any message saved before this field existed (treated as `"dementia"`, the default/no-memory behavior). See `app/README.md`'s "Dementia/Sloth" section. |

## Differences from the in-memory `Conversation`/`ChatMessage` type

The on-disk shape is the same data model, not a redesign, with two deliberate deltas:

1. **`model` and `savedAt` are new** (task 5's requirement, plus a natural "when was this
   written" companion field). `model` was added to the in-memory `Conversation` type too
   (`app/src/types.ts`, optional field) so it round-trips cleanly and survives being displayed
   without a separate side-channel; `savedAt` was *not* added to the in-memory type since nothing
   in the UI currently needs "last saved at" as live state — it's computed fresh at write time
   from the in-memory `Conversation`, which doesn't carry a stale copy of it.
2. **`streaming` and `errored` are dropped.** Both are transient UI-only state on `ChatMessage`
   that never make sense "at rest" in a saved file:
   - `streaming: true` would mean a message got saved mid-generation, which cannot happen — see
     "Why a persisted message is never mid-stream" below.
   - `errored: true` marks a failed-generation placeholder whose `content` is an error string, not
     real generated content; auto-save never fires for a failed generation (see
     "Auto-save default path policy" below), so this case doesn't arise for auto-save. A user
     *could* in principle hit Save/Save As while an error banner is showing on some other,
     previously-failed message; in that case the errored placeholder's text is still saved as
     the message's `content` like any other message, it just won't carry an `errored` flag on
     reload — the same content is shown, just without the special error styling. This is an
     accepted, minor cosmetic gap, not a data-loss risk.

   Neither field is present in `ConversationFileMessage`; loading a file always produces messages
   with both fields simply absent (falsy), which is correct — a freshly loaded message is by
   definition not currently streaming and not in an error state.

### Why a persisted message is never mid-stream

Auto-save (task 4) is wired to fire from `completeGeneration` in `app/src/stores/chat.ts` — the
handler for the Tauri `generation-done` event — not from `appendToken` (fired per-token while
streaming) or `failGeneration`. By the time any write to disk happens, the message it's writing
has already had `streaming` set back to `false` in memory. A hard crash mid-stream loses only that
one in-flight reply (back to whatever was last auto-saved after the previous completed exchange),
never leaves a corrupted "half-written" message in the file. See `app/README.md`'s persistence
section for the full statement of this guarantee.

## Auto-save default path policy

Task 4 requires a real, documented answer for "where does auto-save write when the user has never
manually saved this conversation?" — silently losing history until a manual save is not
acceptable.

**Policy:** the first time a given conversation is auto-saved (i.e. it completes a generation and
has no previously-established file path), Xenon2 writes it to a default location inside the OS
app-data directory, keyed by conversation id:

```
<app-data-dir>/conversations/<conversation-id>.json
```

On Windows this resolves (via Tauri's `app_data_dir()`, driven by the `com.xenon2.app` identifier
in `tauri.conf.json`) to something like:

```
C:\Users\<user>\AppData\Roaming\com.xenon2.app\conversations\<id>.json
```

Once that first write happens, the path is remembered (`filePaths[conversationId]` in the Pinia
store, persisted in `session.json` — see below) and every subsequent auto-save for that
conversation reuses the same path silently — no repeated dialogs, no repeated path decisions.

**Why this over prompting on first auto-save:** a save-file dialog interrupting the user
mid-conversation, immediately after their first reply finishes streaming, would be intrusive and
easy to accidentally cancel (silently discarding the "so don't lose history" guarantee this task
exists for). The app-data default means history is never lost by default, with zero required user
action; File > Save As remains available at any time for the user to explicitly relocate a
conversation to a location of their choosing (see below), and once they do, auto-save follows
that new path instead.

**Interaction with Save As:** File > Save As always opens a dialog (even if a path already
exists) and, once the user picks a location, all subsequent auto-saves for that conversation
switch to the new path — the app-data default is only ever a fallback for conversations no one has
explicitly relocated.

## `session.json` — not part of this schema, but load-bearing for restore-on-launch

A second, small file — `<app-data-dir>/session.json` — is *not* a conversation file and does not
use `schemaVersion`/the shape above. It's app-level bookkeeping the Pinia store reads/writes
(`restoreSession`/`persistSession` in `app/src/stores/chat.ts`, `load_session_file`/
`save_session_file` in `app/src-tauri/src/persistence.rs`) so relaunching Xenon2 can restore the
sidebar and the last-active conversation automatically:

```jsonc
{
  "lastActiveConversationId": "b6e6c8c2-2222-4b3e-9b1a-2f7a6a7b9a11",
  "conversationPaths": {
    "b6e6c8c2-2222-4b3e-9b1a-2f7a6a7b9a11": "C:\\Users\\...\\conversations\\b6e6c8c2....json",
    "3f9d1a00-...": "D:\\my-xenon-chats\\vacation-planning.json"
  }
}
```

`conversationPaths` covers both app-data-default paths and user-chosen Save As paths uniformly —
on startup, the app reads every path it lists, loads each conversation, and populates the sidebar
with all of them (skipping, with a console warning, any individual file that's gone missing or
become corrupted since the last run — one bad file does not block the rest of startup or crash the
app). `lastActiveConversationId` then picks which one is shown initially, satisfying "closing and
reopening the app restores the most recently active conversation without the user needing to
manually click Open."

## Validation on Open

File > Open (and session restore, which uses the same backend command,
`open_conversation_file`) rejects anything that isn't a well-formed file matching this schema,
returning a specific error message rather than crashing or corrupting UI state:

- **Not valid JSON at all** → `"'<path>' is not valid JSON (<parse error>). This does not look
  like a Xenon2 conversation file."`
- **Valid JSON, but missing required fields / wrong types** (e.g. a `.json` file that isn't a
  Xenon2 conversation at all, or a `role` value other than `"user"`/`"assistant"`) →
  `"'<path>' is valid JSON but not a valid Xenon2 conversation (<serde error>). Expected fields:
  schemaVersion, id, title, model, createdAt, savedAt, messages[]."`
- **Valid shape, but an unrecognized `schemaVersion`** → `"'<path>' uses schema version <N>, but
  this build of Xenon2 only supports version 1."`

All three are surfaced to the user via the file-error banner in `App.vue` (`store.fileError`),
distinct from the existing inference-error banner (`store.backendError`) since they're a different
failure domain. None of them throw an unhandled exception or leave the store in a partially-mutated
state — the store only replaces/pushes a conversation *after* a successful parse+validate.

## Versioning

`schemaVersion` exists so a future incompatible change to this shape can be detected and either
migrated or rejected with a clear message, instead of silently misinterpreting old files. There is
no migration logic yet (version 1 is the only version that has ever existed), but any future schema
change must bump this number and add an explicit migration path (or an explicit "version too old,
please re-export" rejection) rather than reusing `1` for an incompatible shape.

## What was actually tested

See `app/README.md`'s "Phase 5 verification" section for the full account of the real dev-build
test pass (save, close/reopen restore, Save As re-pathing, malformed-file rejection) this schema
was verified against — not just unit-level schema checks in isolation.
