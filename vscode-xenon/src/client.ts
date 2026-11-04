// Talks to the Xenon2 app's local coding API (app/src-tauri/src/code_api.rs). Kept free of the
// vscode module so it can be tested with plain Node.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Connection {
  url: string;
  token: string;
}

export interface Settings {
  serverUrl: string;
  token: string;
  maxTokens: number;
  temperature: number;
}

export type Finish = "stop" | "length" | "cancelled";

/** Tauri's app data dir for the app's identifier, where the app writes code-api.json. */
export function discoveryPath(
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env,
  home: string = os.homedir(),
): string {
  const id = "com.xenon2.app";
  if (platform === "win32") {
    return path.join(env.APPDATA ?? path.join(home, "AppData", "Roaming"), id, "code-api.json");
  }
  if (platform === "darwin") {
    return path.join(home, "Library", "Application Support", id, "code-api.json");
  }
  return path.join(env.XDG_DATA_HOME || path.join(home, ".local", "share"), id, "code-api.json");
}

/** Settings win; anything left empty comes from the app's discovery file. */
export function resolveConnection(settings: Pick<Settings, "serverUrl" | "token">, file = discoveryPath()): Connection {
  let found: Partial<Connection> = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    found = { url: parsed.url, token: parsed.token };
  } catch {
    // No file yet: the app has never run, or runs on another machine.
  }
  const url = (settings.serverUrl || found.url || "").replace(/\/+$/, "");
  const token = settings.token || found.token || "";
  if (!url || !token) {
    throw new Error("The Xenon2 app isn't running (no code-api.json found). Start the app, then try again.");
  }
  return { url, token };
}

/** Splits a server-sent event stream into the JSON payloads of its `data:` lines. */
export class SseParser {
  private buffer = "";

  push(chunk: string): string[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const events: string[] = [];
    let end: number;
    while ((end = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) events.push(data);
    }
    return events;
  }
}

/** Rates a reply, or sends the fix when it was wrong. Learning is instant: the next request that
 * resembles this one pulls the rated example into its prompt (see `example_bank.rs`). */
export async function sendFeedback(
  connection: Connection,
  feedback: { messageId: string; rating: "good" | "bad" | "none"; correction?: string; history: ChatMessage[]; reply: string },
): Promise<void> {
  const response = await fetch(`${connection.url}/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${connection.token}` },
    body: JSON.stringify({
      message_id: feedback.messageId,
      rating: feedback.rating,
      correction: feedback.correction,
      history: feedback.history,
      reply: feedback.reply,
    }),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Xenon2 answered ${response.status}: ${detail}`);
  }
}

export async function health(connection: Connection, signal?: AbortSignal): Promise<{ model: string }> {
  const response = await fetch(`${connection.url}/health`, { signal });
  if (!response.ok) throw new Error(`Xenon2 answered ${response.status}`);
  return (await response.json()) as { model: string };
}

/**
 * Streams a reply, calling onText with each new piece. Resolves with why it ended; aborting
 * `signal` stops the model and resolves "cancelled".
 */
export async function streamChat(
  connection: Connection,
  messages: ChatMessage[],
  settings: Pick<Settings, "maxTokens" | "temperature">,
  onText: (text: string) => void,
  signal?: AbortSignal,
): Promise<Finish> {
  let response: Response;
  try {
    response = await fetch(`${connection.url}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${connection.token}` },
      body: JSON.stringify({
        model: "xenon",
        messages,
        stream: true,
        max_tokens: settings.maxTokens,
        temperature: settings.temperature,
      }),
      signal,
    });
  } catch (err) {
    if (signal?.aborted) return "cancelled";
    throw new Error(`Can't reach the Xenon2 app at ${connection.url}. Is it running?`);
  }
  if (response.status === 401) {
    throw new Error("Xenon2 rejected the token. Clear the xenon.token setting, or restart the app.");
  }
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Xenon2 answered ${response.status}: ${detail}`);
  }

  const parser = new SseParser();
  const decoder = new TextDecoder();
  let finish: Finish = "stop";
  try {
    for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
      for (const data of parser.push(decoder.decode(chunk, { stream: true }))) {
        if (data === "[DONE]") return finish;
        const event = JSON.parse(data);
        if (event.error) throw new Error(event.error.message ?? "Xenon2 reported an error");
        const choice = event.choices?.[0];
        if (choice?.delta?.content) onText(choice.delta.content);
        if (choice?.finish_reason === "length") finish = "length";
      }
    }
  } catch (err) {
    if (signal?.aborted) return "cancelled";
    throw err;
  }
  return finish;
}
