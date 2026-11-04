// The Xenon sidebar: a chat that streams replies and can put code from them into the editor.

import * as vscode from "vscode";
import { ChatMessage, Finish, resolveConnection, sendFeedback, streamChat } from "./client";
import { CodeContext, firstCodeBlock, withContext } from "./context";

export interface Host {
  settings(): { serverUrl: string; token: string; maxTokens: number; temperature: number };
  /** Code from the editor the user last worked in, or undefined when there is none. */
  currentContext(): CodeContext | undefined;
  /** The editor and range a "replace" applies to. */
  lastTarget(): { editor: vscode.TextEditor; range: vscode.Range } | undefined;
  setBusy(busy: boolean): void;
}

type FromWebview =
  | { type: "ready" }
  | { type: "send"; text: string; includeContext: boolean }
  | { type: "stop" }
  | { type: "newChat" }
  | { type: "insert" | "replace" | "copy"; code: string }
  | { type: "rate"; messageId: string; rating: "good" | "bad" }
  | { type: "correct"; messageId: string; correction: string };

export class ChatViewProvider implements vscode.WebviewViewProvider {
  static readonly viewId = "xenon.chat";

  private view?: vscode.WebviewView;
  private messages: ChatMessage[] = [];
  /** The code last sent in this chat, so an unchanged file isn't sent (and read) twice. */
  private lastSentCode?: string;
  private abort?: AbortController;
  private queued?: { text: string; context?: CodeContext };
  /** History and reply text for each rateable assistant message, by the id given to the webview. */
  private rateable = new Map<string, { history: ChatMessage[]; reply: string }>();
  private nextMessageId = 0;

  constructor(private readonly extensionUri: vscode.Uri, private readonly host: Host) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")] };
    view.webview.html = this.html(view.webview);
    view.webview.onDidReceiveMessage((message: FromWebview) => this.onMessage(message));
  }

  get generating(): boolean {
    return this.abort !== undefined;
  }

  /** Sends a question from a command, opening the view first if needed. */
  async ask(text: string, context: CodeContext | undefined): Promise<void> {
    if (!this.view) {
      this.queued = { text, context };
      await vscode.commands.executeCommand(`${ChatViewProvider.viewId}.focus`);
      return; // sent once the webview says it's ready
    }
    this.view.show?.(true);
    await this.send(text, context);
  }

  newChat(): void {
    this.stop();
    this.messages = [];
    this.rateable.clear();
    this.lastSentCode = undefined;
    this.post({ type: "reset" });
  }

  stop(): void {
    this.abort?.abort();
  }

  private post(message: unknown): void {
    void this.view?.webview.postMessage(message);
  }

  private async onMessage(message: FromWebview): Promise<void> {
    switch (message.type) {
      case "ready":
        this.post({ type: "history", messages: this.messages });
        if (this.queued) {
          const { text, context } = this.queued;
          this.queued = undefined;
          await this.send(text, context);
        }
        break;
      case "send":
        await this.send(message.text, message.includeContext ? this.host.currentContext() : undefined);
        break;
      case "stop":
        this.stop();
        break;
      case "newChat":
        this.newChat();
        break;
      case "copy":
        await vscode.env.clipboard.writeText(message.code);
        vscode.window.setStatusBarMessage("Copied Xenon's code", 2000);
        break;
      case "insert":
      case "replace":
        await this.applyCode(message.type, message.code);
        break;
      case "rate":
        await this.rate(message.messageId, message.rating);
        break;
      case "correct":
        await this.rate(message.messageId, "bad", message.correction);
        break;
    }
  }

  private async rate(messageId: string, rating: "good" | "bad", correction?: string): Promise<void> {
    const entry = this.rateable.get(messageId);
    if (!entry) return;
    try {
      const connection = resolveConnection(this.host.settings());
      await sendFeedback(connection, { messageId, rating, correction, history: entry.history, reply: entry.reply });
      this.post({ type: "rated", messageId, rating });
    } catch (err) {
      vscode.window.showWarningMessage(`Couldn't save feedback: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private async applyCode(mode: "insert" | "replace", code: string): Promise<void> {
    const target = this.host.lastTarget();
    if (!target) {
      vscode.window.showWarningMessage("Open a file first, then try again.");
      return;
    }
    const { editor, range } = target;
    await vscode.window.showTextDocument(editor.document, editor.viewColumn);
    await editor.edit((edit) => {
      if (mode === "replace" && !range.isEmpty) edit.replace(range, code);
      else edit.insert(editor.selection.active, code);
    });
  }

  private async send(text: string, context: CodeContext | undefined): Promise<void> {
    const question = text.trim();
    if (!question) return;
    if (this.generating) {
      vscode.window.showInformationMessage("Xenon is still answering. Stop it first, or wait.");
      return;
    }
    // Skip code the model already read in this chat; its state still holds it.
    const key = context ? `${context.file}\n${context.code}` : undefined;
    const attach = context && key !== this.lastSentCode ? context : undefined;
    if (attach) this.lastSentCode = key;

    const content = withContext(question, attach);
    this.messages.push({ role: "user", content });
    this.post({
      type: "user",
      text: question,
      contextLabel: attach ? `${attach.file}:${attach.startLine}${attach.isSelection ? " (selection)" : ""}` : undefined,
    });
    this.post({ type: "start", reading: Boolean(attach) });

    const abort = new AbortController();
    this.abort = abort;
    this.host.setBusy(true);
    let reply = "";
    let finish: Finish = "stop";
    try {
      const settings = this.host.settings();
      const connection = resolveConnection(settings);
      finish = await streamChat(
        connection,
        this.messages,
        settings,
        (piece) => {
          reply += piece;
          this.post({ type: "token", text: piece });
        },
        abort.signal,
      );
      const messageId = reply.trim() ? String(this.nextMessageId++) : undefined;
      if (messageId) this.rateable.set(messageId, { history: [...this.messages], reply: reply.trim() });
      this.post({ type: "done", finish, hasCode: firstCodeBlock(reply) !== undefined, messageId });
    } catch (err) {
      this.post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    } finally {
      this.abort = undefined;
      this.host.setBusy(false);
    }
    if (reply.trim()) {
      this.messages.push({ role: "assistant", content: reply.trim() });
    } else {
      // Nothing came back: forget the question so a retry doesn't send it twice.
      this.messages.pop();
      if (attach) this.lastSentCode = undefined;
    }
  }

  private html(webview: vscode.Webview): string {
    const media = (file: string) => webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, "media", file));
    const nonce = [...Array(32)].map(() => Math.floor(Math.random() * 36).toString(36)).join("");
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="${media("chat.css")}" rel="stylesheet">
<title>Xenon</title>
</head>
<body>
<div id="log" aria-live="polite"></div>
<div id="empty">
  <p><strong>Xenon</strong> runs on your own GPU through the Xenon2 app. Nothing leaves this machine.</p>
  <p>Ask about the code you have open, or select some code and right-click → <em>Xenon</em>.</p>
</div>
<form id="form">
  <textarea id="input" rows="3" placeholder="Ask Xenon about your code…  (Enter to send, Shift+Enter for a new line)"></textarea>
  <div class="row">
    <label><input type="checkbox" id="ctx" checked> Include my code</label>
    <span class="spacer"></span>
    <button type="button" id="stop" hidden>Stop</button>
    <button type="submit" id="send">Send</button>
  </div>
</form>
<script nonce="${nonce}" src="${media("chat.js")}"></script>
</body>
</html>`;
  }
}
