// Xenon for VS Code: a sidebar chat, editor commands, a status bar item, and an @xenon chat
// participant, all backed by the Xenon2 app's local coding API.

import * as vscode from "vscode";
import { ChatViewProvider, Host } from "./chatView";
import { ChatMessage, health, resolveConnection, streamChat } from "./client";
import { CodeContext, pickCode, withContext } from "./context";
import { acceptPendingEdit, discardPendingEdit, hasPendingEdit, proposeFileEdit } from "./edit";

const PROMPTS = {
  explain: "Explain what this code does, step by step, and point out anything surprising.",
  fix: "Find the bugs in this code. Explain each one briefly, then show the corrected code in one code block.",
  tests: "Write unit tests for this code, using the usual test framework for this language. Put them in one code block.",
  docs: "Add clear documentation comments to this code. Show the complete updated code in one code block.",
} as const;

let lastEditor: vscode.TextEditor | undefined;
/** The editor and selection the latest question was about; "Replace" writes there. */
let lastTarget: { editor: vscode.TextEditor; range: vscode.Range } | undefined;

function settings() {
  const config = vscode.workspace.getConfiguration("xenon");
  return {
    serverUrl: config.get<string>("serverUrl", ""),
    token: config.get<string>("token", ""),
    maxTokens: config.get<number>("maxTokens", 800),
    temperature: config.get<number>("temperature", 0.3),
    maxContextChars: config.get<number>("maxContextChars", 3000),
  };
}

function contextFrom(editor: vscode.TextEditor | undefined): CodeContext | undefined {
  if (!editor) return undefined;
  const document = editor.document;
  const selection = editor.selection;
  const lines = Array.from({ length: document.lineCount }, (_, i) => document.lineAt(i).text);
  const picked = pickCode(
    lines,
    selection.isEmpty ? undefined : { startLine: selection.start.line, endLine: selection.end.line, text: document.getText(selection) },
    selection.active.line,
    settings().maxContextChars,
  );
  lastTarget = { editor, range: selection.isEmpty ? new vscode.Range(selection.active, selection.active) : selection };
  return {
    ...picked,
    file: vscode.workspace.asRelativePath(document.uri),
    language: document.languageId,
  };
}

export function activate(extension: vscode.ExtensionContext): void {
  lastEditor = vscode.window.activeTextEditor;
  extension.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      // Focus moving to the chat leaves no active editor; keep the one the user came from.
      if (editor) lastEditor = editor;
    }),
  );

  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.command = "xenon.chat.focus";
  status.show();
  extension.subscriptions.push(status);

  let busy = false;
  let online: string | undefined;
  const showStatus = () => {
    if (busy) {
      status.text = "$(loading~spin) Xenon";
      status.tooltip = "Xenon is answering";
    } else if (online) {
      status.text = "$(sparkle) Xenon";
      status.tooltip = `Xenon is ready (${online}), running locally`;
    } else {
      status.text = "$(circle-slash) Xenon";
      status.tooltip = "Xenon2 app not running. Start it to use Xenon.";
    }
  };
  const checkHealth = async () => {
    try {
      const reply = await health(resolveConnection(settings()), AbortSignal.timeout(3000));
      online = reply.model;
    } catch {
      online = undefined;
    }
    showStatus();
  };

  const host: Host = {
    settings,
    currentContext: () => contextFrom(lastEditor),
    lastTarget: () => lastTarget,
    setBusy: (value) => {
      busy = value;
      void vscode.commands.executeCommand("setContext", "xenon.generating", value);
      if (!value) void checkHealth();
      showStatus();
    },
  };

  const chat = new ChatViewProvider(extension.extensionUri, host);
  extension.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewId, chat, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  const askAbout = (prompt: string) => () => {
    const editor = vscode.window.activeTextEditor ?? lastEditor;
    if (!editor) {
      vscode.window.showWarningMessage("Open a file first, then ask Xenon about it.");
      return;
    }
    void chat.ask(prompt, contextFrom(editor));
  };

  extension.subscriptions.push(
    vscode.commands.registerCommand("xenon.ask", async () => {
      const editor = vscode.window.activeTextEditor ?? lastEditor;
      const question = await vscode.window.showInputBox({
        prompt: editor ? "Ask Xenon about this code" : "Ask Xenon",
        placeHolder: "e.g. Why does this loop never end?",
      });
      if (question) await chat.ask(question, contextFrom(editor));
    }),
    vscode.commands.registerCommand("xenon.explain", askAbout(PROMPTS.explain)),
    vscode.commands.registerCommand("xenon.fix", askAbout(PROMPTS.fix)),
    vscode.commands.registerCommand("xenon.tests", askAbout(PROMPTS.tests)),
    vscode.commands.registerCommand("xenon.docs", askAbout(PROMPTS.docs)),
    vscode.commands.registerCommand("xenon.newChat", () => chat.newChat()),
    vscode.commands.registerCommand("xenon.stop", () => chat.stop()),
    vscode.commands.registerCommand("xenon.reconnect", async () => {
      await checkHealth();
      vscode.window.showInformationMessage(online ? `Connected to Xenon (${online}).` : "The Xenon2 app isn't running.");
    }),
    vscode.commands.registerCommand("xenon.editFile", async () => {
      const editor = vscode.window.activeTextEditor ?? lastEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Open a file first, then ask Xenon to edit it.");
        return;
      }
      if (hasPendingEdit()) {
        vscode.window.showWarningMessage("Review the pending Xenon edit first (Apply or Discard), then try again.");
        return;
      }
      const instruction = await vscode.window.showInputBox({
        prompt: `Tell Xenon how to improve ${vscode.workspace.asRelativePath(editor.document.uri)}`,
        placeHolder: "e.g. Make this use type hints and fix the off-by-one in the loop.",
      });
      if (!instruction) return;

      const document = editor.document;
      const abort = new AbortController();
      host.setBusy(true);
      void vscode.commands.executeCommand("setContext", "xenon.editPending", true);
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Xenon is rewriting the file…", cancellable: true },
        async (_progress, token) => {
          token.onCancellationRequested(() => abort.abort());
          try {
            const connection = resolveConnection(settings());
            await proposeFileEdit(connection, settings(), document, instruction, () => {}, abort.signal);
            const action = await vscode.window.showInformationMessage(
              "Xenon's suggestion is open as a diff. Review it, then Apply or Discard.",
              "Apply",
              "Discard",
            );
            if (action === "Apply") await acceptPendingEdit();
            else if (action === "Discard") discardPendingEdit();
          } catch (err) {
            vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
          }
        },
      );
      void vscode.commands.executeCommand("setContext", "xenon.editPending", false);
      host.setBusy(false);
    }),
    vscode.commands.registerCommand("xenon.applyEdit", () => acceptPendingEdit()),
    vscode.commands.registerCommand("xenon.discardEdit", () => discardPendingEdit()),
  );

  registerParticipant(extension);

  void checkHealth();
  const timer = setInterval(() => {
    if (!busy) void checkHealth();
  }, 20000);
  extension.subscriptions.push({ dispose: () => clearInterval(timer) });
}

/** `@xenon` in VS Code's own Chat view, where that view and API exist. */
function registerParticipant(extension: vscode.ExtensionContext): void {
  const chatApi = (vscode as { chat?: typeof vscode.chat }).chat;
  if (!chatApi?.createChatParticipant) return;

  const handler: vscode.ChatRequestHandler = async (request, context, stream, token) => {
    const messages: ChatMessage[] = [];
    for (const turn of context.history) {
      if (turn instanceof vscode.ChatRequestTurn) {
        messages.push({ role: "user", content: turn.prompt });
      } else if (turn instanceof vscode.ChatResponseTurn) {
        const text = turn.response
          .map((part) => (part instanceof vscode.ChatResponseMarkdownPart ? part.value.value : ""))
          .join("");
        if (text.trim()) messages.push({ role: "assistant", content: text });
      }
    }
    const editor = vscode.window.activeTextEditor ?? lastEditor;
    const withCode = editor && !editor.selection.isEmpty ? contextFrom(editor) : undefined;
    messages.push({ role: "user", content: withContext(request.prompt, withCode) });

    const abort = new AbortController();
    token.onCancellationRequested(() => abort.abort());
    try {
      stream.progress(withCode ? "Reading your selection…" : "Thinking…");
      const finish = await streamChat(resolveConnection(settings()), messages, settings(), (text) => stream.markdown(text), abort.signal);
      if (finish === "length") stream.markdown("\n\n*(Stopped at the reply length limit.)*");
    } catch (err) {
      stream.markdown(`**Xenon:** ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const participant = chatApi.createChatParticipant("xenon.participant", handler);
  participant.iconPath = vscode.Uri.joinPath(extension.extensionUri, "media", "xenon.svg");
  extension.subscriptions.push(participant);
}

export function deactivate(): void {}
