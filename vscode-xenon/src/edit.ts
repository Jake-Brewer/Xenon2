// Whole-file edits from an English instruction: ask Xenon to rewrite a file, show the result as a
// diff against the original, and only touch disk once the user accepts it.

import * as os from "node:os";
import * as vscode from "vscode";
import { ChatMessage, Connection, streamChat } from "./client";
import { firstCodeBlock } from "./context";

export interface EditSettings {
  maxTokens: number;
  temperature: number;
}

const INSTRUCTION = (userInstruction: string, file: string, language: string, code: string): string =>
  `This is the complete contents of \`${file}\`:\n\`\`\`${language}\n${code}\n\`\`\`\n` +
  `${userInstruction}\n` +
  `Reply with the complete updated file in one fenced code block, start to finish. Do not omit any part with ` +
  `a comment like "rest unchanged" -- the whole file must be in the block, since it replaces the file as is.`;

/** One pending edit: the diff view it opened, and what to do when the user accepts it. */
class PendingEdit {
  private disposed = false;
  constructor(
    private readonly document: vscode.TextDocument,
    private readonly scratch: vscode.Uri,
    private readonly newText: string,
    private readonly onSettled: () => void,
  ) {}

  async apply(): Promise<void> {
    if (this.disposed) return;
    const edit = new vscode.WorkspaceEdit();
    const fullRange = new vscode.Range(0, 0, this.document.lineCount, 0);
    edit.replace(this.document.uri, fullRange, this.newText);
    await vscode.workspace.applyEdit(edit);
    await this.document.save();
    this.settle();
  }

  discard(): void {
    this.settle();
  }

  private settle(): void {
    if (this.disposed) return;
    this.disposed = true;
    void vscode.workspace.fs.delete(this.scratch).then(undefined, () => undefined);
    this.onSettled();
  }
}

let active: PendingEdit | undefined;

/** Asks Xenon to rewrite `document` per `instruction`, then opens a diff view to review it. */
export async function proposeFileEdit(
  connection: Connection,
  settings: EditSettings,
  document: vscode.TextDocument,
  instruction: string,
  onProgress: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (active) {
    throw new Error("Finish reviewing the last edit first (accept or discard it), then try again.");
  }
  const original = document.getText();
  const messages: ChatMessage[] = [
    { role: "user", content: INSTRUCTION(instruction, vscode.workspace.asRelativePath(document.uri), document.languageId, original) },
  ];
  let reply = "";
  await streamChat(connection, messages, settings, (piece) => {
    reply += piece;
    onProgress(reply);
  }, signal);

  const code = firstCodeBlock(reply);
  if (!code || !code.trim()) {
    throw new Error("Xenon didn't return an updated file. Nothing was changed.");
  }
  if (code.trim() === original.trim()) {
    throw new Error("Xenon's answer didn't change the file.");
  }

  const ext = document.uri.path.split(".").pop() ?? "txt";
  const scratch = vscode.Uri.joinPath(vscode.Uri.file(os.tmpdir()), `xenon-edit-${Date.now()}.${ext}`);
  await vscode.workspace.fs.writeFile(scratch, Buffer.from(code, "utf8"));

  await new Promise<void>((resolve) => {
    active = new PendingEdit(document, scratch, code, () => {
      active = undefined;
      resolve();
    });
    void vscode.commands.executeCommand(
      "vscode.diff",
      document.uri,
      scratch,
      `${vscode.workspace.asRelativePath(document.uri)} ↔ Xenon's suggestion`,
    );
  });
}

export function hasPendingEdit(): boolean {
  return active !== undefined;
}

export async function acceptPendingEdit(): Promise<void> {
  if (!active) {
    vscode.window.showInformationMessage("No pending Xenon edit to apply.");
    return;
  }
  await active.apply();
}

export function discardPendingEdit(): void {
  if (!active) {
    vscode.window.showInformationMessage("No pending Xenon edit to discard.");
    return;
  }
  active.discard();
}
