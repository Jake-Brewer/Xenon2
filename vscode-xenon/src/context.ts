// Turns what's open in the editor into the code block sent with a question.

export interface CodeContext {
  /** Workspace-relative path shown to the model. */
  file: string;
  language: string;
  code: string;
  /** True when this is the user's selection, false for lines around the cursor. */
  isSelection: boolean;
  /** 1-based first line of `code` in the file. */
  startLine: number;
}

/**
 * Picks the lines to send: the selection (cut to `maxChars`), or else as many whole lines around
 * the cursor as fit. `maxChars` 0 means no cap. `lines` is the whole document; positions are
 * 0-based.
 */
export function pickCode(
  lines: string[],
  selection: { startLine: number; endLine: number; text: string } | undefined,
  cursorLine: number,
  maxChars: number,
): { code: string; startLine: number; isSelection: boolean } {
  if (maxChars <= 0) maxChars = Infinity;
  if (selection && selection.text.trim()) {
    const code = selection.text.length > maxChars ? selection.text.slice(0, maxChars) : selection.text;
    return { code, startLine: selection.startLine + 1, isSelection: true };
  }
  let first = Math.min(Math.max(cursorLine, 0), Math.max(lines.length - 1, 0));
  let last = first;
  let size = (lines[first] ?? "").length + 1;
  // Grow outwards, a line above then a line below, so the cursor stays near the middle.
  for (let grew = true; grew; ) {
    grew = false;
    for (const side of ["up", "down"] as const) {
      const next = side === "up" ? first - 1 : last + 1;
      if (next < 0 || next >= lines.length) continue;
      const cost = lines[next].length + 1;
      if (size + cost > maxChars) continue;
      size += cost;
      if (side === "up") first = next;
      else last = next;
      grew = true;
    }
  }
  return { code: lines.slice(first, last + 1).join("\n"), startLine: first + 1, isSelection: false };
}

/** The question text with its code, in the form the chat prompt expects. */
export function withContext(question: string, context: CodeContext | undefined): string {
  if (!context || !context.code.trim()) return question;
  const where = context.isSelection
    ? `Selected code from \`${context.file}\` (line ${context.startLine})`
    : `Code around my cursor in \`${context.file}\` (from line ${context.startLine})`;
  return `${where}:\n\`\`\`${context.language}\n${context.code.replace(/\s+$/, "")}\n\`\`\`\n${question}`;
}

/** First fenced code block in a reply, for "replace selection" actions. */
export function firstCodeBlock(reply: string): string | undefined {
  const match = /```[^\n]*\n([\s\S]*?)```/.exec(reply);
  return match ? match[1].replace(/\n$/, "") : undefined;
}

/** Directories that never help describe a project. */
export const SKIP_DIRS = new Set(["node_modules", ".git", "target", "out", "dist", "build", ".venv", "venv", "__pycache__", ".pytest_cache", ".claude", "models"]);

/**
 * A short description of the open workspace, sent once at the start of each chat (and cached by
 * the server from then on). `entries` are workspace-relative paths, directories ending in "/".
 * `notes` is a hand-written map of the repo when there is one, else the start of the README.
 */
export function formatOverview(
  workspace: { name: string; root: string },
  entries: string[],
  notes: { file: string; text: string } | undefined,
  maxListChars = 2500,
  maxNoteChars = 6000,
): string {
  let listing = "";
  let shown = 0;
  for (const entry of entries) {
    if (listing.length + entry.length + 1 > maxListChars) break;
    listing += entry + "\n";
    shown++;
  }
  const more = entries.length > shown ? `(and ${entries.length - shown} more)\n` : "";
  let text =
    `The programmer has the project \`${workspace.name}\` open, at ${workspace.root}. ` +
    `Questions about "this repo" or "this project" mean this one.\nIts files and folders:\n${listing}${more}`;
  if (notes && notes.text.trim()) {
    const body = notes.text.length > maxNoteChars ? notes.text.slice(0, maxNoteChars) + "\n(cut short)" : notes.text;
    text += `\nFrom \`${notes.file}\`:\n${body.trim()}\n`;
  }
  return text;
}
