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
 * the cursor as fit. `lines` is the whole document; positions are 0-based.
 */
export function pickCode(
  lines: string[],
  selection: { startLine: number; endLine: number; text: string } | undefined,
  cursorLine: number,
  maxChars: number,
): { code: string; startLine: number; isSelection: boolean } {
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
