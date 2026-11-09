// Drives the folder agent (app/src-tauri/src/folder_agent.rs): picks a goal and a set of files,
// runs the autonomous edit/run/fix loop, and shows the result in a webview panel with an Undo
// button wired to the run's git snapshot.

import * as path from "node:path";
import * as vscode from "vscode";
import { Connection, FolderAgentReport, runFolderAgent, undoFolderAgent } from "./client";

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch] as string);
}

function renderReport(report: FolderAgentReport, canUndo: boolean): string {
  const narration = report.narration.map((line) => `<li>${escapeHtml(line)}</li>`).join("");
  const edits = report.edits
    .map(
      (edit) => `
      <section>
        <h3>${escapeHtml(edit.path)}</h3>
        <details><summary>Before</summary><pre>${escapeHtml(edit.before)}</pre></details>
        <details open><summary>After</summary><pre>${escapeHtml(edit.after)}</pre></details>
      </section>`,
    )
    .join("");
  const runs = report.runs
    .map(
      (run) => `
      <section>
        <h3>${escapeHtml(run.command)} <span class="${run.exit_code === 0 ? "pass" : "fail"}">${run.exit_code === 0 ? "passed" : "failed"} (exit ${run.exit_code ?? "?"})</span></h3>
        ${run.stdout.trim() ? `<pre>${escapeHtml(run.stdout)}</pre>` : ""}
        ${run.stderr.trim() ? `<pre class="stderr">${escapeHtml(run.stderr)}</pre>` : ""}
      </section>`,
    )
    .join("");
  return `<!doctype html>
  <html><head><meta charset="utf-8"><style>
    body { font-family: var(--vscode-font-family); padding: 1rem; color: var(--vscode-foreground); }
    h2 { margin-top: 0; }
    pre { background: var(--vscode-textCodeBlock-background); padding: 0.5rem; overflow-x: auto; white-space: pre-wrap; }
    .pass { color: var(--vscode-testing-iconPassed, #3fb950); }
    .fail { color: var(--vscode-testing-iconFailed, #f85149); }
    .stderr { color: var(--vscode-errorForeground); }
    button { margin-top: 1rem; }
  </style></head>
  <body>
    <h2>Xenon's folder agent: ${escapeHtml(report.stopped_reason)}</h2>
    <h3>What it did</h3>
    <ul>${narration || "<li>No changes.</li>"}</ul>
    ${edits}
    ${runs}
    ${canUndo ? '<button id="undo">Undo all of this</button>' : ""}
    <script>
      const vscode = acquireVsCodeApi();
      document.getElementById("undo")?.addEventListener("click", () => vscode.postMessage({ type: "undo" }));
    </script>
  </body></html>`;
}

export async function runFolderAgentCommand(connection: () => Connection): Promise<void> {
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    vscode.window.showWarningMessage("Open a folder first, then ask Xenon to work in it.");
    return;
  }
  const relativeFiles = await vscode.window.showInputBox({
    prompt: "Which Python files should Xenon work on? (comma-separated, relative to the workspace root)",
    placeHolder: "e.g. src/util.py, tests/test_util.py",
  });
  if (!relativeFiles) return;
  const files = relativeFiles
    .split(",")
    .map((f) => f.trim())
    .filter(Boolean);
  if (files.length === 0) return;

  const goal = await vscode.window.showInputBox({
    prompt: "What should Xenon try to do?",
    placeHolder: "e.g. Add type hints and make the failing test pass.",
  });
  if (!goal) return;

  const folder = workspaceFolder.uri.fsPath;
  const abort = new AbortController();

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Xenon is working in the folder…",
      cancellable: true,
    },
    async (progress, token) => {
      token.onCancellationRequested(() => abort.abort());
      try {
        const report = await runFolderAgent(
          connection(),
          { folder, goal, files },
          (line) => progress.report({ message: line }),
          abort.signal,
        );
        showReport(report, connection(), folder);
      } catch (err) {
        vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
      }
    },
  );
}

function showReport(report: FolderAgentReport, connection: Connection, folder: string): void {
  const panel = vscode.window.createWebviewPanel(
    "xenonFolderAgentReport",
    `Xenon: ${path.basename(folder)} report`,
    vscode.ViewColumn.Beside,
    { enableScripts: true },
  );
  panel.webview.html = renderReport(report, report.snapshot_commit !== null);
  panel.webview.onDidReceiveMessage(async (message) => {
    if (message.type !== "undo" || !report.snapshot_commit) return;
    try {
      await undoFolderAgent(connection, folder, report.snapshot_commit);
      vscode.window.showInformationMessage("Reverted. The folder is back to how it was before this run.");
      panel.dispose();
    } catch (err) {
      vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
    }
  });
}
