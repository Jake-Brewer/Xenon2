import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { SseParser, discoveryPath, resolveConnection, runFolderAgent, sendFeedback } from "../client";
import { firstCodeBlock, pickCode, withContext } from "../context";

test("SSE events split across chunks are joined", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push('data: {"a"'), []);
  assert.deepEqual(parser.push(':1}\n\ndata: [DO'), ['{"a":1}']);
  assert.deepEqual(parser.push("NE]\r\n\r\n"), ["[DONE]"]);
});

test("discovery file lives in Tauri's app data dir", () => {
  assert.equal(
    discoveryPath("win32", { APPDATA: "C:\\Users\\a\\AppData\\Roaming" }, "C:\\Users\\a"),
    path.join("C:\\Users\\a\\AppData\\Roaming", "com.xenon2.app", "code-api.json"),
  );
  assert.equal(discoveryPath("linux", {}, "/home/a"), path.join("/home/a", ".local", "share", "com.xenon2.app", "code-api.json"));
});

test("settings override the discovery file, which fills the gaps", () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "xenon-")), "code-api.json");
  fs.writeFileSync(file, JSON.stringify({ url: "http://127.0.0.1:47811", token: "t1" }));
  assert.deepEqual(resolveConnection({ serverUrl: "", token: "" }, file), { url: "http://127.0.0.1:47811", token: "t1" });
  assert.deepEqual(resolveConnection({ serverUrl: "http://h:1/", token: "" }, file), { url: "http://h:1", token: "t1" });
  assert.throws(() => resolveConnection({ serverUrl: "", token: "" }, file + ".missing"), /isn't running/);
});

test("a selection is sent as is, cut to the limit", () => {
  const picked = pickCode(["a", "b"], { startLine: 4, endLine: 5, text: "x".repeat(50) }, 0, 10);
  assert.deepEqual(picked, { code: "x".repeat(10), startLine: 5, isSelection: true });
});

test("without a selection, whole lines around the cursor fit the limit", () => {
  const lines = Array.from({ length: 100 }, (_, i) => `line ${i}`); // 6-7 chars + newline each
  const picked = pickCode(lines, undefined, 50, 40);
  assert.equal(picked.isSelection, false);
  assert.ok(picked.code.length <= 40);
  assert.ok(picked.code.includes("line 50"));
  assert.ok(picked.startLine <= 51 && picked.startLine >= 48);
  // Near the top it grows downwards instead.
  const top = pickCode(lines, undefined, 0, 40);
  assert.equal(top.startLine, 1);
  assert.ok(top.code.startsWith("line 0\nline 1"));
});

test("a limit of 0 sends everything", () => {
  const lines = Array.from({ length: 5000 }, (_, i) => `line ${i}`);
  const picked = pickCode(lines, undefined, 2500, 0);
  assert.equal(picked.code, lines.join("\n"));
  assert.equal(picked.startLine, 1);
});

test("questions carry their code in a fenced block", () => {
  const text = withContext("Why?", { file: "a.py", language: "python", code: "x = 1\n\n", isSelection: true, startLine: 3 });
  assert.equal(text, "Selected code from `a.py` (line 3):\n```python\nx = 1\n```\nWhy?");
  assert.equal(withContext("Why?", undefined), "Why?");
});

test("the first code block of a reply is found", () => {
  assert.equal(firstCodeBlock("Fixed:\n```python\nx = 2\n```\nDone."), "x = 2");
  assert.equal(firstCodeBlock("no code"), undefined);
});

test("feedback posts the rating and correction as JSON", async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return { ok: true, text: async () => "" } as Response;
  }) as typeof fetch;
  try {
    await sendFeedback(
      { url: "http://127.0.0.1:47811", token: "t1" },
      { messageId: "5", rating: "bad", correction: "use list(reversed(xs))", history: [{ role: "user", content: "reverse a list" }], reply: "xs[::-1]" },
    );
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:47811/v1/feedback");
  const body = JSON.parse(calls[0].init.body as string);
  assert.equal(body.message_id, "5");
  assert.equal(body.rating, "bad");
  assert.equal(body.correction, "use list(reversed(xs))");
});

test("the folder agent is started, polled for progress, then its report returned", async () => {
  const report = { stopped_reason: "finished", edits: [], runs: [], narration: ["Edited a.py: fix", "Stopped: done"], snapshot_commit: "abc" };
  const replies = [
    { id: "run-1" },
    { state: "running", narration: ["Edited a.py: fix"], report: null, error: null },
    { state: "done", narration: report.narration, report, error: null },
  ];
  const routes: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    routes.push(`${init.method} ${url}`);
    const body = replies.shift();
    return { ok: true, json: async () => body, text: async () => "" } as Response;
  }) as typeof fetch;
  const steps: string[] = [];
  try {
    const result = await runFolderAgent(
      { url: "http://h", token: "t" },
      { folder: "/f", goal: "g", files: ["a.py"] },
      (line) => steps.push(line),
      undefined,
      0,
    );
    assert.equal(result.snapshot_commit, "abc");
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.deepEqual(routes, ["POST http://h/v1/folder-agent", "GET http://h/v1/folder-agent/run-1", "GET http://h/v1/folder-agent/run-1"]);
  assert.deepEqual(steps, ["Edited a.py: fix", "Stopped: done"]);
});
