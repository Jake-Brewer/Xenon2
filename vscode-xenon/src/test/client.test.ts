import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { SseParser, discoveryPath, resolveConnection, runFolderAgent, sendFeedback, undoFolderAgent } from "../client";
import { firstCodeBlock, formatOverview, pickCode, withContext } from "../context";

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

test("the workspace overview names the project, lists its files, and adds its notes", () => {
  const text = formatOverview(
    { name: "Xenon2", root: "/home/me/Xenon2" },
    ["app/", "app/src/", "PLAN.md"],
    { file: "README.md", text: "An offline assistant." },
  );
  assert.match(text, /`Xenon2` open, at \/home\/me\/Xenon2/);
  assert.match(text, /app\/src\/\nPLAN\.md\n/);
  assert.match(text, /From `README\.md`:\nAn offline assistant\./);
});

test("a long file list is cut and says how much was left out", () => {
  const entries = Array.from({ length: 100 }, (_, i) => `file${i}.py`);
  const text = formatOverview({ name: "p", root: "/p" }, entries, undefined, 50);
  assert.match(text, /\(and \d+ more\)/);
  assert.ok(!text.includes("file99.py"));
});

test("resolveConnection throws when url is missing even if token is present", () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "xenon-")), "code-api.json");
  fs.writeFileSync(file, JSON.stringify({ token: "tok" }));  // no url
  assert.throws(() => resolveConnection({ serverUrl: "", token: "" }, file), /isn't running/);
});

test("resolveConnection throws when token is missing even if url is present", () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "xenon-")), "code-api.json");
  fs.writeFileSync(file, JSON.stringify({ url: "http://127.0.0.1:47811" }));  // no token
  assert.throws(() => resolveConnection({ serverUrl: "", token: "" }, file), /isn't running/);
});

test("resolveConnection strips trailing slash from url", () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "xenon-")), "code-api.json");
  fs.writeFileSync(file, JSON.stringify({ url: "http://127.0.0.1:47811///", token: "tok" }));
  const { url } = resolveConnection({ serverUrl: "", token: "" }, file);
  assert.equal(url, "http://127.0.0.1:47811");
});

test("sendFeedback throws on non-ok response", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => ({ ok: false, status: 500, text: async () => "oops" }) as Response) as typeof fetch;
  try {
    await assert.rejects(
      () => sendFeedback({ url: "http://h", token: "t" }, { messageId: "1", rating: "bad", history: [], reply: "r" }),
      /500/,
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("sendFeedback sends POST with correct headers and body", async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return { ok: true, text: async () => "" } as Response;
  }) as typeof fetch;
  try {
    await sendFeedback({ url: "http://h", token: "mytoken" }, { messageId: "7", rating: "good", history: [], reply: "r" });
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.equal(calls[0].init.method, "POST");
  const headers = calls[0].init.headers as Record<string, string>;
  assert.equal(headers["Content-Type"], "application/json");
  assert.equal(headers["Authorization"], "Bearer mytoken");
});

test("undoFolderAgent sends a POST to the undo endpoint with folder and snapshot", async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return { ok: true, text: async () => "" } as Response;
  }) as typeof fetch;
  try {
    await undoFolderAgent({ url: "http://h", token: "t" }, "/my/folder", "abc123");
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://h/v1/folder-agent/undo");
  assert.equal(calls[0].init.method, "POST");
  const body = JSON.parse(calls[0].init.body as string);
  assert.equal(body.folder, "/my/folder");
  assert.equal(body.snapshot_commit, "abc123");
});

test("undoFolderAgent throws on non-ok response", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => ({ ok: false, status: 503, text: async () => "gone" }) as Response) as typeof fetch;
  try {
    await assert.rejects(
      () => undoFolderAgent({ url: "http://h", token: "t" }, "/f", "sha"),
      /503/,
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("SSE filter removes non-data lines and strips leading space", () => {
  const parser = new SseParser();
  // A block with a comment line and a data line
  assert.deepEqual(parser.push(': comment\ndata: {"x":1}\n\n'), ['{"x":1}']);
});

test("SSE empty data block is not emitted", () => {
  const parser = new SseParser();
  // A block with only a non-data line produces no events
  assert.deepEqual(parser.push(': heartbeat\n\n'), []);
});

test("pickCode selection with whitespace-only text falls through to cursor window", () => {
  const lines = ["line0", "line1", "line2"];
  const result = pickCode(lines, { startLine: 0, endLine: 0, text: "   \n\t  " }, 1, 0);
  assert.equal(result.isSelection, false);
});

test("pickCode respects the maxChars limit exactly (> not >=)", () => {
  // If text.length === maxChars, no slice (> not >=)
  const sel = { startLine: 0, endLine: 0, text: "abc" };
  const exact = pickCode(["abc"], sel, 0, 3);
  assert.equal(exact.code, "abc");  // not sliced: 3 > 3 is false
  const over = pickCode(["abcd"], { startLine: 0, endLine: 0, text: "abcd" }, 0, 3);
  assert.equal(over.code, "abc");   // sliced: 4 > 3 is true
});

test("formatOverview exact boundary: an entry that fits exactly is included", () => {
  // maxListChars=10, entry "abc\n" costs 4; two entries: 4+4=8 <= 10, third would be 12 > 10
  const entries = ["abc", "def", "ghi"];
  const text = formatOverview({ name: "p", root: "/p" }, entries, undefined, 8);
  assert.ok(text.includes("abc\n"));
  assert.ok(text.includes("def\n"));
  assert.ok(!text.includes("ghi\n"));
  assert.match(text, /\(and 1 more\)/);
});

test("formatOverview notes trimmed: whitespace-only notes are omitted", () => {
  const text = formatOverview({ name: "p", root: "/p" }, [], { file: "README.md", text: "   \n  " });
  assert.ok(!text.includes("README.md"));
});

test("formatOverview notes are cut at maxNoteChars with marker", () => {
  const longNote = "x".repeat(200);
  const text = formatOverview({ name: "p", root: "/p" }, [], { file: "notes.md", text: longNote }, 2500, 100);
  assert.match(text, /cut short/);
  assert.ok(text.includes("x".repeat(100)));
  assert.ok(!text.includes("x".repeat(101)));
});
