import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { SseParser, discoveryPath, resolveConnection } from "../client";
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
