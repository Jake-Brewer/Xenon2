// Webview side of the Xenon chat. Renders a small, safe subset of Markdown: fenced code blocks
// (with Copy / Insert / Replace buttons), inline code, bold and line breaks. All text is escaped
// before any markup is added.
(function () {
  const vscode = acquireVsCodeApi();
  const log = document.getElementById("log");
  const empty = document.getElementById("empty");
  const form = document.getElementById("form");
  const input = document.getElementById("input");
  const ctx = document.getElementById("ctx");
  const stopButton = document.getElementById("stop");
  const sendButton = document.getElementById("send");

  let current = null; // { el, text } for the reply being streamed
  let renderQueued = false;

  const saved = vscode.getState() || {};
  if (saved.includeContext === false) ctx.checked = false;
  ctx.addEventListener("change", () => vscode.setState({ ...vscode.getState(), includeContext: ctx.checked }));

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function render(markdown, streaming) {
    const parts = markdown.split(/```/);
    let html = "";
    parts.forEach((part, i) => {
      if (i % 2 === 0) {
        if (part.trim()) html += `<div class="text">${inline(part.replace(/^\n+|\n+$/g, ""))}</div>`;
        return;
      }
      const newline = part.indexOf("\n");
      const lang = newline === -1 ? "" : part.slice(0, newline).trim();
      const code = newline === -1 ? part : part.slice(newline + 1).replace(/\n$/, "");
      const open = streaming && i === parts.length - 1;
      html += `<div class="code"><div class="code-bar"><span>${escapeHtml(lang)}</span>`;
      if (!open) {
        html +=
          `<button data-act="copy">Copy</button>` +
          `<button data-act="insert">Insert</button>` +
          `<button data-act="replace" title="Replace the code this question was about">Replace</button>`;
      }
      html += `</div><pre><code>${escapeHtml(code)}</code></pre></div>`;
    });
    return html;
  }

  function scrollDown() {
    window.scrollTo(0, document.body.scrollHeight);
  }

  function add(kind, html) {
    empty.hidden = true;
    const el = document.createElement("div");
    el.className = `msg ${kind}`;
    el.innerHTML = html;
    log.appendChild(el);
    scrollDown();
    return el;
  }

  function setBusy(busy) {
    stopButton.hidden = !busy;
    sendButton.disabled = busy;
  }

  log.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-act]");
    if (!button) return;
    const act = button.dataset.act;
    if (act === "good" || act === "bad") {
      const messageId = button.closest(".msg").dataset.messageId;
      if (act === "bad") {
        const bar = button.closest(".rate-bar");
        bar.outerHTML = `<div class="rate-bar correcting">
          <textarea class="correction" rows="2" placeholder="What should Xenon have said? (Enter to save)"></textarea>
        </div>`;
        return;
      }
      vscode.postMessage({ type: "rate", messageId, rating: act });
      return;
    }
    const code = button.closest(".code").querySelector("code").textContent;
    vscode.postMessage({ type: act, code });
  });

  log.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || !event.target.classList.contains("correction")) return;
    event.preventDefault();
    const correction = event.target.value.trim();
    if (!correction) return;
    const messageId = event.target.closest(".msg").dataset.messageId;
    vscode.postMessage({ type: "correct", messageId, correction });
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || sendButton.disabled) return;
    input.value = "";
    vscode.postMessage({ type: "send", text, includeContext: ctx.checked });
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  stopButton.addEventListener("click", () => vscode.postMessage({ type: "stop" }));

  window.addEventListener("message", (event) => {
    const message = event.data;
    switch (message.type) {
      case "history":
        log.innerHTML = "";
        empty.hidden = message.messages.length > 0;
        for (const m of message.messages) {
          add(m.role === "user" ? "user" : "bot", m.role === "user" ? inline(m.content) : render(m.content, false));
        }
        break;
      case "reset":
        log.innerHTML = "";
        empty.hidden = false;
        current = null;
        setBusy(false);
        break;
      case "user": {
        const label = message.contextLabel ? `<div class="ctx-label">with ${escapeHtml(message.contextLabel)}</div>` : "";
        add("user", inline(message.text) + label);
        break;
      }
      case "start":
        setBusy(true);
        current = {
          el: add("bot", `<div class="thinking">${message.reading ? "Reading your code…" : "Thinking…"}</div>`),
          text: "",
        };
        break;
      case "token":
        if (!current) break;
        current.text += message.text;
        if (!renderQueued) {
          renderQueued = true;
          requestAnimationFrame(() => {
            renderQueued = false;
            if (current) {
              current.el.innerHTML = render(current.text, true);
              scrollDown();
            }
          });
        }
        break;
      case "done":
        if (current) {
          current.el.innerHTML = render(current.text, false) || `<div class="note">(no reply)</div>`;
          if (message.finish === "length") current.el.insertAdjacentHTML("beforeend", `<div class="note">Stopped at the reply length limit.</div>`);
          if (message.finish === "cancelled") current.el.insertAdjacentHTML("beforeend", `<div class="note">Stopped.</div>`);
          if (message.messageId) {
            current.el.dataset.messageId = message.messageId;
            current.el.insertAdjacentHTML(
              "beforeend",
              `<div class="rate-bar">
                <button data-act="good" title="Good reply">👍</button>
                <button data-act="bad" title="Wrong — tell Xenon the fix">👎</button>
              </div>`,
            );
          }
        }
        current = null;
        setBusy(false);
        scrollDown();
        break;
      case "rated": {
        const el = log.querySelector(`.msg[data-message-id="${message.messageId}"]`);
        const bar = el && el.querySelector(".rate-bar");
        if (bar) bar.outerHTML = `<div class="rate-bar done">${message.rating === "good" ? "Thanks — noted." : "Got it — Xenon will use that fix."}</div>`;
        break;
      }
      case "error":
        if (current) current.el.remove();
        current = null;
        add("error", inline(message.message));
        setBusy(false);
        break;
    }
  });

  vscode.postMessage({ type: "ready" });
  input.focus();
})();
