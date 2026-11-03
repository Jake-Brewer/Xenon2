# Xenon for VS Code

Offline coding help from the Xenon2 app. The model runs on your own GPU through the app; no code
leaves the machine.

## Use it

1. Start the Xenon2 app (`npm run tauri dev` in `app/`). It serves a coding API on
   `127.0.0.1:47811` and writes the address and token to `code-api.json` in its app data folder
   (`%APPDATA%\com.xenon2.app` on Windows). The extension finds it there.
2. Open the **Xenon** view in the activity bar, or press `Ctrl+Alt+X` to ask about the code you have
   open.
3. Select code and right-click → **Xenon** for *Explain*, *Find and Fix Bugs*, *Write Tests* or
   *Add Documentation*.

Code blocks in replies have **Copy**, **Insert** (at the cursor) and **Replace** (the code the
question was about) buttons. The status bar item shows whether the app is running.

Type `@xenon` in VS Code's own Chat view to use Xenon there too, where that view is available.

## What to expect

- **Reading your code is the slow part.** On the RTX 2060 the model reads about 35 tokens a
  second, so 3000 characters of code take around 25 seconds before the first word. Replies then
  stream at about 12 tokens a second. Select just the code you're asking about to keep it quick.
- **Follow-ups are faster.** The app keeps what the model has read, and the extension doesn't
  resend code it already sent in the same chat. Start a **New Chat** for a new topic.
- **The chat window and VS Code take turns.** One model serves both, so a question waits while
  the other is answering.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `xenon.maxContextChars` | 3000 | Most characters of code sent with a question. |
| `xenon.maxTokens` | 800 | Longest reply. |
| `xenon.temperature` | 0.3 | Lower is more predictable. |
| `xenon.serverUrl`, `xenon.token` | empty | Set these only to reach an app that isn't on this machine's default path. |

## Other tools

The API is OpenAI-compatible (`/v1/models`, `/v1/chat/completions` with streaming), so other
editors and extensions can use it too. For [Continue](https://continue.dev), add a model with
provider `openai`, `apiBase: http://127.0.0.1:47811/v1`, model `xenon`, and the token from
`code-api.json` as `apiKey`.

## Develop

```
npm install
npm test          # compile and run the unit tests
npm run package   # build vscode-xenon-<version>.vsix
code --install-extension vscode-xenon-0.1.0.vsix
```
