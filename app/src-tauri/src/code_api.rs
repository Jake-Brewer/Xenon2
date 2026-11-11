//! Local coding API: lets the VS Code extension in `vscode-xenon/` (and any tool that speaks the
//! OpenAI chat-completions format, such as Continue) use the model this app already has loaded.
//!
//! - Listens on 127.0.0.1 only, on `DEFAULT_PORT` or the next free port after it.
//! - Every route except `GET /health` needs `Authorization: Bearer <token>`. The token is random,
//!   kept in `<app data dir>/code-api.json` with the port, and reused across runs so client
//!   configs keep working. A web page can't send that header to us without a CORS preflight we
//!   never answer, so a browser can't drive the model.
//! - `POST /v1/chat/completions` answers with a coding-assistant prompt (`CODE_PREFIX`), streamed
//!   as server-sent events when `"stream": true`. Generation shares the engine lock with the chat
//!   window, so the two take turns. A client that hangs up stops its generation.
//!
//! The HTTP handling is a deliberately small HTTP/1.1 subset (one request per connection,
//! Content-Length bodies, `Connection: close`). That is all the clients above need, and it lets
//! every server-sent event be flushed the moment its token exists.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};

use crate::example_bank;
use crate::feedback::{self, FeedbackEntry, FeedbackTurn};
use crate::folder_agent;
use crate::inference::{self, EngineState, Finish, SamplingParams};

pub const DEFAULT_PORT: u16 = 47811;
const PORT_ATTEMPTS: u16 = 10;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
// Replies are unbounded by design; the client's Stop (hanging up) ends a long one.
const DEFAULT_MAX_TOKENS: i32 = inference::MAX_REPLY_TOKENS;
const MAX_MAX_TOKENS: i32 = inference::MAX_REPLY_TOKENS;
pub const MODEL_ID: &str = "xenon";
/// Feedback agent tag for the coding API, so its examples never mix with chat's.
pub const AGENT: &str = "code";

/// The coding assistant's fixed preamble. One worked example shows the expected shape: a
/// fenced, language-tagged code block and a short explanation. Kept free of blank lines inside
/// turns, since "\n\n" separates turns in this prompt format.
pub const CODE_PREFIX: &str = "The following is a conversation between a programmer and Xenon, an expert programming \
     assistant running offline inside the programmer's code editor. Xenon writes correct, \
     idiomatic code, puts code in fenced Markdown blocks tagged with the language, explains \
     briefly, and says so when it is unsure.\n\n\
     User: How do I check whether a key is in a Python dict?\n\n\
     Xenon: Use the `in` operator:\n\
     ```python\n\
     if \"name\" in person:\n    print(person[\"name\"])\n\
     ```\n\
     To read a key that may be missing without an error, use `person.get(\"name\")`, which \
     returns `None` when it isn't there.\n\n";

/// How replies are labeled in the prompt, which depends on how the loaded model was trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    /// Speaker label for replies.
    pub label: &'static str,
    /// Text after the label that opens a reply.
    pub opener: &'static str,
}

impl Style {
    /// RWKV World models, with the app's own "Xenon:" turns.
    pub const WORLD: Style = Style { label: "Xenon", opener: "" };
    /// RWKV-7 G1 models, trained on "Assistant:" turns with a <think> block; an empty one
    /// ("fake think") answers directly, as the model card recommends.
    pub const G1: Style = Style { label: "Assistant", opener: " <think></think>" };

    pub fn for_model(model_name: &str) -> Style {
        if model_name.to_ascii_lowercase().contains("g1") {
            Style::G1
        } else {
            Style::WORLD
        }
    }
}

/// What the extension reads to find and authenticate to this server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Discovery {
    pub url: String,
    pub port: u16,
    pub token: String,
    pub model: String,
    pub pid: u32,
}

fn discovery_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("code-api.json"))
}

fn new_token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).expect("the OS random source is available");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Starts the server on a background thread. Failures are logged, never fatal: the chat window
/// works the same without it.
pub fn start(app: AppHandle, engine: EngineState, model_name: String) {
    std::thread::spawn(move || {
        let Some(path) = discovery_path(&app) else {
            eprintln!("xenon2: code API off (no app data dir)");
            return;
        };
        let token = std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Discovery>(&raw).ok())
            .map(|d| d.token)
            .filter(|t| t.len() >= 32)
            .unwrap_or_else(new_token);

        let Some((listener, port)) = (0..PORT_ATTEMPTS).find_map(|i| {
            let port = DEFAULT_PORT + i;
            TcpListener::bind(("127.0.0.1", port)).ok().map(|l| (l, port))
        }) else {
            eprintln!("xenon2: code API off (ports {DEFAULT_PORT}+ are all taken)");
            return;
        };

        let discovery = Discovery {
            url: format!("http://127.0.0.1:{port}"),
            port,
            token: token.clone(),
            model: model_name.clone(),
            pid: std::process::id(),
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&discovery).unwrap_or_default()) {
            eprintln!("xenon2: could not write {}: {e}", path.display());
        }
        eprintln!("xenon2: code API listening on {}", discovery.url);

        for stream in listener.incoming().flatten() {
            let app = app.clone();
            let engine = engine.clone();
            let token = token.clone();
            let model_name = model_name.clone();
            std::thread::spawn(move || {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let closer = stream.try_clone();
                if let Err(e) = handle(stream, &app, &engine, &token, &model_name) {
                    eprintln!("xenon2: code API request failed: {e}");
                }
                if let Ok(closer) = closer {
                    close_gracefully(closer);
                }
            });
        }
    });
}

/// Ends a connection so the client gets every byte. Dropping the socket right after the last
/// write let Windows turn the close into a reset that discarded what was still in flight: about
/// one streamed reply in three lost its last piece and `[DONE]`, and the client saw "connection
/// forcibly closed". Sending FIN first and waiting briefly for the client to close its side is the
/// usual graceful close.
fn close_gracefully(mut stream: TcpStream) {
    use std::io::Read;
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut sink = [0u8; 512];
    while matches!(stream.read(&mut sink), Ok(n) if n > 0) {}
}

#[derive(Debug, PartialEq)]
struct Request {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

fn read_request(reader: &mut impl BufRead) -> Result<Request, String> {
    let mut head = Vec::new();
    loop {
        let mut line = Vec::new();
        let n = reader.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed before the request ended".into());
        }
        head.extend_from_slice(&line);
        if head.len() > MAX_HEADER_BYTES {
            return Err("request headers too large".into());
        }
        if line == b"\r\n" || line == b"\n" {
            break;
        }
    }
    let head = String::from_utf8_lossy(&head);
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next().unwrap_or("").to_string();
    let path = first.next().unwrap_or("").split('?').next().unwrap_or("").to_string();
    let mut authorization = None;
    let mut length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "authorization" => authorization = Some(value.to_string()),
            "content-length" => length = value.parse().map_err(|_| "bad Content-Length")?,
            _ => {}
        }
    }
    if length > MAX_BODY_BYTES {
        return Err("request body too large".into());
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, authorization, body })
}

fn authorized(request: &Request, token: &str) -> bool {
    let Some(given) = request.authorization.as_deref().and_then(|h| h.strip_prefix("Bearer ")) else {
        return false;
    };
    // Same work whatever the first wrong byte is.
    given.len() == token.len() && given.bytes().zip(token.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

fn write_json(stream: &mut TcpStream, status: &str, body: &Value) -> std::io::Result<()> {
    let body = body.to_string();
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

fn error_body(message: &str) -> Value {
    json!({ "error": { "message": message, "type": "invalid_request_error" } })
}

/// The responses that need neither the engine nor the app, as (status, body), checking the token
/// first; `None` for the routes `handle` serves itself (chat, starting a folder-agent run,
/// feedback) and for unknown routes. Split out of `handle` so tests can send requests through the
/// router and the auth check: mutation testing found any route, or the token check itself, could
/// be removed without a test failing.
fn simple_route(request: &Request, token: &str, model_name: &str) -> Option<(&'static str, Value)> {
    let bad = |e: String| ("400 Bad Request", error_body(&e));
    Some(match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => ("200 OK", json!({ "ok": true, "name": "xenon2", "model": model_name })),
        _ if !authorized(request, token) => ("401 Unauthorized", error_body("missing or wrong bearer token")),
        ("GET", "/v1/models") => ("200 OK", json!({ "object": "list", "data": [{ "id": MODEL_ID, "object": "model", "owned_by": "xenon2" }] })),
        // The prompt a chat request would get, without generating (and without rated examples,
        // which change as ratings come in). Training data is built through this, so it always
        // matches what the model sees in use.
        ("POST", "/v1/prompt") => {
            let parsed = serde_json::from_slice::<ChatRequest>(&request.body)
                .map_err(|e| format!("invalid request JSON: {e}"))
                .and_then(|r| build_prompt(&r, Style::for_model(model_name)).map(|p| (r, p)));
            match parsed {
                Ok((chat, mut prompt)) => {
                    add_search(&chat, &mut prompt);
                    ("200 OK", json!({ "prior": prompt.prior, "last": prompt.last }))
                }
                Err(e) => bad(e),
            }
        }
        // Code search as a tool: the lines of `workspace` that best match `query`.
        ("POST", "/v1/search") => match serde_json::from_slice::<SearchRequest>(&request.body) {
            Ok(req) if Path::new(&req.workspace).is_dir() => {
                ("200 OK", json!({ "lines": crate::code_search::search(Path::new(&req.workspace), &req.query) }))
            }
            Ok(req) => bad(format!("{} is not a folder", req.workspace)),
            Err(e) => bad(format!("invalid request JSON: {e}")),
        },
        ("GET", path) if path.starts_with("/v1/folder-agent/") => {
            match folder_agent::job_status(path.trim_start_matches("/v1/folder-agent/")) {
                Some(status) => ("200 OK", json!(status)),
                None => ("404 Not Found", error_body("no such run")),
            }
        }
        ("POST", path) if path.starts_with("/v1/folder-agent/") && path.ends_with("/cancel") => {
            let id = path.trim_start_matches("/v1/folder-agent/").trim_end_matches("/cancel");
            if folder_agent::cancel_job(id) {
                ("200 OK", json!({ "ok": true }))
            } else {
                ("404 Not Found", error_body("no such run"))
            }
        }
        ("POST", "/v1/folder-agent/undo") => match serde_json::from_slice::<FolderAgentUndoRequest>(&request.body) {
            Ok(req) => match folder_agent::undo(Path::new(&req.folder), &req.snapshot_commit) {
                Ok(()) => ("200 OK", json!({ "ok": true })),
                Err(e) => ("500 Internal Server Error", error_body(&e)),
            },
            Err(e) => bad(format!("invalid request JSON: {e}")),
        },
        _ => return None,
    })
}

fn handle(stream: TcpStream, app: &AppHandle, engine: &EngineState, token: &str, model_name: &str) -> Result<(), String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut stream = stream;
    let request = match read_request(&mut reader) {
        Ok(r) => r,
        Err(e) => {
            let _ = write_json(&mut stream, "400 Bad Request", &error_body(&e));
            return Err(e);
        }
    };

    let io = |e: std::io::Error| e.to_string();
    if let Some((status, body)) = simple_route(&request, token, model_name) {
        return write_json(&mut stream, status, &body).map_err(io);
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/chat/completions") => {
            let parsed = serde_json::from_slice::<ChatRequest>(&request.body)
                .map_err(|e| format!("invalid request JSON: {e}"))
                .and_then(|r| build_prompt(&r, Style::for_model(model_name)).map(|p| (r, p)));
            match parsed {
                Ok((chat, mut prompt)) => {
                    let last_question = last_question(&chat);
                    add_search(&chat, &mut prompt);
                    let examples = example_bank::examples_for(app, &last_question, AGENT);
                    if !examples.is_empty() {
                        eprintln!("xenon2: code API using {} rated example(s) in this prompt", examples.len());
                        prompt.last = splice_examples("", &examples, Style::for_model(model_name)) + &prompt.last;
                    }
                    chat_completion(&mut stream, app, engine, &chat, &prompt)
                }
                Err(e) => write_json(&mut stream, "400 Bad Request", &error_body(&e)).map_err(io),
            }
        }
        ("POST", "/v1/folder-agent") => {
            let parsed = serde_json::from_slice::<FolderAgentRequest>(&request.body)
                .map_err(|e| format!("invalid request JSON: {e}"));
            match parsed {
                Ok(req) => {
                    let id = format!("run-{}", &new_token()[..16]);
                    let started = folder_agent::start_job(
                        engine.clone(),
                        model_name.to_string(),
                        std::path::PathBuf::from(&req.folder),
                        req.goal,
                        req.files,
                        id.clone(),
                        crate::memory::effective_data_dir(app).ok().filter(|_| req.learn).map(|d| d.join("mistakes.jsonl")),
                    );
                    match started {
                        Ok(()) => write_json(&mut stream, "202 Accepted", &json!({ "id": id })).map_err(io),
                        Err(e) => write_json(&mut stream, "409 Conflict", &error_body(&e)).map_err(io),
                    }
                }
                Err(e) => write_json(&mut stream, "400 Bad Request", &error_body(&e)).map_err(io),
            }
        }
        ("POST", "/v1/feedback") => {
            let parsed = serde_json::from_slice::<FeedbackRequest>(&request.body)
                .map_err(|e| format!("invalid request JSON: {e}"));
            match parsed {
                Ok(fb) => match record_feedback(app, fb) {
                    Ok(summary) => write_json(&mut stream, "200 OK", &json!(summary)).map_err(io),
                    Err(e) => write_json(&mut stream, "500 Internal Server Error", &error_body(&e)).map_err(io),
                },
                Err(e) => write_json(&mut stream, "400 Bad Request", &error_body(&e)).map_err(io),
            }
        }
        _ => write_json(&mut stream, "404 Not Found", &error_body("no such route")).map_err(io),
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<i32>,
    pub max_completion_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Not part of the OpenAI format; the extension may send it.
    pub repeat_penalty: Option<f32>,
    /// Not part of the OpenAI format: the project folder the question is about. When given, the
    /// lines of it that best match the question go in front of it (see `code_search.rs`).
    pub workspace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

/// Message text, from either a plain string or OpenAI's list of `{type: "text", text}` parts.
fn message_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Normalizes text for the "User: ... \n\nXenon: ..." format: blank lines would read as the end
/// of a turn, so runs of newlines become one, as in RWKV World's own chat data.
fn clean(text: &str) -> String {
    let text = text.replace("\r\n", "\n");
    let mut out = String::with_capacity(text.len());
    for ch in text.trim().chars() {
        if ch == '\n' && out.ends_with('\n') {
            continue;
        }
        out.push(ch);
    }
    out
}

#[derive(Debug, PartialEq)]
pub struct Prompt {
    /// Preamble, system text and earlier turns. Cached between requests (see
    /// `inference::generate_code`), so it always ends on a turn boundary.
    pub prior: String,
    /// The new user turn, ending where Xenon's reply starts.
    pub last: String,
    pub params: SamplingParams,
}

pub fn build_prompt(request: &ChatRequest, style: Style) -> Result<Prompt, String> {
    let (last, earlier) = request.messages.split_last().ok_or("messages is empty")?;
    if last.role != "user" {
        return Err("the last message must be from the user".into());
    }
    let mut prior = CODE_PREFIX.replace("

Xenon: ", &format!("

{}: ", style.label));
    if let Some(notes) = request.workspace.as_deref().and_then(self_notes) {
        prior.push_str(&notes);
    }
    let system: Vec<String> = earlier
        .iter()
        .filter(|m| m.role == "system" || m.role == "developer")
        .map(|m| clean(&message_text(&m.content)))
        .filter(|t| !t.is_empty())
        .collect();
    if !system.is_empty() {
        // Kept whole: RWKV has no context window. Long system text only costs reading time
        // (about 35 tokens/s here), and only once, since it sits in the cached prefix.
        prior.push_str(&clean(&system.join("\n")));
        prior.push_str("\n\n");
    }
    for message in earlier {
        let speaker = match message.role.as_str() {
            "user" => "User",
            "assistant" => style.label,
            _ => continue,
        };
        let text = clean(&message_text(&message.content));
        if !text.is_empty() {
            prior.push_str(&format!("{speaker}: {text}\n\n"));
        }
    }
    let question = clean(&message_text(&last.content));
    if question.is_empty() {
        return Err("the last message is empty".into());
    }
    let params = SamplingParams {
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS)
            .clamp(1, MAX_MAX_TOKENS),
        temperature: request.temperature.unwrap_or(0.3).clamp(0.0, 2.0),
        top_p: request.top_p.unwrap_or(0.3).clamp(0.01, 1.0),
        // Code repeats tokens by nature; chat's 1.3 is tuned against canned phrases, not code.
        repeat_penalty: request.repeat_penalty.unwrap_or(1.2).clamp(1.0, 2.0),
    };
    Ok(Prompt { prior, last: format!("User: {question}\n\n{}:{}", style.label, style.opener), params })
}

/// Where a project keeps the description of how it is built that Xenon reads (for Xenon2 itself,
/// Xenon's own description of how it works).
const SELF_NOTES: &str = ".xenon/self.md";

/// The workspace's `.xenon/self.md`, as a block for the cached prior. Asked how it works, Xenon
/// otherwise guessed: it called itself a transformer (15.9% on `training/eval_self.py`). Searching
/// the notes by keyword picked the wrong section for questions like "What GPU do you run on?", so
/// the whole file goes in. It sits right after the preamble, where it is the same on every
/// request, so after the first request it is read from the cache, not again.
fn self_notes(workspace: &str) -> Option<String> {
    let text = std::fs::read_to_string(std::path::Path::new(workspace).join(SELF_NOTES)).ok()?;
    let text = clean(&text);
    (!text.is_empty()).then(|| format!("Xenon's notes on how this project, and Xenon itself, are built:\n{text}\n\n"))
}

/// Puts search results inside the new user turn as question, results, question again. A recurrent
/// model reads once, front to back, so it should know what to look for before the results, and
/// asking again after them helps recall (+11 points on average across 16 recurrent models, RWKV
/// among them: "Just Read Twice", arXiv 2407.05483). Like rated examples, they stay out of the
/// cached `prior`.
fn with_found_lines(last: &str, found: &str) -> String {
    let Some(rest) = last.strip_prefix("User: ") else { return last.to_string() };
    let Some((question, reply_start)) = rest.split_once("\n\n") else { return last.to_string() };
    format!("User: {question}\n{}\n{question}\n\n{reply_start}", clean(found))
}

/// Appends rated examples to `text` as "User: ... / <label>: ..." turns, the same shape as the
/// chat window's example bank (see `example_bank.rs`). They go in the uncached final part of the
/// prompt, just before the new question: in the cached `prior` they would change it, and the next
/// follow-up (whose history doesn't have them) would miss the cache and re-read everything.
fn splice_examples(prior: &str, examples: &[example_bank::Example], style: Style) -> String {
    let mut out = prior.to_string();
    for example in examples {
        out.push_str(&format!("User: {}\n\n{}: {}\n\n", example.question, style.label, example.answer));
    }
    out
}

#[derive(Debug, Deserialize)]
pub struct FolderAgentRequest {
    /// Absolute path to the folder the agent may read, write, and run code in.
    pub folder: String,
    /// The English goal for this run, e.g. "add type hints and fix any failing tests".
    pub goal: String,
    /// Paths, relative to `folder`, the agent may read and edit. Nothing outside this list.
    /// Empty or left out: the agent searches the folder for the goal and picks a few files.
    #[serde(default)]
    pub files: Vec<String>,
    /// Whether mistakes the run fixes go into the next training batch. Benchmarks turn this off,
    /// so Xenon is never trained on its own exam.
    #[serde(default = "default_learn")]
    pub learn: bool,
}

fn default_learn() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub workspace: String,
    pub query: String,
}

fn last_question(chat: &ChatRequest) -> String {
    chat.messages.last().map(|m| message_text(&m.content)).unwrap_or_default()
}

/// When the request names a workspace folder, puts the lines of it that best match the question
/// in front of the question.
fn add_search(chat: &ChatRequest, prompt: &mut Prompt) {
    let Some(root) = chat.workspace.as_deref().map(std::path::Path::new).filter(|p| p.is_dir()) else { return };
    let started = std::time::Instant::now();
    let found = crate::code_search::search(root, &last_question(chat));
    eprintln!("xenon2: code search took {} ms, found {} chars", started.elapsed().as_millis(), found.len());
    if !found.is_empty() {
        prompt.last = with_found_lines(&prompt.last, &found);
    }
}

#[derive(Debug, Deserialize)]
pub struct FolderAgentUndoRequest {
    pub folder: String,
    pub snapshot_commit: String,
}

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    /// Opaque id for the rated reply; any stable string the client chooses.
    pub message_id: String,
    /// "good", "bad", or "none" to clear a rating.
    pub rating: String,
    #[serde(default)]
    pub correction: Option<String>,
    /// Every turn before the rated reply, ending with the user turn it answered.
    pub history: Vec<ChatMessage>,
    pub reply: String,
}

fn record_feedback(app: &AppHandle, request: FeedbackRequest) -> Result<feedback::FeedbackSummary, String> {
    let path = feedback::feedback_path(app)?;
    let entry = FeedbackEntry {
        id: new_token(),
        created_at: now_secs() as i64,
        conversation_id: request.message_id.clone(),
        message_id: request.message_id,
        rating: request.rating,
        correction: request.correction,
        history: request
            .history
            .iter()
            .map(|m| FeedbackTurn { role: m.role.clone(), content: message_text(&m.content) })
            .collect(),
        reply: request.reply,
        model: MODEL_ID.to_string(),
        agent: AGENT.to_string(),
    };
    feedback::record_entry(&path, entry)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Length => "length",
        Finish::Stop | Finish::Cancelled => "stop",
    }
}

fn sse_chunk(id: &str, created: u64, delta: Value, finish: Option<&str>) -> String {
    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    format!("data: {chunk}\n\n")
}

fn chat_completion(stream: &mut TcpStream, app: &AppHandle, engine: &EngineState, chat: &ChatRequest, prompt: &Prompt) -> Result<(), String> {
    let id = format!("chatcmpl-{}", &new_token()[..24]);
    let created = now_secs();
    let _ = app.emit("code-api-busy", true);
    let started = std::time::Instant::now();

    let result = if chat.stream {
        let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        let opened = stream
            .write_all(header.as_bytes())
            .and_then(|_| stream.write_all(sse_chunk(&id, created, json!({ "role": "assistant", "content": "" }), None).as_bytes()))
            .and_then(|_| stream.flush());
        match opened {
            Err(e) => Err(e.to_string()),
            Ok(()) => {
                let mut sink = |text: &str| {
                    stream
                        .write_all(sse_chunk(&id, created, json!({ "content": text }), None).as_bytes())
                        .and_then(|_| stream.flush())
                        .is_ok()
                };
                let outcome = inference::generate_code(engine, &prompt.prior, &prompt.last, prompt.params, &mut sink);
                let tail = match &outcome {
                    Ok(finish) => sse_chunk(&id, created, json!({}), Some(finish_reason(*finish))),
                    Err(e) => format!("data: {}\n\n", error_body(e)),
                };
                let _ = stream.write_all(format!("{tail}data: [DONE]\n\n").as_bytes()).and_then(|_| stream.flush());
                outcome.map(|_| ())
            }
        }
    } else {
        let mut text = String::new();
        let mut sink = |piece: &str| {
            text.push_str(piece);
            true
        };
        match inference::generate_code(engine, &prompt.prior, &prompt.last, prompt.params, &mut sink) {
            Ok(finish) => write_json(
                stream,
                "200 OK",
                &json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": text.trim() },
                        "finish_reason": finish_reason(finish),
                    }],
                }),
            )
            .map_err(|e| e.to_string()),
            Err(e) => {
                let _ = write_json(stream, "500 Internal Server Error", &error_body(&e));
                Err(e)
            }
        }
    };
    let _ = app.emit("code-api-busy", false);
    eprintln!("xenon2: code API reply in {:.1}s", started.elapsed().as_secs_f32());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_prompt_world(request: &ChatRequest) -> Result<Prompt, String> {
        build_prompt(request, Style::WORLD)
    }

    fn request(messages: Value) -> ChatRequest {
        serde_json::from_value(json!({ "messages": messages })).unwrap()
    }

    #[test]
    fn a_workspace_with_self_notes_puts_them_in_the_cached_prior() {
        let dir = std::env::temp_dir().join(format!("xenon-self-notes-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".xenon")).unwrap();
        std::fs::write(dir.join(".xenon/self.md"), "# How Xenon is built\n\n## The model\n\nRWKV-7 2.9B.\n").unwrap();
        let ask = |workspace: &std::path::Path| -> ChatRequest {
            serde_json::from_value(json!({ "messages": [{ "role": "user", "content": "What model?" }], "workspace": workspace })).unwrap()
        };
        let with = build_prompt_world(&ask(&dir)).unwrap();
        assert!(with.prior.starts_with(&build_prompt_world(&request(json!([{ "role": "user", "content": "x" }]))).unwrap().prior));
        assert!(with.prior.contains("## The model\nRWKV-7 2.9B."), "blank lines collapsed: {}", with.prior);
        assert!(with.prior.ends_with("\n\n") && !with.last.contains("RWKV"));
        // The same notes on every request, so the prior (and its cached state) is the same.
        assert_eq!(build_prompt_world(&ask(&dir)).unwrap().prior, with.prior);
        std::fs::remove_file(dir.join(".xenon/self.md")).unwrap();
        assert!(!build_prompt_world(&ask(&dir)).unwrap().prior.contains("notes"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn found_lines_sit_between_two_copies_of_the_question() {
        let last = "User: where is x?\n\nAssistant: <think></think>";
        assert_eq!(
            with_found_lines(last, "a.rs:1: fn x() {}\n"),
            "User: where is x?\na.rs:1: fn x() {}\nwhere is x?\n\nAssistant: <think></think>"
        );
    }

    #[test]
    fn reads_a_request_with_a_body() {
        let raw = b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: x\r\nauthorization: Bearer abc\r\nContent-Length: 2\r\n\r\n{}";
        let parsed = read_request(&mut &raw[..]).unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/v1/chat/completions");
        assert_eq!(parsed.authorization.as_deref(), Some("Bearer abc"));
        assert_eq!(parsed.body, b"{}");
    }

    #[test]
    fn rejects_a_huge_body() {
        let raw = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY_BYTES + 1);
        assert!(read_request(&mut raw.as_bytes()).is_err());
    }

    #[test]
    fn checks_the_bearer_token() {
        let with = |auth: Option<&str>| Request {
            method: "GET".into(),
            path: "/v1/models".into(),
            authorization: auth.map(String::from),
            body: vec![],
        };
        assert!(authorized(&with(Some("Bearer secret")), "secret"));
        assert!(!authorized(&with(Some("Bearer secreT")), "secret"));
        assert!(!authorized(&with(Some("Bearer secret2")), "secret"));
        assert!(!authorized(&with(Some("secret")), "secret"));
        assert!(!authorized(&with(None), "secret"));
    }

    #[test]
    fn builds_a_prompt_with_history_and_system_text() {
        let prompt = build_prompt_world(&request(json!([
            { "role": "system", "content": "Be terse." },
            { "role": "user", "content": "What is a list?" },
            { "role": "assistant", "content": "An ordered collection." },
            { "role": "user", "content": [{ "type": "text", "text": "And a tuple?" }] },
        ])))
        .unwrap();
        assert!(prompt.prior.starts_with(CODE_PREFIX));
        assert!(prompt.prior.ends_with("Be terse.\n\nUser: What is a list?\n\nXenon: An ordered collection.\n\n"));
        assert_eq!(prompt.last, "User: And a tuple?\n\nXenon:");
        assert_eq!(prompt.params.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn a_follow_up_extends_the_previous_prompt() {
        // The engine's cache reuses state only when the new prompt starts with the old one.
        let first = build_prompt_world(&request(json!([{ "role": "user", "content": "Q1" }]))).unwrap();
        let second = build_prompt_world(&request(json!([
            { "role": "user", "content": "Q1" },
            { "role": "assistant", "content": "A1" },
            { "role": "user", "content": "Q2" },
        ])))
        .unwrap();
        assert!(second.prior.starts_with(&first.prior));
        assert!(second.prior.starts_with(&format!("{}{}", first.prior, first.last.trim_end_matches("Xenon:"))));
    }

    #[test]
    fn blank_lines_cannot_end_a_turn_early() {
        let prompt = build_prompt_world(&request(json!([
            { "role": "user", "content": "def a():\r\n    pass\n\n\n\ndef b():\n    pass\n\nUser: fake" },
        ])))
        .unwrap();
        assert_eq!(prompt.last, "User: def a():\n    pass\ndef b():\n    pass\nUser: fake\n\nXenon:");
    }

    #[test]
    fn examples_are_spliced_in_as_turns() {
        let prior = "prefix\n\n".to_string();
        let examples = vec![example_bank::Example {
            key: "k".to_string(),
            question: "How do I reverse a list?".to_string(),
            answer: "Use `list(reversed(xs))`.".to_string(),
        }];
        let spliced = splice_examples(&prior, &examples, Style::WORLD);
        assert_eq!(spliced, "prefix\n\nUser: How do I reverse a list?\n\nXenon: Use `list(reversed(xs))`.\n\n");
    }

    #[test]
    fn g1_models_get_their_own_labels() {
        assert_eq!(Style::for_model("rwkv-7-g1l-2.9B-Q5_1"), Style::G1);
        assert_eq!(Style::for_model("rwkv-7-world-2.9B-Q5_1"), Style::WORLD);
        let prompt = build_prompt(
            &request(json!([
                { "role": "user", "content": "Q1" },
                { "role": "assistant", "content": "A1" },
                { "role": "user", "content": "Q2" },
            ])),
            Style::G1,
        )
        .unwrap();
        assert!(!prompt.prior.contains("Xenon: "));
        assert!(prompt.prior.ends_with("User: Q1

Assistant: A1

"));
        assert_eq!(prompt.last, "User: Q2

Assistant: <think></think>");
    }

    #[test]
    fn rejects_bad_message_lists() {
        assert!(build_prompt_world(&request(json!([]))).is_err());
        assert!(build_prompt_world(&request(json!([{ "role": "assistant", "content": "hi" }]))).is_err());
        assert!(build_prompt_world(&request(json!([{ "role": "user", "content": "  " }]))).is_err());
    }

    #[test]
    fn clamps_sampling_settings() {
        let parsed: ChatRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 10_000_000, "temperature": -1.0, "top_p": 0.0,
        }))
        .unwrap();
        let params = build_prompt_world(&parsed).unwrap().params;
        assert_eq!(params.max_tokens, MAX_MAX_TOKENS);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 0.01);
    }

    #[test]
    fn long_system_text_is_kept_whole() {
        let long = "x".repeat(20_000);
        let prompt = build_prompt_world(&request(json!([
            { "role": "system", "content": long.clone() },
            { "role": "user", "content": "hi" },
        ])))
        .unwrap();
        assert!(prompt.prior.contains(&long));
    }

    #[test]
    fn stream_chunks_are_openai_shaped() {
        let chunk = sse_chunk("id1", 7, json!({ "content": "hi" }), None);
        assert!(chunk.starts_with("data: ") && chunk.ends_with("\n\n"));
        let parsed: Value = serde_json::from_str(chunk.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(parsed["choices"][0]["delta"]["content"], "hi");
        assert_eq!(parsed["object"], "chat.completion.chunk");
    }
    // ---- Written against cargo-mutants' misses: routes, the token check, limits, helpers ----

    const TOKEN: &str = "s3cret-token";

    fn req(method: &str, path: &str, token: Option<&str>, body: &str) -> Request {
        Request {
            method: method.into(),
            path: path.into(),
            authorization: token.map(|t| format!("Bearer {t}")),
            body: body.as_bytes().to_vec(),
        }
    }

    fn route(request: Request) -> Option<(&'static str, Value)> {
        simple_route(&request, TOKEN, "rwkv-test")
    }

    #[test]
    fn health_answers_without_a_token_and_names_the_model() {
        let (status, body) = route(req("GET", "/health", None, "")).unwrap();
        assert_eq!(status, "200 OK");
        assert_eq!(body["model"], "rwkv-test");
    }

    #[test]
    fn every_other_route_needs_the_token() {
        for (method, path) in [
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/prompt"),
            ("POST", "/v1/search"),
            ("POST", "/v1/folder-agent"),
            ("GET", "/v1/folder-agent/run-x"),
            ("POST", "/v1/feedback"),
            ("GET", "/no/such/route"),
        ] {
            for token in [None, Some("wrong")] {
                let (status, _) = route(req(method, path, token, "{}")).unwrap();
                assert_eq!(status, "401 Unauthorized", "{method} {path} with {token:?}");
            }
        }
    }

    #[test]
    fn models_lists_xenon() {
        let (status, body) = route(req("GET", "/v1/models", Some(TOKEN), "")).unwrap();
        assert_eq!(status, "200 OK");
        assert_eq!(body["data"][0]["id"], MODEL_ID);
    }

    #[test]
    fn the_prompt_route_builds_the_chat_prompt_with_search_results() {
        let dir = std::env::temp_dir().join(format!("xenon-api-prompt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("engine.rs"), "fn threads_for(n: u32) -> u32 { n }\n").unwrap();
        let body = json!({ "messages": [{ "role": "user", "content": "Where is `threads_for`?" }], "workspace": dir });
        let (status, out) = route(req("POST", "/v1/prompt", Some(TOKEN), &body.to_string())).unwrap();
        assert_eq!(status, "200 OK");
        let last = out["last"].as_str().unwrap();
        assert!(last.contains("Where is `threads_for`?"), "{last}");
        assert!(last.contains("engine.rs:1: fn threads_for"), "search results are spliced in: {last}");
        assert!(out["prior"].as_str().unwrap().contains("programming"), "the coding preamble");

        let (status, _) = route(req("POST", "/v1/prompt", Some(TOKEN), "not json")).unwrap();
        assert_eq!(status, "400 Bad Request");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_search_route_searches_a_folder_and_refuses_anything_else() {
        let dir = std::env::temp_dir().join(format!("xenon-api-search-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.py"), "def zebra_marker():\n    pass\n").unwrap();
        let ask = |workspace: &Path| json!({ "workspace": workspace, "query": "zebra_marker" }).to_string();

        let (status, body) = route(req("POST", "/v1/search", Some(TOKEN), &ask(&dir))).unwrap();
        assert_eq!(status, "200 OK");
        assert!(body["lines"].as_str().unwrap().contains("a.py:1: def zebra_marker"), "{body}");

        let (status, body) = route(req("POST", "/v1/search", Some(TOKEN), &ask(&dir.join("a.py")))).unwrap();
        assert_eq!(status, "400 Bad Request");
        assert!(body["error"]["message"].as_str().unwrap().contains("is not a folder"));

        let (status, _) = route(req("POST", "/v1/search", Some(TOKEN), "{")).unwrap();
        assert_eq!(status, "400 Bad Request");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn folder_agent_status_cancel_and_undo_answer_for_unknown_runs() {
        let (status, _) = route(req("GET", "/v1/folder-agent/run-missing", Some(TOKEN), "")).unwrap();
        assert_eq!(status, "404 Not Found");
        let (status, _) = route(req("POST", "/v1/folder-agent/run-missing/cancel", Some(TOKEN), "")).unwrap();
        assert_eq!(status, "404 Not Found");
        let (status, _) = route(req("POST", "/v1/folder-agent/undo", Some(TOKEN), "{}")).unwrap();
        assert_eq!(status, "400 Bad Request", "undo needs a folder and a commit");
        let missing = json!({ "folder": std::env::temp_dir().join("xenon-no-such-folder"), "snapshot_commit": "abc" });
        let (status, _) = route(req("POST", "/v1/folder-agent/undo", Some(TOKEN), &missing.to_string())).unwrap();
        assert_eq!(status, "500 Internal Server Error");
    }

    #[test]
    fn engine_routes_and_unknown_routes_are_left_to_handle() {
        for (method, path) in [("POST", "/v1/chat/completions"), ("POST", "/v1/folder-agent"), ("POST", "/v1/feedback"), ("GET", "/nope")] {
            assert!(route(req(method, path, Some(TOKEN), "{}")).is_none(), "{method} {path}");
        }
    }

    #[test]
    fn a_wrong_token_is_refused_even_when_its_differences_cancel_out() {
        // "ab" and "ba" differ by the same bits in both places; XOR-ing the differences together
        // would cancel them out and let the wrong token in.
        let with = |token: &str| req("GET", "/v1/models", Some(token), "");
        assert!(!authorized(&with("ba"), "ab"));
        assert!(authorized(&with("ab"), "ab"));
    }

    #[test]
    fn a_body_of_exactly_the_limit_is_read() {
        let body = "x".repeat(MAX_BODY_BYTES);
        let raw = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", MAX_BODY_BYTES);
        assert_eq!(read_request(&mut raw.as_bytes()).unwrap().body.len(), MAX_BODY_BYTES);
    }

    #[test]
    fn headers_up_to_the_limit_are_read_and_one_byte_more_is_not() {
        // The request line (16 bytes), one filler header, and the blank line (2) make the head.
        let head = |size: usize| format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(size - 16 - 5 - 2));
        assert_eq!(head(MAX_HEADER_BYTES).len(), MAX_HEADER_BYTES);
        assert!(read_request(&mut head(MAX_HEADER_BYTES).as_bytes()).is_ok());
        assert!(read_request(&mut head(MAX_HEADER_BYTES + 1).as_bytes()).is_err());
    }

    #[test]
    fn a_json_reply_arrives_whole_over_a_real_socket() {
        use std::io::Read;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            write_json(&mut stream, "200 OK", &json!({ "ok": true, "pad": "x".repeat(100_000) })).unwrap();
            close_gracefully(stream);
        });
        let mut client = TcpStream::connect(address).unwrap();
        let mut reply = String::new();
        client.read_to_string(&mut reply).unwrap();
        server.join().unwrap();
        let (head, body) = reply.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert!(head.contains(&format!("Content-Length: {}", body.len())), "{head}");
        assert_eq!(serde_json::from_str::<Value>(body).unwrap()["ok"], true);
    }

    #[test]
    fn small_helpers() {
        assert_eq!(error_body("boom")["error"]["message"], "boom");
        let (a, b) = (new_token(), new_token());
        assert_eq!(a.len(), 48);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "tokens are random");
        assert_eq!(finish_reason(Finish::Length), "length");
        assert_eq!(finish_reason(Finish::Stop), "stop");
        assert_eq!(finish_reason(Finish::Cancelled), "stop");
        assert!(now_secs() > 1_700_000_000, "a real clock, not zero");
        let agent: FolderAgentRequest = serde_json::from_str(r#"{"folder": "f", "goal": "g"}"#).unwrap();
        assert!(agent.learn, "runs learn unless told not to");
        assert!(agent.files.is_empty());
        let chat = request(json!([{ "role": "user", "content": "first" }, { "role": "user", "content": "last one" }]));
        assert_eq!(last_question(&chat), "last one");
    }
}
