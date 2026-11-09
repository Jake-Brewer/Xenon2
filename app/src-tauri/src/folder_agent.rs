//! Project-folder agent: point Xenon at a folder and an English goal, and it edits, checks, and
//! fixes the code in that folder on its own (Python, Rust, TypeScript, Vue; see `check_for`;
//! Xenon2 itself is the main target), batch style, stopping on a hard budget and
//! reporting what it did. This is a different shape from `code_api.rs`'s single question/answer:
//! a loop that owns a working directory for a while, writes files, and runs code it chose to run.
//!
//! Safety, in order of how a run can go wrong:
//! - *Containment*: every path the agent writes to is canonicalized and checked to be inside the
//!   target folder before anything touches disk. The agent is only ever given paths relative to
//!   that folder, so there is no "../../" escape route in a model-chosen filename.
//! - *Rollback*: before the first edit, the folder's git state is snapshotted with a throwaway
//!   commit (only if the tree was already clean — a dirty tree before the run starts gets no
//!   snapshot, since "revert to the snapshot" would otherwise discard the user's own in-progress
//!   work). `undo()` resets straight back to that commit.
//! - *Budget*: a run stops itself after `MAX_EDITS` file writes, `MAX_RUNS` test/script
//!   executions, or `MAX_SECONDS` of wall time, whichever comes first.
//! - *Execution*: checks run with their working directory inside the target folder. This is
//!   not a hard OS sandbox (Windows has no cheap equivalent of a container here) — it relies on
//!   the containment check plus the rollback point, not on stopping the subprocess from reading
//!   elsewhere on disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::code_api::{self, ChatMessage, ChatRequest, Style};
use crate::inference::{self, EngineState};

pub const MAX_EDITS: u32 = 5;
pub const MAX_RUNS: u32 = 3;
pub const MAX_SECONDS: u64 = 10 * 60;
/// Lines shown on each side of the lines Xenon rewrites, as context it can't change.
const CONTEXT_LINES: usize = 3;
/// Lines Xenon rewrites from a definition down (its signature and the start of its body).
const EDIT_LINES: usize = 4;
/// Files to work on when none are named: few, since each is read whole into every prompt.
const FILES_FROM_SEARCH: usize = 3;

const SNAPSHOT_MESSAGE: &str = "xenon-agent: snapshot before run";

/// The RTX 2060 here is in a laptop: before each model turn, wait for a hot GPU to cool (see
/// `gpu_temp.rs`, which also throttles between tokens).
use crate::gpu_temp::{COOL_C, HOT_C};
const MAX_COOL_WAIT: Duration = Duration::from_secs(5 * 60);

fn gpu_temp() -> Option<u32> {
    crate::gpu_temp::read()
}

/// Waits out a hot GPU. Returns a narration line when it had to wait.
fn cool_down(cancel: &AtomicBool) -> Option<String> {
    let hot = gpu_temp().filter(|t| *t >= HOT_C)?;
    let started = Instant::now();
    while started.elapsed() < MAX_COOL_WAIT && !cancel.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(5));
        if gpu_temp().map_or(true, |t| t <= COOL_C) {
            break;
        }
    }
    Some(format!("Paused {}s to let the GPU cool (it was at {hot}°C)", started.elapsed().as_secs()))
}

/// The reply's one-line reason after `tag`, cut where the model runs on into an invented turn.
fn reason_after(text: &str, tag: &str) -> Option<String> {
    let rest = &text[text.find(tag)? + tag.len()..];
    let rest = rest.split('\n').next().unwrap_or("");
    let rest = rest.split("User:").next().unwrap_or("").trim();
    Some(rest.chars().take(300).collect())
}

/// The start of a reply the loop couldn't use, for the report.
fn excerpt(reply: &str) -> String {
    let text: String = reply.trim().chars().take(400).collect();
    if text.is_empty() {
        "(nothing)".to_string()
    } else {
        format!("{text:?}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStep {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Whether the run stopped because it finished, or hit a budget limit.
    pub stopped_reason: String,
    pub edits: Vec<FileChange>,
    pub runs: Vec<RunStep>,
    /// One line per step explaining why the agent did it, in order taken.
    pub narration: Vec<String>,
    /// The snapshot commit to revert to, if one was taken (the tree was clean at the start).
    pub snapshot_commit: Option<String>,
}

/// Checks that `candidate` (already joined to `root`) stays inside `root` once symlinks and
/// `..` are resolved, so a model-chosen relative path can never escape the folder it was given.
fn contained(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    let resolved = if candidate.exists() {
        candidate.canonicalize().map_err(|e| format!("couldn't resolve {}: {e}", candidate.display()))?
    } else {
        // New file: resolve the existing parent, then re-attach the file name.
        let parent = candidate.parent().unwrap_or(root);
        let parent = parent.canonicalize().map_err(|e| format!("couldn't resolve {}: {e}", parent.display()))?;
        parent.join(candidate.file_name().ok_or("empty file name")?)
    };
    let root = root.canonicalize().map_err(|e| format!("couldn't resolve {}: {e}", root.display()))?;
    if resolved.starts_with(&root) {
        Ok(resolved)
    } else {
        Err(format!("{} is outside the target folder {}", candidate.display(), root.display()))
    }
}

/// A subprocess with no console window (the app has none, so Windows would flash one up per
/// command). npm and npx are `.cmd` scripts on Windows, which only `cmd /C` can start.
fn command(program: &str) -> Command {
    #[cfg(windows)]
    let mut cmd = if matches!(program, "npm" | "npx") {
        let mut c = Command::new("cmd");
        c.args(["/C", program]);
        c
    } else {
        Command::new(program)
    };
    #[cfg(not(windows))]
    let cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = command("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("couldn't run git: {e}"))?;
    if !output.status.success() {
        return Err(format!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Snapshots `root` with a throwaway commit if its tree is clean, so a run has a rollback point
/// that survives a crash (unlike a stash). Returns `None`, not an error, when the tree already
/// has changes in progress — snapshotting over those would make them look like part of the run.
/// Untracked files and changes inside submodules don't count: `undo`'s `reset --hard` leaves both
/// alone. (This repo always has both: `.claude/` and the patched `rwkv.cpp` checkout.)
fn snapshot(root: &Path) -> Result<Option<String>, String> {
    let status = git(root, &["status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty"])?;
    if !status.is_empty() {
        return Ok(None);
    }
    git(root, &["commit", "--allow-empty", "-m", SNAPSHOT_MESSAGE])?;
    Ok(Some(git(root, &["rev-parse", "HEAD"])?))
}

/// Resets `root` back to its pre-run state. Safe to call only because `snapshot` only commits
/// over an already-clean tree, so this never discards work that predates the run.
pub fn undo(root: &Path, snapshot_commit: &str) -> Result<(), String> {
    git(root, &["reset", "--hard", snapshot_commit])?;
    git(root, &["reset", "--soft", "HEAD~1"])?;
    Ok(())
}

struct Budget {
    started: Instant,
    edits: u32,
    runs: u32,
}

impl Budget {
    fn new() -> Self {
        Budget { started: Instant::now(), edits: 0, runs: 0 }
    }
    fn exhausted(&self) -> Option<String> {
        if self.started.elapsed() > Duration::from_secs(MAX_SECONDS) {
            Some(format!("hit the {}-minute time budget", MAX_SECONDS / 60))
        } else if self.edits >= MAX_EDITS {
            Some(format!("hit the {MAX_EDITS}-edit budget"))
        } else if self.runs >= MAX_RUNS {
            Some(format!("hit the {MAX_RUNS}-run budget"))
        } else {
            None
        }
    }
}

/// Asks the model a plain coding-API question and returns its reply as plain text (no
/// streaming, no example-bank splicing — this loop's own prompts carry their own context).
///
/// `start` is written into the reply before the model begins, and comes back at the front of the
/// returned text. Starting the reply with `FILE:` keeps G1, which was trained on tool use, from
/// answering a long code prompt with a `<tool_call>` instead of an edit. Generation ends when
/// `done` says the text so far is complete.
///
/// The question goes in the cached part of the prompt, so asking it again with a different
/// `start` doesn't re-read the files in it.
fn ask(
    engine: &EngineState,
    model_name: &str,
    question: &str,
    start: &str,
    done: &dyn Fn(&str) -> bool,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let style = Style::for_model(model_name);
    let request = ChatRequest {
        messages: vec![ChatMessage { role: "user".to_string(), content: serde_json::Value::String(question.to_string()) }],
        stream: false,
        max_tokens: Some(inference::MAX_REPLY_TOKENS),
        max_completion_tokens: None,
        temperature: Some(0.2),
        top_p: None,
        repeat_penalty: None,
        workspace: None,
    };
    let prompt = code_api::build_prompt(&request, style)?;
    // `last` is "User: ...\n\n<label>: ..."; the user turn ends on a turn boundary, which is
    // where cached prompts must end.
    let (user_turn, reply_start) = match prompt.last.rfind("\n\n") {
        Some(i) => prompt.last.split_at(i + 2),
        None => ("", prompt.last.as_str()),
    };
    let prior = format!("{}{user_turn}", prompt.prior);
    let last = format!("{reply_start}{start}");
    let mut text = start.to_string();
    let mut sink = |piece: &str| {
        text.push_str(piece);
        !done(&text) && Instant::now() < deadline && !cancel.load(Ordering::Relaxed)
    };
    inference::generate_code(engine, &prior, &last, prompt.params, &mut sink)?;
    Ok(text.trim().to_string())
}

/// True once the text after `FILE:` has a complete non-blank line (or has run long).
fn first_line_done(text: &str) -> bool {
    let rest = text.strip_prefix("FILE:").unwrap_or(text);
    rest.trim_start().contains('\n') || rest.len() > 300
}

/// The changed lines between two versions of a file, with two lines around them, in `-`/`+`
/// form. Enough to judge a small edit.
fn short_diff(before: &str, after: &str) -> String {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    let prefix = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suffix = a[prefix..].iter().rev().zip(b[prefix..].iter().rev()).take_while(|(x, y)| x == y).count();
    let mut out = Vec::new();
    for line in &a[prefix.saturating_sub(2)..prefix] {
        out.push(format!(" {line}"));
    }
    out.extend(a[prefix..a.len() - suffix].iter().map(|l| format!("-{l}")));
    out.extend(b[prefix..b.len() - suffix].iter().map(|l| format!("+{l}")));
    let after_end = (a.len() - suffix + 2).min(a.len());
    for line in &a[a.len() - suffix..after_end] {
        out.push(format!(" {line}"));
    }
    out.join("\n")
}

/// A command that checks an edit: tests where the project has them, else a typecheck or a run.
#[derive(Debug, PartialEq)]
struct Check {
    program: &'static str,
    args: Vec<String>,
    /// Relative to the run's root; "" is the root itself.
    dir: String,
}

impl Check {
    fn display(&self) -> String {
        let command = std::iter::once(self.program.to_string()).chain(self.args.iter().cloned()).collect::<Vec<_>>().join(" ");
        if self.dir.is_empty() {
            command
        } else {
            format!("{command} (in {})", self.dir)
        }
    }
}

fn is_python_test(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.starts_with("test_") || name.ends_with("_test.py")
}

/// The nearest folder at or above `relative`'s folder, inside `root`, that holds `marker`.
fn nearest_with(root: &Path, relative: &str, marker: &str) -> Option<String> {
    let mut dir = Path::new(relative).parent();
    while let Some(d) = dir {
        if root.join(d).join(marker).is_file() {
            return Some(d.to_string_lossy().replace('\\', "/"));
        }
        dir = d.parent();
    }
    None
}

/// Picks the check for an edit to `edited`: pytest (or the script itself) for Python, `cargo
/// test` for Rust, and the package's `test` script or a typecheck for TypeScript and Vue.
fn check_for(root: &Path, edited: &str, files: &[String]) -> Option<Check> {
    let ext = edited.rsplit('.').next().unwrap_or("");
    match ext {
        "py" => {
            let target = if is_python_test(edited) {
                edited.to_string()
            } else {
                files.iter().find(|f| is_python_test(f)).cloned().unwrap_or_else(|| edited.to_string())
            };
            let args = if is_python_test(&target) { vec!["-m".into(), "pytest".into(), target] } else { vec![target] };
            Some(Check { program: "python", args, dir: String::new() })
        }
        "rs" => {
            let dir = nearest_with(root, edited, "Cargo.toml")?;
            let mut args = vec!["test".to_string()];
            if root.join(&dir).join("src/lib.rs").is_file() {
                args.push("--lib".into());
            }
            Some(Check { program: "cargo", args, dir })
        }
        "ts" | "tsx" | "js" | "mjs" | "vue" => {
            let dir = nearest_with(root, edited, "package.json")?;
            let package = std::fs::read_to_string(root.join(&dir).join("package.json")).unwrap_or_default();
            let package: serde_json::Value = serde_json::from_str(&package).unwrap_or_default();
            let args: Vec<String> = if package["scripts"]["test"].is_string() {
                vec!["test".into()]
            } else if package["devDependencies"]["vue-tsc"].is_string() || package["dependencies"]["vue-tsc"].is_string() {
                vec!["vue-tsc".into(), "--noEmit".into()]
            } else {
                vec!["tsc".into(), "--noEmit".into()]
            };
            let program = if args[0] == "test" { "npm" } else { "npx" };
            Some(Check { program, args, dir })
        }
        _ => None,
    }
}

/// Runs `check`, killing it at `deadline` or on cancel. Output is read on its own threads so a
/// chatty command can't fill the pipe and stall.
fn run_check(root: &Path, check: &Check, deadline: Instant, cancel: &AtomicBool) -> Result<RunStep, String> {
    use std::io::Read;
    use std::process::Stdio;

    let display = check.display();
    let mut child = command(check.program)
        .args(&check.args)
        .current_dir(root.join(&check.dir))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("couldn't run {display}: {e}"))?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let mut stopped = None;
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(status);
        }
        if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
            stopped = Some(if cancel.load(Ordering::Relaxed) { "cancelled" } else { "out of time" });
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let mut stderr = stderr.join().unwrap_or_default();
    if let Some(why) = stopped {
        stderr.push_str(&format!("\n(stopped: {why})"));
    }
    Ok(RunStep {
        command: display,
        stdout: stdout.join().unwrap_or_default(),
        stderr,
        exit_code: status.and_then(|s| s.code()),
    })
}

/// The file in `files` that a model-written name means: an exact match, else the one entry ending
/// in it at a folder boundary (small models shorten paths, e.g. `src/x.rs` for `app/src/x.rs`).
fn snap_to_list(written: &str, files: &[String]) -> Option<String> {
    let written = written.replace('\\', "/");
    let written = written.trim_start_matches("./").trim_start_matches('/');
    if let Some(exact) = files.iter().find(|f| f.as_str() == written) {
        return Some(exact.clone());
    }
    let suffix = format!("/{written}");
    let mut found = files.iter().filter(|f| format!("/{f}").ends_with(&suffix));
    let first = found.next()?;
    found.next().is_none().then(|| first.clone())
}

/// The lines Xenon rewrites around line `best` (0-based, end exclusive): `best` and its
/// neighbours, or for a definition (`def`, `fn`, `function`...) its first lines, since what needs
/// changing is usually in the body (with the range centred on `def last_n`, Xenon couldn't reach
/// its `return` line). Blank lines are kept off both edges: Xenon dropped them when they were in.
fn edit_range(lines: &[&str], best: usize) -> (usize, usize) {
    const DEFINERS: &[&str] = &["def ", "fn ", "function ", "class ", "async def ", "pub fn ", "pub(crate) fn ", "export function "];
    let trimmed = lines[best].trim_start();
    let (mut from, mut to) = if DEFINERS.iter().any(|d| trimmed.starts_with(d)) {
        (best, (best + EDIT_LINES).min(lines.len()))
    } else {
        (best.saturating_sub(1), (best + 2).min(lines.len()))
    };
    while from < best && lines[from].trim().is_empty() {
        from += 1;
    }
    while to > best + 1 && lines[to - 1].trim().is_empty() {
        to -= 1;
    }
    (from, to)
}

/// `text` with its lines `from..to` (0-based, end exclusive) replaced by `new`, keeping the file's
/// line endings and its final newline.
fn replace_lines(text: &str, from: usize, to: usize, new: &str) -> String {
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = lines[..from].to_vec();
    out.extend(new.lines());
    out.extend(&lines[to..]);
    let mut joined = out.join(newline);
    if text.ends_with('\n') {
        joined.push_str(newline);
    }
    joined
}

/// The language name for a file's code fence.
fn fence(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "py" => "python",
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "mjs" => "javascript",
        "vue" => "vue",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cu" | "cuh" => "cpp",
        "sh" => "bash",
        "md" => "markdown",
        "toml" => "toml",
        "json" => "json",
        _ => "",
    }
}

/// Under `tauri dev`, writing to the app's own Rust source restarts the app, and the run with it.
/// Debug builds therefore leave that folder alone; an installed (release) app can edit it.
fn restarts_this_app(path: &Path) -> bool {
    if !cfg!(debug_assertions) {
        return false;
    }
    let own = Path::new(env!("CARGO_MANIFEST_DIR"));
    match (path.canonicalize().or_else(|_| path.parent().map_or(Err(std::io::ErrorKind::NotFound.into()), Path::canonicalize)), own.canonicalize()) {
        (Ok(p), Ok(own)) => p.starts_with(own),
        _ => false,
    }
}

/// Writes `new_text` to `relative_path` inside `root`, after confirming the path stays
/// contained, and returns the before/after pair for the report.
fn write_file(root: &Path, relative_path: &str, new_text: &str) -> Result<FileChange, String> {
    let target = contained(root, &root.join(relative_path))?;
    if restarts_this_app(&target) {
        return Err(format!(
            "{relative_path} is this app's own Rust code; editing it under `tauri dev` would restart the app mid-run. Use a release build for that."
        ));
    }
    let before = std::fs::read_to_string(&target).unwrap_or_default();
    std::fs::write(&target, new_text).map_err(|e| format!("couldn't write {}: {e}", target.display()))?;
    Ok(FileChange { path: relative_path.to_string(), before, after: new_text.to_string() })
}

/// A run's state as the client polls it. A run takes minutes, longer than an HTTP client should
/// hold one request open, so runs are background jobs rather than one long request.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatus {
    /// "running", "done", or "failed".
    pub state: String,
    pub narration: Vec<String>,
    pub report: Option<Report>,
    pub error: Option<String>,
}

struct Job {
    status: JobStatus,
    cancel: Arc<AtomicBool>,
}

fn jobs() -> &'static Mutex<HashMap<String, Job>> {
    static JOBS: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();
    JOBS.get_or_init(Default::default)
}

fn update(id: &str, change: impl FnOnce(&mut JobStatus)) {
    let mut jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
    if let Some(job) = jobs.get_mut(id) {
        change(&mut job.status);
    }
}

/// Starts a run on its own thread. Only one run at a time: they share the one loaded model, and
/// stacking them only queues GPU work on a laptop that is already warm.
pub fn start_job(engine: EngineState, model_name: String, root: PathBuf, goal: String, files: Vec<String>, id: String) -> Result<(), String> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
        if jobs.values().any(|j| j.status.state == "running") {
            return Err("a folder-agent run is already in progress; wait for it or cancel it".into());
        }
        let status = JobStatus { state: "running".into(), narration: Vec::new(), report: None, error: None };
        jobs.insert(id.clone(), Job { status, cancel: cancel.clone() });
    }
    std::thread::spawn(move || {
        let progress = |narration: &[String]| update(&id, |s| s.narration = narration.to_vec());
        let outcome = run(&engine, &model_name, &root, &goal, &files, &progress, &cancel);
        update(&id, |s| match outcome {
            Ok(report) => {
                s.state = "done".into();
                s.narration = report.narration.clone();
                s.report = Some(report);
            }
            Err(e) => {
                s.state = "failed".into();
                s.error = Some(e);
            }
        });
    });
    Ok(())
}

pub fn job_status(id: &str) -> Option<JobStatus> {
    jobs().lock().unwrap_or_else(|p| p.into_inner()).get(id).map(|j| j.status.clone())
}

/// Asks a run to stop before its next step; the model turn already under way finishes first.
pub fn cancel_job(id: &str) -> bool {
    match jobs().lock().unwrap_or_else(|p| p.into_inner()).get(id) {
        Some(job) => {
            job.cancel.store(true, Ordering::Relaxed);
            true
        }
        None => false,
    }
}

/// The whole folder-agent loop: edit, run, read the result, decide whether to continue, stopping
/// on success, a budget limit, cancellation, or the model declining to propose further changes.
pub fn run(
    engine: &EngineState,
    model_name: &str,
    root: &Path,
    goal: &str,
    files: &[String],
    progress: &dyn Fn(&[String]),
    cancel: &AtomicBool,
) -> Result<Report, String> {
    if !root.is_dir() {
        return Err(format!("{} is not a folder", root.display()));
    }
    let snapshot_commit = snapshot(root)?;
    let mut report = Report {
        stopped_reason: String::new(),
        edits: Vec::new(),
        runs: Vec::new(),
        narration: Vec::new(),
        snapshot_commit: snapshot_commit.clone(),
    };
    let mut budget = Budget::new();
    let mut last_failure = String::new();

    // Names typed as `inference.rs` or `src/inference.rs` mean the one real file ending that way.
    // With no names at all, search the folder for the goal and work on the best matches.
    let mut files: Vec<String> = files
        .iter()
        .map(|f| crate::code_search::snap_path(root, f).unwrap_or_else(|| f.clone()))
        .collect();
    if files.is_empty() {
        files = crate::code_search::best_files(root, goal, FILES_FROM_SEARCH).into_iter().map(|h| h.path).collect();
        if files.is_empty() {
            report.stopped_reason = "no files were given, and searching the folder for the goal found none".to_string();
            return Ok(report);
        }
        report.narration.push(format!("Searched the folder for the goal and picked {}", files.join(", ")));
    }
    let files = files.as_slice();
    let mut listing = String::new();
    for file in files {
        let path = match contained(root, &root.join(file)) {
            Ok(p) => p,
            Err(e) => {
                report.stopped_reason = e;
                return Ok(report);
            }
        };
        let text = std::fs::read_to_string(&path).map_err(|e| format!("couldn't read {file}: {e}"))?;
        listing.push_str(&format!("File `{file}`:\n```{}\n{text}\n```\n\n", fence(file)));
    }

    loop {
        progress(&report.narration);
        if cancel.load(Ordering::Relaxed) {
            report.stopped_reason = "cancelled".to_string();
            break;
        }
        if let Some(reason) = budget.exhausted() {
            report.stopped_reason = reason;
            break;
        }

        if let Some(line) = cool_down(cancel) {
            report.narration.push(line);
        }

        // The goal comes before the files and again after them: a recurrent model reads once, so
        // it should know what it's looking for while it reads (see CODING_STRATEGIES.md). The
        // edit's shape is set by how its reply is started (below), so it isn't described here.
        let question = format!(
            "Goal: {goal}\n\n{listing}{last_failure}Goal: {goal}\n\
             Reply with a line `FILE: <path>` naming the one file above to change. If nothing needs \
             to change, reply `FILE: none` and then `DONE: <one-sentence reason>`."
        );
        let deadline = budget.started + Duration::from_secs(MAX_SECONDS);

        // Two steps, each started for the model so it can only go one way: first the file (G1
        // otherwise answers long code prompts with a <tool_call>), then the edit, begun with the
        // SEARCH marker so the next thing written is lines from that file.
        let path_reply = ask(engine, model_name, &question, "FILE:", &first_line_done, deadline, cancel)?;
        let written = path_reply["FILE:".len()..].lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().trim_matches('`').to_string();
        if written.is_empty() || written.eq_ignore_ascii_case("none") {
            let done = ask(engine, model_name, &question, "FILE: none\nDONE:", &|t: &str| t.ends_with('\n'), deadline, cancel)?;
            let reason = reason_after(&done, "DONE:").filter(|r| !r.is_empty()).unwrap_or_else(|| "no reason given".to_string());
            report.narration.push(format!("Stopped: {reason}"));
            report.stopped_reason = "finished".to_string();
            break;
        }
        let Some(relative_path) = snap_to_list(&written, files) else {
            report.narration.push(format!("Xenon's reply began: {}", excerpt(&path_reply)));
            report.stopped_reason = format!("the model named {written}, which wasn't in the given file list");
            break;
        };
        let current = std::fs::read_to_string(root.join(&relative_path)).unwrap_or_default();
        // Xenon recalls lines it read far back poorly, and given SEARCH/REPLACE it either
        // misremembered a long line (five times in a row) or, shown the lines first, copied them
        // into both halves. What it does well is rewrite lines shown just before its reply. So
        // the agent picks the line that best matches the goal and has Xenon rewrite it and its
        // two neighbours. A few lines on either side are shown as context but never replaced:
        // when they were part of the rewrite, Xenon dropped one of them.
        let file_lines: Vec<&str> = current.lines().collect();
        let Some(best) = crate::code_search::best_line(&current, goal) else {
            report.stopped_reason = format!("couldn't find where in {relative_path} the goal applies");
            break;
        };
        let (from, to) = edit_range(&file_lines, best);
        let target = file_lines[from..to].join("\n");
        let before = file_lines[from.saturating_sub(CONTEXT_LINES)..from].join("\n");
        let after = file_lines[to..(to + CONTEXT_LINES).min(file_lines.len())].join("\n");
        let lang = fence(&relative_path);
        let start = format!(
            "FILE: {relative_path}\nLines {first}-{to} of it are:\n```{lang}\n{target}\n```\n\
             (just before them:\n```{lang}\n{before}\n```\nand just after:\n```{lang}\n{after}\n```)\n\
             With the goal done, lines {first}-{to} become:\n```{lang}\n",
            first = from + 1
        );
        let start_len = start.len();
        let closed = move |t: &str| t.get(start_len..).is_some_and(|rest| rest.contains("```"));
        let reply = ask(engine, model_name, &question, &start, &closed, deadline, cancel)?;
        let Some(written) = reply.get(start_len..).and_then(|rest| rest.split("```").next()) else {
            report.narration.push(format!("Xenon's reply began: {}", excerpt(&reply)));
            report.stopped_reason = "the model's reply had no new lines".to_string();
            break;
        };
        let written = written.trim_end_matches(['\n', '\r']);
        if written.trim() == target.trim() {
            // Small models often copy the lines back unchanged while meaning to change them.
            budget.edits += 1;
            report.narration.push(format!("Xenon's edit to {relative_path} changed nothing; asking again"));
            last_failure = format!(
                "Your last edit to `{relative_path}` wrote these lines back unchanged:\n```{lang}\n{target}\n```\n\
                 This time, change them so the goal is done.\n\n"
            );
            continue;
        }
        let code = replace_lines(&current, from, to, written);
        let why = format!("rewrote lines {}-{to}", from + 1);
        if code.trim() == current.trim() {
            report.narration.push(format!("Stopped: Xenon proposed {relative_path} unchanged"));
            report.stopped_reason = "finished".to_string();
            break;
        }

        let change = match write_file(root, &relative_path, &code) {
            Ok(c) => c,
            Err(e) => {
                report.stopped_reason = e;
                break;
            }
        };
        budget.edits += 1;
        report.narration.push(format!("Edited {relative_path}: {why}"));
        report.edits.push(change.clone());
        progress(&report.narration);

        let Some(check) = check_for(root, &relative_path, files) else {
            report.narration.push(format!("No check runs for {relative_path}'s file type, so stopping after the edit"));
            report.stopped_reason = "finished: edited, nothing to check it with".to_string();
            break;
        };
        let step = match run_check(root, &check, deadline, cancel) {
            Ok(s) => s,
            Err(e) => {
                report.stopped_reason = e;
                break;
            }
        };
        budget.runs += 1;
        let passed = step.exit_code == Some(0);
        let output = format!("{}{}", step.stdout, step.stderr);
        report.runs.push(step);
        if passed {
            report.narration.push(format!("Ran `{}`: passed", check.display()));
            report.stopped_reason = "finished: the tests pass".to_string();
            break;
        }
        // A failed edit is undone, so the next try starts from the original file and a run that
        // runs out of budget never leaves a broken one (one did, before this).
        if let Err(e) = write_file(root, &relative_path, &change.before) {
            report.stopped_reason = e;
            break;
        }
        report.edits.pop();
        report.narration.push(format!("Ran `{}`: failed, so undid the edit", check.display()));
        let tail: String = output.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
        last_failure = format!(
            "Your last edit to `{relative_path}` was:\n```diff\n{}\n```\nIt was undone because `{}` failed with:\n```\n{}\n```\n\n",
            short_diff(&change.before, &change.after),
            check.display(),
            tail.trim()
        );
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_repo() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xenon-folder-agent-test-{}", std::process::id().wrapping_add(rand_suffix())));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q"]).unwrap();
        git(&dir, &["config", "user.email", "test@xenon.local"]).unwrap();
        git(&dir, &["config", "user.name", "xenon-test"]).unwrap();
        fs::write(dir.join("a.py"), "x = 1\n").unwrap();
        git(&dir, &["add", "."]).unwrap();
        git(&dir, &["commit", "-q", "-m", "init"]).unwrap();
        dir
    }

    fn rand_suffix() -> u32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos()
    }

    #[test]
    fn contained_accepts_paths_inside_root() {
        let dir = temp_repo();
        let result = contained(&dir, &dir.join("a.py"));
        assert!(result.is_ok());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn contained_rejects_escaping_paths() {
        let dir = temp_repo();
        let outside = dir.parent().unwrap().join("outside.py");
        fs::write(&outside, "evil = 1\n").unwrap();
        let result = contained(&dir, &dir.join("../outside.py"));
        assert!(result.is_err());
        fs::remove_file(&outside).ok();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn contained_rejects_new_files_outside_root() {
        let dir = temp_repo();
        let result = contained(&dir, &dir.join("../escape.py"));
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_commits_a_clean_tree_and_undo_reverts_it() {
        let dir = temp_repo();
        let before = git(&dir, &["rev-parse", "HEAD"]).unwrap();
        let snap = snapshot(&dir).unwrap();
        assert!(snap.is_some());
        fs::write(dir.join("a.py"), "x = 2\n").unwrap();
        undo(&dir, &snap.unwrap()).unwrap();
        let after = git(&dir, &["rev-parse", "HEAD"]).unwrap();
        assert_eq!(before, after);
        assert_eq!(fs::read_to_string(dir.join("a.py")).unwrap().trim(), "x = 1");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_skips_a_dirty_tree() {
        let dir = temp_repo();
        fs::write(dir.join("a.py"), "x = 99\n").unwrap();
        let snap = snapshot(&dir).unwrap();
        assert!(snap.is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn budget_trips_on_edit_count() {
        let mut budget = Budget::new();
        budget.edits = MAX_EDITS;
        assert!(budget.exhausted().is_some());
    }

    #[test]
    fn budget_trips_on_run_count() {
        let mut budget = Budget::new();
        budget.runs = MAX_RUNS;
        assert!(budget.exhausted().is_some());
    }

    #[test]
    fn reasons_stop_where_the_model_runs_on() {
        let reply = "FILE: a.py\n```python\nx = 2\n```\nWHY: fixed the sign.User: You are a senior engineer";
        assert_eq!(reason_after(reply, "WHY:").as_deref(), Some("fixed the sign."));
        assert_eq!(reason_after("DONE: nothing to do\nUser: more", "DONE:").as_deref(), Some("nothing to do"));
        assert_eq!(reason_after("no tag here", "WHY:"), None);
    }

    #[test]
    fn the_path_step_ends_at_the_end_of_its_line() {
        assert!(!first_line_done("FILE:"));
        assert!(!first_line_done("FILE:\n"));
        assert!(!first_line_done("FILE: src/a.rs"));
        assert!(first_line_done("FILE: src/a.rs\n"));
        assert!(first_line_done("FILE:\nsrc/a.rs\n"));
    }

    #[test]
    fn the_edit_range_reaches_into_a_definition_and_skips_blank_edges() {
        let lines = ["x = 1", "", "def last_n(items, n):", "    \"\"\"The last n.\"\"\"", "    return items[-n - 1:]", "", ""];
        assert_eq!(edit_range(&lines, 2), (2, 5), "a def takes its body, not the blank lines after");
        let lines = ["A = 1", "", "B = 2", "", "C = 3"];
        assert_eq!(edit_range(&lines, 2), (2, 3), "blank neighbours stay out");
        assert_eq!(edit_range(&["A = 1", "B = 2", "C = 3"], 1), (0, 3));
    }

    #[test]
    fn replacing_lines_keeps_the_rest_and_the_line_endings() {
        assert_eq!(replace_lines("a\nb\nc\nd\n", 1, 3, "B\nC2\nC3"), "a\nB\nC2\nC3\nd\n");
        assert_eq!(replace_lines("a\r\nb\r\nc\r\n", 1, 2, "x"), "a\r\nx\r\nc\r\n");
        assert_eq!(replace_lines("a\nb", 0, 1, ""), "b");
    }

    #[test]
    fn a_short_diff_shows_the_change_with_two_lines_around_it() {
        let before = "a\nb\nc\nd\ne\nf\n";
        let after = "a\nb\nc\nD\ne\nf\n";
        assert_eq!(short_diff(before, after), " b\n c\n-d\n+D\n e\n f");
        assert_eq!(short_diff("x\n", "x\ny\n"), " x\n+y");
    }

    #[test]
    fn unknown_runs_have_no_status_and_cannot_be_cancelled() {
        assert!(job_status("run-missing").is_none());
        assert!(!cancel_job("run-missing"));
    }

    #[test]
    fn write_file_rejects_paths_outside_root() {
        let dir = temp_repo();
        let result = write_file(&dir, "../escape.py", "evil = 1\n");
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_ignores_untracked_files() {
        let dir = temp_repo();
        fs::write(dir.join("scratch.txt"), "notes\n").unwrap();
        assert!(snapshot(&dir).unwrap().is_some());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn picks_a_check_for_each_language() {
        let dir = temp_repo();
        fs::create_dir_all(dir.join("crate/src")).unwrap();
        fs::write(dir.join("crate/Cargo.toml"), "[package]\n").unwrap();
        fs::write(dir.join("crate/src/lib.rs"), "").unwrap();
        fs::create_dir_all(dir.join("ext/src")).unwrap();
        fs::write(dir.join("ext/package.json"), r#"{"scripts":{"test":"node --test"}}"#).unwrap();
        fs::create_dir_all(dir.join("ui/src")).unwrap();
        fs::write(dir.join("ui/package.json"), r#"{"devDependencies":{"vue-tsc":"^2"}}"#).unwrap();
        let files = vec!["a.py".to_string(), "test_a.py".to_string()];

        let check = |path: &str| check_for(&dir, path, &files).map(|c| c.display());
        assert_eq!(check("a.py").as_deref(), Some("python -m pytest test_a.py"));
        assert_eq!(check("crate/src/lib.rs").as_deref(), Some("cargo test --lib (in crate)"));
        assert_eq!(check("ext/src/client.ts").as_deref(), Some("npm test (in ext)"));
        assert_eq!(check("ui/src/App.vue").as_deref(), Some("npx vue-tsc --noEmit (in ui)"));
        assert_eq!(check("README.md"), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_check_reports_failure_and_output() {
        let dir = temp_repo();
        let check = Check { program: "git", args: vec!["no-such-command".into()], dir: String::new() };
        let step = run_check(&dir, &check, Instant::now() + Duration::from_secs(30), &AtomicBool::new(false)).unwrap();
        assert_ne!(step.exit_code, Some(0));
        assert!(step.stderr.contains("no-such-command"), "{}", step.stderr);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_shortened_file_name_snaps_to_the_list() {
        let files = vec!["app/src/x.rs".to_string(), "lib/y.py".to_string(), "a/y.py".to_string()];
        assert_eq!(snap_to_list("app/src/x.rs", &files).as_deref(), Some("app/src/x.rs"));
        assert_eq!(snap_to_list("src/x.rs", &files).as_deref(), Some("app/src/x.rs"));
        assert_eq!(snap_to_list("/x.rs", &files).as_deref(), Some("app/src/x.rs"));
        assert_eq!(snap_to_list("y.py", &files), None, "two files end in y.py");
        assert_eq!(snap_to_list("z.rs", &files), None);
    }

    #[test]
    fn listings_fence_code_in_its_own_language() {
        assert_eq!(fence("app/src-tauri/src/lib.rs"), "rust");
        assert_eq!(fence("vscode-xenon/src/client.ts"), "typescript");
        assert_eq!(fence("training/eval_code.py"), "python");
    }

    #[test]
    fn debug_builds_leave_their_own_rust_code_alone() {
        assert!(restarts_this_app(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")));
        assert!(!restarts_this_app(&std::env::temp_dir().join("elsewhere.rs")));
    }
}
