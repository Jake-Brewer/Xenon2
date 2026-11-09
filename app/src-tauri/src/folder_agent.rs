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
use crate::edit_blocks;
use crate::inference::{self, EngineState};

pub const MAX_EDITS: u32 = 5;
pub const MAX_RUNS: u32 = 3;
pub const MAX_SECONDS: u64 = 10 * 60;
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

/// True once a reply has its closing `WHY:`/`DONE:` line, so generation can stop there instead
/// of running on into a made-up next turn (which only costs GPU time and heat).
fn reply_complete(text: &str) -> bool {
    ["WHY:", "DONE:"].iter().any(|tag| {
        text.find(tag).is_some_and(|i| {
            let rest = &text[i..];
            rest.contains('\n') || rest.contains("User:")
        })
    })
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
/// answering a long code prompt with a `<tool_call>` instead of an edit.
fn ask(engine: &EngineState, model_name: &str, question: &str, start: &str, deadline: Instant, cancel: &AtomicBool) -> Result<String, String> {
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
    let mut prompt = code_api::build_prompt(&request, style)?;
    prompt.last.push_str(start);
    let mut text = start.to_string();
    let mut sink = |piece: &str| {
        text.push_str(piece);
        !reply_complete(&text) && Instant::now() < deadline && !cancel.load(Ordering::Relaxed)
    };
    inference::generate_code(engine, &prompt.prior, &prompt.last, prompt.params, &mut sink)?;
    Ok(text.trim().to_string())
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
        // it should know what it's looking for while it reads (see CODING_STRATEGIES.md).
        let question = format!(
            "Goal: {goal}\n\n{listing}{last_failure}Goal: {goal}\n\
             Reply in this exact shape: a line `FILE: <path>` naming one file above, then one or \
             more edit blocks, each copying a few lines exactly from that file and giving their \
             replacement:\n\
             <<<<<<< SEARCH\n    let x = 1;\n=======\n    let x = 2;\n>>>>>>> REPLACE\n\
             then a line starting with `WHY:` giving a one-sentence reason. If the goal is already \
             met and nothing needs to change, reply with exactly `DONE: <one-sentence reason>` instead."
        );
        let deadline = budget.started + Duration::from_secs(MAX_SECONDS);
        let reply = ask(engine, model_name, &question, "FILE:", deadline, cancel)?;

        // With the reply started as `FILE:`, "nothing to do" shows up as a DONE: line and no edit.
        if edit_blocks::parse(&reply).is_empty() && extract_code_block(&reply).is_none() {
            if let Some(reason) = reason_after(&reply, "DONE:") {
                report.narration.push(format!("Stopped: {reason}"));
                report.stopped_reason = "finished".to_string();
                break;
            }
        }

        let Some(file_line) = reply.lines().find(|l| l.trim_start().starts_with("FILE:")) else {
            report.narration.push(format!("Xenon's reply began: {}", excerpt(&reply)));
            report.stopped_reason = "the model's reply didn't name a file to edit".to_string();
            break;
        };
        let file_line = file_line.trim_start();
        let written = file_line.trim_start_matches("FILE:").trim().trim_matches('`').to_string();
        let Some(relative_path) = snap_to_list(&written, files) else {
            report.stopped_reason = format!("the model named {written}, which wasn't in the given file list");
            break;
        };
        let current = std::fs::read_to_string(root.join(&relative_path)).unwrap_or_default();
        // Edit blocks, or failing those a whole new file in a code block.
        let blocks = edit_blocks::parse(&reply);
        if !blocks.is_empty() && blocks.iter().all(|b| b.search.trim() == b.replace.trim()) {
            // Small models often copy SEARCH into REPLACE while saying they changed it.
            budget.edits += 1;
            report.narration.push(format!("Xenon's edit to {relative_path} changed nothing; asking again"));
            last_failure = format!(
                "In your last reply the lines after `=======` were the same as the SEARCH lines, so nothing \
                 changed. This time, after `=======`, write the lines as they should be after the fix.\n\n"
            );
            continue;
        }
        let code = if !blocks.is_empty() {
            match edit_blocks::apply_all(&current, &blocks) {
                Ok(code) => code,
                Err(e) => {
                    // A failed edit still costs one from the budget, so a model that keeps
                    // missing can't loop forever; the reason goes back to it.
                    budget.edits += 1;
                    report.narration.push(format!("An edit to {relative_path} didn't apply: {e}"));
                    last_failure = format!("Your last edit to `{relative_path}` didn't apply: {e}\n\n");
                    continue;
                }
            }
        } else if let Some(code) = extract_code_block(&reply) {
            code
        } else {
            report.narration.push(format!("Xenon's reply began: {}", excerpt(&reply)));
            report.stopped_reason = "the model's reply had no edit blocks or code block".to_string();
            break;
        };
        let why = reason_after(&reply, "WHY:").filter(|r| !r.is_empty()).unwrap_or_else(|| "no reason given".to_string());
        if code.trim() == current.trim() {
            report.narration.push(format!("Stopped: Xenon proposed {relative_path} unchanged ({why})"));
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
        report.narration.push(format!(
            "Ran `{}`: {}",
            step.command,
            if passed { "passed" } else { "failed" }
        ));
        let output = format!("{}{}", step.stdout, step.stderr);
        report.runs.push(step);
        if passed {
            report.stopped_reason = "finished: the tests pass".to_string();
            break;
        }
        let tail: String = output.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
        last_failure = format!("After your last edit, `{}` failed with:\n```\n{}\n```\n\n", check.display(), tail.trim());

        let lang = fence(&relative_path);
        listing = listing.replace(
            &format!("File `{relative_path}`:\n```{lang}\n{}", change.before),
            &format!("File `{relative_path}`:\n```{lang}\n{}", change.after),
        );
    }

    Ok(report)
}

fn extract_code_block(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_fence = &text[start + 3..];
    let body_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
    let body = &after_fence[body_start..];
    let end = body.find("```")?;
    Some(body[..end].trim_end().to_string())
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
    fn extract_code_block_reads_the_first_fenced_block() {
        let reply = "FILE: a.py\n```python\nx = 2\n```\nWHY: fixed it\n";
        assert_eq!(extract_code_block(reply), Some("x = 2".to_string()));
    }

    #[test]
    fn extract_code_block_is_none_without_a_fence() {
        assert_eq!(extract_code_block("no code here"), None);
    }

    #[test]
    fn reasons_stop_where_the_model_runs_on() {
        let reply = "FILE: a.py\n```python\nx = 2\n```\nWHY: fixed the sign.User: You are a senior engineer";
        assert_eq!(reason_after(reply, "WHY:").as_deref(), Some("fixed the sign."));
        assert_eq!(reason_after("DONE: nothing to do\nUser: more", "DONE:").as_deref(), Some("nothing to do"));
        assert_eq!(reason_after("no tag here", "WHY:"), None);
    }

    #[test]
    fn generation_stops_once_the_reason_line_ends() {
        assert!(!reply_complete("FILE: a.py\n```python\nx = 2\n```\nWHY: fixed"));
        assert!(reply_complete("FILE: a.py\n```python\nx = 2\n```\nWHY: fixed it.\n"));
        assert!(reply_complete("WHY: fixed it.User:"));
        assert!(reply_complete("DONE: already passes\n"));
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
