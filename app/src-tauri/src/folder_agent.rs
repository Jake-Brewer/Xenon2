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
/// The most steps a goal is broken into.
const MAX_STEPS: usize = 5;
/// Tries at one place in a file before moving on to the next-best place.
const TRIES_PER_PLACE: usize = 2;
/// Lines Xenon rewrites from a definition down (its signature and the start of its body).
const EDIT_LINES: usize = 4;
/// Files to work on when none are named: few, since each is read whole into every prompt.
const FILES_FROM_SEARCH: usize = 3;

const SNAPSHOT_MESSAGE: &str = "xenon-agent: snapshot before run";

/// The RTX 2060 here is in a laptop: before each model turn, wait for a hot GPU to cool (see
/// `gpu_temp.rs`, which also throttles between tokens).
use crate::gpu_temp::{COOL_C, HOT_C};
const MAX_COOL_WAIT: Duration = Duration::from_secs(5 * 60);
/// How often a cool-down re-reads the temperature (quick in tests).
const COOL_POLL: Duration = if cfg!(test) { Duration::from_millis(5) } else { Duration::from_secs(5) };

/// Waits out a hot GPU. Returns a narration line when it had to wait.
fn cool_down(model: &dyn Model, cancel: &AtomicBool) -> Option<String> {
    let hot = model.gpu_temp().filter(|t| *t >= HOT_C)?;
    let started = Instant::now();
    while started.elapsed() < MAX_COOL_WAIT && !cancel.load(Ordering::Relaxed) {
        std::thread::sleep(COOL_POLL);
        if model.gpu_temp().map_or(true, |t| t <= COOL_C) {
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Report {
    /// Whether the run stopped because it finished, or hit a budget limit.
    pub stopped_reason: String,
    pub edits: Vec<FileChange>,
    pub runs: Vec<RunStep>,
    /// One line per step explaining why the agent did it, in order taken.
    pub narration: Vec<String>,
    /// The snapshot commit to revert to, if one was taken (the tree was clean at the start).
    pub snapshot_commit: Option<String>,
    /// Mistakes this run fixed, for the next training batch. Not sent to clients: prompts are large.
    #[serde(skip)]
    pub lessons: Vec<Lesson>,
}

/// A mistake Xenon later fixed: the prompt of an edit that failed its checks, and the edit that
/// passed at the same lines. Never a failed edit as the answer: Xenon learns the repair.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Lesson {
    pub prompt: String,
    pub answer: String,
    pub passed: bool,
    pub source: &'static str,
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

/// cargo, from PATH or else rustup's toolchains: on Windows rustup often leaves it off PATH, and an
/// app started from the desktop doesn't get a shell's PATH anyway.
fn find_cargo() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "cargo.exe" } else { "cargo" };
    let on_path = std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(exe)).find(|p| p.is_file());
    on_path.or_else(|| {
        let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
        let home = PathBuf::from(home);
        let cargo_bin = home.join(".cargo").join("bin").join(exe);
        if cargo_bin.is_file() {
            return Some(cargo_bin);
        }
        let mut toolchains: Vec<PathBuf> = std::fs::read_dir(home.join(".rustup").join("toolchains"))
            .ok()?
            .flatten()
            .map(|e| e.path().join("bin").join(exe))
            .filter(|p| p.is_file())
            .collect();
        // Prefer stable when several toolchains are installed.
        toolchains.sort_by_key(|p| !p.to_string_lossy().contains("stable"));
        toolchains.into_iter().next()
    })
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
    let (prior, last, params) = ask_prompt(model_name, question, start)?;
    let mut text = start.to_string();
    let mut sink = |piece: &str| {
        text.push_str(piece);
        !done(&text) && Instant::now() < deadline && !cancel.load(Ordering::Relaxed)
    };
    inference::generate_code(engine, &prior, &last, params, &mut sink)?;
    Ok(text.trim().to_string())
}

/// The prompt `ask` gives the model, as (cached part, the rest, sampling settings). The two
/// strings together are exactly the text the model reads before its reply.
fn ask_prompt(model_name: &str, question: &str, start: &str) -> Result<(String, String, inference::SamplingParams), String> {
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
    Ok((prior, last, prompt.params))
}

/// What the agent's loop asks things of: the model, and the GPU it runs on. The app's engine in
/// use; a scripted stand-in in tests, so the loop (undoing failed edits, the budget, lessons, the
/// fall back to steps) can be tested without a GPU. Mutation testing found none of it was.
trait Model {
    /// Xenon's reply to `question`, begun with `start` (which comes back at its front), ending
    /// once `done` says the text is complete. As `ask`.
    fn ask(&self, question: &str, start: &str, done: &dyn Fn(&str) -> bool, deadline: Instant, cancel: &AtomicBool) -> Result<String, String>;
    /// Exactly the text the model reads before its reply to `question` begun with `start`.
    fn prompt_text(&self, question: &str, start: &str) -> String;
    /// The GPU's temperature now, if there is one to read.
    fn gpu_temp(&self) -> Option<u32>;
}

/// The app's loaded model.
struct Engine<'a> {
    engine: &'a EngineState,
    model_name: &'a str,
}

impl Model for Engine<'_> {
    fn ask(&self, question: &str, start: &str, done: &dyn Fn(&str) -> bool, deadline: Instant, cancel: &AtomicBool) -> Result<String, String> {
        ask(self.engine, self.model_name, question, start, done, deadline, cancel)
    }
    fn prompt_text(&self, question: &str, start: &str) -> String {
        ask_prompt(self.model_name, question, start).map(|(a, b, _)| a + &b).unwrap_or_default()
    }
    fn gpu_temp(&self) -> Option<u32> {
        crate::gpu_temp::read()
    }
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
            // The edited file's own test beside it, else a test among the files, else the file.
            let (dir, name) = edited.rsplit_once('/').map_or(("", edited), |(d, n)| (d, n));
            let stem = name.trim_end_matches(".py");
            let beside = [format!("test_{stem}.py"), format!("{stem}_test.py")]
                .into_iter()
                .map(|n| if dir.is_empty() { n } else { format!("{dir}/{n}") })
                .find(|p| root.join(p).is_file());
            let target = if is_python_test(edited) {
                edited.to_string()
            } else {
                beside
                    .or_else(|| files.iter().find(|f| is_python_test(f)).cloned())
                    .unwrap_or_else(|| edited.to_string())
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
    let dir = root.join(&check.dir);
    let mut cmd = match (check.program, find_cargo()) {
        ("cargo", Some(cargo)) => Command::new(cargo),
        _ => command(check.program),
    };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW, as in `command`
    }
    if check.program == "cargo" {
        // Crates that load DLLs at test time (this app's engine) find them in target/debug.
        let mut path = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).collect::<Vec<_>>();
        path.insert(0, dir.join("target").join("debug"));
        if let Ok(joined) = std::env::join_paths(path) {
            cmd.env("PATH", joined);
        }
    }
    let mut child = cmd
        .args(&check.args)
        .current_dir(&dir)
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

/// True when the goal says a test changes along with the code ("Update the code and its test").
/// Only then may a failing test be rewritten; otherwise a failing test means the code is wrong.
fn wants_test_update(goal: &str) -> bool {
    goal.to_lowercase().contains("test")
}

/// The test file and line a check's failure points at: pytest's `test_x.py:12:` location.
fn failing_test_line(failure: &str) -> Option<(String, usize)> {
    failure.lines().rev().find_map(|line| {
        let line = line.trim().replace('\\', "/");
        let (path, rest) = line.split_once(".py:")?;
        let path = format!("{}.py", path.rsplit([' ', '"', '(']).next().unwrap_or(path));
        let number: usize = rest.split(':').next()?.trim().parse().ok()?;
        is_python_test(&path).then_some((path, number))
    })
}

/// The Test-writer: rewrites the assertion a failing check points at so it expects what `goal`
/// asks for. Returns the change, already written, or `None` when there's no test line to change,
/// Xenon wrote nothing new, or the rewrite would drop an assertion (loosening a test until broken
/// code passes is the classic way to "fix" one).
fn update_failing_test(ctx: &RunContext, goal: &str, failure: &str, report: &mut Report) -> Result<Option<FileChange>, String> {
    let Some((test_path, number)) = failing_test_line(failure) else { return Ok(None) };
    let Ok(text) = std::fs::read_to_string(ctx.root.join(&test_path)) else { return Ok(None) };
    let Some(line) = text.lines().nth(number.saturating_sub(1)) else { return Ok(None) };
    let files = vec![test_path.clone()];
    let test_ctx = RunContext { files: &files, searched: false, ..*ctx };
    let test_goal = format!("{goal} The code is already changed; update this test in {test_path} so it expects the new behavior.");
    let focus = format!("{test_path} {line}");
    let mut tried = HashMap::new();
    let Attempt::Edit { path, current, code, .. } = attempt(&test_ctx, &test_goal, &focus, failure, &mut tried, report)? else {
        return Ok(None);
    };
    if code.matches("assert").count() < current.matches("assert").count() {
        report.narration.push(format!("Refused a change to {path} that removed an assertion"));
        return Ok(None);
    }
    write_file(ctx.root, &path, &code).map(Some)
}

/// The one file in `files` that `text` names, by path or file name, as a whole word (so
/// `shop.py` isn't found in `test_shop.py`). `None` when it names none of them, or several.
fn file_named_in(text: &str, files: &[String]) -> Option<String> {
    let names = |file: &str| -> bool {
        let base = file.rsplit('/').next().unwrap_or(file);
        [file, base].iter().any(|name| {
            text.match_indices(name).any(|(at, _)| {
                let before = text[..at].chars().next_back();
                let after = text[at + name.len()..].chars().next();
                let edge = |c: Option<char>| c.map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '/'));
                edge(before) && edge(after)
            })
        })
    };
    let mut found = files.iter().filter(|f| names(f));
    let first = found.next()?;
    found.next().is_none().then(|| first.clone())
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

/// The compiler warnings in a check's output, each as its message plus where it points (rustc's
/// `warning: ...` line and its ` --> file:line` line), without line numbers, so a warning that
/// only moved because lines were added above it still counts as the same one. Summary lines
/// ("`app` (lib) generated 1 warning") are left out.
fn warnings(output: &str) -> Vec<String> {
    let lines: Vec<&str> = output.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(message) = line.strip_prefix("warning: ") else { continue };
        if message.contains("generated") && message.starts_with('`') {
            continue;
        }
        let place = lines
            .get(i + 1)
            .and_then(|l| l.trim_start().strip_prefix("--> "))
            .map(|p| p.split(':').next().unwrap_or(p).replace('\\', "/"))
            .unwrap_or_default();
        out.push(format!("{message} ({place})"));
    }
    out
}

/// True for goals like "silence the dead_code warning on X" or "fix the unused-import warnings".
fn asks_to_remove_a_warning(goal: &str) -> bool {
    let goal = goal.to_lowercase();
    goal.contains("warning") && ["silence", "remove", "fix", "get rid of", "clear"].iter().any(|v| goal.contains(v))
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
/// `lessons_file` (the app's `mistakes.jsonl`) gets one line per mistake the run fixed, for the
/// next training batch (training/build_batch.py).
pub fn start_job(
    engine: EngineState,
    model_name: String,
    root: PathBuf,
    goal: String,
    files: Vec<String>,
    id: String,
    lessons_file: Option<PathBuf>,
) -> Result<(), String> {
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
        if let (Ok(report), Some(path)) = (&outcome, &lessons_file) {
            if let Err(e) = save_lessons(path, &report.lessons) {
                eprintln!("xenon2: couldn't save the run's lessons: {e}");
            }
        }
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

/// Appends lessons to `path` as JSON lines.
fn save_lessons(path: &Path, lessons: &[Lesson]) -> Result<(), String> {
    use std::io::Write;
    if lessons.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| e.to_string())?;
    for lesson in lessons {
        let line = serde_json::to_string(lesson).map_err(|e| e.to_string())?;
        writeln!(file, "{line}").map_err(|e| e.to_string())?;
    }
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
/// What stays the same for every attempt in a run.
struct RunContext<'a> {
    model: &'a dyn Model,
    root: &'a Path,
    files: &'a [String],
    /// The files came from search, so Xenon may name another real file in the folder.
    searched: bool,
    listing: &'a str,
    deadline: Instant,
    cancel: &'a AtomicBool,
}

/// What one attempt at an edit came to.
enum Attempt {
    /// A rewrite of lines `from..to` of `path`: `code` is the whole file afterwards.
    Edit { path: String, from: usize, to: usize, written: String, current: String, code: String, clean_prompt: String },
    /// Xenon wrote the lines back unchanged.
    Unchanged { path: String, from: usize, to: usize, target: String, clean_prompt: String },
    /// Xenon said nothing needs to change, with its reason.
    Done(String),
    /// The run can't go on, with the reason.
    Stop(String),
}

/// One edit for `goal`: Xenon picks the file, the agent picks the lines, Xenon rewrites them.
///
/// `focus` is what the agent searches the file for to pick the lines; `goal` is what Xenon is
/// told. They differ for a step, whose prompt carries the whole goal for context but whose lines
/// must be found by the step alone: searching for the whole goal's words put the step on another
/// step's lines.
fn attempt(
    ctx: &RunContext,
    goal: &str,
    focus: &str,
    last_failure: &str,
    tried: &mut HashMap<String, Vec<(usize, usize)>>,
    report: &mut Report,
) -> Result<Attempt, String> {
    let RunContext { model, root, files, searched, listing, deadline, cancel } = *ctx;
    // The goal comes before the files and again after them: a recurrent model reads once, so
    // it should know what it's looking for while it reads (see CODING_STRATEGIES.md). The
    // edit's shape is set by how its reply is started (below), so it isn't described here.
    let question = format!(
        "Goal: {goal}\n\n{listing}{last_failure}Goal: {goal}\n\
         Reply with a line `FILE: <path>` naming the one file above to change. If nothing needs \
         to change, reply `FILE: none` and then `DONE: <one-sentence reason>`."
    );

    // The reply is started as `FILE:` so it can only be a path (G1 otherwise answers long code
    // prompts with a <tool_call>).
    // A goal or step that names one of the files outright needs no model call to find it: asked
    // anyway, Xenon named shop.py for "Update the import in report.py".
    let path = if let Some(path) = file_named_in(focus, files) {
        path
    } else {
        let path_reply = model.ask(&question, "FILE:", &first_line_done, deadline, cancel)?;
        let written = path_reply["FILE:".len()..].lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().trim_matches('`').to_string();
        if written.is_empty() || written.eq_ignore_ascii_case("none") {
            let done = model.ask(&question, "FILE: none\nDONE:", &|t: &str| t.ends_with('\n'), deadline, cancel)?;
            let reason = reason_after(&done, "DONE:").filter(|r| !r.is_empty()).unwrap_or_else(|| "no reason given".to_string());
            return Ok(Attempt::Done(reason));
        }
        // Files named by the caller are the limit. Files picked by search are only a guess, so
        // Xenon may name another real file in the folder (search had missed walker.py where
        // other files repeated the goal's words; Xenon named it anyway).
        let named = snap_to_list(&written, files).or_else(|| {
            searched
                .then(|| crate::code_search::snap_path(root, &written))
                .flatten()
                .filter(|p| contained(root, &root.join(p)).is_ok())
        });
        let Some(path) = named else {
            report.narration.push(format!("Xenon's reply began: {}", excerpt(&path_reply)));
            return Ok(Attempt::Stop(format!("the model named {written}, which wasn't in the given file list")));
        };
        path
    };
    let current = std::fs::read_to_string(root.join(&path)).unwrap_or_default();
    // Xenon recalls lines it read far back poorly, and given SEARCH/REPLACE it either
    // misremembered a long line (five times in a row) or, shown the lines first, copied them
    // into both halves. What it does well is rewrite lines shown just before its reply. So
    // the agent picks the line that best matches the goal and has Xenon rewrite it and its
    // neighbours. A few lines on either side are shown as context but never replaced: when
    // they were part of the rewrite, Xenon dropped one of them.
    let file_lines: Vec<&str> = current.lines().collect();
    // Each place gets two tries, then the next-best place: retrying the same lines got the
    // same wrong edit three times running, but a second try with the failure in front of it
    // often fixes the first.
    let tried_here = tried.entry(path.clone()).or_default();
    let Some((from, to)) = crate::code_search::ranked_lines(&current, focus)
        .into_iter()
        .map(|line| edit_range(&file_lines, line))
        .find(|range| tried_here.iter().filter(|r| *r == range).count() < TRIES_PER_PLACE)
    else {
        return Ok(Attempt::Stop(format!("tried every place in {path} that matches the goal")));
    };
    tried_here.push((from, to));
    let target = file_lines[from..to].join("\n");
    let before = file_lines[from.saturating_sub(CONTEXT_LINES)..from].join("\n");
    let after = file_lines[to..(to + CONTEXT_LINES).min(file_lines.len())].join("\n");
    let lang = fence(&path);
    // A short question of its own, without the file listing: with three large files read
    // first, Xenon dropped entries from a line it was asked to add one to. But the lines to
    // change stay in the reply's start, right before "become:": with them in the question
    // instead, Xenon passed 2 of 7 edit tasks rather than 6 of 6.
    let question_with = |failure: &str| {
        format!(
            "Goal: {goal}\n\n{failure}In `{path}`, the lines just before the ones to change:\n```{lang}\n{before}\n```\n\
             and just after them:\n```{lang}\n{after}\n```\nGoal: {goal}"
        )
    };
    let start = format!(
        "Lines {first}-{to} of `{path}` are:\n```{lang}\n{target}\n```\nWith the goal done, they become:\n```{lang}\n",
        first = from + 1
    );
    let start_len = start.len();
    let closed = move |t: &str| t.get(start_len..).is_some_and(|rest| rest.contains("```"));
    let reply = model.ask(&question_with(last_failure), &start, &closed, deadline, cancel)?;
    let Some(written) = reply.get(start_len..).and_then(|rest| rest.split("```").next()) else {
        report.narration.push(format!("Xenon's reply began: {}", excerpt(&reply)));
        return Ok(Attempt::Stop("the model's reply had no new lines".to_string()));
    };
    let written = written.trim_end_matches(['\n', '\r']).to_string();
    // The prompt this place gets without any failure feedback: the one a lesson pairs with the
    // edit that finally passes here.
    let clean_prompt = model.prompt_text(&question_with(""), &start);
    if written.trim() == target.trim() {
        return Ok(Attempt::Unchanged { path, from, to, target, clean_prompt });
    }
    let code = replace_lines(&current, from, to, &written);
    Ok(Attempt::Edit { path, from, to, written, current, code, clean_prompt })
}

/// Asks Xenon to break `goal` into numbered steps, each one change in one place. The reply is
/// started as "1." so it can only be a list. At most `MAX_STEPS` come back.
fn plan(ctx: &RunContext, goal: &str, last_failure: &str) -> Result<Vec<String>, String> {
    let question = format!(
        "Goal: {goal}\n\n{}{last_failure}Goal: {goal}\n\
         Break the goal into steps. Each step is one change in one place in one of the files above, \
         and names that file. Number the steps. If one change does it, write one step.",
        ctx.listing
    );
    let done = |t: &str| t.len() > 1500 || t.contains("\n\n") || t.contains("User:");
    let reply = ctx.model.ask(&question, "1.", &done, ctx.deadline, ctx.cancel)?;
    Ok(parse_steps(&reply))
}

/// The steps of a numbered list ("1. ...", "2) ..."), in order, at most `MAX_STEPS`.
fn parse_steps(reply: &str) -> Vec<String> {
    reply
        .lines()
        .map(str::trim)
        .take_while(|l| !l.starts_with("User:"))
        .filter_map(|l| {
            let digits = l.chars().take_while(char::is_ascii_digit).count();
            let rest = l.get(digits..)?;
            (digits > 0 && (rest.starts_with('.') || rest.starts_with(')'))).then(|| rest[1..].trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .take(MAX_STEPS)
        .collect()
}

/// Runs `check` once on the untouched code if it hasn't been yet, recording its warnings.
fn ensure_baseline(ctx: &RunContext, check: &Check, baselines: &mut HashMap<String, Vec<String>>, report: &mut Report) -> Result<(), String> {
    let key = check.display();
    if baselines.contains_key(&key) {
        return Ok(());
    }
    let step = run_check(ctx.root, check, ctx.deadline, ctx.cancel)?;
    // Failing already is often the point ("make the failing test pass"); the run succeeds
    // only once the whole check passes, so it just goes in the report.
    if step.exit_code != Some(0) {
        report.narration.push(format!("`{key}` fails before any edit"));
    }
    baselines.insert(key, warnings(&format!("{}{}", step.stdout, step.stderr)));
    Ok(())
}

/// Runs `check` after an edit. Ok(true) when it passes with no new warnings (and, for a goal to
/// remove a warning, fewer of them); otherwise Ok(false) with the reason in `failure`.
fn passes(
    ctx: &RunContext,
    check: &Check,
    goal: &str,
    baselines: &HashMap<String, Vec<String>>,
    report: &mut Report,
    failure: &mut String,
) -> Result<bool, String> {
    let step = run_check(ctx.root, check, ctx.deadline, ctx.cancel)?;
    let mut output = format!("{}{}", step.stdout, step.stderr);
    let before_warnings = baselines.get(&check.display()).cloned().unwrap_or_default();
    let after_warnings = warnings(&output);
    let new_warnings: Vec<String> = after_warnings.iter().filter(|w| !before_warnings.contains(w)).cloned().collect();
    // A goal to remove a warning is only met when there are fewer of them afterwards.
    let warning_stayed = asks_to_remove_a_warning(goal) && after_warnings.len() >= before_warnings.len();
    let passed = step.exit_code == Some(0) && new_warnings.is_empty() && !warning_stayed;
    if step.exit_code == Some(0) && !new_warnings.is_empty() {
        output = format!("The tests passed, but the edit added compiler warnings:\n{}", new_warnings.join("\n"));
    } else if step.exit_code == Some(0) && warning_stayed {
        output = format!("The tests passed, but the warnings the goal is about are still there:\n{}", after_warnings.join("\n"));
    }
    report.runs.push(step);
    report.narration.push(format!("Ran `{}`: {}", check.display(), if passed { "passed" } else { "failed" }));
    let tail: String = output.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
    *failure = format!("`{}` failed with:\n```\n{}\n```\n", check.display(), tail.trim());
    Ok(passed)
}

pub fn run(
    engine: &EngineState,
    model_name: &str,
    root: &Path,
    goal: &str,
    files: &[String],
    progress: &dyn Fn(&[String]),
    cancel: &AtomicBool,
) -> Result<Report, String> {
    run_with(&Engine { engine, model_name }, root, goal, files, progress, cancel)
}

/// `run`, with the model passed in (a scripted stand-in in tests).
fn run_with(
    model: &dyn Model,
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
        lessons: Vec::new(),
    };

    // Names typed as `inference.rs` or `src/inference.rs` mean the one real file ending that way.
    // With no names at all, search the folder for the goal and work on the best matches.
    let mut files: Vec<String> = files
        .iter()
        .map(|f| crate::code_search::snap_path(root, f).unwrap_or_else(|| f.clone()))
        .collect();
    let searched = files.is_empty();
    if searched {
        files = crate::code_search::best_files(root, goal, FILES_FROM_SEARCH).into_iter().map(|h| h.path).collect();
        if files.is_empty() {
            report.stopped_reason = "no files were given, and searching the folder for the goal found none".to_string();
            return Ok(report);
        }
        report.narration.push(format!("Searched the folder for the goal and picked {}", files.join(", ")));
    }
    let mut listing = String::new();
    for file in &files {
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
    let budget_started = Instant::now();
    let ctx = RunContext {
        model,
        root,
        files: &files,
        searched,
        listing: &listing,
        deadline: budget_started + Duration::from_secs(MAX_SECONDS),
        cancel,
    };

    // Try the goal as a single sequence of edits first. The one-edit retry loop is more robust
    // than multi-step mode for a small model (which over-decomposes simple goals), so steps are a
    // fallback, used only when one edit at a time can't finish (a rename whose callers break until
    // a later edit, say). A run that finished, was cancelled, or hit the time limit is not retried.
    run_one(&ctx, goal, &mut report, progress)?;
    if report.stopped_reason.starts_with("finished")
        || report.stopped_reason == "cancelled"
        || ctx.cancel.load(Ordering::Relaxed)
        || ctx.deadline <= Instant::now()
    {
        return Ok(report);
    }
    let steps = plan(&ctx, goal, "")?;
    if steps.len() > 1 {
        report.narration.push(format!("One edit at a time didn't finish; broke the goal into {} steps: {}", steps.len(), steps.join(" / ")));
        run_steps(&ctx, goal, steps, &mut report, progress)?;
    }
    Ok(report)
}

/// The one-edit loop: edit, check, and on failure undo and try again with the failure shown.
fn run_one(ctx: &RunContext, goal: &str, report: &mut Report, progress: &dyn Fn(&[String])) -> Result<(), String> {
    let mut budget = Budget::new();
    let mut last_failure = String::new();
    // Per check command: the compiler warnings it gave before any edit.
    let mut baselines: HashMap<String, Vec<String>> = HashMap::new();
    // Per file: the line ranges already rewritten.
    let mut tried: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    // Per place (file and line range): the clean prompt of its first failed edit there.
    let mut failed_at: HashMap<(String, usize, usize), String> = HashMap::new();

    loop {
        progress(&report.narration);
        if ctx.cancel.load(Ordering::Relaxed) {
            report.stopped_reason = "cancelled".to_string();
            return Ok(());
        }
        if let Some(reason) = budget.exhausted() {
            report.stopped_reason = reason;
            return Ok(());
        }
        if let Some(line) = cool_down(ctx.model, ctx.cancel) {
            report.narration.push(line);
        }

        let (path, from, to, written, current, code, clean_prompt) = match attempt(ctx, goal, goal, &last_failure, &mut tried, report)? {
            Attempt::Edit { path, from, to, written, current, code, clean_prompt } => (path, from, to, written, current, code, clean_prompt),
            Attempt::Unchanged { path, from, to, target, clean_prompt } => {
                // Small models often copy the lines back unchanged while meaning to change them.
                budget.edits += 1;
                report.narration.push(format!("Xenon's edit to {path} changed nothing; asking again"));
                failed_at.entry((path.clone(), from, to)).or_insert(clean_prompt);
                last_failure = format!(
                    "Your last edit to `{path}` wrote these lines back unchanged:\n```{lang}\n{target}\n```\n\
                     This time, change them so the goal is done.\n\n",
                    lang = fence(&path)
                );
                continue;
            }
            Attempt::Done(reason) => {
                report.narration.push(format!("Stopped: {reason}"));
                report.stopped_reason = "finished".to_string();
                return Ok(());
            }
            Attempt::Stop(reason) => {
                report.stopped_reason = reason;
                return Ok(());
            }
        };
        if code.trim() == current.trim() {
            report.narration.push(format!("Stopped: Xenon proposed {path} unchanged"));
            report.stopped_reason = "finished".to_string();
            return Ok(());
        }

        // Before the first edit a check covers, run it on the untouched code: tests that already
        // fail would make every edit look wrong, and its compiler warnings are the bar an edit
        // must not raise (Xenon once "removed" a warning by re-indenting, and every test passed).
        let check = check_for(ctx.root, &path, ctx.files);
        if let Some(check) = &check {
            ensure_baseline(ctx, check, &mut baselines, report)?;
        }
        let change = write_file(ctx.root, &path, &code)?;
        budget.edits += 1;
        report.narration.push(format!("Edited {path}: rewrote lines {}-{to}", from + 1));
        report.edits.push(change.clone());
        progress(&report.narration);

        let Some(check) = check else {
            report.narration.push(format!("No check runs for {path}'s file type, so stopping after the edit"));
            report.stopped_reason = "finished: edited, nothing to check it with".to_string();
            return Ok(());
        };
        budget.runs += 1;
        let mut failure = String::new();
        let place = (path.clone(), from, to);
        if passes(ctx, &check, goal, &baselines, report, &mut failure)? {
            if let Some(prompt) = failed_at.remove(&place) {
                report.lessons.push(Lesson { prompt, answer: format!("{written}\n```"), passed: true, source: "agent" });
            }
            report.stopped_reason = "finished: the tests pass".to_string();
            return Ok(());
        }
        // A goal that changes behavior and says its test changes too can fail only because a test
        // still asserts the old behavior. Then the Test-writer updates that assertion, and the
        // code and test edits are checked as one change, instead of the code edit being undone.
        if wants_test_update(goal) && !budget.exhausted().is_some() {
            if let Some(test_change) = update_failing_test(ctx, goal, &failure, report)? {
                budget.edits += 1;
                budget.runs += 1;
                let mut test_failure = String::new();
                if passes(ctx, &check, goal, &baselines, report, &mut test_failure)? {
                    report.narration.push(format!("Updated {} to expect the new behavior", test_change.path));
                    report.edits.push(test_change);
                    report.stopped_reason = "finished: the tests pass".to_string();
                    return Ok(());
                }
                write_file(ctx.root, &test_change.path, &test_change.before)?;
                failure = test_failure;
            }
        }
        // A failed edit is undone, so the next try starts from the original file and a run that
        // runs out of budget never leaves a broken one (one did, before this).
        write_file(ctx.root, &path, &change.before)?;
        report.edits.pop();
        report.narration.push("Undid the edit".to_string());
        failed_at.entry(place).or_insert(clean_prompt);
        last_failure = format!(
            "Your last edit to `{path}` was:\n```diff\n{}\n```\nIt was undone because {failure}\n",
            short_diff(&change.before, &change.after)
        );
    }
}

/// Several steps: make every step's edit, then check once. Checking after each step would undo
/// every partial change (a renamed function breaks its callers until the next step). On a
/// failure all of the steps are undone and the goal is planned again with the failure shown.
fn run_steps(ctx: &RunContext, goal: &str, mut steps: Vec<String>, report: &mut Report, progress: &dyn Fn(&[String])) -> Result<(), String> {
    let mut budget = Budget::new();
    let mut baselines: HashMap<String, Vec<String>> = HashMap::new();
    let mut tried: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    let mut last_failure = String::new();

    // The baselines come first, before any step changes a file.
    let mut checks: Vec<Check> = Vec::new();
    for file in ctx.files {
        if let Some(check) = check_for(ctx.root, file, ctx.files) {
            if !checks.iter().any(|c| c.display() == check.display()) {
                ensure_baseline(ctx, &check, &mut baselines, report)?;
                checks.push(check);
            }
        }
    }

    loop {
        progress(&report.narration);
        if ctx.cancel.load(Ordering::Relaxed) {
            report.stopped_reason = "cancelled".to_string();
            return Ok(());
        }
        if let Some(reason) = budget.exhausted() {
            report.stopped_reason = reason;
            return Ok(());
        }
        if let Some(line) = cool_down(ctx.model, ctx.cancel) {
            report.narration.push(line);
        }

        // A failed round undoes every step, so each round starts from the original files and
        // every place is fresh again. Kept across rounds, the record skipped nearly every step
        // of rounds 2 and 3 as "tried every place".
        tried.clear();
        let mut changes: Vec<FileChange> = Vec::new();
        for (i, step) in steps.iter().enumerate() {
            let step_goal = format!("{step} (step {} of {}; the whole goal: {goal})", i + 1, steps.len());
            match attempt(ctx, &step_goal, step, &last_failure, &mut tried, report)? {
                Attempt::Edit { path, from, to, code, .. } => {
                    let change = write_file(ctx.root, &path, &code)?;
                    budget.edits += 1;
                    report.narration.push(format!("Step {}: edited {path}, lines {}-{to}", i + 1, from + 1));
                    changes.push(change);
                }
                Attempt::Unchanged { path, .. } => report.narration.push(format!("Step {}: Xenon left {path} unchanged", i + 1)),
                Attempt::Done(reason) => report.narration.push(format!("Step {}: nothing to change ({reason})", i + 1)),
                Attempt::Stop(reason) => report.narration.push(format!("Step {}: skipped ({reason})", i + 1)),
            }
            progress(&report.narration);
        }
        if changes.is_empty() {
            report.stopped_reason = "no step changed anything".to_string();
            return Ok(());
        }

        // Every check that covers a changed file, once each.
        let mut failure = String::new();
        let mut all_passed = true;
        let mut ran: Vec<String> = Vec::new();
        for change in &changes {
            let Some(check) = check_for(ctx.root, &change.path, ctx.files) else { continue };
            if ran.contains(&check.display()) {
                continue;
            }
            ran.push(check.display());
            ensure_baseline(ctx, &check, &mut baselines, report)?;
            budget.runs += 1;
            if !passes(ctx, &check, goal, &baselines, report, &mut failure)? {
                all_passed = false;
                break;
            }
        }
        if all_passed {
            report.edits.extend(changes);
            report.stopped_reason = if ran.is_empty() { "finished: edited, nothing to check it with" } else { "finished: the tests pass" }.to_string();
            return Ok(());
        }

        // Undo every step, newest first, so each file returns to how it was.
        for change in changes.iter().rev() {
            write_file(ctx.root, &change.path, &change.before)?;
        }
        report.narration.push("Undid every step".to_string());
        let diffs: Vec<String> = changes.iter().map(|c| format!("`{}`:\n```diff\n{}\n```", c.path, short_diff(&c.before, &c.after))).collect();
        last_failure = format!("Your last edits were:\n{}\nThey were undone because {failure}\n", diffs.join("\n"));
        steps = plan(ctx, goal, &last_failure)?;
        if steps.is_empty() {
            report.stopped_reason = "Xenon gave no steps".to_string();
            return Ok(());
        }
        report.narration.push(format!("Planned again: {}", steps.join(" / ")));
    }
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
    fn warnings_are_read_without_line_numbers_or_summaries() {
        let output = "warning: function `split_into_parts` is never used\n  --> src\\model_parts.rs:84:8\n   |\n\
                      warning: `app` (lib) generated 1 warning\n";
        assert_eq!(warnings(output), vec!["function `split_into_parts` is never used (src/model_parts.rs)"]);
        let moved = output.replace(":84:8", ":90:8");
        assert_eq!(warnings(&moved), warnings(output), "a warning that moved is the same warning");
    }

    #[test]
    fn goals_about_removing_a_warning_are_recognised() {
        assert!(asks_to_remove_a_warning("Silence the dead_code warning on split_into_parts"));
        assert!(!asks_to_remove_a_warning("Add \"models\" to SKIP_DIRS"));
        assert!(!asks_to_remove_a_warning("Log a warning when the GPU is hot"));
    }

    #[test]
    fn steps_are_read_from_a_numbered_list() {
        let reply = "1. Rename calc_total in shop.py\n2) Update the import in report.py\n   3. Update test_shop.py\nUser: next";
        assert_eq!(
            parse_steps(reply),
            vec!["Rename calc_total in shop.py", "Update the import in report.py", "Update test_shop.py"]
        );
        assert_eq!(parse_steps("1. Only one change"), vec!["Only one change"]);
        assert!(parse_steps("Sure, here is a plan:").is_empty());
        let many: String = (1..=9).map(|i| format!("{i}. step {i}\n")).collect();
        assert_eq!(parse_steps(&many).len(), MAX_STEPS);
    }

    #[test]
    fn lessons_are_saved_as_json_lines() {
        let dir = temp_repo();
        let path = dir.join("data").join("mistakes.jsonl");
        let lesson = Lesson { prompt: "P".into(), answer: "A
```".into(), passed: true, source: "agent" };
        save_lessons(&path, &[lesson.clone()]).unwrap();
        save_lessons(&path, &[]).unwrap();
        save_lessons(&path, &[lesson]).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
        let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["answer"], "A
```");
        assert_eq!(first["passed"], true);
        fs::remove_dir_all(&dir).ok();
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
    // ---- The loop, with a scripted model --------------------------------------------------
    // Mutation testing found none of the loop was tested: run_one could be replaced with "do
    // nothing" and every test still passed. These run it end to end against a stand-in model
    // and real files, checked with `python`.

    use std::cell::{Cell, RefCell};

    /// A stand-in for the model: `reply(question, start)` is what it writes after `start`.
    struct Scripted<F: Fn(&str, &str) -> String> {
        reply: F,
        temps: RefCell<Vec<u32>>,
        /// The first question asked with an edit step's reply start.
        edit_question: RefCell<String>,
    }

    fn scripted<F: Fn(&str, &str) -> String>(reply: F) -> Scripted<F> {
        Scripted { reply, temps: RefCell::new(Vec::new()), edit_question: RefCell::new(String::new()) }
    }

    impl<F: Fn(&str, &str) -> String> Model for Scripted<F> {
        /// Writes the scripted reply a character at a time and stops once `done` says so, as the
        /// engine stops generating, so the loop's stop rules are tested too.
        fn ask(&self, question: &str, start: &str, done: &dyn Fn(&str) -> bool, _deadline: Instant, _cancel: &AtomicBool) -> Result<String, String> {
            if start.starts_with("Lines ") && self.edit_question.borrow().is_empty() {
                *self.edit_question.borrow_mut() = question.to_string();
            }
            let mut text = start.to_string();
            for ch in (self.reply)(question, start).chars() {
                text.push(ch);
                if done(&text) {
                    break;
                }
            }
            Ok(text.trim().to_string())
        }
        fn prompt_text(&self, question: &str, start: &str) -> String {
            format!("{question}|{start}")
        }
        fn gpu_temp(&self) -> Option<u32> {
            let mut temps = self.temps.borrow_mut();
            (!temps.is_empty()).then(|| temps.remove(0))
        }
    }

    /// The lines an edit step's reply start quotes: the ones Xenon is asked to rewrite.
    fn quoted_lines(start: &str) -> String {
        let after = start.split_once("are:\n```").map(|(_, r)| r).unwrap_or("");
        let after = after.split_once('\n').map(|(_, r)| r).unwrap_or("");
        after.split("\n```").next().unwrap_or("").to_string()
    }

    /// Replies as Xenon would to an edit run: names `file`, rewrites the quoted lines with `edit`,
    /// and plans one step unless `steps` says otherwise.
    fn editing(file: &'static str, steps: &'static str, edit: impl Fn(&str, &str) -> String) -> impl Fn(&str, &str) -> String {
        move |question, start| {
            if start == "FILE:" {
                format!(" {file}\n")
            } else if start.starts_with("Lines ") {
                format!("{}\n```", edit(question, &quoted_lines(start)))
            } else if start == "1." {
                steps.to_string()
            } else {
                String::new()
            }
        }
    }

    fn repo_with(files: &[(&str, &str)]) -> PathBuf {
        let dir = temp_repo();
        for (rel, text) in files {
            fs::write(dir.join(rel), text).unwrap();
        }
        git(&dir, &["add", "."]).unwrap();
        git(&dir, &["commit", "-q", "-m", "task"]).unwrap();
        dir
    }

    fn run_scripted(model: &dyn Model, dir: &Path, goal: &str, files: &[&str]) -> Report {
        let files: Vec<String> = files.iter().map(|f| f.to_string()).collect();
        run_with(model, dir, goal, &files, &|_| {}, &AtomicBool::new(false)).unwrap()
    }

    const VALUE_PY: &str = "VALUE = 1\nassert VALUE == 2, VALUE\n";

    #[test]
    fn a_failed_edit_is_undone_and_the_one_that_passes_is_kept_and_learned() {
        let dir = repo_with(&[("a.py", VALUE_PY)]);
        let tries = Cell::new(0);
        let model = scripted(editing("a.py", " One step\n", |_, lines| {
            tries.set(tries.get() + 1);
            lines.replace("VALUE = 1", if tries.get() == 1 { "VALUE = 3" } else { "VALUE = 2" })
        }));
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        assert_eq!(report.stopped_reason, "finished: the tests pass", "{:?}", report.narration);
        assert!(fs::read_to_string(dir.join("a.py")).unwrap().contains("VALUE = 2"));
        assert!(report.narration.iter().any(|n| n == "Undid the edit"), "{:?}", report.narration);
        assert_eq!(report.edits.len(), 1, "only the kept edit is reported");
        assert_eq!(report.lessons.len(), 1, "a fail then a pass at the same lines is a lesson");
        assert!(report.lessons[0].answer.starts_with("VALUE = 2"), "{}", report.lessons[0].answer);
        assert!(!report.lessons[0].prompt.contains("undone"), "the lesson's prompt has no failure feedback");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edits_that_keep_failing_leave_the_file_as_it_was() {
        let dir = repo_with(&[("a.py", VALUE_PY)]);
        let model = scripted(editing("a.py", " One step\n", |_, lines| lines.replace("VALUE = 1", "VALUE = 3")));
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        assert!(!report.stopped_reason.starts_with("finished"), "{}", report.stopped_reason);
        assert_eq!(fs::read_to_string(dir.join("a.py")).unwrap().replace("\r\n", "\n"), VALUE_PY);
        assert!(report.edits.is_empty() && report.lessons.is_empty());
        assert!(report.runs.len() >= 2, "each edit was checked");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lines_written_back_unchanged_are_noted_and_never_written() {
        let dir = repo_with(&[("a.py", VALUE_PY)]);
        let model = scripted(editing("a.py", " One step\n", |_, lines| lines.to_string()));
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        let unchanged = report.narration.iter().filter(|n| n.contains("changed nothing")).count();
        assert_eq!(unchanged, TRIES_PER_PLACE, "two tries at the one place: {:?}", report.narration);
        assert!(report.stopped_reason.contains("tried every place"), "{}", report.stopped_reason);
        assert!(report.edits.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nothing_to_change_ends_the_run_without_an_edit() {
        let dir = repo_with(&[("a.py", VALUE_PY)]);
        let model = scripted(|_: &str, start: &str| match start {
            "FILE:" => " none\n".to_string(),
            s if s.starts_with("FILE: none") => " already done\n".to_string(),
            _ => String::new(),
        });
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        assert_eq!(report.stopped_reason, "finished");
        assert!(report.narration.iter().any(|n| n == "Stopped: already done"), "{:?}", report.narration);
        assert!(report.edits.is_empty() && report.runs.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_outside_the_given_list_is_refused() {
        let dir = repo_with(&[("a.py", VALUE_PY), ("b.py", "x = 1\n")]);
        let model = scripted(editing("b.py", " One step\n", |_, lines| lines.to_string()));
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        assert!(report.stopped_reason.contains("wasn't in the given file list"), "{}", report.stopped_reason);
        assert_eq!(fs::read_to_string(dir.join("b.py")).unwrap().replace("\r\n", "\n"), "x = 1\n");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn steps_are_tried_only_after_one_edit_at_a_time_gives_up() {
        let dir = repo_with(&[("a.py", "ALPHA = 1\nBETA = 1\nassert ALPHA == 2 and BETA == 2\n")]);
        // One edit at a time sets a wrong value every time; each planned step sets the right one.
        let model = scripted(editing(
            "a.py",
            " Set ALPHA to 2 in a.py\n2. Set BETA to 2 in a.py\n",
            |question, lines| lines.replace("= 1", if question.contains("(step ") { "= 2" } else { "= 3" }),
        ));
        let report = run_scripted(&model, &dir, "Set ALPHA and BETA to 2", &["a.py"]);

        assert_eq!(report.stopped_reason, "finished: the tests pass", "{:?}", report.narration);
        assert!(report.narration.iter().any(|n| n.contains("broke the goal into 2 steps")), "{:?}", report.narration);
        let text = fs::read_to_string(dir.join("a.py")).unwrap();
        assert!(text.contains("ALPHA = 2") && text.contains("BETA = 2"), "{text}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_goal_finished_in_one_edit_is_never_broken_into_steps() {
        let dir = repo_with(&[("a.py", VALUE_PY)]);
        let planned = Cell::new(false);
        let edit = editing("a.py", " A\n2. B\n", |_, lines| lines.replace("VALUE = 1", "VALUE = 2"));
        let model = scripted(|question: &str, start: &str| {
            if start == "1." {
                planned.set(true);
            }
            edit(question, start)
        });
        let report = run_scripted(&model, &dir, "Set VALUE to 2", &["a.py"]);

        assert_eq!(report.stopped_reason, "finished: the tests pass");
        assert!(!planned.get(), "no plan was asked for");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hot_gpu_is_waited_out_and_a_cool_one_is_not() {
        let model = scripted(|_: &str, _: &str| String::new());
        let cancel = AtomicBool::new(false);
        assert_eq!(cool_down(&model, &cancel), None, "no GPU");
        *model.temps.borrow_mut() = vec![HOT_C - 1];
        assert_eq!(cool_down(&model, &cancel), None, "below HOT_C");
        *model.temps.borrow_mut() = vec![HOT_C + 5, COOL_C + 3, COOL_C];
        let line = cool_down(&model, &cancel).expect("waited");
        assert!(line.contains(&format!("{}°C", HOT_C + 5)), "{line}");
        assert!(model.temps.borrow().is_empty(), "read until it was cool");
    }

    #[test]
    fn the_budget_stops_at_each_limit() {
        let mut budget = Budget::new();
        assert_eq!(budget.exhausted(), None);
        budget.edits = MAX_EDITS - 1;
        budget.runs = MAX_RUNS - 1;
        assert_eq!(budget.exhausted(), None, "one short of each limit");
        budget.edits = MAX_EDITS;
        assert!(budget.exhausted().unwrap().contains("edit budget"));
        let mut budget = Budget::new();
        budget.runs = MAX_RUNS;
        assert!(budget.exhausted().unwrap().contains("run budget"));
        // Windows counts Instant from boot, so "11 minutes ago" doesn't exist in a fresh boot's
        // first minutes; the time limit is then checked on the next run instead.
        if let Some(started) = Instant::now().checked_sub(Duration::from_secs(MAX_SECONDS + 1)) {
            let budget = Budget { started, edits: 0, runs: 0 };
            assert!(budget.exhausted().unwrap().contains("time budget"));
        }
    }

    #[test]
    fn checks_fail_on_new_warnings_and_on_a_warning_the_goal_wanted_gone() {
        let dir = temp_repo();
        let model = scripted(|_: &str, _: &str| String::new());
        let cancel = AtomicBool::new(false);
        let files: Vec<String> = Vec::new();
        let ctx = RunContext {
            model: &model,
            root: &dir,
            files: &files,
            searched: false,
            listing: "",
            deadline: Instant::now() + Duration::from_secs(60),
            cancel: &cancel,
        };
        let printing = |text: &str| Check { program: "python", args: vec!["-c".into(), format!("print({text:?})")], dir: String::new() };
        let old = "warning: old\n  --> src/a.rs:1:1";
        let check = printing(old);
        let mut baselines = HashMap::new();
        baselines.insert(check.display(), warnings(old));
        let mut report = Report::default();
        let mut failure = String::new();

        assert!(passes(&ctx, &check, "Add a flag", &baselines, &mut report, &mut failure).unwrap(), "same warnings as before");
        assert!(!passes(&ctx, &check, "Silence the old warning", &baselines, &mut report, &mut failure).unwrap());
        assert!(failure.contains("still there"), "{failure}");

        let added = printing("warning: old\n  --> src/a.rs:1:1\nwarning: new\n  --> src/a.rs:9:1");
        let mut baselines = HashMap::new();
        baselines.insert(added.display(), warnings(old));
        assert!(!passes(&ctx, &added, "Add a flag", &baselines, &mut report, &mut failure).unwrap());
        assert!(failure.contains("added compiler warnings") && failure.contains("new"), "{failure}");

        let failing = Check { program: "python", args: vec!["-c".into(), "raise SystemExit(1)".into()], dir: String::new() };
        assert!(!passes(&ctx, &failing, "Add a flag", &HashMap::new(), &mut report, &mut failure).unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn job_status_and_cancel_follow_the_job() {
        let id = "run-test-status".to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let status = JobStatus { state: "running".into(), narration: Vec::new(), report: None, error: None };
        jobs().lock().unwrap().insert(id.clone(), Job { status, cancel: cancel.clone() });
        update(&id, |s| s.narration.push("a step".into()));
        assert_eq!(job_status(&id).unwrap().narration, vec!["a step".to_string()]);
        assert!(cancel_job(&id));
        assert!(cancel.load(Ordering::Relaxed), "cancelling reaches the running loop");
        jobs().lock().unwrap().remove(&id);
    }

    #[test]
    fn cargo_is_found_when_tests_run_under_it() {
        let cargo = find_cargo().expect("cargo is running these tests");
        assert!(cargo.file_stem().is_some_and(|s| s == "cargo"), "{}", cargo.display());
        assert!(cargo.is_file());
    }

    /// A run context over `root` with `model`, as `run_with` would make one.
    fn context<'a>(model: &'a dyn Model, root: &'a Path, files: &'a [String], cancel: &'a AtomicBool) -> RunContext<'a> {
        RunContext { model, root, files, searched: false, listing: "", deadline: Instant::now() + Duration::from_secs(60), cancel }
    }

    #[test]
    fn a_plan_ends_at_the_first_blank_line() {
        let dir = temp_repo();
        let model = scripted(|_: &str, start: &str| if start == "1." { " A\n2. B\n\n3. not part of the plan\n".to_string() } else { String::new() });
        let cancel = AtomicBool::new(false);
        let files: Vec<String> = Vec::new();
        assert_eq!(plan(&context(&model, &dir, &files, &cancel), "goal", "").unwrap(), vec!["A", "B"]);
        fs::remove_dir_all(&dir).ok();
    }

    const TEXT_PY: &str = "import re\n\n\ndef slugify(title):\n    return '-'.join(w.lower() for w in re.findall(r'[a-zA-Z]+', title))\n";
    const TEST_TEXT_PY: &str = "from text import slugify\n\n\ndef test_digits():\n    assert slugify('Room 101') == 'room'\n";

    /// Edits as Xenon would for "keep digits": the code's regex, then (asked by the Test-writer)
    /// the test's expected value, via `test_edit`.
    fn keeping_digits(test_edit: fn(&str) -> String) -> impl Fn(&str, &str) -> String {
        editing("text.py", " One step\n", move |question, lines| {
            if question.contains("update this test") {
                test_edit(lines)
            } else {
                lines.replace("[a-zA-Z]+", "[a-zA-Z0-9]+")
            }
        })
    }

    #[test]
    fn a_behavior_change_updates_its_test_and_keeps_both() {
        let dir = repo_with(&[("text.py", TEXT_PY), ("test_text.py", TEST_TEXT_PY)]);
        let model = scripted(keeping_digits(|lines| lines.replace("== 'room'", "== 'room-101'")));
        let report = run_scripted(&model, &dir, "slugify should keep digits. Update the code and its test.", &["text.py", "test_text.py"]);

        assert_eq!(report.stopped_reason, "finished: the tests pass", "{:?}", report.narration);
        assert!(fs::read_to_string(dir.join("text.py")).unwrap().contains("[a-zA-Z0-9]+"));
        assert!(fs::read_to_string(dir.join("test_text.py")).unwrap().contains("== 'room-101'"));
        assert_eq!(report.edits.len(), 2, "the code and its test, kept together");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_test_change_that_drops_the_assertion_is_refused_and_both_are_undone() {
        let dir = repo_with(&[("text.py", TEXT_PY), ("test_text.py", TEST_TEXT_PY)]);
        let model = scripted(keeping_digits(|lines| lines.replace("    assert slugify('Room 101') == 'room'", "    pass")));
        let report = run_scripted(&model, &dir, "slugify should keep digits. Update the code and its test.", &["text.py", "test_text.py"]);

        assert!(report.narration.iter().any(|n| n.contains("removed an assertion")), "{:?}", report.narration);
        assert!(!report.stopped_reason.starts_with("finished"), "{}", report.stopped_reason);
        assert_eq!(fs::read_to_string(dir.join("test_text.py")).unwrap().replace("\r\n", "\n"), TEST_TEXT_PY);
        assert_eq!(fs::read_to_string(dir.join("text.py")).unwrap().replace("\r\n", "\n"), TEXT_PY);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_goal_that_doesnt_mention_its_test_never_changes_one() {
        let dir = repo_with(&[("text.py", TEXT_PY), ("test_text.py", TEST_TEXT_PY)]);
        let model = scripted(keeping_digits(|lines| lines.replace("== 'room'", "== 'room-101'")));
        let report = run_scripted(&model, &dir, "slugify should keep digits.", &["text.py", "test_text.py"]);

        assert!(!report.stopped_reason.starts_with("finished"), "the old test fails, so the edit is wrong");
        assert_eq!(fs::read_to_string(dir.join("test_text.py")).unwrap().replace("\r\n", "\n"), TEST_TEXT_PY);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_failing_test_line_is_read_from_pytest_output() {
        let output = "    def test_digits():\n>       assert slugify('Room 101') == 'room'\nE       AssertionError\n\ntest_text.py:5: AssertionError\n";
        assert_eq!(failing_test_line(output), Some(("test_text.py".to_string(), 5)));
        assert_eq!(failing_test_line("C:\\tmp\\repo\\tests\\test_a.py:12: in test_x"), Some(("C:/tmp/repo/tests/test_a.py".to_string(), 12)));
        assert_eq!(failing_test_line("text.py:3: SyntaxError"), None, "not a test file");
        assert!(wants_test_update("Update the code and its Test.") && !wants_test_update("Fix the bug"));
    }

    #[test]
    fn a_step_naming_a_file_is_located_there() {
        let files = vec!["shop.py".to_string(), "report.py".to_string(), "test_shop.py".to_string(), "src/util.rs".to_string()];
        assert_eq!(file_named_in("Update the import in `report.py`.", &files).as_deref(), Some("report.py"));
        assert_eq!(file_named_in("Rename calc_total in shop.py", &files).as_deref(), Some("shop.py"), "not test_shop.py");
        assert_eq!(file_named_in("Fix test_shop.py", &files).as_deref(), Some("test_shop.py"));
        assert_eq!(file_named_in("Tidy util.rs", &files).as_deref(), Some("src/util.rs"), "by its file name");
        assert_eq!(file_named_in("Rename it in shop.py and report.py", &files), None, "two files: Xenon chooses");
        assert_eq!(file_named_in("Rename calc_total", &files), None);
    }

    #[test]
    fn a_step_names_its_file_and_xenon_is_not_asked() {
        let dir = repo_with(&[("a.py", VALUE_PY), ("b.py", "x = 1\n")]);
        // Xenon would name the wrong file if asked; the goal names the right one.
        let model = scripted(editing("b.py", " One step\n", |_, lines| lines.replace("VALUE = 1", "VALUE = 2")));
        let report = run_scripted(&model, &dir, "Set VALUE to 2 in a.py", &["a.py", "b.py"]);
        assert_eq!(report.stopped_reason, "finished: the tests pass", "{:?}", report.narration);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn only_numbered_lines_are_steps() {
        assert_eq!(parse_steps(". no number\n3 apples\n1. real step\n"), vec!["real step"]);
    }

    #[test]
    fn a_baseline_is_run_once_and_a_failing_one_is_noted() {
        let dir = temp_repo();
        let model = scripted(|_: &str, _: &str| String::new());
        let cancel = AtomicBool::new(false);
        let files: Vec<String> = Vec::new();
        let ctx = context(&model, &dir, &files, &cancel);
        let mut report = Report::default();
        let mut baselines = HashMap::new();

        let failing = Check { program: "python", args: vec!["-c".into(), "print('warning: w\\n  --> a.rs:1:1'); raise SystemExit(1)".into()], dir: String::new() };
        ensure_baseline(&ctx, &failing, &mut baselines, &mut report).unwrap();
        assert_eq!(baselines.get(&failing.display()).map(Vec::len), Some(1), "its warnings are kept");
        assert!(report.narration.iter().any(|n| n.contains("fails before any edit")), "{:?}", report.narration);

        let passing = Check { program: "python", args: vec!["-c".into(), "pass".into()], dir: String::new() };
        ensure_baseline(&ctx, &passing, &mut baselines, &mut report).unwrap();
        assert_eq!(report.narration.len(), 1, "a passing baseline isn't noted");

        // Run again, the stored baseline is used: no new narration even though it still fails.
        ensure_baseline(&ctx, &failing, &mut baselines, &mut report).unwrap();
        assert_eq!(report.narration.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_check_is_stopped_at_its_deadline_or_on_cancel() {
        let dir = temp_repo();
        let sleeper = Check { program: "python", args: vec!["-c".into(), "import time; time.sleep(30)".into()], dir: String::new() };
        let started = Instant::now();
        let step = run_check(&dir, &sleeper, Instant::now() + Duration::from_millis(300), &AtomicBool::new(false)).unwrap();
        assert!(started.elapsed() < Duration::from_secs(10), "killed, not waited out");
        assert!(step.stderr.contains("out of time"), "{}", step.stderr);
        assert_ne!(step.exit_code, Some(0));

        let step = run_check(&dir, &sleeper, Instant::now() + Duration::from_secs(60), &AtomicBool::new(true)).unwrap();
        assert!(step.stderr.contains("cancelled"), "{}", step.stderr);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_cool_down_stops_waiting_when_cancelled() {
        let model = scripted(|_: &str, _: &str| String::new());
        *model.temps.borrow_mut() = vec![HOT_C + 5, HOT_C + 5, HOT_C + 5];
        assert!(cool_down(&model, &AtomicBool::new(true)).is_some(), "it was hot, so it says so");
        assert_eq!(model.temps.borrow().len(), 2, "only the first reading: cancel ends the wait at once");
    }

    #[test]
    fn an_edit_is_shown_three_lines_after_it_and_no_more() {
        // Names of three or more letters: search ignores shorter words.
        let lines: Vec<String> = (0..10).map(|i| format!("VALUE{i} = {i}")).collect();
        let text = lines.join("\n") + "\nassert VALUE2 == 9\n";
        let dir = repo_with(&[("a.py", &text)]);
        let model = scripted(editing("a.py", " One step\n", |_, lines| lines.to_string()));
        run_scripted(&model, &dir, "Set VALUE2 to 9", &["a.py"]);
        let question = model.edit_question.borrow().clone();
        // VALUE2 is rewritten with both neighbours (VALUE1..VALUE3); the three after are VALUE4..6.
        assert!(question.contains("VALUE6 = 6") && !question.contains("VALUE7 = 7"), "{question}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_language_gets_its_fence() {
        let cases = [
            ("a.py", "python"), ("a.rs", "rust"), ("a.ts", "typescript"), ("a.tsx", "typescript"),
            ("a.js", "javascript"), ("a.mjs", "javascript"), ("a.vue", "vue"), ("a.c", "c"), ("a.h", "c"),
            ("a.cpp", "cpp"), ("a.cu", "cpp"), ("a.cuh", "cpp"), ("a.hpp", "cpp"), ("a.sh", "bash"),
            ("a.md", "markdown"), ("a.toml", "toml"), ("a.json", "json"), ("a.txt", ""),
        ];
        for (file, lang) in cases {
            assert_eq!(fence(file), lang, "{file}");
        }
    }

    #[test]
    fn small_helpers_at_their_edges() {
        // The path step ends on a long line even without a newline.
        assert!(first_line_done(&format!("FILE: {}", "x".repeat(301))));
        assert!(!first_line_done(&format!("FILE: {}", "x".repeat(299))));
        // Numbered lines only; a list can run past nine.
        assert_eq!(parse_steps("x. not a step\n10. tenth\n"), vec!["tenth"]);
        assert!(parse_steps("1.\n2. \n").is_empty(), "numbers with nothing after them");
        // Only two lines of context after a change near the top of a long file.
        assert_eq!(short_diff("a\nb\nc\nd\ne\nf\ng\n", "a\nB\nc\nd\ne\nf\ng\n"), " a\n-b\n+B\n c\n d");
        // A reply the loop couldn't use is quoted, or named as empty.
        assert_eq!(excerpt("  "), "(nothing)");
        assert_eq!(excerpt(" hi\n"), "\"hi\"");
        // A warning that merely mentions "generated" is still a warning.
        assert_eq!(warnings("warning: field `generated` is never read\n --> src/a.rs:3:5\n").len(), 1);
    }
}
