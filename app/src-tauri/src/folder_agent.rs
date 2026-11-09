//! Project-folder agent: point Xenon at a folder and an English goal, and it edits, runs, and
//! fixes the Python in that folder on its own, batch style, stopping on a hard budget and
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
//! - *Execution*: `python` runs with its working directory pinned to the target folder. This is
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

const SNAPSHOT_MESSAGE: &str = "xenon-agent: snapshot before run";

/// The RTX 2060 here is in a laptop. At or above `HOT_C` the agent waits, before its next model
/// turn, until the GPU is back down to `COOL_C`.
const HOT_C: u32 = 80;
const COOL_C: u32 = 72;
const MAX_COOL_WAIT: Duration = Duration::from_secs(5 * 60);

fn gpu_temp() -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
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

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
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
fn snapshot(root: &Path) -> Result<Option<String>, String> {
    let status = git(root, &["status", "--porcelain"])?;
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
fn ask(engine: &EngineState, model_name: &str, question: &str) -> Result<String, String> {
    let style = Style::for_model(model_name);
    let request = ChatRequest {
        messages: vec![ChatMessage { role: "user".to_string(), content: serde_json::Value::String(question.to_string()) }],
        stream: false,
        max_tokens: Some(1200),
        max_completion_tokens: None,
        temperature: Some(0.2),
        top_p: None,
        repeat_penalty: None,
    };
    let prompt = code_api::build_prompt(&request, style)?;
    let mut text = String::new();
    let mut sink = |piece: &str| {
        text.push_str(piece);
        !reply_complete(&text)
    };
    inference::generate_code(engine, &prompt.prior, &prompt.last, prompt.params, &mut sink)?;
    Ok(text.trim().to_string())
}

/// Runs `python -m pytest <file>` for a test file (by the `test_*.py`/`*_test.py` convention
/// pytest itself uses, or when `file` is a directory) and plain `python <file>` otherwise, with
/// the working directory pinned to `root`, so relative imports and relative file access the
/// script makes stay inside the sandboxed folder by construction, not by special-casing its own
/// I/O.
fn run_python(root: &Path, relative_target: &str) -> Result<RunStep, String> {
    let target = contained(root, &root.join(relative_target))?;
    let file_name = target.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let is_test = target.is_dir() || file_name.starts_with("test_") || file_name.ends_with("_test.py");
    let command = if is_test {
        if target.is_dir() {
            "python -m pytest".to_string()
        } else {
            format!("python -m pytest {relative_target}")
        }
    } else {
        format!("python {relative_target}")
    };
    let mut cmd = Command::new("python");
    if is_test {
        cmd.args(["-m", "pytest"]);
        if !target.is_dir() {
            cmd.arg(&target);
        }
    } else {
        cmd.arg(&target);
    }
    let output = cmd
        .current_dir(root)
        .output()
        .map_err(|e| format!("couldn't run {command}: {e}"))?;
    Ok(RunStep {
        command,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code(),
    })
}

/// Writes `new_text` to `relative_path` inside `root`, after confirming the path stays
/// contained, and returns the before/after pair for the report.
fn write_file(root: &Path, relative_path: &str, new_text: &str) -> Result<FileChange, String> {
    let target = contained(root, &root.join(relative_path))?;
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
        listing.push_str(&format!("File `{file}`:\n```python\n{text}\n```\n\n"));
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

        let question = format!(
            "{listing}{last_failure}Goal: {goal}\n\
             Reply in this exact shape: a line `FILE: <path>` naming one file above to rewrite, \
             then its complete new contents in one fenced code block, then a line starting with \
             `WHY:` giving a one-sentence reason. If the goal is already met and nothing needs to \
             change, reply with exactly `DONE: <one-sentence reason>` instead."
        );
        let reply = ask(engine, model_name, &question)?;

        if !reply.contains("FILE:") {
            if let Some(reason) = reason_after(&reply, "DONE:") {
                report.narration.push(format!("Stopped: {reason}"));
                report.stopped_reason = "finished".to_string();
                break;
            }
        }

        let Some(file_line) = reply.lines().find(|l| l.starts_with("FILE:")) else {
            report.stopped_reason = "the model's reply didn't name a file to edit".to_string();
            break;
        };
        let relative_path = file_line.trim_start_matches("FILE:").trim().to_string();
        if !files.contains(&relative_path) {
            report.stopped_reason = format!("the model named {relative_path}, which wasn't in the given file list");
            break;
        }
        let Some(code) = extract_code_block(&reply) else {
            report.stopped_reason = "the model's reply had no code block".to_string();
            break;
        };
        let why = reason_after(&reply, "WHY:").filter(|r| !r.is_empty()).unwrap_or_else(|| "no reason given".to_string());
        let current = std::fs::read_to_string(root.join(&relative_path)).unwrap_or_default();
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

        let run_target = if relative_path.ends_with("_test.py") || relative_path.starts_with("test_") {
            relative_path.clone()
        } else {
            match files.iter().find(|f| f.ends_with("_test.py") || f.starts_with("test_")) {
                Some(t) => t.clone(),
                None => relative_path.clone(),
            }
        };
        let step = match run_python(root, &run_target) {
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
        last_failure = format!("After your last edit, `{run_target}` failed with:\n```\n{}\n```\n\n", tail.trim());

        listing = listing.replace(
            &format!("File `{relative_path}`:\n```python\n{}", change.before),
            &format!("File `{relative_path}`:\n```python\n{}", change.after),
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
}
