//! Finds the lines of a project that best match a question, so Xenon can answer "where is X" and
//! "which file does Y" from the code itself. A 2.9B model can't hold a codebase in its head; it
//! can read twenty relevant lines. Plain keyword search weighted by rarity, re-run per question:
//! a few hundred source files take tens of milliseconds, with no index to keep fresh.

use std::collections::HashSet;
use std::path::Path;

/// Folders never searched. Git submodules (a folder holding a `.git` file) are skipped too:
/// they are someone else's code.
const SKIP_DIRS: &[&str] = &[
    "node_modules", ".git", "target", "out", "dist", "build", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".claude", "models",
];
const EXTENSIONS: &[&str] = &[
    "rs", "ts", "js", "vue", "py", "c", "cc", "cpp", "h", "hpp", "cu", "cuh", "sh", "ps1", "md",
    "toml", "json", "yml", "yaml", "css", "html",
];
const MAX_FILE_BYTES: u64 = 300_000;
const MAX_FILES: usize = 3;
const MAX_LINES_PER_FILE: usize = 5;
const MAX_LINE_CHARS: usize = 160;
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "what", "which", "where", "file", "files", "does",
    "how", "why", "when", "who", "are", "was", "repo", "project", "code", "xenon", "xenon2", "you",
    "your", "into", "from", "there", "here", "about", "answer", "path", "only", "tell", "show",
    "find", "defines", "defined", "define", "use", "uses", "used", "make", "can", "should", "would",
    "handle", "handles", "have", "has", "its", "not", "but", "all", "any", "out", "get", "set",
];
const DEFINERS: &[&str] = &[
    "fn ", "def ", "function ", "class ", "struct ", "enum ", "const ", "static ", "trait ",
    "interface ", "type ", "let ", "impl ",
];

struct Keyword {
    text: String,
    /// Looks like an identifier (snake_case, camelCase, or quoted in backticks): worth more.
    strong: bool,
}

fn keywords(question: &str) -> Vec<Keyword> {
    let quoted: HashSet<&str> = question.split('`').skip(1).step_by(2).collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for word in question.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let lower = word.to_lowercase();
        if lower.len() < 3 || STOPWORDS.contains(&lower.as_str()) || !seen.insert(lower.clone()) {
            continue;
        }
        let camel = word.chars().skip(1).any(char::is_uppercase) && word.chars().any(char::is_lowercase);
        let strong = word.contains('_') || camel || quoted.contains(word);
        out.push(Keyword { text: lower, strong });
    }
    out
}

fn source_files(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) && !path.join(".git").is_file() {
                    stack.push(path);
                }
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let small = entry.metadata().map(|m| m.len() <= MAX_FILE_BYTES).unwrap_or(false);
            if !kind.is_file() || !small || !EXTENSIONS.contains(&ext) || name.ends_with(".lock") || name == "package-lock.json" {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                out.push((rel, text));
            }
        }
    }
    out.sort();
    out
}

fn is_word_at(line: &str, at: usize, len: usize) -> bool {
    let before = line[..at].chars().next_back();
    let after = line[at + len..].chars().next();
    let edge = |c: Option<char>| c.map_or(true, |c| !(c.is_alphanumeric() || c == '_'));
    edge(before) && edge(after)
}

/// Words that may come before a definer at the start of a definition line.
const MODIFIERS: &[&str] = &["pub", "pub(crate)", "pub(super)", "async", "export", "default", "unsafe", "extern", "static", "abstract"];

/// True when `line` defines `name` (e.g. `pub fn name(`, `def name`, `export const name =`), and
/// not merely mentions it, as in a string or a comment.
fn defines(line: &str, name: &str) -> bool {
    let lower = line.to_lowercase();
    let mut from = 0;
    while let Some(found) = lower[from..].find(name) {
        let at = from + found;
        if is_word_at(&lower, at, name.len()) {
            for definer in DEFINERS {
                if let Some(before) = lower[..at].strip_suffix(definer) {
                    if before.split_whitespace().all(|w| MODIFIERS.contains(&w)) {
                        return true;
                    }
                }
            }
        }
        from = at + name.len();
    }
    false
}

/// The file in `root` that a model-written path most likely means. Small models shorten paths
/// (`src/inference.rs` for `app/src-tauri/src/inference.rs`) or add a stray leading `/`: this
/// keeps an exact match, else takes the one file whose path ends with the given one. `None` when
/// no file matches, or several do.
pub fn snap_path(root: &Path, written: &str) -> Option<String> {
    let written = written.trim().trim_matches('`').replace('\\', "/");
    let written = written.trim_start_matches("./").trim_start_matches('/');
    if written.is_empty() {
        return None;
    }
    if root.join(written).is_file() {
        return Some(written.to_string());
    }
    let suffix = format!("/{}", written.to_lowercase());
    let mut found = source_files(root)
        .into_iter()
        .map(|(rel, _)| rel)
        .filter(|rel| format!("/{}", rel.to_lowercase()).ends_with(&suffix));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// Every line of `text` that matches `question` at all (0-based), best first, weighted as
/// `search` weighs files: common words count for nothing, rare ones and identifiers for more, a
/// definition for most. Ties go to the earlier line.
pub fn ranked_lines(text: &str, question: &str) -> Vec<usize> {
    let words = keywords(question);
    let lowered: Vec<String> = text.lines().map(str::to_lowercase).collect();
    let weights: Vec<f32> = words
        .iter()
        .map(|w| {
            let df = lowered.iter().filter(|l| l.contains(&w.text)).count();
            if df == 0 {
                0.0
            } else {
                ((lowered.len() as f32 + 1.0) / df as f32).ln() * if w.strong { 3.0 } else { 1.0 }
            }
        })
        .collect();
    let mut scored: Vec<(usize, f32)> = text
        .lines()
        .zip(&lowered)
        .enumerate()
        .map(|(i, (line, lower))| {
            let score: f32 = words
                .iter()
                .zip(&weights)
                .filter(|(w, _)| lower.contains(&w.text))
                .map(|(w, weight)| if w.strong && defines(line, &w.text) { weight * 5.0 } else { *weight })
                .sum();
            (i, score)
        })
        .filter(|&(_, score)| score > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// One file's match: its path and best lines (1-based number, text), in file order.
pub struct Hit {
    pub path: String,
    pub lines: Vec<(usize, String)>,
}

/// The best-matching lines of `root` for `question`, formatted for the prompt, or "" when nothing
/// matches well enough to help.
pub fn search(root: &Path, question: &str) -> String {
    let hits = best_files(root, question, MAX_FILES);
    if hits.is_empty() {
        return String::new();
    }
    // grep's `path:line: text`, with the whole path on every line: the model copies paths it sees
    // whole, and shortens ones it has to piece together from a heading.
    let mut out = String::from("Matching lines from the project (path:line: text):\n");
    for hit in hits {
        for (n, line) in hit.lines {
            out.push_str(&format!("{}:{n}: {line}\n", hit.path));
        }
    }
    out
}

/// Up to `limit` files of `root` that best match `question`, best first.
pub fn best_files(root: &Path, question: &str, limit: usize) -> Vec<Hit> {
    let words = keywords(question);
    if words.is_empty() {
        return Vec::new();
    }
    let files = source_files(root);
    let lowered: Vec<String> = files.iter().map(|(_, text)| text.to_lowercase()).collect();
    // Rarer words say more: weight each by log(files / files containing it).
    let weights: Vec<f32> = words
        .iter()
        .map(|w| {
            let df = lowered.iter().filter(|t| t.contains(&w.text)).count();
            if df == 0 {
                0.0
            } else {
                ((files.len() as f32 + 1.0) / df as f32).ln() * if w.strong { 3.0 } else { 1.0 }
            }
        })
        .collect();

    let mut ranked = Vec::new();
    for ((rel, text), lower) in files.iter().zip(&lowered) {
        let rel_lower = rel.to_lowercase();
        let mut file_score = 0.0;
        for (w, weight) in words.iter().zip(&weights) {
            if lower.contains(&w.text) {
                file_score += weight * 0.5;
            }
            if rel_lower.contains(&w.text) {
                file_score += weight;
            }
        }
        if file_score == 0.0 {
            continue;
        }
        let mut lines: Vec<(f32, usize, &str)> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line_lower = line.to_lowercase();
            let mut score = 0.0;
            for (w, weight) in words.iter().zip(&weights) {
                if line_lower.contains(&w.text) {
                    score += weight;
                    if w.strong && defines(line, &w.text) {
                        score += weight * 4.0;
                    }
                }
            }
            if score > 0.0 {
                lines.push((score, n + 1, line));
            }
        }
        lines.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        lines.truncate(MAX_LINES_PER_FILE);
        let best = lines.first().map_or(0.0, |l| l.0);
        ranked.push((file_score + best, rel, lines));
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(_, rel, mut lines)| {
            lines.sort_by_key(|l| l.1);
            Hit {
                path: rel.clone(),
                lines: lines.into_iter().map(|(_, n, line)| (n, line.trim().chars().take(MAX_LINE_CHARS).collect())).collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Project(std::path::PathBuf);

    impl Project {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Project {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A throwaway folder holding `files` (path, contents), removed when dropped.
    fn project_with(files: &[(&str, String)]) -> Project {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = Project(std::env::temp_dir().join(format!(
            "xenon-code-search-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        for (rel, text) in files {
            let path = dir.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        dir
    }

    fn project() -> Project {
        project_with(&[
            ("src/engine.rs", "fn threads_for(layers: u32) -> u32 {\n    1\n}\n".into()),
            ("src/caller.rs", "fn load() {\n    let t = threads_for(3);\n}\n".into()),
            ("ui/App.vue", "<button class=\"stop-btn\" @click=\"stopReply\">Stop</button>\n".into()),
            ("node_modules/x/index.js", "function threads_for() {}\n".into()),
            ("vendor/.git", "gitdir: elsewhere\n".into()),
            ("vendor/lib.rs", "fn threads_for() {}\n".into()),
        ])
    }

    // The tests below were written against cargo-mutants' misses: each fails if the code it
    // covers is broken in the way named, which the earlier tests didn't notice.

    #[test]
    fn keywords_skip_short_and_common_words_and_mark_identifiers() {
        // "api" (3 letters) stays; two-letter words and stopwords go; a quoted word, a
        // snake_case name and a camelCase name are identifiers, a plain word is not.
        let found: Vec<(String, bool)> = keywords("Is the api of `Stop` in threads_for or loadModel?")
            .into_iter()
            .map(|k| (k.text, k.strong))
            .collect();
        let expected = [("api", false), ("stop", true), ("threads_for", true), ("loadmodel", true)];
        assert_eq!(found, expected.map(|(t, s)| (t.to_string(), s)));
    }

    #[test]
    fn big_files_lockfiles_and_unknown_types_are_not_searched() {
        let dir = project_with(&[
            ("real.rs", "let zebra_marker = 1;\n".into()),
            ("big.rs", format!("// zebra_marker\n{}", "x".repeat(MAX_FILE_BYTES as usize + 1))),
            ("Cargo.lock", "zebra_marker\n".into()),
            ("package-lock.json", "{\"zebra_marker\": 1}\n".into()),
            ("weights.bin", "zebra_marker\n".into()),
        ]);
        let names: Vec<String> = source_files(dir.path()).into_iter().map(|(rel, _)| rel).collect();
        assert_eq!(names, vec!["real.rs"]);
    }

    #[test]
    fn a_definition_is_matched_as_a_whole_word() {
        assert!(!defines("fn threads_for_all() {}", "threads_for"), "a longer name");
        assert!(!defines("fn xthreads_for() {}", "threads_for"), "a name ending the same way");
        assert!(defines("fn threads_for() {}", "threads_for"));
    }

    #[test]
    fn a_rare_word_outranks_a_common_one() {
        // "alpha" is on five lines, "zeta" on one: the zeta line says more.
        let text = "alpha 1\nalpha 2\nalpha 3\nalpha 4\nalpha 5\nzeta here\n\n";
        assert_eq!(ranked_lines(text, "alpha zeta").first(), Some(&5));

        // The same across files: the one file with "zeta" beats five with "alpha".
        let mut files: Vec<(&str, String)> = ["a1.rs", "a2.rs", "a3.rs", "a4.rs", "a5.rs"]
            .iter()
            .map(|n| (*n, "alpha\nalpha\nalpha\n".to_string()))
            .collect();
        files.push(("z.rs", "zeta\n".into()));
        let dir = project_with(&files);
        assert_eq!(best_files(dir.path(), "alpha zeta", 3)[0].path, "z.rs");
    }

    #[test]
    fn the_definition_line_outranks_the_lines_that_only_use_it() {
        let text = "run_job(1);\nrun_job(2);\nrun_job(3);\nfn run_job(n: u32) {}\n";
        assert_eq!(ranked_lines(text, "`run_job`").first(), Some(&3));
    }

    #[test]
    fn a_file_matching_more_of_the_question_comes_first() {
        // Both files' best line matches one word; p.rs also has the other word on another line.
        let dir = project_with(&[("p.rs", "alpha\nzeta\n".into()), ("q.rs", "zeta\n".into())]);
        assert_eq!(best_files(dir.path(), "alpha zeta", 2)[0].path, "p.rs");
    }

    #[test]
    fn a_file_named_for_the_question_comes_first() {
        let dir = project_with(&[
            ("walker.py", "def walk(): pass  # the walker\n".into()),
            ("other.py", "# uses the walker\nimport os\n".into()),
        ]);
        assert_eq!(best_files(dir.path(), "walker", 2)[0].path, "walker.py");
    }

    #[test]
    fn only_matching_lines_are_returned() {
        let dir = project_with(&[("a.rs", "one\nzeta two\nthree\nfour\n".into())]);
        let hits = best_files(dir.path(), "zeta", 1);
        assert_eq!(hits[0].lines, vec![(2, "zeta two".to_string())]);
    }

    #[test]
    fn the_defining_file_comes_first() {
        let dir = project();
        let found = search(dir.path(), "Which file defines `threads_for`?");
        let first = found.lines().nth(1).unwrap();
        assert_eq!(first, "src/engine.rs:1: fn threads_for(layers: u32) -> u32 {", "{found}");
    }

    #[test]
    fn shortened_paths_snap_to_the_one_real_file() {
        let dir = project();
        assert_eq!(snap_path(dir.path(), "src/engine.rs").as_deref(), Some("src/engine.rs"));
        assert_eq!(snap_path(dir.path(), "/engine.rs").as_deref(), Some("src/engine.rs"));
        assert_eq!(snap_path(dir.path(), "`App.vue`").as_deref(), Some("ui/App.vue"));
        assert_eq!(snap_path(dir.path(), "src/missing.rs"), None);
        assert_eq!(snap_path(dir.path(), "ine.rs"), None, "a suffix must start at a folder boundary");
    }

    #[test]
    fn the_best_line_ignores_common_words() {
        let text = "// Turns what's open in the editor into the code block.\n\
                    /** Directories that never help. */\n\
                    export const SKIP_DIRS = new Set([\"node_modules\", \".git\"]);\n\
                    export function formatOverview(workspace) {}\n";
        let goal = "Add \"models\" to SKIP_DIRS in the extension's context.ts, so the workspace overview stops listing the model files.";
        assert_eq!(ranked_lines(text, goal).first(), Some(&2));
        assert!(ranked_lines(text, "what is the weather").is_empty());
    }

    #[test]
    fn a_definition_must_start_its_line() {
        assert!(defines("pub(crate) async fn threads_for(x: u32) {", "threads_for"));
        assert!(defines("export function threadsFor() {", "threadsfor"));
        assert!(!defines("write(\"a.rs\", \"fn threads_for() {}\");", "threads_for"));
        assert!(!defines("// see fn threads_for", "threads_for"));
    }

    #[test]
    fn plain_words_find_ui_code() {
        let dir = project();
        let found = search(dir.path(), "which file has the Stop button?");
        assert!(found.lines().nth(1).unwrap().starts_with("ui/App.vue:1: "), "{found}");
    }

    #[test]
    fn skips_dependencies_and_submodules() {
        let dir = project();
        let found = search(dir.path(), "threads_for");
        assert!(!found.contains("node_modules") && !found.contains("vendor"), "{found}");
    }

    #[test]
    fn nothing_for_a_question_with_no_matches() {
        let dir = project();
        assert_eq!(search(dir.path(), "what is the weather like"), "");
    }
}
