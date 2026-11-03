//! Ratings on Xenon's replies, collected for learning from feedback (see `ONLINE_LEARNING.md`).
//!
//! Each rating is one JSON object per line in `<data dir>/feedback.jsonl`. The file is
//! append-only: changing a rating appends a new line, and the latest line for a message wins, so
//! a crash mid-write can only lose the newest rating, never earlier ones. Each line carries the
//! whole prompt history, so a training run never needs the conversation files.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::AppHandle;

const FILE_NAME: &str = "feedback.jsonl";
const RATINGS: &[&str] = &["good", "bad", "none"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedbackEntry {
    pub id: String,
    pub created_at: i64,
    pub conversation_id: String,
    pub message_id: String,
    /// "good", "bad", or "none" (the rating was cleared).
    pub rating: String,
    /// What Xenon should have said. A bad rating without one cannot be trained on directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction: Option<String>,
    /// Every turn before the rated reply, ending with the user turn it answered.
    pub history: Vec<FeedbackTurn>,
    pub reply: String,
    pub model: String,
    #[serde(default)]
    pub agent: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedbackSummary {
    pub good: usize,
    pub bad: usize,
    pub corrections: usize,
}

pub fn feedback_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(crate::memory::effective_data_dir(app)?.join(FILE_NAME))
}

fn append(path: &Path, entry: &FeedbackEntry) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create '{}': {e}", parent.display()))?;
    }
    let line = serde_json::to_string(entry).map_err(|e| format!("Could not serialize feedback: {e}"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Could not open '{}': {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("Could not write '{}': {e}", path.display()))
}

/// Every readable line. A damaged line is skipped, so one bad write cannot hide the rest.
pub fn read_entries(path: &Path) -> Vec<FeedbackEntry> {
    let Ok(raw) = fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// The current rating for each rated reply, in the order they were first rated. Cleared ratings
/// are left out. A regenerated reply keeps its message id, so the reply text is part of the key:
/// rating the new reply must not overwrite what the user said about the old one.
pub fn latest_by_message(entries: &[FeedbackEntry]) -> Vec<FeedbackEntry> {
    let mut order: Vec<(String, String)> = Vec::new();
    let mut latest: HashMap<(String, String), FeedbackEntry> = HashMap::new();
    for entry in entries {
        let key = (entry.message_id.clone(), entry.reply.clone());
        if !latest.contains_key(&key) {
            order.push(key.clone());
        }
        latest.insert(key, entry.clone());
    }
    order
        .into_iter()
        .filter_map(|id| latest.remove(&id))
        .filter(|entry| entry.rating != "none")
        .collect()
}

pub fn summarize(entries: &[FeedbackEntry]) -> FeedbackSummary {
    let mut summary = FeedbackSummary::default();
    for entry in latest_by_message(entries) {
        match entry.rating.as_str() {
            "good" => summary.good += 1,
            "bad" => summary.bad += 1,
            _ => {}
        }
        if entry.correction.is_some() {
            summary.corrections += 1;
        }
    }
    summary
}

/// Appends the lines from `incoming` that `local` does not already have, matched by entry id.
/// Import uses this so ratings collected on two machines are combined, not overwritten.
pub fn merge_into(local: &Path, incoming: &Path) -> Result<usize, String> {
    let known: HashSet<String> = read_entries(local).into_iter().map(|e| e.id).collect();
    let mut added = 0;
    for entry in read_entries(incoming) {
        if !known.contains(&entry.id) {
            append(local, &entry)?;
            added += 1;
        }
    }
    Ok(added)
}

fn validate(mut entry: FeedbackEntry) -> Result<FeedbackEntry, String> {
    if !RATINGS.contains(&entry.rating.as_str()) {
        return Err(format!("Unknown rating '{}'. Expected good, bad or none.", entry.rating));
    }
    entry.correction = entry
        .correction
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty());
    Ok(entry)
}

#[tauri::command]
pub fn record_feedback(app: AppHandle, entry: FeedbackEntry) -> Result<FeedbackSummary, String> {
    let path = feedback_path(&app)?;
    append(&path, &validate(entry)?)?;
    Ok(summarize(&read_entries(&path)))
}

#[tauri::command]
pub fn feedback_summary(app: AppHandle) -> Result<FeedbackSummary, String> {
    Ok(summarize(&read_entries(&feedback_path(&app)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, message: &str, rating: &str, correction: Option<&str>) -> FeedbackEntry {
        FeedbackEntry {
            id: id.to_string(),
            created_at: 0,
            conversation_id: "c".to_string(),
            message_id: message.to_string(),
            rating: rating.to_string(),
            correction: correction.map(str::to_string),
            history: vec![FeedbackTurn { role: "user".to_string(), content: "hi".to_string() }],
            reply: "hello".to_string(),
            model: "m".to_string(),
            agent: "dementia".to_string(),
        }
    }

    fn temp_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("xenon2-feedback-test-{}-{name}", std::process::id()));
        let _ = fs::remove_file(&path);
        path
    }

    #[test]
    fn latest_rating_wins_and_cleared_ratings_drop_out() {
        let entries = vec![
            entry("1", "a", "good", None),
            entry("2", "b", "bad", None),
            entry("3", "a", "bad", Some("Paris.")),
            entry("4", "b", "none", None),
        ];
        let current = latest_by_message(&entries);
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].id, "3");
        assert_eq!(summarize(&entries), FeedbackSummary { good: 0, bad: 1, corrections: 1 });
    }

    #[test]
    fn a_regenerated_reply_keeps_its_own_rating() {
        let mut old = entry("1", "a", "good", None);
        old.reply = "first reply".to_string();
        let mut new = entry("2", "a", "bad", None);
        new.reply = "second reply".to_string();
        let current = latest_by_message(&[old, new]);
        assert_eq!(current.len(), 2);
        assert_eq!(summarize(&current), FeedbackSummary { good: 1, bad: 1, corrections: 0 });
    }

    #[test]
    fn appends_survive_a_damaged_line() {
        let path = temp_file("damaged");
        append(&path, &entry("1", "a", "good", None)).unwrap();
        fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"{not json\n").unwrap();
        append(&path, &entry("2", "b", "bad", None)).unwrap();
        assert_eq!(read_entries(&path).len(), 2);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn merge_adds_only_entries_it_does_not_have() {
        let local = temp_file("local");
        let incoming = temp_file("incoming");
        append(&local, &entry("1", "a", "good", None)).unwrap();
        append(&incoming, &entry("1", "a", "good", None)).unwrap();
        append(&incoming, &entry("2", "b", "bad", None)).unwrap();
        assert_eq!(merge_into(&local, &incoming).unwrap(), 1);
        assert_eq!(merge_into(&local, &incoming).unwrap(), 0);
        assert_eq!(read_entries(&local).len(), 2);
        let _ = fs::remove_file(&local);
        let _ = fs::remove_file(&incoming);
    }

    #[test]
    fn rejects_unknown_ratings_and_blank_corrections() {
        assert!(validate(entry("1", "a", "great", None)).is_err());
        assert_eq!(validate(entry("1", "a", "bad", Some("   "))).unwrap().correction, None);
    }
}
