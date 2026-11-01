//! Phase 7 follow-up: persistent cross-conversation memory for the "Sloth" agent.
//!
//! Dementia (the default agent) behaves exactly like the app always has -- each conversation's
//! own history is used, nothing survives outside that chat window. Sloth additionally reads and
//! writes a small store of short, distilled "facts" (`<app-data-dir>/sloth_memory.json`) that
//! persists across every conversation and every app restart -- see `inference.rs`'s
//! `generate_blocking` for where facts get injected into the prompt (every Sloth turn) and
//! extracted from it (after every Sloth turn completes).
//!
//! Facts are automatically extracted by asking the model itself (a second, small, low-
//! temperature generate() call after the main reply) whether the exchange revealed anything new
//! and durable worth remembering -- not user-curated. Given the underlying model is small, this
//! extraction step will sometimes get it wrong (miss something, or invent something) -- the
//! "Sloth Memory..." UI (see App.vue) lets facts be deleted individually or all at once so a bad
//! extraction doesn't have to live forever.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// Hard cap on stored facts, so the JSON file rewritten on every append stays small. Oldest facts
/// are dropped first once the cap is hit (see `append_fact`).
const MAX_FACTS: usize = 10_000;

/// How many facts one Sloth turn sends to the model. The facts sit in the volatile header, which
/// is re-evaluated on every turn, so each one costs tokens per reply. Storage can grow without
/// slowing replies down because only this many are chosen (see `relevant_facts`).
pub const MAX_PROMPT_FACTS: usize = 8;

const STOP_WORDS: &[&str] = &[
    "a", "about", "am", "an", "and", "are", "at", "be", "can", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "should", "so", "that", "the", "their", "them", "they", "this", "to", "user",
    "users", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
];

fn keywords(text: &str) -> Vec<String> {
    let mut words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOP_WORDS.contains(w))
        .map(str::to_string)
        .collect();
    words.sort();
    words.dedup();
    words
}

/// Chooses the facts to send for one user message: the best keyword matches, with rarer words
/// counting for more (a query word found in few facts says more than one found in most), then the
/// newest facts in whatever slots are left. A model this small cannot call a tool to look
/// something up, so the choice has to be made here, before generation.
pub fn relevant_facts(facts: &[SlothFact], query: &str, limit: usize) -> Vec<SlothFact> {
    let query_words = keywords(query);
    let fact_words: Vec<Vec<String>> = facts.iter().map(|f| keywords(&f.text)).collect();

    // Counted once per query word. Counting inside the loop below would be quadratic when a
    // common word appears in most of the store.
    let weights: Vec<f64> = query_words
        .iter()
        .map(|word| {
            let containing = fact_words.iter().filter(|w| w.binary_search(word).is_ok()).count();
            (1.0 + facts.len() as f64 / containing.max(1) as f64).ln()
        })
        .collect();

    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (index, words) in fact_words.iter().enumerate() {
        let mut score = 0.0;
        for (word, weight) in query_words.iter().zip(&weights) {
            if words.binary_search(word).is_ok() {
                score += weight;
            }
        }
        if score > 0.0 {
            scored.push((score, index));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));

    let mut chosen: Vec<usize> = scored.into_iter().take(limit).map(|(_, index)| index).collect();
    for index in (0..facts.len()).rev() {
        if chosen.len() >= limit {
            break;
        }
        if !chosen.contains(&index) {
            chosen.push(index);
        }
    }
    chosen.sort_unstable();
    chosen.into_iter().map(|index| facts[index].clone()).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlothFact {
    pub id: String,
    pub text: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlothMemory {
    #[serde(default)]
    pub facts: Vec<SlothFact>,
}

fn memory_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not resolve the app data directory: {e}"))?;
    Ok(dir.join("sloth_memory.json"))
}

/// Loads the fact store. A missing file (Sloth never used yet) is normal and returns an empty
/// store, not an error.
pub fn load_memory(app: &AppHandle) -> Result<SlothMemory, String> {
    let path = memory_path(app)?;
    if !path.exists() {
        return Ok(SlothMemory::default());
    }
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("Could not read '{}': {}", path.display(), e))?;
    serde_json::from_str(&raw).map_err(|e| format!("Could not parse sloth_memory.json: {e}"))
}

fn save_memory(app: &AppHandle, memory: &SlothMemory) -> Result<(), String> {
    let path = memory_path(app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Could not create '{}': {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(memory)
        .map_err(|e| format!("Could not serialize sloth_memory.json: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Could not write '{}': {}", path.display(), e))
}

/// Appends one newly-extracted fact, dropping the oldest fact first if already at `MAX_FACTS`.
/// Called from `inference.rs` after a Sloth turn's extraction step produces something worth
/// keeping. Returns the updated fact list (so the caller can emit it to the frontend without a
/// second read).
pub fn append_fact(app: &AppHandle, text: String, created_at: i64) -> Result<Vec<SlothFact>, String> {
    let mut memory = load_memory(app)?;
    if memory.facts.iter().any(|f| f.text.eq_ignore_ascii_case(&text)) {
        return Ok(memory.facts);
    }
    memory.facts.push(SlothFact {
        id: created_at.to_string(),
        text,
        created_at,
    });
    if memory.facts.len() > MAX_FACTS {
        let excess = memory.facts.len() - MAX_FACTS;
        memory.facts.drain(0..excess);
    }
    save_memory(app, &memory)?;
    Ok(memory.facts)
}

#[tauri::command]
pub fn list_sloth_facts(app: AppHandle) -> Result<Vec<SlothFact>, String> {
    Ok(load_memory(&app)?.facts)
}

#[tauri::command]
pub fn delete_sloth_fact(app: AppHandle, id: String) -> Result<Vec<SlothFact>, String> {
    let mut memory = load_memory(&app)?;
    memory.facts.retain(|f| f.id != id);
    save_memory(&app, &memory)?;
    Ok(memory.facts)
}

#[tauri::command]
pub fn clear_sloth_facts(app: AppHandle) -> Result<(), String> {
    save_memory(&app, &SlothMemory::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(texts: &[&str]) -> Vec<SlothFact> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| SlothFact { id: i.to_string(), text: text.to_string(), created_at: i as i64 })
            .collect()
    }

    fn texts(facts: &[SlothFact]) -> Vec<&str> {
        facts.iter().map(|f| f.text.as_str()).collect()
    }

    #[test]
    fn finds_an_old_fact_among_thousands() {
        let mut items = vec!["The user's dog is named Rex.".to_string()];
        items.extend((0..5000).map(|i| format!("The user visited city number {i}.")));
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        let chosen = relevant_facts(&store(&refs), "What is my dog called?", MAX_PROMPT_FACTS);
        assert_eq!(chosen.len(), MAX_PROMPT_FACTS);
        assert!(texts(&chosen).contains(&"The user's dog is named Rex."));
    }

    #[test]
    fn rare_words_outrank_common_ones() {
        let facts = store(&[
            "The user likes coffee in the morning.",
            "The user likes tea in the morning.",
            "The user owns a telescope.",
        ]);
        let chosen = relevant_facts(&facts, "telescope morning", 1);
        assert_eq!(texts(&chosen), vec!["The user owns a telescope."]);
    }

    #[test]
    fn fills_spare_slots_with_the_newest_facts() {
        let facts = store(&["first", "second", "third", "fourth"]);
        let chosen = relevant_facts(&facts, "nothing matches this", 2);
        assert_eq!(texts(&chosen), vec!["third", "fourth"]);
    }

    #[test]
    fn a_word_in_every_fact_stays_fast_at_the_cap() {
        let items: Vec<String> = (0..MAX_FACTS).map(|i| format!("The user visited city number {i}.")).collect();
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        let facts = store(&refs);
        let started = std::time::Instant::now();
        let chosen = relevant_facts(&facts, "Which city did I visit?", MAX_PROMPT_FACTS);
        assert_eq!(chosen.len(), MAX_PROMPT_FACTS);
        assert!(started.elapsed() < std::time::Duration::from_secs(1), "took {:?}", started.elapsed());
    }

    #[test]
    fn handles_an_empty_store() {
        assert!(relevant_facts(&[], "anything", MAX_PROMPT_FACTS).is_empty());
    }
}
