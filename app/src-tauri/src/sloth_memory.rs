//! Persistent cross-conversation memory for the "Sloth" agent.
//!
//! Dementia (the default agent) behaves exactly like the app always has -- each conversation's
//! own history is used, nothing survives outside that chat window. Sloth additionally reads and
//! writes a store of short, distilled "facts" that persists across every conversation and every
//! app restart -- see `inference.rs`'s `generate_blocking` for where facts get injected into the
//! prompt (every Sloth turn) and extracted from it (after every Sloth turn completes).
//!
//! Facts are automatically extracted by asking the model itself whether the exchange revealed
//! anything new and durable worth remembering. A small model sometimes gets that wrong, so the
//! "Sloth Memory..." UI (see App.vue) lets facts be deleted individually or all at once.
//!
//! Storage is SQLite at `<data dir>/sloth_memory.sqlite`, with no cap on the number of facts.
//! Each fact can carry a 384-dimension sentence embedding from the voice sidecar
//! (`voice-pipeline/embedder.py`), so facts are found by meaning: "what is my pet called" finds
//! "the user's dog is named Rex". Facts saved before the embedder was ready are embedded later
//! (see `missing_embeddings`), and keyword matching covers any fact without one, or every fact
//! when the sidecar is not running. The pre-SQLite `sloth_memory.json` is imported once and
//! left in place.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// How many facts one Sloth turn sends to the model. The facts sit in the volatile header, which
/// is re-evaluated on every turn, so each one costs tokens per reply. The store can grow without
/// slowing replies down because only this many are chosen (see `relevant_facts`).
pub const MAX_PROMPT_FACTS: usize = 8;

/// The Memory panel shows the newest facts only. Thousands of rows would make it unusable.
const PANEL_FACTS: usize = 1000;

/// Below this cosine similarity, an embedded fact is not considered related to the message.
const MIN_SIMILARITY: f32 = 0.2;

/// Keyword overlap adds this much at most on top of similarity, to break near-ties toward facts
/// that also share words with the message.
const KEYWORD_WEIGHT: f32 = 0.15;

const DB_NAME: &str = "sloth_memory.sqlite";
const LEGACY_JSON: &str = "sloth_memory.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlothFact {
    pub id: String,
    pub text: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredFact {
    pub fact: SlothFact,
    pub embedding: Option<Vec<f32>>,
    /// The fact's keywords, sorted. Worked out once when loaded, since splitting every fact's
    /// text on every turn was most of the search time for a large store.
    pub keywords: Vec<String>,
}

impl StoredFact {
    pub fn new(fact: SlothFact, embedding: Option<Vec<f32>>) -> Self {
        let keywords = keywords(&fact.text);
        StoredFact { fact, embedding, keywords }
    }
}

#[derive(Debug, Default, Deserialize)]
struct LegacyMemory {
    #[serde(default)]
    facts: Vec<SlothFact>,
}

/// Every fact, kept in memory between turns so a large store is not reread on each one. Any
/// write updates it, and a different database path (the data directory changed) reloads it.
static CACHE: Mutex<Option<(PathBuf, Vec<StoredFact>)>> = Mutex::new(None);

pub fn db_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(crate::memory::effective_data_dir(app)?.join(DB_NAME))
}

fn legacy_json_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|dir| dir.join(LEGACY_JSON))
}

fn to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_blob(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.is_empty() || blob.len() % 4 != 0 {
        return None;
    }
    Some(blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Opens (creating if needed) the store at `path`, importing `legacy_json` if the store is new.
pub fn open_at(path: &Path, legacy_json: Option<&Path>) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create '{}': {e}", parent.display()))?;
    }
    let conn = Connection::open(path).map_err(|e| format!("Could not open '{}': {e}", path.display()))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS facts (
             id TEXT PRIMARY KEY,
             text TEXT NOT NULL,
             created_at INTEGER NOT NULL,
             embedding BLOB
         );
         CREATE INDEX IF NOT EXISTS facts_created ON facts(created_at);",
    )
    .map_err(|e| format!("Could not prepare the Sloth memory store: {e}"))?;

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;
    if count == 0 {
        if let Some(legacy) = legacy_json.filter(|p| p.exists()) {
            let raw = fs::read_to_string(legacy).map_err(|e| format!("Could not read '{}': {e}", legacy.display()))?;
            let old: LegacyMemory = serde_json::from_str(&raw).unwrap_or_default();
            for fact in old.facts {
                conn.execute(
                    "INSERT OR IGNORE INTO facts (id, text, created_at) VALUES (?1, ?2, ?3)",
                    params![fact.id, fact.text, fact.created_at],
                )
                .map_err(|e| format!("Could not import Sloth facts: {e}"))?;
            }
        }
    }
    Ok(conn)
}

fn open(app: &AppHandle) -> Result<(PathBuf, Connection), String> {
    let path = db_path(app)?;
    let conn = open_at(&path, legacy_json_path(app).as_deref())?;
    Ok((path, conn))
}

fn read_all(conn: &Connection) -> Result<Vec<StoredFact>, String> {
    let mut stmt = conn
        .prepare("SELECT id, text, created_at, embedding FROM facts ORDER BY created_at, id")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            let blob: Option<Vec<u8>> = row.get(3)?;
            Ok(StoredFact::new(
                SlothFact { id: row.get(0)?, text: row.get(1)?, created_at: row.get(2)? },
                blob.as_deref().and_then(from_blob),
            ))
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// Every fact, oldest first, from the cache when it is current.
pub fn load_facts(app: &AppHandle) -> Result<Vec<StoredFact>, String> {
    let (path, conn) = open(app)?;
    let mut cache = CACHE.lock().map_err(|_| "Sloth memory cache lock poisoned".to_string())?;
    if let Some((cached_path, facts)) = cache.as_ref() {
        if *cached_path == path {
            return Ok(facts.clone());
        }
    }
    let facts = read_all(&conn)?;
    *cache = Some((path, facts.clone()));
    Ok(facts)
}

fn invalidate_cache() {
    if let Ok(mut cache) = CACHE.lock() {
        *cache = None;
    }
}

fn panel_view(facts: &[StoredFact]) -> Vec<SlothFact> {
    let start = facts.len().saturating_sub(PANEL_FACTS);
    facts[start..].iter().map(|f| f.fact.clone()).collect()
}

/// Saves one newly-extracted fact, skipping exact duplicates. Returns the facts the Memory panel
/// shows, so the caller can emit them without a second read.
pub fn append_fact(
    app: &AppHandle,
    text: String,
    created_at: i64,
    embedding: Option<Vec<f32>>,
) -> Result<Vec<SlothFact>, String> {
    let (_path, conn) = open(app)?;
    let duplicate: Option<i64> = conn
        .query_row("SELECT 1 FROM facts WHERE lower(text) = lower(?1) LIMIT 1", params![text], |row| row.get(0))
        .optional()
        .map_err(|e| e.to_string())?;
    if duplicate.is_none() {
        conn.execute(
            "INSERT INTO facts (id, text, created_at, embedding) VALUES (?1, ?2, ?3, ?4)",
            params![
                format!("{created_at}-{}", uuid_suffix()),
                text,
                created_at,
                embedding.as_deref().map(to_blob)
            ],
        )
        .map_err(|e| format!("Could not save Sloth fact: {e}"))?;
        invalidate_cache();
    }
    Ok(panel_view(&load_facts(app)?))
}

/// Facts that have no embedding yet, oldest first, up to `limit`.
pub fn missing_embeddings(facts: &[StoredFact], limit: usize) -> Vec<SlothFact> {
    facts.iter().filter(|f| f.embedding.is_none()).take(limit).map(|f| f.fact.clone()).collect()
}

pub fn store_embeddings(app: &AppHandle, items: &[(String, Vec<f32>)]) -> Result<(), String> {
    if items.is_empty() {
        return Ok(());
    }
    let (_path, mut conn) = open(app)?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for (id, vector) in items {
        tx.execute("UPDATE facts SET embedding = ?1 WHERE id = ?2", params![to_blob(vector), id])
            .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    invalidate_cache();
    Ok(())
}

/// Adds the facts from another store that this one does not already have. Import uses this so
/// memories from two machines are combined, not overwritten.
pub fn merge_from(local: &Path, incoming: &Path) -> Result<usize, String> {
    let conn = open_at(local, None)?;
    let before: i64 = conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0)).map_err(|e| e.to_string())?;
    conn.execute("ATTACH DATABASE ?1 AS incoming", params![incoming.to_string_lossy()])
        .map_err(|e| format!("Could not read '{}': {e}", incoming.display()))?;
    conn.execute(
        "INSERT OR IGNORE INTO facts (id, text, created_at, embedding)
         SELECT id, text, created_at, embedding FROM incoming.facts
         WHERE lower(text) NOT IN (SELECT lower(text) FROM main.facts)",
        [],
    )
    .map_err(|e| format!("Could not merge Sloth facts: {e}"))?;
    conn.execute("DETACH DATABASE incoming", []).map_err(|e| e.to_string())?;
    let after: i64 = conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0)).map_err(|e| e.to_string())?;
    invalidate_cache();
    Ok((after - before) as usize)
}

fn uuid_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    format!("{nanos:09}")
}

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

/// Keyword overlap for every fact, with rarer words counting for more (a query word found in few
/// facts says more than one found in most). Zero where nothing is shared.
fn keyword_scores(facts: &[StoredFact], query: &str) -> Vec<f32> {
    let query_words = keywords(query);
    let fact_words: Vec<&Vec<String>> = facts.iter().map(|f| &f.keywords).collect();
    // Counted once per query word. Counting inside the loop below would be quadratic when a
    // common word appears in most of the store.
    let weights: Vec<f32> = query_words
        .iter()
        .map(|word| {
            let containing = fact_words.iter().filter(|w| w.binary_search(word).is_ok()).count();
            (1.0 + facts.len() as f32 / containing.max(1) as f32).ln()
        })
        .collect();
    fact_words
        .iter()
        .map(|words| {
            query_words
                .iter()
                .zip(&weights)
                .filter(|(word, _)| words.binary_search(word).is_ok())
                .map(|(_, weight)| weight)
                .sum()
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Chooses the facts to send for one user message. With `query_embedding`, embedded facts rank by
/// similarity of meaning, nudged by shared keywords. Facts with no embedding, or every fact when
/// there is no query embedding, rank by keywords alone. Spare slots go to the newest facts. A
/// model this small cannot call a tool to look something up, so the choice is made here.
pub fn relevant_facts(
    facts: &[StoredFact],
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: usize,
) -> Vec<SlothFact> {
    let keyword = keyword_scores(facts, query);
    let top_keyword = keyword.iter().cloned().fold(0.0f32, f32::max);

    let mut scored: Vec<(f32, usize)> = Vec::new();
    for (index, fact) in facts.iter().enumerate() {
        let kw = if top_keyword > 0.0 { keyword[index] / top_keyword } else { 0.0 };
        let score = match (query_embedding, fact.embedding.as_deref()) {
            (Some(q), Some(v)) if v.len() == q.len() => {
                let similarity = dot(q, v);
                if similarity < MIN_SIMILARITY && kw == 0.0 {
                    continue;
                }
                similarity + KEYWORD_WEIGHT * kw
            }
            _ => {
                if kw == 0.0 {
                    continue;
                }
                // Keyword-only matches rank below a clear meaning match.
                MIN_SIMILARITY * kw
            }
        };
        scored.push((score, index));
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
    chosen.into_iter().map(|index| facts[index].fact.clone()).collect()
}

#[tauri::command]
pub fn list_sloth_facts(app: AppHandle) -> Result<Vec<SlothFact>, String> {
    Ok(panel_view(&load_facts(&app)?))
}

#[tauri::command]
pub fn delete_sloth_fact(app: AppHandle, id: String) -> Result<Vec<SlothFact>, String> {
    let (_path, conn) = open(&app)?;
    conn.execute("DELETE FROM facts WHERE id = ?1", params![id]).map_err(|e| e.to_string())?;
    invalidate_cache();
    Ok(panel_view(&load_facts(&app)?))
}

#[tauri::command]
pub fn clear_sloth_facts(app: AppHandle) -> Result<(), String> {
    let (_path, conn) = open(&app)?;
    conn.execute("DELETE FROM facts", []).map_err(|e| e.to_string())?;
    invalidate_cache();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(texts: &[&str]) -> Vec<StoredFact> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                StoredFact::new(SlothFact { id: i.to_string(), text: text.to_string(), created_at: i as i64 }, None)
            })
            .collect()
    }

    fn texts(facts: &[SlothFact]) -> Vec<&str> {
        facts.iter().map(|f| f.text.as_str()).collect()
    }

    fn unit(values: &[f32]) -> Vec<f32> {
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter().map(|v| v / norm).collect()
    }

    fn temp_db(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("xenon2-sloth-test-{}-{name}.sqlite", std::process::id()));
        let _ = fs::remove_file(&path);
        path
    }

    #[test]
    fn finds_an_old_fact_among_thousands() {
        let mut items = vec!["The user's dog is named Rex.".to_string()];
        items.extend((0..5000).map(|i| format!("The user visited city number {i}.")));
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        let chosen = relevant_facts(&stored(&refs), "What is my dog called?", None, MAX_PROMPT_FACTS);
        assert_eq!(chosen.len(), MAX_PROMPT_FACTS);
        assert!(texts(&chosen).contains(&"The user's dog is named Rex."));
    }

    #[test]
    fn rare_words_outrank_common_ones() {
        let facts = stored(&[
            "The user likes coffee in the morning.",
            "The user likes tea in the morning.",
            "The user owns a telescope.",
        ]);
        let chosen = relevant_facts(&facts, "telescope morning", None, 1);
        assert_eq!(texts(&chosen), vec!["The user owns a telescope."]);
    }

    #[test]
    fn meaning_beats_shared_words() {
        let mut facts = stored(&["The user's dog is named Rex.", "The user called their mother."]);
        facts[0].embedding = Some(unit(&[1.0, 0.1, 0.0]));
        facts[1].embedding = Some(unit(&[0.0, 0.2, 1.0]));
        let query = unit(&[0.9, 0.2, 0.1]);
        // "called" is shared with the wrong fact. The embedding still wins.
        let chosen = relevant_facts(&facts, "What is my pet called?", Some(&query), 1);
        assert_eq!(texts(&chosen), vec!["The user's dog is named Rex."]);
    }

    #[test]
    fn unrelated_embedded_facts_give_way_to_the_newest() {
        let mut facts = stored(&["old unrelated", "newest fact"]);
        facts[0].embedding = Some(unit(&[1.0, 0.0]));
        facts[1].embedding = Some(unit(&[1.0, 0.0]));
        let chosen = relevant_facts(&facts, "anything", Some(&unit(&[0.0, 1.0])), 1);
        assert_eq!(texts(&chosen), vec!["newest fact"]);
    }

    #[test]
    fn fills_spare_slots_with_the_newest_facts() {
        let facts = stored(&["first", "second", "third", "fourth"]);
        let chosen = relevant_facts(&facts, "nothing matches this", None, 2);
        assert_eq!(texts(&chosen), vec!["third", "fourth"]);
    }

    #[test]
    fn handles_an_empty_store() {
        assert!(relevant_facts(&[], "anything", None, MAX_PROMPT_FACTS).is_empty());
    }

    #[test]
    fn searching_100k_embedded_facts_is_fast() {
        let dims = 384;
        let facts: Vec<StoredFact> = (0..100_000)
            .map(|i| {
                StoredFact::new(
                    SlothFact { id: i.to_string(), text: format!("fact number {i}"), created_at: i },
                    Some(unit(&(0..dims).map(|d| ((i * 31 + d * 7) % 97) as f32 + 1.0).collect::<Vec<_>>())),
                )
            })
            .collect();
        let query = unit(&(0..dims).map(|d| (d % 5) as f32 + 1.0).collect::<Vec<_>>());
        let started = std::time::Instant::now();
        let chosen = relevant_facts(&facts, "fact", Some(&query), MAX_PROMPT_FACTS);
        let took = started.elapsed();
        assert_eq!(chosen.len(), MAX_PROMPT_FACTS);
        // Debug builds are several times slower than release. This bounds the debug build.
        assert!(took < std::time::Duration::from_secs(3), "took {took:?}");
        eprintln!("100k-fact search took {took:?}");
    }

    #[test]
    fn imports_the_old_json_store_once_and_round_trips_embeddings() {
        let db = temp_db("import");
        let legacy = std::env::temp_dir().join(format!("xenon2-sloth-test-{}-legacy.json", std::process::id()));
        fs::write(&legacy, r#"{"facts":[{"id":"1","text":"The user's dog is named Rex.","createdAt":5}]}"#).unwrap();

        let conn = open_at(&db, Some(&legacy)).unwrap();
        conn.execute("UPDATE facts SET embedding = ?1 WHERE id = '1'", params![to_blob(&[0.5, -0.25])]).unwrap();
        drop(conn);
        let facts = read_all(&open_at(&db, Some(&legacy)).unwrap()).unwrap();
        assert_eq!(facts.len(), 1, "a second open must not import again");
        assert_eq!(facts[0].fact.text, "The user's dog is named Rex.");
        assert_eq!(facts[0].embedding, Some(vec![0.5, -0.25]));
        assert!(legacy.exists(), "the old file is kept");

        let _ = fs::remove_file(&db);
        let _ = fs::remove_file(&legacy);
    }

    #[test]
    fn merge_adds_only_new_facts() {
        let local = temp_db("local");
        let incoming = temp_db("incoming");
        let a = open_at(&local, None).unwrap();
        a.execute("INSERT INTO facts (id, text, created_at) VALUES ('1', 'Likes tea.', 1)", []).unwrap();
        drop(a);
        let b = open_at(&incoming, None).unwrap();
        b.execute("INSERT INTO facts (id, text, created_at) VALUES ('9', 'likes tea.', 1)", []).unwrap();
        b.execute("INSERT INTO facts (id, text, created_at) VALUES ('2', 'Owns a telescope.', 2)", []).unwrap();
        drop(b);
        assert_eq!(merge_from(&local, &incoming).unwrap(), 1);
        assert_eq!(merge_from(&local, &incoming).unwrap(), 0);
        let _ = fs::remove_file(&local);
        let _ = fs::remove_file(&incoming);
    }
}
