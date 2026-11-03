//! Learning from ratings without training: rated replies become examples in later prompts.
//!
//! A reply rated good, or a bad reply with a correction (the correction is the good answer),
//! becomes a demonstrated exchange. Before each reply, the examples whose question means about
//! the same as the new message go into the volatile header as "User: ... / Xenon: ..." turns. This
//! base model imitates demonstrated turns far more reliably than instructions (see app/README.md
//! on the closing-question fix), so a rating changes later replies on the very next turn.
//!
//! Examples are re-fed on every turn they appear in, so each one costs prefill time. That is why
//! only close matches are used, at most `MAX_EXAMPLES`, each cut to `MAX_EXAMPLE_CHARS`.

use std::collections::HashMap;
use std::sync::Mutex;

use tauri::AppHandle;

use crate::feedback::{self, FeedbackEntry};

pub const MAX_EXAMPLES: usize = 2;
/// Cosine similarity between the new message and an example's question. Below this, an example
/// is more likely to steer the reply off topic than help it.
const MIN_SIMILARITY: f32 = 0.55;
const MAX_EXAMPLE_CHARS: usize = 400;
/// Examples embedded per turn, so a long backlog of ratings catches up gradually.
const EMBED_PER_TURN: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub key: String,
    pub question: String,
    pub answer: String,
}

/// Question embeddings by example key. Rebuilt from feedback.jsonl as needed, so it never has to
/// be saved.
static EMBEDDINGS: Mutex<Option<HashMap<String, Vec<f32>>>> = Mutex::new(None);

fn cut(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    if let Some(end) = out.rfind(['.', '!', '?']) {
        out.truncate(end + 1);
    }
    out
}

/// The current examples: each reply rated good, and each bad reply with a correction, using the
/// latest rating for each reply.
pub fn examples_from(entries: &[FeedbackEntry]) -> Vec<Example> {
    feedback::latest_by_message(entries)
        .into_iter()
        .filter_map(|entry| {
            let question = entry.history.iter().rev().find(|t| t.role == "user")?.content.clone();
            let answer = match (entry.rating.as_str(), &entry.correction) {
                ("bad", Some(correction)) => correction.clone(),
                ("good", _) => entry.reply.clone(),
                _ => return None,
            };
            if question.trim().is_empty() || answer.trim().is_empty() {
                return None;
            }
            Some(Example {
                key: format!("{}:{}", entry.message_id, answer),
                question: cut(&question, MAX_EXAMPLE_CHARS),
                answer: cut(&answer, MAX_EXAMPLE_CHARS),
            })
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The closest examples to the message, best last so the strongest one sits nearest the reply.
pub fn choose(
    examples: &[Example],
    embeddings: &HashMap<String, Vec<f32>>,
    query: &[f32],
    limit: usize,
) -> Vec<Example> {
    let mut scored: Vec<(f32, &Example)> = examples
        .iter()
        .filter_map(|example| {
            let vector = embeddings.get(&example.key)?;
            let similarity = dot(query, vector);
            (vector.len() == query.len() && similarity >= MIN_SIMILARITY).then_some((similarity, example))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<Example> = scored.into_iter().take(limit).map(|(_, e)| e.clone()).collect();
    chosen.reverse();
    chosen
}

/// Examples for one turn, or none when there are no ratings or the voice sidecar cannot embed.
pub fn examples_for(app: &AppHandle, message: &str) -> Vec<Example> {
    let Ok(path) = feedback::feedback_path(app) else {
        return Vec::new();
    };
    let examples = examples_from(&feedback::read_entries(&path));
    if examples.is_empty() {
        return Vec::new();
    }
    let Ok(mut guard) = EMBEDDINGS.lock() else {
        return Vec::new();
    };
    let embeddings = guard.get_or_insert_with(HashMap::new);

    let missing: Vec<&Example> = examples
        .iter()
        .filter(|e| !embeddings.contains_key(&e.key))
        .take(EMBED_PER_TURN)
        .collect();
    let mut texts = vec![message.to_string()];
    texts.extend(missing.iter().map(|e| e.question.clone()));
    let Some(mut vectors) = crate::voice::embed_texts(app, &texts) else {
        return Vec::new();
    };
    let query = vectors.remove(0);
    for (example, vector) in missing.iter().zip(vectors) {
        embeddings.insert(example.key.clone(), vector);
    }
    choose(&examples, embeddings, &query, MAX_EXAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::FeedbackTurn;

    fn rated(id: &str, question: &str, reply: &str, rating: &str, correction: Option<&str>) -> FeedbackEntry {
        FeedbackEntry {
            id: id.to_string(),
            created_at: 0,
            conversation_id: "c".to_string(),
            message_id: format!("m{id}"),
            rating: rating.to_string(),
            correction: correction.map(str::to_string),
            history: vec![FeedbackTurn { role: "user".to_string(), content: question.to_string() }],
            reply: reply.to_string(),
            model: "m".to_string(),
            agent: "dementia".to_string(),
        }
    }

    #[test]
    fn good_replies_and_corrections_become_examples() {
        let examples = examples_from(&[
            rated("1", "Capital of France?", "Paris.", "good", None),
            rated("2", "Capital of Spain?", "Lisbon.", "bad", Some("Madrid.")),
            rated("3", "Capital of Peru?", "Quito.", "bad", None),
        ]);
        let pairs: Vec<(&str, &str)> = examples.iter().map(|e| (e.question.as_str(), e.answer.as_str())).collect();
        assert_eq!(pairs, vec![("Capital of France?", "Paris."), ("Capital of Spain?", "Madrid.")]);
    }

    #[test]
    fn only_close_examples_are_used_strongest_last() {
        let examples = examples_from(&[
            rated("1", "a", "close", "good", None),
            rated("2", "b", "closest", "good", None),
            rated("3", "c", "far", "good", None),
        ]);
        let mut embeddings = HashMap::new();
        embeddings.insert(examples[0].key.clone(), vec![0.8, 0.6]);
        embeddings.insert(examples[1].key.clone(), vec![1.0, 0.0]);
        embeddings.insert(examples[2].key.clone(), vec![0.0, 1.0]);
        let chosen = choose(&examples, &embeddings, &[1.0, 0.0], MAX_EXAMPLES);
        let answers: Vec<&str> = chosen.iter().map(|e| e.answer.as_str()).collect();
        assert_eq!(answers, vec!["close", "closest"]);
    }

    #[test]
    fn long_answers_are_cut_at_a_sentence() {
        let long = format!("{} End.", "word ".repeat(200));
        let cut_text = cut(&long, MAX_EXAMPLE_CHARS);
        assert!(cut_text.chars().count() <= MAX_EXAMPLE_CHARS);
    }
}
