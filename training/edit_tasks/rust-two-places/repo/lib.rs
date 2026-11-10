/// Truncates a list to at most 10 items.
pub fn truncate(items: Vec<String>) -> Vec<String> {
    items.into_iter().take(10).collect()
}

/// Returns true if the list has 10 or fewer items.
pub fn validate(items: &[String]) -> bool {
    items.len() <= 10
}
