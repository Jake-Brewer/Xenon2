/// Truncates a list to at most 10 items.
pub fn truncate(items: Vec<String>) -> Vec<String> {
    items.into_iter().take(10).collect()
}

/// Returns true if the list has 10 or fewer items.
pub fn validate(items: &[String]) -> bool {
    items.len() <= 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_up_to_ten() {
        let v: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        assert_eq!(truncate(v).len(), 10);
    }

    #[test]
    fn validate_accepts_ten() {
        let v: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        assert!(validate(&v));
    }

    #[test]
    fn validate_rejects_eleven() {
        let v: Vec<String> = (0..11).map(|i| i.to_string()).collect();
        assert!(!validate(&v));
    }
}
