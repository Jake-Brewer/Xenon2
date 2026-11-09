//! SEARCH/REPLACE edit blocks: the model writes only what changes, not the whole file.
//!
//! At ~23 tokens/s a 300-line rewrite takes about two minutes and risks dropping code the model
//! didn't mean to touch; an edit block takes seconds. The format is aider's, which models have
//! seen a lot of:
//!
//! ```text
//! <<<<<<< SEARCH
//! lines as they are now
//! =======
//! lines as they should be
//! >>>>>>> REPLACE
//! ```
//!
//! Applying is lenient, as aider found it must be (strict matching gave 9x more failed edits): an
//! exact match first, then one that ignores indentation and trailing spaces, with the new lines
//! re-indented to fit. A block must match exactly one place, so an edit never lands somewhere the
//! model didn't mean.

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub search: String,
    pub replace: String,
}

const SEARCH: &str = "<<<<<<< SEARCH";
const DIVIDER: &str = "=======";
const REPLACE: &str = ">>>>>>> REPLACE";

/// Every complete block in `text`, in order. An unfinished block at the end (a reply cut short)
/// is left out.
pub fn parse(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if !line.trim_end().ends_with(SEARCH) {
            continue;
        }
        let mut search = Vec::new();
        let mut replace = Vec::new();
        let mut in_replace = false;
        let mut closed = false;
        for line in lines.by_ref() {
            let marker = line.trim();
            if !in_replace && marker == DIVIDER {
                in_replace = true;
            } else if in_replace && marker.starts_with(REPLACE) {
                closed = true;
                break;
            } else if in_replace {
                replace.push(line);
            } else {
                search.push(line);
            }
        }
        if closed {
            blocks.push(Block { search: search.join("\n"), replace: replace.join("\n") });
        }
    }
    blocks
}

fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Where `search` occurs in `lines` (start index), comparing with `same`. `None` unless exactly
/// one place matches.
fn find_unique(lines: &[&str], search: &[&str], same: impl Fn(&str, &str) -> bool) -> Option<usize> {
    if search.is_empty() || search.len() > lines.len() {
        return None;
    }
    let mut found = None;
    for start in 0..=lines.len() - search.len() {
        if search.iter().enumerate().all(|(i, s)| same(lines[start + i], s)) {
            if found.is_some() {
                return None;
            }
            found = Some(start);
        }
    }
    found
}

/// The few lines of the file that share the most words with `wanted`, to show the model what is
/// really there when its SEARCH lines miss.
fn closest_lines(lines: &[&str], wanted: &str) -> Option<String> {
    let words = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|w| w.len() > 1).map(str::to_lowercase).collect()
    };
    let wanted = words(wanted);
    let (best, shared) = lines
        .iter()
        .enumerate()
        .map(|(i, l)| (i, words(l).intersection(&wanted).count()))
        .max_by_key(|&(i, n)| (n, std::cmp::Reverse(i)))?;
    if shared == 0 {
        return None;
    }
    let from = best.saturating_sub(1);
    let to = (best + 3).min(lines.len());
    Some(lines[from..to].join("\n"))
}

/// Applies `block` to `text`, or says why it couldn't.
pub fn apply(text: &str, block: &Block) -> Result<String, String> {
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let ends_with_newline = text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();
    // Blank lines at either end of a block are the model's spacing, not content to match.
    let search: Vec<&str> = block.search.lines().collect();
    let trim = |v: &[&str]| -> (usize, usize) {
        let first = v.iter().position(|l| !l.trim().is_empty()).unwrap_or(v.len());
        let last = v.iter().rposition(|l| !l.trim().is_empty()).map_or(first, |i| i + 1);
        (first, last)
    };
    let (first, last) = trim(&search);
    let search = &search[first..last];
    if search.is_empty() {
        return Err("a SEARCH section was empty".into());
    }
    let replace: Vec<&str> = block.replace.lines().collect();

    // A line the model copied without its start, e.g. `const X = ...` for `export const X = ...`.
    // Each missing start must leave at least half the line, so short fragments don't match.
    let clipped = |a: &str, b: &str| {
        let (a, b) = (a.trim(), b.trim());
        !b.is_empty() && a.ends_with(b) && b.len() * 2 >= a.len()
    };
    let (start, reindent, exact_lines) = if let Some(start) = find_unique(&lines, search, |a, b| a.trim_end() == b.trim_end()) {
        (start, false, true)
    } else if let Some(start) = find_unique(&lines, search, |a, b| a.trim() == b.trim()) {
        (start, true, true)
    } else if let Some(start) = find_unique(&lines, search, clipped) {
        (start, true, false)
    } else {
        let first_line = search[0].trim();
        let count = lines.iter().filter(|l| l.trim() == first_line).count();
        return Err(if count > 1 {
            format!("the SEARCH lines starting `{first_line}` match {count} places; include more lines so they match one")
        } else {
            let mut message = format!("the SEARCH lines starting `{first_line}` aren't in the file; copy them exactly from it");
            if let Some(near) = closest_lines(&lines, first_line) {
                message.push_str(&format!(". The closest lines in the file are:\n{near}"));
            }
            message
        });
    };
    // Matched ignoring indentation: shift the new lines by the difference on the first line.
    let (from, to) = (indent_of(search[0]).to_string(), indent_of(lines[start]).to_string());
    // What the model left off each matched line, to put back on the line that replaces it.
    let missing: Vec<&str> = if exact_lines {
        Vec::new()
    } else {
        search
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let real = lines[start + i].trim();
                &real[..real.len() - s.trim().len()]
            })
            .collect()
    };

    let mut out: Vec<String> = lines[..start].iter().map(|l| l.to_string()).collect();
    for (i, line) in replace.iter().enumerate() {
        let mut line = if !reindent {
            line.to_string()
        } else if line.trim().is_empty() {
            String::new()
        } else if let Some(rest) = line.strip_prefix(from.as_str()) {
            format!("{to}{rest}")
        } else {
            line.to_string()
        };
        // Line i of the replacement stands for line i of the search when the counts agree;
        // otherwise only the first line can be paired up.
        let paired = if replace.len() == search.len() { missing.get(i) } else if i == 0 { missing.first() } else { None };
        if let Some(head) = paired.filter(|h| !h.is_empty()) {
            let body = line.trim_start();
            if !body.is_empty() && !body.starts_with(head.trim()) {
                line = format!("{}{head}{body}", indent_of(&line));
            }
        }
        out.push(line);
    }
    out.extend(lines[start + search.len()..].iter().map(|l| l.to_string()));
    let mut joined = out.join(newline);
    if ends_with_newline {
        joined.push_str(newline);
    }
    Ok(joined)
}

/// Applies every block in order. All or nothing: if one fails, the text is unchanged.
pub fn apply_all(text: &str, blocks: &[Block]) -> Result<String, String> {
    let mut out = text.to_string();
    for (i, block) in blocks.iter().enumerate() {
        out = apply(&out, block).map_err(|e| format!("edit {} of {}: {e}", i + 1, blocks.len()))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n";

    fn block(search: &str, replace: &str) -> Block {
        Block { search: search.into(), replace: replace.into() }
    }

    #[test]
    fn parses_blocks_and_skips_an_unfinished_one() {
        let reply = "FILE: a.rs\n<<<<<<< SEARCH\n    let x = 1;\n=======\n    let x = 2;\n>>>>>>> REPLACE\nWHY: two\n<<<<<<< SEARCH\ncut";
        assert_eq!(parse(reply), vec![block("    let x = 1;", "    let x = 2;")]);
    }

    #[test]
    fn applies_an_exact_match() {
        let out = apply(FILE, &block("    let x = 1;", "    let x = 2;")).unwrap();
        assert_eq!(out, FILE.replace("x = 1", "x = 2"));
    }

    #[test]
    fn a_match_ignoring_indentation_is_reindented() {
        let out = apply(FILE, &block("let x = 1;\nprintln!(\"{x}\");", "let x = 2;\nprintln!(\"{x}\");")).unwrap();
        assert_eq!(out, FILE.replace("x = 1", "x = 2"));
    }

    #[test]
    fn a_line_copied_without_its_start_still_matches_and_keeps_it() {
        let file = "/** Skip these. */\nexport const SKIP = new Set([\"a\", \"b\"]);\n";
        let out = apply(file, &block("const SKIP = new Set([\"a\", \"b\"]);", "const SKIP = new Set([\"a\", \"b\", \"models\"]);")).unwrap();
        assert_eq!(out, "/** Skip these. */\nexport const SKIP = new Set([\"a\", \"b\", \"models\"]);\n");
    }

    #[test]
    fn a_short_fragment_does_not_match_a_long_line() {
        assert!(apply(FILE, &block("1;", "2;")).is_err());
    }

    #[test]
    fn a_miss_shows_the_closest_real_lines() {
        let err = apply(FILE, &block("let x = 5;\nprint(x)", "")).unwrap_err();
        assert!(err.contains("closest lines in the file are:\nfn main() {\n    let x = 1;"), "{err}");
    }

    #[test]
    fn keeps_windows_line_endings() {
        let crlf = FILE.replace('\n', "\r\n");
        let out = apply(&crlf, &block("    let x = 1;", "    let x = 2;")).unwrap();
        assert_eq!(out, crlf.replace("x = 1", "x = 2"));
    }

    #[test]
    fn refuses_a_block_that_matches_twice_or_not_at_all() {
        let twice = "a\nb\na\n";
        assert!(apply(twice, &block("a", "c")).unwrap_err().contains("2 places"));
        assert!(apply(FILE, &block("let y = 1;", "")).unwrap_err().contains("aren't in the file"));
    }

    #[test]
    fn deleting_lines_works() {
        let out = apply(FILE, &block("    let x = 1;\n", "")).unwrap();
        assert_eq!(out, "fn main() {\n    println!(\"{x}\");\n}\n");
    }

    #[test]
    fn all_or_nothing() {
        let blocks = [block("    let x = 1;", "    let x = 2;"), block("nope", "")];
        assert!(apply_all(FILE, &blocks).unwrap_err().starts_with("edit 2 of 2"));
    }
}
