//! Document sync: how long a change waits before analysis, and how an
//! incremental change message becomes the new text.

use std::time::{Duration, Instant};

use tower_lsp::lsp_types::{Position, TextDocumentContentChangeEvent};

use crate::DEBOUNCE;

/// A change that follows the previous one within this time is typing.
const TYPING_GAP: Duration = Duration::from_millis(400);
const MIN_DEBOUNCE_MS: f64 = 20.0;
const EWMA_ALPHA: f64 = 0.5;

/// What kind of run a wall time came from.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Run {
    /// No base to reuse: the cost of a whole analysis.
    Cold,
    Incremental,
    /// An incremental run that had to recompute everything.
    Fallback,
}

/// The change rhythm and the analysis cost of one document.
#[derive(Default)]
pub struct Pace {
    last_change: Option<Instant>,
    ewma_ms: Option<f64>,
    /// `ewma_ms` came from incremental runs, not from a whole analysis.
    warm: bool,
    /// The next sample is not representative (a root change forced it).
    skip_sample: bool,
}

impl Pace {
    pub fn opened(&mut self, now: Instant) {
        *self = Pace {
            last_change: Some(now),
            ..Pace::default()
        };
    }

    /// Record a change at `now` and return the quiet time it waits for.
    pub fn changed(&mut self, now: Instant) -> Duration {
        let typing = self
            .last_change
            .is_some_and(|t| now.saturating_duration_since(t) < TYPING_GAP);
        self.last_change = Some(now);
        match self.ewma_ms {
            Some(ms) if !typing => Duration::from_secs_f64(
                (2.0 * ms).clamp(MIN_DEBOUNCE_MS, DEBOUNCE.as_secs_f64() * 1000.0) / 1000.0,
            ),
            _ => DEBOUNCE,
        }
    }

    pub fn sampled(&mut self, wall: Duration, run: Run) {
        if std::mem::take(&mut self.skip_sample) {
            return;
        }
        let ms = wall.as_secs_f64() * 1000.0;
        match run {
            Run::Fallback => self.failed(),
            Run::Cold => {
                self.ewma_ms = Some(ms);
                self.warm = false;
            }
            Run::Incremental => {
                self.ewma_ms = match self.ewma_ms {
                    Some(old) if self.warm => Some(EWMA_ALPHA * ms + (1.0 - EWMA_ALPHA) * old),
                    _ => Some(ms),
                };
                self.warm = true;
            }
        }
    }

    pub fn failed(&mut self) {
        self.ewma_ms = None;
        self.warm = false;
    }

    /// The next analysis runs with new roots: its cost says nothing.
    pub fn reset(&mut self) {
        self.failed();
        self.skip_sample = true;
    }
}

/// The text after `changes`, applied in order, or `None` if one cannot be
/// applied: the caller then keeps the old text. A change without a range
/// replaces the whole text; `current` is `None` for an unknown document.
/// A position past the end of a line or of the text means that end.
pub fn apply_changes(
    current: Option<&str>,
    changes: &[TextDocumentContentChangeEvent],
) -> Option<String> {
    let (mut text, ranged) = match changes.iter().rposition(|c| c.range.is_none()) {
        Some(i) => (changes[i].text.clone(), &changes[i + 1..]),
        None => (current?.to_owned(), changes),
    };
    if changes.is_empty() {
        return None;
    }
    for change in ranged {
        let range = change.range?;
        let start = offset(&text, range.start)?;
        let end = offset(&text, range.end)?;
        if start > end {
            return None;
        }
        text.replace_range(start..end, &change.text);
    }
    Some(text)
}

/// Byte offset of a UTF-16 position. Lines end at `\n`, `\r\n`, or `\r`.
fn offset(text: &str, position: Position) -> Option<usize> {
    let bytes = text.as_bytes();
    let is_break = |b: &u8| *b == b'\n' || *b == b'\r';
    let mut start = 0;
    for _ in 0..position.line {
        let Some(p) = bytes[start..].iter().position(is_break) else {
            return Some(text.len());
        };
        start += p;
        start += if bytes[start] == b'\r' && bytes.get(start + 1) == Some(&b'\n') {
            2
        } else {
            1
        };
    }
    let end = bytes[start..]
        .iter()
        .position(is_break)
        .map_or(text.len(), |p| start + p);
    let mut units = 0u32;
    for (i, ch) in text[start..end].char_indices() {
        if units == position.character {
            return Some(start + i);
        }
        units += ch.len_utf16() as u32;
        if units > position.character {
            return None;
        }
    }
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Range;

    fn edit(sl: u32, sc: u32, el: u32, ec: u32, text: &str) -> TextDocumentContentChangeEvent {
        TextDocumentContentChangeEvent {
            range: Some(Range::new(Position::new(sl, sc), Position::new(el, ec))),
            range_length: None,
            text: text.into(),
        }
    }

    fn full(text: &str) -> TextDocumentContentChangeEvent {
        TextDocumentContentChangeEvent {
            range: None,
            range_length: None,
            text: text.into(),
        }
    }

    #[test]
    fn changes_apply_in_order() {
        let got = apply_changes(
            Some("ab\ncd\n"),
            &[edit(0, 1, 0, 1, "X"), edit(1, 0, 1, 2, "Y")],
        );
        assert_eq!(got.as_deref(), Some("aXb\nY\n"));
        let got = apply_changes(
            Some("abc"),
            &[edit(0, 0, 0, 1, "xyz"), edit(0, 3, 0, 4, "")],
        );
        assert_eq!(got.as_deref(), Some("xyzc"));
    }

    #[test]
    fn change_without_range_replaces_the_text() {
        let got = apply_changes(
            Some("old"),
            &[edit(0, 0, 0, 1, "z"), full("new"), edit(0, 3, 0, 3, "!")],
        );
        assert_eq!(got.as_deref(), Some("new!"));
        assert_eq!(
            apply_changes(None, &[full("fresh")]).as_deref(),
            Some("fresh")
        );
        assert_eq!(apply_changes(None, &[edit(0, 0, 0, 0, "x")]), None);
        assert_eq!(apply_changes(Some("a"), &[]), None);
    }

    #[test]
    fn columns_count_utf16_units() {
        let text = "é\u{1F600}x\n";
        assert_eq!(
            apply_changes(Some(text), &[edit(0, 3, 0, 4, "y")]).as_deref(),
            Some("é\u{1F600}y\n")
        );
        assert_eq!(
            apply_changes(Some(text), &[edit(0, 1, 0, 3, "")]).as_deref(),
            Some("éx\n")
        );
        assert_eq!(apply_changes(Some(text), &[edit(0, 2, 0, 3, "")]), None);
    }

    #[test]
    fn crlf_and_lone_cr_end_a_line() {
        let text = "a\r\nb\rc\n";
        assert_eq!(
            apply_changes(Some(text), &[edit(1, 0, 1, 1, "B")]).as_deref(),
            Some("a\r\nB\rc\n")
        );
        assert_eq!(
            apply_changes(Some(text), &[edit(2, 1, 2, 1, "C")]).as_deref(),
            Some("a\r\nb\rcC\n")
        );
        assert_eq!(
            apply_changes(Some(text), &[edit(0, 9, 0, 9, "!")]).as_deref(),
            Some("a!\r\nb\rc\n")
        );
    }

    #[test]
    fn positions_past_the_end_clamp() {
        assert_eq!(
            apply_changes(Some("ab\ncd"), &[edit(0, 99, 0, 99, "!")]).as_deref(),
            Some("ab!\ncd")
        );
        assert_eq!(
            apply_changes(Some("ab\ncd"), &[edit(7, 0, 7, 3, "!")]).as_deref(),
            Some("ab\ncd!")
        );
        assert_eq!(apply_changes(Some("ab"), &[edit(0, 2, 0, 1, "!")]), None);
    }

    #[test]
    fn pace_follows_the_rule() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut p = Pace::default();
        assert_eq!(p.changed(t0), DEBOUNCE, "no sample");
        p.sampled(ms(10), Run::Cold);
        assert_eq!(p.changed(t0 + ms(2000)), ms(20));
        assert_eq!(p.changed(t0 + ms(2100)), DEBOUNCE, "typing");
        p.sampled(ms(40), Run::Incremental);
        assert_eq!(p.changed(t0 + ms(3000)), ms(80));
        p.sampled(ms(20), Run::Incremental);
        assert_eq!(p.changed(t0 + ms(4000)), ms(60), "average of 40 and 20");
        p.sampled(ms(900), Run::Incremental);
        assert_eq!(p.changed(t0 + ms(5000)), DEBOUNCE, "capped");
        p.sampled(ms(1), Run::Fallback);
        assert_eq!(p.changed(t0 + ms(6000)), DEBOUNCE, "fallback");
        p.sampled(ms(1), Run::Incremental);
        p.failed();
        assert_eq!(p.changed(t0 + ms(7000)), DEBOUNCE, "failure");
        p.sampled(ms(1), Run::Incremental);
        p.reset();
        p.sampled(ms(1), Run::Incremental);
        assert_eq!(p.changed(t0 + ms(8000)), DEBOUNCE, "root change");
        p.sampled(ms(1), Run::Incremental);
        assert_eq!(p.changed(t0 + ms(9000)), ms(20));
    }
}
