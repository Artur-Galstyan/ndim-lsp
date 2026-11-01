//! Document sync: how long a change waits before analysis, and how an
//! incremental change message becomes the new text.

use std::time::Duration;

use tower_lsp::lsp_types::{Position, TextDocumentContentChangeEvent};

use crate::DEBOUNCE;

/// The floor merges a burst: changes closer than this share one analysis.
const MIN_DEBOUNCE_MS: f64 = 20.0;
/// Quiet time per millisecond of a known incremental run.
const WARM_FACTOR: f64 = 0.5;
/// Quiet time per millisecond of a whole analysis: the next run can cost as
/// much, so the machine gets twice that time free.
const WHOLE_FACTOR: f64 = 2.0;
const EWMA_ALPHA: f64 = 0.5;

/// What kind of run a wall time came from.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Run {
    /// No base to reuse, or an incremental run that had to recompute
    /// everything: the cost of a whole analysis.
    Whole,
    Incremental,
}

/// The analysis cost of one document.
#[derive(Default)]
pub struct Pace {
    ewma_ms: Option<f64>,
    /// `ewma_ms` came from incremental runs, not from a whole analysis.
    warm: bool,
    /// Runs up to this rev are not representative (a root change forced them).
    forced_rev: Option<u64>,
}

impl Pace {
    /// The quiet time a change waits for. No keystroke rhythm enters: a key
    /// costs one analysis at most, and an estimate keeps that analysis cheap.
    pub fn debounce(&self) -> Duration {
        let max_ms = DEBOUNCE.as_secs_f64() * 1000.0;
        let ms = self.ewma_ms.map_or(max_ms, |ms| {
            let factor = if self.warm { WARM_FACTOR } else { WHOLE_FACTOR };
            (factor * ms).clamp(MIN_DEBOUNCE_MS, max_ms)
        });
        Duration::from_secs_f64(ms / 1000.0)
    }

    /// Record the cost of the run for `rev`, also if newer text replaced that
    /// text: a costly run must raise the debounce during typing.
    pub fn sampled(&mut self, rev: u64, wall: Duration, run: Run) {
        if self.forced_rev.is_some_and(|forced| rev <= forced) {
            return;
        }
        let ms = wall.as_secs_f64() * 1000.0;
        match run {
            Run::Whole => {
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

    /// The runs up to `forced_rev` use old roots or fill a cache that a root
    /// change emptied: their cost says nothing.
    pub fn reset(&mut self, forced_rev: u64) {
        self.failed();
        self.forced_rev = Some(forced_rev);
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
    fn pace_follows_the_cost() {
        let ms = Duration::from_millis;
        let mut p = Pace::default();
        assert_eq!(p.debounce(), DEBOUNCE, "no sample");
        p.sampled(1, ms(5), Run::Whole);
        assert_eq!(p.debounce(), ms(20), "floor");
        p.sampled(2, ms(60), Run::Whole);
        assert_eq!(p.debounce(), ms(120), "a whole analysis counts twice");
        p.sampled(3, ms(400), Run::Whole);
        assert_eq!(p.debounce(), DEBOUNCE, "capped");
        p.sampled(4, ms(80), Run::Incremental);
        assert_eq!(p.debounce(), ms(40), "the first warm sample replaces whole");
        p.sampled(5, ms(40), Run::Incremental);
        assert_eq!(p.debounce(), ms(30), "half of the average of 80 and 40");
        p.sampled(6, ms(900), Run::Incremental);
        assert_eq!(p.debounce(), DEBOUNCE, "capped");
        p.sampled(7, ms(1), Run::Incremental);
        p.failed();
        assert_eq!(p.debounce(), DEBOUNCE, "failure");
        p.sampled(8, ms(1), Run::Incremental);
        p.reset(10);
        p.sampled(9, ms(1), Run::Incremental);
        p.sampled(10, ms(1), Run::Incremental);
        assert_eq!(p.debounce(), DEBOUNCE, "root change");
        p.sampled(11, ms(1), Run::Incremental);
        assert_eq!(p.debounce(), ms(20));
    }
}
