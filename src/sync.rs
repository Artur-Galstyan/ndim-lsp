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
    enc: Encoding,
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
        let lines = LineIndex::new(&text);
        let start = lines.offset_clamped(&text, range.start, enc)?;
        let end = lines.offset_clamped(&text, range.end, enc)?;
        if start > end {
            return None;
        }
        text.replace_range(start..end, &change.text);
    }
    Some(text)
}

/// The unit of `Position.character`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Encoding {
    Utf8,
    #[default]
    Utf16,
}

impl Encoding {
    fn units(self, ch: char) -> u32 {
        match self {
            Encoding::Utf8 => ch.len_utf8() as u32,
            Encoding::Utf16 => ch.len_utf16() as u32,
        }
    }

    fn count(self, s: &str) -> u32 {
        match self {
            Encoding::Utf8 => s.len() as u32,
            Encoding::Utf16 => s.encode_utf16().count() as u32,
        }
    }
}

/// The lines of one text under the LSP line model: a line ends at `\n`,
/// `\r\n`, or `\r`. Tree-sitter rows end at `\n` only, so a row can hold
/// more than one LSP line.
#[derive(Debug, Default)]
pub struct LineIndex {
    /// Byte offset of each line start.
    starts: Vec<usize>,
    /// The line of each tree-sitter row start.
    rows: Vec<u32>,
}

impl LineIndex {
    pub fn new(text: &str) -> LineIndex {
        let bytes = text.as_bytes();
        let mut starts = vec![0];
        let mut rows = vec![0];
        for (i, &b) in bytes.iter().enumerate() {
            let ends = b == b'\n' || (b == b'\r' && bytes.get(i + 1) != Some(&b'\n'));
            if ends {
                starts.push(i + 1);
            }
            if b == b'\n' {
                rows.push(starts.len() as u32 - 1);
            }
        }
        LineIndex { starts, rows }
    }

    /// The start of `line` and the end of its text, before the line break.
    fn bounds(&self, text: &str, line: usize) -> Option<(usize, usize)> {
        let start = *self.starts.get(line)?;
        let end = match self.starts.get(line + 1) {
            Some(&next) if text.as_bytes()[..next].ends_with(b"\r\n") => next - 2,
            Some(&next) => next - 1,
            None => text.len(),
        };
        Some((start, end))
    }

    pub fn position(&self, text: &str, byte: usize, enc: Encoding) -> Position {
        let line = self.starts.partition_point(|&s| s <= byte) - 1;
        let (start, end) = self.bounds(text, line).unwrap_or((0, 0));
        let character = text.get(start..byte.min(end)).map_or(0, |s| enc.count(s));
        Position::new(line as u32, character)
    }

    /// The byte of `position`, or `None` if the line or the character is past
    /// its end, or inside a char.
    pub fn offset(&self, text: &str, position: Position, enc: Encoding) -> Option<usize> {
        let (start, end) = self.bounds(text, position.line as usize)?;
        let byte = self.offset_in(text, start, end, position.character, enc)?;
        (byte < end || self.character_count(text, start, end, enc) == position.character)
            .then_some(byte)
    }

    /// As `offset`, but a position past the end of a line or of the text
    /// means that end, as the LSP spec says.
    pub fn offset_clamped(&self, text: &str, position: Position, enc: Encoding) -> Option<usize> {
        let Some((start, end)) = self.bounds(text, position.line as usize) else {
            return Some(text.len());
        };
        self.offset_in(text, start, end, position.character, enc)
    }

    fn offset_in(&self, text: &str, start: usize, end: usize, character: u32, enc: Encoding) -> Option<usize> {
        let mut units = 0u32;
        for (i, ch) in text[start..end].char_indices() {
            if units == character {
                return Some(start + i);
            }
            units += enc.units(ch);
            if units > character {
                return None;
            }
        }
        Some(end)
    }

    fn character_count(&self, text: &str, start: usize, end: usize, enc: Encoding) -> u32 {
        enc.count(&text[start..end])
    }

    /// The length of the text of `line`.
    pub fn line_length(&self, text: &str, line: u32, enc: Encoding) -> Option<u32> {
        let (start, end) = self.bounds(text, line as usize)?;
        Some(self.character_count(text, start, end, enc))
    }

    /// The lines that the tree-sitter row `row` holds.
    pub fn row_lines(&self, row: usize) -> std::ops::Range<u32> {
        let Some(&first) = self.rows.get(row) else {
            return 0..0;
        };
        let next = self.rows.get(row + 1).copied().unwrap_or(self.starts.len() as u32);
        first..next
    }
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

    fn apply(current: Option<&str>, changes: &[TextDocumentContentChangeEvent]) -> Option<String> {
        apply_changes(current, changes, Encoding::Utf16)
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
        let got = apply(
            Some("ab\ncd\n"),
            &[edit(0, 1, 0, 1, "X"), edit(1, 0, 1, 2, "Y")],
        );
        assert_eq!(got.as_deref(), Some("aXb\nY\n"));
        let got = apply(
            Some("abc"),
            &[edit(0, 0, 0, 1, "xyz"), edit(0, 3, 0, 4, "")],
        );
        assert_eq!(got.as_deref(), Some("xyzc"));
    }

    #[test]
    fn change_without_range_replaces_the_text() {
        let got = apply(
            Some("old"),
            &[edit(0, 0, 0, 1, "z"), full("new"), edit(0, 3, 0, 3, "!")],
        );
        assert_eq!(got.as_deref(), Some("new!"));
        assert_eq!(
            apply(None, &[full("fresh")]).as_deref(),
            Some("fresh")
        );
        assert_eq!(apply(None, &[edit(0, 0, 0, 0, "x")]), None);
        assert_eq!(apply(Some("a"), &[]), None);
    }

    #[test]
    fn columns_count_utf16_units() {
        let text = "é\u{1F600}x\n";
        assert_eq!(
            apply(Some(text), &[edit(0, 3, 0, 4, "y")]).as_deref(),
            Some("é\u{1F600}y\n")
        );
        assert_eq!(
            apply(Some(text), &[edit(0, 1, 0, 3, "")]).as_deref(),
            Some("éx\n")
        );
        assert_eq!(apply(Some(text), &[edit(0, 2, 0, 3, "")]), None);
    }

    #[test]
    fn crlf_and_lone_cr_end_a_line() {
        let text = "a\r\nb\rc\n";
        assert_eq!(
            apply(Some(text), &[edit(1, 0, 1, 1, "B")]).as_deref(),
            Some("a\r\nB\rc\n")
        );
        assert_eq!(
            apply(Some(text), &[edit(2, 1, 2, 1, "C")]).as_deref(),
            Some("a\r\nb\rcC\n")
        );
        assert_eq!(
            apply(Some(text), &[edit(0, 9, 0, 9, "!")]).as_deref(),
            Some("a!\r\nb\rc\n")
        );
    }

    #[test]
    fn line_index_maps_bytes_and_rows_to_lsp_lines() {
        let text = "a\r\nb\rcé\n";
        let index = LineIndex::new(text);
        let p = |line, character| Position::new(line, character);
        let cases = [(0, p(0, 0)), (1, p(0, 1)), (3, p(1, 0)), (5, p(2, 0)), (6, p(2, 1)), (9, p(3, 0))];
        for (byte, position) in cases {
            assert_eq!(index.position(text, byte, Encoding::Utf16), position, "{byte}");
            assert_eq!(index.offset(text, position, Encoding::Utf16), Some(byte), "{byte}");
        }
        assert_eq!(index.position(text, 8, Encoding::Utf8), p(2, 3));
        assert_eq!(index.offset(text, p(2, 3), Encoding::Utf16), None);
        assert_eq!(index.offset(text, p(4, 0), Encoding::Utf16), None);
        assert_eq!(index.offset_clamped(text, p(2, 9), Encoding::Utf16), Some(8));
        assert_eq!(index.line_length(text, 0, Encoding::Utf16), Some(1));
        assert_eq!(index.line_length(text, 2, Encoding::Utf8), Some(3));
        assert_eq!((index.row_lines(0), index.row_lines(1), index.row_lines(2)), (0..1, 1..3, 3..4));
    }

    #[test]
    fn positions_past_the_end_clamp() {
        assert_eq!(
            apply(Some("ab\ncd"), &[edit(0, 99, 0, 99, "!")]).as_deref(),
            Some("ab!\ncd")
        );
        assert_eq!(
            apply(Some("ab\ncd"), &[edit(7, 0, 7, 3, "!")]).as_deref(),
            Some("ab\ncd!")
        );
        assert_eq!(apply(Some("ab"), &[edit(0, 2, 0, 1, "!")]), None);
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
