use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Range;

use crate::layers::map_layer_dims;

/// Name → shape resolution for the apply helpers. Implemented by plain
/// per-scope `HashMap`s and by `ScopeShapes` in `analysis`, which also sees
/// synthetic bindings without cloning the scope map (#43).
pub trait ShapeLookup {
    fn shape(&self, name: &str) -> Option<&Vec<String>>;
}

impl ShapeLookup for HashMap<String, Vec<String>> {
    fn shape(&self, name: &str) -> Option<&Vec<String>> {
        self.get(name)
    }
}

/// Starts a name in a dim that holds a value the analysis does not know: a
/// parameter, a dataclass field, or a size argument of an unknown name. Such a
/// dim is not a claim like an annotation symbol: it can equal any other dim,
/// so it never conflicts (`dims_conflict`). The mark stays in a shape for the
/// full analysis. `LayerShapeAnalysis::without_free_marks` removes it at the
/// end, so that only the analysis sees it.
pub(crate) const FREE_DIM: &str = "__free_";

/// The free dim of an axis that keeps its place in the shape but has a size
/// the file does not give. The reader sees `?`.
pub(crate) const UNKNOWN_DIM: &str = "__free_?";

/// The longest dim text that the analysis keeps. No dim of a real file comes
/// near it. A longer dim is `UNKNOWN_DIM`, so that a chain of lines that doubles
/// a dim stays small and the axis stays in the shape.
pub(crate) const MAX_DIM_TEXT: usize = 256;

/// The highest rank that the analysis keeps. A value with a higher rank has no
/// shape: the axes of a shape cannot stay when the rank itself is the problem.
pub(crate) const MAX_RANK: usize = 64;

/// `dim`, or `UNKNOWN_DIM` when `dim` is too long or holds an unknown size. A
/// form of an unknown size says nothing about the axis, so it does not grow.
pub(crate) fn bounded_dim(dim: String) -> String {
    if dim.len() > MAX_DIM_TEXT || (dim.contains('?') && dim.contains(UNKNOWN_DIM) && dim != UNKNOWN_DIM) {
        UNKNOWN_DIM.to_string()
    } else {
        dim
    }
}

/// `shape` with each dim in its limit. `None` when the rank is over its limit.
pub(crate) fn bounded_shape(shape: Vec<String>) -> Option<Vec<String>> {
    (shape.len() <= MAX_RANK).then(|| shape.into_iter().map(bounded_dim).collect())
}

/// `parts` with `separator` between them. The length comes first, so that many
/// long parts allocate nothing.
pub(crate) fn joined_dim<S: AsRef<str>>(parts: &[S], separator: &str) -> String {
    let separators = separator.len().saturating_mul(parts.len().saturating_sub(1));
    let text: usize = parts.iter().map(|part| part.as_ref().len()).sum();
    if text.saturating_add(separators) > MAX_DIM_TEXT {
        return UNKNOWN_DIM.to_string();
    }
    let parts: Vec<&str> = parts.iter().map(AsRef::as_ref).collect();
    bounded_dim(parts.join(separator))
}

pub(crate) fn is_free_dim(dim: &str) -> bool {
    dim.contains(FREE_DIM) || dim.contains(crate::layers::UNPROVEN_DIM)
}

/// Whether `dim` subtracts from a name: `n-1`. The value of the name is not
/// known, so only the same name proves that such a dim differs.
fn is_difference(dim: &str) -> bool {
    dim.contains('-') && dim.chars().any(|c| c.is_alphabetic() || c == '_')
}

/// Whether `dim` is one size with names that a form can use. `_` and `_n` are
/// separate sizes, `#n` can broadcast, `*b` and `...` have a variable rank, and
/// `n=4` and `?` are not names.
pub(crate) fn is_named_size(dim: &str) -> bool {
    let text = dim.replace(FREE_DIM, "");
    !text.contains(['#', '=', '?'])
        && !text.contains("...")
        && !text.starts_with('*')
        && !text.split(|c: char| !c.is_alphanumeric() && c != '_').any(|word| word.starts_with('_'))
}

/// Whether the annotation dim `dim` stands for any number of axes.
pub(crate) fn is_variadic_dim(dim: &str) -> bool {
    dim == "..." || dim.trim_start_matches(['#', '_']).starts_with('*')
}

/// The limit `l` and the size `e` of the whole form `min(l, e)`, when `l` is an
/// integer.
pub(crate) fn min_form(dim: &str) -> Option<(i64, &str)> {
    let inner = dim.strip_prefix(FREE_DIM).unwrap_or(dim).strip_prefix("min(")?.strip_suffix(')')?;
    let (limit, size) = inner.split_once(", ")?;
    let mut depth = 0i32;
    for c in size.chars() {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth < 0 {
            return None;
        }
    }
    Some((limit.parse().ok()?, size))
}

/// The parts of `text` between the commas outside all brackets.
pub(crate) fn split_top_level(text: &str) -> Vec<&str> {
    let (mut parts, mut depth, mut start) = (Vec::new(), 0usize, 0);
    for (at, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&text[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// `dim` in brackets when an operator of `operators` is outside all its
/// brackets. A sign at the start is no operator.
fn bracket_on(dim: &str, operators: &[char]) -> String {
    let mut depth = 0usize;
    for (at, c) in dim.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 && at > 0 && operators.contains(&c) => return format!("({dim})"),
            _ => {}
        }
    }
    dim.to_string()
}

/// `dim` as a factor of a product or a dividend: `n-1` becomes `(n-1)`.
pub(crate) fn factor_text(dim: &str) -> String {
    bracket_on(dim, &['+', '-'])
}

/// `dim` next to a `*` or a `/`: `t*u` becomes `(t*u)`, so that `s/(t*u)` keeps
/// its value.
pub(crate) fn operand_text(dim: &str) -> String {
    bracket_on(dim, &['+', '-', '*', '/', '%'])
}

/// `dim` less `amount`, with the integer at the end of `dim` folded: `n-1`
/// less 1 is `n-2`, and `n+1` less 1 is `n`. `None` when the integer overflows.
/// A `min` form less more than its limit is below 0 at every size, so the size
/// is `UNKNOWN_DIM`.
pub(crate) fn offset_dim(dim: &str, amount: i64) -> Option<String> {
    let (base, held) = match dim.rfind(['+', '-']) {
        Some(at) if at > 0 && dim[at + 1..].bytes().all(|b| b.is_ascii_digit()) => {
            (&dim[..at], dim[at..].parse::<i64>().ok()?)
        }
        _ => (dim, 0),
    };
    let total = held.checked_sub(amount)?;
    if min_form(base).is_some_and(|(limit, _)| limit.checked_add(total).is_none_or(|low| low < 0)) {
        return Some(UNKNOWN_DIM.to_string());
    }
    Some(match total {
        0 => base.to_string(),
        total => format!("{base}{total:+}"),
    })
}

/// Joins a free name to its integer: `__free_D__is_64`. The file binds `D` to
/// the literal 64 and an annotation also uses `D` as a symbol, for a size that
/// the value of the name does not bind. So the dim equals both the integer and
/// the symbol. It conflicts only with another integer.
pub(crate) const LITERAL_DIM: &str = "__is_";

/// The integer of a dim that is one `LITERAL_DIM` name.
pub(crate) fn literal_dim(dim: &str) -> Option<&str> {
    let (name, value) = dim.strip_prefix(FREE_DIM)?.rsplit_once(LITERAL_DIM)?;
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b >= 0x80;
    (name.bytes().all(is_word) && !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()))
        .then_some(value)
}

/// `dim`, an expression of `LITERAL_DIM` names and integers, as one
/// `LITERAL_DIM` name of its value: it still equals an annotation, but
/// conflicts with another integer. A value of 1 stays an integer, as a dim of
/// size 1 broadcasts. No size is negative, so such a value is `UNPROVEN_DIM`.
/// A dim with another name stays as it is.
pub(crate) fn fold_literal_marks(dim: String) -> String {
    if !dim.contains(LITERAL_DIM) || literal_dim(&dim).is_some() {
        return dim;
    }
    match fold_integers(&without_dim_marks(&dim)) {
        Some(value) => literal_mark(&dim, value),
        None => dim,
    }
}

/// `value` as the result of a size rule whose inputs have the literal marks of
/// `source`.
pub(crate) fn literal_mark(source: &str, value: i64) -> String {
    let name = source.find(FREE_DIM).and_then(|start| {
        let rest = &source[start + FREE_DIM.len()..];
        rest.find(LITERAL_DIM).map(|end| &rest[..end])
    });
    match name {
        _ if value < 0 => crate::layers::UNPROVEN_DIM.to_string(),
        Some(name) if value != 1 => format!("{FREE_DIM}{name}{LITERAL_DIM}{value}"),
        _ => value.to_string(),
    }
}

/// `text` as a reader sees it: a `LITERAL_DIM` name is its integer, and a free
/// name is the name.
fn without_dim_marks(text: &str) -> String {
    let mut text = text.to_string();
    while let Some(end) = text.find(LITERAL_DIM) {
        let start = text[..end].rfind(FREE_DIM).unwrap_or(end);
        text.replace_range(start..end + LITERAL_DIM.len(), "");
    }
    text.replace(FREE_DIM, "")
}

/// The value of `text` when it is integers with `+`, `-`, `*`, `//`, and
/// parentheses.
pub(crate) fn fold_integers(text: &str) -> Option<i64> {
    let tokens: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut at = 0;
    let value = fold_sum(&tokens, &mut at)?;
    (at == tokens.len()).then_some(value)
}

fn fold_sum(tokens: &[u8], at: &mut usize) -> Option<i64> {
    let mut value = fold_product(tokens, at)?;
    while let Some(&op @ (b'+' | b'-')) = tokens.get(*at) {
        *at += 1;
        let right = fold_product(tokens, at)?;
        value = if op == b'+' { value.checked_add(right)? } else { value.checked_sub(right)? };
    }
    Some(value)
}

fn fold_product(tokens: &[u8], at: &mut usize) -> Option<i64> {
    let mut value = fold_atom(tokens, at)?;
    loop {
        match tokens.get(*at..*at + 2) {
            Some([b'*', b'*']) => return None,
            Some([b'/', b'/']) => {
                *at += 2;
                let right = fold_atom(tokens, at)?;
                value = value.checked_div_euclid(Some(right).filter(|r| *r > 0)?)?;
            }
            _ if tokens.get(*at) == Some(&b'*') => {
                *at += 1;
                value = value.checked_mul(fold_atom(tokens, at)?)?;
            }
            _ => return Some(value),
        }
    }
}

fn fold_atom(tokens: &[u8], at: &mut usize) -> Option<i64> {
    match tokens.get(*at)? {
        b'(' => {
            *at += 1;
            let value = fold_sum(tokens, at)?;
            (tokens.get(*at) == Some(&b')')).then(|| *at += 1)?;
            Some(value)
        }
        b'-' => {
            *at += 1;
            fold_atom(tokens, at)?.checked_neg()
        }
        _ => {
            let start = *at;
            while tokens.get(*at).is_some_and(u8::is_ascii_digit) {
                *at += 1;
            }
            std::str::from_utf8(&tokens[start..*at]).ok()?.parse().ok()
        }
    }
}

/// Whether the two dims are different at run time, as far as the file proves.
/// A free dim conflicts with nothing. A difference conflicts only with a dim of
/// the same name and another integer: `n-1` with `n`, but not with `m` or `4`.
pub(crate) fn dims_conflict(a: &str, b: &str) -> bool {
    let other_integer = |value: &str, dim: &str| dim.parse::<i64>().is_ok() && dim != value;
    match (literal_dim(a), literal_dim(b)) {
        (Some(left), Some(right)) => left != right,
        (Some(value), None) => other_integer(value, b),
        (None, Some(value)) => other_integer(value, a),
        (None, None) => {
            a != b
                && !is_free_dim(a)
                && !is_free_dim(b)
                && is_named_size(a)
                && is_named_size(b)
                && !dims_canonically_equal(a, b)
                && (!(is_difference(a) || is_difference(b)) || same_names(a, b))
        }
    }
}

/// Whether two flat sums have the same terms apart from their integers.
fn same_names(a: &str, b: &str) -> bool {
    let names = |dim: &str| {
        let stripped: String = dim.chars().filter(|c| !c.is_whitespace()).collect();
        flat_sum(&stripped).map(|(terms, _)| terms)
    };
    names(a).is_some_and(|terms| !terms.is_empty() && names(b).as_ref() == Some(&terms))
}

/// The dim that two dims without a conflict give together. A free dim takes
/// the value of the other side, and a difference takes an exact dim. Two
/// different free dims give `UNPROVEN_DIM`, except that `UNKNOWN_DIM` says
/// nothing and so takes any other side.
pub(crate) fn unified_dim(a: &str, b: &str) -> String {
    if b == UNKNOWN_DIM {
        return a.to_string();
    }
    let rank = |dim: &str| match (is_free_dim(dim), is_difference(dim)) {
        (true, _) => 2,
        (false, true) => 1,
        (false, false) => 0,
    };
    match rank(a).cmp(&rank(b)) {
        std::cmp::Ordering::Greater => b.to_string(),
        std::cmp::Ordering::Less => a.to_string(),
        std::cmp::Ordering::Equal if a == UNKNOWN_DIM => b.to_string(),
        std::cmp::Ordering::Equal if rank(a) > 0 && !dims_canonically_equal(a, b) => {
            crate::layers::UNPROVEN_DIM.to_string()
        }
        std::cmp::Ordering::Equal => a.to_string(),
    }
}

/// Shape equality under [`dims_canonically_equal`].
pub(crate) fn shapes_canonically_equal(a: &[String], b: &[String]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| dims_canonically_equal(a, b))
}

/// Canonicalize a symbolic-dim expression for equality comparison.
///
/// Dims are bare strings (`"d_state*2"`, `"2*d_state"`, `"batch"`, `"3"`, …)
/// compared across independent call sites (constructor args, jaxtyping
/// annotations, other assignments). Two spellings of the same product/sum
/// commute, but plain `==` treats them as different, producing false
/// mismatches. This is **not** a CAS: it only understands a flat sum of
/// `+`-separated terms, each a flat product of `*`-separated integer
/// literals and identifiers. It:
///
/// * strips whitespace;
/// * folds each term's literal factors into one coefficient (`2*3*d` →
///   `6*d`, `d*1` → `d`);
/// * sorts each term's identifier factors, and sorts the terms themselves,
///   so commutative reorderings compare equal (`d*2` ≡ `2*d`, `a+b*2` ≡
///   `2*b+a`);
/// * drops additive-identity `0` terms (`d+0` → `d`), unless every term is
///   zero;
/// * adds the integer terms into one (`4+d+4` → `8+d`, `n-1+1` → `n`).
///
/// A subtraction is a term with a sign. Anything containing parens or
/// division is returned unchanged (whitespace-stripped only) — those aren't
/// provably commutative under this simple a token shuffle, so we don't risk a
/// false "equal" — except for the one unambiguous case `fold_paren_cancellation` recognizes
/// (a leading paren group whose contents are cancelled by the very next
/// operation, e.g. `(d-1)+1` → `d`): additive-inverse cancellation is sound
/// regardless of what the parenthesized expression is, so it's folded
/// before the conservative bail-out.
pub fn canonicalize_dim(dim: &str) -> String {
    let stripped: String = dim.chars().filter(|c| !c.is_whitespace()).collect();
    let stripped = if stripped.contains(FREE_DIM) {
        without_dim_marks(&stripped)
    } else {
        stripped
    };
    if stripped.is_empty() {
        return stripped;
    }
    if stripped.contains(['(', ')', '/']) {
        if let Some(folded) = fold_paren_cancellation(&stripped) {
            return canonicalize_dim(&folded);
        }
        return stripped;
    }
    let Some((terms, integers)) = flat_sum(&stripped) else {
        return stripped;
    };
    let (subtracted, mut added): (Vec<String>, Vec<String>) =
        terms.into_iter().partition(|term| term.starts_with('-'));
    if integers > 0 || (integers == 0 && added.is_empty() && subtracted.is_empty()) {
        added.push(integers.to_string());
    }
    added.sort_unstable();
    let mut canonical = added.join("+");
    canonical.extend(subtracted);
    if integers < 0 {
        canonical.push_str(&integers.to_string());
    }
    canonical
}

/// The sorted terms of a flat sum without its integers, with a `-` on a
/// subtracted term, and the sum of the integers. `None` for a sign that does
/// not start a term: `n*-1`.
fn flat_sum(stripped: &str) -> Option<(Vec<String>, i64)> {
    let mut integers = 0i64;
    let mut terms: Vec<String> = Vec::new();
    let mut rest = stripped;
    while !rest.is_empty() {
        let negative = rest.starts_with('-');
        rest = rest.strip_prefix(['+', '-']).unwrap_or(rest);
        let end = rest.find(['+', '-']).unwrap_or(rest.len());
        if end == 0 || rest[..end].ends_with('*') {
            return None;
        }
        let term = canonicalize_product_term(&rest[..end]);
        rest = &rest[end..];
        match term.parse::<i64>() {
            Ok(value) => {
                integers = integers.saturating_add(if negative { -value } else { value });
            }
            Err(_) if negative => terms.push(format!("-{term}")),
            Err(_) => terms.push(term),
        }
    }
    terms.sort_unstable();
    Some((terms, integers))
}

/// Fold the narrow, unambiguous shape `(expr<op1><lit1>)<op2><lit2>` → `expr`
/// when `<op1><lit1>` and `<op2><lit2>` are additive inverses (e.g.
/// `(d-1)+1` → `d`, `(d+2)-2` → `d`). Only a single leading paren group
/// followed by exactly one more `+`/`-` literal term is recognized —
/// anything else (a coefficient before the paren like `2*(d+1)`, division,
/// multiple paren groups, non-literal trailing terms) returns `None` and is
/// left for the caller's conservative "leave unchanged" fallback.
fn fold_paren_cancellation(stripped: &str) -> Option<String> {
    if !stripped.starts_with('(') {
        return None;
    }
    let close = stripped.find(')')?;
    let inner = &stripped[1..close];
    let rest = &stripped[close + 1..];
    let mut rest_chars = rest.chars();
    let outer_op = rest_chars.next()?;
    if outer_op != '+' && outer_op != '-' {
        return None;
    }
    let outer_lit: i64 = rest_chars.as_str().parse().ok()?;

    // Split `inner` at its last `+`/`-` (not at position 0, to avoid a
    // leading unary sign) into `expr` and a trailing `<op><literal>` term.
    let inner_op_idx = inner
        .char_indices()
        .skip(1)
        .filter(|&(_, c)| c == '+' || c == '-')
        .last()
        .map(|(i, _)| i)?;
    let (expr, inner_term) = inner.split_at(inner_op_idx);
    if expr.is_empty() {
        return None;
    }
    let mut inner_chars = inner_term.chars();
    let inner_op = inner_chars.next()?;
    let inner_lit: i64 = inner_chars.as_str().parse().ok()?;

    let inner_signed = if inner_op == '-' { -inner_lit } else { inner_lit };
    let outer_signed = if outer_op == '-' { -outer_lit } else { outer_lit };
    if inner_signed + outer_signed == 0 {
        Some(expr.to_string())
    } else {
        None
    }
}

/// Canonicalize one `+`-separated term: a flat product of int literals and
/// identifiers (`"d_state*2"` → `"2*d_state"`, `"d*1"` → `"d"`).
fn canonicalize_product_term(term: &str) -> String {
    if term.is_empty() {
        return term.to_string();
    }
    let mut literal: i64 = 1;
    let mut idents: Vec<&str> = Vec::new();
    for factor in term.split('*') {
        if let Ok(n) = factor.parse::<i64>() {
            literal = literal.saturating_mul(n);
        } else {
            idents.push(factor);
        }
    }
    idents.sort_unstable();
    if idents.is_empty() {
        return literal.to_string();
    }
    if literal == 1 {
        idents.join("*")
    } else {
        let mut parts = vec![literal.to_string()];
        parts.extend(idents.iter().map(|s| s.to_string()));
        parts.join("*")
    }
}

/// Dim equality under [`canonicalize_dim`], with or without the `FREE_DIM`
/// mark. A check that reports a mismatch uses [`dims_conflict`].
pub fn dims_canonically_equal(a: &str, b: &str) -> bool {
    canonicalize_dim(a) == canonicalize_dim(b)
}

#[cfg(test)]
mod canonicalize_dim_tests {
    use super::*;

    #[test]
    fn test_commutative_product_matches() {
        assert_eq!(canonicalize_dim("d*2"), canonicalize_dim("2*d"));
        assert_eq!(
            canonicalize_dim("d_state*2"),
            canonicalize_dim("2*d_state")
        );
    }

    #[test]
    fn test_literal_folding() {
        assert_eq!(canonicalize_dim("2*3*d"), "6*d");
        assert_eq!(canonicalize_dim("d*1"), "d");
        assert_eq!(canonicalize_dim("d+0"), "d");
    }

    #[test]
    fn test_commutative_sum_matches() {
        assert_eq!(canonicalize_dim("a+b*2"), canonicalize_dim("2*b+a"));
    }

    #[test]
    fn test_integer_terms_add_up() {
        assert_eq!(canonicalize_dim("4+4"), "8");
        assert_eq!(canonicalize_dim("4+d+4"), canonicalize_dim("8+d"));
    }

    #[test]
    fn test_all_zero_sum_collapses_to_zero() {
        assert_eq!(canonicalize_dim("0+0"), "0");
    }

    #[test]
    fn test_plain_identifier_and_literal_unchanged() {
        assert_eq!(canonicalize_dim("batch"), "batch");
        assert_eq!(canonicalize_dim("3"), "3");
    }

    #[test]
    fn test_whitespace_stripped() {
        assert_eq!(canonicalize_dim(" d * 2 "), "2*d");
    }

    #[test]
    fn test_non_commutative_forms_returned_unchanged() {
        // Parens, division: not a CAS, returned as-is
        // (whitespace-stripped) rather than risking a false "equal".
        assert_eq!(canonicalize_dim("(d)"), "(d)");
        assert_eq!(canonicalize_dim("d-1"), "d-1");
        assert_eq!(canonicalize_dim("d/2"), "d/2");
    }

    #[test]
    fn test_subtraction_is_a_term_with_a_sign() {
        assert_eq!(canonicalize_dim("n-1+1"), "n");
        assert_eq!(canonicalize_dim("n-1-1"), canonicalize_dim("n-2"));
        assert_eq!(canonicalize_dim("1-n+2"), canonicalize_dim("3-n"));
        assert_eq!(canonicalize_dim("a-b-c"), canonicalize_dim("a-c-b"));
        assert_ne!(canonicalize_dim("a-b"), canonicalize_dim("b-a"));
        assert_eq!(canonicalize_dim("n*-1"), "n*-1");
    }

    #[test]
    fn test_a_difference_conflicts_only_with_its_own_name() {
        assert!(dims_conflict("n-1", "n"));
        assert!(dims_conflict("n", "n-1"));
        assert!(dims_conflict("n-1", "n-2"));
        assert!(dims_conflict("n-1", "n+1"));
        assert!(!dims_conflict("n-1", "n-1"));
        assert!(!dims_conflict("n-1", "1-1+n-1"));
        assert!(!dims_conflict("n-1", "4"));
        assert!(!dims_conflict("4", "n-1"));
        assert!(!dims_conflict("n-1", "m"));
        assert!(!dims_conflict("n-1", "m-1"));
        assert!(!dims_conflict("__free_n-1", "n"));
        assert!(!dims_conflict("(n-1)*2", "6"));
        for other in ["4", "n", "m", "n-1", "2*n", "__free_k"] {
            assert!(!dims_conflict("__free_min(4, n)", other), "{other}");
            assert!(!dims_conflict(other, "__free_min(4, n)"), "{other}");
            assert!(!dims_conflict(UNKNOWN_DIM, other), "{other}");
        }
    }

    #[test]
    fn test_a_long_dim_is_unknown() {
        let at_limit = "n".repeat(MAX_DIM_TEXT);
        assert_eq!(bounded_dim(at_limit.clone()), at_limit);
        assert_eq!(bounded_dim("n".repeat(MAX_DIM_TEXT + 1)), UNKNOWN_DIM);
        assert_eq!(bounded_dim(format!("{UNKNOWN_DIM}+n")), UNKNOWN_DIM);
        assert_eq!(joined_dim(&["n", "m"], "+"), "n+m");
        assert_eq!(joined_dim(&vec![at_limit.as_str(); 1 << 20], "+"), UNKNOWN_DIM);
        assert_eq!(bounded_shape(vec!["n".to_string(); MAX_RANK]).map(|s| s.len()), Some(MAX_RANK));
        assert_eq!(bounded_shape(vec!["n".to_string(); MAX_RANK + 1]), None);
    }

    #[test]
    fn test_a_top_level_split_keeps_a_call() {
        assert_eq!(split_top_level("min(k, 2), [1, 2], 3"), vec!["min(k, 2)", " [1, 2]", " 3"]);
        assert_eq!(split_top_level("n"), vec!["n"]);
    }

    #[test]
    fn test_an_offset_folds_the_last_integer() {
        assert_eq!(offset_dim("n", 1).as_deref(), Some("n-1"));
        assert_eq!(offset_dim("n-1", 2).as_deref(), Some("n-3"));
        assert_eq!(offset_dim("n+1", 1).as_deref(), Some("n"));
        assert_eq!(offset_dim("n+3", 1).as_deref(), Some("n+2"));
        assert_eq!(offset_dim("2*n", 1).as_deref(), Some("2*n-1"));
        assert_eq!(offset_dim("(n-1)//2", 1).as_deref(), Some("(n-1)//2-1"));
        assert_eq!(offset_dim("min(4, n-1)", 1).as_deref(), Some("min(4, n-1)-1"));
    }

    #[test]
    fn test_a_factor_keeps_a_sum_together() {
        assert_eq!(factor_text("n-1"), "(n-1)");
        assert_eq!(factor_text("n+1"), "(n+1)");
        assert_eq!(factor_text("n*2"), "n*2");
        assert_eq!(factor_text("min(4, n-1)"), "min(4, n-1)");
        assert_eq!(factor_text("-1"), "-1");
    }

    #[test]
    fn test_paren_cancellation_folds() {
        assert_eq!(canonicalize_dim("(d-1)+1"), "d");
        assert_eq!(canonicalize_dim("(d+2)-2"), "d");
        assert_eq!(canonicalize_dim("(d_state-1)+1"), "d_state");
        // Folds fully back down into commutative canonicalization too.
        assert_eq!(canonicalize_dim("(d-1)+1"), canonicalize_dim("d"));
    }

    #[test]
    fn test_paren_cancellation_is_conservative() {
        // A coefficient before the paren isn't the narrow single-group
        // shape this recognizes — left unchanged.
        assert_eq!(canonicalize_dim("2*(d+1)"), "2*(d+1)");
        // Non-cancelling literals: not an identity, left unchanged.
        assert_eq!(canonicalize_dim("(d-1)+2"), "(d-1)+2");
        // Division inside/after the group still bails (no cancellation to
        // find, and dividing stays unfolded regardless).
        assert_eq!(canonicalize_dim("(d-1)/2"), "(d-1)/2");
    }

    #[test]
    fn test_dims_canonically_equal() {
        assert!(dims_canonically_equal("d*2", "2*d"));
        assert!(!dims_canonically_equal("d*2", "d*3"));
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportPath {
    pub dots: usize,
    pub module: Vec<String>,
    pub name: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct CallInfo {
    pub variable: String,
    pub target: String,
    pub args_node_range: Range,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOp {
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedTarget {
    pub dots: usize,
    pub parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedModuleTarget {
    pub dots: usize,
    pub module_parts: Vec<String>,
    pub file_path: PathBuf,
    pub symbol_parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PythonSymbol {
    Class { name: String },
    Function { name: String },
    Import { name: String, path: ImportPath },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedImplementation {
    pub target: ResolvedModuleTarget,
    pub symbol: Option<PythonSymbol>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct PythonCallableSignature {
    pub owner: Option<String>,
    pub name: String,
    pub params: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum CallArgument {
    Positional { value: String },
    Keyword { name: String, value: String },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedCallSignature {
    pub implementation: ResolvedImplementation,
    pub signature: PythonCallableSignature,
    pub arguments: Vec<CallArgument>,
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LayerKind {
    Linear {
        in_features: String,
        out_features: String,
    },
    /// Conv1d/2d/3d (torch + equinox), channels-first.
    Conv {
        spatial_rank: usize,
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
        dilation: String,
    },
    /// flax.linen.Dense — channels-last: replaces the last dim with
    /// `features`. Input width is runtime-inferred by flax, so there is
    /// nothing to check.
    Dense {
        features: String,
    },
    /// flax.linen.Conv: channels-last, spatial dims follow `kernel_size` and
    /// `padding` (default SAME), last dim becomes `features`. Non-default
    /// strides are refused at classification time.
    FlaxConv {
        features: String,
        spatial_rank: usize,
        kernel_size: String,
        padding: String,
    },
    /// Max/Avg pooling, channels-first like Conv: channels preserved, the
    /// trailing `spatial_rank` dims follow the conv output formula.
    /// `dilation` is `"1"` for AvgPool and equinox. `ceil_mode` is the torch
    /// text; a value that is not a literal bool gives unknown spatial dims.
    Pool {
        name: String,
        spatial_rank: usize,
        kernel_size: String,
        stride: String,
        padding: String,
        dilation: String,
        ceil_mode: String,
    },
    /// Adaptive pooling: the trailing `spatial_rank` dims all become
    /// `output_size`, channels preserved.
    AdaptivePool {
        name: String,
        spatial_rank: usize,
        output_size: String,
    },
    /// torch.nn.MultiheadAttention — returns an (output, weights) tuple:
    /// output has the query's shape, weights are `(N, L, S)` (or
    /// `(N, num_heads, L, S)`), without `N` when unbatched. Only tuple-unpacking LHS is modelled.
    ///
    /// `feature_dim` is the ctor's per-token model dimension: `embed_dim`
    /// for `torch.nn.MultiheadAttention(embed_dim, num_heads)`, or
    /// `query_size` for `equinox.nn.MultiheadAttention(num_heads,
    /// query_size, ...)` — equinox's real positional order is reversed and
    /// uses a different name, so `classify_layer_call` must branch on the
    /// resolved module path rather than reusing one binding key for both
    /// frameworks (previously it always read torch's `embed_dim` key, so
    /// equinox ctors were silently unclassified/misbound).
    ///
    /// torch only (`returns_weights`): `batch_first`, `num_heads`, and
    /// `added_keys`, the count of key positions that `add_bias_kv` and
    /// `add_zero_attn` add (`UNPROVEN_DIM` when not literal). equinox returns
    /// the output only.
    MultiheadAttention {
        feature_dim: String,
        num_heads: String,
        batch_first: String,
        added_keys: String,
        returns_weights: bool,
    },
    /// Index lookup table: appends `embedding_size` to the input shape
    /// (scalar index → `(embedding_size,)`, `(batch, seq)` → `(batch, seq, embedding_size)`).
    Embedding {
        embedding_size: String,
    },
    ShapePreserving {
        name: String,
    },
    /// torch.nn.Flatten — collapses dims `[start_dim, end_dim]` (Python
    /// negative indices resolved against the input rank at apply time) into
    /// a single dim: product of literal ints when all concrete, otherwise a
    /// symbolic `d0*d1*...` string.
    Flatten {
        start_dim: String,
        end_dim: String,
    },
    /// torch.nn.Unflatten(dim, sizes) — expands `dim` into the components of
    /// `sizes` (raw ctor tuple/list text, split at apply time).
    Unflatten {
        dim: String,
        sizes: String,
    },
    /// torch.nn.Upsample — scales the trailing spatial dims (all dims after
    /// the leading batch+channel dims, determined at apply time from the
    /// input rank since `Upsample` itself is rank-agnostic) by
    /// `scale_factor`, or sets them directly to `size`. Exactly one of the
    /// two is expected to be usable; anything else is treated as unknown
    /// (`Ok(None)`).
    Upsample {
        scale_factor: Option<String>,
        size: Option<String>,
    },
    /// ConvTranspose1d/2d/3d (torch + equinox): `out = (in-1)*stride -
    /// 2*padding + dilation*(kernel-1) + output_padding + 1`. Channels-first
    /// layout, same convention as `Conv`.
    ConvTranspose {
        spatial_rank: usize,
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
        output_padding: String,
        dilation: String,
    },
    /// torch.nn.RNN/LSTM/GRU full-sequence modules, `(output, h_n)` or
    /// `(output, (h_n, c_n))`. A direct application gives `output` only:
    /// last dim `D*H_out`, with `D` 2 when bidirectional and `H_out` the
    /// `proj_size` of an LSTM when it is above 0, else `hidden_size`. The
    /// state is `[D*num_layers, N, H_out]` (`c_n` keeps `hidden_size`).
    /// `batch_first`, `bidirectional`, `num_layers` and `proj_size` are the
    /// constructor text; a value that is not a literal gives an unknown dim.
    Rnn {
        name: String,
        input_size: String,
        hidden_size: String,
        num_layers: String,
        bidirectional: String,
        proj_size: String,
        batch_first: String,
    },
    /// torch.nn.RNNCell/GRUCell/LSTMCell and equinox.nn.LSTMCell/GRUCell —
    /// single-step cells. RNNCell/GRUCell genuinely return one tensor of
    /// shape `hidden_size`; LSTMCell returns `(h, c)` — modelled as just the
    /// `h` component (same shape as GRUCell's output), an approximation for
    /// the same tuple-output reason as `Rnn`.
    RnnCell {
        name: String,
        input_size: String,
        hidden_size: String,
    },
    /// torch.nn.PixelShuffle(upscale_factor): `(*, C*r^2, H, W) -> (*, C, H*r, W*r)`.
    PixelShuffle {
        upscale_factor: String,
    },
    /// torch.nn.PixelUnshuffle(downscale_factor): `(*, C, H*r, W*r) -> (*, C*r^2, H, W)`.
    PixelUnshuffle {
        downscale_factor: String,
    },
    /// Constant/Zero/Reflection/Replication Pad Nd: pads the trailing
    /// `spatial_rank` dims. `padding` is the raw ctor arg text: a single int
    /// means uniform padding on every side of every spatial dim; an explicit
    /// `2*spatial_rank`-length tuple follows torch's reverse-axis pad-pair
    /// convention (last spatial dim's (left,right) first).
    Pad {
        name: String,
        spatial_rank: usize,
        padding: String,
    },
    /// torch.nn.Bilinear(in1_features, in2_features, out_features): this
    /// analyzer only tracks the first positional input per layer call (see
    /// `extract_layer_applications`), so this only validates/transforms
    /// against `x1`; `x2` and its broadcasting are not modelled.
    Bilinear {
        in1_features: String,
        in2_features: String,
        out_features: String,
    },
    /// torch.nn.CosineSimilarity(dim): removes the reduced axis from `x1`'s
    /// shape (same single-tracked-input limitation as `Bilinear`).
    CosineSimilarity {
        dim: String,
    },
    /// equinox.nn.MLP: last-dim transform `in_size -> out_size`, same rule as
    /// `Linear`.
    Mlp {
        in_size: String,
        out_size: String,
    },
    /// `torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)` —
    /// im2col: extracts sliding `kernel_size`-shaped blocks from a
    /// channels-first `(C, H, W)` / `(N, C, H, W)` input into columns:
    /// `(C*kh*kw, L)` / `(N, C*kh*kw, L)`, where `L = num_blocks_h *
    /// num_blocks_w` follows the same block-count formula as
    /// `conv_spatial_dim`. Each of `kernel_size`/`stride`/`padding`/
    /// `dilation` is the raw ctor arg text: a single value applies to both
    /// H and W, or an explicit 2-tuple gives them independently (parsed at
    /// apply time via `parse_pair_arg`; anything else is unknown). Exact
    /// when H/W and the geometry args are literal ints, or when `dilation`
    /// is literal `1` (the overwhelmingly common default, which folds away
    /// even against a symbolic kernel size); any other non-literal
    /// `dilation` combination isn't foldable and yields `Ok(None)`.
    Unfold {
        kernel_size: String,
        stride: String,
        padding: String,
        dilation: String,
    },
    /// `torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0,
    /// stride=1)` — inverse of `Unfold`: input `(C*kh*kw, L)` /
    /// `(N, C*kh*kw, L)` folds back into `(C, oh, ow)` / `(N, C, oh, ow)`.
    /// `output_size` and `kernel_size` are the raw ctor arg text (each a
    /// single value or 2-tuple, parsed at apply time); the channel dim is
    /// recovered by dividing the input's second-to-last dim by `kh*kw`
    /// (`Err` when literal and not evenly divisible, a symbolic
    /// `dim/(kh*kw)` string otherwise). The `L` dim itself is not
    /// cross-checked against `output_size`/kernel geometry.
    Fold {
        output_size: String,
        kernel_size: String,
    },
    /// `torch.nn.Sequential` / `equinox.nn.Sequential`: a composite container
    /// wrapping an ordered list of child layers (torch: variadic positional
    /// ctor args; equinox: a single list-literal positional arg — both forms
    /// are unwrapped into `children` at classification time). Application
    /// threads the input shape through each child's own apply rule in order,
    /// via the shared `apply_layer_kind` helper; a mid-chain error reports as
    /// that child's own error message, and an unknown (`Ok(None)`) child
    /// makes the whole chain unknown. Classification itself is honest: if any
    /// ctor argument isn't itself a resolvable layer-constructor call
    /// (arbitrary callable, `*args` unpacking, etc.), the whole `Sequential`
    /// is left unclassified.
    Sequential { children: Vec<LayerKind> },
    /// `einops.layers.torch.Rearrange` / `Reduce` (and the flax equivalents)
    /// — the layer-object form of the free-function einops pattern algebra.
    /// `name` is `"Rearrange"`/`"Reduce"` (cosmetic only: the reduction op
    /// itself isn't shape-relevant, same as the free-function form).
    /// `pattern` is the raw (still-quoted) ctor pattern string; `kwargs` are
    /// axis-length keyword bindings from the ctor (e.g. `h=14`). Both are fed
    /// verbatim into `known_functions::apply_known_einops`'s existing pattern
    /// engine at apply time.
    EinopsPattern {
        name: String,
        pattern: String,
        kwargs: HashMap<String, String>,
    },
    /// A `torch.nn` loss module (`MSELoss`, `CrossEntropyLoss`, ...). It takes
    /// the prediction and a target, so it applies through
    /// `LayerApplication::extra_inputs`. `reduction` is the raw ctor text
    /// (`'mean'` by default). It is `None` when a deprecated `size_average` or
    /// `reduce` argument decides the reduction.
    Loss {
        name: String,
        reduction: Option<String>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum KnownFunction {
    Concatenate,
    Stack,
    Reshape,
    Transpose,
    ExpandDims,
    Squeeze,
    /// torch `squeeze`: a dim whose size is not 1 stays in the shape.
    TorchSqueeze,
    Sum,
    Mean,
    /// A reduction by the slots of one library signature. Each field is a
    /// positional index, the input is 0. `keepdims: None` is keyword only.
    /// The shape of the quantile argument `q` leads the result.
    Reduction {
        axis: usize,
        keepdims: Option<usize>,
        q: Option<usize>,
    },
    Max,
    Min,
    /// `torch.max`/`torch.min`: two tensors broadcast elementwise, a `dim`
    /// returns a `(values, indices)` namedtuple, otherwise a full reduction.
    TorchMaxMin,
    MinMax,
    Prod,
    Std,
    Var,
    Matmul,
    /// `jax.lax.dot_general(lhs, rhs, ((lc, rc), (lb, rb)))`.
    DotGeneral,
    /// `torch.bmm` / `Tensor.bmm`: rank-3 operands, equal batch (no broadcast).
    Bmm,
    /// `torch.mm` / `Tensor.mm`: two rank-2 operands.
    Mm,
    /// `torch.mv` / `Tensor.mv`: a rank-2 and a rank-1 operand.
    Mv,
    /// `torch.addmm(input, mat1, mat2)`: `mm`, and `input` broadcasts to it.
    Addmm,
    /// `torch.baddbmm(input, batch1, batch2)`: `bmm`, and `input` broadcasts to it.
    Baddbmm,
    /// `Tensor.t()`: the transpose of a tensor of rank 2 or less.
    TorchT,
    /// `torch.nn.functional.linear(input, weight, bias)`, `weight` is `[out, in]`.
    Linear,
    /// `torch.nn.functional.scaled_dot_product_attention(query, key, value)`.
    ScaledDotProductAttention,
    /// A `torch.nn.functional` loss with the rule of the loss module `layer`.
    /// `reduction` is the positional index of the `reduction` argument, and
    /// `deprecated` the indices of `size_average` and `reduce`.
    FunctionalLoss {
        layer: &'static str,
        reduction: usize,
        deprecated: Option<(usize, usize)>,
    },
    LinalgEigvals,
    LinalgMatrixPower,
    LinalgMultiDot,
    /// `linalg.slogdet`: a `(sign, logabsdet)` tuple.
    LinalgSlogdet,
    Dot,
    Einsum,
    Split,
    ArraySplit,
    /// Real torch `split`/`Tensor.split` semantics: the 2nd arg is a
    /// `split_size` (or list of sizes), not a section *count* like
    /// `jnp.split`/`np.split` or `torch.tensor_split`.
    TorchSplit,
    Tile,
    /// numpy `repeat(a, repeats, axis)`, torch `repeat_interleave`.
    Repeat,
    /// The method `repeat_interleave`: `Repeat` in every library.
    RepeatInterleave,
    /// torch `Tensor.repeat(*sizes)`: a tile with at least one size per dim.
    TorchRepeat,
    Flatten,
    /// `torch.flatten(x, start_dim, end_dim)` / `Tensor.flatten(start_dim, end_dim)`.
    TorchFlatten,
    Ravel,
    MoveAxis,
    SwapAxes,
    Where,
    Zeros,
    Ones,
    Full,
    /// `torch.zeros/ones/empty/rand/randn`: dims as varargs or one tuple/list.
    TorchSizeConstructor,
    Arange,
    Eye,
    Vmap,
    Scan,
    FlaxPool,
    EinopsRearrange,
    EinopsReduce,
    EinopsRepeat,
    BroadcastTo,
    BroadcastArrays,
    AtLeast1D,
    AtLeast2D,
    AtLeast3D,
    Pad,
    NumpyPad,
    Roll,
    Flip,
    Rot90,
    Take,
    Diag,
    Diagonal,
    Trace,
    Triu,
    Tril,
    Meshgrid,
    Vstack,
    Hstack,
    Dstack,
    ColumnStack,
    Block,
    Permute,
    Array,
    AsArray,
    Empty,
    ZerosLike,
    OnesLike,
    FullLike,
    EmptyLike,
    Identity,
    Linspace,
    Logspace,
    All,
    Any,
    ArgMax,
    ArgMin,
    Argsort,
    Sort,
    Cumsum,
    Cumprod,
    TensorDot,
    Outer,
    Inner,
    Vdot,
    LinalgInv,
    LinalgMatrixSqrth,
    LinalgDet,
    LinalgSvd,
    LinalgQr,
    LinalgEig,
    Astype,
    Copy,
    Clamp,
    Detach,
    Contiguous,
    To,
    // jax.lax higher-order / structured ops
    LaxMap,
    LaxCond,
    LaxSwitch,
    LaxWhileLoop,
    LaxForiLoop,
    LaxConvGeneralDilated,
    LaxGather,
    LaxScatter,
    LaxReduceWindow,
    LaxTopK,
    LaxSort,
    LaxSortKeyVal,
    LaxPad,
    LaxBroadcast,
    LaxBroadcastInDim,
    LaxSlice,
    LaxDynamicSlice,
    LaxDynamicUpdateSlice,
    LaxAssociativeScan,
    LaxPolynomial,
    RandomMultinomial,
    // jax.numpy / numpy array creation
    Diagflat,
    Tri,
    Indices,
    BinCount,
    Unique,
    Select,
    // jax.numpy / numpy shape transforms
    RollAxis,
    Resize,
    Insert,
    Delete,
    Append,
    // jax.numpy / numpy joining and splitting
    HSplit,
    VSplit,
    DSplit,
    Kron,
    // jax.numpy / numpy indexing and selection
    TakeAlongAxis,
    PutAlongAxis,
    Nonzero,
    Argwhere,
    SearchSorted,
    Extract,
    Compress,
    Histogram,
    // linear algebra
    Cross,
    TorchCross,
    LinalgSolve,
    LinalgLstsq,
    LinalgPinv,
    LinalgMatrixRank,
    // einops
    EinopsEinsum,
    EinopsPack,
    EinopsUnpack,
    EinopsParseShape,
    // torch.nn.functional losses
    CrossEntropy,
    NllLoss,
    // jax.nn
    OneHot,
    DotProductAttention,
    // torch tensor indexing / selection methods
    Gather,
    Scatter,
    IndexSelect,
    Narrow,
    SelectDim,
    MaskedSelect,
    MaskedFill,
    Unfold,
    ShapeAs,
    Item,
    NewConstructor,
    /// `x.new_zeros/new_ones/new_empty`: like `TorchSizeConstructor` after the receiver.
    NewSizeConstructor,
    // torch tuple-output methods
    TopK,
    Chunk,
    Unbind,
    KthValue,
    MedianDim,
    // torch combinatorics
    Combinations,
    CartesianProd,
    BlockDiag,
    // torch.nn.functional
    Interpolate,
    FunctionalConv1d,
    FunctionalConv2d,
    FunctionalConv3d,
    FunctionalMaxPool1d,
    FunctionalMaxPool2d,
    FunctionalMaxPool3d,
    FunctionalAvgPool1d,
    FunctionalAvgPool2d,
    FunctionalAvgPool3d,
    FunctionalEmbedding,
    /// `torch.nn.functional.glu(input, dim=-1)` — halves `dim` (must be
    /// even-sized): `(*, 2*n, *) -> (*, n, *)`. Not shape-preserving, unlike
    /// the rest of the `torch.nn.functional` activation family.
    FunctionalGlu,
    // torch.nn.utils.rnn
    PadSequence,
    /// Elementwise array arguments, in signature order. Other parameters do
    /// not affect shape. LAX primitives can prohibit non-scalar rank promotion.
    Elementwise {
        parameters: &'static [&'static str],
        rank_promotion: bool,
    },
    BroadcastLike,
    // Structured matrix constructors in jax.scipy.linalg.
    Hadamard,
    Dft,
    InvHilbert,
    InvPascal,
    Helmert,
    Circulant,
    Fiedler,
    Companion,
    FiedlerCompanion,
    Leslie,
    ConvolutionMatrix,
    QrMultiply,
}

#[derive(Debug, Clone)]
pub struct LayerApplication {
    pub variable: String,
    pub layer: String,
    pub input: String,
    /// The inputs after the first, in call order: the target of a loss layer.
    pub extra_inputs: Vec<String>,
    pub kind: LayerKind,
    pub range: Range,
}

impl PartialEq for LayerApplication {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
            && self.layer == other.layer
            && self.input == other.input
            && self.extra_inputs == other.extra_inputs
            && self.kind == other.kind
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LayerAssignment {
    pub name: String,
    pub kind: LayerKind,
    pub byte_position: usize,
}

/// A `self.<attr> = <layer ctor>` binding together with the byte range of the
/// class it was defined in, so same-named attrs in different classes don't
/// collide at lookup time.
#[derive(Debug, PartialEq, Clone)]
pub struct ScopedSelfAttrLayer {
    pub class_start: usize,
    pub class_end: usize,
    pub kind: LayerKind,
    /// Where the constructor call stands.
    pub byte: usize,
}

/// A `self.<attr> = <ident>` alias binding together with the byte range of
/// the class it was defined in, so same-named attrs in different classes
/// don't collide at lookup time. Mirrors `ScopedSelfAttrLayer`.
#[derive(Debug, PartialEq, Clone)]
pub struct ScopedSelfAttrAlias {
    pub class_start: usize,
    pub class_end: usize,
    pub value: String,
}

/// Classifies a `ShapeError` by how confident the rule that raised it is.
///
/// `Mismatch`: a genuine shape contradiction under the rule's modeled
/// semantics. `Approximation`: raised by a rule that is documented (see
/// `llm.txt`/`TO_IMPLEMENT.md`) as an approximation of the real op, so the
/// "error" may be a false positive rather than an actual bug in the analyzed
/// code. Drives diagnostic severity in `main.rs`.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ShapeErrorKind {
    Mismatch,
    Approximation,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ShapeFix {
    AppendTranspose {
        expression_range: Range,
        operand_range: Range,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ShapeError {
    pub variable: String,
    pub message: String,
    pub range: Range,
    pub kind: ShapeErrorKind,
    /// Optional related-information location: a second range (e.g. the
    /// *other* operand's node in a binary-op mismatch) plus a short message,
    /// surfaced as LSP `DiagnosticRelatedInformation` by `main.rs`. `None`
    /// for errors that don't have a natural second location.
    pub related: Option<(Range, String)>,
    pub fix: Option<ShapeFix>,
}

impl ShapeError {
    /// A genuine shape contradiction under the rule's modeled semantics.
    pub fn mismatch(variable: impl Into<String>, message: impl Into<String>, range: Range) -> Self {
        ShapeError {
            variable: variable.into(),
            message: message.into(),
            range,
            kind: ShapeErrorKind::Mismatch,
            related: None,
            fix: None,
        }
    }

    /// Raised by a rule that is a documented approximation of the real op
    /// (e.g. `jax.lax.dot_general` modeled as matmul), so the error may be a
    /// false positive rather than an actual bug in the analyzed code.
    pub fn approximation(
        variable: impl Into<String>,
        message: impl Into<String>,
        range: Range,
    ) -> Self {
        ShapeError {
            variable: variable.into(),
            message: message.into(),
            range,
            kind: ShapeErrorKind::Approximation,
            related: None,
            fix: None,
        }
    }

    /// Attach a related-information location/message, e.g. the *other*
    /// operand's node range in a two-operand shape mismatch. Builder-style:
    /// chain after `mismatch`/`approximation`.
    pub fn with_related(mut self, range: Range, message: impl Into<String>) -> Self {
        self.related = Some((range, message.into()));
        self
    }

    pub fn with_fix(mut self, fix: ShapeFix) -> Self {
        self.fix = Some(fix);
        self
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ShapeDimensionSite {
    pub binding: Option<String>,
    pub axis: usize,
    pub value: String,
    pub range: Range,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionShapeScope {
    pub function_name: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub shapes: HashMap<String, Vec<String>>,
    pub return_shape: Option<Vec<String>>,
    /// Parameter names with jaxtyping annotations, in declaration order.
    /// Used by cross-function shape propagation to match positional call
    /// arguments to declared parameter shapes.
    pub param_order: Vec<String>,
    /// EVERY declared parameter name (annotated or not), in declaration
    /// order, including `self`/`cls`. Used by lazy call-site parameter
    /// seeding (`analysis::specialize_callee_call`) to map a call's
    /// positional/keyword arguments onto the callee's full parameter list —
    /// `param_order` can't be reused for this since it only contains
    /// jaxtyping-annotated params, so its index doesn't match the true
    /// declared position when annotated and un-annotated params are mixed.
    pub all_params: Vec<String>,
    pub dimension_sites: Vec<ShapeDimensionSite>,
}

/// One record per (non-annotated) assignment that produced a shape, in the
/// order assignments were processed. Unlike `FunctionShapeScope::shapes`
/// (which keeps only the last shape per name), this preserves the shape at
/// *each* assignment site so inlay hints can be emitted per reassignment.
#[derive(Debug, PartialEq, Clone)]
pub struct AssignmentShape {
    /// 0-based row of the assignment's LHS.
    pub line: u32,
    pub name: String,
    pub shape: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct LayerShapeAnalysis {
    pub scopes: Vec<FunctionShapeScope>,
    pub layers: HashMap<String, LayerKind>,
    pub applications: Vec<LayerApplication>,
    pub errors: Vec<ShapeError>,
    pub assignment_shapes: Vec<AssignmentShape>,
}

impl LayerShapeAnalysis {
    /// The analysis without the `FREE_DIM` marks, for the reader of the result:
    /// `UNKNOWN_DIM` becomes `?`.
    /// A shape or a message has no `UNPROVEN_DIM`: the analysis drops those. A
    /// layer of an application keeps it, so it becomes `?` here.
    pub(crate) fn without_free_marks(mut self) -> Self {
        fn clean(text: &mut String) {
            if text.contains(FREE_DIM) {
                *text = without_dim_marks(text);
            }
        }
        for scope in &mut self.scopes {
            scope.shapes.values_mut().flatten().for_each(clean);
            scope.return_shape.iter_mut().flatten().for_each(clean);
            for site in &mut scope.dimension_sites {
                clean(&mut site.value);
            }
        }
        for binding in &mut self.assignment_shapes {
            binding.shape.iter_mut().for_each(clean);
        }
        for error in &mut self.errors {
            clean(&mut error.message);
            if let Some((_, message)) = &mut error.related {
                clean(message);
            }
        }
        for application in &mut self.applications {
            application.kind = map_layer_dims(application.kind.clone(), &|dim| {
                without_dim_marks(dim)
                    .replace(crate::layers::UNPROVEN_DIM, "?")
                    .replace(crate::layers::PADDING_NAME, "")
            });
        }
        self
    }

    pub fn scope_for_byte(&self, byte: usize) -> Option<&FunctionShapeScope> {
        scope_for_byte(&self.scopes, byte)
    }
}

pub fn scope_for_byte(scopes: &[FunctionShapeScope], byte: usize) -> Option<&FunctionShapeScope> {
    scope_index_for_byte(scopes, byte).map(|i| &scopes[i])
}

/// Fast `scope_index_for_byte` for scopes whose ranges are nested or disjoint
/// (every parsed file). Other range sets use the linear scan. Built from the
/// ranges only, so later changes to a scope's shapes leave it valid.
#[derive(Default)]
pub struct ScopeIndex {
    ranges: Vec<(usize, usize)>,
    /// Non-empty ranges ordered by start, then larger first, then lower index.
    sorted: Vec<IndexedScope>,
    nested: bool,
}

struct IndexedScope {
    start: usize,
    end: usize,
    index: usize,
    /// Position in `sorted` of the smallest range that encloses this one.
    parent: Option<usize>,
}

impl ScopeIndex {
    pub fn new(scopes: &[FunctionShapeScope]) -> Self {
        Self::from_ranges(scopes.iter().map(|s| (s.start_byte, s.end_byte)).collect())
    }

    pub fn from_ranges(ranges: Vec<(usize, usize)>) -> Self {
        let mut order: Vec<usize> = (0..ranges.len())
            .filter(|&i| ranges[i].0 < ranges[i].1)
            .collect();
        order.sort_by_key(|&i| (ranges[i].0, std::cmp::Reverse(ranges[i].1), i));
        let mut sorted: Vec<IndexedScope> = Vec::with_capacity(order.len());
        let mut open: Vec<usize> = Vec::new();
        let mut nested = true;
        for index in order {
            let (start, end) = ranges[index];
            while open.last().is_some_and(|&o| sorted[o].end <= start) {
                open.pop();
            }
            let parent = open.last().copied();
            if parent.is_some_and(|p| end > sorted[p].end) {
                nested = false;
                break;
            }
            open.push(sorted.len());
            sorted.push(IndexedScope {
                start,
                end,
                index,
                parent,
            });
        }
        Self {
            ranges,
            sorted,
            nested,
        }
    }

    /// Same answer as `scope_index_for_byte`: the smallest scope containing
    /// `byte`, the later one when two have the same size.
    pub fn lookup(&self, byte: usize) -> Option<usize> {
        if !self.nested {
            return linear_scope_index(self.ranges.iter().copied(), byte);
        }
        let mut cur = self
            .sorted
            .partition_point(|e| e.start <= byte)
            .checked_sub(1);
        while let Some(c) = cur {
            let e = &self.sorted[c];
            if byte < e.end {
                return Some(e.index);
            }
            cur = e.parent;
        }
        None
    }
}

fn linear_scope_index(ranges: impl Iterator<Item = (usize, usize)>, byte: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (i, (start, end)) in ranges.enumerate() {
        if start <= byte && byte < end {
            let size = end - start;
            if best.is_none_or(|(_, prev)| size <= prev) {
                best = Some((i, size));
            }
        }
    }
    best.map(|(i, _)| i)
}

pub fn scope_index_for_byte(scopes: &[FunctionShapeScope], byte: usize) -> Option<usize> {
    linear_scope_index(scopes.iter().map(|s| (s.start_byte, s.end_byte)), byte)
}

#[cfg(test)]
mod scope_index_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            if p.is_dir() {
                python_files(&p, out);
            } else if p.extension().is_some_and(|x| x == "py") {
                out.push(p);
            }
        }
    }

    fn scopes_of(text: &str) -> Vec<FunctionShapeScope> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(text, None).unwrap();
        crate::python_ast::extract_jaxtyping_shapes(tree.root_node(), text).unwrap()
    }

    /// Lookup is constant between scope boundaries, so the bytes next to every
    /// boundary plus a stride cover each distinct answer; small files get
    /// every byte.
    fn bytes_to_check(scopes: &[FunctionShapeScope], len: usize) -> Vec<usize> {
        if len <= 64 * 1024 {
            return (0..=len).collect();
        }
        let mut bytes: Vec<usize> = (0..=len).step_by(97).collect();
        for s in scopes {
            for b in [s.start_byte, s.end_byte] {
                bytes.extend([b.saturating_sub(1), b, b + 1]);
            }
        }
        bytes.retain(|&b| b <= len);
        bytes
    }

    #[test]
    fn index_matches_linear_scan_on_all_files() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        python_files(&root.join("corpus"), &mut files);
        python_files(&root.join("bench"), &mut files);
        assert!(files.len() > 20);
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            let scopes = scopes_of(&text);
            let index = ScopeIndex::new(&scopes);
            for byte in bytes_to_check(&scopes, text.len()) {
                assert_eq!(
                    index.lookup(byte),
                    scope_index_for_byte(&scopes, byte),
                    "{} byte {byte}",
                    path.display()
                );
            }
        }
    }

    fn scope(start_byte: usize, end_byte: usize) -> FunctionShapeScope {
        FunctionShapeScope {
            function_name: None,
            start_byte,
            end_byte,
            shapes: HashMap::new(),
            return_shape: None,
            param_order: Vec::new(),
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
        }
    }

    #[test]
    fn equal_size_scopes_resolve_to_the_later_one() {
        let scopes = vec![scope(0, 100), scope(10, 50), scope(10, 50), scope(60, 70)];
        let index = ScopeIndex::new(&scopes);
        for byte in 0..110 {
            assert_eq!(index.lookup(byte), scope_index_for_byte(&scopes, byte));
        }
        assert_eq!(index.lookup(20), Some(2));
    }

    #[test]
    fn overlapping_or_unordered_scopes_match_linear_scan() {
        let scopes = vec![
            scope(40, 90),
            scope(0, usize::MAX),
            scope(10, 60),
            scope(30, 30),
            scope(20, 5),
        ];
        let index = ScopeIndex::new(&scopes);
        for byte in (0..120).chain([usize::MAX - 1, usize::MAX]) {
            assert_eq!(index.lookup(byte), scope_index_for_byte(&scopes, byte));
        }
    }
}
