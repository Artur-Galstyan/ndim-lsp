//! The values of the names in the dims of a layer constructor.
//!
//! A layer keeps each dim as the text of its argument, so that the memo of a
//! statement does not depend on the other statements. Every lookup of a layer
//! writes that text with values (`ShapeCtx::layer_at`). A name becomes:
//!
//! - its integer, when the file proves one: every write gives the same literal
//!   and, in the module scope, every write stands before the reader.
//! - itself with the `FREE_DIM` mark, when it means a size the caller gives or
//!   the file does not bind it: a parameter, a dataclass field, or a name with
//!   no binding. The value is not known here, so the dim conflicts with nothing.
//! - itself with the `LITERAL_DIM` mark and its integer, when the file proves
//!   the integer and an annotation also uses the word as a dim: the dim equals
//!   both. An expression of such a name gets only the `FREE_DIM` mark.
//! - itself, when an annotation uses it as a dim and the file does not bind it:
//!   the name is a claim.
//! - `UNPROVEN_DIM` otherwise. The output of the layer stays unknown.
//!
//! A slice writes its size in the same way (`ShapeCtx::slice_size`).

use tree_sitter::Node;

use super::modules::{KEYWORDS, PROBE_FUNCTION};
use super::{
    NameBinding, ShapeCtx, collect_name_values, function_scope, literal_constant, nearest_binding,
    normalize_dim,
};
use crate::layers::{PADDING_NAME, UNPROVEN_DIM, map_layer_dims};
use crate::python_ast::dim_symbols;
use crate::types::{
    CallArgument, FREE_DIM, LITERAL_DIM, LayerKind, MAX_DIM_TEXT, UNKNOWN_DIM, dims_canonically_equal, factor_text,
    fold_integers, has_mark, is_free_dim, is_named_size, literal_dim, min_form, offset_dim,
};

enum Value {
    Integer(String),
    /// A literal of a name that an annotation also uses as a dim.
    Literal { name: String, value: String },
    Name(String),
    Free(String),
    Unproven,
}

impl ShapeCtx<'_> {
    /// `kind` with its dims written with values, for a constructor at `byte`.
    pub(super) fn layer_at(&self, kind: &LayerKind, byte: usize) -> LayerKind {
        map_layer_dims(kind.clone(), &|dim| self.dim_at(dim, byte))
    }

    /// The text of a constructor argument at `byte` with its values. `None`
    /// when the file does not prove what it holds.
    pub(super) fn argument_value(&self, value: &str, byte: usize) -> Option<String> {
        Some(self.dim_at(value, byte)).filter(|written| written != UNPROVEN_DIM)
    }

    /// `shape`, the result of a call, with the `FREE_DIM` mark on each dim that
    /// has a name from the text of an argument: `torch.zeros(3, self.d)`. An
    /// attribute gets the mark also when the class writes one integer to it:
    /// code outside the class can write another.
    pub(super) fn with_free_argument_dims(
        &self,
        shape: Vec<String>,
        args: &[CallArgument],
        byte: usize,
    ) -> Vec<String> {
        let values = || {
            args.iter().map(|arg| match arg {
                CallArgument::Positional { value } | CallArgument::Keyword { value, .. } => value,
            })
        };
        let from_argument = |dim: &str| {
            dim_symbols(dim).any(|name| values().flat_map(|v| dim_symbols(v)).any(|word| word == name))
        };
        // A name of the file with the text of a mark.
        let marked_name = |dim: &str| {
            dim_symbols(dim).any(|name| {
                has_mark(name)
                    && !self.synthetic_text.contains_key(name)
                    && values().flat_map(|v| dim_symbols(v)).any(|word| word == name)
            })
        };
        shape
            .into_iter()
            .map(|dim| {
                if marked_name(&dim) {
                    return UNPROVEN_DIM.to_string();
                }
                if dim.contains(FREE_DIM) || !from_argument(&dim) {
                    return dim;
                }
                let dim = normalize_dim(&dim, self.aliases, self.hierarchy, byte);
                let written = self.dim_at(&dim, byte);
                if written.contains(UNPROVEN_DIM) || dim.contains("self.") {
                    format!("{FREE_DIM}{dim}")
                } else {
                    written
                }
            })
            .collect()
    }

    /// The size of `[start:stop:step]` on an axis of size `axis`. An integer
    /// axis with proven bounds gives the size of Python, and a symbolic axis
    /// gives `symbolic_slice_len`. A bound that is not proven gives
    /// `UNKNOWN_DIM`: the axis stays. `None` for a step of 0, which Python
    /// refuses.
    pub(super) fn slice_size(
        &self,
        [start, stop, step]: [&str; 3],
        axis: &str,
        byte: usize,
    ) -> Option<String> {
        let unknown = || Some(UNKNOWN_DIM.to_string());
        let (Some(step), Some(start)) = (self.slice_bound(step, byte), self.slice_bound(start, byte))
        else {
            return unknown();
        };
        let step = step.unwrap_or(1);
        let stop = if step > 0 && self.stop_is_axis(stop, axis, byte) {
            None
        } else {
            match self.slice_bound(stop, byte) {
                Some(stop) => stop,
                None => return unknown(),
            }
        };
        let whole = stop.is_none()
            && match step {
                1 => start.unwrap_or(0) == 0,
                -1 => start.is_none(),
                _ => false,
            };
        if whole {
            return Some(axis.to_string());
        }
        if step == 0 {
            return None;
        }
        match literal_dim(axis).unwrap_or(axis).parse() {
            Ok(axis) => Some(python_slice_len(start, stop, step, axis).to_string()),
            Err(_) => symbolic_slice_len(start, stop, step, axis).or_else(unknown),
        }
    }

    /// Whether the value of `stop` is the size of `axis`: an integer, or a name
    /// that is an annotation symbol and that no scope binds. An attribute that
    /// holds a parameter stands for the name of the parameter, when no other
    /// write reaches the attribute.
    fn stop_is_axis(&self, stop: &str, axis: &str, byte: usize) -> bool {
        if stop.is_empty() || self.dim_at(stop, byte).contains(UNPROVEN_DIM) {
            return false;
        }
        let written = self.dim_at(&normalize_dim(stop, self.aliases, self.hierarchy, byte), byte);
        !is_free_dim(&written) && dims_canonically_equal(&written, axis)
    }

    /// Whether `step` is a proven negative integer.
    pub(super) fn is_negative_step(&self, step: &str, byte: usize) -> bool {
        self.slice_bound(step, byte).flatten().is_some_and(|step| step < 0)
    }

    /// The value of one part of a slice: `Some(None)` when the part is absent.
    fn slice_bound(&self, bound: &str, byte: usize) -> Option<Option<i64>> {
        if bound.is_empty() || bound == "None" {
            return Some(None);
        }
        let bound = normalize_dim(bound, self.aliases, self.hierarchy, byte);
        if bound.contains("self.") {
            return None;
        }
        let written = self.dim_at(&bound, byte);
        literal_dim(&written)
            .and_then(|value| value.parse().ok())
            .or_else(|| fold_integers(&written))
            .map(Some)
    }

    fn dim_at(&self, dim: &str, byte: usize) -> String {
        if !dim.bytes().any(|b| b.is_ascii_alphabetic() || b == b'_') {
            return dim.to_string();
        }
        let key = (byte, dim.to_string());
        if let Some(known) = self.layer_dims.borrow().get(&key) {
            return known.clone();
        }
        let at = self.root.descendant_for_byte_range(byte, byte).unwrap_or(self.root);
        let written = self.dim_with_values(dim, at);
        self.layer_dims.borrow_mut().insert(key, written.clone());
        written
    }

    fn dim_with_values(&self, dim: &str, at: Node) -> String {
        let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b >= 0x80;
        let bytes = dim.as_bytes();
        let mut written = String::with_capacity(dim.len());
        let mut replaced = false;
        let mut i = 0;
        while i < dim.len() {
            if written.len() > MAX_DIM_TEXT {
                return UNKNOWN_DIM.to_string();
            }
            if matches!(bytes[i], b'"' | b'\'') {
                let literal = i == 0 && bytes.last() == Some(&bytes[0]);
                return if literal { dim.to_string() } else { UNPROVEN_DIM.to_string() };
            }
            if !is_word(bytes[i]) {
                written.push(bytes[i] as char);
                i += 1;
                continue;
            }
            let start = i;
            while i < dim.len() && is_word(bytes[i]) {
                i += 1;
            }
            let word = &dim[start..i];
            if bytes[start].is_ascii_digit() || matches!(word, "True" | "False" | "None") {
                written.push_str(word);
                continue;
            }
            let call = dim[i..].trim_start().starts_with('(') && !word.starts_with(FREE_DIM);
            if KEYWORDS.contains(&word) || call {
                return UNPROVEN_DIM.to_string();
            }
            match self.word_value(word, at) {
                Value::Integer(value) => {
                    replaced = true;
                    written.push_str(&value);
                }
                Value::Literal { name, value } if word.len() == dim.len() => {
                    written.push_str(FREE_DIM);
                    written.push_str(&name);
                    written.push_str(LITERAL_DIM);
                    written.push_str(&value);
                }
                Value::Name(name) => written.push_str(&name),
                Value::Literal { name, .. } | Value::Free(name) => {
                    written.push_str(FREE_DIM);
                    written.push_str(&name);
                }
                Value::Unproven => return UNPROVEN_DIM.to_string(),
            }
        }
        if !replaced {
            return written;
        }
        match fold_integers(&written) {
            Some(value) => value.to_string(),
            None if written.contains(',') || written.bytes().any(|b| b.is_ascii_alphabetic()) => written,
            None => UNPROVEN_DIM.to_string(),
        }
    }

    fn word_value(&self, word: &str, at: Node) -> Value {
        if let Some(name) = word.strip_prefix(FREE_DIM) {
            // The file may bind the marked text itself as a name. The probe
            // function takes marked dims of the caller as parameters.
            let (owner, binding) = nearest_binding(at, word, self);
            let probe = owner.kind() == "function_definition"
                && owner.child_by_field_name("name").and_then(|n| self.text.get(n.byte_range()))
                    == Some(PROBE_FUNCTION);
            let user = has_mark(name) || !(matches!(binding, NameBinding::Unbound) || probe);
            return if user { Value::Unproven } else { Value::Free(name.to_string()) };
        }
        if let Some(name) = word.strip_prefix(PADDING_NAME) {
            return match self.word_value(name, at) {
                Value::Name(_) => Value::Unproven,
                Value::Free(_) if !self.is_integer_parameter(name, at) => Value::Unproven,
                value => value,
            };
        }
        match word.strip_prefix("self.") {
            Some(attr) if !attr.contains('.') => self.attr_value(attr, at),
            None if !word.contains('.') => self.name_value(word, at),
            _ => Value::Unproven,
        }
    }

    fn name_value(&self, name: &str, at: Node) -> Value {
        if has_mark(name) {
            return Value::Unproven;
        }
        let (owner, binding) = nearest_binding(at, name, self);
        let parameter = owner.kind() == "function_definition"
            && function_scope(owner, self)
                .is_some_and(|idx| self.scopes[idx].all_params.iter().any(|p| p == name));
        if parameter {
            return if has_string_default(owner, name, self.text) {
                Value::Unproven
            } else {
                Value::Free(name.to_string())
            };
        }
        let symbol = self.is_annotation_symbol(name);
        if symbol && matches!(binding, NameBinding::Unbound) {
            return Value::Name(name.to_string());
        }
        if let Some(value) = self.constant_before(at, name) {
            return literal_value(name, value, symbol);
        }
        if matches!(binding, NameBinding::Unbound) {
            Value::Free(name.to_string())
        } else {
            Value::Unproven
        }
    }

    fn is_integer_parameter(&self, name: &str, at: Node) -> bool {
        let (owner, _) = nearest_binding(at, name, self);
        owner.kind() == "function_definition" && has_int_annotation(owner, name, self.text)
    }

    /// The integer `name` holds at `at`. A module global holds it only when
    /// every write stands before the top-level statement of the reader, as in
    /// `module_global`.
    fn constant_before(&self, at: Node, name: &str) -> Option<String> {
        let value = literal_constant(at, name, self)?;
        let (owner, binding) = nearest_binding(at, name, self);
        if owner.kind() != "module" {
            return Some(value);
        }
        let NameBinding::Values(writes) = binding else {
            return None;
        };
        let mut top = at;
        while let Some(parent) = top.parent().filter(|p| p.kind() != "module") {
            top = parent;
        }
        (!self.class_body_binds(owner, name)
            && writes.iter().all(|write| write.end_byte() <= top.start_byte()))
        .then_some(value)
    }

    /// What `self.<attr>` holds when any class that the instance can be built
    /// from may write it: every write must agree, in `__init__`, in any other
    /// method, and in a class body.
    fn attr_value(&self, attr: &str, at: Node) -> Value {
        match self.attr_held(attr, at) {
            Value::Integer(value) => literal_value(attr, value, self.is_annotation_symbol(attr)),
            held => held,
        }
    }

    fn attr_held(&self, attr: &str, at: Node) -> Value {
        let mut class = at;
        while class.kind() != "class_definition" {
            let Some(parent) = class.parent() else {
                return Value::Unproven;
            };
            class = parent;
        }
        let family = self.hierarchy.family((class.start_byte(), class.end_byte()));
        let mut integer: Option<String> = None;
        let mut named = false;
        let mut found = |value: Value| match value {
            Value::Integer(value) if *integer.get_or_insert_with(|| value.clone()) == value => true,
            Value::Name(_) | Value::Free(_) => {
                named = true;
                true
            }
            _ => false,
        };
        for write in self.attr_writes {
            if write.attr == attr && family.contains(&write.class) && !found(self.write_value(write.value)) {
                return Value::Unproven;
            }
        }
        for &class in &family {
            let held = if self.hierarchy.fields(class).iter().any(|field| field == attr) {
                Value::Free(attr.to_string())
            } else if self.hierarchy.binds_attr(class, attr) {
                self.class_constant(class, attr).map_or(Value::Unproven, Value::Integer)
            } else {
                continue;
            };
            if !found(held) {
                return Value::Unproven;
            }
        }
        match (integer, named) {
            (Some(value), false) => Value::Integer(value),
            (None, true) => Value::Free(attr.to_string()),
            _ => Value::Unproven,
        }
    }

    fn write_value(&self, value: Option<Node>) -> Value {
        let Some(value) = value else {
            return Value::Unproven;
        };
        let Some(source) = self.text.get(value.byte_range()) else {
            return Value::Unproven;
        };
        if value.kind() == "identifier" {
            return self.name_value(source, value);
        }
        if source.contains("self.") {
            return Value::Unproven;
        }
        let written = self.dim_with_values(source, value);
        if written.bytes().all(|b| b.is_ascii_digit()) {
            Value::Integer(written)
        } else {
            Value::Unproven
        }
    }

    /// The integer literal that the body of `class` assigns to `attr`, when
    /// every binding of the name is such an assignment and they agree.
    fn class_constant(&self, class: (usize, usize), attr: &str) -> Option<String> {
        let mut node = self.root.descendant_for_byte_range(class.0, class.1)?;
        while node.kind() != "class_definition" {
            node = node.parent()?;
        }
        let body = node.child_by_field_name("body")?;
        let (mut values, mut bound) = (Vec::new(), false);
        if !collect_name_values(body, attr, self.text, &mut values, &mut bound) {
            return None;
        }
        let mut literal = None;
        for value in values {
            let text = self.text.get(value.byte_range())?;
            let plain = value.kind() == "integer"
                && value.parent().is_some_and(|p| {
                    p.kind() == "assignment" && p.child_by_field_name("type").is_none()
                });
            if !plain || *literal.get_or_insert(text) != text {
                return None;
            }
        }
        literal.map(str::to_string)
    }
}

/// The node of parameter `name` of `function`.
fn parameter<'t>(function: Node<'t>, name: &str, text: &str) -> Option<Node<'t>> {
    let parameters = function.child_by_field_name("parameters")?;
    (0..parameters.named_child_count()).filter_map(|i| parameters.named_child(i as u32)).find(|parameter| {
        let target = match parameter.kind() {
            "default_parameter" | "typed_default_parameter" => parameter.child_by_field_name("name"),
            "typed_parameter" => parameter.named_child(0),
            _ => Some(*parameter),
        };
        target.is_some_and(|n| n.utf8_text(text.as_bytes()) == Ok(name))
    })
}

/// Whether parameter `name` of `function` has a string as its default.
fn has_string_default(function: Node, name: &str, text: &str) -> bool {
    parameter(function, name, text)
        .and_then(|parameter| parameter.child_by_field_name("value"))
        .is_some_and(|value| value.kind() == "string")
}

/// Whether the annotation of parameter `name` of `function` is `int`.
pub(super) fn has_int_annotation(function: Node, name: &str, text: &str) -> bool {
    parameter_annotation(function, name, text) == Some("int")
}

/// The text of the annotation of parameter `name` of `function`.
pub(super) fn parameter_annotation<'t>(function: Node, name: &str, text: &'t str) -> Option<&'t str> {
    let annotation = parameter(function, name, text)?.child_by_field_name("type")?;
    text.get(annotation.byte_range())
}

/// The value of a literal. `1` stays an integer: a dim of size 1 broadcasts.
fn literal_value(name: &str, value: String, symbol: bool) -> Value {
    if symbol && value != "1" {
        Value::Literal { name: name.to_string(), value }
    } else {
        Value::Integer(value)
    }
}

/// The size of `[start:stop:step]` on the symbolic axis `n`. `step` is not 0.
/// A positive step `k` gives `ceil(s/k)` with the size `s` of `[start:stop]`. A
/// negative step gives a form only without bounds. `None` when no form exists,
/// which is also the result for an axis that `is_named_size` refuses.
fn symbolic_slice_len(start: Option<i64>, stop: Option<i64>, step: i64, n: &str) -> Option<String> {
    if !is_named_size(n) || (step < 0 && (start.is_some() || stop.is_some())) {
        return None;
    }
    let size = form_slice_len(start, stop, n)?;
    if step.abs() == 1 || size == "0" {
        return Some(size);
    }
    Some(format!("{FREE_DIM}ceil({}/{})", factor_text(&size), step.checked_abs()?))
}

/// The size of `[start:stop]` on the axis `n`. A slice of `min(l, e)` is the
/// same as a slice of `e` with the limit `l` in its bounds, so that `[5:]` on
/// `min(4, n)` is 0. A form with `ceil` or another `min` is not below 0 and can
/// have a `min` with an integer. Any other size assumes more than its text
/// proves, so `None`.
fn form_slice_len(start: Option<i64>, stop: Option<i64>, n: &str) -> Option<String> {
    let first = start.unwrap_or(0);
    if let Some((limit, size)) = min_form(n) {
        if start.is_none() && stop.is_none() {
            return Some(n.to_string());
        }
        let (start, stop) = match stop {
            None if first >= 0 => (first, limit),
            Some(stop) if first >= 0 && stop >= 0 => (first, stop.min(limit)),
            Some(stop) if first >= 0 => (first.checked_sub(stop)?, limit),
            None => (0, first.checked_neg()?.min(limit)),
            Some(stop) if stop < 0 => (stop.checked_neg()?, first.checked_neg()?.min(limit)),
            Some(_) => return None,
        };
        return unit_slice_len((start != 0).then_some(start), Some(stop), size);
    }
    if n.contains("min(") || n.contains("ceil(") {
        let fits = match stop {
            None => first <= 0,
            Some(stop) => first == 0 && stop >= 0 || (first < 0) == (stop < 0) && stop <= first,
        };
        if !fits {
            return None;
        }
    }
    unit_slice_len(start, stop, n)
}

/// The size of `[start:stop]` on the symbolic axis `n`, for `a`, `b` > 0:
///
/// - `[a:]` is `n-a`, `[:-b]` is `n-b`, and `[a:-b]` is `n-a-b`. These forms
///   assume that the axis has no less than that many items, as a shorter axis
///   gives 0.
/// - `[:b]` and `[-b:]` are `min(b, n)` for every `n`.
/// - `[a:b]` is `min(b-a, n-a)` and `[-a:-b]` is `min(a-b, n-b)`, with the same
///   assumption for `n-a` and `n-b`. A range with no items is 0.
/// - `[-a:b]` depends on `n` in a way no form here gives: `None`.
///
/// A `min` form has the `FREE_DIM` mark, so it conflicts with nothing.
fn unit_slice_len(start: Option<i64>, stop: Option<i64>, n: &str) -> Option<String> {
    let min = |limit: i64, axis: &str| format!("{FREE_DIM}min({limit}, {axis})");
    let start = start.unwrap_or(0);
    Some(match stop {
        None if start >= 0 => offset_dim(n, start)?,
        None => min(start.checked_neg()?, n),
        Some(stop) if start >= 0 && stop < 0 => offset_dim(n, start.checked_sub(stop)?)?,
        Some(stop) if start < 0 && stop >= 0 => return None,
        Some(stop) if stop <= start => "0".to_string(),
        Some(stop) if start > 0 => min(stop.checked_sub(start)?, &offset_dim(n, start)?),
        Some(stop) if stop < 0 => min(stop.checked_sub(start)?, &offset_dim(n, stop.checked_neg()?)?),
        Some(stop) => min(stop, n),
    })
}

/// The number of items of `range(n)[start:stop:step]`. `step` is not 0.
fn python_slice_len(start: Option<i64>, stop: Option<i64>, step: i64, n: i64) -> i64 {
    let (n, step) = (i128::from(n), i128::from(step));
    let (low, high) = if step > 0 { (0, n) } else { (-1, n - 1) };
    let index = |bound: Option<i64>, default: i128| {
        bound.map_or(default, |bound| {
            let bound = i128::from(bound);
            if bound < 0 { (bound + n).max(low) } else { bound.min(high) }
        })
    };
    let (first, last) = if step > 0 {
        (index(start, low), index(stop, high))
    } else {
        (index(start, high), index(stop, low))
    };
    let span = if step > 0 { last - first } else { first - last };
    if span <= 0 { 0 } else { ((span - 1) / step.abs() + 1) as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value of a slice form at `n`, with `ceil` and `min` as in Python.
    fn value(form: &str, n: i64) -> i64 {
        let form = form.replace(FREE_DIM, "").replace(' ', "").replace("min", "M").replace('n', &n.to_string());
        fn eval(form: &str) -> i64 {
            if let Some(inner) = form.strip_prefix("ceil(").and_then(|f| f.strip_suffix(')')) {
                let (size, step) = inner.rsplit_once('/').unwrap();
                return (eval(size) as f64 / step.parse::<f64>().unwrap()).ceil() as i64;
            }
            if let Some(inner) = form.strip_prefix("M(").and_then(|f| f.strip_suffix(')')) {
                let (left, right) = inner.split_once(',').unwrap();
                return eval(left).min(eval(right));
            }
            fold_integers(form).unwrap()
        }
        eval(&form)
    }

    #[test]
    fn every_slice_form_equals_python_on_an_axis_that_is_big_enough() {
        let bounds = [None, Some(0), Some(1), Some(2), Some(5), Some(-1), Some(-2), Some(-5)];
        let mut forms = 0;
        for start in bounds {
            for stop in bounds {
                for step in [1, 2, 3, -1, -2] {
                    let Some(form) = symbolic_slice_len(start, stop, step, "n") else {
                        continue;
                    };
                    forms += 1;
                    let assumed = start.unwrap_or(0).abs() + stop.unwrap_or(0).abs();
                    for n in assumed..assumed + 12 {
                        let label = format!("{start:?} {stop:?} {step} {form} n={n}");
                        assert_eq!(value(&form, n), python_slice_len(start, stop, step, n), "{label}");
                    }
                }
            }
        }
        assert!(forms > 60, "{forms}");
    }

    #[test]
    fn a_slice_of_a_limited_axis_equals_python() {
        let bounds = [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(7), Some(-1), Some(-2), Some(-3), Some(-4), Some(-7)];
        for (base, take) in [("__free_min(4, n)", 4), ("__free_min(1, n)", 1)] {
            for start in bounds {
                for stop in bounds {
                    for step in [1, 2, 3, -1, -2] {
                        let Some(form) = symbolic_slice_len(start, stop, step, base) else {
                            continue;
                        };
                        let assumed = start.unwrap_or(0).abs() + stop.unwrap_or(0).abs();
                        for n in assumed..assumed + 12 {
                            let size = python_slice_len(None, Some(take), 1, n);
                            let label = format!("{base} {start:?} {stop:?} {step} {form} n={n}");
                            assert_eq!(value(&form, n), python_slice_len(start, stop, step, size), "{label}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_slice_of_a_ceil_axis_equals_python_at_every_size() {
        let bounds = [None, Some(0), Some(1), Some(2), Some(5), Some(-1), Some(-2), Some(-5)];
        for start in bounds {
            for stop in bounds {
                for step in [1, 2, -1] {
                    let base = "__free_ceil(n/2)";
                    let Some(form) = symbolic_slice_len(start, stop, step, base) else {
                        continue;
                    };
                    for n in 0..24 {
                        let size = python_slice_len(None, None, 2, n);
                        let label = format!("{start:?} {stop:?} {step} {form} n={n}");
                        assert_eq!(value(&form, n), python_slice_len(start, stop, step, size), "{label}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_min_form_needs_no_assumption() {
        for n in 0..12 {
            for (form, start, stop) in
                [("__free_min(4, n)", None, Some(4)), ("__free_min(4, n)", Some(-4), None)]
            {
                assert_eq!(value(form, n), python_slice_len(start, stop, 1, n), "{form} n={n}");
            }
        }
    }

    #[test]
    fn a_form_never_overflows() {
        for (start, stop) in [(Some(i64::MIN), None), (Some(i64::MAX), Some(i64::MIN))] {
            assert_eq!(unit_slice_len(start, stop, "n"), None);
        }
    }
}
