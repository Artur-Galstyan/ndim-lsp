//! The values of the names in the dims of a layer constructor.
//!
//! A layer keeps each dim as the text of its argument, so that the memo of a
//! statement does not depend on the other statements. Every lookup of a layer
//! writes that text with values (`ShapeCtx::layer_at`). A name becomes:
//!
//! - its integer, when the file proves one: every write gives the same literal
//!   and, in the module scope, every write stands before the reader.
//! - itself, when it means a size the caller gives or the file does not bind
//!   it: a parameter, a dataclass field, a name that an annotation uses as a
//!   dim, or a name with no binding.
//! - `UNPROVEN_DIM` otherwise. A symbol would differ from any integer, and the
//!   value can be that integer, so the output of the layer stays unknown.

use tree_sitter::Node;

use super::modules::KEYWORDS;
use super::{NameBinding, ShapeCtx, collect_name_values, function_scope, literal_constant, nearest_binding};
use crate::layers::{UNPROVEN_DIM, map_layer_dims};
use crate::types::LayerKind;

enum Value {
    Integer(String),
    Name(String),
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
            if matches!(bytes[i], b'"' | b'\'') {
                return dim.to_string();
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
            if KEYWORDS.contains(&word) || dim[i..].trim_start().starts_with('(') {
                return UNPROVEN_DIM.to_string();
            }
            match self.word_value(word, at) {
                Value::Integer(value) => {
                    replaced = true;
                    written.push_str(&value);
                }
                Value::Name(name) => written.push_str(&name),
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
        match word.strip_prefix("self.") {
            Some(attr) if !attr.contains('.') => self.attr_value(attr, at),
            None if !word.contains('.') => self.name_value(word, at),
            _ => Value::Unproven,
        }
    }

    fn name_value(&self, name: &str, at: Node) -> Value {
        if self.is_annotation_symbol(name) {
            return Value::Name(name.to_string());
        }
        if let Some(value) = self.constant_before(at, name) {
            return Value::Integer(value);
        }
        let (owner, binding) = nearest_binding(at, name, self);
        let parameter = owner.kind() == "function_definition"
            && function_scope(owner, self)
                .is_some_and(|idx| self.scopes[idx].all_params.iter().any(|p| p == name));
        if parameter || matches!(binding, NameBinding::Unbound) {
            Value::Name(name.to_string())
        } else {
            Value::Unproven
        }
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
        if self.is_annotation_symbol(attr) {
            return Value::Name(attr.to_string());
        }
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
            Value::Name(_) => {
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
                Value::Name(attr.to_string())
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
            (None, true) => Value::Name(attr.to_string()),
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

/// The value of `text` when it is integers with `+`, `-`, `*`, `//`, and
/// parentheses.
fn fold_integers(text: &str) -> Option<i64> {
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
