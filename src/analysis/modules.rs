//! User modules held in containers, and modules imported from other files.
//!
//! A `self.<attr>` field holds one module (`Field::One`) or a sequence of them
//! (`Field::Many`: `nn.ModuleList`, `nn.Sequential`, a list or tuple, or a
//! comprehension of one constructor call). The elements are user modules of
//! this file, modules of another file, or framework layers.
//!
//! What the analyzer says about a sequence must hold for every element that
//! the code can pick:
//! - `nn.Sequential` and `eqx.nn.Sequential` chain their elements in order.
//! - A loop `for layer in self.layers: x = layer(x)` is the same chain, so it
//!   replays as one statement, not as a loop (`ShapeCtx::chain_loop`).
//! - `self.layers[i](x)` with a literal `i` is one element. Any other index is
//!   known only when every element agrees.
//! - Elements that are one constructor call repeated (`Repeat`) apply with the
//!   same shape rule each time. An unknown count (`range(n)`) is known only
//!   when one application gives back the input shape, because `n` may be 0 or
//!   large and the result must hold for every `n`.
//!
//! A module of another file is analyzed as if its file had a function that
//! builds the module and calls it with the shapes of this call (a probe). The
//! result is cached with the disk state it depends on.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use tree_sitter::{Node, Range};

use super::{
    ShapeCtx, Submodule, apply_submodule_call, constructed_module, extra_layer_inputs,
    insert_identifiers, normalize_dim, resolve_call_args,
};
use crate::imports::is_imported;
use crate::layers::{apply_layer_application, classify_inline_constructor};
use crate::python_ast::{ClassHierarchy, dim_symbols, dynamic_attribute, extract_call_arguments};
use crate::resolution::{
    ModuleProbe, resolve_call_target, resolve_implementation, search_roots_fingerprint,
};
use crate::types::*;

/// The most elements a comprehension with a literal count is chained over.
const MAX_REPEAT: usize = 64;

/// How many probes may run inside one another.
const MAX_PROBE_DEPTH: usize = 2;

/// The largest file a probe is run on.
const MAX_PROBE_FILE: usize = 512 << 10;

/// Python keywords: an expression with one of them is not copied to a probe.
pub(super) const KEYWORDS: [&str; 30] = [
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "while", "yield",
];

/// Packages whose classes are not user modules.
pub(crate) const LIBRARIES: [&str; 18] = [
    "torch",
    "jax",
    "flax",
    "equinox",
    "numpy",
    "einops",
    "optax",
    "jaxtyping",
    "scipy",
    "tensorflow",
    "keras",
    "math",
    "functools",
    "dataclasses",
    "typing",
    "collections",
    "itertools",
    "abc",
];

const PROBE_FUNCTION: &str = "__ndim_probe__";
const PROBE_OUTPUT: &str = "__ndim_out__";

thread_local! {
    static PROBE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// A module of another file built as `Class(...)` or `lib.Class(...)`.
#[derive(Clone)]
pub(super) struct ImportedModule {
    /// The callee as written.
    callee: String,
    ctor_args: Vec<CallArgument>,
    /// Start of the constructor call, to resolve `self.<attr>` in its arguments.
    byte: usize,
}

#[derive(Clone)]
pub(super) enum Module {
    Local(Submodule),
    Imported(ImportedModule),
}

/// One element of a sequence: a module, or a framework layer.
#[derive(Clone)]
pub(super) enum Held {
    Module(Module),
    /// A layer and where its constructor call stands.
    Layer(LayerKind, usize),
}

#[derive(Clone)]
pub(super) enum Items {
    List(Vec<Held>),
    /// One constructor call, repeated. `count` is known for `range(<literal>)`.
    Repeat {
        item: Box<Held>,
        count: Option<usize>,
    },
}

#[derive(Clone)]
pub(super) struct Sequence {
    items: Items,
    /// `nn.Sequential` can be called; a list cannot.
    callable: bool,
}

#[derive(Clone)]
pub(super) enum Field {
    One(Module),
    Many(Sequence),
}

impl Module {
    fn same(&self, other: &Module) -> bool {
        match (self, other) {
            (Module::Local(a), Module::Local(b)) => {
                a.class == b.class && a.ctor_args == b.ctor_args
            }
            (Module::Imported(a), Module::Imported(b)) => {
                a.callee == b.callee && a.ctor_args == b.ctor_args
            }
            _ => false,
        }
    }
}

impl Held {
    fn same(&self, other: &Held) -> bool {
        match (self, other) {
            (Held::Module(a), Held::Module(b)) => a.same(b),
            (Held::Layer(a, _), Held::Layer(b, _)) => a == b,
            _ => false,
        }
    }
}

impl Field {
    pub(super) fn same(&self, other: &Field) -> bool {
        match (self, other) {
            (Field::One(a), Field::One(b)) => a.same(b),
            (Field::Many(a), Field::Many(b)) => {
                a.callable == b.callable
                    && match (&a.items, &b.items) {
                        (Items::List(x), Items::List(y)) => {
                            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| x.same(y))
                        }
                        (
                            Items::Repeat { item: x, count: m },
                            Items::Repeat { item: y, count: n },
                        ) => m == n && x.same(y),
                        _ => false,
                    }
            }
            _ => false,
        }
    }

    pub(super) fn is_sequence(&self) -> bool {
        matches!(self, Field::Many(_))
    }
}

// ---------------------------------------------------------------- fields

/// What `value` (the right side of `self.<attr> = value`) makes the field hold.
/// `None` when it is anything else, so the field stays unknown.
pub(super) fn field_of(
    value: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Field> {
    // A framework layer is held through `self_attr_layers`, which a call
    // looks up first.
    if let Some(sequence) = sequence_of(value, text, import_map, hierarchy) {
        return Some(Field::Many(sequence));
    }
    module_of(value, text, import_map, hierarchy).map(Field::One)
}

fn module_of(
    call: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Module> {
    if let Some(module) = constructed_module(call, text, import_map, hierarchy) {
        return Some(Module::Local(module));
    }
    imported_module_of(call, text, import_map).map(Module::Imported)
}

/// `Class(...)` or `lib.Class(...)` where the first name is imported.
pub(super) fn imported_module_of(
    call: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
) -> Option<ImportedModule> {
    if call.kind() != "call" {
        return None;
    }
    let func = call.child_by_field_name("function")?;
    if !matches!(func.kind(), "identifier" | "attribute") {
        return None;
    }
    let callee = func.utf8_text(text.as_bytes()).ok()?;
    let mut names = callee.split('.');
    if !names.clone().all(is_identifier) {
        return None;
    }
    // A framework or standard-library callee is not a user module.
    let root = names.next()?;
    if !is_imported(import_map, root) {
        return None;
    }
    let root = import_map.get(root)?;
    if LIBRARIES.contains(&root.module.first().unwrap_or(&root.name).as_str()) {
        return None;
    }
    Some(ImportedModule {
        callee: callee.to_string(),
        ctor_args: extract_call_arguments(call.child_by_field_name("arguments")?, text).ok()?,
        byte: call.start_byte(),
    })
}

fn held_of(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Held> {
    if node.kind() != "call" {
        return None;
    }
    if let Some(kind) = classify_inline_constructor(node, text, import_map) {
        return Some(Held::Layer(kind, node.start_byte()));
    }
    module_of(node, text, import_map, hierarchy).map(Held::Module)
}

fn sequence_of(
    value: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Sequence> {
    let items = |node: Node| items_of(node, text, import_map, hierarchy);
    match value.kind() {
        "list" | "tuple" | "list_comprehension" | "generator_expression" => Some(Sequence {
            items: items(value)?,
            callable: false,
        }),
        "call" => {
            let func = value.child_by_field_name("function")?;
            let callee = func.utf8_text(text.as_bytes()).ok()?;
            if !callee.ends_with("ModuleList") && !callee.ends_with("Sequential") {
                return None;
            }
            let target = resolve_call_target(callee, import_map);
            let parts: Vec<&str> = target.parts.iter().map(String::as_str).collect();
            let args = value.child_by_field_name("arguments")?;
            let arguments: Vec<Node> = match args.kind() {
                "generator_expression" => vec![args],
                _ => (0..args.named_child_count())
                    .filter_map(|i| args.named_child(i as u32))
                    .filter(|n| n.kind() != "comment")
                    .collect(),
            };
            match (target.dots, parts.as_slice()) {
                (0, ["torch", "nn", "ModuleList"]) => {
                    let [only] = arguments.as_slice() else {
                        return None;
                    };
                    Some(Sequence {
                        items: items(*only)?,
                        callable: false,
                    })
                }
                (0, ["equinox", "nn", "Sequential"]) => {
                    let [only] = arguments.as_slice() else {
                        return None;
                    };
                    Some(Sequence {
                        items: items(*only)?,
                        callable: true,
                    })
                }
                (0, ["torch", "nn", "Sequential"]) => {
                    let items = match arguments.as_slice() {
                        [only] if only.kind() == "list_splat" => items(only.named_child(0)?)?,
                        calls => normalize(
                            calls
                                .iter()
                                .map(|n| held_of(*n, text, import_map, hierarchy))
                                .collect::<Option<Vec<_>>>()?,
                        ),
                    };
                    Some(Sequence {
                        items,
                        callable: true,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The elements a list, tuple, or comprehension builds.
fn items_of(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Items> {
    match node.kind() {
        "list" | "tuple" => {
            let mut held = Vec::new();
            for i in 0..node.named_child_count() {
                let child = node.named_child(i as u32)?;
                if child.kind() != "comment" {
                    held.push(held_of(child, text, import_map, hierarchy)?);
                }
            }
            Some(normalize(held))
        }
        "list_comprehension" | "generator_expression" => {
            let item = held_of(
                node.child_by_field_name("body")?,
                text,
                import_map,
                hierarchy,
            )?;
            let mut used = HashSet::new();
            insert_identifiers(node.child_by_field_name("body")?, text, &mut used);
            let mut bound = HashSet::new();
            let (mut loops, mut filters) = (Vec::new(), false);
            for i in 0..node.named_child_count() {
                let child = node.named_child(i as u32)?;
                match child.kind() {
                    "for_in_clause" => {
                        insert_identifiers(child.child_by_field_name("left")?, text, &mut bound);
                        loops.push(child);
                    }
                    "if_clause" => filters = true,
                    _ => {}
                }
            }
            // Every element must be the same call: nothing it reads may be
            // a loop variable.
            if loops.is_empty() || !used.is_disjoint(&bound) {
                return None;
            }
            let count = match (loops.as_slice(), filters) {
                ([only], false) => range_count(only.child_by_field_name("right")?, text),
                _ => None,
            };
            Some(Items::Repeat {
                item: Box::new(item),
                count,
            })
        }
        _ => None,
    }
}

/// `n` for `range(n)` with an integer literal `n` no larger than `MAX_REPEAT`.
fn range_count(iterable: Node, text: &str) -> Option<usize> {
    if iterable.kind() != "call" {
        return None;
    }
    let func = iterable.child_by_field_name("function")?;
    if func.utf8_text(text.as_bytes()).ok()? != "range" {
        return None;
    }
    let args = iterable.child_by_field_name("arguments")?;
    if args.named_child_count() != 1 {
        return None;
    }
    let only = args.named_child(0)?;
    if only.kind() != "integer" {
        return None;
    }
    let digits = only.utf8_text(text.as_bytes()).ok()?;
    digits
        .bytes()
        .all(|b| b.is_ascii_digit())
        .then(|| digits.parse::<usize>().ok())
        .flatten()
        .filter(|&n| n <= MAX_REPEAT)
}

/// A list of equal elements is one element repeated.
fn normalize(held: Vec<Held>) -> Items {
    match held.as_slice() {
        [first, rest @ ..] if !rest.is_empty() && rest.iter().all(|h| h.same(first)) => {
            Items::Repeat {
                count: Some(held.len()),
                item: Box::new(held.into_iter().next().expect("one element at least")),
            }
        }
        _ => Items::List(held),
    }
}

/// Methods that change a list in place.
const MUTATORS: [&str; 14] = [
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "sort",
    "reverse",
    "add_module",
    "register_module",
    "update",
    "__setitem__",
    "__delitem__",
    "__iadd__",
];

/// An attribute that some code changes, and where: the class around the use
/// when the object is `self` (the list of an instance of that class, of a
/// class it inherits from, or of a class that inherits from it), else
/// anywhere.
pub(super) type Changed = (String, Option<(usize, usize)>);

/// The class that `object` stands for when it is `self`, as the range of the
/// class around it. `None` for any other object.
fn class_of_self(object: Option<Node>, text: &str) -> Option<(usize, usize)> {
    let object = object.filter(|o| o.utf8_text(text.as_bytes()) == Ok("self"))?;
    let mut up = object.parent();
    while let Some(node) = up {
        if node.kind() == "class_definition" {
            return Some((node.start_byte(), node.end_byte()));
        }
        up = node.parent();
    }
    None
}

/// The attributes among `attrs` that some code changes in place or lets go
/// of, so that the constructor's list may not be the list a call sees:
/// `<x>.<attr>.append(...)`, `<x>.<attr>[i] = ...`, `del <x>.<attr>[i]`,
/// `<x>.<attr> = ...` on another object, and every use of `<x>.<attr>` that
/// keeps the list under another name (`ls = self.layers`, `return
/// self.layers`, an argument of a call) or reaches it by a string
/// (`getattr(self, "layers")`). Reading it (`for layer in self.layers`,
/// `len(self.layers)`, `self.layers[0](x)`) changes nothing.
pub(super) fn changed_in_place(root: Node, text: &str, attrs: &HashSet<&str>) -> Vec<Changed> {
    let mut changed: HashSet<Changed> = HashSet::new();
    for attr in attrs {
        // Look at the places the name appears, not at every node.
        for (start, _) in word_matches(text, attr) {
            let Some(name) = root.descendant_for_byte_range(start, start + attr.len()) else {
                continue;
            };
            let moved = match name.kind() {
                "identifier" => name
                    .parent()
                    .filter(|a| {
                        a.kind() == "attribute" && a.child_by_field_name("attribute") == Some(name)
                    })
                    .filter(|attribute| attribute_is_changed(*attribute, text))
                    .map(|attribute| class_of_self(attribute.child_by_field_name("object"), text)),
                "string_content" => name
                    .parent()
                    .and_then(|string| string.parent())
                    .and_then(|arguments| arguments.parent())
                    .filter(|call| dynamic_attribute(*call, text) == Some(attr))
                    .map(|call| {
                        class_of_self(
                            call.child_by_field_name("arguments").and_then(|a| a.named_child(0)),
                            text,
                        )
                    }),
                _ => None,
            };
            if let Some(class) = moved {
                changed.insert((attr.to_string(), class));
                // Anywhere covers every class.
                if class.is_none() {
                    break;
                }
            }
        }
    }
    changed.into_iter().collect()
}

/// The names that a `for` statement binds over one of the fields `attrs`:
/// `for layer in self.layers` and `for i, layer in enumerate(self.layers)`.
pub(super) fn loop_variables(root: Node, text: &str, attrs: &HashSet<&str>) -> HashSet<String> {
    let mut names = HashSet::new();
    for attr in attrs {
        for (start, _) in word_matches(text, attr) {
            let Some(name) = root.descendant_for_byte_range(start, start + attr.len()) else {
                continue;
            };
            let Some(mut iterated) = name.parent().filter(|p| p.kind() == "attribute") else {
                continue;
            };
            // `enumerate(self.layers)`: the loop iterates the call.
            if let Some(arguments) = iterated.parent().filter(|p| p.kind() == "argument_list")
                && let Some(call) = arguments.parent().filter(|p| p.kind() == "call")
            {
                iterated = call;
            }
            if let Some(statement) = iterated.parent().filter(|p| p.kind() == "for_statement")
                && statement.child_by_field_name("right") == Some(iterated)
                && let Some(left) = statement.child_by_field_name("left")
            {
                insert_identifiers(left, text, &mut names);
            }
        }
    }
    names
}

/// Where `word` appears in `text` as a whole word.
fn word_matches<'a>(text: &'a str, word: &'a str) -> impl Iterator<Item = (usize, &'a str)> {
    let bytes = text.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    text.match_indices(word).filter(move |(start, _)| {
        let end = start + word.len();
        !(*start > 0 && is_word(bytes[start - 1])) && !bytes.get(end).is_some_and(|&b| is_word(b))
    })
}

/// Functions that read a list without keeping it.
const READERS: [&str; 7] = ["len", "enumerate", "list", "tuple", "reversed", "zip", "iter"];

fn attribute_is_changed(attribute: Node, text: &str) -> bool {
    let mut at = attribute;
    while let Some(parent) = at.parent().filter(|p| p.kind() == "parenthesized_expression") {
        at = parent;
    }
    let Some(parent) = at.parent() else {
        return false;
    };
    let is_field = |name: &str| parent.child_by_field_name(name) == Some(at);
    match parent.kind() {
        // `<x>.<attr>.append(...)`
        "attribute" if is_field("object") => parent
            .child_by_field_name("attribute")
            .and_then(|m| m.utf8_text(text.as_bytes()).ok())
            .is_some_and(|m| MUTATORS.contains(&m)),
        // `<x>.<attr>[i] = ...`, `del <x>.<attr>[i]`
        "subscript" if is_field("value") => is_store_target(parent),
        // `self.net(x)`
        "call" if is_field("function") => false,
        // `for layer in self.layers`
        "for_statement" | "for_in_clause" if is_field("right") => false,
        "if_statement" | "elif_clause" | "while_statement" if is_field("condition") => false,
        "comparison_operator" | "not_operator" | "boolean_operator" => false,
        "argument_list" => !parent
            .parent()
            .filter(|call| call.kind() == "call")
            .and_then(|call| call.child_by_field_name("function"))
            .and_then(|f| f.utf8_text(text.as_bytes()).ok())
            .is_some_and(|f| READERS.contains(&f)),
        _ => {
            let on_self = attribute
                .child_by_field_name("object")
                .and_then(|o| o.utf8_text(text.as_bytes()).ok())
                == Some("self");
            // `self.<attr> = ...` is a write the field already accounts for.
            // Any other use keeps the list under another name.
            !(on_self && is_store_target(attribute))
        }
    }
}

/// Whether `node` is assigned to, deleted, or the target of a loop or `with`.
fn is_store_target(node: Node) -> bool {
    let mut child = node;
    while let Some(parent) = child.parent() {
        match parent.kind() {
            "pattern_list"
            | "tuple_pattern"
            | "list_pattern"
            | "list_splat_pattern"
            | "parenthesized_expression"
            | "expression_list" => child = parent,
            "assignment" | "augmented_assignment" | "for_statement" | "for_in_clause" => {
                return parent.child_by_field_name("left") == Some(child);
            }
            "delete_statement" => return true,
            _ => return false,
        }
    }
    false
}

// ----------------------------------------------------------------- loops

/// Whether an identifier node is bound there, not read.
pub(super) fn is_binding(node: Node) -> bool {
    let mut child = node;
    while let Some(parent) = child.parent() {
        match parent.kind() {
            "pattern_list"
            | "tuple_pattern"
            | "list_pattern"
            | "list_splat_pattern"
            | "parenthesized_expression"
            | "expression_list"
            | "as_pattern_target" => child = parent,
            "assignment" | "augmented_assignment" | "for_statement" | "for_in_clause" => {
                return parent.child_by_field_name("left") == Some(child);
            }
            "named_expression" => return parent.child_by_field_name("name") == Some(child),
            "as_pattern" => return parent.child_by_field_name("alias") == Some(child),
            "function_definition" | "class_definition" => {
                return parent.child_by_field_name("name") == Some(child);
            }
            "delete_statement"
            | "parameters"
            | "lambda_parameters"
            | "default_parameter"
            | "typed_parameter"
            | "typed_default_parameter"
            | "global_statement"
            | "nonlocal_statement"
            | "import_statement"
            | "import_from_statement"
            | "aliased_import"
            | "dotted_name" => return true,
            _ => return false,
        }
    }
    false
}

/// The byte where each name is bound last under `node`, not counting the
/// bodies of nested functions and lambdas. With `imports` false, a name that
/// only an `import` statement binds is left out.
pub(super) fn binders(node: Node, text: &str, imports: bool) -> HashMap<String, usize> {
    let in_import = |mut n: Node| {
        for _ in 0..4 {
            match n.parent() {
                Some(p) if p.kind().starts_with("import") || p.kind() == "future_import_statement" => {
                    return true;
                }
                Some(p) => n = p,
                None => break,
            }
        }
        false
    };
    let mut last = HashMap::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "identifier" => {
                if is_binding(n)
                    && (imports || !in_import(n))
                    && let Ok(name) = n.utf8_text(text.as_bytes())
                {
                    let at = last.entry(name.to_string()).or_insert(0);
                    *at = (*at).max(n.start_byte());
                }
            }
            "function_definition" | "class_definition" => {
                if let Some(name) = n.child_by_field_name("name") {
                    stack.push(name);
                }
            }
            "lambda" => {}
            _ => {
                let mut cursor = n.walk();
                stack.extend(n.named_children(&mut cursor));
            }
        }
    }
    last
}

/// Whether an identifier under `root` is `name`.
fn names_identifier(root: Node, text: &str, name: &str) -> bool {
    if root.kind() == "identifier" {
        return root.utf8_text(text.as_bytes()) == Ok(name);
    }
    (0..root.named_child_count())
        .filter_map(|i| root.named_child(i as u32))
        .any(|child| names_identifier(child, text, name))
}

/// Whether `name` is bound anywhere under `root` outside `except`.
fn binds_name(root: Node, text: &str, name: &str, except: Node) -> bool {
    let mut cursor = root.walk();
    'walk: loop {
        let node = cursor.node();
        let skip = node.id() == except.id();
        if !skip
            && node.kind() == "identifier"
            && node.utf8_text(text.as_bytes()) == Ok(name)
            && is_binding(node)
        {
            return true;
        }
        if !skip && cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                break 'walk;
            }
        }
    }
    false
}

/// The attribute `A` of `self.A`, or of `enumerate(self.A)`; and which of the
/// two forms it was.
fn iterated_field<'t>(iterable: Node<'t>, text: &'t str, enumerated: bool) -> Option<&'t str> {
    let text_of = |n: Node<'t>| -> Option<&'t str> { n.utf8_text(text.as_bytes()).ok() };
    let target = if enumerated {
        if iterable.kind() != "call" {
            return None;
        }
        let args = iterable.child_by_field_name("arguments")?;
        if text_of(iterable.child_by_field_name("function")?)? != "enumerate"
            || args.named_child_count() != 1
        {
            return None;
        }
        args.named_child(0)?
    } else {
        iterable
    };
    if target.kind() != "attribute" || text_of(target.child_by_field_name("object")?)? != "self" {
        return None;
    }
    text_of(target.child_by_field_name("attribute")?)
}

/// The variable of `for <name> in self.A` or `for <index>, <name> in
/// enumerate(self.A)`, and `A`.
fn loop_variable<'t>(for_node: Node<'t>, text: &'t str) -> Option<(&'t str, &'t str)> {
    let left = for_node.child_by_field_name("left")?;
    let right = for_node.child_by_field_name("right")?;
    let name_of = |n: Node<'t>| n.utf8_text(text.as_bytes()).ok();
    match left.kind() {
        "identifier" => Some((name_of(left)?, iterated_field(right, text, false)?)),
        "pattern_list" | "tuple_pattern" if left.named_child_count() == 2 => {
            let (index, name) = (left.named_child(0)?, left.named_child(1)?);
            if index.kind() != "identifier" || name.kind() != "identifier" {
                return None;
            }
            Some((name_of(name)?, iterated_field(right, text, true)?))
        }
        _ => None,
    }
}

/// `x = layer(x)`: a loop body that is only this statement. Gives the loop
/// variable and `x`.
fn chain_statement<'t>(for_node: Node<'t>, text: &'t str) -> Option<(&'t str, &'t str)> {
    if for_node.kind() != "for_statement" || for_node.child_by_field_name("alternative").is_some() {
        return None;
    }
    let (variable, _) = loop_variable(for_node, text)?;
    let body = for_node.child_by_field_name("body")?;
    let mut statements = (0..body.named_child_count())
        .filter_map(|i| body.named_child(i as u32))
        .filter(|n| n.kind() != "comment");
    let statement = statements.next()?;
    if statements.next().is_some()
        || statement.kind() != "expression_statement"
        || statement.named_child_count() != 1
    {
        return None;
    }
    let assignment = statement.named_child(0)?;
    if assignment.kind() != "assignment" || assignment.child_by_field_name("type").is_some() {
        return None;
    }
    let (left, right) = (
        assignment.child_by_field_name("left")?,
        assignment.child_by_field_name("right")?,
    );
    let text_of = |n: Node<'t>| n.utf8_text(text.as_bytes()).ok();
    if left.kind() != "identifier" || right.kind() != "call" {
        return None;
    }
    let callee = right.child_by_field_name("function")?;
    let args = right.child_by_field_name("arguments")?;
    if callee.kind() != "identifier"
        || text_of(callee)? != variable
        || args.named_child_count() != 1
    {
        return None;
    }
    let argument = args.named_child(0)?;
    (argument.kind() == "identifier" && text_of(argument)? == text_of(left)?)
        .then_some((variable, text_of(left)?))
}

impl ShapeCtx<'_> {
    /// The sequence that `name` ranges over at `at`, when `name` is the
    /// variable of an enclosing `for ... in self.<attr>` (or its `enumerate`)
    /// and nothing else in that loop binds the name.
    fn loop_binding<'t>(&self, at: Node<'t>, name: &str) -> Option<(Node<'t>, String, Sequence)> {
        if !self.submodules.loop_variables.contains(name) {
            return None;
        }
        let mut up = at.parent();
        while let Some(node) = up {
            match node.kind() {
                "function_definition" | "lambda" | "class_definition" => return None,
                "for_statement" => {
                    let body = node.child_by_field_name("body")?;
                    let left = node.child_by_field_name("left")?;
                    if names_identifier(left, self.text, name)
                        && body.byte_range().contains(&at.start_byte())
                    {
                        let (variable, attr) = loop_variable(node, self.text)?;
                        if variable != name || binds_name(node, self.text, name, left) {
                            return None;
                        }
                        let field = self.submodule_at(attr, node.start_byte())?;
                        let Some(Field::Many(sequence)) = &field.module else {
                            return None;
                        };
                        return Some((node, attr.to_string(), sequence.clone()));
                    }
                }
                _ => {}
            }
            up = node.parent();
        }
        None
    }

    /// Whether a loop is a chain, which is replayed as one statement: it runs
    /// over a field that holds a sequence, and its only statement is
    /// `x = layer(x)`.
    pub(super) fn chain_loop(&self, for_node: Node) -> bool {
        if self.submodules.loop_variables.is_empty() {
            return false;
        }
        let Some((variable, _)) = chain_statement(for_node, self.text) else {
            return false;
        };
        let Some(body) = for_node.child_by_field_name("body") else {
            return false;
        };
        self.loop_binding(body, variable)
            .is_some_and(|(node, _, _)| node.id() == for_node.id())
    }

    /// Whether the call `name(...)` at `at` calls a loop variable over a
    /// sequence field.
    pub(super) fn is_loop_module(&self, at: Node, name: &str) -> bool {
        self.loop_binding(at, name).is_some()
    }
}

// ----------------------------------------------------------------- calls

fn quietly<T>(ctx: &mut ShapeCtx, f: impl FnOnce(&mut ShapeCtx) -> T) -> T {
    let errors = ctx.errors.len();
    let result = f(ctx);
    ctx.errors.truncate(errors);
    result
}

pub(super) fn apply_module(
    module: &Module,
    display: &str,
    args: &[CallArgument],
    call_byte: usize,
    scope: usize,
    range: Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    match module {
        Module::Local(module) => {
            apply_submodule_call(module, display, args, call_byte, scope, range, ctx)
        }
        Module::Imported(module) => apply_imported_module(module, display, args, scope, range, ctx),
    }
}

fn apply_held(
    held: &Held,
    display: &str,
    args: &[CallArgument],
    call_byte: usize,
    scope: usize,
    range: Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    match held {
        Held::Module(module) => apply_module(module, display, args, call_byte, scope, range, ctx),
        Held::Layer(kind, byte) => {
            let Some(CallArgument::Positional { value: input }) = args.first() else {
                return None;
            };
            let kind = ctx.layer_at(kind, *byte);
            let application = LayerApplication {
                variable: String::new(),
                layer: display.to_string(),
                input: input.clone(),
                extra_inputs: extra_layer_inputs(args, &kind),
                kind,
                range,
            };
            match apply_layer_application(&application, &ctx.scope_shapes(scope)) {
                Ok(shape) => shape,
                Err(message) => {
                    ctx.report(ShapeError::mismatch(String::new(), message, range));
                    None
                }
            }
        }
    }
}

/// The call of a field that holds a sequence: `self.net(x)` for a sequential
/// container. The elements apply in order.
pub(super) fn apply_sequence_call(
    sequence: &Sequence,
    display: &str,
    args: &[CallArgument],
    call_byte: usize,
    scope: usize,
    range: Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if !sequence.callable {
        return None;
    }
    // An equinox container takes a `key` that does not change a shape.
    let inputs: Vec<&CallArgument> = args
        .iter()
        .filter(|a| !matches!(a, CallArgument::Keyword { name, .. } if name == "key"))
        .collect();
    let [CallArgument::Positional { value: input }] = inputs.as_slice() else {
        return None;
    };
    apply_chain(sequence, display, input, call_byte, scope, range, ctx)
}

/// Apply each element to the output of the one before.
fn apply_chain(
    sequence: &Sequence,
    display: &str,
    input: &str,
    call_byte: usize,
    scope: usize,
    range: Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let start = ctx.resolve_shape(input, scope)?;
    let step = |held: &Held, label: String, name: &str, ctx: &mut ShapeCtx| {
        let args = [CallArgument::Positional {
            value: name.to_string(),
        }];
        apply_held(held, &label, &args, call_byte, scope, range, ctx)
    };
    match &sequence.items {
        Items::List(items) => {
            let (mut shape, mut name) = (start, input.to_string());
            for (i, held) in items.iter().enumerate() {
                let label = format!("{display}[{i}]");
                shape = step(held, label.clone(), &name, ctx)?;
                name =
                    ctx.bind_synthetic(shape.clone(), scope, call_byte, &format!("{label}(...)"));
            }
            Some(shape)
        }
        // The count may be 0 or large: the shape must be the same for every
        // count, so one application must give back the input.
        Items::Repeat { item, count: None } => {
            let out = quietly(ctx, |ctx| step(item, format!("{display}[0]"), input, ctx))?;
            shapes_canonically_equal(&out, &start).then_some(start)
        }
        Items::Repeat {
            item,
            count: Some(count),
        } => {
            let (mut shape, mut name) = (start, input.to_string());
            for i in 0..*count {
                let label = format!("{display}[{i}]");
                let out = step(item, label.clone(), &name, ctx)?;
                // The same module on the same shape: nothing changes again.
                if shapes_canonically_equal(&out, &shape) {
                    break;
                }
                name = ctx.bind_synthetic(out.clone(), scope, call_byte, &format!("{label}(...)"));
                shape = out;
            }
            Some(shape)
        }
    }
}

/// A call of the variable of `for layer in self.<attr>`. `None` when `name` is
/// not such a variable. For one, `Some(None)` is an unknown result.
pub(super) fn shape_of_loop_module_call(
    call: Node,
    callee: Node,
    args_node: Node,
    scope: usize,
    ctx: &mut ShapeCtx,
) -> Option<Option<Vec<String>>> {
    let name = callee.utf8_text(ctx.text.as_bytes()).ok()?;
    let (for_node, attr, sequence) = ctx.loop_binding(callee, name)?;
    let display = format!("self.{attr}");
    let (call_byte, range) = (call.start_byte(), args_node.range());
    let args = extract_call_arguments(args_node, ctx.text)
        .ok()
        .and_then(|raw| resolve_call_args(raw, args_node, scope, ctx));
    let Some(args) = args else {
        return Some(None);
    };
    if let Some((_, variable)) = chain_statement(for_node, ctx.text)
        && ctx.chain_loop(for_node)
    {
        return Some(apply_chain(
            &sequence, &display, variable, call_byte, scope, range, ctx,
        ));
    }
    // Any other use sees the element of one iteration. Elements that are
    // not all the same call tell apart only by the iteration.
    let result = match &sequence.items {
        Items::Repeat { count: Some(0), .. } => None,
        Items::Repeat {
            item,
            count: Some(_),
        } => apply_held(
            item,
            &format!("{display}[0]"),
            &args,
            call_byte,
            scope,
            range,
            ctx,
        ),
        Items::Repeat { item, count: None } => quietly(ctx, |ctx| {
            apply_held(
                item,
                &format!("{display}[0]"),
                &args,
                call_byte,
                scope,
                range,
                ctx,
            )
        }),
        Items::List(items) => match items.as_slice() {
            [only] => apply_held(
                only,
                &format!("{display}[0]"),
                &args,
                call_byte,
                scope,
                range,
                ctx,
            ),
            _ => None,
        },
    };
    Some(result)
}

/// `self.<attr>[index](...)`. `None` when the callee is not that form.
pub(super) fn shape_of_indexed_module_call(
    call: Node,
    callee: Node,
    args_node: Node,
    scope: usize,
    ctx: &mut ShapeCtx,
) -> Option<Option<Vec<String>>> {
    let value = callee.child_by_field_name("value")?;
    if value.kind() != "attribute"
        || value
            .child_by_field_name("object")?
            .utf8_text(ctx.text.as_bytes())
            .ok()?
            != "self"
    {
        return None;
    }
    let attr = value
        .child_by_field_name("attribute")?
        .utf8_text(ctx.text.as_bytes())
        .ok()?;
    let field = ctx.submodule_at(attr, value.start_byte())?;
    let Some(Field::Many(sequence)) = field.module.clone() else {
        return Some(None);
    };
    let subscript = callee.child_by_field_name("subscript")?;
    // A slice is a container, not an element.
    if subscript.kind() == "slice" || callee.named_child_count() != 2 {
        return Some(None);
    }
    let index = literal_index(subscript, ctx.text);
    let display = format!("self.{attr}");
    let (call_byte, range) = (call.start_byte(), args_node.range());
    let args = extract_call_arguments(args_node, ctx.text)
        .ok()
        .and_then(|raw| resolve_call_args(raw, args_node, scope, ctx));
    let Some(args) = args else {
        return Some(None);
    };
    let label = |i: i64| format!("{display}[{i}]");
    let result = match (&sequence.items, index) {
        (Items::List(items), Some(i)) => {
            let at = if i < 0 { i + items.len() as i64 } else { i };
            usize::try_from(at)
                .ok()
                .and_then(|at| items.get(at))
                .and_then(|held| apply_held(held, &label(i), &args, call_byte, scope, range, ctx))
        }
        (Items::List(items), None) => {
            // Any element can be picked: known only if all agree.
            let mut agreed: Option<Vec<String>> = None;
            for held in items {
                let out = quietly(ctx, |ctx| {
                    apply_held(held, &display, &args, call_byte, scope, range, ctx)
                });
                match (&agreed, out) {
                    (_, None) => return Some(None),
                    (Some(seen), Some(out)) if *seen != out => return Some(None),
                    (_, out) => agreed = out,
                }
            }
            agreed
        }
        (Items::Repeat { item, count }, index) => {
            let in_range = match (count, index) {
                (Some(count), Some(i)) => (-(*count as i64)..*count as i64).contains(&i),
                _ => true,
            };
            if in_range {
                apply_held(
                    item,
                    &label(index.unwrap_or(0)),
                    &args,
                    call_byte,
                    scope,
                    range,
                    ctx,
                )
            } else {
                None
            }
        }
    };
    Some(result)
}

/// An integer literal index: `0`, `-1`.
fn literal_index(node: Node, text: &str) -> Option<i64> {
    let source = node.utf8_text(text.as_bytes()).ok()?;
    let digits = source.strip_prefix('-').unwrap_or(source);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    source.parse().ok()
}

// -------------------------------------------------------------- imported

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// A constructor argument as source for a probe, and the plain names it
/// reads, which the probe takes as parameters. `None` for anything the probe
/// cannot carry over: an attribute, a call, a keyword, a string with escapes.
fn probe_expression(value: &str) -> Option<(String, Vec<String>)> {
    let value = value.trim();
    if value.is_empty()
        || !value
            .chars()
            .all(|c| c.is_alphanumeric() || " _+-*/(),.[]'\"".contains(c))
    {
        return None;
    }
    let chars: Vec<char> = value.chars().collect();
    let mut names: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if chars[i].is_alphabetic() || chars[i] == '_' {
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let attribute = (start > 0 && chars[start - 1] == '.') || chars.get(i) == Some(&'.');
            if attribute {
                return None;
            }
            match word.as_str() {
                "True" | "False" | "None" => {}
                w if KEYWORDS.contains(&w) || w.starts_with("__ndim") => return None,
                _ if names.contains(&word) => {}
                _ => names.push(word),
            }
        } else if chars[i].is_ascii_digit() {
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    Some((value.to_string(), names))
}

/// The dims of a shape as a `jaxtyping` annotation, when each can be written
/// there. A scalar cannot.
fn probe_annotation(shape: &[String]) -> Option<String> {
    let writable = |dim: &String| {
        !dim.is_empty()
            && dim
                .chars()
                .all(|c| c.is_alphanumeric() || "_+-*".contains(c))
            && !has_probe_name(dim)
    };
    (!shape.is_empty() && shape.iter().all(writable))
        .then(|| format!("Float[Array, \"{}\"]", shape.join(" ")))
}

fn has_probe_name(text: &str) -> bool {
    text.contains("__ndim")
}

/// A module of another file. The class is found through the import map like
/// any imported name. Its file gets a probe function that builds the module
/// from this call's constructor arguments and calls it on this call's argument
/// shapes, and the file is analyzed. The probe function's variable gives the
/// result. The errors that the probe call adds to the file are errors of this
/// call.
fn apply_imported_module(
    module: &ImportedModule,
    display: &str,
    args: &[CallArgument],
    scope: usize,
    range: Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if args
        .iter()
        .any(|arg| matches!(arg, CallArgument::Positional { value } if value.starts_with('*')))
        || PROBE_DEPTH.with(Cell::get) >= MAX_PROBE_DEPTH
    {
        return None;
    }
    // Everything the probe depends on is read inside one recording: the
    // resolution of the class and the files the analysis reads. The result
    // stays valid while none of those files changes.
    let (outcome, deps) = match ctx.cache {
        Some(cache) => cache.record_deps(|| probe_module(module, args, scope, ctx)),
        None => (probe_module(module, args, scope, ctx), None),
    };
    let (prepared, result, fresh) = outcome?;
    if fresh && let (Some(cache), Some(deps)) = (ctx.cache, deps) {
        cache.probe_store(prepared.key.clone(), deps, result.clone());
    }
    let restore = |text: &str| {
        let mut text = text.replace(&prepared.constructed, display);
        for (parameter, value) in &prepared.inputs {
            text = text.replace(parameter.as_str(), value);
        }
        text
    };
    for message in &result.errors {
        ctx.report(ShapeError::mismatch(
            display.to_string(),
            restore(message),
            range,
        ));
    }
    result.shape.clone()
}

/// A probe ready to run.
struct Prepared {
    key: String,
    class: String,
    /// The constructor call as the probe writes it.
    constructed: String,
    path: std::path::PathBuf,
    probe: String,
    /// The probe parameter of each call argument, and the argument.
    inputs: Vec<(String, String)>,
    /// The names of the probe that stand for something of the caller: the
    /// plain names of the constructor arguments, and the dims of the inputs.
    caller_names: HashSet<String>,
}

/// The file and the name of the user class that `module` calls, found through
/// the import machinery. A library class is not one: it is not analyzed as a
/// user module.
pub(super) fn imported_class(
    module: &ImportedModule,
    ctx: &ShapeCtx,
) -> Option<(std::path::PathBuf, String)> {
    let target = resolve_call_target(&module.callee, ctx.imports.at(module.byte));
    let implementation = resolve_implementation(
        target,
        ctx.search_roots,
        ctx.read_file,
        ctx.max_depth,
        ctx.cache,
    )
    .ok()??;
    let Some(PythonSymbol::Class { name: class }) = implementation.symbol.clone() else {
        return None;
    };
    let path = implementation.target.file_path;
    let library = path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some("site-packages" | "dist-packages")
        )
    });
    (implementation.target.symbol_parts == [class.clone()] && !library).then_some((path, class))
}

/// Find the class, write the probe, and use the cached result if there is
/// one. The last value says whether the probe ran just now.
fn probe_module(
    module: &ImportedModule,
    args: &[CallArgument],
    scope: usize,
    ctx: &ShapeCtx,
) -> Option<(Prepared, std::sync::Arc<ModuleProbe>, bool)> {
    let (path, class) = imported_class(module, ctx)?;
    let mut parameters: Vec<String> = Vec::new();
    let mut caller_names: HashSet<String> = HashSet::new();
    let mut constructor = Vec::new();
    for arg in &module.ctor_args {
        let (keyword, value) = match arg {
            CallArgument::Positional { value } => (None, value),
            CallArgument::Keyword { name, value } => (Some(name), value),
        };
        let value = normalize_dim(value, ctx.aliases, ctx.hierarchy, module.byte);
        probe_expression(&value)?;
        // A probe name keeps an unproven value out of the result.
        let Some(value) = ctx.argument_value(&value, module.byte) else {
            let parameter = format!("__ndim_u{}__", parameters.len());
            constructor.push(match keyword {
                Some(name) => format!("{name}={parameter}"),
                None => parameter.clone(),
            });
            parameters.push(parameter);
            continue;
        };
        let (source, names) = probe_expression(&value)?;
        for name in names {
            if name == class {
                return None;
            }
            if !parameters.contains(&name) {
                caller_names.insert(name.clone());
                parameters.push(name);
            }
        }
        constructor.push(match keyword {
            Some(name) => format!("{name}={source}"),
            None => source,
        });
    }
    let mut call_arguments = Vec::new();
    let mut inputs = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        let (keyword, value) = match arg {
            CallArgument::Positional { value } => (None, value),
            CallArgument::Keyword { name, value } => (Some(name), value),
        };
        let parameter = format!("__ndim_a{i}__");
        let shape = ctx.resolve_shape(value, scope);
        let annotation = shape.as_deref().and_then(probe_annotation);
        if annotation.is_some() {
            caller_names.extend(shape.iter().flatten().flat_map(|dim| dim_symbols(dim)).map(str::to_string));
        }
        parameters.push(match annotation {
            Some(annotation) => format!("{parameter}: {annotation}"),
            None => parameter.clone(),
        });
        call_arguments.push(match keyword {
            Some(name) => format!("{name}={parameter}"),
            None => parameter.clone(),
        });
        inputs.push((parameter, value.clone()));
    }
    let probe = format!(
        "\n\ndef {PROBE_FUNCTION}({}):\n    {PROBE_OUTPUT} = {class}({})({})\n",
        parameters.join(", "),
        constructor.join(", "),
        call_arguments.join(", "),
    );
    let key = format!(
        "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{probe}",
        path.display(),
        search_roots_fingerprint(ctx.search_roots),
        ctx.max_depth,
        PROBE_DEPTH.with(Cell::get),
    );
    let prepared = Prepared {
        key,
        class: class.clone(),
        constructed: format!("{class}({})", constructor.join(", ")),
        path,
        probe,
        inputs,
        caller_names,
    };
    if let Some(found) = ctx
        .cache
        .and_then(|cache| cache.probe_lookup(&prepared.key))
    {
        return Some((prepared, found, false));
    }
    let result = run_probe(
        &prepared.path,
        &prepared.class,
        &prepared.probe,
        &prepared.caller_names,
        ctx,
    );
    Some((prepared, result, true))
}

/// Analyze `path` with `probe` added at its end.
fn run_probe(
    path: &std::path::Path,
    class: &str,
    probe: &str,
    caller_names: &HashSet<String>,
    ctx: &ShapeCtx,
) -> std::sync::Arc<ModuleProbe> {
    let unknown = || {
        std::sync::Arc::new(ModuleProbe {
            shape: None,
            errors: Vec::new(),
        })
    };
    let Some(source) = (ctx.read_file)(&path.to_path_buf()).filter(|f| f.len() <= MAX_PROBE_FILE)
    else {
        return unknown();
    };
    // Only the class and what it names run, so a large file costs no more
    // than the class does.
    let Some(file) = (match ctx.cache {
        Some(cache) => cache.class_source(&source, class, || class_closure(&source, class)),
        None => class_closure(&source, class).map(std::sync::Arc::new),
    }) else {
        return unknown();
    };
    let text = format!("{file}{probe}");
    let Ok(tree) = parse(&text) else {
        return unknown();
    };
    let read = |p: &std::path::PathBuf| (ctx.read_file)(p);
    PROBE_DEPTH.with(|d| d.set(d.get() + 1));
    let analysis = super::analyze_tree(
        tree.root_node(),
        &text,
        ctx.search_roots,
        &read,
        ctx.max_depth,
        ctx.cache,
    );
    PROBE_DEPTH.with(|d| d.set(d.get() - 1));
    let Ok(analysis) = analysis else {
        return unknown();
    };
    // A name of the file that the class reads, like a constant, means nothing
    // to the caller. A dim made of such a name is not a dim of the caller, and
    // a message that names one is not a message about the call.
    let constants = module_constants(tree.root_node(), &text);
    let of_caller = |name: &str| caller_names.contains(name) && !constants.contains(name);
    let shape = analysis
        .scopes
        .iter()
        .find(|s| s.function_name.as_deref() == Some(PROBE_FUNCTION))
        .and_then(|s| s.shapes.get(PROBE_OUTPUT))
        .filter(|shape| {
            !shape.iter().any(|dim| has_probe_name(dim))
                && shape.iter().flat_map(|dim| dim_symbols(dim)).all(of_caller)
        })
        .cloned();

    // An error inside the file that the probe call does not cause is not an
    // error of the call.
    let mut errors: Vec<&ShapeError> = analysis.errors.iter().collect();
    let own = file.len();
    if errors.iter().any(|e| e.range.start_byte < own)
        && let Ok(base_tree) = parse(&file)
        && let Ok(base) = super::analyze_tree(
            base_tree.root_node(),
            &file,
            ctx.search_roots,
            &read,
            ctx.max_depth,
            ctx.cache,
        )
    {
        errors.retain(|e| {
            e.range.start_byte >= own
                || !base
                    .errors
                    .iter()
                    .any(|b| b.message == e.message && b.range == e.range)
        });
    }
    let mut messages: Vec<String> = Vec::new();
    for error in errors {
        let foreign = error
            .message
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(|word| constants.contains(word.strip_prefix(FREE_DIM).unwrap_or(word)));
        if (!has_probe_name(&error.message) || error.message.contains("__ndim_a"))
            && !foreign
            && !messages.contains(&error.message)
        {
            messages.push(error.message.clone());
        }
    }
    std::sync::Arc::new(ModuleProbe {
        shape,
        errors: messages,
    })
}

/// The names that plain assignments of the top-level statements bind.
fn module_constants(root: Node, source: &str) -> HashSet<String> {
    let mut names = Vec::new();
    for i in 0..root.named_child_count() {
        if let Some(unit) = root.named_child(i as u32)
            && unit.kind() == "expression_statement"
        {
            bound_names(unit, source, &mut names);
        }
    }
    names.into_iter().collect()
}

/// The top-level statements of `source` that the class `class` needs, in
/// source order: the imports, the class, and every statement that binds a name
/// that an included statement uses, again and again. `None` when the class is
/// not defined by exactly one statement or the file rebinds module names in a
/// way that this does not follow (`global`).
fn class_closure(source: &str, class: &str) -> Option<String> {
    if source.contains("global ") {
        return Some(source.to_string());
    }
    let tree = parse(source).ok()?;
    let root = tree.root_node();
    let units: Vec<Node> = (0..root.named_child_count())
        .filter_map(|i| root.named_child(i as u32))
        .filter(|n| n.kind() != "comment")
        .collect();
    let mut binders: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    let mut included = vec![false; units.len()];
    let mut queue = Vec::new();
    for (i, unit) in units.iter().enumerate() {
        let mut names = Vec::new();
        bound_names(*unit, source, &mut names);
        for name in names {
            binders.entry(name).or_default().push(i);
        }
        if matches!(
            unit.kind(),
            "import_statement" | "import_from_statement" | "future_import_statement"
        ) {
            included[i] = true;
            queue.push(i);
        }
    }
    let definers = binders.get(class)?;
    let [definer] = definers.as_slice() else {
        return None;
    };
    if !included[*definer] {
        included[*definer] = true;
        queue.push(*definer);
    }
    while let Some(i) = queue.pop() {
        let mut used = HashSet::new();
        insert_identifiers(units[i], source, &mut used);
        for name in used {
            for &j in binders.get(&name).into_iter().flatten() {
                if !included[j] {
                    included[j] = true;
                    queue.push(j);
                }
            }
        }
    }
    let mut out = String::new();
    for (unit, _) in units.iter().zip(&included).filter(|(_, keep)| **keep) {
        out.push_str(unit.utf8_text(source.as_bytes()).ok()?);
        out.push_str("\n\n");
    }
    Some(out)
}

/// The module names that the top-level statement `unit` binds.
fn bound_names(unit: Node, source: &str, out: &mut Vec<String>) {
    let name_of = |n: Node| n.utf8_text(source.as_bytes()).ok().map(str::to_string);
    let definition = if unit.kind() == "decorated_definition" {
        unit.child_by_field_name("definition")
    } else {
        Some(unit)
    };
    if let Some(definition) = definition
        && matches!(
            definition.kind(),
            "function_definition" | "class_definition"
        )
    {
        out.extend(definition.child_by_field_name("name").and_then(name_of));
        return;
    }
    // Any other statement: every name it binds, however deep (an `if`, a
    // `try`, a plain assignment). A name bound inside a function is local, but
    // listing it only adds text to the probe.
    let mut cursor = unit.walk();
    'walk: loop {
        let node = cursor.node();
        if node.kind() == "identifier" && is_binding(node) {
            out.extend(name_of(node));
        }
        if cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                break 'walk;
            }
        }
    }
}

fn parse(text: &str) -> Result<tree_sitter::Tree, String> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;
    parser
        .parse(text, None)
        .ok_or_else(|| "failed to parse".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expression(value: &str) -> Option<(String, Vec<String>)> {
        probe_expression(value)
    }

    #[test]
    fn a_probe_carries_literals_and_plain_names_only() {
        assert_eq!(expression("8"), Some(("8".to_string(), vec![])));
        assert_eq!(expression("0.5"), Some(("0.5".to_string(), vec![])));
        assert_eq!(
            expression("'gelu'"),
            Some(("'gelu'".to_string(), vec!["gelu".to_string()]))
        );
        assert_eq!(
            expression("2 * d + k"),
            Some((
                "2 * d + k".to_string(),
                vec!["d".to_string(), "k".to_string()]
            ))
        );
        assert_eq!(expression("(3, 3)"), Some(("(3, 3)".to_string(), vec![])));
        assert_eq!(expression("True"), Some(("True".to_string(), vec![])));
        assert_eq!(
            expression("f(3)"),
            Some(("f(3)".to_string(), vec!["f".to_string()]))
        );
        for refused in [
            "cfg.dim",
            "self.dim",
            "x if y else z",
            "lambda: 1",
            "{'a': 1}",
            "a.b",
            "__ndim_a0__",
            "",
            "'a\\n'",
        ] {
            assert_eq!(expression(refused), None, "{refused}");
        }
    }

    #[test]
    fn a_literal_index_is_an_integer_without_a_slice_or_a_name() {
        assert_eq!(literal_index_of("0"), Some(0));
        assert_eq!(literal_index_of("-2"), Some(-2));
        assert_eq!(literal_index_of("i"), None);
        assert_eq!(literal_index_of("1:"), None);
        assert_eq!(literal_index_of("-"), None);
    }

    fn literal_index_of(source: &str) -> Option<i64> {
        let text = format!("x[{source}]\n");
        let tree = parse(&text).unwrap();
        let subscript = tree.root_node().named_child(0)?.named_child(0)?;
        literal_index(subscript.child_by_field_name("subscript")?, &text)
    }

    #[test]
    fn a_class_closure_keeps_what_the_class_names() {
        let source = "import torch.nn as nn\nfrom m import helper\n\nWIDTH = 4\nWIDTH = 8\n\ndef scale(d):\n    return d * WIDTH\n\nclass Base(nn.Module):\n    pass\n\nclass Block(Base):\n    def __init__(self, d):\n        self.fc = nn.Linear(scale(d), 3)\n\nclass Other(nn.Module):\n    pass\n\nunused = 3\n";
        let closure = class_closure(source, "Block").unwrap();
        for kept in [
            "import torch.nn as nn",
            "from m import helper",
            "WIDTH = 4",
            "WIDTH = 8",
            "def scale",
            "class Base",
            "class Block",
        ] {
            assert!(closure.contains(kept), "{kept}\n{closure}");
        }
        for left in ["class Other", "unused"] {
            assert!(!closure.contains(left), "{left}\n{closure}");
        }
        // Source order is kept.
        assert!(closure.find("class Base") < closure.find("class Block"));
    }

    #[test]
    fn a_class_closure_needs_one_definition_and_gives_the_file_for_global() {
        let twice = "class A:\n    pass\n\nclass A:\n    pass\n";
        assert_eq!(class_closure(twice, "A"), None);
        assert_eq!(class_closure("class A:\n    pass\n", "B"), None);
        let global =
            "COUNT = 0\n\ndef bump():\n    global COUNT\n    COUNT += 1\n\nclass A:\n    pass\n";
        assert_eq!(class_closure(global, "A").as_deref(), Some(global));
    }

    #[test]
    fn a_name_in_a_conditional_block_binds_for_the_closure() {
        let source = "import torch.nn as nn\ntry:\n    from fast import Act\nexcept ImportError:\n    Act = nn.ReLU\n\nclass Block(nn.Module):\n    def __init__(self):\n        self.act = Act()\n";
        let closure = class_closure(source, "Block").unwrap();
        assert!(
            closure.contains("from fast import Act") && closure.contains("Act = nn.ReLU"),
            "{closure}"
        );
    }
}
