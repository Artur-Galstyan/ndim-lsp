//! Branch merge for the assignment replay.
//!
//! The replay walks the assignments of a scope in source order and keeps one
//! binding per name. A branch breaks that: only one arm runs. This module
//! reads the compound statements of the tree as events (`collect_flow`) and
//! keeps the state of each open statement (`Replay`). Each arm starts from the
//! state before the statement. At the end of the statement, a name keeps a
//! shape only if every path that reaches the end agrees on it.
//!
//! A path that ends in `return` or `raise` does not reach the end. A `break`
//! or `continue` reaches the end of the loop, not the end of its block.
//! Everything here depends only on the statement and its function, so a
//! replayed top-level statement gives what a fresh run gives.

use std::collections::{HashMap, HashSet};

use tree_sitter::Node;

use super::modules::binders;
use super::{
    Lhs, ShapeCtx, TupleTarget, collect_assignment_items, collect_global_names, insert_identifiers,
};
use crate::imports::ImportScopes;
use crate::resolution::resolve_call_target;
use crate::types::{ImportPath, shapes_canonically_equal};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Role {
    /// `elif`, `else` of an `if`, or a later `case`.
    Next,
    Handler,
    Else,
    Finally,
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Jump {
    Return,
    Raise,
    Break,
    Continue,
}

pub(super) enum Kind {
    /// `if` and `match`: one arm runs. Without a final `else` or an
    /// irrefutable `case`, no arm may run.
    Branch { exhaustive: bool },
    /// `for` and `while`: the body runs zero or more times.
    Loop { has_else: bool },
    Try {
        has_else: bool,
        /// Names the `finally` clause assigns.
        finally_writes: Vec<String>,
    },
    /// A `with` whose manager may swallow an exception.
    Suppress,
}

pub(super) enum Event<'t> {
    Open(Kind, Node<'t>),
    Arm(Role),
    Close,
    Jump(Jump),
    /// Names bound in a way the replay does not model (loop target, `with`
    /// target, `except` target, case capture, walrus, `def`, `class`, `import`,
    /// `del`): their shape is unknown from here on. The scope is the one at
    /// the byte of the event.
    Bind(Vec<String>),
    Done,
}

pub(super) struct Marked<'t> {
    byte: usize,
    event: Event<'t>,
}

/// Context managers whose `__exit__` may swallow an exception, so that the
/// body can stop at any point.
const SWALLOWING: [&str; 7] = [
    "suppress",
    "raises",
    "assertRaises",
    "assertRaisesRegex",
    "subTest",
    "ExitStack",
    "AsyncExitStack",
];

/// What `visit` reads besides the tree.
struct Source<'a> {
    text: &'a str,
    imports: &'a ImportScopes,
    /// The statement that `visit` is at is in a class body, with no function
    /// between: the names it binds belong to the class.
    in_class: std::cell::Cell<bool>,
}

/// The events of the compound statements under `root` (and `root` itself), in
/// source order. Nested functions and classes are entered: the flat replay
/// runs their statements in the same pass.
pub(super) fn collect_flow<'t>(
    root: Node<'t>,
    text: &str,
    imports: &ImportScopes,
    in_class: bool,
) -> Vec<Marked<'t>> {
    let mut out = Vec::new();
    let src = Source {
        text,
        imports,
        in_class: std::cell::Cell::new(in_class),
    };
    visit(root, &src, &mut out);
    out
}

fn push<'t>(out: &mut Vec<Marked<'t>>, byte: usize, event: Event<'t>) {
    out.push(Marked { byte, event });
}

fn visit<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    match node.kind() {
        "if_statement" => visit_if(node, src, out),
        "while_statement" | "for_statement" => visit_loop(node, src, out),
        "try_statement" => visit_try(node, src, out),
        "match_statement" => visit_match(node, src, out),
        "with_statement" => visit_with(node, src, out),
        "return_statement" => push(out, node.start_byte(), Event::Jump(Jump::Return)),
        "raise_statement" => push(out, node.start_byte(), Event::Jump(Jump::Raise)),
        "break_statement" => push(out, node.start_byte(), Event::Jump(Jump::Break)),
        "continue_statement" => push(out, node.start_byte(), Event::Jump(Jump::Continue)),
        "function_definition" | "class_definition" => {
            // The name is bound in the scope around the statement, which is
            // the scope of the byte before it. A method or a class field is a
            // member of the class, not a name of the scope.
            if !src.in_class.get()
                && let Some(name) = node.child_by_field_name("name")
            {
                let mut names = Vec::new();
                bound_names(name, src.text, &mut names);
                bind(out, node.start_byte().saturating_sub(1), names);
            }
            let outer = src.in_class.replace(node.kind() == "class_definition");
            if let Some(body) = node.child_by_field_name("body") {
                visit(body, src, out);
            }
            src.in_class.set(outer);
        }
        "import_statement" | "import_from_statement" => {
            let mut names = Vec::new();
            imported_names(node, src.text, &mut names);
            bind(out, node.start_byte(), names);
        }
        "delete_statement" => {
            let mut names = Vec::new();
            deleted_names(node, src.text, &mut names);
            bind(out, node.start_byte(), names);
        }
        "module"
        | "block"
        | "decorated_definition"
        | "elif_clause"
        | "else_clause"
        | "except_clause"
        | "except_group_clause"
        | "finally_clause"
        | "case_clause" => visit_children(node, src, out),
        // A simple statement, or the header of a compound one: only a walrus
        // in it binds a name.
        _ => walrus(node, src.text, node.start_byte(), out),
    }
}

fn visit_children<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        visit(child, src, out);
    }
}

/// A walrus under `node` binds its name where the statement runs, and
/// whether the walrus runs at all (`a and (x := f())`, a comprehension that
/// has no item) is not known, so its name is unknown from the statement on.
/// A lambda has a scope of its own.
fn walrus<'t>(node: Node<'t>, text: &str, byte: usize, out: &mut Vec<Marked<'t>>) {
    if !text.get(node.byte_range()).is_some_and(|source| source.contains(":=")) {
        return;
    }
    let mut names = Vec::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "lambda" | "function_definition" | "class_definition" | "block" => {}
            "named_expression" => {
                if let Some(name) = n.child_by_field_name("name") {
                    bound_names(name, text, &mut names);
                }
                if let Some(value) = n.child_by_field_name("value") {
                    stack.push(value);
                }
            }
            _ => {
                let mut cursor = n.walk();
                stack.extend(n.named_children(&mut cursor));
            }
        }
    }
    bind(out, byte, names);
}

/// The names an `import` or `from ... import` statement binds.
fn imported_names(node: Node, text: &str, out: &mut Vec<String>) {
    let mut cursor = node.walk();
    for item in node.children_by_field_name("name", &mut cursor) {
        match item.kind() {
            // `import a.b.c` binds `a`.
            "dotted_name" => {
                if let Some(first) = item.named_child(0) {
                    bound_names(first, text, out);
                }
            }
            "aliased_import" => {
                if let Some(alias) = item.child_by_field_name("alias") {
                    bound_names(alias, text, out);
                }
            }
            _ => {}
        }
    }
}

/// The plain names a `del` statement unbinds. `del a[0]` and `del a.b` keep
/// `a` bound.
fn deleted_names(node: Node, text: &str, out: &mut Vec<String>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "identifier" => bound_names(n, text, out),
            "delete_statement" | "expression_list" | "tuple" | "list" | "parenthesized_expression" => {
                let mut cursor = n.walk();
                stack.extend(n.named_children(&mut cursor));
            }
            _ => {}
        }
    }
}

fn has_child_of_kind(node: Node, kind: &str) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).any(|c| c.kind() == kind)
}

/// Names a pattern binds. An attribute or a subscript binds no name.
fn bound_names(node: Node, text: &str, out: &mut Vec<String>) {
    match node.kind() {
        "identifier" => {
            if let Ok(name) = node.utf8_text(text.as_bytes()) {
                out.push(name.to_string());
            }
        }
        "pattern_list"
        | "tuple_pattern"
        | "list_pattern"
        | "list_splat_pattern"
        | "parenthesized_pattern"
        | "as_pattern_target"
        | "tuple"
        | "list"
        | "list_splat" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                bound_names(child, text, out);
            }
        }
        _ => {}
    }
}

/// The names that `... as <name>` binds in the clause or item `node`.
fn alias_names(node: Node, text: &str, out: &mut Vec<String>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "as_pattern" => {
                if let Some(alias) = child.child_by_field_name("alias") {
                    bound_names(alias, text, out);
                }
            }
            "with_item" => alias_names(child, text, out),
            _ => {}
        }
    }
}

fn bind<'t>(out: &mut Vec<Marked<'t>>, byte: usize, names: Vec<String>) {
    if !names.is_empty() {
        push(out, byte, Event::Bind(names));
    }
}

fn visit_if<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let exhaustive = has_child_of_kind(node, "else_clause");
    push(
        out,
        node.start_byte(),
        Event::Open(Kind::Branch { exhaustive }, node),
    );
    if let Some(condition) = node.child_by_field_name("condition") {
        walrus(condition, src.text, condition.start_byte(), out);
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "block" => visit(child, src, out),
            "elif_clause" | "else_clause" => {
                push(out, child.start_byte(), Event::Arm(Role::Next));
                visit(child, src, out);
            }
            _ => {}
        }
    }
    push(out, node.end_byte(), Event::Close);
}

fn visit_loop<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let text = src.text;
    let has_else = has_child_of_kind(node, "else_clause");
    // What a `for` loop iterates over runs once, before the loop.
    if let Some(iterated) = node.child_by_field_name("right") {
        walrus(iterated, text, iterated.start_byte(), out);
    }
    push(
        out,
        node.start_byte(),
        Event::Open(Kind::Loop { has_else }, node),
    );
    if let Some(condition) = node.child_by_field_name("condition") {
        walrus(condition, text, condition.start_byte(), out);
    }
    if let Some(target) = node.child_by_field_name("left") {
        let mut names = Vec::new();
        bound_names(target, text, &mut names);
        bind(out, node.start_byte(), names);
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "block" => visit(child, src, out),
            "else_clause" => {
                push(out, child.start_byte(), Event::Arm(Role::Else));
                visit(child, src, out);
            }
            _ => {}
        }
    }
    push(out, node.end_byte(), Event::Close);
}

fn visit_try<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let text = src.text;
    let has_else = has_child_of_kind(node, "else_clause");
    let mut finally_writes = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "finally_clause" {
            for (lhs, _, _) in collect_assignment_items(child, text).unwrap_or_default() {
                finally_writes.extend(written_names(&lhs).into_iter().map(str::to_string));
            }
        }
    }
    push(
        out,
        node.start_byte(),
        Event::Open(
            Kind::Try {
                has_else,
                finally_writes,
            },
            node,
        ),
    );
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "block" => visit(child, src, out),
            "except_clause" | "except_group_clause" => {
                push(out, child.start_byte(), Event::Arm(Role::Handler));
                let mut names = Vec::new();
                alias_names(child, text, &mut names);
                bind(out, child.start_byte(), names);
                visit(child, src, out);
            }
            "else_clause" => {
                push(out, child.start_byte(), Event::Arm(Role::Else));
                visit(child, src, out);
            }
            "finally_clause" => {
                push(out, child.start_byte(), Event::Arm(Role::Finally));
                visit(child, src, out);
            }
            _ => {}
        }
    }
    push(out, node.end_byte(), Event::Close);
}

/// Whether a `case` takes every subject: `case _:` or a bare capture, with no
/// guard.
fn irrefutable(case: Node) -> bool {
    if case.child_by_field_name("guard").is_some() {
        return false;
    }
    let mut cursor = case.walk();
    let patterns: Vec<Node> = case
        .named_children(&mut cursor)
        .filter(|c| c.kind() == "case_pattern")
        .collect();
    let [pattern] = patterns.as_slice() else {
        return false;
    };
    match pattern.named_child_count() {
        0 => true,
        1 => pattern
            .named_child(0)
            .is_some_and(|c| c.kind() == "dotted_name" && c.named_child_count() == 1),
        _ => false,
    }
}

/// The names a pattern of a `case` captures. A bare name captures, a dotted
/// name is a value. In `Point(x=w)` the keyword `x` and the class `Point` are
/// not captures.
fn captured_names(pattern: Node, text: &str, out: &mut Vec<String>) {
    match pattern.kind() {
        "dotted_name" => {
            if pattern.named_child_count() == 1
                && let Some(name) = pattern.named_child(0)
            {
                bound_names(name, text, out);
            }
        }
        // `case ... as w` and `*rest` or `**rest`: the identifier captures.
        "identifier" => bound_names(pattern, text, out),
        "keyword_pattern" => {
            if let Some(value) = pattern.named_child(1) {
                captured_names(value, text, out);
            }
        }
        "class_pattern" => {
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor).skip(1) {
                captured_names(child, text, out);
            }
        }
        _ => {
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor) {
                captured_names(child, text, out);
            }
        }
    }
}

fn visit_match<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let Some(body) = node.child_by_field_name("body") else {
        return;
    };
    let mut cursor = body.walk();
    let cases: Vec<Node> = body
        .named_children(&mut cursor)
        .filter(|c| c.kind() == "case_clause")
        .collect();
    let exhaustive = cases.iter().any(|c| irrefutable(*c));
    if let Some(subject) = node.child_by_field_name("subject") {
        walrus(subject, src.text, subject.start_byte(), out);
    }
    // A pattern that fails may have bound a name before it failed, so every
    // case starts with the names of all the patterns unknown.
    let mut captured = Vec::new();
    for case in &cases {
        let mut inner = case.walk();
        for pattern in case
            .named_children(&mut inner)
            .filter(|c| c.kind() == "case_pattern")
        {
            captured_names(pattern, src.text, &mut captured);
        }
    }
    captured.sort_unstable();
    captured.dedup();
    push(
        out,
        node.start_byte(),
        Event::Open(Kind::Branch { exhaustive }, node),
    );
    for (i, case) in cases.iter().enumerate() {
        if i > 0 {
            push(out, case.start_byte(), Event::Arm(Role::Next));
        }
        bind(out, case.start_byte(), captured.clone());
        visit(*case, src, out);
    }
    push(out, node.end_byte(), Event::Close);
}

/// Whether the callee `target` is a context manager that may swallow an
/// exception, by the name as written or by the name it is imported from.
fn is_swallowing(target: &str, import_map: &HashMap<String, ImportPath>) -> bool {
    let named = |name: &str| SWALLOWING.contains(&name.rsplit('.').next().unwrap_or(name));
    named(target)
        || resolve_call_target(target, import_map)
            .parts
            .last()
            .is_some_and(|last| named(last))
}

fn swallows(node: Node, src: &Source) -> bool {
    let mut cursor = node.walk();
    let mut stack: Vec<Node> = node
        .named_children(&mut cursor)
        .filter(|c| c.kind() == "with_clause")
        .collect();
    while let Some(n) = stack.pop() {
        match n.kind() {
            "call" => {
                let callee = n
                    .child_by_field_name("function")
                    .and_then(|f| f.utf8_text(src.text.as_bytes()).ok())
                    .unwrap_or("");
                if is_swallowing(callee, src.imports.at(n.start_byte())) {
                    return true;
                }
            }
            _ => {
                let mut inner = n.walk();
                stack.extend(n.named_children(&mut inner));
            }
        }
    }
    false
}

fn visit_with<'t>(node: Node<'t>, src: &Source, out: &mut Vec<Marked<'t>>) {
    let mut names = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "with_clause" {
            walrus(child, src.text, node.start_byte(), out);
            alias_names(child, src.text, &mut names);
        }
    }
    bind(out, node.start_byte(), names);
    let body = node.child_by_field_name("body");
    if swallows(node, src) {
        push(out, node.start_byte(), Event::Open(Kind::Suppress, node));
        if let Some(body) = body {
            visit(body, src, out);
        }
        push(out, node.end_byte(), Event::Close);
    } else if let Some(body) = body {
        visit(body, src, out);
    }
}

fn written_names(lhs: &Lhs) -> Vec<&str> {
    match lhs {
        Lhs::Single(name) | Lhs::Augmented(name) => vec![name],
        Lhs::Tuple(targets) => targets.iter().flat_map(TupleTarget::names).collect(),
    }
}

/// What a scope holds for one name.
#[derive(Clone, PartialEq, Debug)]
struct Binding {
    shape: Option<Vec<String>>,
    carry: bool,
    dim: Option<String>,
}

impl Binding {
    fn same_as(&self, other: &Binding) -> bool {
        self == other
            || (self.carry == other.carry
                && self.dim == other.dim
                && matches!((&self.shape, &other.shape),
                    (Some(a), Some(b)) if shapes_canonically_equal(a, b)))
    }

    fn unknown() -> Self {
        Binding {
            shape: None,
            carry: false,
            dim: None,
        }
    }
}

impl ShapeCtx<'_> {
    fn binding(&self, scope: usize, name: &str) -> Binding {
        let key = (scope, name.to_string());
        Binding {
            shape: self.scopes[scope].shapes.get(name).cloned(),
            carry: self.tuple_carries.contains(&key),
            dim: self.scalar_dims.get(&key).cloned(),
        }
    }

    fn put_binding(&mut self, scope: usize, name: &str, binding: &Binding) {
        let key = (scope, name.to_string());
        match &binding.shape {
            Some(shape) => {
                self.scopes[scope]
                    .shapes
                    .insert(name.to_string(), shape.clone());
            }
            None => {
                self.scopes[scope].shapes.remove(name);
            }
        }
        if binding.carry {
            self.tuple_carries.insert(key.clone());
        } else if !self.tuple_carries.is_empty() {
            self.tuple_carries.remove(&key);
        }
        match &binding.dim {
            Some(dim) => {
                self.scalar_dims.insert(key, dim.clone());
            }
            None => {
                self.scalar_dims.remove(&key);
            }
        }
    }
}

/// The bindings of the names a statement wrote, and whether the path wrote
/// each one. A name that is not here still has its binding from the entry.
type Exit = HashMap<String, (Binding, bool)>;

#[derive(Default)]
struct LoopState {
    /// States at the end of an iteration: the end of the body and each
    /// `continue`.
    backs: Vec<Exit>,
    breaks: Vec<Exit>,
}

#[derive(Default)]
struct TryState {
    /// The state when the body ended, and whether it ended dead.
    body: Option<(Exit, bool)>,
    /// How many names the body touched.
    body_len: usize,
    /// The join of the paths that reach the `finally` clause normally.
    normal: Vec<(String, Binding)>,
    normal_live: bool,
    in_finally: bool,
}

struct Open<'t> {
    kind: Kind,
    node: Node<'t>,
    scope: usize,
    start: usize,
    arm: usize,
    /// The current path ended in a jump.
    dead: bool,
    /// Binding of each name written in the statement, as it was before the
    /// first write.
    entry: HashMap<String, Binding>,
    order: Vec<String>,
    /// How many inlay hints the pass had made when the statement began.
    hints_from: usize,
    /// Names written on the current path.
    written: HashSet<String>,
    live: Vec<Exit>,
    dead_exits: Vec<Exit>,
    looping: LoopState,
    trying: TryState,
}

impl Open<'_> {
    fn touch(&mut self, name: &str, ctx: &ShapeCtx) {
        if !self.entry.contains_key(name) {
            self.entry
                .insert(name.to_string(), ctx.binding(self.scope, name));
            self.order.push(name.to_string());
        }
    }

    fn snapshot(&self, ctx: &ShapeCtx) -> Exit {
        self.order
            .iter()
            .map(|n| {
                (
                    n.clone(),
                    (ctx.binding(self.scope, n), self.written.contains(n)),
                )
            })
            .collect()
    }

    fn restore(&self, ctx: &mut ShapeCtx) {
        for name in &self.order {
            ctx.put_binding(self.scope, name, &self.entry[name]);
        }
    }

    fn end_path(&mut self, ctx: &ShapeCtx) {
        let exit = self.snapshot(ctx);
        if self.dead {
            self.dead_exits.push(exit);
        } else {
            self.live.push(exit);
        }
    }

    /// Start the next arm from the entry state.
    fn next_arm(&mut self, ctx: &mut ShapeCtx) {
        self.end_path(ctx);
        self.restore(ctx);
        self.written.clear();
        self.dead = false;
        self.arm += 1;
    }
}

/// Where a scope first mentions each name.
struct Mentions {
    first: HashMap<String, usize>,
    /// The first `from m import *`.
    wildcard: Option<usize>,
}

/// An identifier that the top-level name model (`incremental/names.rs`)
/// links to the statements that bind the same module name. It leaves out the
/// names it skips and the ones that are not module names (a keyword, a class
/// name, an attribute).
fn links_module_name(node: Node) -> bool {
    let Some(parent) = node.parent() else {
        return true;
    };
    let field = match parent.kind() {
        "attribute" => "attribute",
        "keyword_argument" | "class_definition" => "name",
        _ => return true,
    };
    parent
        .child_by_field_name(field)
        .is_none_or(|n| n.id() != node.id())
}

/// Whether the node at `cursor` is in a class body, with no function between.
fn in_class_body(cursor: &tree_sitter::TreeCursor) -> bool {
    let mut up = cursor.clone();
    while up.goto_parent() {
        match up.node().kind() {
            "class_definition" => return true,
            "function_definition" => return false,
            _ => {}
        }
    }
    false
}

/// The first byte of each identifier under `root`. The module scope leaves
/// out function bodies and lambdas, whose names are not module names, and
/// the identifiers that `links_module_name` leaves out. A `def` or a `class`
/// binds its name, but a member of a class does not bind a module name.
fn first_mentions(root: Node, text: &str, module: bool) -> Mentions {
    let mut found = Mentions {
        first: HashMap::new(),
        wildcard: None,
    };
    let mut cursor = root.walk();
    'walk: loop {
        let node = cursor.node();
        let mut descend = true;
        match node.kind() {
            "identifier" => {
                descend = false;
                if (!module || links_module_name(node))
                    && let Ok(name) = node.utf8_text(text.as_bytes())
                    && !found.first.contains_key(name)
                {
                    found.first.insert(name.to_string(), node.start_byte());
                }
            }
            "function_definition" | "class_definition" if module => {
                if !in_class_body(&cursor)
                    && let Some(name) = node.child_by_field_name("name")
                    && let Ok(name) = name.utf8_text(text.as_bytes())
                {
                    found
                        .first
                        .entry(name.to_string())
                        .or_insert(node.start_byte());
                }
                descend = node.kind() == "class_definition";
            }
            "lambda" if module => descend = false,
            "wildcard_import" => {
                found.wildcard.get_or_insert(node.start_byte());
            }
            _ => {}
        }
        if descend && cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                break 'walk;
            }
        }
    }
    found
}

/// A name the body reads and writes ends one iteration with a shape that
/// the next iteration starts from, so replaying the body once is only
/// right when that shape did not change (`x = f(x)`). Otherwise the
/// iteration count decides (`x, y = y, x`), and the name is unknown after
/// the loop. So is any name written from such a name.
fn unstable_names(open: &Open, backs: &[&Exit], ctx: &ShapeCtx) -> HashSet<String> {
    if open.order.is_empty() {
        return HashSet::new();
    }
    let items = collect_assignment_items(open.node, ctx.text).unwrap_or_default();
    let mut deps: Vec<(String, HashSet<String>)> = Vec::new();
    let mut reads = HashSet::new();
    for (lhs, rhs, _) in &items {
        let mut read = HashSet::new();
        insert_identifiers(*rhs, ctx.text, &mut read);
        if let Lhs::Augmented(name) = lhs {
            read.insert(name.clone());
        }
        reads.extend(read.iter().cloned());
        deps.extend(
            written_names(lhs)
                .into_iter()
                .map(|n| (n.to_string(), read.clone())),
        );
    }
    let mut unknown: HashSet<String> = HashSet::new();
    for (name, _) in &deps {
        let Some(entry) = open.entry.get(name).and_then(|b| b.shape.as_ref()) else {
            continue;
        };
        if reads.contains(name)
            && backs.iter().any(|b| {
                b.get(name)
                    .is_some_and(|(binding, _)| {
                        !binding.shape.as_ref().is_some_and(|shape| shapes_canonically_equal(shape, entry))
                    })
            })
        {
            unknown.insert(name.clone());
        }
    }
    loop {
        let before = unknown.len();
        for (name, read) in &deps {
            if !read.is_disjoint(&unknown) {
                unknown.insert(name.clone());
            }
        }
        if unknown.len() == before {
            break;
        }
    }
    unknown
}

/// What a loop leaves unknown (see `unstable_names`): those names have no
/// shape from here on. The hints of the body that read the first-iteration
/// shape of such a name are taken back (`stale_hints`), because the later
/// iterations start from another shape. `retract` is false once the `else`
/// clause took them back, as that clause comes after the body.
fn settle_loop(open: &Open, backs: &[&Exit], retract: bool, ctx: &mut ShapeCtx) {
    let unknown = unstable_names(open, backs, ctx);
    if unknown.is_empty() {
        return;
    }
    for name in &unknown {
        ctx.put_binding(open.scope, name, &Binding::unknown());
    }
    if retract {
        let stale = stale_hints(open, unknown, ctx);
        let from = open.hints_from;
        let mut index = 0;
        ctx.assignment_shapes.retain(|hint| {
            index += 1;
            index <= from || !stale.get(&hint.line).is_some_and(|names| names.contains(&hint.name))
        });
    }
}

/// The names, by row, whose hint in the body of a loop shows a shape that
/// only the first iteration gives. `tainted` holds the names that may still
/// have the value of the iteration before: the unstable ones, until a
/// statement of the body (not of a branch in it) writes them from values of
/// this iteration. A statement that reads one of them, and so what it writes,
/// has a hint that only the first iteration gives.
fn stale_hints(open: &Open, mut tainted: HashSet<String>, ctx: &ShapeCtx) -> HashMap<u32, HashSet<String>> {
    let mut stale: HashMap<u32, HashSet<String>> = HashMap::new();
    let items = collect_assignment_items(open.node, ctx.text).unwrap_or_default();
    for (lhs, rhs, assignment) in &items {
        if ctx.links.scope_index.lookup(rhs.start_byte()) != Some(open.scope) {
            continue;
        }
        let mut read = HashSet::new();
        insert_identifiers(*rhs, ctx.text, &mut read);
        if let Lhs::Augmented(name) = lhs {
            read.insert(name.clone());
        }
        let written = written_names(lhs);
        if !read.is_disjoint(&tainted) {
            let row = assignment.start_position().row as u32;
            stale.entry(row).or_default().extend(written.iter().map(|n| n.to_string()));
            tainted.extend(written.iter().map(|n| n.to_string()));
        } else if assignment
            .parent()
            .and_then(|statement| statement.parent())
            .and_then(|block| block.parent())
            .is_some_and(|owner| owner.id() == open.node.id())
        {
            for name in written {
                tainted.remove(name);
            }
        }
    }
    stale
}

pub(super) struct Replay<'t> {
    events: Vec<Marked<'t>>,
    next: usize,
    stack: Vec<Open<'t>>,
    mentions: HashMap<usize, Mentions>,
    /// Per loop (a node id): the byte where each name is bound last in it.
    binders: HashMap<usize, HashMap<String, usize>>,
}

impl<'t> Replay<'t> {
    pub(super) fn new(events: Vec<Marked<'t>>) -> Self {
        Replay {
            events,
            next: 0,
            stack: Vec::new(),
            mentions: HashMap::new(),
            binders: HashMap::new(),
        }
    }

    /// Run every event that starts or ends at or before `upto`.
    pub(super) fn advance(&mut self, upto: usize, ctx: &mut ShapeCtx) {
        while self.next < self.events.len() && self.events[self.next].byte <= upto {
            let byte = self.events[self.next].byte;
            let event = std::mem::replace(&mut self.events[self.next].event, Event::Done);
            self.next += 1;
            self.apply(byte, event, ctx);
        }
    }

    pub(super) fn finish(mut self, ctx: &mut ShapeCtx) {
        self.advance(usize::MAX, ctx);
    }

    /// Note, before an assignment runs, the names it writes.
    pub(super) fn wrote(&mut self, scope: usize, lhs: &Lhs, ctx: &ShapeCtx) {
        if self.stack.is_empty() {
            return;
        }
        self.write_names(scope, &written_names(lhs), ctx);
    }

    fn write_names(&mut self, scope: usize, names: &[&str], ctx: &ShapeCtx) {
        for open in self.stack.iter_mut().filter(|o| o.scope == scope) {
            for name in names {
                open.touch(name, ctx);
                open.written.insert(name.to_string());
            }
        }
    }

    fn apply(&mut self, byte: usize, event: Event<'t>, ctx: &mut ShapeCtx) {
        match event {
            Event::Open(kind, node) => {
                // A loop over the modules of a sequence that applies each one
                // to `x` is one statement, not a loop (see `modules`).
                let straight = matches!(kind, Kind::Loop { .. }) && ctx.chain_loop(node);
                let scope = ctx
                    .links
                    .scope_index
                    .lookup(byte)
                    .filter(|&s| s < ctx.scopes.len() && !straight)
                    .unwrap_or(usize::MAX);
                self.stack.push(Open {
                    kind,
                    node,
                    scope,
                    start: byte,
                    arm: 0,
                    dead: false,
                    entry: HashMap::new(),
                    order: Vec::new(),
                    hints_from: ctx.assignment_shapes.len(),
                    written: HashSet::new(),
                    live: Vec::new(),
                    dead_exits: Vec::new(),
                    looping: LoopState::default(),
                    trying: TryState::default(),
                });
            }
            Event::Arm(role) => {
                if let Some(open) = self.stack.pop() {
                    let open = self.start_arm(open, role, ctx);
                    self.stack.push(open);
                }
            }
            Event::Close => {
                if let Some(open) = self.stack.pop() {
                    self.close(open, ctx);
                }
            }
            Event::Jump(jump) => self.jump(jump, byte, ctx),
            Event::Bind(names) => {
                let Some(scope) = ctx
                    .links
                    .scope_index
                    .lookup(byte)
                    .filter(|&s| s < ctx.scopes.len())
                else {
                    return;
                };
                let names: Vec<&str> = names.iter().map(String::as_str).collect();
                self.write_names(scope, &names, ctx);
                for name in names {
                    ctx.evict_binding(scope, name);
                }
            }
            Event::Done => {}
        }
    }

    fn start_arm(&mut self, mut open: Open<'t>, role: Role, ctx: &mut ShapeCtx) -> Open<'t> {
        if open.scope == usize::MAX {
            open.arm += 1;
            return open;
        }
        match open.kind {
            Kind::Branch { .. } => open.next_arm(ctx),
            Kind::Loop { .. } => {
                // The `else` clause runs after the loop ends without a `break`:
                // from the entry (no iteration) or from the end of an iteration.
                if !open.dead {
                    let end = open.snapshot(ctx);
                    open.looping.backs.push(end);
                }
                let entry = Exit::new();
                let mut paths: Vec<&Exit> = vec![&entry];
                paths.extend(open.looping.backs.iter());
                let results = self.join(&open, &paths, &[], ctx);
                let written: HashSet<String> = results
                    .iter()
                    .filter(|(_, _, w)| *w)
                    .map(|(n, _, _)| n.clone())
                    .collect();
                for (name, binding, _) in &results {
                    ctx.put_binding(open.scope, name, binding);
                }
                // The `else` clause follows the last iteration, which starts
                // from the state of the one before.
                let backs: Vec<&Exit> = open.looping.backs.iter().collect();
                settle_loop(&open, &backs, true, ctx);
                open.written = written;
                open.dead = false;
                open.arm += 1;
            }
            Kind::Try { has_else, .. } => {
                let mut trying = std::mem::take(&mut open.trying);
                if open.arm == 0 {
                    let end = open.snapshot(ctx);
                    trying.body_len = open.order.len();
                    // With an `else` clause the normal path goes on in it.
                    if !has_else {
                        if open.dead {
                            open.dead_exits.push(end.clone());
                        } else {
                            open.live.push(end.clone());
                        }
                    }
                    trying.body = Some((end, open.dead));
                } else {
                    open.end_path(ctx);
                }
                match role {
                    Role::Handler => {
                        // An exception can come at any point of the body.
                        for (i, name) in open.order.iter().enumerate() {
                            if i < trying.body_len {
                                ctx.put_binding(open.scope, name, &Binding::unknown());
                            } else {
                                ctx.put_binding(open.scope, name, &open.entry[name]);
                            }
                        }
                        open.written = open.order[..trying.body_len].iter().cloned().collect();
                        open.dead = false;
                    }
                    Role::Else => {
                        let (end, dead) = trying.body.clone().unwrap_or_default();
                        for name in &open.order {
                            let binding = end.get(name).map_or(&open.entry[name], |(b, _)| b);
                            ctx.put_binding(open.scope, name, binding);
                        }
                        open.written = end
                            .iter()
                            .filter(|(_, (_, w))| *w)
                            .map(|(n, _)| n.clone())
                            .collect();
                        open.dead = dead;
                    }
                    _ => {
                        let live: Vec<&Exit> = open.live.iter().collect();
                        let dead: Vec<&Exit> = open.dead_exits.iter().collect();
                        trying.normal = self
                            .join(&open, &live, &dead, ctx)
                            .into_iter()
                            .map(|(n, b, _)| (n, b))
                            .collect();
                        trying.normal_live = !open.live.is_empty();
                        // Any path can reach the clause: the body may stop
                        // anywhere, and a handler may too.
                        for name in &open.order {
                            ctx.put_binding(open.scope, name, &Binding::unknown());
                        }
                        trying.in_finally = true;
                        open.written.clear();
                        open.dead = false;
                    }
                }
                open.trying = trying;
                open.arm += 1;
            }
            Kind::Suppress => open.arm += 1,
        }
        open
    }

    fn close(&mut self, mut open: Open<'t>, ctx: &mut ShapeCtx) {
        if open.scope == usize::MAX {
            return;
        }
        let scope = open.scope;
        let live_left = match open.kind {
            Kind::Branch { exhaustive } => {
                open.end_path(ctx);
                if !exhaustive {
                    open.live.push(Exit::new());
                }
                let live: Vec<&Exit> = open.live.iter().collect();
                let dead: Vec<&Exit> = open.dead_exits.iter().collect();
                self.apply_join(&open, &live, &dead, ctx);
                !open.live.is_empty()
            }
            Kind::Loop { has_else } => self.close_loop(&mut open, has_else, ctx),
            Kind::Try { .. } => {
                let trying = std::mem::take(&mut open.trying);
                if trying.in_finally {
                    for (name, binding) in &trying.normal {
                        if !open.written.contains(name) {
                            ctx.put_binding(scope, name, binding);
                        }
                    }
                    trying.normal_live && !open.dead
                } else {
                    open.end_path(ctx);
                    let live: Vec<&Exit> = open.live.iter().collect();
                    let dead: Vec<&Exit> = open.dead_exits.iter().collect();
                    self.apply_join(&open, &live, &dead, ctx);
                    !open.live.is_empty()
                }
            }
            Kind::Suppress => {
                for name in &open.order {
                    ctx.put_binding(scope, name, &Binding::unknown());
                }
                true
            }
        };
        if !live_left
            && let Some(parent) = self.stack.last_mut()
            && parent.scope == scope
        {
            parent.dead = true;
        }
    }

    fn apply_join(&mut self, open: &Open, live: &[&Exit], dead: &[&Exit], ctx: &mut ShapeCtx) {
        for (name, binding, _) in self.join(open, live, dead, ctx) {
            ctx.put_binding(open.scope, &name, &binding);
        }
    }

    /// The state after a loop: it ran no iteration, or it left at the end of
    /// one, or a `break` left it. Only the first two reach an `else` clause.
    fn close_loop(&mut self, open: &mut Open<'t>, has_else: bool, ctx: &mut ShapeCtx) -> bool {
        let end = (!open.dead).then(|| open.snapshot(ctx));
        let entry = Exit::new();
        let mut paths: Vec<&Exit> = Vec::new();
        if has_else {
            paths.extend(end.iter());
        } else {
            paths.push(&entry);
            paths.extend(open.looping.backs.iter());
            paths.extend(end.iter());
        }
        paths.extend(open.looping.breaks.iter());
        let reachable = !paths.is_empty();
        self.apply_join(open, &paths, &[], ctx);
        let mut backs: Vec<&Exit> = open.looping.backs.iter().collect();
        if !has_else {
            backs.extend(end.iter());
        }
        settle_loop(open, &backs, !has_else, ctx);
        reachable
    }

    /// The binding of each written name after the paths in `live` meet: the
    /// shape they all give it, else unknown. A path that does not write a new
    /// name (see `is_new`) has no say. When no live path is left to decide, the
    /// paths in `dead` (that jumped away) decide, because the code in their
    /// arms is still read against the state after the statement.
    fn join(
        &mut self,
        open: &Open,
        live: &[&Exit],
        dead: &[&Exit],
        ctx: &ShapeCtx,
    ) -> Vec<(String, Binding, bool)> {
        let mut results = Vec::with_capacity(open.order.len());
        for name in &open.order {
            let entry = &open.entry[name];
            let unwritten = |e: &&Exit| !e.get(name).is_some_and(|(_, w)| *w);
            let new_name = entry.shape.is_none()
                && live.iter().chain(dead).any(unwritten)
                && self.is_new(open, name, ctx);
            let pick = |paths: &[&Exit]| -> Option<Binding> {
                let mut chosen: Option<Binding> = None;
                let mut conflict = false;
                for exit in paths {
                    let (binding, written) =
                        exit.get(name).map_or((entry, false), |(b, w)| (b, *w));
                    if !written && new_name {
                        continue;
                    }
                    match &chosen {
                        None => chosen = Some(binding.clone()),
                        Some(seen) if !seen.same_as(binding) => conflict = true,
                        Some(_) => {}
                    }
                }
                chosen.map(|c| if conflict { Binding::unknown() } else { c })
            };
            let binding = pick(live)
                .or_else(|| pick(dead))
                .unwrap_or_else(|| entry.clone());
            let written = live
                .iter()
                .chain(dead)
                .any(|e| e.get(name).is_some_and(|(_, w)| *w));
            results.push((name.clone(), binding, written));
        }
        results
    }

    /// Whether `name` is new at the statement: nothing before it can have
    /// bound the name, so a path that does not write it has no value for it
    /// (a use there is an error) and the other paths decide.
    ///
    /// In a function this is a name the function does not mention before the
    /// statement. At module level it is a name that no statement before the
    /// statement mentions as a module name, that no `global` statement of the
    /// file names, and that no earlier wildcard import may bind. These facts
    /// come from statements that share the name with this one, or from the
    /// file-wide digest of `global` names and wildcard imports (see
    /// `incremental::names`), so a replayed statement agrees with a fresh run.
    fn is_new(&mut self, open: &Open, name: &str, ctx: &ShapeCtx) -> bool {
        let module = open.scope == 0;
        let root = if module {
            let mut root = open.node;
            while let Some(parent) = root.parent() {
                root = parent;
            }
            root
        } else {
            match ctx.scope_function_nodes.get(open.scope) {
                Some(Some(func)) => *func,
                _ => return false,
            }
        };
        if module {
            let globals = ctx.global_names.get_or_init(|| {
                let mut names = HashSet::new();
                if ctx.text.contains("global") {
                    collect_global_names(root, ctx.text, &mut names);
                }
                names
            });
            if globals.contains(name) {
                return false;
            }
        }
        let mentions = self
            .mentions
            .entry(open.scope)
            .or_insert_with(|| first_mentions(root, ctx.text, module));
        mentions.wildcard.is_none_or(|at| at >= open.start)
            && mentions.first.get(name).is_none_or(|&at| at >= open.start)
            && !self.fed_back(open, name, ctx)
    }

    /// Whether a loop around the statement binds `name` after the statement.
    /// The next iteration then brings that value to the paths of the statement
    /// that do not write the name.
    fn fed_back(&mut self, open: &Open, name: &str, ctx: &ShapeCtx) -> bool {
        let end = open.node.end_byte();
        for ancestor in &self.stack {
            if ancestor.scope != open.scope
                || ancestor.arm != 0
                || !matches!(ancestor.kind, Kind::Loop { .. })
            {
                continue;
            }
            let binders = self
                .binders
                .entry(ancestor.node.id())
                .or_insert_with(|| binders(ancestor.node, ctx.text, true));
            if binders.get(name).is_some_and(|&at| at >= end) {
                return true;
            }
        }
        false
    }

    fn jump(&mut self, jump: Jump, byte: usize, ctx: &mut ShapeCtx) {
        let Some(scope) = ctx.links.scope_index.lookup(byte) else {
            return;
        };
        if matches!(jump, Jump::Break | Jump::Continue) {
            self.leave_loop(jump, scope, ctx);
        }
        if let Some(top) = self.stack.last_mut()
            && top.scope == scope
        {
            top.dead = true;
        }
    }

    /// Record the state at a `break` or `continue` for the loop it leaves.
    fn leave_loop(&mut self, jump: Jump, scope: usize, ctx: &ShapeCtx) {
        let mut poison: Vec<String> = Vec::new();
        let mut target = None;
        for i in (0..self.stack.len()).rev() {
            let open = &self.stack[i];
            if open.scope != scope {
                break;
            }
            match &open.kind {
                Kind::Loop { .. } if open.arm == 0 => {
                    target = Some(i);
                    break;
                }
                // The `finally` clause runs on the way out and may rewrite.
                Kind::Try { finally_writes, .. } if !open.trying.in_finally => {
                    poison.extend(finally_writes.iter().cloned());
                }
                _ => {}
            }
        }
        let Some(i) = target else {
            return;
        };
        let open = &mut self.stack[i];
        for name in &poison {
            open.touch(name, ctx);
        }
        let mut exit = open.snapshot(ctx);
        for name in poison {
            exit.insert(name, (Binding::unknown(), true));
        }
        if jump == Jump::Break {
            open.looping.breaks.push(exit);
        } else {
            open.looping.backs.push(exit);
        }
    }
}
