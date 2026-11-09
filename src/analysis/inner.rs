//! Shapes through the scope of a lambda or a comprehension.
//!
//! Both bind names that live only while the body runs. The body is evaluated
//! where it stands, with those names set in the scope around it (`Frame`), and
//! every name is put back afterwards, so nothing leaks: not a binding, not a
//! synthetic name, not an error that depends on a value that changes from one
//! call or iteration to the next.
//!
//! A body is evaluated once, with the shapes of one call or one iteration, so
//! its result stands only when it does not depend on which one it was.

use std::collections::HashSet;

use tree_sitter::Node;

use super::{
    NameBinding, ShapeCtx, find_positional_arg_node, insert_target_names, is_array_expr,
    is_scalar_expr, is_tuple_value, nearest_binding, scalar_dim_expr, shape_of_expression,
};
use crate::resolution::resolve_call_target;
use crate::types::FREE_DIM;

/// The longest sequence a comprehension is expanded to for `stack` and its
/// family, which take their count from the number of names in the list.
const MAX_ELEMENTS: usize = 256;

/// What a name bound by a lambda or a comprehension holds.
pub(super) enum Inner {
    Array(Vec<String>),
    /// A Python number that differs between calls and iterations.
    Scalar,
    Unknown,
}

struct Undo {
    name: String,
    shape: Option<Vec<String>>,
    set: Option<Vec<String>>,
    carry: bool,
    scalar_dim: Option<String>,
    masked: bool,
}

/// The names bound for the body of one lambda or comprehension.
pub(super) struct Frame {
    scope: usize,
    owner: usize,
    undo: Vec<Undo>,
    names: Vec<String>,
    scalars: Vec<String>,
    errors: usize,
}

impl Frame {
    pub(super) fn new(ctx: &ShapeCtx, scope: usize, owner: Node) -> Self {
        Self {
            scope,
            owner: owner.id(),
            undo: Vec::new(),
            names: Vec::new(),
            scalars: Vec::new(),
            errors: ctx.errors.len(),
        }
    }

    /// Bind `name` over whatever the scope around holds for it.
    pub(super) fn bind(&mut self, ctx: &mut ShapeCtx, name: String, inner: Inner) {
        ctx.name_checks.borrow_mut().clear();
        let key = (self.scope, name.clone());
        let shape = ctx.scopes[self.scope].shapes.remove(&name);
        let carry = ctx.tuple_carries.remove(&key);
        let scalar_dim = ctx.scalar_dims.remove(&key);
        let mut set = None;
        let mut masked = false;
        match inner {
            Inner::Array(dims) => {
                if !dims.is_empty() {
                    ctx.inner_arrays.insert((self.owner, name.clone()));
                }
                ctx.scopes[self.scope]
                    .shapes
                    .insert(name.clone(), dims.clone());
                set = Some(dims);
            }
            Inner::Scalar => {
                ctx.inner_scalars.insert((self.owner, name.clone()));
                self.scalars.push(name.clone());
                ctx.inner_masked.push(key);
                masked = true;
            }
            Inner::Unknown => {
                ctx.inner_masked.push(key);
                masked = true;
            }
        }
        self.names.push(name.clone());
        self.undo.push(Undo {
            name,
            shape,
            set,
            carry,
            scalar_dim,
            masked,
        });
    }

    /// Put every name back and drop the errors that read a changing number.
    pub(super) fn restore(&mut self, ctx: &mut ShapeCtx) {
        ctx.name_checks.borrow_mut().clear();
        while let Some(undo) = self.undo.pop() {
            let key = (self.scope, undo.name.clone());
            ctx.inner_arrays.remove(&(self.owner, undo.name.clone()));
            ctx.inner_scalars.remove(&(self.owner, undo.name.clone()));
            if undo.masked {
                ctx.inner_masked.pop();
            }
            // A call into this very function may have replaced the map.
            let shapes = &mut ctx.scopes[self.scope].shapes;
            if shapes.get(&undo.name) == undo.set.as_ref() {
                match undo.shape {
                    Some(shape) => shapes.insert(undo.name.clone(), shape),
                    None => shapes.remove(&undo.name),
                };
            }
            if undo.carry {
                ctx.tuple_carries.insert(key.clone());
            }
            if let Some(dim) = undo.scalar_dim {
                ctx.scalar_dims.insert(key, dim);
            }
        }
        let mut kept = self.errors;
        for i in self.errors..ctx.errors.len() {
            if !ctx.errors[i].reads.iter().any(|name| self.scalars.contains(name)) {
                ctx.errors.swap(kept, i);
                kept += 1;
            }
        }
        ctx.errors.truncate(kept);
    }

    /// Whether no dim of `shape` is spelled with a name of the frame: such a
    /// dim stands for a value that is not the same at every call or iteration.
    pub(super) fn untainted(&self, shape: &[String]) -> bool {
        shape.iter().all(|dim| {
            !dim.split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|token| {
                    let token = token.strip_prefix(FREE_DIM).unwrap_or(token);
                    self.names.iter().any(|name| name == token)
                })
        })
    }
}

/// The node with the parentheses around it taken off.
fn unparenthesized(mut node: Node) -> Option<Node> {
    while node.kind() == "parenthesized_expression" {
        node = node.named_child(0)?;
    }
    Some(node)
}

fn is_sequence_literal(node: Node) -> bool {
    matches!(node.kind(), "tuple" | "expression_list" | "list")
}

/// Whether `node` is a call of the Python builtin `name`: not imported from a
/// module, not bound by a scope around the call.
pub(super) fn is_builtin_call(node: Node, name: &str, ctx: &ShapeCtx) -> bool {
    node.kind() == "call"
        && node.child_by_field_name("function").is_some_and(|func| {
            func.kind() == "identifier"
                && ctx.text.get(func.byte_range()) == Some(name)
                && resolve_call_target(name, ctx.imports.at(node.start_byte())).parts == [name]
                && matches!(nearest_binding(node, name, ctx).1, NameBinding::Unbound)
        })
}

/// The arguments of a call as plain nodes, or `None` for a keyword or a splat.
fn plain_arguments<'t>(call: Node<'t>) -> Option<Vec<Node<'t>>> {
    let args = call.child_by_field_name("arguments")?;
    if args.kind() != "argument_list" {
        return None;
    }
    let mut cursor = args.walk();
    let mut nodes = Vec::new();
    for arg in args.named_children(&mut cursor) {
        match arg.kind() {
            "comment" => {}
            "keyword_argument" | "list_splat" | "dictionary_splat" => return None,
            _ => nodes.push(arg),
        }
    }
    Some(nodes)
}

/// The integer a `range` argument stands for, or the length of the first axis
/// of a named array: `len(x)` and `x.shape[0]`.
fn static_integer(node: Node, scope_idx: usize, ctx: &ShapeCtx) -> Option<i64> {
    if let Some(dim) = scalar_dim_expr(node, scope_idx, ctx) {
        return dim.parse().ok();
    }
    let text = |n: Node| ctx.text.get(n.byte_range());
    let leading = |name: Node| -> Option<i64> {
        if name.kind() != "identifier" {
            return None;
        }
        ctx.resolve_shape(text(name)?, scope_idx)?
            .first()?
            .parse()
            .ok()
    };
    if is_builtin_call(node, "len", ctx) {
        let [arg] = plain_arguments(node)?[..] else {
            return None;
        };
        return if is_array_expr(arg, ctx, &mut Vec::new()) {
            leading(arg)
        } else {
            None
        };
    }
    if node.kind() != "subscript" || node.named_child_count() != 2 {
        return None;
    }
    let value = node.child_by_field_name("value")?;
    let is_shape = value.kind() == "attribute"
        && value.child_by_field_name("attribute").and_then(text) == Some("shape");
    if is_shape && text(node.named_child(1)?) == Some("0") {
        leading(value.child_by_field_name("object")?)
    } else {
        None
    }
}

/// How many items `range(...)` yields.
fn range_count(args: &[Node], scope_idx: usize, ctx: &ShapeCtx) -> Option<usize> {
    let values: Option<Vec<i64>> = args
        .iter()
        .map(|arg| static_integer(*arg, scope_idx, ctx))
        .collect();
    let (start, stop, step) = match values?[..] {
        [stop] => (0, stop, 1),
        [start, stop] => (start, stop, 1),
        [start, stop, step] if step != 0 => (start, stop, step),
        _ => return None,
    };
    let span = if step > 0 { stop.checked_sub(start)? } else { start.checked_sub(stop)? };
    let count = if span <= 0 { 0 } else { (span - 1) / step.checked_abs()? + 1 };
    usize::try_from(count).ok()
}

/// The first axis and the rows of an iterated array.
enum Iterated {
    /// A value that is not provably an array, or has no shape.
    Unknown,
    /// A rank-0 array: Python raises when it is iterated.
    Scalar,
    Array {
        lead: String,
        row: Vec<String>,
    },
}

fn rows_of(node: Node, ctx: &mut ShapeCtx) -> Iterated {
    let shape = shape_of_expression(node, ctx);
    match shape {
        Some(shape) if is_array_expr(node, ctx, &mut Vec::new()) => match shape.split_first() {
            Some((lead, row)) => Iterated::Array {
                lead: lead.clone(),
                row: row.to_vec(),
            },
            None => Iterated::Scalar,
        },
        _ => Iterated::Unknown,
    }
}

/// The names a clause binds and what each holds.
type Bindings = Vec<(String, Inner)>;

/// The names a `for` target binds, when it is one name or a flat list of names.
enum Target {
    One(String),
    Many(Vec<String>),
    /// A nested pattern: each name is bound, and none has a known value.
    Nested(Vec<String>),
}

fn target_of(left: Node, ctx: &ShapeCtx) -> Option<Target> {
    let name = |node: Node| ctx.text.get(node.byte_range()).map(str::to_string);
    match left.kind() {
        "identifier" => Some(Target::One(name(left)?)),
        "pattern_list" | "tuple_pattern" | "list_pattern" => {
            let mut cursor = left.walk();
            let items: Vec<Node> = left.named_children(&mut cursor).collect();
            if items.iter().all(|item| item.kind() == "identifier") {
                return Some(Target::Many(
                    items.into_iter().map(name).collect::<Option<_>>()?,
                ));
            }
            let mut names = HashSet::new();
            insert_target_names(left, ctx.text, &mut names);
            let all = |node: Node| {
                let mut stack = vec![node];
                while let Some(n) = stack.pop() {
                    match n.kind() {
                        "identifier"
                        | "pattern_list"
                        | "tuple_pattern"
                        | "list_pattern"
                        | "list_splat_pattern"
                        | "parenthesized_pattern" => {
                            let mut cursor = n.walk();
                            stack.extend(n.named_children(&mut cursor));
                        }
                        _ => return false,
                    }
                }
                true
            };
            all(left).then(|| Target::Nested(names.into_iter().collect()))
        }
        _ => None,
    }
}

fn unknown_names(names: Vec<String>) -> Bindings {
    names
        .into_iter()
        .map(|name| (name, Inner::Unknown))
        .collect()
}

/// The values one `for` clause binds, and how many iterations it makes. `None`
/// when Python would raise, or the clause is not understood.
fn clause_bindings(
    clause: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<(Bindings, Option<usize>)> {
    if clause.child(0).is_some_and(|first| first.kind() == "async") {
        return None;
    }
    let target = target_of(clause.child_by_field_name("left")?, ctx)?;
    let mut cursor = clause.walk();
    let mut rights = clause.children_by_field_name("right", &mut cursor);
    let right = rights.next()?;
    if rights.next().is_some() {
        return None;
    }
    let count_of = |lead: &str| lead.parse::<usize>().ok();
    let row_inner = |row: &[String]| Inner::Array(row.to_vec());

    if is_builtin_call(right, "range", ctx) {
        let args = plain_arguments(right)?;
        let count = range_count(&args, scope_idx, ctx);
        return match target {
            Target::One(name) => Some((vec![(name, Inner::Scalar)], count)),
            _ => None,
        };
    }

    if is_builtin_call(right, "enumerate", ctx) {
        let args = plain_arguments(right)?;
        let (first, start) = match args[..] {
            [first] => (first, None),
            [first, start] => (first, Some(start)),
            _ => return None,
        };
        if start.is_some_and(|start| !matches!(start.kind(), "integer")) {
            return None;
        }
        let rows = rows_of(first, ctx);
        let (count, inner) = match &rows {
            Iterated::Scalar => return None,
            Iterated::Array { lead, row } => (count_of(lead), row_inner(row)),
            Iterated::Unknown => (None, Inner::Unknown),
        };
        return Some((
            match target {
                Target::Many(names) if names.len() == 2 => {
                    let [index, item] = <[String; 2]>::try_from(names).ok()?;
                    vec![(index, Inner::Scalar), (item, inner)]
                }
                Target::One(name) => vec![(name, Inner::Unknown)],
                Target::Many(names) | Target::Nested(names) => unknown_names(names),
            },
            count,
        ));
    }

    if is_builtin_call(right, "zip", ctx) {
        let args = plain_arguments(right)?;
        let mut leads = Vec::new();
        let mut inners = Vec::new();
        for arg in &args {
            match rows_of(*arg, ctx) {
                Iterated::Scalar => return None,
                Iterated::Array { lead, row } => {
                    leads.push(Some(lead));
                    inners.push(row_inner(&row));
                }
                Iterated::Unknown => {
                    leads.push(None);
                    inners.push(Inner::Unknown);
                }
            }
        }
        let count = zip_count(&leads);
        return Some((
            match target {
                Target::Many(names) if names.len() == inners.len() => {
                    names.into_iter().zip(inners).collect()
                }
                Target::One(name) => vec![(name, Inner::Unknown)],
                Target::Many(names) | Target::Nested(names) => unknown_names(names),
            },
            count,
        ));
    }

    let rows = rows_of(right, ctx);
    let (lead, row) = match rows {
        Iterated::Scalar => return None,
        Iterated::Array { lead, row } => (lead, row),
        Iterated::Unknown => {
            let names = match target {
                Target::One(name) => vec![name],
                Target::Many(names) | Target::Nested(names) => names,
            };
            return Some((unknown_names(names), None));
        }
    };
    let count = count_of(&lead);
    Some((
        match target {
            Target::One(name) => vec![(name, row_inner(&row))],
            Target::Many(names)
                if row
                    .first()
                    .is_some_and(|dim| dim.parse() == Ok(names.len())) =>
            {
                names
                    .into_iter()
                    .map(|name| (name, Inner::Array(row[1..].to_vec())))
                    .collect()
            }
            Target::Many(names) | Target::Nested(names) => unknown_names(names),
        },
        count,
    ))
}

/// `zip` stops at the shortest argument: known when the axes agree, or are all
/// numbers.
fn zip_count(leads: &[Option<String>]) -> Option<usize> {
    let leads: Vec<&String> = leads.iter().map(Option::as_ref).collect::<Option<_>>()?;
    let numbers: Option<Vec<usize>> = leads.iter().map(|lead| lead.parse().ok()).collect();
    numbers?.into_iter().min()
}

fn contains_kind(node: Node, kind: &str) -> bool {
    node.kind() == kind
        || (0..node.named_child_count())
            .filter_map(|i| node.named_child(i as u32))
            .any(|child| contains_kind(child, kind))
}

/// The shape of one element of a list or generator comprehension, and how many
/// elements it has when that is a number from 1 through `MAX_ELEMENTS`.
pub(super) fn comprehension_elements(
    node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<(Vec<String>, Option<usize>)> {
    let body = node.child_by_field_name("body")?;
    // A walrus binds in the scope around the comprehension.
    if contains_kind(node, "named_expression") {
        return None;
    }
    let mut cursor = node.walk();
    let mut clauses = Vec::new();
    let mut filtered = false;
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "for_in_clause" => clauses.push(child),
            "if_clause" => filtered = true,
            _ => {}
        }
    }
    if clauses.is_empty() {
        return None;
    }
    let mut frame = Frame::new(ctx, scope_idx, node);
    let mut count = None;
    let element = (|| {
        for clause in &clauses {
            let (bindings, n) = clause_bindings(*clause, scope_idx, ctx)?;
            count = n;
            for (name, inner) in bindings {
                frame.bind(ctx, name, inner);
            }
        }
        shape_of_expression(body, ctx)
    })();
    frame.restore(ctx);
    let element = element.filter(|shape| frame.untainted(shape))?;
    let count = if clauses.len() == 1 && !filtered {
        count.filter(|n| (1..=MAX_ELEMENTS).contains(n))
    } else {
        None
    };
    Some((element, count))
}

/// The comprehension an argument list holds alone: `f([...])`, or `f(x for ...)`
/// where the generator is the whole argument list.
pub(super) fn sole_comprehension<'t>(args: Node<'t>) -> Option<Node<'t>> {
    let is_comprehension =
        |n: Node| matches!(n.kind(), "list_comprehension" | "generator_expression");
    if is_comprehension(args) {
        return Some(args);
    }
    let mut cursor = args.walk();
    let mut named = args
        .named_children(&mut cursor)
        .filter(|n| n.kind() != "comment");
    let only = named.next()?;
    (named.next().is_none() && is_comprehension(only)).then_some(only)
}

/// `sum(<comprehension>)`: the elements added to the integer 0, so the shape of
/// an element. Python raises on an empty sum of arrays only when the elements
/// are not numbers, but an empty sum is the integer 0, so the count must be 1
/// or more.
pub(super) fn shape_of_sum(
    call: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if !is_builtin_call(call, "sum", ctx) {
        return None;
    }
    let comprehension = sole_comprehension(call.child_by_field_name("arguments")?)?;
    let (element, count) = comprehension_elements(comprehension, scope_idx, ctx)?;
    count.map(|_| element)
}

/// The names that stand for the elements of a list comprehension in a call of
/// `stack` or its family, which count their inputs by name.
pub(super) fn comprehension_names(
    node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if node.kind() != "list_comprehension" {
        return None;
    }
    let (element, count) = comprehension_elements(node, scope_idx, ctx)?;
    let name = ctx.bind_synthetic(
        element,
        scope_idx,
        node.start_byte(),
        ctx.text
            .get(node.child_by_field_name("body")?.byte_range())
            .unwrap_or_default(),
    );
    Some(vec![name; count?])
}

/// The parameters of a lambda and the default of each, or `None` for a lambda
/// with `*args`, `**kwargs`, or a separator.
fn lambda_signature<'t>(
    lambda: Node<'t>,
    text: &str,
) -> Option<(Vec<String>, Vec<Option<Node<'t>>>)> {
    let mut names = Vec::new();
    let mut defaults = Vec::new();
    let Some(params) = lambda.child_by_field_name("parameters") else {
        return Some((names, defaults));
    };
    let mut cursor = params.walk();
    for param in params.named_children(&mut cursor) {
        match param.kind() {
            "identifier" => {
                names.push(text.get(param.byte_range())?.to_string());
                defaults.push(None);
            }
            "default_parameter" => {
                let name = param.child_by_field_name("name")?;
                names.push(text.get(name.byte_range())?.to_string());
                defaults.push(param.child_by_field_name("value"));
            }
            "comment" => {}
            _ => return None,
        }
    }
    Some((names, defaults))
}

/// What an argument holds: its shape, a number, or nothing known. A tuple
/// stands for several values, so the one shape of the evaluator is not its.
fn argument_inner(node: Node, scope_idx: usize, ctx: &mut ShapeCtx) -> Inner {
    if is_tuple_value(node, scope_idx, ctx) {
        return Inner::Unknown;
    }
    match shape_of_expression(node, ctx) {
        Some(shape) => Inner::Array(shape),
        None if is_scalar_expr(node, ctx, &mut Vec::new()) => Inner::Scalar,
        None => Inner::Unknown,
    }
}

/// The value of each parameter of a lambda for a call. A parameter the call
/// leaves out takes its default, which is a number only when it is a literal.
fn complete_arguments(
    lambda: Node,
    given: Vec<(Option<String>, Inner)>,
    ctx: &ShapeCtx,
) -> Option<Vec<(String, Inner)>> {
    let (names, defaults) = lambda_signature(lambda, ctx.text)?;
    let mut values: Vec<Option<Inner>> = names.iter().map(|_| None).collect();
    let mut position = 0;
    for (keyword, inner) in given {
        let at = match keyword {
            Some(keyword) => names.iter().position(|name| *name == keyword)?,
            None => {
                position += 1;
                position - 1
            }
        };
        let slot = values.get_mut(at)?;
        if slot.is_some() {
            return None;
        }
        *slot = Some(inner);
    }
    names
        .into_iter()
        .zip(values)
        .zip(defaults)
        .map(|((name, value), default)| {
            let inner = match (value, default) {
                (Some(inner), _) => inner,
                (None, Some(default)) if matches!(default.kind(), "integer" | "float") => {
                    Inner::Scalar
                }
                (None, Some(_)) => Inner::Unknown,
                (None, None) => return None,
            };
            Some((name, inner))
        })
        .collect()
}

/// Evaluate `eval` on the body of `lambda` with its parameters bound.
fn eval_lambda(
    lambda: Node,
    scope_idx: usize,
    params: Vec<(String, Inner)>,
    ctx: &mut ShapeCtx,
    eval: impl FnOnce(Node, &mut ShapeCtx) -> Option<Vec<String>>,
) -> Option<Vec<String>> {
    let body = lambda.child_by_field_name("body")?;
    let mut frame = Frame::new(ctx, scope_idx, lambda);
    for (name, inner) in params {
        frame.bind(ctx, name, inner);
    }
    let shape = eval(body, ctx);
    frame.restore(ctx);
    shape.filter(|shape| frame.untainted(shape))
}

/// The shape of a body that is one array: a tuple is several values.
fn single_output(body: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    if is_sequence_literal(unparenthesized(body)?) {
        return None;
    }
    shape_of_expression(body, ctx)
}

/// `(lambda x: ...)(args)`.
pub(super) fn shape_of_direct_lambda(
    lambda: Node,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let mut given = Vec::new();
    let mut cursor = args_node.walk();
    for arg in args_node.named_children(&mut cursor) {
        match arg.kind() {
            "comment" => {}
            "list_splat" | "dictionary_splat" => return None,
            "keyword_argument" => {
                let name = ctx
                    .text
                    .get(arg.child_by_field_name("name")?.byte_range())?;
                let inner = argument_inner(arg.child_by_field_name("value")?, scope_idx, ctx);
                given.push((Some(name.to_string()), inner));
            }
            _ => given.push((None, argument_inner(arg, scope_idx, ctx))),
        }
    }
    let params = complete_arguments(lambda, given, ctx)?;
    eval_lambda(lambda, scope_idx, params, ctx, single_output)
}

/// A lambda whose parameters are the per-example shapes `examples` in order,
/// and whose result is batched again by `finish`.
pub(super) fn shape_of_batched_lambda(
    lambda: Node,
    scope_idx: usize,
    examples: Vec<Vec<String>>,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let given = examples
        .into_iter()
        .map(|shape| (None, Inner::Array(shape)))
        .collect();
    let params = complete_arguments(lambda, given, ctx)?;
    eval_lambda(lambda, scope_idx, params, ctx, single_output)
}

/// Whether an argument of a mapped call is one array and not a tuple of them.
pub(super) fn is_single_array_argument(node: Node, scope_idx: usize, ctx: &ShapeCtx) -> bool {
    !matches!(
        node.kind(),
        "keyword_argument" | "list_splat" | "dictionary_splat"
    ) && !is_tuple_value(node, scope_idx, ctx)
}

/// The per-step output of `jax.lax.scan(lambda carry, x: (carry, y), init, xs)`,
/// without the leading axis.
pub(super) fn scan_lambda_output(
    lambda: Node,
    carry: Vec<String>,
    carry_is_tuple: bool,
    element: Vec<String>,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let (names, _) = lambda_signature(lambda, ctx.text)?;
    let [carry_name, element_name] = <[String; 2]>::try_from(names).ok()?;
    let carry = if carry_is_tuple {
        Inner::Unknown
    } else {
        Inner::Array(carry)
    };
    let params = vec![(carry_name, carry), (element_name, Inner::Array(element))];
    eval_lambda(lambda, scope_idx, params, ctx, |body, ctx| {
        let body = unparenthesized(body)?;
        if body.kind() != "tuple" {
            return None;
        }
        let mut cursor = body.walk();
        let items: Vec<Node> = body
            .named_children(&mut cursor)
            .filter(|n| n.kind() != "comment")
            .collect();
        let [carry_out, output] = items[..] else {
            return None;
        };
        shape_of_expression(carry_out, ctx);
        if is_sequence_literal(unparenthesized(output)?) || is_tuple_value(output, scope_idx, ctx) {
            return None;
        }
        shape_of_expression(output, ctx)
    })
}

/// The lambda an argument node is, when it is one.
pub(super) fn lambda_argument<'t>(args: Node<'t>, position: usize) -> Option<Node<'t>> {
    find_positional_arg_node(args, position).filter(|node| node.kind() == "lambda")
}

/// The lambda called by `(lambda ...)(...)`.
pub(super) fn called_lambda(func: Node) -> Option<Node> {
    let inner = unparenthesized(func)?;
    (inner.kind() == "lambda").then_some(inner)
}
