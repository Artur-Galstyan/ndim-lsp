use std::collections::HashMap;
use std::path::PathBuf;

use tree_sitter::Node;

use crate::known_functions::{
    apply_known_function, apply_method_call, classify_known_function, classify_method_call,
};
use crate::layers::{apply_layer_application, extract_layer_assignments_scoped};
use crate::python_ast::{
    build_import_map, extract_call_arguments,
    extract_jaxtyping_shapes,
};
use crate::resolution::{ResolutionCache, resolve_call_target};

use crate::types::*;

pub fn analyze_layer_shapes<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut scopes = extract_jaxtyping_shapes(node, text)?;
    let import_map = build_import_map(node, text)?;
    let layer_records =
        extract_layer_assignments_scoped(node, text, search_roots, read_file, max_depth, cache)?;

    let (applications, errors) =
        propagate_calls(node, text, &import_map, &layer_records, &mut scopes)?;

    let mut layers = HashMap::new();
    for rec in &layer_records {
        layers.insert(rec.name.clone(), rec.kind.clone());
    }

    Ok(LayerShapeAnalysis {
        scopes,
        layers,
        applications,
        errors,
    })
}

fn find_scoped_layer<'a>(
    records: &'a [LayerAssignment],
    scopes: &[FunctionShapeScope],
    call_byte: usize,
    target: &str,
) -> Option<&'a LayerKind> {
    let mut best: Option<(usize, usize)> = None;
    for (i, rec) in records.iter().enumerate() {
        if rec.name != target {
            continue;
        }
        let Some(rec_scope_idx) = scope_index_for_byte(scopes, rec.byte_position) else {
            continue;
        };
        let rec_scope = &scopes[rec_scope_idx];
        if rec_scope.start_byte <= call_byte && call_byte < rec_scope.end_byte {
            let size = rec_scope.end_byte - rec_scope.start_byte;
            match best {
                None => best = Some((i, size)),
                Some((_, prev)) if size <= prev => best = Some((i, size)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| &records[i].kind)
}

// ── Recursive shape evaluator (#34) ─────────────────────────────────────
//
// Replaces the flat query-based extraction (extract_calls /
// extract_method_calls / extract_binary_ops) with a recursive tree walk.
// For each assignment `lhs = rhs`, we call shape_of_expression(rhs) which
// recurses into nested calls, chained methods, unary wraps, parenthesised
// expressions, identifiers, and subscripts.

/// Context shared across recursive shape_of_expression calls within a single
/// propagate_calls pass.
struct ShapeCtx<'a> {
    text: &'a str,
    import_map: &'a HashMap<String, ImportPath>,
    layer_records: &'a [LayerAssignment],
    scopes: &'a mut [FunctionShapeScope],
    vmap_targets: &'a mut HashMap<String, VmapInfo>,
    applications: &'a mut Vec<LayerApplication>,
    errors: &'a mut Vec<ShapeError>,
    /// Monotonic counter for synthetic variable names used to bind inline
    /// expression shapes so that known-function / method-call helpers can
    /// look them up by name.
    synthetic_counter: usize,
    /// Side-channel map for synthetic bindings. Entries here are visible
    /// to `resolve_shape` lookups but are NOT persisted to
    /// `scopes[...].shapes`, preventing __synth_* keys from leaking
    /// into the LSP's inlay hints.
    synthetics: HashMap<(usize, String), Vec<String>>,
}

impl<'a> ShapeCtx<'a> {
    /// Insert a shape under a synthetic name and return that name.
    /// Used when an inline expression (e.g. a nested call) produces a shape
    /// but has no variable binding of its own. The helpers in
    /// `known_functions` look up arguments by name in `shapes`, so we need
    /// a name.
    fn bind_synthetic(&mut self, shape: Vec<String>, scope_idx: usize) -> String {
        let name = format!("__synth_{}", self.synthetic_counter);
        self.synthetic_counter += 1;
        self.synthetics.insert((scope_idx, name.clone()), shape);
        name
    }

    /// Look up a shape by name in the given scope, checking both real
    /// user-visible shapes and the synthetic side-channel.
    fn resolve_shape(&self, name: &str, scope_idx: usize) -> Option<Vec<String>> {
        self.scopes[scope_idx].shapes.get(name).cloned()
            .or_else(|| self.synthetics.get(&(scope_idx, name.to_string())).cloned())
    }

    /// Produce a merged shapes map for a given scope that includes both
    /// user-visible shapes and synthetic bindings. Used when passing shapes
    /// to helper functions like `apply_known_function` that accept
    /// `&HashMap<String, Vec<String>>`.
    fn merged_shapes(&self, scope_idx: usize) -> HashMap<String, Vec<String>> {
        let mut merged = self.scopes[scope_idx].shapes.clone();
        for ((si, name), shape) in &self.synthetics {
            if *si == scope_idx {
                merged.insert(name.clone(), shape.clone());
            }
        }
        merged
    }
}

/// Recursively compute the shape of an expression node.
///
/// Returns `Some(shape)` if the shape could be determined, `None` if not
/// (silently skipped — unrecognised function, missing variable shape, etc.).
/// On error (dimension mismatch etc.) records a `ShapeError` and returns `None`.
fn shape_of_expression(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    match node.kind() {
        "call" => shape_of_call(node, ctx),
        "identifier" => shape_of_identifier(node, ctx),
        "attribute" => shape_of_attribute(node, ctx),
        "unary_operator" => shape_of_unary(node, ctx),
        "parenthesized_expression" => {
            // Transparent — recurse into the inner expression.
            let inner = node.named_child(0)?;
            shape_of_expression(inner, ctx)
        }
        "subscript" => shape_of_subscript(node, ctx),
        "binary_operator" => shape_of_binary_operator(node, ctx),
        _ => None,
    }
}

/// Look up a bare identifier in the current scope's shapes.
/// Checks the user-visible scope.shapes first, then falls back to the
/// synthetic side-channel (for inline expression bindings).
fn shape_of_identifier(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let name = node.utf8_text(ctx.text.as_bytes()).ok()?;
    let scope_idx = scope_index_for_byte(ctx.scopes, node.start_byte())?;
    ctx.resolve_shape(name, scope_idx)
}

/// An `attribute` node that is *not* the `function` field of a call.
/// This can appear in chained expressions like `x.reshape(3,4).sum(axis=1)`
/// where the tree-sitter parse produces a single `call` whose `function`
/// field is an attribute chain — that case is handled in `shape_of_call`.
/// A standalone attribute (e.g. `obj.field`) doesn't have a shape rule in v1.
fn shape_of_attribute(_node: Node, _ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    None
}

/// Unary operator: propagate the operand's shape unchanged.
/// Subsumes issue #32: `-x`, `+x`, `~x` all preserve shape.
fn shape_of_unary(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    // The unary_operator node has one named child: the operand.
    let operand = node.named_child(0)?;
    shape_of_expression(operand, ctx)
}

/// Subscript: `x[0]`, `x[i:j]`, etc.  No shape rule in v1 — return None.
fn shape_of_subscript(_node: Node, _ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    None
}

/// Find the operator text between left and right children of a binary_operator node.
fn find_binary_op_text<'a>(
    node: Node<'a>,
    left_node: &Node<'a>,
    right_node: &Node<'a>,
    text: &str,
) -> Option<String> {
    for i in 0..node.child_count() {
        let child = node.child(i as u32)?;
        if !child.is_named()
            && child.start_byte() > left_node.end_byte()
            && child.end_byte() < right_node.start_byte()
            && let Ok(t) = child.utf8_text(text.as_bytes()) {
                return Some(t.trim().to_string());
            }
    }
    None
}

/// Binary operator: delegates to the existing `apply_binary_op`-style logic
/// but works on tree-sitter nodes directly rather than `BinaryOpInfo`.
fn shape_of_binary_operator(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let left_node = node.child_by_field_name("left")?;
    let right_node = node.child_by_field_name("right")?;

    // Both sides must be identifiers for v1 binary-op resolution.
    if left_node.kind() != "identifier" || right_node.kind() != "identifier" {
        return None;
    }

    let left_name = left_node.utf8_text(ctx.text.as_bytes()).ok()?.to_string();
    let right_name = right_node.utf8_text(ctx.text.as_bytes()).ok()?.to_string();

    // Determine the operator text.
    let op_text = find_binary_op_text(node, &left_node, &right_node, ctx.text)?;

    let op = match op_text.as_str() {
        "@" => BinaryOp::MatMul,
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        _ => return None,
    };

    let scope_idx = scope_index_for_byte(ctx.scopes, node.start_byte())?;
    let left_shape = ctx.resolve_shape(&left_name, scope_idx)?;
    let right_shape = ctx.resolve_shape(&right_name, scope_idx)?;

    let result = match op {
        BinaryOp::MatMul => apply_matmul_shape(&left_shape, &right_shape, &left_name, &right_name),
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
            apply_elementwise_shape(&left_shape, &right_shape, op)
        }
    };

    match result {
        Ok(Some(shape)) => Some(shape),
        Ok(None) => None,
        Err(message) => {
            // We don't have a natural variable name here — use the full
            // expression text as the "variable" in the error.
            let var_text = node.utf8_text(ctx.text.as_bytes()).unwrap_or("?").to_string();
            ctx.errors.push(ShapeError {
                variable: var_text,
                message,
                range: node.range(),
            });
            None
        }
    }
}

/// Resolve a `call` node's shape.
///
/// This is the main dispatch point that handles:
/// 1. Chained method calls (e.g. `x.reshape(3,4).sum(axis=1)`)
/// 2. Free function calls (e.g. `jnp.exp(...)`)
/// 3. Layer applications
/// 4. vmap bindings and calls
/// 5. User-defined function propagation
/// 6. Known-function resolution
fn shape_of_call(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let call_byte = node.start_byte();
    let scope_idx = scope_index_for_byte(ctx.scopes, call_byte)?;

    let func_node = node.child_by_field_name("function")?;
    let args_node = node.child_by_field_name("arguments")?;

    // ── Chained method calls ──
    // In tree-sitter Python, `x.reshape(3,4).sum(axis=1)` parses as a
    // single `call` node whose `function` is an `attribute` like
    // `x.reshape(3,4).sum`.  The attribute's `object` is a `call`
    // (`x.reshape(3,4)`) and its `attribute` is `sum`.
    //
    // We handle this by resolving the object (receiver) recursively,
    // then treating the outer call as a method call on that receiver.

    if func_node.kind() == "attribute" {
        let attr_node = func_node;
        let obj_node = attr_node.child_by_field_name("object")?;
        let method_name_node = attr_node.child_by_field_name("attribute")?;
        let method_name = method_name_node
            .utf8_text(ctx.text.as_bytes())
            .ok()?
            .to_string();

        // Distinguish between:
        //   (a) Chained method call: x.reshape(3,4).sum(axis=1)
        //       where obj_node is itself a `call` node.
        //   (b) Qualified free function: jnp.reshape(x, (5,3))
        //       where obj_node is an identifier (jnp).
        //   (c) Simple method call: x.reshape(3, 4)
        //       where obj_node is an identifier (x).
        //
        // For (a), we resolve the receiver recursively.
        // For (b), we fall through to free-function resolution.
        // For (c), we dispatch to classify_method_call.

        let is_chained = obj_node.kind() == "call";

        if is_chained {
            // (a) Chained method call: resolve the receiver (the inner
            // call), then treat the outer call as a method call on the result.
            let receiver_shape = shape_of_expression(obj_node, ctx);
            let receiver_name = if let Some(shape) = receiver_shape {
                ctx.bind_synthetic(shape, scope_idx)
            } else {
                return None;
            };

            let args = extract_call_arguments(args_node, ctx.text).ok()?;

            if let Some(known) = classify_method_call(&method_name) {
                let merged = ctx.merged_shapes(scope_idx);
                let result = apply_method_call(
                    &known,
                    &receiver_name,
                    &args,
                    &merged,
                );
                return match result {
                    Ok(Some(shape)) => Some(shape),
                    Ok(None) => None,
                    Err(message) => {
                        ctx.errors.push(ShapeError {
                            variable: method_name,
                            message,
                            range: args_node.range(),
                        });
                        None
                    }
                };
            }
            return None;
        }

        // (c) Simple method call: x.reshape(3, 4)
        //     where obj_node is an identifier that is NOT an import alias.
        if obj_node.kind() == "identifier" {
            let receiver_name = obj_node
                .utf8_text(ctx.text.as_bytes())
                .ok()?
                .to_string();

            // If the receiver is an import alias (like jnp), this is a
            // qualified free function — fall through.
            if !ctx.import_map.contains_key(&receiver_name) {
                let args = extract_call_arguments(args_node, ctx.text).ok()?;

                if let Some(known) = classify_method_call(&method_name) {
                    let merged = ctx.merged_shapes(scope_idx);
                    let result = apply_method_call(
                        &known,
                        &receiver_name,
                        &args,
                        &merged,
                    );
                    return match result {
                        Ok(Some(shape)) => Some(shape),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(ShapeError {
                                variable: receiver_name,
                                message,
                                range: args_node.range(),
                            });
                            None
                        }
                    };
                }

                // Check if receiver is a layer — layers are called like layer(x)
                // but they use identifier func_node, not attribute. Skip here.
            }
        }

        // For (b) and remaining cases: fall through to free-function
        // resolution. The `target` string includes the full qualified name.
    }

    // ── Free function call ──
    // func_node is an identifier (e.g. `my_func`) or an attribute chain
    // (e.g. `jnp.exp`, `jax.nn.softplus`).

    let target = func_node
        .utf8_text(ctx.text.as_bytes())
        .ok()?
        .to_string();

    // Resolve arguments — but first, recursively evaluate any inline
    // expression arguments (nested calls, unary ops, etc.) so they have
    // shapes in the scope.
    let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
    let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

    // 1. Layer check
    if let Some(kind) = find_scoped_layer(ctx.layer_records, ctx.scopes, call_byte, &target) {
        let Some(CallArgument::Positional { value: input }) = args.first().cloned() else {
            return None;
        };
        // Use the LHS variable name if available, or a synthetic one.
        // For now we don't have the LHS name here; the caller
        // (propagate_calls) handles the binding.
        let application = LayerApplication {
            variable: String::new(), // filled by caller
            layer: target.clone(),
            input,
            kind: kind.clone(),
            range: args_node.range(),
        };
        let merged = ctx.merged_shapes(scope_idx);
        let result = apply_layer_application(&application, &merged);
        ctx.applications.push(LayerApplication {
            variable: application.variable.clone(),
            layer: application.layer.clone(),
            input: application.input.clone(),
            kind: application.kind.clone(),
            range: application.range,
        });
        match result {
            Ok(Some(output)) => return Some(output),
            Ok(None) => return None,
            Err(message) => {
                // We'll record the error with a variable name from the caller.
                // For now use the target as placeholder — caller replaces it.
                ctx.errors.push(ShapeError {
                    variable: target.clone(),
                    message,
                    range: application.range,
                });
                return None;
            }
        }
    }

    // 2. vmap-target check
    if let Some(info) = ctx.vmap_targets.get(&target).cloned() {
        let merged = ctx.merged_shapes(scope_idx);
        let result = apply_vmap_call(
            &info,
            &args,
            &merged,
            ctx.scopes,
        );
        return match result {
            Ok(Some(shape)) => Some(shape),
            Ok(None) => None,
            Err(message) => {
                ctx.errors.push(ShapeError {
                    variable: target,
                    message,
                    range: args_node.range(),
                });
                None
            }
        };
    }

    // 3. User-defined function propagation
    if !target.contains('.')
        && let Some(result) = apply_user_function(
            &target,
            call_byte,
            &args,
            &ctx.merged_shapes(scope_idx),
            ctx.scopes,
        ) {
            return match result {
                Ok(Some(shape)) => Some(shape),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError {
                        variable: target,
                        message,
                        range: args_node.range(),
                    });
                    None
                }
            };
        }

    // 4. vmap *recording* (e.g. `vf = jax.vmap(f)`)
    let resolved = resolve_call_target(&target, ctx.import_map);
    if let Some(KnownFunction::Vmap) = classify_known_function(&resolved)
        && let Some(_info) = parse_vmap_call(&args) {
            // The caller (propagate_calls) will handle binding the LHS
            // variable name to the vmap target. For now we just signal
            // that a vmap binding was recorded by returning None (vmap
            // itself has no shape).
            // We return the info via a side channel — but since
            // propagate_calls handles the LHS binding, we use a special
            // marker. Actually, we'll handle vmap recording in
            // propagate_calls directly when we see the assignment.
            // Here we just return None — the vmap binding is done by the
            // caller which has the LHS variable name.
            return None;
        }

    // 5. Known-function check
    if let Some(known) = classify_known_function(&resolved) {
        let merged = ctx.merged_shapes(scope_idx);
        let result = apply_known_function(&known, &args, &merged);
        return match result {
            Ok(Some(shape)) => Some(shape),
            Ok(None) => None,
            Err(message) => {
                ctx.errors.push(ShapeError {
                    variable: target,
                    message,
                    range: args_node.range(),
                });
                None
            }
        };
    }

    // 6. Elementwise / activation fallback (not in classify_known_function)
    //
    // Functions like jnp.exp, jnp.sin, jax.nn.relu, jax.nn.softplus, etc.
    // are shape-preserving (they return the input array's shape unchanged).
    // They're not in `classify_known_function` to avoid bloating the enum,
    // but we handle them here with a simple shape-preserving rule.
    if is_shape_preserving_call(&resolved) {
        let Some(CallArgument::Positional { value }) = args.first() else {
            return None;
        };
        return ctx.resolve_shape(value, scope_idx);
    }

    None
}

/// Given the raw `CallArgument`s extracted from a call's argument_list node,
/// recursively evaluate any argument that is itself a non-trivial expression
/// (nested call, unary operator, etc.) and bind its shape under a synthetic
/// name so that downstream lookups in `apply_known_function` etc. can find it.
///
/// Simple identifier arguments pass through unchanged.
fn resolve_call_args(
    raw_args: Vec<CallArgument>,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    let mut resolved = Vec::with_capacity(raw_args.len());

    for (i, arg) in raw_args.into_iter().enumerate() {
        match arg {
            CallArgument::Positional { value } => {
                // Check if the argument text looks like a simple identifier.
                // This is a fast-path heuristic: strings matching
                // [A-Za-z0-9_]+ are either identifiers (which are already
                // in shapes or not) or numeric literals (which won't
                // match any shape entry and are silently skipped).
                // Complex expressions (dots, parens, operators) fall through
                // to the recursive evaluator.
                let looks_like_identifier = value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_');

                if looks_like_identifier {
                    resolved.push(CallArgument::Positional { value });
                } else {
                    // Find the corresponding named child in args_node.
                    // Positional args map to named children that aren't
                    // keyword_arguments.
                    let child_node = find_positional_arg_node(args_node, i)?;
                    let shape = shape_of_expression(child_node, ctx);
                    if let Some(shape) = shape {
                        let synth_name = ctx.bind_synthetic(shape, scope_idx);
                        resolved.push(CallArgument::Positional {
                            value: synth_name,
                        });
                    } else {
                        // Can't resolve this argument — pass through as-is.
                        // Downstream lookups will fail and return None,
                        // which is the expected fallback.
                        resolved.push(CallArgument::Positional { value });
                    }
                }
            }
            CallArgument::Keyword { name, value } => {
                // For keyword args, check if the value looks like a simple identifier.
                let looks_like_identifier = value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_');
                if looks_like_identifier {
                    resolved.push(CallArgument::Keyword { name, value });
                } else {
                    // Find the keyword argument node and evaluate its value.
                    let kw_node = find_keyword_arg_node(args_node, &name, ctx.text.as_bytes())?;
                    let value_node = kw_node.child_by_field_name("value")?;
                    let shape = shape_of_expression(value_node, ctx);
                    if let Some(shape) = shape {
                        let synth_name = ctx.bind_synthetic(shape, scope_idx);
                        resolved.push(CallArgument::Keyword {
                            name,
                            value: synth_name,
                        });
                    } else {
                        resolved.push(CallArgument::Keyword { name, value });
                    }
                }
            }
        }
    }

    Some(resolved)
}

/// Find the i-th positional argument node in an argument_list.
/// Find the Nth positional argument node in an argument_list.
///
/// Invariant: Python requires positional args to precede keyword args,
/// so `positional_index` matches the raw iteration index for positional
/// children. This function skips keyword_argument nodes to count only
/// positionals, making the index correct regardless of arg ordering.
fn find_positional_arg_node<'a>(args_node: Node<'a>, positional_index: usize) -> Option<Node<'a>> {
    let mut count = 0;
    for i in 0..args_node.named_child_count() {
        let child = args_node.named_child(i as u32)?;
        if child.kind() != "keyword_argument" {
            if count == positional_index {
                return Some(child);
            }
            count += 1;
        }
    }
    None
}

/// Find a keyword argument node by name in an argument_list.
fn find_keyword_arg_node<'a>(args_node: Node<'a>, name: &str, source_bytes: &[u8]) -> Option<Node<'a>> {
    for i in 0..args_node.named_child_count() {
        let child = args_node.named_child(i as u32)?;
        if child.kind() == "keyword_argument" {
            let name_node = child.child_by_field_name("name")?;
            let name_text = name_node.utf8_text(source_bytes).ok()?;
            if name_text == name {
                return Some(child);
            }
        }
    }
    None
}

// ── End recursive shape evaluator ────────────────────────────────────────

// ── vmap support ──────────────────────────────────────────────────────────
//
// v1 limitations (documented in code comments and PR description):
// - Only scalar-integer `in_axes` / `out_axes` are supported. Tuple,
//   PyTree, or `None` values cause `parse_vmap_call` to return None and
//   the call is silently skipped.
// - Nested vmap (`vmap(vmap(f))`) is not supported: the first positional
//   arg must be a bare identifier, so a `vmap(...)` expression won't parse
//   as a function name.
// - Cross-file wrapped functions are not supported: if `f` is imported
//   from another module, scope lookup will fail and we return Ok(None).
// - `vmap_targets` is global to the analysis pass, not per-scope. Nested-
//   scope shadowing of a vmap name is out of scope for v1.
// - `axis_size` keyword is not supported.

/// Metadata recorded when `jax.vmap(f, ...)` or `equinox.filter_vmap(f, ...)`
/// is encountered during the walk.
#[derive(Debug, Clone)]
struct VmapInfo {
    /// Name of the function being vmapped (bare identifier, e.g. "f").
    wrapped: String,
    /// `in_axes` value — default 0. Only scalar integers are honoured in v1.
    in_axes: isize,
    /// `out_axes` value — default 0. Only scalar integers are honoured in v1.
    out_axes: isize,
}

/// Parse a vmap/filter_vmap call's arguments into a `VmapInfo`.
///
/// Returns `None` if:
/// - The first positional arg is not a bare identifier (contains `.` or is
///   not simple).
/// - `in_axes` or `out_axes` is present but not a literal integer (tuple,
///   variable, `None`, etc.).
fn parse_vmap_call(args: &[CallArgument]) -> Option<VmapInfo> {
    // First positional arg = wrapped function name.
    let first_positional = args.iter().find_map(|arg| match arg {
        CallArgument::Positional { value } => Some(value.clone()),
        _ => None,
    })?;

    // Must be a bare identifier — no dots (no qualified names like
    // `module.func`), and no expressions like `lambda x: x`.
    if first_positional.contains('.') {
        return None;
    }

    // Read in_axes (default 0). Only scalar literal ints are honoured.
    let in_axes = parse_int_keyword(args, "in_axes", 0)?;
    // Read out_axes (default 0). Same rule.
    let out_axes = parse_int_keyword(args, "out_axes", 0)?;

    Some(VmapInfo {
        wrapped: first_positional,
        in_axes,
        out_axes,
    })
}

/// Extract a keyword argument as `isize`, falling back to `default`.
/// Returns `None` if the keyword is present but its value is not a
/// parseable integer (e.g. tuple, variable, None).
fn parse_int_keyword(args: &[CallArgument], name: &str, default: isize) -> Option<isize> {
    for arg in args {
        if let CallArgument::Keyword {
            name: kw_name,
            value,
        } = arg
            && kw_name == name
        {
            return value.parse::<isize>().ok();
        }
    }
    Some(default) // keyword not present → use default
}

/// Peel the batch dim from a shape at position `axis`.
/// Returns `Ok((peeled_shape, batch_dim))` or `Err(msg)`.
fn peel_batch_dim(shape: &[String], axis: isize) -> Result<(Vec<String>, String), String> {
    let len = shape.len() as isize;
    if len == 0 {
        return Err("cannot peel axis from scalar (rank-0)".to_string());
    }
    let axis = if axis < 0 { axis + len } else { axis };
    if axis < 0 || axis >= len {
        return Err(format!("axis {} out of bounds for rank {}", axis, len));
    }
    let axis = axis as usize;
    let batch_dim = shape[axis].clone();
    let mut peeled = shape.to_vec();
    peeled.remove(axis);
    Ok((peeled, batch_dim))
}

/// Insert a batch dim at position `axis` in the shape.
/// Negative axes count from the right *relative to the shape after
/// insertion* (numpy `np.insert` semantics). For v1 we clamp the
/// effective axis to `0..=len`.
///
/// The `len + axis + 1` formula for negative axes follows np.insert
/// convention: axis = -1 means "insert before the last element of
/// the result after insertion", which for a pre-insertion length of
/// `len` means position `len` (i.e. append at the end).
///
/// Example: shape = ["m"] (post-peel rank 1), axis = -1
///   → effective axis = 1 + (-1) + 1 = 1 → insert at position 1 (end of result)
///   → result = ["m", "B"]
///
/// Example: shape = ["m"] (post-peel rank 1), axis = 1
///   → effective axis = 1 (clamped to len=1) → insert at position 1 (end)
///   → result = ["m", "B"]
///
/// Example: shape = ["m", "k"] (post-peel rank 2), axis = -1
///   → effective axis = 2 + (-1) + 1 = 2 → insert at position 2 (end)
///   → result = ["m", "k", "B"]
fn prepend_batch_dim(mut shape: Vec<String>, axis: isize, dim: String) -> Vec<String> {
    let len = shape.len() as isize;
    // Normalise negative axis: -1 means "insert before the last element"
    // which means position len for len elements, or position len-1+1=len.
    // np.insert convention: axis = -1 → insert at the end.
    let axis = if axis < 0 {
        (len + axis + 1).max(0) as usize
    } else {
        (axis as usize).min(shape.len())
    };
    shape.insert(axis, dim);
    shape
}

fn propagate_calls(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    layer_records: &[LayerAssignment],
    scopes: &mut [FunctionShapeScope],
) -> Result<(Vec<LayerApplication>, Vec<ShapeError>), String> {
    let mut applications = Vec::new();
    let mut errors = Vec::new();
    let mut vmap_targets: HashMap<String, VmapInfo> = HashMap::new();

    // Collect all assignments (with identifier LHS) in source order.
    let assignments = collect_identifier_assignments(node, text)?;

    let mut ctx = ShapeCtx {
        text,
        import_map,
        layer_records,
        scopes,
        vmap_targets: &mut vmap_targets,
        applications: &mut applications,
        errors: &mut errors,
        synthetic_counter: 0,
        synthetics: HashMap::new(),
    };

    for (lhs_name, rhs_node, _assignment_node) in assignments {
        let scope_idx = match scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) {
            Some(idx) => idx,
            None => continue,
        };

        // Special-case vmap *recording* before delegating to
        // shape_of_expression.  vmap bindings like `vf = jax.vmap(f)`
        // produce no output shape of their own but need to be recorded
        // so that later calls to `vf(...)` can be expanded.
        //
        // TODO: Consider threading lhs_name through ShapeCtx so that
        // shape_of_call can own the vmap-recording path too, eliminating
        // this duplication.  Currently shape_of_call doesn't know the
        // LHS variable name for the assignment.
        if rhs_node.kind() == "call"
            && let Some(func_node) = rhs_node.child_by_field_name("function") {
                let target = func_node.utf8_text(text.as_bytes()).ok().unwrap_or("");
                let resolved = resolve_call_target(target, import_map);
                if let Some(KnownFunction::Vmap) = classify_known_function(&resolved) {
                    let args_node = rhs_node.child_by_field_name("arguments");
                    if let Some(an) = args_node
                        && let Ok(args) = extract_call_arguments(an, text)
                            && let Some(info) = parse_vmap_call(&args) {
                                ctx.vmap_targets.insert(lhs_name.clone(), info);
                            }
                    // vmap binding has no output shape — skip to next assignment.
                    continue;
                }
            }

        // Also need to handle the old-style layer applications. The
        // layer-application path in shape_of_call doesn't know the LHS
        // variable name, so we do a pre-check here for layer calls.
        // Inline expression arguments (e.g. `layer(jnp.exp(x))`) are
        // resolved via resolve_call_args so nested inputs get a shape.
        //
        // TODO: Same as vmap above — threading lhs_name through ShapeCtx
        // would let shape_of_call own the layer-application path, removing
        // this duplication.
        if rhs_node.kind() == "call"
            && let Some(func_node) = rhs_node.child_by_field_name("function") {
                let target = func_node.utf8_text(text.as_bytes()).ok().unwrap_or("");
                let call_byte = rhs_node.start_byte();
                if let Some(kind) =
                    find_scoped_layer(ctx.layer_records, ctx.scopes, call_byte, target)
                {
                    let args_node = rhs_node.child_by_field_name("arguments");
                    if let Some(an) = args_node
                        && let Ok(raw_args) = extract_call_arguments(an, text)
                            // Recursively evaluate inline expression args (e.g.
                            // `layer(jnp.exp(x))`) so the input resolves to a
                            // synthetic name with a known shape.
                            && let Some(args) =
                                resolve_call_args(raw_args, an, scope_idx, &mut ctx)
                            && let Some(CallArgument::Positional { value: input }) =
                                args.first().cloned()
                            {
                                let application = LayerApplication {
                                    variable: lhs_name.clone(),
                                    layer: target.to_string(),
                                    input,
                                    kind: kind.clone(),
                                    range: an.range(),
                                };
                                let merged = ctx.merged_shapes(scope_idx);
                                let result = apply_layer_application(&application, &merged);
                                match result {
                                    Ok(Some(output)) => {
                                        ctx.scopes[scope_idx]
                                            .shapes
                                            .insert(lhs_name.clone(), output);
                                    }
                                    Ok(None) => {}
                                    Err(message) => ctx.errors.push(ShapeError {
                                        variable: lhs_name.clone(),
                                        message,
                                        range: application.range,
                                    }),
                                }
                                ctx.applications.push(application);
                                continue;
                            }
                }
            }

        // Delegate to the recursive evaluator.
        let errors_before = ctx.errors.len();
        let result = shape_of_expression(rhs_node, &mut ctx);

        // Fix up error variable names: shape_of_expression doesn't know the
        // LHS variable name, so it records errors with placeholder names
        // (the function target or expression text). Replace with the actual
        // LHS name.
        for err in &mut ctx.errors[errors_before..] {
            err.variable = lhs_name.clone();
        }

        if let Some(shape) = result {
            ctx.scopes[scope_idx].shapes.insert(lhs_name.clone(), shape);
        }
    }

    // Also handle binary-operator expressions in return/yield/assert
    // statements. The recursive evaluator handles the expression itself,
    // but we need to walk these statement types to catch shape errors
    // (e.g., matmul dimension mismatches) that don't appear in assignments.
    let value_statements = collect_value_statements(node, text)?;
    for stmt in value_statements {
        let errors_before = ctx.errors.len();
        let _ = shape_of_expression(stmt, &mut ctx);
        // Fix up error variable names: for return/yield/assert contexts,
        // there is no LHS variable, so set the variable name to empty.
        for err in &mut ctx.errors[errors_before..] {
            err.variable = String::new();
        }
    }

    Ok((applications, errors))
}

/// Collect all assignment statements where the LHS is a simple identifier,
/// in source order. Returns (lhs_name, rhs_node, assignment_node).
fn collect_identifier_assignments<'a>(
    node: Node<'a>,
    text: &str,
) -> Result<Vec<(String, Node<'a>, Node<'a>)>, String> {
    let mut result = Vec::new();
    collect_assignments_recursive(node, text, &mut result)?;
    result.sort_by_key(|(_, rhs, _)| rhs.start_byte());
    Ok(result)
}

fn collect_assignments_recursive<'a>(
    node: Node<'a>,
    text: &str,
    out: &mut Vec<(String, Node<'a>, Node<'a>)>,
) -> Result<(), String> {
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32).unwrap();

        if child.kind() == "expression_statement" {
            // Extract assignments from within expression_statement, but
            // do NOT recurse into it — the inner walk already handled
            // all its children.
            for j in 0..child.named_child_count() {
                let inner = child.named_child(j as u32).unwrap();
                if inner.kind() == "assignment" {
                    try_extract_assignment(inner, text, out);
                }
            }
            continue;
        } else if child.kind() == "assignment" {
            try_extract_assignment(child, text, out);
            // Also don't recurse into bare assignment nodes to avoid
            // double-counting.
            continue;
        }

        collect_assignments_recursive(child, text, out)?;
    }
    Ok(())
}

fn try_extract_assignment<'a>(
    assignment: Node<'a>,
    text: &str,
    out: &mut Vec<(String, Node<'a>, Node<'a>)>,
) {
    let left = assignment.child_by_field_name("left");
    let right = assignment.child_by_field_name("right");

    if let (Some(lhs), Some(rhs)) = (left, right)
        && lhs.kind() == "identifier"
            && let Ok(name) = lhs.utf8_text(text.as_bytes()) {
                out.push((name.to_string(), rhs, assignment));
            }
}


/// Collect the inner expression nodes from return_statement, yield, and
/// assert_statement nodes. These are value-position contexts where
/// shape errors should be detected but no shape is assigned to a variable.
fn collect_value_statements<'a>(
    node: Node<'a>,
    text: &str,
) -> Result<Vec<Node<'a>>, String> {
    let mut result = Vec::new();
    collect_value_statements_recursive(node, text, &mut result)?;
    result.sort_by_key(|n| n.start_byte());
    Ok(result)
}

fn collect_value_statements_recursive<'a>(
    node: Node<'a>,
    _text: &str,
    out: &mut Vec<Node<'a>>,
) -> Result<(), String> {
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32).unwrap();
        match child.kind() {
            "return_statement" | "yield" | "assert_statement" => {
                // The value expression is the first named child.
                // If it's an expression_list (tuple), expand into individual
                // expressions so each element gets evaluated.
                if let Some(expr) = child.named_child(0) {
                    if expr.kind() == "expression_list" {
                        for j in 0..expr.named_child_count() {
                            if let Some(elem) = expr.named_child(j as u32) {
                                out.push(elem);
                            }
                        }
                    } else {
                        out.push(expr);
                    }
                }
            }
            "expression_statement" => {
                // Bare call expressions like `jnp.matmul(x, y)` whose result
                // is discarded.  Also handle yield/return/assert inside
                // expression_statement wrappers.
                // If the immediate child is a call, evaluate it so that
                // shape errors are detected even without an assignment.
                if let Some(expr) = child.named_child(0) {
                    match expr.kind() {
                        "call" => out.push(expr),
                        "yield" | "return_statement" | "assert_statement" => {
                            // Unwrap the value expression from these
                            // statement-like wrappers.
                            if let Some(inner) = expr.named_child(0) {
                                if inner.kind() == "expression_list" {
                                    for j in 0..inner.named_child_count() {
                                        if let Some(elem) = inner.named_child(j as u32) {
                                            out.push(elem);
                                        }
                                    }
                                } else {
                                    out.push(inner);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // Don't recurse — assignments inside expression_statement
                // are already handled by collect_identifier_assignments.
            }
            _ => {
                collect_value_statements_recursive(child, _text, out)?;
            }
        }
    }
    Ok(())
}

/// Expand a call to a vmap-bound name.
///
/// `info` describes the vmap binding (wrapped function name, in_axes, out_axes).
/// `args` are the call's arguments (positional variable names).
/// `caller_shapes` are the caller's known shapes.
/// `scopes` are all function scopes (used to find the wrapped function).
///
/// Returns:
/// - `Ok(Some(shape))` — output shape with batch dim prepended.
/// - `Ok(None)` — silently skipped (missing shapes, no annotations, etc.).
/// - `Err(msg)` — shape error (rank mismatch, batch dim disagreement, etc.).
fn apply_vmap_call(
    info: &VmapInfo,
    args: &[CallArgument],
    caller_shapes: &HashMap<String, Vec<String>>,
    scopes: &[FunctionShapeScope],
) -> Result<Option<Vec<String>>, String> {
    // a. Resolve each positional arg's shape from caller_shapes.
    //    Peel the batch dim from each.
    let mut peeled_arg_shapes: Vec<(&str, Vec<String>)> = Vec::new(); // (arg_name, peeled_shape)
    let mut batch_dim: Option<String> = None;

    for arg in args {
        let CallArgument::Positional { value } = arg else {
            // Skip non-positional args silently (v1 doesn't pass kwargs through vmap).
            continue;
        };
        let Some(shape) = caller_shapes.get(value.as_str()) else {
            // Arg has no known shape — skip silently.
            return Ok(None);
        };
        // b. Peel the batch dim at in_axes.
        match peel_batch_dim(shape, info.in_axes) {
            Ok((peeled, dim)) => {
                // d. All peeled batch dims must match.
                if let Some(ref existing) = batch_dim {
                    if existing != &dim {
                        return Err(format!(
                            "vmap input batch dims disagree: '{}' vs '{}'",
                            existing, dim
                        ));
                    }
                } else {
                    batch_dim = Some(dim);
                }
                peeled_arg_shapes.push((value.as_str(), peeled));
            }
            Err(msg) => {
                return Err(format!(
                    "vmap: argument '{}' rank insufficient for in_axes={}: {}",
                    value, info.in_axes, msg
                ));
            }
        }
    }

    // e. Find the wrapped function's FunctionShapeScope.
    let callee = match find_callee_scope(&info.wrapped, None, scopes) {
        Some(idx) => &scopes[idx],
        None => return Ok(None), // wrapped function not found — skip silently
    };

    // If the callee has no jaxtyping annotations at all, skip.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return Ok(None);
    }

    // f. Map positional arg shapes to param names (using param_order),
    //    then bind and substitute.
    let param_names = &callee.param_order;
    let arg_shapes: Vec<(&str, Vec<String>)> = peeled_arg_shapes
        .iter()
        .enumerate()
        .filter_map(|(idx, (_, shape))| param_names.get(idx).map(|p| (p.as_str(), shape.clone())))
        .collect();

    let result = bind_and_substitute(callee, &info.wrapped, &arg_shapes)?;

    // g. If return_shape is None after substitution, no output to propagate.
    let Some(substituted) = result else {
        return Ok(None);
    };

    // h. Prepend the batch dim at out_axes.
    let Some(ref dim) = batch_dim else {
        // No positional args at all — can't determine batch dim.
        return Ok(None);
    };

    let output = prepend_batch_dim(substituted, info.out_axes, dim.clone());

    // i. Store result — done by caller.
    Ok(Some(output))
}

/// Search for a `FunctionShapeScope` whose `function_name` matches `target`,
/// excluding the module scope (index 0) and optionally excluding scopes
/// whose byte range contains `call_byte`.
///
/// - `Some(call_byte)`: exclude scopes whose byte range contains this byte
///   (used by user-function calls to avoid self-recursive binding).
/// - `None`: don't exclude any scope (used by vmap where the call is not
///   inside the wrapped function).
///
/// Returns the index of the best (smallest-scope) matching scope, or `None`.
fn find_callee_scope(
    target: &str,
    call_byte: Option<usize>,
    scopes: &[FunctionShapeScope],
) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (scope index, scope size)
    for (i, scope) in scopes.iter().enumerate() {
        if i == 0 {
            continue; // skip module scope
        }
        if scope.function_name.as_deref() != Some(target) {
            continue;
        }
        // Don't bind a call to itself (recursive call in the same function body).
        if let Some(byte) = call_byte
            && scope.start_byte <= byte
            && byte < scope.end_byte
        {
            continue;
        }
        let size = scope.end_byte - scope.start_byte;
        match best {
            None => best = Some((i, size)),
            Some((_, prev_size)) if size < prev_size => best = Some((i, size)),
            _ => {} // keep first on tie
        }
    }
    best.map(|(i, _)| i)
}

/// Bind a callee's parameter dims to provided arg shapes and substitute
/// into the callee's return shape.
///
/// `positional_arg_shapes` is `[(param_name, shape)]` — callers must
/// map argument variable names to the callee's declared parameter names
/// before calling this function. Both the user-function path and the
/// vmap path do this mapping upstream.
///
/// Returns:
/// - `Ok(Some(substituted_return_shape))` if binding and substitution
///   succeeded and the callee has a return annotation.
/// - `Ok(None)` if the callee has no return annotation (or no shapes at all).
/// - `Err(msg)` if there's a rank or dim mismatch.
fn bind_and_substitute(
    callee: &FunctionShapeScope,
    target_name: &str,
    positional_arg_shapes: &[(&str, Vec<String>)],
) -> Result<Option<Vec<String>>, String> {
    // If the callee has no jaxtyping annotations at all, skip.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return Ok(None);
    }

    // Build a binding: declared dim name → resolved dim from the caller's arg.
    let mut binding: HashMap<String, String> = HashMap::new();
    for (param_name, arg_shape) in positional_arg_shapes {
        let Some(param_shape) = callee.shapes.get(*param_name) else {
            continue;
        };

        if param_shape.len() != arg_shape.len() {
            return Err(format!(
                "call to {}: argument '{}' expected rank {}, got rank {}",
                target_name,
                param_name,
                param_shape.len(),
                arg_shape.len()
            ));
        }

        for (dim_idx, (param_dim, arg_dim)) in param_shape.iter().zip(arg_shape.iter()).enumerate()
        {
            let is_param_concrete = param_dim.parse::<usize>().is_ok();

            if is_param_concrete {
                if param_dim != arg_dim {
                    return Err(format!(
                        "call to {}: argument '{}' dim {} expected {}, got {}",
                        target_name, param_name, dim_idx, param_dim, arg_dim
                    ));
                }
            } else if let Some(existing) = binding.get(param_dim) {
                if existing != arg_dim {
                    return Err(format!(
                        "call to {}: dim '{}' cannot be both '{}' and '{}'",
                        target_name, param_dim, existing, arg_dim
                    ));
                }
            } else {
                binding.insert(param_dim.clone(), arg_dim.clone());
            }
        }
    }

    // Substitute into return_shape.
    let Some(ref return_shape) = callee.return_shape else {
        return Ok(None);
    };

    let substituted: Vec<String> = return_shape
        .iter()
        .map(|dim| binding.get(dim).cloned().unwrap_or_else(|| dim.clone()))
        .collect();

    Ok(Some(substituted))
}

/// Attempt to resolve `target` as a user-defined function in the same file
/// and propagate its declared return shape to the call site.
///
/// Returns:
/// - `Some(Ok(Some(shape)))` if a matching function was found and its
///   return shape could be computed after binding param dims to arg dims.
/// - `Some(Ok(None))` if a matching function was found but has no return
///   annotation — argument validation still ran but nothing to propagate.
/// - `Some(Err(msg))` if argument shapes don't unify with declared param shapes.
/// - `None` if no matching user-defined function was found (fall through to
///   the known-function branch).
///
/// v1 limitations (documented in PR):
/// - Only positional arguments are matched. Keyword args that match a param
///   name are honoured; otherwise the call is skipped with Ok(None).
/// - No cross-file resolution.
/// - Qualified names ("module.func") are excluded at the call site.
/// - Fresh output dims (not in the binding) pass through unchanged.
fn apply_user_function(
    target: &str,
    call_byte: usize,
    args: &[CallArgument],
    caller_shapes: &HashMap<String, Vec<String>>,
    scopes: &[FunctionShapeScope],
) -> Option<Result<Option<Vec<String>>, String>> {
    let scope_idx = find_callee_scope(target, Some(call_byte), scopes)?;
    let callee = &scopes[scope_idx];

    // If the callee has no jaxtyping annotations at all, fall through to
    // the known-function branch.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return None;
    }

    // Resolve each positional arg to a shape from the caller's scope.
    // We match positional args to declared params in declaration order.
    let param_names = &callee.param_order;
    let mut arg_shapes: Vec<(&str, Vec<String>)> = Vec::new(); // (param_name, arg_shape)
    let mut positional_idx = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                let Some(param_name) = param_names.get(positional_idx) else {
                    break; // more positional args than annotated params — ignore extras
                };
                let Some(shape) = caller_shapes.get(value.as_str()) else {
                    // Arg has no known shape (literal, untracked variable, etc.)
                    // — skip entire function. v1 intentionally bails early
                    // rather than partially validating some args, to avoid
                    // noisy diagnostics on calls that mix typed and untyped
                    // arguments.
                    return Some(Ok(None));
                };
                arg_shapes.push((param_name, shape.clone()));
                positional_idx += 1;
            }
            CallArgument::Keyword { name, value } => {
                // v1: honour keyword args whose name matches a declared param.
                if callee.shapes.contains_key(name) {
                    let Some(shape) = caller_shapes.get(value.as_str()) else {
                        return Some(Ok(None));
                    };
                    arg_shapes.push((name, shape.clone()));
                }
                // Non-matching keyword args are silently ignored in v1.
            }
        }
    }

    // Delegate to the shared bind_and_substitute helper.
    let result = bind_and_substitute(callee, target, &arg_shapes);
    match result {
        Ok(substituted) => Some(Ok(substituted)),
        Err(msg) => Some(Err(msg)),
    }
}

fn apply_matmul_shape(
    left: &[String],
    right: &[String],
    left_name: &str,
    right_name: &str,
) -> Result<Option<Vec<String>>, String> {
    // Rank < 2: return Ok(None) for v1
    if left.len() < 2 || right.len() < 2 {
        return Ok(None);
    }

    // Batch dims must match exactly (no broadcasting in v1)
    let left_batch = &left[..left.len() - 2];
    let right_batch = &right[..right.len() - 2];
    if left_batch != right_batch {
        return Err(format!(
            "matmul batch dimension mismatch: {} batch [{}] != {} batch [{}]",
            left_name,
            left_batch.join(", "),
            right_name,
            right_batch.join(", ")
        ));
    }

    // Last dim of LHS must equal second-to-last dim of RHS.
    // Invariant: left.len() >= 2 and right.len() >= 2 (guard above).
    let lhs_last = left
        .last()
        .expect("invariant: left.len() >= 2 checked above");
    let rhs_second_last = &right[right.len() - 2];

    if lhs_last != rhs_second_last {
        return Err(format!(
            "matmul dimension mismatch: {} last dim {} != {} second-to-last dim {}",
            left_name, lhs_last, right_name, rhs_second_last
        ));
    }

    let mut output = left[..left.len() - 1].to_vec();
    output.push(
        right
            .last()
            .expect("invariant: right.len() >= 2 checked above")
            .clone(),
    );
    Ok(Some(output))
}

fn apply_elementwise_shape(
    left: &[String],
    right: &[String],
    op: BinaryOp,
) -> Result<Option<Vec<String>>, String> {
    // Scalar / rank-0: Ok(None) for now
    if left.is_empty() || right.is_empty() {
        return Ok(None);
    }

    let op_symbol = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::MatMul => unreachable!(),
    };

    let rank = left.len().max(right.len());
    let mut result = Vec::with_capacity(rank);
    for i in 0..rank {
        // right-align: None = leading pad of the shorter operand
        let a = (i + left.len()).checked_sub(rank).map(|j| &left[j]);
        let b = (i + right.len()).checked_sub(rank).map(|j| &right[j]);
        let dim = match (a, b) {
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (Some(a), Some(b)) => {
                if a == b || b == "1" {
                    a.clone()
                } else if a == "1" {
                    b.clone()
                } else {
                    // ponytail: unequal non-"1" dims are incompatible, including
                    // distinct symbolic names ("batch" vs "seq"). Strict in v1;
                    // upgrade to a unification step if it proves too strict.
                    return Err(format!(
                        "elementwise {} incompatible shapes, got [{}] and [{}]",
                        op_symbol,
                        left.join(", "),
                        right.join(", ")
                    ));
                }
            }
            (None, None) => unreachable!("i < max(left.len(), right.len())"),
        };
        result.push(dim);
    }

    Ok(Some(result))
}

/// Check if a resolved call target is a shape-preserving function
/// (elementwise math or activation) that isn't in the `KnownFunction` enum.
///
/// These functions always return the input array's shape unchanged,
/// so we can handle them with a simple "return first arg's shape" rule
/// without adding each one to the enum.
fn is_shape_preserving_call(resolved: &ResolvedTarget) -> bool {
    let Some((name, module)) = resolved.parts.split_last() else {
        return false;
    };
    let name: &str = name;

    // jax.nn activations
    if module == ["jax", "nn"] {
        return matches!(name,
            "relu" | "sigmoid" | "softplus" | "silu" | "swish" | "gelu"
            | "elu" | "leaky_relu" | "selu" | "hard_sigmoid" | "hard_silu"
            | "hard_tanh" | "hard_swish" | "mish" | "celu" | "log_sigmoid"
            | "log_softmax" | "softmax" | "standardize" | "one_hot"
        );
    }

    // jax.numpy / numpy elementwise math
    if module == ["jax", "numpy"] || module == ["numpy"] {
        return matches!(name,
            "exp" | "exp2" | "expm1" | "log" | "log2" | "log10" | "log1p"
            | "sqrt" | "cbrt" | "square" | "abs" | "fabs" | "sign" | "conj"
            | "real" | "imag" | "angle" | "clip" | "ceil" | "floor" | "trunc"
            | "rint" | "round" | "around" | "fix" | "positive" | "negative"
            | "absolute" | "reciprocal"
            | "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "arcsinh" | "arccosh" | "arctanh"
            | "deg2rad" | "rad2deg" | "degrees" | "radians"
            | "add" | "subtract" | "multiply" | "divide" | "true_divide"
            | "floor_divide" | "power" | "float_power" | "remainder" | "mod"
            | "fmod" | "maximum" | "minimum" | "fmax" | "fmin" | "copysign"
            | "heaviside" | "logical_not" | "logical_and" | "logical_or"
            | "logical_xor" | "equal" | "not_equal" | "less" | "less_equal"
            | "greater" | "greater_equal" | "isnan" | "isfinite" | "isinf"
            | "isneginf" | "isposinf" | "signbit" | "nextafter"
        );
    }

    // torch elementwise math + activations
    if module == ["torch"] {
        return matches!(name,
            "exp" | "log" | "sqrt" | "abs" | "sign" | "clip" | "ceil" | "floor"
            | "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "sigmoid" | "relu" | "gelu"
            | "silu" | "mish" | "softplus" | "softsign" | "prelu"
            | "add" | "sub" | "mul" | "div" | "pow" | "neg"
            | "isnan" | "isfinite" | "isinf"
        );
    }

    false
}

#[cfg(test)]
mod shape_of_expression_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn analyze_simple(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        let import_map = build_import_map(tree.root_node(), code).unwrap();
        let layer_records = Vec::new();
        let mut scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let mut applications = Vec::new();
        let mut errors = Vec::new();
        let mut vmap_targets = HashMap::new();

        let assignments = collect_identifier_assignments(tree.root_node(), code).unwrap();

        let mut ctx = ShapeCtx {
            text: code,
            import_map: &import_map,
            layer_records: &layer_records,
            scopes: &mut scopes,
            vmap_targets: &mut vmap_targets,
            applications: &mut applications,
            errors: &mut errors,
            synthetic_counter: 0,
            synthetics: HashMap::new(),
        };

        for (lhs_name, rhs_node, _assignment_node) in assignments {
            let scope_idx = match scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) {
                Some(idx) => idx,
                None => continue,
            };
            let result = shape_of_expression(rhs_node, &mut ctx);
            if let Some(shape) = result {
                ctx.scopes[scope_idx].shapes.insert(lhs_name, shape);
            }
        }

        let layers = HashMap::new();
        LayerShapeAnalysis {
            scopes,
            layers,
            applications,
            errors,
        }
    }

    fn find_shape<'a>(analysis: &'a LayerShapeAnalysis, var: &str) -> Option<&'a Vec<String>> {
        analysis.scopes.iter().find_map(|s| s.shapes.get(var))
    }

    #[test]
    fn test_nested_call_jnp_exp_reshape() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "4 8"]):
    y = jnp.exp(jnp.reshape(x, (4, 8)))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["4", "8"]))
        );
    }

    #[test]
    fn test_unary_minus_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = -x
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_parenthesized_expression_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = (x)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_chained_method_call_reshape_sum() {
        let code = r#"
def f(x: Float[Array, "3 4"]):
    y = x.reshape(3, 4).sum(axis=1)
"#;
        let analysis = analyze_simple(code);
        // reshape(3,4) → [3, 4], sum(axis=1) → [3]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3"]))
        );
    }

    #[test]
    fn test_existing_single_call_still_works() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jnp.reshape(x, (5, 3))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["5", "3"]))
        );
    }

    #[test]
    fn test_nested_call_two_levels() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "12"]):
    y = jnp.exp(jnp.reshape(x, (3, 4)))
"#;
        let analysis = analyze_simple(code);
        // reshape [12] → [3, 4], exp preserves shape → [3, 4]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "4"]))
        );
    }

    #[test]
    fn test_jax_nn_softplus_shape_preserving() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jax.nn.softplus(x)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_chained_method_reshape_then_flatten() {
        let code = r#"
def f(x: Float[Array, "3 4"]):
    y = x.reshape(3, 4).flatten()
"#;
        let analysis = analyze_simple(code);
        // reshape(3,4) → [3, 4], flatten() → [12]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["12"]))
        );
    }

    #[test]
    fn test_unary_tilde_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = ~x
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_subscript_returns_none() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = x[0]
"#;
        let analysis = analyze_simple(code);
        // Subscript not supported in v1 — should not produce a shape.
        assert_eq!(find_shape(&analysis, "y"), None);
    }

    #[test]
    fn test_parenthesized_call_preserves_shape() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = (jnp.sum(x, axis=1))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3"]))
        );
    }

    #[test]
    fn test_jax_nn_softplus_wrapping_shaped_call() {
        // Acceptance criteria: jax.nn.softplus(some_already_shaped_call(...)) gets a shape
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jax.nn.softplus(jnp.reshape(x, (5, 3)))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["5", "3"]))
        );
    }

    #[test]
    fn test_no_synth_keys_leak_into_scope_shapes() {
        // After analyzing code with nested calls, no __synth_* keys should
        // appear in any scope's public shapes map.
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jnp.exp(jnp.reshape(x, (5, 3)))
"#;
        let analysis = analyze_simple(code);
        for scope in &analysis.scopes {
            for key in scope.shapes.keys() {
                assert!(
                    !key.starts_with("__synth_"),
                    "__synth_* key leaked into scope.shapes: {}",
                    key
                );
            }
        }
    }

    #[test]
    fn test_broadcast_equal_shapes_unchanged() {
        let l = shape(&["batch", "features"]);
        assert_eq!(
            apply_elementwise_shape(&l, &l, BinaryOp::Add),
            Ok(Some(shape(&["batch", "features"])))
        );
    }

    #[test]
    fn test_broadcast_trailing_one_each_side() {
        let a = shape(&["3", "1"]);
        let b = shape(&["3", "5"]);
        assert_eq!(
            apply_elementwise_shape(&a, &b, BinaryOp::Mul),
            Ok(Some(shape(&["3", "5"])))
        );
        assert_eq!(
            apply_elementwise_shape(&b, &a, BinaryOp::Mul),
            Ok(Some(shape(&["3", "5"])))
        );
    }

    #[test]
    fn test_broadcast_rank_mismatch_leading_dims_pass_through() {
        let x = shape(&["batch", "f"]);
        let bias = shape(&["f"]);
        assert_eq!(
            apply_elementwise_shape(&x, &bias, BinaryOp::Add),
            Ok(Some(shape(&["batch", "f"])))
        );
        assert_eq!(
            apply_elementwise_shape(&bias, &x, BinaryOp::Add),
            Ok(Some(shape(&["batch", "f"])))
        );
    }

    #[test]
    fn test_broadcast_scalar_empty_returns_none() {
        let x = shape(&["3", "5"]);
        assert_eq!(apply_elementwise_shape(&[], &x, BinaryOp::Add), Ok(None));
        assert_eq!(apply_elementwise_shape(&x, &[], BinaryOp::Add), Ok(None));
    }

    #[test]
    fn test_broadcast_incompatible_concrete_dims() {
        assert!(apply_elementwise_shape(&shape(&["3"]), &shape(&["4"]), BinaryOp::Add).is_err());
    }

    #[test]
    fn test_broadcast_incompatible_symbolic_dims() {
        assert!(
            apply_elementwise_shape(&shape(&["batch"]), &shape(&["seq"]), BinaryOp::Add).is_err()
        );
    }
}
