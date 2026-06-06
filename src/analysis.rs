use std::collections::HashMap;
use std::path::PathBuf;

use tree_sitter::Node;

use crate::known_functions::{
    apply_known_function, apply_method_call, classify_known_function, classify_method_call,
};
use crate::layers::{apply_layer_application, extract_layer_assignments_scoped};
use crate::python_ast::{
    build_import_map, extract_binary_ops, extract_call_arguments, extract_calls,
    extract_jaxtyping_shapes, extract_method_calls,
};
use crate::resolution::resolve_call_target;

use crate::types::*;

pub fn analyze_layer_shapes<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut scopes = extract_jaxtyping_shapes(node, text)?;
    let import_map = build_import_map(node, text)?;
    let layer_records =
        extract_layer_assignments_scoped(node, text, search_roots, read_file, max_depth)?;

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

enum CallEntry {
    Free(CallInfo),
    Method(MethodCallInfo),
    BinaryOp(BinaryOpInfo),
}

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
        if let CallArgument::Keyword { name: kw_name, value } = arg
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
        return Err(format!(
            "axis {} out of bounds for rank {}",
            axis, len
        ));
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
    let free_calls = extract_calls(node, text)?;
    let method_calls = extract_method_calls(node, text)?;
    let binary_ops = extract_binary_ops(node, text)?;

    let mut entries: Vec<(usize, CallEntry)> = Vec::new();
    for call in free_calls {
        entries.push((call.args_node_range.start_byte, CallEntry::Free(call)));
    }
    for method_call in method_calls {
        if import_map.contains_key(&method_call.receiver) {
            continue;
        }
        entries.push((
            method_call.args_node_range.start_byte,
            CallEntry::Method(method_call),
        ));
    }
    for binary_op in binary_ops {
        entries.push((binary_op.range.start_byte, CallEntry::BinaryOp(binary_op)));
    }
    entries.sort_by_key(|(position, _)| *position);

    let mut applications = Vec::new();
    let mut errors = Vec::new();

    // vmap_targets is global to the analysis pass, not per-scope.
    // Nested-scope shadowing of a vmap name is out of scope for v1.
    let mut vmap_targets: HashMap<String, VmapInfo> = HashMap::new();

    for (position, entry) in entries {
        let Some(scope_idx) = scope_index_for_byte(scopes, position) else {
            continue;
        };

        match entry {
            CallEntry::Free(call) => {
                let Some(args_node) = node.descendant_for_byte_range(
                    call.args_node_range.start_byte,
                    call.args_node_range.end_byte,
                ) else {
                    continue;
                };
                let args = extract_call_arguments(args_node, text)?;

                // 1. Layer check (existing)
                if let Some(kind) =
                    find_scoped_layer(layer_records, scopes, position, &call.target)
                {
                    let Some(CallArgument::Positional { value: input }) = args.first().cloned()
                    else {
                        continue;
                    };
                    let application = LayerApplication {
                        variable: call.variable.clone(),
                        layer: call.target.clone(),
                        input,
                        kind: kind.clone(),
                        range: call.args_node_range,
                    };
                    let scope_shapes = &mut scopes[scope_idx].shapes;
                    match apply_layer_application(&application, scope_shapes) {
                        Ok(Some(output)) => {
                            scope_shapes.insert(call.variable.clone(), output);
                        }
                        Ok(None) => {}
                        Err(message) => errors.push(ShapeError {
                            variable: call.variable.clone(),
                            message,
                            range: application.range,
                        }),
                    }
                    applications.push(application);
                    continue;
                }

                // 2. vmap-target check (Phase 3)
                // If call.target is a name that was bound by a prior
                // `vf = jax.vmap(f)` call, expand it here.
                if let Some(info) = vmap_targets.get(&call.target).cloned() {
                    let result = apply_vmap_call(
                        &info,
                        &args,
                        &scopes[scope_idx].shapes,
                        scopes,
                    );
                    match result {
                        Ok(Some(output_shape)) => {
                            scopes[scope_idx]
                                .shapes
                                .insert(call.variable.clone(), output_shape);
                        }
                        Ok(None) => {
                            // Silently skipped — missing arg shape, no
                            // annotations on wrapped function, etc.
                        }
                        Err(message) => errors.push(ShapeError {
                            variable: call.variable.clone(),
                            message,
                            range: call.args_node_range,
                        }),
                    }
                    continue;
                }

                // 3. User-defined function propagation (Phase 2)
                // Try to resolve `call.target` as a user-defined function in the
                // same file. Qualified names (e.g. "mod.f") are skipped — no
                // cross-module resolution yet. Method calls are dispatched
                // separately above via CallEntry::Method.
                if !call.target.contains('.')
                    && let Some(result) = apply_user_function(
                        &call.target,
                        position,
                        &args,
                        &scopes[scope_idx].shapes,
                        scopes,
                    )
                {
                    match result {
                        Ok(Some(output_shape)) => {
                            scopes[scope_idx]
                                .shapes
                                .insert(call.variable.clone(), output_shape);
                        }
                        Ok(None) => {
                            // No return annotation — nothing to propagate,
                            // but any ShapeErrors from argument validation
                            // were already handled below (they'd be Err).
                        }
                        Err(message) => errors.push(ShapeError {
                            variable: call.variable.clone(),
                            message,
                            range: call.args_node_range,
                        }),
                    }
                    continue;
                    // If no matching user-function scope was found, fall through
                    // to the known-function branch below.
                }
                // Qualified names (e.g. "module.func") are out of scope for
                // user-function propagation in v1 — no cross-module resolution.

                // Before known-function check: intercept vmap/filter_vmap calls
                // that *record* a vmap binding (e.g. `vf = jax.vmap(f)`).
                let resolved = resolve_call_target(&call.target, import_map);
                if let Some(KnownFunction::Vmap) = classify_known_function(&resolved) {
                    if let Some(info) = parse_vmap_call(&args) {
                        vmap_targets.insert(call.variable.clone(), info);
                    }
                    // Whether or not we recorded the binding, don't fall
                    // through to `apply_known_function` — Vmap has no shape
                    // rule of its own.
                    continue;
                }

                // 4. Known-function check (existing)
                let Some(known) = classify_known_function(&resolved) else {
                    continue;
                };
                let scope_shapes = &mut scopes[scope_idx].shapes;
                record_result(
                    apply_known_function(&known, &args, scope_shapes),
                    &call.variable,
                    call.args_node_range,
                    scope_shapes,
                    &mut errors,
                );
            }
            CallEntry::Method(method_call) => {
                let Some(args_node) = node.descendant_for_byte_range(
                    method_call.args_node_range.start_byte,
                    method_call.args_node_range.end_byte,
                ) else {
                    continue;
                };
                let args = extract_call_arguments(args_node, text)?;
                let Some(known) = classify_method_call(&method_call.method) else {
                    continue;
                };
                let scope_shapes = &mut scopes[scope_idx].shapes;
                record_result(
                    apply_method_call(&known, &method_call.receiver, &args, scope_shapes),
                    &method_call.variable,
                    method_call.args_node_range,
                    scope_shapes,
                    &mut errors,
                );
            }
            CallEntry::BinaryOp(binop) => {
                let scope_shapes = &mut scopes[scope_idx].shapes;
                record_result(
                    apply_binary_op(&binop, scope_shapes),
                    &binop.variable,
                    binop.range,
                    scope_shapes,
                    &mut errors,
                );
            }
        }
    }

    Ok((applications, errors))
}

fn record_result(
    result: Result<Option<Vec<String>>, String>,
    variable: &str,
    range: tree_sitter::Range,
    shapes: &mut HashMap<String, Vec<String>>,
    errors: &mut Vec<ShapeError>,
) {
    match result {
        Ok(Some(output)) => {
            if !variable.is_empty() {
                shapes.insert(variable.to_string(), output);
            }
        }
        Ok(None) => {}
        Err(message) => errors.push(ShapeError {
            variable: variable.to_string(),
            message,
            range,
        }),
    }
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
        .filter_map(|(idx, (_, shape))| {
            param_names.get(idx).map(|p| (p.as_str(), shape.clone()))
        })
        .collect();

    let result = bind_and_substitute(
        callee,
        &info.wrapped,
        &arg_shapes,
    )?;

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
            && scope.start_byte <= byte && byte < scope.end_byte
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
                target_name, param_name, param_shape.len(), arg_shape.len()
            ));
        }

        for (dim_idx, (param_dim, arg_dim)) in
            param_shape.iter().zip(arg_shape.iter()).enumerate()
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
                arg_shapes.push((param_name.as_str(), shape.clone()));
                positional_idx += 1;
            }
            CallArgument::Keyword { name, value } => {
                // v1: honour keyword args whose name matches a declared param.
                if callee.shapes.contains_key(name) {
                    let Some(shape) = caller_shapes.get(value.as_str()) else {
                        return Some(Ok(None));
                    };
                    arg_shapes.push((name.as_str(), shape.clone()));
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

fn apply_binary_op(
    binop: &BinaryOpInfo,
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(left_shape) = shapes.get(&binop.left) else {
        return Ok(None);
    };
    let Some(right_shape) = shapes.get(&binop.right) else {
        return Ok(None);
    };

    match binop.op {
        BinaryOp::MatMul => apply_matmul_shape(
            left_shape,
            right_shape,
            &binop.left,
            &binop.right,
        ),
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
            apply_elementwise_shape(left_shape, right_shape, binop.op)
        }
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
    let lhs_last = left.last().expect("invariant: left.len() >= 2 checked above");
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

    if left != right {
        return Err(format!(
            "elementwise {} expected equal shapes, got [{}] and [{}]",
            op_symbol,
            left.join(", "),
            right.join(", ")
        ));
    }

    Ok(Some(left.to_vec()))
}
