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

                // --- User-defined function propagation (Phase 2) ---
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

                let resolved = resolve_call_target(&call.target, import_map);
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
            shapes.insert(variable.to_string(), output);
        }
        Ok(None) => {}
        Err(message) => errors.push(ShapeError {
            variable: variable.to_string(),
            message,
            range,
        }),
    }
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
    // Search for a FunctionShapeScope whose function_name matches `target`,
    // excluding the module scope (index 0) and excluding scopes whose
    // byte range contains the call site (to avoid self-recursive binding).
    // If multiple candidates remain, prefer the one with the smallest scope
    // (most specific / innermost). If still tied, take the first match.
    let mut best: Option<(usize, usize)> = None; // (scope index, scope size)
    for (i, scope) in scopes.iter().enumerate() {
        if i == 0 {
            continue; // skip module scope
        }
        if scope.function_name.as_deref() != Some(target) {
            continue;
        }
        // Don't bind a call to itself (recursive call in the same function body).
        if scope.start_byte <= call_byte && call_byte < scope.end_byte {
            continue;
        }
        let size = scope.end_byte - scope.start_byte;
        match best {
            None => best = Some((i, size)),
            Some((_, prev_size)) if size < prev_size => best = Some((i, size)),
            _ => {} // keep first on tie
        }
    }
    let scope_idx = best.map(|(i, _)| i)?;
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

    // Build a binding: declared dim name → resolved dim from the caller's arg.
    // For each param, walk its dims in order against the arg's dims.
    let mut binding: HashMap<String, String> = HashMap::new();
    for (param_name, arg_shape) in &arg_shapes {
        let Some(param_shape) = callee.shapes.get(*param_name) else {
            continue;
        };

        // Rank mismatch
        if param_shape.len() != arg_shape.len() {
            return Some(Err(format!(
                "call to {}: argument '{}' expected rank {}, got rank {}",
                target, param_name, param_shape.len(), arg_shape.len()
            )));
        }

        for (dim_idx, (param_dim, arg_dim)) in
            param_shape.iter().zip(arg_shape.iter()).enumerate()
        {
            let is_param_concrete = param_dim.parse::<usize>().is_ok();

            if is_param_concrete {
                // Concrete param dim must match the arg dim exactly.
                if param_dim != arg_dim {
                    return Some(Err(format!(
                        "call to {}: argument '{}' dim {} expected {}, got {}",
                        target, param_name, dim_idx, param_dim, arg_dim
                    )));
                }
            } else {
                // Symbolic param dim: bind it.
                if let Some(existing) = binding.get(param_dim) {
                    if existing != arg_dim {
                        return Some(Err(format!(
                            "call to {}: dim '{}' cannot be both '{}' and '{}'" ,
                            target, param_dim, existing, arg_dim
                        )));
                    }
                } else {
                    binding.insert(param_dim.clone(), arg_dim.clone());
                }
            }
        }
    }

    // Substitute into return_shape using the binding.
    // Dims not in the binding pass through unchanged (fresh output dims).
    let Some(ref return_shape) = callee.return_shape else {
        // No return annotation — nothing to propagate, but argument
        // validation above succeeded without error.
        return Some(Ok(None));
    };

    let substituted: Vec<String> = return_shape
        .iter()
        .map(|dim| binding.get(dim).cloned().unwrap_or_else(|| dim.clone()))
        .collect();

    Some(Ok(Some(substituted)))
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
