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
