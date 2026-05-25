use std::collections::HashMap;
use std::path::PathBuf;

use tree_sitter::Node;

use crate::known_functions::{
    apply_known_function, apply_method_call, classify_known_function, classify_method_call,
};
use crate::layers::{apply_layer_application, extract_layer_assignments};
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
    let layers = extract_layer_assignments(node, text, search_roots, read_file, max_depth)?;

    let (applications, errors) = propagate_calls(node, text, &import_map, &layers, &mut scopes)?;

    Ok(LayerShapeAnalysis {
        scopes,
        layers,
        applications,
        errors,
    })
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
    layers: &HashMap<String, LayerKind>,
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

                if let Some(kind) = layers.get(&call.target) {
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
                match apply_known_function(&known, &args, scope_shapes) {
                    Ok(Some(output)) => {
                        scope_shapes.insert(call.variable.clone(), output);
                    }
                    Ok(None) => {}
                    Err(message) => errors.push(ShapeError {
                        variable: call.variable.clone(),
                        message,
                        range: call.args_node_range,
                    }),
                }
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
                match apply_method_call(&known, &method_call.receiver, &args, scope_shapes) {
                    Ok(Some(output)) => {
                        scope_shapes.insert(method_call.variable.clone(), output);
                    }
                    Ok(None) => {}
                    Err(message) => errors.push(ShapeError {
                        variable: method_call.variable.clone(),
                        message,
                        range: method_call.args_node_range,
                    }),
                }
            }
            CallEntry::BinaryOp(binop) => {
                let scope_shapes = &mut scopes[scope_idx].shapes;
                match apply_binary_op(&binop, scope_shapes) {
                    Ok(Some(output)) => {
                        scope_shapes.insert(binop.variable.clone(), output);
                    }
                    Ok(None) => {}
                    Err(message) => errors.push(ShapeError {
                        variable: binop.variable.clone(),
                        message,
                        range: binop.range,
                    }),
                }
            }
        }
    }

    Ok((applications, errors))
}

fn apply_binary_op(
    binop: &BinaryOpInfo,
    shapes: &mut HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(left_shape) = shapes.get(&binop.left).cloned() else {
        return Ok(None);
    };
    let Some(right_shape) = shapes.get(&binop.right).cloned() else {
        return Ok(None);
    };

    match binop.op {
        BinaryOp::MatMul => apply_matmul_shape(
            &left_shape,
            &right_shape,
            &binop.left,
            &binop.right,
        ),
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
            apply_elementwise_shape(&left_shape, &right_shape, &binop.op)
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

    // Batch matmul: output = LHS[:-1] ++ [RHS[-1]]
    // Last dim of LHS must equal second-to-last dim of RHS
    let lhs_last = left.last().unwrap();
    let rhs_second_last = &right[right.len() - 2];

    if lhs_last != rhs_second_last {
        return Err(format!(
            "matmul dimension mismatch: {} last dim {} != {} second-to-last dim {}",
            left_name, lhs_last, right_name, rhs_second_last
        ));
    }

    let mut output = left[..left.len() - 1].to_vec();
    output.push(right.last().unwrap().clone());
    Ok(Some(output))
}

fn apply_elementwise_shape(
    left: &[String],
    right: &[String],
    op: &BinaryOp,
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
