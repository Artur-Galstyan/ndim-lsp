use std::collections::HashMap;
use std::path::PathBuf;

use tree_sitter::Node;

use crate::known_functions::{
    apply_known_function, apply_method_call, classify_known_function, classify_method_call,
};
use crate::layers::{apply_layer_application, extract_layer_assignments};
use crate::python_ast::{
    build_import_map, extract_call_arguments, extract_calls, extract_jaxtyping_shapes,
    extract_method_calls,
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
        }
    }

    Ok((applications, errors))
}
