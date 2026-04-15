// analyzer.rs
use crate::{
    assignments::get_assignments,
    binary_operators::get_binary_operators,
    helpers::{self, get_functions},
    imports::get_imports,
    layers::layers::try_parse_layer_constructor,
    shape_resolvers::shape_resolver::{ParamKind, ShapeInfo, ShapeResult, resolve_shape},
};
use std::collections::HashMap;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use tree_sitter::{Node, Parser};

pub struct AnalysisResult {
    pub diagnostics: Vec<Diagnostic>,
    pub shapes: HashMap<String, HashMap<String, ParamKind>>,
    pub imports: HashMap<String, String>,
}

pub fn analyze_document(text: &str) -> Option<AnalysisResult> {
    let mut parser = Parser::new();
    let language = tree_sitter_python::LANGUAGE;
    parser
        .set_language(&language.into())
        .expect("Failed to set language");
    let Some(tree) = parser.parse(text, None) else {
        return None;
    };
    let root = tree.root_node();

    let mut shapes: HashMap<String, HashMap<String, ParamKind>> = HashMap::new();
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let imports = get_imports(root, text).map_err(|e| e.to_string()).ok()?;

    let functions = get_functions(root, text).map_err(|e| e.to_string()).ok()?;
    for function_node in functions {
        let Some(function_node_name) = function_node.child_by_field_name("name") else {
            continue;
        };

        let Ok(function_name) = function_node_name.utf8_text(&text.as_bytes()) else {
            continue;
        };

        let mut params = HashMap::new();
        // Fill the shapes from the fn args into params dict
        helpers::get_shapes_from_fn_args(function_node, &mut params, text);

        let assignment_nodes = get_assignments(function_node, text)
            .map_err(|e| e.to_string())
            .ok()?;

        for assignment_node in assignment_nodes {
            let Some(right_child) = assignment_node.child_by_field_name("right") else {
                continue;
            };

            if right_child.kind() == "call" {
                if let Some(layer_info) = try_parse_layer_constructor(right_child, &imports, text) {
                    let Some(var_name) = assignment_node
                        .child_by_field_name("left")
                        .and_then(|n| n.utf8_text(text.as_bytes()).ok())
                    else {
                        continue;
                    };

                    params.insert(var_name.to_string(), ParamKind::Layer(layer_info));
                    continue;
                }
            }

            let resolved_shape = match resolve_shape(right_child, &params, &imports, text) {
                ShapeResult::Ok(shape) => shape,
                ShapeResult::Error(msg) => {
                    let start = right_child.start_position();
                    let end = right_child.end_position();
                    diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position {
                                line: start.row as u32,
                                character: start.column as u32,
                            },
                            end: Position {
                                line: end.row as u32,
                                character: end.column as u32,
                            },
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("ndim-lsp".to_string()),
                        message: msg,
                        ..Default::default()
                    });
                    continue;
                }
                ShapeResult::Unknown => continue,
            };

            let Some(assign_left) = assignment_node.child_by_field_name("left") else {
                continue;
            };
            let Ok(var_name) = assign_left.utf8_text(text.as_bytes()) else {
                continue;
            };
            params.insert(
                var_name.to_string(),
                ParamKind::Shape(ShapeInfo {
                    dims: resolved_shape,
                    line: assignment_node.end_position().row as u32,
                    character: assignment_node.end_position().column as u32,
                    is_inferred: true,
                }),
            );
        }

        let binary_operator_nodes: Vec<Node<'_>> = get_binary_operators(function_node, text)
            .map_err(|e| e.to_string())
            .ok()?;

        for binary_operator_node in binary_operator_nodes {
            match resolve_shape(binary_operator_node, &params, &imports, text) {
                ShapeResult::Ok(_) => {}
                ShapeResult::Error(msg) => {
                    let start = binary_operator_node.start_position();
                    let end = binary_operator_node.end_position();
                    diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position {
                                line: start.row as u32,
                                character: start.column as u32,
                            },
                            end: Position {
                                line: end.row as u32,
                                character: end.column as u32,
                            },
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("ndim-lsp".to_string()),
                        message: msg,
                        ..Default::default()
                    });
                }
                ShapeResult::Unknown => {}
            }
        }
        shapes.insert(function_name.to_string(), params);
    }

    Some(AnalysisResult {
        diagnostics,
        shapes,
        imports,
    })
}
