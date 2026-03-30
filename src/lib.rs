use std::collections::HashMap;
use std::usize;
use tokio::sync::RwLock;

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams, MarkedString, MessageType, OneOf,
    Position, Range, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use tower_lsp::{Client, LanguageServer};
use tree_sitter::{Node, Parser};

enum ShapeResult {
    Ok(Vec<String>),
    Error(String),
    Unknown,
}

pub struct ShapeInfo {
    dims: Vec<String>,
    line: u32,
    character: u32,
    is_inferred: bool,
}

pub struct Backend {
    pub client: Client,
    pub shapes: RwLock<HashMap<String, HashMap<String, ShapeInfo>>>,
    pub import_alias_map: RwLock<HashMap<String, String>>,
    pub document_text: RwLock<HashMap<String, String>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                inlay_hint_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "File opened")
            .await;

        self.on_change(&params.text_document.uri, &params.text_document.text)
            .await
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.on_change(&params.text_document.uri, &change.text)
                .await;
            let mut doc_lock = self.document_text.write().await;
            doc_lock.insert(params.text_document.uri.to_string(), change.text);
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let doc_lock = self.document_text.read().await;
        let Some(text) = doc_lock.get(&uri.to_string()) else {
            return Ok(None);
        };
        self.on_hover(text, &pos).await
    }

    async fn inlay_hint(&self, _params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let mut hints = Vec::new();

        let shapes_lock = self.shapes.read().await;

        for (_, fn_shapes) in shapes_lock.iter() {
            for (_name, info) in fn_shapes {
                if !info.is_inferred {
                    continue;
                }
                hints.push(InlayHint {
                    position: Position {
                        line: info.line,
                        character: info.character,
                    },
                    label: InlayHintLabel::String(format!(": ({})", info.dims.join(", "))),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: None,
                    padding_left: Some(true),
                    padding_right: None,
                    data: None,
                });
            }
        }

        Ok(Some(hints))
    }
}

impl Backend {
    async fn on_change(&self, uri: &Url, text: &str) {
        self.client
            .log_message(MessageType::INFO, format!("changed uri {}", uri))
            .await;

        let mut parser = Parser::new();

        let language = tree_sitter_python::LANGUAGE;

        parser
            .set_language(&language.into())
            .expect("Failed to set language");

        let Some(tree) = parser.parse(text, None) else {
            return;
        };

        let mut diagnostics: Vec<Diagnostic> = Vec::new();
        let root = tree.root_node();

        let import_alias_map = build_import_map(root, text);

        let mut typed_functions: Vec<Node<'_>> = Vec::new();
        find_node_by_kind(root, "function_definition", &mut typed_functions);

        let mut shapes: HashMap<String, HashMap<String, ShapeInfo>> = HashMap::new();

        // ITERATE OVER EVERY FUNCTION DEFINITION
        for function_node in typed_functions {
            let Some(function_node_name) = function_node.child_by_field_name("name") else {
                continue;
            };

            let Ok(function_name) = function_node_name.utf8_text(&text.as_bytes()) else {
                continue;
            };

            let mut params = HashMap::new();
            // Fill the shapes from the fn args into params dict
            get_shapes_from_fn_args(function_node, &mut params, text);

            let mut assignment_nodes: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(function_node, "assignment", &mut assignment_nodes);

            for assignment_node in assignment_nodes {
                let Some(right_child) = assignment_node.child_by_field_name("right") else {
                    continue;
                };

                let resolved_shape =
                    match resolve_shape(right_child, &params, &import_alias_map, text) {
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
                    ShapeInfo {
                        dims: resolved_shape,
                        line: assignment_node.end_position().row as u32,
                        character: assignment_node.end_position().column as u32,
                        is_inferred: true,
                    },
                );
            }

            let mut binary_operator_nodes: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(function_node, "binary_operator", &mut binary_operator_nodes);

            for binary_operator_node in binary_operator_nodes {
                match resolve_shape(binary_operator_node, &params, &import_alias_map, text) {
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
        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;

        let mut shapes_lock = self.shapes.write().await;
        *shapes_lock = shapes;
        let mut import_alias_map_lock = self.import_alias_map.write().await;
        *import_alias_map_lock = import_alias_map;
    }

    async fn on_hover(&self, text: &str, pos: &Position) -> Result<Option<Hover>> {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser
            .set_language(&language.into())
            .expect("Failed to set language");
        let Some(tree) = parser.parse(text, None) else {
            return Ok(None);
        };

        let root = tree.root_node();
        let point = tree_sitter::Point::new(pos.line as usize, pos.character as usize);

        let Some(node) = root.descendant_for_point_range(point, point) else {
            return Ok(None);
        };

        let Ok(name) = node.utf8_text(text.as_bytes()) else {
            return Ok(None);
        };

        let Some(parent) = get_first_matching_parent(node, "function_definition") else {
            return Ok(None);
        };

        let mut function_name = None;
        if let Some(name_node) = parent.child_by_field_name("name") {
            function_name = name_node.utf8_text(&text.as_bytes()).ok();
        }

        let shapes_lock = self.shapes.read().await;
        let Some(fn_name) = function_name else {
            return Ok(None);
        };
        let Some(fn_shapes) = shapes_lock.get(fn_name) else {
            return Ok(None);
        };
        let Some(param) = fn_shapes.get(name) else {
            return Ok(None);
        };

        Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::String(format!(
                "shape: {:?}",
                param.dims
            ))),
            range: None,
        }))
    }
}

fn find_node_by_kind<'a>(node: Node<'a>, kind: &str, results: &mut Vec<Node<'a>>) {
    if node.kind() == kind {
        results.push(node);
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        find_node_by_kind(child, kind, results);
    }
}

fn get_first_matching_parent<'a>(node: Node<'a>, target_type: &str) -> Option<Node<'a>> {
    let mut parent = node.parent();

    while let Some(p) = parent {
        if p.kind() == target_type {
            return Some(p);
        } else {
            parent = p.parent();
        }
    }

    return None;
}

fn resolve_shape(
    node: Node<'_>,
    params: &HashMap<String, ShapeInfo>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    match node.kind() {
        "identifier" => {
            let param_name = node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");
            match params.get(param_name) {
                Some(info) => ShapeResult::Ok(info.dims.clone()),
                None => ShapeResult::Unknown,
            }
        }
        "call" => {
            let Some(func_node) = node.child_by_field_name("function") else {
                return ShapeResult::Unknown;
            };
            let Some(args_node) = node.child_by_field_name("arguments") else {
                return ShapeResult::Unknown;
            };

            if func_node.kind() == "attribute" {
                let Some(obj) = func_node.child_by_field_name("object") else {
                    return ShapeResult::Unknown;
                };
                let Some(attr) = func_node.child_by_field_name("attribute") else {
                    return ShapeResult::Unknown;
                };
                let Some(obj_name) = obj.utf8_text(text.as_bytes()).ok() else {
                    return ShapeResult::Unknown;
                };
                let Some(attr_name) = attr.utf8_text(text.as_bytes()).ok() else {
                    return ShapeResult::Unknown;
                };
                let resolved_object = import_alias_map
                    .get(obj_name)
                    .map(|s| s.as_str())
                    .unwrap_or(obj_name);
                match (resolved_object, attr_name) {
                    ("jax.numpy", "transpose") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Unknown;
                        };
                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
                                ShapeResult::Ok(dims) => dims,
                                other => return other,
                            };

                        let Some(tuple_node) = get_arg(args_node, 1, "axes", text) else {
                            return ShapeResult::Unknown;
                        };

                        let mut new_shape = Vec::new();
                        let mut cursor = tuple_node.walk();
                        for child in tuple_node.named_children(&mut cursor) {
                            if child.kind() == "integer" {
                                let Ok(text) = child.utf8_text(text.as_bytes()) else {
                                    return ShapeResult::Unknown;
                                };
                                let Ok(idx) = text.parse::<usize>() else {
                                    return ShapeResult::Unknown;
                                };

                                let Some(dim) = input_shape.get(idx) else {
                                    return ShapeResult::Error(format!(
                                        "Axis {} is out of bounds for shape with {} dims",
                                        idx,
                                        input_shape.len()
                                    ));
                                };

                                new_shape.push(dim.clone());
                            }
                        }
                        ShapeResult::Ok(new_shape)
                    }
                    ("jax.numpy", "sum")
                    | ("jax.numpy", "mean")
                    | ("jax.numpy", "max")
                    | ("jax.numpy", "min") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Unknown;
                        };

                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
                                ShapeResult::Ok(dims) => dims,
                                other => return other,
                            };

                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            return ShapeResult::Unknown;
                        };

                        let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
                            return ShapeResult::Unknown;
                        };

                        let Ok(parsed_axis) = axis_str.parse::<usize>() else {
                            return ShapeResult::Unknown;
                        };

                        if parsed_axis >= input_shape.len() {
                            return ShapeResult::Error(format!(
                                "Axis index {} is out of bounds for shape with {} dimensions",
                                parsed_axis,
                                input_shape.len()
                            ));
                        }

                        let keepdims = match get_arg(args_node, 4, "keepdims", text) {
                            Some(n) => n.utf8_text(text.as_bytes()).ok() == Some("True"),
                            None => false,
                        };
                        let mut result = input_shape.clone();

                        if keepdims {
                            result[parsed_axis] = "1".to_string();
                        } else {
                            result.remove(parsed_axis);
                        }
                        return ShapeResult::Ok(result);
                    }
                    ("jax.numpy", "expand_dims") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get input shape"
                            ));
                        };
                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get axis"
                            ));
                        };

                        let shape = match resolve_shape(input_node, params, import_alias_map, text)
                        {
                            ShapeResult::Ok(items) => items,
                            other => return other,
                        };

                        let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get axis string"
                            ));
                        };

                        let Ok(parsed_axis) = axis_str.parse::<usize>() else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to parse axis string"
                            ));
                        };

                        let mut current_dims = shape.clone();
                        if parsed_axis > shape.len() {
                            return ShapeResult::Error(format!(
                                "Axis {} is out of bounds for expand_dims on shape with {} dims",
                                parsed_axis,
                                shape.len()
                            ));
                        }
                        current_dims.insert(parsed_axis, "1".to_string());
                        return ShapeResult::Ok(current_dims);
                    }
                    ("jax.numpy", "squeeze") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get input shape"
                            ));
                        };
                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
                                ShapeResult::Ok(items) => items,
                                other => return other,
                            };

                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            let result: Vec<String> =
                                input_shape.into_iter().filter(|d| d != "1").collect();
                            return ShapeResult::Ok(result);
                        };

                        let Some(parsed_axis) = parse_axis(axis_node, text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to parse axis string"
                            ));
                        };

                        if parsed_axis >= input_shape.len() {
                            return ShapeResult::Error(format!(
                                "Axis {} is out of bounds for shape with {} dims",
                                parsed_axis,
                                input_shape.len()
                            ));
                        }

                        if input_shape[parsed_axis] != "1" {
                            return ShapeResult::Error(format!(
                                "Cannot squeeze axis {} with dim '{}' — only dims of size 1 can be squeezed",
                                parsed_axis, input_shape[parsed_axis]
                            ));
                        }

                        let mut new_dims = input_shape.clone();
                        new_dims.remove(parsed_axis);
                        return ShapeResult::Ok(new_dims);
                    }
                    _ => return ShapeResult::Unknown,
                }
            } else {
                return ShapeResult::Unknown;
            }
        }
        "parenthesized_expression" => {
            let binary_operator_child = node
                .named_child(0)
                .expect("A parenthesized_expression always has a child");

            return resolve_shape(binary_operator_child, params, import_alias_map, text);
        }
        "attribute" => {
            let Some(attribute_identifier_node) = node.child_by_field_name("attribute") else {
                return ShapeResult::Unknown;
            };

            let attribute_name = attribute_identifier_node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");

            match attribute_name {
                "T" => {
                    let Some(object_node) = node.child_by_field_name("object") else {
                        return ShapeResult::Error(format!(
                            "Unexpected TS error: Failed to get object child from attribute node"
                        ));
                    };
                    let mut shape = match resolve_shape(object_node, params, import_alias_map, text)
                    {
                        ShapeResult::Ok(shape) => shape,
                        other => return other,
                    };
                    shape.reverse();
                    return ShapeResult::Ok(shape);
                }
                _ => ShapeResult::Unknown,
            }
        }
        "binary_operator" => {
            let Some(op) = node.child_by_field_name("operator") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get operator child from binary_operator node"
                        .to_string(),
                );
            };

            let op_text = op
                .utf8_text(&text.as_bytes())
                .expect("Operator has no text");

            let Some(left_node) = node.child_by_field_name("left") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get left child from binary_operator node"
                        .to_string(),
                );
            };
            let Some(right_node) = node.child_by_field_name("right") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get right child from binary_operator node"
                        .to_string(),
                );
            };

            let left_shape = match resolve_shape(left_node, params, import_alias_map, text) {
                ShapeResult::Ok(shape) => shape,
                other => return other,
            };
            let right_shape = match resolve_shape(right_node, params, import_alias_map, text) {
                ShapeResult::Ok(shape) => shape,
                other => return other,
            };

            match op_text {
                "@" => {
                    if &left_shape.last().unwrap() != &right_shape.first().unwrap() {
                        return ShapeResult::Error(format!(
                            "Invalid shapes found: {:?} and {:?}",
                            left_shape, right_shape
                        ));
                    } else {
                        return ShapeResult::Ok(vec![
                            left_shape.first().unwrap().clone(),
                            right_shape.last().unwrap().clone(),
                        ]);
                    }
                }
                "+" | "-" | "*" | "/" => handle_elementwise_ops(left_shape, right_shape),
                _ => return ShapeResult::Unknown,
            }
        }
        _ => return ShapeResult::Unknown,
    }
}

fn handle_elementwise_ops(left_shape: Vec<String>, right_shape: Vec<String>) -> ShapeResult {
    if left_shape.len() != right_shape.len() {
        return ShapeResult::Error(format!(
            "Invalid shapes found: {:?} and {:?}",
            left_shape, right_shape
        ));
    }

    for i in 0..left_shape.len() {
        if left_shape[i] != right_shape[i] {
            return ShapeResult::Error(format!(
                "Invalid shapes found: {:?} and {:?}",
                left_shape, right_shape
            ));
        }
    }

    return ShapeResult::Ok(left_shape.clone());
}

fn get_shapes_from_fn_args(
    function_node: Node<'_>,
    params: &mut HashMap<String, ShapeInfo>,
    text: &str,
) {
    let mut typed_params: Vec<Node<'_>> = Vec::new();
    find_node_by_kind(function_node, "typed_parameter", &mut typed_params);

    for param_node in typed_params {
        let Some(node_identifier) = param_node.child(0) else {
            return;
        };

        let param_name = node_identifier
            .utf8_text(text.as_bytes())
            .expect("Failed to get node identifier");

        let Some(node_type) = param_node.child_by_field_name("type") else {
            return;
        };

        let mut string_content_results: Vec<Node<'_>> = Vec::new();
        find_node_by_kind(node_type, "string_content", &mut string_content_results);

        let string_content_node = string_content_results
            .into_iter()
            .next()
            .expect("Expected to find at least one string_content");

        let string_content = string_content_node
            .utf8_text(&text.as_bytes())
            .expect("Failed to get string_content");

        let dims: Vec<String> = string_content
            .split_whitespace()
            .map(String::from)
            .collect();
        let shape_info = ShapeInfo {
            dims,
            line: 0,
            character: 0,
            is_inferred: false,
        };
        params.insert(param_name.to_string(), shape_info);
    }
}

// usize::MAX is a sentinel to indicate that I only care about the kwargs
fn get_arg<'a>(args_node: Node<'a>, position: usize, name: &str, text: &str) -> Option<Node<'a>> {
    let mut positional_index = 0;
    let mut cursor = args_node.walk();

    for child in args_node.named_children(&mut cursor) {
        if child.kind() == "keyword_argument" {
            let name_node = child.child_by_field_name("name")?;
            let kw_name = name_node.utf8_text(text.as_bytes()).ok()?;
            if kw_name == name {
                return child.child_by_field_name("value");
            }
        } else {
            if positional_index == position {
                return Some(child);
            } else {
                positional_index += 1;
            }
        }
    }

    None
}

fn parse_axis(axis_node: Node<'_>, text: &str) -> Option<usize> {
    let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
        return None;
    };
    let Ok(parsed_axis) = axis_str.parse::<usize>() else {
        return None;
    };
    Some(parsed_axis)
}

fn build_import_map(root: Node, text: &str) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut import_alias_nodes: Vec<Node<'_>> = Vec::new();
    find_node_by_kind(root, "import_statement", &mut import_alias_nodes);

    for node in import_alias_nodes {
        let Some(name_child) = node.child_by_field_name("name") else {
            continue;
        };

        if name_child.kind() == "aliased_import" {
            let Some(dotted) = name_child.child_by_field_name("name") else {
                continue;
            };
            let Some(alias) = name_child.child_by_field_name("alias") else {
                continue;
            };
            let Ok(full_name) = dotted.utf8_text(text.as_bytes()) else {
                continue;
            };
            let Ok(alias_name) = alias.utf8_text(text.as_bytes()) else {
                continue;
            };
            map.insert(alias_name.to_string(), full_name.to_string());
        } else if name_child.kind() == "dotted_name" {
            let Ok(full_name) = name_child.utf8_text(text.as_bytes()) else {
                continue;
            };
            map.insert(full_name.to_string(), full_name.to_string());
        }
    }

    let mut import_from_nodes = Vec::new();
    find_node_by_kind(root, "import_from_statement", &mut import_from_nodes);

    for node in import_from_nodes {
        let Some(module_node) = node.child_by_field_name("module_name") else {
            continue;
        };
        let Ok(module_name) = module_node.utf8_text(text.as_bytes()) else {
            continue;
        };
        let Some(name_child) = node.child_by_field_name("name") else {
            continue;
        };

        if name_child.kind() == "aliased_import" {
            let Some(dotted) = name_child.child_by_field_name("name") else {
                continue;
            };
            let Some(alias) = name_child.child_by_field_name("alias") else {
                continue;
            };
            let Ok(original) = dotted.utf8_text(text.as_bytes()) else {
                continue;
            };
            let Ok(alias_name) = alias.utf8_text(text.as_bytes()) else {
                continue;
            };
            map.insert(
                alias_name.to_string(),
                format!("{}.{}", module_name, original),
            );
        } else if name_child.kind() == "dotted_name" {
            let Ok(name) = name_child.utf8_text(text.as_bytes()) else {
                continue;
            };
            map.insert(name.to_string(), format!("{}.{}", module_name, name));
        }
    }

    map
}
