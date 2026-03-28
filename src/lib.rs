use std::collections::HashMap;
use std::hash::Hash;
use std::str::FromStr;
use tokio::sync::RwLock;

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    MarkedString, MessageType, Position, Range, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Url,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tree_sitter::{Node, Parser};

pub struct Backend {
    pub client: Client,
    pub shapes: RwLock<HashMap<String, HashMap<String, Vec<String>>>>,
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

        let mut typed_functions: Vec<Node<'_>> = Vec::new();
        find_node_by_kind(root, "function_definition", &mut typed_functions);

        let mut shapes: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();

        // ITERATE OVER EVERY FUNCTION DEFINITION
        for function_node in typed_functions {
            let Some(function_node_name) = function_node.child_by_field_name("name") else {
                continue;
            };

            let Ok(function_name) = function_node_name.utf8_text(&text.as_bytes()) else {
                continue;
            };

            // START: GET SHAPES FROM FUNCTION ARGS
            let mut typed_params: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(function_node, "typed_parameter", &mut typed_params);
            let mut params = HashMap::new();

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
                params.insert(param_name.to_string(), dims);
                self.client
                    .log_message(MessageType::INFO, format!("{:?}", params))
                    .await;
            }
            // END: GET SHAPES FROM FUNCTION ARGS

            let mut assignment_nodes: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(function_node, "assignment", &mut assignment_nodes);

            for assignment_node in assignment_nodes {
                let Some(right_child) = assignment_node.child_by_field_name("right") else {
                    continue;
                };

                let Some(resolved_shape) = resolve_shape(right_child, &params, text) else {
                    continue;
                };

                let Some(assign_left) = assignment_node.child_by_field_name("left") else {
                    continue;
                };
                let Ok(var_name) = assign_left.utf8_text(text.as_bytes()) else {
                    continue;
                };
                params.insert(var_name.to_string(), resolved_shape);
            }

            let mut binary_operator_nodes: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(function_node, "binary_operator", &mut binary_operator_nodes);

            for binary_operator_node in binary_operator_nodes {
                if resolve_shape(binary_operator_node, &params, text).is_none() {
                    let left_node = binary_operator_node.child_by_field_name("left");
                    let right_node = binary_operator_node.child_by_field_name("right");

                    let (Some(left), Some(right)) = (left_node, right_node) else {
                        continue;
                    };

                    let Some(left_shape) = resolve_shape(left, &params, text) else {
                        continue;
                    };

                    let Some(right_shape) = resolve_shape(right, &params, text) else {
                        continue;
                    };

                    let start = binary_operator_node.start_position();
                    let end = binary_operator_node.end_position();
                    let Some(op) = binary_operator_node.child_by_field_name("operator") else {
                        continue;
                    };

                    let op_text = op
                        .utf8_text(&text.as_bytes())
                        .expect("Operator has no text");

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
                        message: format!(
                            "Invalid shapes for this operation: {:?} {} {:?}",
                            left_shape, op_text, right_shape
                        ),
                        ..Default::default()
                    });
                }
            }
            shapes.insert(function_name.to_string(), params);
        }
        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;

        let mut shapes_lock = self.shapes.write().await;
        *shapes_lock = shapes;
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
            contents: HoverContents::Scalar(MarkedString::String(format!("shape: {:?}", param))),
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

fn get_child_param_names(binary_operator: Node<'_>, text: &str) -> Option<(String, String)> {
    let left_child = binary_operator
        .child_by_field_name("left")
        .expect("Failed to find left child");
    let left_child_param_name = left_child
        .utf8_text(&text.as_bytes())
        .expect("Failed to get param name of left child");

    let right_child = binary_operator
        .child_by_field_name("right")
        .expect("Failed to find right child");
    let right_child_param_name = right_child
        .utf8_text(&text.as_bytes())
        .expect("Failed to get param name of right child");

    Some((
        left_child_param_name.to_string(),
        right_child_param_name.to_string(),
    ))
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
    params: &HashMap<String, Vec<String>>,
    text: &str,
) -> Option<Vec<String>> {
    match node.kind() {
        "identifier" => {
            let param_name = node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");
            return params.get(param_name).cloned();
        }
        "parenthesized_expression" => {
            let binary_operator_child = node
                .named_child(0)
                .expect("A parenthesized_expression always has a child");

            return resolve_shape(binary_operator_child, params, text);
        }
        "attribute" => {
            let Some(attribute_identifier_node) = node.child_by_field_name("attribute") else {
                return None;
            };

            let attribute_name = attribute_identifier_node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");

            match attribute_name {
                "T" => {
                    let Some(object_node) = node.child_by_field_name("object") else {
                        return None;
                    };
                    let mut shape = resolve_shape(object_node, params, text)?;
                    shape.reverse();
                    Some(shape)
                }
                _ => None,
            }
        }
        "binary_operator" => {
            let Some(op) = node.child_by_field_name("operator") else {
                return None;
            };

            let op_text = op
                .utf8_text(&text.as_bytes())
                .expect("Operator has no text");

            let left_node = node.child_by_field_name("left")?;
            let right_node = node.child_by_field_name("right")?;
            let Some(left_shape) = resolve_shape(left_node, params, text) else {
                return None;
            };
            let Some(right_shape) = resolve_shape(right_node, params, text) else {
                return None;
            };

            match op_text {
                "@" => {
                    if &left_shape.last().unwrap() != &right_shape.first().unwrap() {
                        return None;
                    } else {
                        return Some(vec![
                            left_shape.first().unwrap().clone(),
                            right_shape.last().unwrap().clone(),
                        ]);
                    }
                }
                "+" | "-" | "*" | "/" => handle_elementwise_ops(left_shape, right_shape),
                _ => return None,
            }
        }
        _ => return None,
    }
}

fn handle_elementwise_ops(
    left_shape: Vec<String>,
    right_shape: Vec<String>,
) -> Option<Vec<String>> {
    if left_shape.len() != right_shape.len() {
        return None;
    }

    for i in 0..left_shape.len() {
        if left_shape[i] != right_shape[i] {
            return None;
        }
    }

    return Some(left_shape.clone());
}
