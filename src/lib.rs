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

        self.on_hover(&uri, text, &pos).await
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

        for node in typed_functions {
            let Some(node_name) = node.child_by_field_name("name") else {
                continue;
            };

            let Ok(function_name) = node_name.utf8_text(&text.as_bytes()) else {
                continue;
            };

            let mut typed_params: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(node, "typed_parameter", &mut typed_params);
            let mut params = HashMap::new();

            for node in typed_params {
                let Some(node_identifier) = node.child(0) else {
                    return;
                };

                let param_name = node_identifier
                    .utf8_text(text.as_bytes())
                    .expect("Failed to get node identifier");

                let Some(node_type) = node.child_by_field_name("type") else {
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

            let mut binary_operator_nodes: Vec<Node<'_>> = Vec::new();
            find_node_by_kind(node, "binary_operator", &mut binary_operator_nodes);

            for binary_operator in binary_operator_nodes {
                let Some(op) = binary_operator.child_by_field_name("operator") else {
                    continue;
                };

                let op_text = op
                    .utf8_text(&text.as_bytes())
                    .expect("Operator has no text");

                if op_text != "@" {
                    continue;
                }

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

                let left_dims = params
                    .get(left_child_param_name)
                    .expect("Failed to find left child param in params hashmap");
                let right_dims = params
                    .get(right_child_param_name)
                    .expect("Failed to find right child param in params hashmap");

                if left_dims.last().unwrap() != right_dims.first().unwrap() {
                    self.client
                        .log_message(
                            MessageType::ERROR,
                            format!(
                                "can't matmut with these shapes {:?} @ {:?}",
                                left_dims, right_dims
                            ),
                        )
                        .await;

                    let start = binary_operator.start_position();
                    let end = binary_operator.end_position();

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
                            "can't matmul with these shapes {:?} @ {:?}",
                            left_dims, right_dims
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

    async fn on_hover(&self, uri: &Url, text: &str, pos: &Position) -> Result<Option<Hover>> {
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

        let shapes_lock = self.shapes.read().await;

        for shape_fn in shapes_lock.iter() {
            let (_, map) = shape_fn;
            let Some(param) = map.get(name) else {
                continue;
            };
            return Ok(Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(format!(
                    "shape: {:?}",
                    param
                ))),
                range: None,
            }));
        }

        Ok(None)
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
