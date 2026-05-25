use std::collections::HashMap;
use std::path::PathBuf;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidChangeWorkspaceFoldersParams,
    DidOpenTextDocumentParams, Hover, HoverContents, HoverParams, HoverProviderCapability,
    InitializeParams, InitializeResult, InlayHint, InlayHintParams, MarkedString, MessageType,
    OneOf, Position, Range, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    Url, WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tree_sitter::Parser;

use ndim_lsp::{analyze_layer_shapes, ShapeError};

pub struct Backend {
    pub client: Client,
    pub document_text: RwLock<HashMap<Url, String>>,
    pub workspace_roots: RwLock<Vec<PathBuf>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        // Seed workspace roots from the initial workspace folders
        if let Some(folders) = params.workspace_folders {
            let roots: Vec<PathBuf> = folders
                .into_iter()
                .filter_map(|f| f.uri.to_file_path().ok())
                .collect();
            let mut lock = self.workspace_roots.write().await;
            *lock = roots;
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                inlay_hint_provider: Some(OneOf::Left(true)),
                workspace: Some(WorkspaceServerCapabilities {
                    workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Left(true)),
                    }),
                    file_operations: None,
                }),
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

        self.republish_diagnostics(
            &params.text_document.uri,
            &params.text_document.text,
            params.text_document.version,
        )
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.republish_diagnostics(
                &params.text_document.uri,
                &change.text,
                params.text_document.version,
            )
            .await;
        }
    }

    async fn did_change_workspace_folders(&self, params: DidChangeWorkspaceFoldersParams) {
        let mut lock = self.workspace_roots.write().await;

        // Remove folders that were removed
        for removed in &params.event.removed {
            if let Ok(path) = removed.uri.to_file_path() {
                lock.retain(|p| p != &path);
            }
        }

        // Add folders that were added
        for added in params.event.added {
            if let Ok(path) = added.uri.to_file_path()
                && !lock.contains(&path)
            {
                lock.push(path);
            }
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let doc_lock = self.document_text.read().await;
        let Some(text) = doc_lock.get(&uri) else {
            return Ok(None);
        };
        self.on_hover(text, &pos).await
    }

    async fn inlay_hint(&self, _params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let hints = Vec::new();
        Ok(Some(hints))
    }
}

impl Backend {
    async fn republish_diagnostics(&self, uri: &Url, text: &str, version: i32) {
        // Store the document text
        {
            let mut doc_lock = self.document_text.write().await;
            doc_lock.insert(uri.clone(), text.to_string());
        }

        // Parse with tree-sitter-python
        let mut parser = Parser::new();
        if parser.set_language(&tree_sitter_python::LANGUAGE.into()).is_err() {
            self.client
                .log_message(
                    MessageType::ERROR,
                    "failed to set tree-sitter-python language",
                )
                .await;
            self.client
                .publish_diagnostics(uri.clone(), Vec::new(), Some(version))
                .await;
            return;
        }
        let Some(tree) = parser.parse(text, None) else {
            self.client
                .publish_diagnostics(uri.clone(), Vec::new(), Some(version))
                .await;
            return;
        };

        // Use cached workspace roots
        let search_roots = self.workspace_roots.read().await.clone();

        // Read-file implementation
        let read_file = |path: &PathBuf| std::fs::read_to_string(path).ok();

        // Run analysis
        let diagnostics = match analyze_layer_shapes(
            tree.root_node(),
            text,
            &search_roots,
            read_file,
            5,
        ) {
            Ok(analysis) => analysis
                .errors
                .into_iter()
                .map(shape_error_to_diagnostic)
                .collect(),
            Err(message) => {
                self.client
                    .log_message(
                        MessageType::WARNING,
                        format!("analysis failed: {}", message),
                    )
                    .await;
                Vec::new()
            }
        };

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, Some(version))
            .await;
    }

    async fn on_hover(&self, text: &str, pos: &Position) -> Result<Option<Hover>> {
        let mut parser = Parser::new();
        if parser.set_language(&tree_sitter_python::LANGUAGE.into()).is_err() {
            return Ok(None);
        }
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

        // TODO: use `name` for real hover content
        let _ = name;

        Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::String("shape:".to_string())),
            range: None,
        }))
    }
}

fn shape_error_to_diagnostic(error: ShapeError) -> Diagnostic {
    Diagnostic {
        range: ts_range_to_lsp_range(error.range),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("ndim-lsp".to_string()),
        message: format!("{}: {}", error.variable, error.message),
        code: None,
        related_information: None,
        ..Default::default()
    }
}

fn ts_range_to_lsp_range(ts: tree_sitter::Range) -> Range {
    Range {
        start: Position {
            line: ts.start_point.row as u32,
            character: ts.start_point.column as u32,
        },
        end: Position {
            line: ts.end_point.row as u32,
            character: ts.end_point.column as u32,
        },
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        document_text: Default::default(),
        workspace_roots: Default::default(),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_range_to_lsp_range_converts_correctly() {
        let ts = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 5),
            end_point: tree_sitter::Point::new(2, 10),
            start_byte: 0,
            end_byte: 0,
        };
        let lsp = ts_range_to_lsp_range(ts);
        assert_eq!(lsp.start.line, 0);
        assert_eq!(lsp.start.character, 5);
        assert_eq!(lsp.end.line, 2);
        assert_eq!(lsp.end.character, 10);
    }

    #[test]
    fn ts_range_to_lsp_range_zero_based() {
        let ts = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 1),
            start_byte: 0,
            end_byte: 0,
        };
        let lsp = ts_range_to_lsp_range(ts);
        assert_eq!(lsp.start.line, 0);
        assert_eq!(lsp.start.character, 0);
        assert_eq!(lsp.end.line, 0);
        assert_eq!(lsp.end.character, 1);
    }
}
