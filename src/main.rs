use tower_lsp::{LspService, Server};

use core::str;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokio::sync::RwLock;

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InlayHint, InlayHintParams,
    MarkedString, MessageType, OneOf, Position, ServerCapabilities, SignatureInformation,
    TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use tower_lsp::{Client, LanguageServer};
use tree_sitter::{Parser, Tree};

pub struct Backend {
    pub client: Client,
    pub document_text: RwLock<HashMap<Url, String>>,
    pub site_packages_path: RwLock<String>,
    pub global_state: GlobalState,
}

#[derive(Default)]
pub struct GlobalState {
    trees: HashMap<Url, Tree>,
    module_resolution: HashMap<String, Url>,
    signatures: HashMap<Url, HashMap<String, SignatureInformation>>,
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

        let mut py_path = PathBuf::from(".venv/bin/python");

        if !py_path.exists() {
            self.client
                .log_message(
                    MessageType::INFO,
                    "No virtual env found, using system python",
                )
                .await;
            py_path = PathBuf::from("python3");
        }

        let site_packages_res = Command::new(&py_path)
            .args(["-c", "import site; print(site.getsitepackages())"])
            .output()
            .ok();

        if let Some(output) = site_packages_res
            && let Ok(site_packages) = std::str::from_utf8(&output.stdout)
        {
            let trimmed = site_packages.trim().to_string();
            {
                let mut site_packages_lock = self.site_packages_path.write().await;
                *site_packages_lock = trimmed.clone();
            }

            self.client
                .log_message(MessageType::INFO, self.site_packages_path.read().await)
                .await;
        }

        self.on_change(&params.text_document.uri, &params.text_document.text)
            .await
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.on_change(&params.text_document.uri, &change.text)
                .await;
            let mut doc_lock = self.document_text.write().await;
            doc_lock.insert(params.text_document.uri, change.text);
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
        let mut hints = Vec::new();

        Ok(Some(hints))
    }
}

impl Backend {
    async fn on_change(&self, uri: &Url, text: &str) {
        self.client
            .log_message(MessageType::INFO, format!("changed uri {}", uri))
            .await;
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

        Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::String(format!("shape:",))),
            range: None,
        }))
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        document_text: Default::default(),
        site_packages_path: Default::default(),
        global_state: Default::default(),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}
