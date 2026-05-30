use std::collections::{HashMap, HashSet};
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

        // Use cached workspace roots, extended with discovered site-packages
        let workspace_roots = self.workspace_roots.read().await.clone();
        let mut search_roots = workspace_roots.clone();
        search_roots.extend(python_site_packages_roots(&workspace_roots));

        // Read-file implementation
        let read_file = |path: &PathBuf| std::fs::read_to_string(path).ok();

        // Run analysis
        let diagnostics = match analyze_layer_shapes(
            tree.root_node(),
            text,
            &search_roots,
            read_file,
            8,
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

        // Only hover on identifiers (skip keywords, whitespace, etc.)
        if node.kind() != "identifier" {
            return Ok(None);
        }

        let Ok(var_name) = node.utf8_text(text.as_bytes()) else {
            return Ok(None);
        };

        // Compute cursor byte from the identifier node's start byte
        let cursor_byte = node.start_byte();

        // Use cached workspace roots for analysis, extended with discovered site-packages
        let workspace_roots = self.workspace_roots.read().await.clone();
        let mut search_roots = workspace_roots.clone();
        search_roots.extend(python_site_packages_roots(&workspace_roots));
        let read_file = |path: &PathBuf| std::fs::read_to_string(path).ok();

        let analysis = match analyze_layer_shapes(root, text, &search_roots, read_file, 8) {
            Ok(a) => a,
            Err(msg) => {
                self.client
                    .log_message(MessageType::WARNING, format!("hover analysis failed: {}", msg))
                    .await;
                return Ok(None);
            }
        };

        // Walk from innermost scope outward, looking for the variable
        let shape = match find_shape_for_variable(&analysis.scopes, cursor_byte, var_name) {
            Some(s) => s,
            None => return Ok(None),
        };

        let hover_content = format_hover(var_name, &shape);
        let hover_range = ts_range_to_lsp_range(node.range());

        Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::LanguageString(
                tower_lsp::lsp_types::LanguageString {
                    language: "python".into(),
                    value: hover_content,
                },
            )),
            range: Some(hover_range),
        }))
    }
}

/// Walk from the innermost scope containing `cursor_byte` outward until a
/// shape entry for `var_name` is found, then return a clone of it.
fn find_shape_for_variable(
    scopes: &[ndim_lsp::FunctionShapeScope],
    cursor_byte: usize,
    var_name: &str,
) -> Option<Vec<String>> {
    // Collect all enclosing scope indices, ordered innermost-first.
    let mut enclosing: Vec<usize> = Vec::new();
    for (i, scope) in scopes.iter().enumerate() {
        if scope.start_byte <= cursor_byte && cursor_byte < scope.end_byte {
            enclosing.push(i);
        }
    }
    // Sort innermost-first: smallest byte span first, tie-break by later index
    // (mirrors scope_index_for_byte logic).
    enclosing.sort_by(|&a, &b| {
        let size_a = scopes[a].end_byte - scopes[a].start_byte;
        let size_b = scopes[b].end_byte - scopes[b].start_byte;
        size_a.cmp(&size_b).then_with(|| b.cmp(&a))
    });

    for idx in enclosing {
        if let Some(shape) = scopes[idx].shapes.get(var_name) {
            return Some(shape.clone());
        }
    }
    None
}

/// Format the variable name and shape into a Python-annotated string.
/// Example: `x: Float[Array, "batch features"]`
/// Scalar (zero-rank) shapes render as `x: Scalar`.
fn format_hover(var_name: &str, shape: &[String]) -> String {
    if shape.is_empty() {
        format!("{}: Scalar", var_name)
    } else {
        let dims = shape.join(" ");
        format!("{}: Float[Array, \"{}\"]", var_name, dims)
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

/// Statically discover Python `site-packages` directories without invoking
/// Python or shelling out. Probes (in priority order):
///
/// 1. `$VIRTUAL_ENV`
/// 2. `$CONDA_PREFIX`
/// 3. `<workspace>/.venv` and `<workspace>/venv` for each workspace root
///
/// For each candidate venv root, looks for `lib/python*/site-packages`
/// (Unix/macOS) and `Lib/site-packages` (Windows). Only existing paths are
/// returned. Duplicates are collapsed by canonical path so a symlinked
/// `.venv` does not double-count against its target. Never panics; IO errors
/// are silently ignored.
fn python_site_packages_roots(workspace_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    let mut probe_venv = |venv_root: PathBuf| {
        // Unix/macOS layout: <venv>/lib/python*/site-packages
        let lib = venv_root.join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with("python") {
                    let site = entry.path().join("site-packages");
                    if site.exists() {
                        candidates.push(site);
                    }
                }
            }
        }
        // Windows layout: <venv>/Lib/site-packages
        let win = venv_root.join("Lib").join("site-packages");
        if win.exists() {
            candidates.push(win);
        }
    };

    if let Ok(v) = std::env::var("VIRTUAL_ENV")
        && !v.is_empty()
    {
        probe_venv(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("CONDA_PREFIX")
        && !v.is_empty()
    {
        probe_venv(PathBuf::from(v));
    }
    for root in workspace_roots {
        probe_venv(root.join(".venv"));
        probe_venv(root.join("venv"));
    }

    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut result: Vec<PathBuf> = Vec::new();
    for c in candidates {
        let key = std::fs::canonicalize(&c).unwrap_or_else(|_| c.clone());
        if seen.insert(key) {
            result.push(c);
        }
    }
    result
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

    #[test]
    fn format_hover_basic() {
        assert_eq!(
            format_hover("x", &["batch".into(), "features".into()]),
            "x: Float[Array, \"batch features\"]"
        );
    }

    #[test]
    fn format_hover_single_dim() {
        assert_eq!(
            format_hover("vec", &["n".into()]),
            "vec: Float[Array, \"n\"]"
        );
    }

    #[test]
    fn format_hover_scalar() {
        assert_eq!(format_hover("s", &[]), "s: Scalar");
    }

    #[test]
    fn find_shape_innermost_scope_wins() {
        use ndim_lsp::FunctionShapeScope;
        use std::collections::HashMap;

        let mut outer_shapes: HashMap<String, Vec<String>> = HashMap::new();
        outer_shapes.insert("x".into(), vec!["outer_dim".into()]);

        let mut inner_shapes: HashMap<String, Vec<String>> = HashMap::new();
        inner_shapes.insert("x".into(), vec!["inner_dim".into()]);

        let scopes = vec![
            FunctionShapeScope {
                function_name: None,
                start_byte: 0,
                end_byte: 200,
                shapes: outer_shapes,
                return_shape: None,
                param_order: Vec::new(),
            },
            FunctionShapeScope {
                function_name: Some("foo".into()),
                start_byte: 20,
                end_byte: 180,
                shapes: inner_shapes,
                return_shape: None,
                param_order: Vec::new(),
            },
        ];

        // Cursor at byte 50 falls in both scopes; inner should win
        let result = find_shape_for_variable(&scopes, 50, "x");
        assert_eq!(result, Some(vec!["inner_dim".into()]));
    }

    #[test]
    fn find_shape_falls_back_to_outer() {
        use ndim_lsp::FunctionShapeScope;
        use std::collections::HashMap;

        let mut outer_shapes: HashMap<String, Vec<String>> = HashMap::new();
        outer_shapes.insert("y".into(), vec!["outer".into()]);

        let inner_shapes: HashMap<String, Vec<String>> = HashMap::new();

        let scopes = vec![
            FunctionShapeScope {
                function_name: None,
                start_byte: 0,
                end_byte: 200,
                shapes: outer_shapes,
                return_shape: None,
                param_order: Vec::new(),
            },
            FunctionShapeScope {
                function_name: Some("foo".into()),
                start_byte: 20,
                end_byte: 180,
                shapes: inner_shapes,
                return_shape: None,
                param_order: Vec::new(),
            },
        ];

        // "y" is only in the outer scope
        let result = find_shape_for_variable(&scopes, 50, "y");
        assert_eq!(result, Some(vec!["outer".into()]));
    }

    #[test]
    fn find_shape_not_found() {
        use ndim_lsp::FunctionShapeScope;
        use std::collections::HashMap;

        let scopes = vec![FunctionShapeScope {
            function_name: None,
            start_byte: 0,
            end_byte: 100,
            shapes: HashMap::new(),
            return_shape: None,
            param_order: Vec::new(),
        }];

        let result = find_shape_for_variable(&scopes, 50, "z");
        assert_eq!(result, None);
    }

    /// Drop guard that saves an environment variable's value and restores it
    /// on drop, so env-mutating subcases inside a single test don't leak state.
    struct EnvGuard {
        var: &'static str,
        prev: Option<String>,
    }
    impl EnvGuard {
        fn save_and_clear(var: &'static str) -> Self {
            let prev = std::env::var(var).ok();
            if prev.is_some() {
                unsafe {
                    std::env::remove_var(var);
                }
            }
            Self { var, prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.var, v),
                    None => std::env::remove_var(self.var),
                }
            }
        }
    }

    fn make_site_packages(base: &std::path::Path, py_dir: &str) -> PathBuf {
        let site = base.join("lib").join(py_dir).join("site-packages");
        std::fs::create_dir_all(&site).unwrap();
        site
    }

    fn make_win_site_packages(base: &std::path::Path) -> PathBuf {
        let site = base.join("Lib").join("site-packages");
        std::fs::create_dir_all(&site).unwrap();
        site
    }

    /// All discovery cases bundled into a single `#[test]` so they
    /// run sequentially — they share global env-var state, which would
    /// otherwise race under cargo's parallel test runner. EnvGuard restores
    /// any prior `VIRTUAL_ENV` / `CONDA_PREFIX` even on panic.
    #[test]
    fn python_site_packages_discovery_cases() {
        // Save and clear both env vars for a known baseline.
        let _ve = EnvGuard::save_and_clear("VIRTUAL_ENV");
        let _ce = EnvGuard::save_and_clear("CONDA_PREFIX");

        // Case: empty workspace, no env vars -> empty.
        {
            let tmp = tempfile::tempdir().unwrap();
            let roots = vec![tmp.path().to_path_buf()];
            assert!(python_site_packages_roots(&roots).is_empty());
        }

        // Case: workspace contains .venv/lib/python3.11/site-packages.
        {
            let tmp = tempfile::tempdir().unwrap();
            let site = make_site_packages(&tmp.path().join(".venv"), "python3.11");
            let roots = vec![tmp.path().to_path_buf()];
            let got = python_site_packages_roots(&roots);
            assert_eq!(got.len(), 1);
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&site).unwrap()
            );
        }

        // Case: workspace contains venv/lib/python3.12/site-packages.
        {
            let tmp = tempfile::tempdir().unwrap();
            let site = make_site_packages(&tmp.path().join("venv"), "python3.12");
            let roots = vec![tmp.path().to_path_buf()];
            let got = python_site_packages_roots(&roots);
            assert_eq!(got.len(), 1);
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&site).unwrap()
            );
        }

        // Case: workspace contains .venv/Lib/site-packages (Windows-style).
        {
            let tmp = tempfile::tempdir().unwrap();
            let site = make_win_site_packages(&tmp.path().join(".venv"));
            let roots = vec![tmp.path().to_path_buf()];
            let got = python_site_packages_roots(&roots);
            assert_eq!(got.len(), 1);
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&site).unwrap()
            );
        }

        // Case: VIRTUAL_ENV points at a tempdir with lib/python3.12/site-packages.
        {
            let tmp = tempfile::tempdir().unwrap();
            let site = make_site_packages(tmp.path(), "python3.12");
            unsafe {
                std::env::set_var("VIRTUAL_ENV", tmp.path());
            }
            let got = python_site_packages_roots(&[]);
            unsafe {
                std::env::remove_var("VIRTUAL_ENV");
            }
            assert_eq!(got.len(), 1);
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&site).unwrap()
            );
        }

        // Case: CONDA_PREFIX points at a tempdir with lib/python3.10/site-packages.
        {
            let tmp = tempfile::tempdir().unwrap();
            let site = make_site_packages(tmp.path(), "python3.10");
            unsafe {
                std::env::set_var("CONDA_PREFIX", tmp.path());
            }
            let got = python_site_packages_roots(&[]);
            unsafe {
                std::env::remove_var("CONDA_PREFIX");
            }
            assert_eq!(got.len(), 1);
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&site).unwrap()
            );
        }

        // Case: VIRTUAL_ENV set AND workspace has .venv (different paths) ->
        // both returned, no duplicates.
        {
            let venv_tmp = tempfile::tempdir().unwrap();
            let ws_tmp = tempfile::tempdir().unwrap();
            let venv_site = make_site_packages(venv_tmp.path(), "python3.11");
            let ws_site = make_site_packages(&ws_tmp.path().join(".venv"), "python3.11");
            unsafe {
                std::env::set_var("VIRTUAL_ENV", venv_tmp.path());
            }
            let got = python_site_packages_roots(&[ws_tmp.path().to_path_buf()]);
            unsafe {
                std::env::remove_var("VIRTUAL_ENV");
            }
            assert_eq!(got.len(), 2);
            let canon: HashSet<PathBuf> = got
                .iter()
                .map(|p| std::fs::canonicalize(p).unwrap())
                .collect();
            assert!(canon.contains(&std::fs::canonicalize(&venv_site).unwrap()));
            assert!(canon.contains(&std::fs::canonicalize(&ws_site).unwrap()));
        }

        // Case: nonexistent candidate paths are filtered (workspace has no
        // .venv or venv -> empty).
        {
            let tmp = tempfile::tempdir().unwrap();
            let roots = vec![tmp.path().to_path_buf()];
            assert!(python_site_packages_roots(&roots).is_empty());
        }

        // Case: symlinked .venv -> a real venv dir; both candidate paths
        // would otherwise appear, but they canonicalize to the same target.
        #[cfg(unix)]
        {
            let real_tmp = tempfile::tempdir().unwrap();
            let real_site = make_site_packages(real_tmp.path(), "python3.11");

            let ws_tmp = tempfile::tempdir().unwrap();
            // Make ws/.venv a symlink to real_tmp.
            std::os::unix::fs::symlink(real_tmp.path(), ws_tmp.path().join(".venv")).unwrap();

            unsafe {
                std::env::set_var("VIRTUAL_ENV", real_tmp.path());
            }
            let got = python_site_packages_roots(&[ws_tmp.path().to_path_buf()]);
            unsafe {
                std::env::remove_var("VIRTUAL_ENV");
            }
            // Both probes find real_site (once directly, once via symlink),
            // but canonicalization should collapse them to one entry.
            assert_eq!(
                got.len(),
                1,
                "expected symlinked .venv to be deduped against VIRTUAL_ENV target, got {:?}",
                got
            );
            assert_eq!(
                std::fs::canonicalize(&got[0]).unwrap(),
                std::fs::canonicalize(&real_site).unwrap()
            );
        }
    }

    #[test]
    fn position_to_byte_via_tree_sitter() {
        // Verify that converting an LSP Position through tree-sitter's
        // descendant_for_point_range gives the correct byte offset.
        let text = "hello\nworld\n";
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(text, None).unwrap();
        let root = tree.root_node();

        // Point at line 1, column 0 should land on 'w' which is byte 6
        let point = tree_sitter::Point::new(1, 0);
        let node = root.descendant_for_point_range(point, point).unwrap();
        // The node should start at byte 6 (after "hello\n")
        assert_eq!(node.start_byte(), 6);
    }
}
