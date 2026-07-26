use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticRelatedInformation, DiagnosticSeverity, DidChangeTextDocumentParams,
    DidChangeWorkspaceFoldersParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams, InlayHintTooltip, Location,
    MarkupContent, MarkupKind, MessageType, OneOf,
    Position, PositionEncodingKind, Range, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Url,
    WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tree_sitter::{Parser, Tree};

use ndim_lsp::{
    LayerShapeAnalysis, ResolutionCache, ShapeError, ShapeErrorKind, analyze_layer_shapes,
    clear_resolution_cache, new_resolution_cache,
};

/// Analysis plus the exact text/tree it was computed from, so hover and
/// inlay hints reuse the parse instead of re-parsing per request.
#[derive(Clone)]
pub struct CachedAnalysis {
    pub analysis: Arc<LayerShapeAnalysis>,
    pub text: Arc<str>,
    pub tree: Arc<Tree>,
}

pub struct Backend {
    pub client: Client,
    pub document_text: RwLock<HashMap<Url, String>>,
    pub workspace_roots: RwLock<Vec<PathBuf>>,
    /// Cache: URI → (version, analysis+snapshot). Invalidated on text change.
    /// Stored as `Arc`s so cache hits clone pointers, not the whole analysis.
    pub analysis_cache: RwLock<HashMap<Url, (i32, CachedAnalysis)>>,
    /// Current version for each URI (set on did_open/did_change).
    pub document_version: RwLock<HashMap<Url, i32>>,
    /// Session-lifetime cache for resolved import targets.
    /// Keyed on (import-path-segments, search-roots-fingerprint).
    /// Invalidated when workspace folders change.
    pub resolution_cache: Arc<ResolutionCache>,
    /// Severity for `ShapeErrorKind::Mismatch` diagnostics, set from
    /// `initializationOptions.diagnostic_severity`.
    pub diagnostic_severity: RwLock<DiagnosticSeverity>,
    /// Severity for `ShapeErrorKind::Approximation` diagnostics, set from
    /// `initializationOptions.approximation_severity` (defaults to WARNING,
    /// independent of `diagnostic_severity`).
    pub approximation_severity: RwLock<DiagnosticSeverity>,
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

        if let Some(opts) = params.initialization_options.as_ref() {
            *self.diagnostic_severity.write().await =
                parse_severity(opts, "diagnostic_severity", DiagnosticSeverity::ERROR);
            *self.approximation_severity.write().await =
                parse_severity(opts, "approximation_severity", DiagnosticSeverity::WARNING);
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                position_encoding: Some(PositionEncodingKind::UTF16),
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
        let Some(change) = params.content_changes.into_iter().last() else {
            return;
        };
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        // Record the new text immediately so hover/inlay requests see it...
        {
            let mut doc_lock = self.document_text.write().await;
            doc_lock.insert(uri.clone(), change.text);
        }
        {
            let mut ver_lock = self.document_version.write().await;
            ver_lock.insert(uri.clone(), version);
        }
        // ...but debounce the analysis: if another keystroke lands during the
        // window, this run bows out and the newer one does the work.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        {
            let ver_lock = self.document_version.read().await;
            if ver_lock.get(&uri).copied() != Some(version) {
                return;
            }
        }
        self.publish_diagnostics_for(&uri, version).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = &params.text_document.uri;
        // Evict all per-document state so long-lived sessions don't leak.
        self.document_text.write().await.remove(uri);
        self.document_version.write().await.remove(uri);
        self.analysis_cache.write().await.remove(uri);
        // Clear any diagnostics the client may still be showing.
        self.client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
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

        // Workspace roots changed → site-packages may have shifted.
        // Clear the resolution cache so stale entries don't survive.
        clear_resolution_cache(&self.resolution_cache);
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        // Verify doc exists
        let doc_lock = self.document_text.read().await;
        if !doc_lock.contains_key(&uri) {
            return Ok(None);
        }
        drop(doc_lock);
        self.on_hover(&uri, &pos).await
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = params.text_document.uri;
        let visible_range = params.range;
        // Verify doc exists
        let doc_lock = self.document_text.read().await;
        if !doc_lock.contains_key(&uri) {
            return Ok(None);
        }
        drop(doc_lock);
        Ok(self.compute_inlay_hints(&uri, &visible_range).await)
    }
}

impl Backend {
    /// Run full analysis, or return cached result if version matches.
    /// Populates the analysis cache on miss. Caller passes the version it
    /// believes is current; if the document advances during analysis the
    /// freshly-computed entry is dropped instead of poisoning a newer cached
    /// result.
    async fn get_analysis(&self, uri: &Url, version: i32) -> Option<CachedAnalysis> {
        // Check cache hit first (read-only lock, cheap)
        {
            let cache = self.analysis_cache.read().await;
            if let Some((cached_ver, cached)) = cache.get(uri)
                && *cached_ver == version
            {
                return Some(cached.clone());
            }
        }

        // Cache miss — run full analysis. Read text + version together so the
        // value we cache can't disagree with the version it's keyed on.
        let t_total = Instant::now();
        let (text, snapshot_version) = {
            let doc_lock = self.document_text.read().await;
            let ver_lock = self.document_version.read().await;
            let text = doc_lock.get(uri)?.clone();
            let snapshot_version = ver_lock.get(uri).copied()?;
            (text, snapshot_version)
        };
        if snapshot_version != version {
            // Document moved while we were waiting. Let the caller re-request
            // with the current version rather than analyzing stale text.
            return None;
        }
        let text_bytes = text.len();
        let text_lines = text.lines().count();

        let mut parser = Parser::new();
        if parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .is_err()
        {
            return None;
        }
        let t_parse = Instant::now();
        let tree = parser.parse(&text, None)?;
        let parse_ms = t_parse.elapsed().as_millis();

        let t_roots = Instant::now();
        let workspace_roots = self.workspace_roots.read().await.clone();
        let mut search_roots = workspace_roots.clone();
        search_roots.extend(python_site_packages_roots(&workspace_roots));
        let roots_ms = t_roots.elapsed().as_millis();
        let search_roots_count = search_roots.len();

        let read_count = AtomicUsize::new(0);
        let read_bytes = AtomicUsize::new(0);
        let read_file = |path: &PathBuf| {
            let result = std::fs::read_to_string(path).ok();
            if let Some(ref s) = result {
                read_count.fetch_add(1, Ordering::Relaxed);
                read_bytes.fetch_add(s.len(), Ordering::Relaxed);
            }
            result
        };

        let t_analyze = Instant::now();
        let analysis = match analyze_layer_shapes(
            tree.root_node(),
            &text,
            &search_roots,
            read_file,
            8,
            Some(&self.resolution_cache),
        ) {
            Ok(a) => a,
            Err(msg) => {
                self.client
                    .log_message(MessageType::WARNING, format!("analysis failed: {}", msg))
                    .await;
                return None;
            }
        };
        let analyze_ms = t_analyze.elapsed().as_millis();
        let total_ms = t_total.elapsed().as_millis();

        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "analysis: total={}ms parse={}ms roots={}ms analyze={}ms \
                     | text={}B/{}L roots={} reads={}/{}B res_cache_entries={} hits={} misses={}",
                    total_ms,
                    parse_ms,
                    roots_ms,
                    analyze_ms,
                    text_bytes,
                    text_lines,
                    search_roots_count,
                    read_count.load(Ordering::Relaxed),
                    read_bytes.load(Ordering::Relaxed),
                    self.resolution_cache.map.read().unwrap().len(),
                    self.resolution_cache.hits.load(Ordering::Relaxed),
                    self.resolution_cache.misses.load(Ordering::Relaxed),
                ),
            )
            .await;

        let cached = CachedAnalysis {
            analysis: Arc::new(analysis),
            text: Arc::from(text),
            tree: Arc::new(tree),
        };
        // Only publish to the cache if the document hasn't advanced past the
        // version we analyzed. Otherwise we'd overwrite a (potentially newer)
        // entry with stale results.
        {
            let ver_lock = self.document_version.read().await;
            let current = ver_lock.get(uri).copied();
            drop(ver_lock);
            if current == Some(version) {
                let mut cache = self.analysis_cache.write().await;
                cache.insert(uri.clone(), (version, cached.clone()));
            }
        }

        Some(cached)
    }

    async fn republish_diagnostics(&self, uri: &Url, text: &str, version: i32) {
        // Store the document text and version. LSP versions are monotonic
        // per did_change, so any existing cache entry for this URI will be
        // at a strictly older version — `get_analysis` misses on the version
        // check and overwrites, no explicit invalidation needed.
        {
            let mut doc_lock = self.document_text.write().await;
            doc_lock.insert(uri.clone(), text.to_string());
        }
        {
            let mut ver_lock = self.document_version.write().await;
            ver_lock.insert(uri.clone(), version);
        }
        self.publish_diagnostics_for(uri, version).await;
    }

    /// Analyze (or hit the cache) and publish diagnostics for `version`.
    async fn publish_diagnostics_for(&self, uri: &Url, version: i32) {
        let Some(cached) = self.get_analysis(uri, version).await else {
            self.client
                .publish_diagnostics(uri.clone(), Vec::new(), Some(version))
                .await;
            return;
        };

        let mismatch_severity = *self.diagnostic_severity.read().await;
        let approximation_severity = *self.approximation_severity.read().await;
        let diagnostics: Vec<Diagnostic> = cached
            .analysis
            .errors
            .iter()
            .cloned()
            .map(|e| shape_error_to_diagnostic(e, uri, mismatch_severity, approximation_severity))
            .collect();

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, Some(version))
            .await;
    }

    async fn compute_inlay_hints(
        &self,
        uri: &Url,
        visible_range: &Range,
    ) -> Option<Vec<InlayHint>> {
        let t_total = Instant::now();

        let version = {
            let ver_lock = self.document_version.read().await;
            ver_lock.get(uri).copied()?
        };

        // Snapshot cache state *before* `get_analysis` populates it on miss,
        // so the telemetry below actually distinguishes hits from misses.
        let was_cached = {
            let cache = self.analysis_cache.read().await;
            cache.get(uri).is_some_and(|(v, _)| *v == version)
        };

        let cached = self.get_analysis(uri, version).await?;
        let analysis = &cached.analysis;

        // Line lengths (UTF-16) for the visible window, computed in one pass
        // over the snapshot text the analysis was built from.
        let line_lens =
            line_lengths_in_range(&cached.text, visible_range.start.line, visible_range.end.line);

        let t_build = Instant::now();
        let mut hints: Vec<InlayHint> = Vec::new();
        let mut seen: HashSet<(u32, String)> = HashSet::new();

        // One hint per (non-annotated) assignment site, in source order, so a
        // reassigned variable shows its shape at each line (issue #28).
        for rec in &analysis.assignment_shapes {
            if rec.name.starts_with('_') {
                continue;
            }
            let line = rec.line;

            if !seen.insert((line, rec.name.clone())) {
                continue;
            }

            if line < visible_range.start.line || line > visible_range.end.line {
                continue;
            }

            let label = if rec.shape.is_empty() {
                ": Scalar".to_string()
            } else {
                format!(": [{}]", rec.shape.join(", "))
            };
            let char_pos = line_lens.get(&line).copied().unwrap_or(0);
            // Same text hover shows for this variable (e.g. `x: Float[Array,
            // "batch features"]"`), so both surfaces stay consistent.
            let tooltip = InlayHintTooltip::String(format_hover(&rec.name, &rec.shape));

            hints.push(InlayHint {
                position: Position {
                    line,
                    character: char_pos,
                },
                label: InlayHintLabel::String(label),
                kind: Some(InlayHintKind::TYPE),
                text_edits: None,
                tooltip: Some(tooltip),
                padding_left: None,
                padding_right: Some(true),
                data: None,
            });
        }

        let build_ms = t_build.elapsed().as_millis();
        let total_ms = t_total.elapsed().as_millis();
        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "inlay_hint: total={}ms build={}ms cached={} \
                     | hints={} scopes={} visible={}..{}",
                    total_ms,
                    build_ms,
                    was_cached,
                    hints.len(),
                    analysis.scopes.len(),
                    visible_range.start.line,
                    visible_range.end.line,
                ),
            )
            .await;

        Some(hints)
    }

    async fn on_hover(&self, uri: &Url, pos: &Position) -> Result<Option<Hover>> {
        let t_total = Instant::now();

        let version = {
            let ver_lock = self.document_version.read().await;
            ver_lock.get(uri).copied()
        };

        // If no version tracked, doc might not be open — fall back
        let Some(version) = version else {
            return Ok(None);
        };

        // Snapshot cache state *before* `get_analysis` may populate it on
        // miss, so telemetry distinguishes hits from misses.
        let was_cached = {
            let cache = self.analysis_cache.read().await;
            cache.get(uri).is_some_and(|(v, _)| *v == version)
        };

        // The cached snapshot carries the text and tree the analysis was
        // computed from, so `cursor_byte` and `analysis.scopes` byte ranges
        // are consistent by construction — no re-parse per hover.
        let cached = match self.get_analysis(uri, version).await {
            Some(c) => c,
            None => return Ok(None),
        };
        let root = cached.tree.root_node();
        let point = tree_sitter::Point::new(pos.line as usize, pos.character as usize);
        let Some(node) = root.descendant_for_point_range(point, point) else {
            return Ok(None);
        };
        if node.kind() != "identifier" {
            return Ok(None);
        }
        let var_name = match node.utf8_text(cached.text.as_bytes()) {
            Ok(v) => v.to_string(),
            Err(_) => return Ok(None),
        };
        let cursor_byte = node.start_byte();

        // If the document advanced while we were waiting on `get_analysis`,
        // `cursor_byte` points into stale text and `analysis.scopes` byte
        // ranges may have moved. Bail; the editor will re-request.
        {
            let ver_lock = self.document_version.read().await;
            if ver_lock.get(uri).copied() != Some(version) {
                return Ok(None);
            }
        }

        let total_ms = t_total.elapsed().as_millis();
        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "hover '{}': total={}ms cached={}",
                    var_name, total_ms, was_cached,
                ),
            )
            .await;

        // Walk from innermost scope outward, looking for the variable
        let shape = match find_shape_for_variable(&cached.analysis.scopes, cursor_byte, &var_name) {
            Some(s) => s,
            None => return Ok(None),
        };

        let hover_content = format_hover(&var_name, &shape);
        let hover_range = ts_range_to_lsp_range(node.range());

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("```python\n{}\n```", hover_content),
            }),
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


/// UTF-16 code-unit counts (LSP `Position.character` units) for each 0-based
/// line in `start..=end`, computed in a single pass over `text`. Lines past
/// the end of the text are simply absent from the map.
fn line_lengths_in_range(text: &str, start: u32, end: u32) -> HashMap<u32, u32> {
    text.lines()
        .enumerate()
        .skip(start as usize)
        .take((end as usize).saturating_sub(start as usize) + 1)
        .map(|(i, l)| (i as u32, l.encode_utf16().count() as u32))
        .collect()
}

/// Map an `initializationOptions` value at `key` to a diagnostic severity.
/// Unknown / missing / malformed -> `default`. Used for both
/// `diagnostic_severity` (Mismatch errors, defaults to ERROR) and
/// `approximation_severity` (Approximation errors, defaults to WARNING) —
/// the two are independent settings, not one derived from the other.
fn parse_severity(
    opts: &serde_json::Value,
    key: &str,
    default: DiagnosticSeverity,
) -> DiagnosticSeverity {
    match opts
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("error") => DiagnosticSeverity::ERROR,
        Some("warning") => DiagnosticSeverity::WARNING,
        Some("information") => DiagnosticSeverity::INFORMATION,
        Some("hint") => DiagnosticSeverity::HINT,
        _ => default,
    }
}

/// Resolve a `ShapeError` to a diagnostic, picking severity by `error.kind`:
/// `Mismatch` uses `mismatch_severity` (`initializationOptions.diagnostic_severity`,
/// defaults to ERROR); `Approximation` uses `approximation_severity`
/// (`initializationOptions.approximation_severity`, defaults to WARNING).
/// `error.related` (e.g. the other operand's node range in a binary-op
/// mismatch) becomes a single-entry `related_information`, anchored at
/// `uri` (the document the diagnostic itself belongs to — `ShapeError`'s
/// range doesn't carry a URI of its own).
fn shape_error_to_diagnostic(
    error: ShapeError,
    uri: &Url,
    mismatch_severity: DiagnosticSeverity,
    approximation_severity: DiagnosticSeverity,
) -> Diagnostic {
    let severity = match error.kind {
        ShapeErrorKind::Mismatch => mismatch_severity,
        ShapeErrorKind::Approximation => approximation_severity,
    };
    let related_information = error.related.map(|(range, message)| {
        vec![DiagnosticRelatedInformation {
            location: Location {
                uri: uri.clone(),
                range: ts_range_to_lsp_range(range),
            },
            message,
        }]
    });
    Diagnostic {
        range: ts_range_to_lsp_range(error.range),
        severity: Some(severity),
        source: Some("ndim-lsp".to_string()),
        message: if error.variable.is_empty() {
            error.message
        } else {
            format!("{}: {}", error.variable, error.message)
        },
        code: None,
        related_information,
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
        analysis_cache: Default::default(),
        document_version: Default::default(),
        resolution_cache: new_resolution_cache(),
        diagnostic_severity: RwLock::new(DiagnosticSeverity::ERROR),
        approximation_severity: RwLock::new(DiagnosticSeverity::WARNING),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_severity_maps_values() {
        use serde_json::json;
        let key = "diagnostic_severity";
        let default = DiagnosticSeverity::ERROR;
        assert_eq!(
            parse_severity(&json!({"diagnostic_severity": "warning"}), key, default),
            DiagnosticSeverity::WARNING
        );
        assert_eq!(
            parse_severity(&json!({"diagnostic_severity": "ERROR"}), key, default),
            DiagnosticSeverity::ERROR
        );
        assert_eq!(
            parse_severity(&json!({"diagnostic_severity": "information"}), key, default),
            DiagnosticSeverity::INFORMATION
        );
        assert_eq!(
            parse_severity(&json!({"diagnostic_severity": "hint"}), key, default),
            DiagnosticSeverity::HINT
        );
        assert_eq!(parse_severity(&json!({}), key, default), DiagnosticSeverity::ERROR);
        assert_eq!(
            parse_severity(&json!({"diagnostic_severity": "garbage"}), key, default),
            DiagnosticSeverity::ERROR
        );
    }

    #[test]
    fn parse_severity_reads_approximation_key_with_its_own_default() {
        use serde_json::json;
        let key = "approximation_severity";
        let default = DiagnosticSeverity::WARNING;
        // Missing/malformed falls back to the *approximation* default
        // (WARNING), independent of the mismatch default (ERROR).
        assert_eq!(parse_severity(&json!({}), key, default), DiagnosticSeverity::WARNING);
        assert_eq!(
            parse_severity(&json!({"approximation_severity": "hint"}), key, default),
            DiagnosticSeverity::HINT
        );
        assert_eq!(
            parse_severity(&json!({"approximation_severity": "error"}), key, default),
            DiagnosticSeverity::ERROR
        );
        // A `diagnostic_severity` key present alongside doesn't leak in.
        assert_eq!(
            parse_severity(
                &json!({"diagnostic_severity": "error", "approximation_severity": "information"}),
                key,
                default
            ),
            DiagnosticSeverity::INFORMATION
        );
    }

    #[test]
    fn shape_error_to_diagnostic_uses_kind_specific_severity() {
        let range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 1),
            start_byte: 0,
            end_byte: 1,
        };
        let uri = Url::parse("file:///test.py").unwrap();
        let mismatch = ShapeError::mismatch("x", "bad shape", range);
        let approximation = ShapeError::approximation("y", "approx shape", range);

        let diag = shape_error_to_diagnostic(
            mismatch,
            &uri,
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));

        let diag = shape_error_to_diagnostic(
            approximation,
            &uri,
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));

        // Approximation severity is independent of a custom mismatch severity.
        let approximation2 = ShapeError::approximation("z", "approx shape", range);
        let diag = shape_error_to_diagnostic(
            approximation2,
            &uri,
            DiagnosticSeverity::HINT,
            DiagnosticSeverity::INFORMATION,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::INFORMATION));
    }

    #[test]
    fn shape_error_to_diagnostic_maps_related_information() {
        let range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 1),
            start_byte: 0,
            end_byte: 1,
        };
        let related_range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(1, 2),
            end_point: tree_sitter::Point::new(1, 5),
            start_byte: 10,
            end_byte: 13,
        };
        let uri = Url::parse("file:///test.py").unwrap();
        let error = ShapeError::mismatch("y", "matmul dimension mismatch", range)
            .with_related(related_range, "other operand `b`: shape [n, d]");

        let diag = shape_error_to_diagnostic(
            error,
            &uri,
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );

        let related = diag.related_information.expect("related_information");
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].location.uri, uri);
        assert_eq!(related[0].location.range, ts_range_to_lsp_range(related_range));
        assert_eq!(related[0].message, "other operand `b`: shape [n, d]");
    }

    #[test]
    fn shape_error_to_diagnostic_no_related_when_absent() {
        let range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 1),
            start_byte: 0,
            end_byte: 1,
        };
        let uri = Url::parse("file:///test.py").unwrap();
        let error = ShapeError::mismatch("x", "bad shape", range);

        let diag = shape_error_to_diagnostic(
            error,
            &uri,
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );

        assert!(diag.related_information.is_none());
    }

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

    #[test]
    fn line_length_basic() {
        let text = "x = 1\ny = 2\n";
        let lens = line_lengths_in_range(text, 0, 1);
        assert_eq!(lens.get(&0).copied(), Some(5));
        assert_eq!(lens.get(&1).copied(), Some(5));
    }

    #[test]
    fn line_length_utf16() {
        // α is U+03B1 → 2 UTF-16 code units. "α = 1" → 2+3 = 5 code units, but 5+2=7 bytes.
        let text = "\u{03b1} = 1\n";
        let lens = line_lengths_in_range(text, 0, 0);
        assert_eq!(lens.get(&0).copied(), Some(5)); // UTF-16 code units, not bytes
        assert_ne!(
            lens[&0] as usize,
            text.lines().next().unwrap().len()
        ); // bytes ≠ code units
    }

    #[test]
    fn line_length_empty_line() {
        let text = "\n\n";
        let lens = line_lengths_in_range(text, 0, 1);
        assert_eq!(lens.get(&0).copied(), Some(0));
        assert_eq!(lens.get(&1).copied(), Some(0));
    }

    #[test]
    fn line_length_out_of_range() {
        let text = "abc\n";
        // Lines past the end of the text are absent; callers default to 0.
        assert_eq!(line_lengths_in_range(text, 99, 120).get(&99), None);
    }

    #[test]
    fn line_length_window_excludes_outside_lines() {
        let text = "a\nbb\nccc\ndddd\n";
        let lens = line_lengths_in_range(text, 1, 2);
        assert_eq!(lens.get(&0), None);
        assert_eq!(lens.get(&1).copied(), Some(2));
        assert_eq!(lens.get(&2).copied(), Some(3));
        assert_eq!(lens.get(&3), None);
    }

}
