use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use tokio::sync::{RwLock, watch};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CodeAction, CodeActionKind, CodeActionOptions, CodeActionOrCommand, CodeActionParams,
    CodeActionProviderCapability, CodeActionResponse, Diagnostic, DiagnosticRelatedInformation,
    DiagnosticSeverity, DidChangeTextDocumentParams, DidChangeWatchedFilesParams,
    DidChangeWatchedFilesRegistrationOptions, DidChangeWorkspaceFoldersParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    DocumentChanges, FileSystemWatcher, GlobPattern, GotoDefinitionParams,
    GotoDefinitionResponse, Hover, HoverContents, HoverParams, HoverProviderCapability,
    InitializeParams, InitializeResult, InitializedParams, InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams,
    InlayHintTooltip, Location, MarkupContent, MarkupKind, MessageType, OneOf,
    OptionalVersionedTextDocumentIdentifier, Position, PositionEncodingKind, Range, Registration,
    ServerCapabilities, TextDocumentEdit, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TextDocumentSyncSaveOptions, TextEdit, Url, WorkspaceEdit, WorkspaceFoldersServerCapabilities, WorkspaceServerCapabilities,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tree_sitter::{Parser, Tree};

use ndim_lsp::{
    LayerShapeAnalysis, ResolutionCache, ShapeError, ShapeErrorKind, analyze_layer_shapes,
    clear_resolution_cache, new_resolution_cache,
};

/// Quiet time after a change before its text is analyzed.
const DEBOUNCE: Duration = Duration::from_millis(150);

/// Analysis plus the exact text/tree it was computed from, so hover and
/// inlay hints reuse the parse instead of re-parsing per request.
#[derive(Clone)]
pub struct CachedAnalysis {
    pub analysis: Arc<LayerShapeAnalysis>,
    pub text: Arc<str>,
    pub tree: Arc<Tree>,
    pub version: i32,
    pub rev: u64,
}

/// One text snapshot of an open document. `rev` is unique per text write, so
/// results keyed on it can never belong to a closed or replaced text.
#[derive(Clone)]
pub struct Document {
    text: Arc<str>,
    version: i32,
    rev: u64,
    /// Analysis of this text may start at this time (the debounce deadline).
    ready_at: Instant,
}

type Inflight = Mutex<HashMap<(Url, u64), watch::Receiver<Option<Option<CachedAnalysis>>>>>;

/// Every field is shared, so a clone can move into a spawned task.
#[derive(Clone)]
pub struct Backend {
    pub client: Client,
    /// Text and version change together under one lock, in message order:
    /// the handlers that write them never await.
    pub documents: Arc<StdRwLock<HashMap<Url, Document>>>,
    pub next_rev: Arc<AtomicU64>,
    pub workspace_roots: Arc<StdRwLock<Vec<PathBuf>>>,
    /// Newest completed analysis per URI. Shared `Arc`s, so a hit clones
    /// pointers, not the whole analysis.
    pub analysis_cache: Arc<StdRwLock<HashMap<Url, CachedAnalysis>>>,
    /// Single-flight: one running analysis per (URI, rev); every caller for
    /// that text waits on it.
    pub inflight: Arc<Inflight>,
    /// One analysis runs at a time per URI. A later one waits here, and skips
    /// the work if newer text arrived meanwhile.
    pub gates: Arc<Mutex<HashMap<Url, Arc<tokio::sync::Mutex<()>>>>>,
    /// Session-lifetime cache for resolved import targets.
    /// Keyed on (import-path-segments, search-roots-fingerprint).
    /// Invalidated when workspace folders change.
    pub resolution_cache: Arc<ResolutionCache>,
    /// Severity for `ShapeErrorKind::Mismatch` diagnostics, set from
    /// `initializationOptions.diagnostic_severity`.
    pub diagnostic_severity: Arc<RwLock<DiagnosticSeverity>>,
    /// Severity for `ShapeErrorKind::Approximation` diagnostics, set from
    /// `initializationOptions.approximation_severity` (defaults to WARNING,
    /// independent of `diagnostic_severity`).
    pub approximation_severity: Arc<RwLock<DiagnosticSeverity>>,
    /// The client accepts `workspace/didChangeWatchedFiles` registration.
    pub can_watch_files: Arc<AtomicBool>,
    /// The client accepts `workspace/inlayHint/refresh`.
    pub can_refresh_hints: Arc<AtomicBool>,
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
            *self.workspace_roots.write().unwrap() = roots;
        }

        if let Some(ws) = params.capabilities.workspace.as_ref() {
            let watch = ws
                .did_change_watched_files
                .as_ref()
                .and_then(|c| c.dynamic_registration);
            self.can_watch_files
                .store(watch == Some(true), Ordering::Relaxed);
            let refresh = ws.inlay_hint.as_ref().and_then(|c| c.refresh_support);
            self.can_refresh_hints
                .store(refresh == Some(true), Ordering::Relaxed);
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
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        save: Some(TextDocumentSyncSaveOptions::Supported(true)),
                        ..Default::default()
                    },
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                inlay_hint_provider: Some(OneOf::Left(true)),
                definition_provider: Some(OneOf::Left(true)),
                code_action_provider: Some(CodeActionProviderCapability::Options(
                    CodeActionOptions {
                        code_action_kinds: Some(vec![CodeActionKind::QUICKFIX]),
                        resolve_provider: Some(false),
                        ..Default::default()
                    },
                )),
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

    async fn initialized(&self, _: InitializedParams) {
        if !self.can_watch_files.load(Ordering::Relaxed) {
            return;
        }
        let options = DidChangeWatchedFilesRegistrationOptions {
            watchers: vec![FileSystemWatcher {
                glob_pattern: GlobPattern::String("**/*.py".to_string()),
                kind: None,
            }],
        };
        let registration = Registration {
            id: "ndim-watch-python".to_string(),
            method: "workspace/didChangeWatchedFiles".to_string(),
            register_options: serde_json::to_value(options).ok(),
        };
        if let Err(e) = self.client.register_capability(vec![registration]).await {
            self.client
                .log_message(MessageType::WARNING, format!("file watch failed: {e}"))
                .await;
        }
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        // The write comes first: no await may let a later change overtake it.
        let doc = params.text_document;
        if let Some(rev) = self.set_document(&doc.uri, doc.text, doc.version, Duration::ZERO, false)
        {
            self.spawn_publish(doc.uri, rev);
        }
        self.client
            .log_message(MessageType::INFO, "File opened")
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let Some(change) = params.content_changes.into_iter().last() else {
            return;
        };
        let uri = params.text_document.uri;
        // No await before the write: handlers start in message order, so the
        // newest text always wins. The analysis runs in its own task after
        // the debounce, so this handler does not hold a concurrency slot.
        let Some(rev) = self.set_document(
            &uri,
            change.text,
            params.text_document.version,
            DEBOUNCE,
            true,
        ) else {
            return;
        };
        self.spawn_publish(uri, rev);
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        if let Ok(path) = params.text_document.uri.to_file_path() {
            self.files_changed(&[path]);
        }
    }

    async fn did_change_watched_files(&self, params: DidChangeWatchedFilesParams) {
        let paths: Vec<PathBuf> = params
            .changes
            .iter()
            .filter_map(|c| c.uri.to_file_path().ok())
            .collect();
        self.files_changed(&paths);
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = &params.text_document.uri;
        // Evict all per-document state so long-lived sessions don't leak.
        // An analysis still running finds no matching text and drops itself.
        self.documents.write().unwrap().remove(uri);
        self.analysis_cache.write().unwrap().remove(uri);
        self.gates.lock().unwrap().remove(uri);
        // Clear any diagnostics the client may still be showing.
        self.client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
    }

    async fn did_change_workspace_folders(&self, params: DidChangeWorkspaceFoldersParams) {
        {
            let mut lock = self.workspace_roots.write().unwrap();

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

        // Workspace roots changed → site-packages may have shifted.
        // Clear the resolution cache so stale entries don't survive, and
        // analyze the open documents again with the new roots.
        clear_resolution_cache(&self.resolution_cache);
        self.reanalyze_open_documents();
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        if self.document(&uri).is_none() {
            return Ok(None);
        }
        self.on_hover(&uri, &pos).await
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = params.text_document.uri;
        let visible_range = params.range;
        if self.document(&uri).is_none() {
            return Ok(None);
        }
        Ok(self.compute_inlay_hints(&uri, &visible_range).await)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        self.on_goto_definition(&uri, &position).await
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = params.text_document.uri;
        let Some(cached) = self.get_current_analysis(&uri).await else {
            return Ok(None);
        };
        Ok(Some(transpose_quick_fixes(
            &cached.analysis,
            &cached.text,
            &uri,
            cached.version,
            &params.range,
            params.context.only.as_deref(),
        )))
    }
}

impl Backend {
    fn document(&self, uri: &Url) -> Option<Document> {
        self.documents.read().unwrap().get(uri).cloned()
    }

    /// Files changed on disk. If an import resolution depended on one, the
    /// cache is dropped and every open document is analyzed again: the cache
    /// does not record which document used which file, so this is the safe set.
    fn files_changed(&self, paths: &[PathBuf]) {
        let dropped = self.resolution_cache.revalidate(paths);
        // A resolution that is running may have read the old file before it
        // was tracked: its analysis is stale even if nothing was dropped.
        let running = !paths.is_empty() && !self.inflight.lock().unwrap().is_empty();
        if dropped || running {
            self.reanalyze_open_documents();
        }
    }

    /// New revs for every open document, then an analysis and a publication
    /// for each, so no result built before this call is used again.
    fn reanalyze_open_documents(&self) {
        let now = Instant::now();
        let mut revs = Vec::new();
        for (uri, doc) in self.documents.write().unwrap().iter_mut() {
            doc.rev = self.next_rev.fetch_add(1, Ordering::Relaxed);
            doc.ready_at = now;
            revs.push((uri.clone(), doc.rev));
        }
        // After the new revs: an analysis of an old rev can no longer fill it.
        self.analysis_cache.write().unwrap().clear();
        for (uri, rev) in revs {
            self.spawn_publish(uri, rev);
        }
        if self.can_refresh_hints.load(Ordering::Relaxed) {
            let client = self.client.clone();
            tokio::spawn(async move {
                let _ = client.inlay_hint_refresh().await;
            });
        }
    }

    /// Store new text. `monotonic` drops a version older than the stored one.
    /// Returns the new rev, or `None` if the write was dropped.
    fn set_document(
        &self,
        uri: &Url,
        text: String,
        version: i32,
        debounce: Duration,
        monotonic: bool,
    ) -> Option<u64> {
        let mut docs = self.documents.write().unwrap();
        if monotonic && docs.get(uri).is_some_and(|d| d.version > version) {
            return None;
        }
        let rev = self.next_rev.fetch_add(1, Ordering::Relaxed);
        docs.insert(
            uri.clone(),
            Document {
                text: Arc::from(text),
                version,
                rev,
                ready_at: Instant::now() + debounce,
            },
        );
        Some(rev)
    }

    fn cache_hit(&self, uri: &Url, rev: u64) -> Option<CachedAnalysis> {
        let cache = self.analysis_cache.read().unwrap();
        cache.get(uri).filter(|c| c.rev == rev).cloned()
    }

    fn is_current(&self, uri: &Url, rev: u64) -> bool {
        self.document(uri).is_some_and(|d| d.rev == rev)
    }

    /// Analysis of the newest text, and nothing else: `None` if the document
    /// is closed, the analysis failed, or newer text arrived while this call
    /// waited. A request that needs fresh data waits out the debounce, then
    /// shares the one analysis with the diagnostics task and other requests.
    async fn get_analysis(&self, uri: &Url) -> Option<CachedAnalysis> {
        let doc = self.document(uri)?;
        if let Some(hit) = self.cache_hit(uri, doc.rev) {
            return Some(hit);
        }
        let wait = doc.ready_at.saturating_duration_since(Instant::now());
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
            if !self.is_current(uri, doc.rev) {
                return None;
            }
        }
        self.join_analysis(uri, &doc).await
    }

    /// `get_analysis`, but `None` if the text moved on before the caller can
    /// use the result (its byte and line positions would be wrong).
    async fn get_current_analysis(&self, uri: &Url) -> Option<CachedAnalysis> {
        let cached = self.get_analysis(uri).await?;
        self.is_current(uri, cached.rev).then_some(cached)
    }

    async fn join_analysis(&self, uri: &Url, doc: &Document) -> Option<CachedAnalysis> {
        let mut rx = {
            let mut inflight = self.inflight.lock().unwrap();
            // Checked under the lock: a finished task fills the cache before it
            // leaves `inflight`, so a miss here means no result exists yet.
            if let Some(hit) = self.cache_hit(uri, doc.rev) {
                return Some(hit);
            }
            let key = (uri.clone(), doc.rev);
            match inflight.get(&key) {
                Some(rx) => rx.clone(),
                None => {
                    let (tx, rx) = watch::channel(None);
                    inflight.insert(key, rx.clone());
                    let this = self.clone();
                    let (uri, doc) = (uri.clone(), doc.clone());
                    tokio::spawn(async move { this.run_analysis(uri, doc, tx).await });
                    rx
                }
            }
        };
        let done = rx.wait_for(|v| v.is_some()).await.ok()?.clone();
        done.flatten()
    }

    /// The one analysis for (`uri`, `doc.rev`). CPU work runs on the blocking
    /// pool so the runtime threads keep reading input.
    async fn run_analysis(
        self,
        uri: Url,
        doc: Document,
        tx: watch::Sender<Option<Option<CachedAnalysis>>>,
    ) {
        let gate = self
            .gates
            .lock()
            .unwrap()
            .entry(uri.clone())
            .or_default()
            .clone();
        let _turn = gate.lock().await;
        if !self.is_current(&uri, doc.rev) {
            self.inflight.lock().unwrap().remove(&(uri, doc.rev));
            let _ = tx.send(Some(None));
            return;
        }
        let t_total = Instant::now();
        let roots = self.workspace_roots.read().unwrap().clone();
        let resolution_cache = self.resolution_cache.clone();
        let text = doc.text.clone();
        let run =
            tokio::task::spawn_blocking(move || compute_analysis(&text, &roots, &resolution_cache))
                .await;

        let result = match run {
            Ok(Ok(run)) => {
                let total_ms = t_total.elapsed().as_millis();
                self.client
                    .log_message(
                        MessageType::INFO,
                        format!(
                            "analysis: total={}ms parse={}ms roots={}ms analyze={}ms \
                             | text={}B/{}L roots={} reads={}/{}B res_cache_entries={} hits={} misses={}",
                            total_ms,
                            run.parse_ms,
                            run.roots_ms,
                            run.analyze_ms,
                            doc.text.len(),
                            doc.text.lines().count(),
                            run.search_roots,
                            run.read_count,
                            run.read_bytes,
                            self.resolution_cache.map.read().unwrap().len(),
                            self.resolution_cache.hits.load(Ordering::Relaxed),
                            self.resolution_cache.misses.load(Ordering::Relaxed),
                        ),
                    )
                    .await;
                Some(CachedAnalysis {
                    analysis: Arc::new(run.analysis),
                    text: doc.text.clone(),
                    tree: Arc::new(run.tree),
                    version: doc.version,
                    rev: doc.rev,
                })
            }
            Ok(Err(msg)) => {
                self.client
                    .log_message(MessageType::WARNING, format!("analysis failed: {}", msg))
                    .await;
                None
            }
            Err(_) => None,
        };

        // Cache only if this is still the newest text; a stale entry would
        // never be read and a closed document must not be re-populated.
        if let Some(cached) = &result {
            let docs = self.documents.read().unwrap();
            if docs.get(&uri).is_some_and(|d| d.rev == doc.rev) {
                self.analysis_cache
                    .write()
                    .unwrap()
                    .insert(uri.clone(), cached.clone());
            }
        }
        self.inflight.lock().unwrap().remove(&(uri, doc.rev));
        let _ = tx.send(Some(result));
    }

    /// Publish diagnostics for `rev` in a task of its own, after the debounce.
    fn spawn_publish(&self, uri: Url, rev: u64) {
        let this = self.clone();
        tokio::spawn(async move { this.publish_diagnostics_for(&uri, rev).await });
    }

    /// Analyze (or hit the cache) and publish diagnostics, but only if `rev`
    /// is still the newest text when the result is ready.
    async fn publish_diagnostics_for(&self, uri: &Url, rev: u64) {
        let cached = self.get_analysis(uri).await;
        let mismatch_severity = *self.diagnostic_severity.read().await;
        let approximation_severity = *self.approximation_severity.read().await;
        let Some(doc) = self.document(uri) else {
            return;
        };
        if doc.rev != rev {
            return;
        }
        let diagnostics: Vec<Diagnostic> = match &cached {
            Some(cached) if cached.rev == rev => cached
                .analysis
                .errors
                .iter()
                .cloned()
                .map(|e| {
                    shape_error_to_diagnostic(
                        e,
                        uri,
                        &cached.text,
                        mismatch_severity,
                        approximation_severity,
                    )
                })
                .collect(),
            Some(_) => return,
            None => Vec::new(),
        };

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, Some(doc.version))
            .await;
    }

    async fn compute_inlay_hints(
        &self,
        uri: &Url,
        visible_range: &Range,
    ) -> Option<Vec<InlayHint>> {
        let t_total = Instant::now();

        // Snapshot cache state *before* `get_current_analysis` populates it on
        // miss, so the telemetry below actually distinguishes hits from misses.
        let was_cached = {
            let doc = self.document(uri)?;
            self.cache_hit(uri, doc.rev).is_some()
        };

        let cached = self.get_current_analysis(uri).await?;
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
            if line < visible_range.start.line || line > visible_range.end.line {
                continue;
            }

            if !seen.insert((line, rec.name.clone())) {
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

        // Snapshot cache state *before* `get_current_analysis` may populate it
        // on miss, so telemetry distinguishes hits from misses.
        let Some(doc) = self.document(uri) else {
            return Ok(None);
        };
        let was_cached = self.cache_hit(uri, doc.rev).is_some();

        // The cached snapshot carries the text and tree the analysis was
        // computed from, so `cursor_byte` and `analysis.scopes` byte ranges
        // are consistent by construction — no re-parse per hover. A text
        // change while waiting returns `None`; the editor will re-request.
        let Some(cached) = self.get_current_analysis(uri).await else {
            return Ok(None);
        };
        let root = cached.tree.root_node();
        let Some(cursor_byte) = lsp_position_to_byte(&cached.text, pos) else {
            return Ok(None);
        };
        let Some(node) = root.descendant_for_byte_range(cursor_byte, cursor_byte) else {
            return Ok(None);
        };
        if node.kind() != "identifier" {
            return Ok(None);
        }
        let var_name = match node.utf8_text(cached.text.as_bytes()) {
            Ok(v) => v.to_string(),
            Err(_) => return Ok(None),
        };
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
        let hover_range = ts_range_to_lsp_range(&cached.text, node.range());

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("```python\n{}\n```", hover_content),
            }),
            range: Some(hover_range),
        }))
    }

    async fn on_goto_definition(
        &self,
        uri: &Url,
        position: &Position,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let Some(cached) = self.get_current_analysis(uri).await else {
            return Ok(None);
        };
        let Some(range) = find_dimension_definition(
            &cached.analysis.scopes,
            &cached.text,
            position,
        ) else {
            return Ok(None);
        };
        Ok(Some(GotoDefinitionResponse::Scalar(Location {
            uri: uri.clone(),
            range: ts_range_to_lsp_range(&cached.text, range),
        })))
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

fn is_symbolic_dimension(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(c) if c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn find_dimension_definition(
    scopes: &[ndim_lsp::FunctionShapeScope],
    text: &str,
    position: &Position,
) -> Option<tree_sitter::Range> {
    let byte = lsp_position_to_byte(text, position)?;
    let scope = ndim_lsp::scope_for_byte(scopes, byte)?;
    let current = scope
        .dimension_sites
        .iter()
        .find(|site| site.range.start_byte <= byte && byte < site.range.end_byte)?;
    if !is_symbolic_dimension(&current.value) {
        return None;
    }
    scope
        .dimension_sites
        .iter()
        .filter(|site| site.value == current.value)
        .min_by_key(|site| site.range.start_byte)
        .map(|site| site.range)
}

fn position_is_before_or_equal(left: &Position, right: &Position) -> bool {
    left.line < right.line || (left.line == right.line && left.character <= right.character)
}

fn ranges_intersect(left: &Range, right: &Range) -> bool {
    position_is_before_or_equal(&left.start, &right.end)
        && position_is_before_or_equal(&right.start, &left.end)
}

fn transpose_quick_fixes(
    analysis: &LayerShapeAnalysis,
    text: &str,
    uri: &Url,
    version: i32,
    requested_range: &Range,
    only: Option<&[CodeActionKind]>,
) -> CodeActionResponse {
    if only.is_some_and(|kinds| {
        !kinds
            .iter()
            .any(|kind| kind.as_str() == CodeActionKind::QUICKFIX.as_str())
    }) {
        return Vec::new();
    }

    let mut actions = Vec::new();
    for error in &analysis.errors {
        let Some(ndim_lsp::ShapeFix::AppendTranspose {
            expression_range,
            operand_range,
        }) = error.fix.as_ref()
        else {
            continue;
        };
        let error_range = ts_range_to_lsp_range(text, error.range);
        let expression_range = ts_range_to_lsp_range(text, *expression_range);
        if !ranges_intersect(requested_range, &error_range)
            && !ranges_intersect(requested_range, &expression_range)
        {
            continue;
        }
        let Some(position) = byte_to_lsp_position(text, operand_range.end_byte) else {
            continue;
        };
        let edit = TextEdit {
            range: Range {
                start: position,
                end: position,
            },
            new_text: ".T".to_string(),
        };
        let workspace_edit = WorkspaceEdit {
            document_changes: Some(DocumentChanges::Edits(vec![TextDocumentEdit {
                text_document: OptionalVersionedTextDocumentIdentifier::new(uri.clone(), version),
                edits: vec![OneOf::Left(edit)],
            }])),
            ..Default::default()
        };
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Append .T to the right operand".to_string(),
            kind: Some(CodeActionKind::QUICKFIX),
            edit: Some(workspace_edit),
            is_preferred: Some(true),
            ..Default::default()
        }));
    }
    actions
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
    text: &str,
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
                range: ts_range_to_lsp_range(text, range),
            },
            message,
        }]
    });
    Diagnostic {
        range: ts_range_to_lsp_range(text, error.range),
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

fn byte_to_lsp_position(text: &str, byte: usize) -> Option<Position> {
    if byte > text.len() || !text.is_char_boundary(byte) {
        return None;
    }
    let prefix = &text[..byte];
    let line = prefix.bytes().filter(|&b| b == b'\n').count() as u32;
    let line_start = prefix.rfind('\n').map_or(0, |i| i + 1);
    Some(Position {
        line,
        character: text[line_start..byte].encode_utf16().count() as u32,
    })
}

fn lsp_position_to_byte(text: &str, position: &Position) -> Option<usize> {
    let line_start = if position.line == 0 {
        0
    } else {
        text.match_indices('\n')
            .nth(position.line as usize - 1)
            .map(|(i, _)| i + 1)?
    };
    let line_end = text[line_start..]
        .find('\n')
        .map_or(text.len(), |i| line_start + i);
    let line = &text[line_start..line_end];
    let mut utf16 = 0u32;
    for (offset, ch) in line.char_indices() {
        if utf16 == position.character {
            return Some(line_start + offset);
        }
        utf16 += ch.len_utf16() as u32;
        if utf16 > position.character {
            return None;
        }
    }
    (utf16 == position.character).then_some(line_end)
}

fn ts_range_to_lsp_range(text: &str, ts: tree_sitter::Range) -> Range {
    let fallback_start = Position {
        line: ts.start_point.row as u32,
        character: ts.start_point.column as u32,
    };
    let fallback_end = Position {
        line: ts.end_point.row as u32,
        character: ts.end_point.column as u32,
    };
    Range {
        start: byte_to_lsp_position(text, ts.start_byte).unwrap_or(fallback_start),
        end: byte_to_lsp_position(text, ts.end_byte).unwrap_or(fallback_end),
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

struct AnalysisRun {
    analysis: LayerShapeAnalysis,
    tree: Tree,
    parse_ms: u128,
    roots_ms: u128,
    analyze_ms: u128,
    search_roots: usize,
    read_count: usize,
    read_bytes: usize,
}

/// Parse and analyze `text`. Pure CPU and file reads: call from a blocking
/// thread.
fn compute_analysis(
    text: &str,
    workspace_roots: &[PathBuf],
    resolution_cache: &ResolutionCache,
) -> std::result::Result<AnalysisRun, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;
    let t_parse = Instant::now();
    let tree = parser.parse(text, None).ok_or("parse failed")?;
    let parse_ms = t_parse.elapsed().as_millis();

    let t_roots = Instant::now();
    let mut search_roots = workspace_roots.to_vec();
    search_roots.extend(python_site_packages_roots(workspace_roots));
    let roots_ms = t_roots.elapsed().as_millis();

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
    let analysis = analyze_layer_shapes(
        tree.root_node(),
        text,
        &search_roots,
        read_file,
        8,
        Some(resolution_cache),
    )?;
    Ok(AnalysisRun {
        analysis,
        tree,
        parse_ms,
        roots_ms,
        analyze_ms: t_analyze.elapsed().as_millis(),
        search_roots: search_roots.len(),
        read_count: read_count.load(Ordering::Relaxed),
        read_bytes: read_bytes.load(Ordering::Relaxed),
    })
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        documents: Default::default(),
        next_rev: Default::default(),
        workspace_roots: Default::default(),
        analysis_cache: Default::default(),
        inflight: Default::default(),
        gates: Default::default(),
        resolution_cache: new_resolution_cache(),
        diagnostic_severity: Arc::new(RwLock::new(DiagnosticSeverity::ERROR)),
        approximation_severity: Arc::new(RwLock::new(DiagnosticSeverity::WARNING)),
        can_watch_files: Default::default(),
        can_refresh_hints: Default::default(),
    });

    // Requests that wait for the debounce hold a slot: keep enough free for
    // notifications and shutdown.
    Server::new(stdin, stdout, socket)
        .concurrency_level(32)
        .serve(service)
        .await;
    // Do not wait for a running analysis on the blocking pool.
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

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
            "x",
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));

        let diag = shape_error_to_diagnostic(
            approximation,
            &uri,
            "y",
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));

        let approximation2 = ShapeError::approximation("z", "approx shape", range);
        let diag = shape_error_to_diagnostic(
            approximation2,
            &uri,
            "z",
            DiagnosticSeverity::HINT,
            DiagnosticSeverity::INFORMATION,
        );
        assert_eq!(diag.severity, Some(DiagnosticSeverity::INFORMATION));
    }

    #[test]
    fn shape_error_to_diagnostic_maps_related_information() {
        let text = "x\n  abc";
        let range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 1),
            start_byte: 0,
            end_byte: 1,
        };
        let related_range = tree_sitter::Range {
            start_point: tree_sitter::Point::new(1, 2),
            end_point: tree_sitter::Point::new(1, 5),
            start_byte: 4,
            end_byte: 7,
        };
        let uri = Url::parse("file:///test.py").unwrap();
        let error = ShapeError::mismatch("y", "matmul dimension mismatch", range)
            .with_related(related_range, "other operand `b`: shape [n, d]");

        let diag = shape_error_to_diagnostic(
            error,
            &uri,
            text,
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );

        let related = diag.related_information.expect("related_information");
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].location.uri, uri);
        assert_eq!(related[0].location.range, ts_range_to_lsp_range(text, related_range));
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
            "x",
            DiagnosticSeverity::ERROR,
            DiagnosticSeverity::WARNING,
        );
        assert!(diag.related_information.is_none());
    }

    #[test]
    fn lsp_positions_use_utf16_code_units() {
        let text = "πx\n😀z";
        let ts = tree_sitter::Range {
            start_point: tree_sitter::Point::new(0, 2),
            end_point: tree_sitter::Point::new(1, 5),
            start_byte: 2,
            end_byte: 9,
        };
        let lsp = ts_range_to_lsp_range(text, ts);
        assert_eq!(lsp.start, Position { line: 0, character: 1 });
        assert_eq!(lsp.end, Position { line: 1, character: 3 });
        assert_eq!(lsp_position_to_byte(text, &lsp.start), Some(2));
        assert_eq!(lsp_position_to_byte("😀", &Position { line: 0, character: 1 }), None);
    }

    #[test]
    fn dimension_definition_uses_first_name_in_the_same_scope() {
        let code = r#"def first(x: Float[Array, "n width"], y: Float[Array, "n width"]):
    pass

def second(x: Float[Array, "n width"], y: Float[Array, "n width"]):
    pass"#;
        let analysis = analyze(code);
        let use_byte = code.rfind("n width").unwrap();
        let position = byte_to_lsp_position(code, use_byte).unwrap();
        let definition = find_dimension_definition(&analysis.scopes, code, &position).unwrap();
        let second_start = code.find("def second").unwrap();

        assert!(definition.start_byte > second_start);
        assert_eq!(&code[definition.start_byte..definition.end_byte], "n");
    }

    #[test]
    fn dimension_definition_skips_concrete_and_expression_dimensions() {
        let code = r#"def f(x: Float[Array, "3 hidden*2"]):
    pass"#;
        let analysis = analyze(code);
        for token in ["3", "hidden*2"] {
            let byte = code.find(token).unwrap();
            let position = byte_to_lsp_position(code, byte).unwrap();
            assert!(find_dimension_definition(&analysis.scopes, code, &position).is_none());
        }
    }

    #[test]
    fn transpose_quick_fix_is_versioned_and_utf16_safe() {
        let code = r#"def f(α: Float[Array, "m k"], b: Float[Array, "n k"]):
    y = α @ b"#;
        let analysis = analyze(code);
        let ndim_lsp::ShapeFix::AppendTranspose { expression_range, .. } =
            analysis.errors[0].fix.as_ref().unwrap();
        let request_range = ts_range_to_lsp_range(code, *expression_range);
        let uri = Url::parse("file:///test.py").unwrap();
        let actions = transpose_quick_fixes(&analysis, code, &uri, 7, &request_range, None);

        assert_eq!(actions.len(), 1);
        let CodeActionOrCommand::CodeAction(action) = &actions[0] else {
            panic!("expected code action");
        };
        assert_eq!(action.title, "Append .T to the right operand");
        let changes = action.edit.as_ref().unwrap().document_changes.as_ref().unwrap();
        let DocumentChanges::Edits(edits) = changes else {
            panic!("expected versioned edits");
        };
        assert_eq!(edits[0].text_document.version, Some(7));
        let OneOf::Left(edit) = &edits[0].edits[0] else {
            panic!("expected text edit");
        };
        assert_eq!(edit.new_text, ".T");
        assert_eq!(edit.range.start, edit.range.end);
        assert_eq!(lsp_position_to_byte(code, &edit.range.start), code.rfind('b').map(|i| i + 1));

        let filtered = transpose_quick_fixes(
            &analysis,
            code,
            &uri,
            7,
            &request_range,
            Some(&[CodeActionKind::REFACTOR]),
        );
        assert!(filtered.is_empty());
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
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
            },
            FunctionShapeScope {
                function_name: Some("foo".into()),
                start_byte: 20,
                end_byte: 180,
                shapes: inner_shapes,
                return_shape: None,
                param_order: Vec::new(),
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
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
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
            },
            FunctionShapeScope {
                function_name: Some("foo".into()),
                start_byte: 20,
                end_byte: 180,
                shapes: inner_shapes,
                return_shape: None,
                param_order: Vec::new(),
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
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
        all_params: Vec::new(),
            dimension_sites: Vec::new(),
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
