use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::SystemTime,
};

use tree_sitter::{Node, Parser, Tree};

use crate::python_ast::{
    extract_call_arguments, extract_callable_signature, extract_jaxtyping_shapes,
    binds_name, find_top_level_symbol, undecorated,
};
use crate::types::*;

/// Session-lifetime cache for resolved import targets.
///
/// Keyed on `(import-path-segments, search-roots-fingerprint)` so that
/// a change to workspace/site-packages roots invalidates stale entries.
///
/// `search_roots_fingerprint` is a stable u64 hash of the ordered
/// search-root path list at resolution time. When roots change
/// (workspace folder added/removed), the fingerprint shifts and old
/// entries naturally fall out of lookup — no explicit eviction needed
/// beyond clearing the entire map.
///
/// Note: only the final resolution result (start → ResolvedImplementation)
/// is cached. Re-export chain intermediates are not cached individually,
/// which is correct for the LSP workload (queries target first-class
/// imports, not intermediate re-export hops). A resolution that found nothing
/// is cached as `None`.
///
/// Every file or path a resolution looked at, including the ones that did not
/// exist, has a stamp, and each entry knows which paths it looked at.
/// `revalidate` compares the stamps with the disk and drops only the entries
/// that looked at a changed path, so a saved, created, or deleted file never
/// leaves a stale entry. Imports read files through the caller's `read_file`,
/// which in the server gives the editor text of an open document; a change
/// of that text drops entries through `buffer_changed`.
pub struct ResolutionCache {
    pub map: std::sync::RwLock<ResolvedMap>,
    /// Session-lifetime cache for extracted jaxtyping `FunctionShapeScope`s of
    /// cross-file helper functions (cross-file return-type tracing). Keyed by
    /// the resolved implementation's file path plus function name, since one
    /// file can define several functions. `None` caches a lookup that found
    /// the function but no matching jaxtyping scope, so repeat calls to an
    /// unannotated cross-file helper don't re-parse the file.
    pub signatures: std::sync::RwLock<HashMap<(PathBuf, String), Option<FunctionShapeScope>>>,
    /// The parameters of each class or function, per file and symbol (see
    /// `callable_signature`).
    callables: std::sync::RwLock<HashMap<(PathBuf, String), Option<PythonCallableSignature>>>,
    /// Lock order: `tracked`, then `map`, then `signatures`, then `callables`.
    tracked: std::sync::RwLock<Tracked>,
    /// Disk state of each path an editor event already dropped entries for,
    /// so a second event for the same change does not drop them again.
    handled_events: std::sync::Mutex<HashMap<PathBuf, Stamp>>,
    /// Bumped by every drop. An insert whose resolution started before a
    /// drop is skipped: it may have read the old file.
    epoch: AtomicU64,
    /// Tells this cache from another, for results that name a cache.
    id: u64,
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    /// Number of implementation files parsed.
    pub parses: AtomicUsize,
    /// Last parse of each implementation file with the text it came from. A
    /// tree is reused only when the text read now is identical, so it needs no
    /// invalidation of its own. `Tree` is `Send + Sync`, and a clone is a cheap
    /// reference-counted copy that does not borrow the cache.
    parsed: std::sync::Mutex<HashMap<PathBuf, (Arc<str>, Tree)>>,
    /// Results of module probes (see `analysis::modules`) with the disk state
    /// each one read. An entry is used while `deps_valid` holds for it.
    module_probes: std::sync::Mutex<HashMap<String, (Deps, Arc<ModuleProbe>)>>,
    /// The part of a file that one class needs (see `analysis::modules`),
    /// per hash of the file's text and class name. The text is part of the
    /// key, so an entry needs no invalidation.
    class_sources: std::sync::Mutex<ClassSources>,
}

/// What analyzing a module of another file on one call gave: the shape it
/// returned and the messages of the errors the call caused.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ModuleProbe {
    pub(crate) shape: Option<Vec<String>>,
    pub(crate) errors: Vec<String>,
}

/// The part of a file that a class needs, per hash of the text and class name.
type ClassSources = HashMap<(u64, String), Option<Arc<String>>>;

/// Module probes kept at most. The map is emptied when it is full.
const MAX_MODULE_PROBES: usize = 512;

/// Parsed files kept at most. The map is emptied when it is full.
const MAX_PARSED_FILES: usize = 128;

/// Handled editor events kept at most. The map is emptied when it is full:
/// a second event for a forgotten change then drops entries once more.
const MAX_HANDLED_EVENTS: usize = 4096;

#[derive(Default)]
struct Tracked {
    stamps: HashMap<PathBuf, Stamp>,
    /// Paths each `map` entry looked at.
    map_paths: HashMap<(Vec<String>, u64), Arc<[PathBuf]>>,
    /// The file each `signatures` entry was read from.
    signature_files: HashMap<(PathBuf, String), PathBuf>,
    /// Number of times `revalidate` marked each path stale. A path never
    /// marked has generation 0. `clear` empties it: the clear count makes
    /// every older `Deps` invalid.
    generations: HashMap<PathBuf, u64>,
    /// Number of times the whole cache was cleared.
    clears: u64,
}

/// The disk state a computation looked at: every path it probed or read with
/// the generation of the path then, and the cache and clear count it ran
/// under. The computation's result is still valid while `ResolutionCache::deps_valid`
/// holds.
#[derive(Debug, Clone, PartialEq)]
pub struct Deps {
    cache: u64,
    clears: u64,
    paths: Vec<(PathBuf, u64)>,
}

/// What the resolutions of one `ResolutionCache::record_deps` call touched.
#[derive(Default)]
struct Touches {
    /// The paths of each resolution, shared with the cache entry.
    paths: Vec<Arc<[PathBuf]>>,
    /// A resolution whose paths the cache does not track, so a change to them
    /// would go unseen.
    untracked: bool,
}

thread_local! {
    static TOUCHES: RefCell<Option<Touches>> = const { RefCell::new(None) };
    /// The file of the document under analysis, for its relative imports.
    static DOCUMENT: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Run `f` with `document` as the file whose relative imports resolve. With
/// `None` (an untitled buffer, or a nested analysis of another file) a
/// relative import resolves to nothing.
pub fn with_document<T>(document: Option<&std::path::Path>, f: impl FnOnce() -> T) -> T {
    struct Put(Option<PathBuf>);
    impl Drop for Put {
        fn drop(&mut self) {
            DOCUMENT.with(|d| *d.borrow_mut() = self.0.take());
        }
    }
    let _restore = Put(DOCUMENT.with(|d| d.replace(document.map(std::path::Path::to_path_buf))));
    f()
}

/// A relative target of the document as an absolute one, with the roots to
/// look it up in. The import names a file next to the document, so its
/// package is the directory of the document, `dots - 1` levels up. The module
/// path starts at the longest search root above that package, or else at the
/// package's parent, which then comes first among the roots.
fn from_document(target: &ResolvedTarget, search_roots: &[PathBuf]) -> Option<(ResolvedTarget, Vec<PathBuf>)> {
    let document = DOCUMENT.with(|d| d.borrow().clone())?;
    let mut package = document.parent()?;
    for _ in 1..target.dots {
        package = package.parent()?;
    }
    let root = search_roots
        .iter()
        .filter(|root| package.starts_with(root) && package != root.as_path())
        .max_by_key(|root| root.components().count())
        .cloned()
        .or_else(|| package.parent().map(std::path::Path::to_path_buf))?;
    let mut parts = Vec::new();
    for part in package.strip_prefix(&root).ok()?.components() {
        parts.push(part.as_os_str().to_str()?.to_string());
    }
    parts.extend(target.parts.iter().cloned());
    let mut roots = vec![root.clone()];
    roots.extend(search_roots.iter().filter(|r| **r != root).cloned());
    Some((ResolvedTarget { dots: 0, parts }, roots))
}

/// Puts back the recording that was active before a nested one, also when
/// the nested one panics.
struct Restore(Option<Touches>);

impl Drop for Restore {
    fn drop(&mut self) {
        TOUCHES.with(|t| *t.borrow_mut() = self.0.take());
    }
}

fn recording() -> bool {
    TOUCHES.with(|t| t.borrow().is_some())
}

fn touch(paths: &Arc<[PathBuf]>) {
    TOUCHES.with(|t| {
        if let Some(touches) = t.borrow_mut().as_mut()
            && !touches.paths.iter().any(|seen| Arc::ptr_eq(seen, paths))
        {
            touches.paths.push(paths.clone());
        }
    });
}

fn touch_untracked() {
    TOUCHES.with(|t| {
        if let Some(touches) = t.borrow_mut().as_mut() {
            touches.untracked = true;
        }
    });
}

/// Resolved implementation per (import segments, roots fingerprint).
pub type ResolvedMap = HashMap<(Vec<String>, u64), Option<ResolvedImplementation>>;

/// Modification time and size, or `None` for a path that does not exist.
type Stamp = Option<(Option<SystemTime>, u64)>;

type Probes = Vec<(PathBuf, Stamp)>;

fn stamp_of(path: &std::path::Path) -> Stamp {
    std::fs::metadata(path)
        .ok()
        .map(|m| (m.modified().ok(), m.len()))
}

impl Tracked {
    fn add_stamps(&mut self, probes: &Probes) {
        // The oldest stamp wins: it is the one the cached data may predate.
        for (path, stamp) in probes {
            self.stamps.entry(path.clone()).or_insert(*stamp);
        }
    }
}

impl ResolutionCache {
    /// Drop the entries that looked at a path that changed on disk, or at one
    /// of `changed` (an editor event, which counts even when the modification
    /// time did not move). Returns whether it dropped anything.
    pub fn revalidate(&self, changed: &[PathBuf]) -> bool {
        let stale: HashSet<PathBuf> = {
            let tracked = self.tracked.read().unwrap();
            let mut handled = self.handled_events.lock().unwrap();
            let mut stale = HashSet::new();
            for p in changed.iter().filter(|p| tracked.stamps.contains_key(*p)) {
                let now = stamp_of(p);
                if handled.get(p) != Some(&now) {
                    if handled.len() >= MAX_HANDLED_EVENTS {
                        handled.clear();
                    }
                    handled.insert(p.clone(), now);
                    stale.insert(p.clone());
                }
            }
            for (p, old) in &tracked.stamps {
                if stamp_of(p) != *old {
                    stale.insert(p.clone());
                }
            }
            stale
        };
        if stale.is_empty() {
            // A resolution that started before this event may have read the
            // old file and not be tracked yet: it must not insert.
            if !changed.is_empty() {
                self.epoch.fetch_add(1, Ordering::SeqCst);
            }
            return false;
        }
        let mut tracked = self.tracked.write().unwrap();
        self.epoch.fetch_add(1, Ordering::SeqCst);
        for path in &stale {
            *tracked.generations.entry(path.clone()).or_default() += 1;
        }
        let mut map = self.map.write().unwrap();
        tracked.map_paths.retain(|key, paths| {
            let keep = !paths.iter().any(|p| stale.contains(p));
            if !keep {
                map.remove(key);
            }
            keep
        });
        let mut signatures = self.signatures.write().unwrap();
        tracked.signature_files.retain(|key, file| {
            let keep = !stale.contains(file);
            if !keep {
                signatures.remove(key);
            }
            keep
        });
        self.callables.write().unwrap().retain(|(file, _), _| !stale.contains(file));
        self.parsed.lock().unwrap().retain(|p, _| !stale.contains(p));
        tracked.stamps.retain(|p, _| !stale.contains(p));
        true
    }

    /// The editor text of `path` changed, and its disk state may not have.
    /// A resolution that runs now may read the old text, so it must not insert.
    /// If an entry looked at `path`, drop the entries as `revalidate` does for
    /// a saved file. Returns whether it dropped anything.
    pub fn buffer_changed(&self, path: &std::path::Path) -> bool {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        if !self.tracked.read().unwrap().stamps.contains_key(path) {
            return false;
        }
        self.handled_events.lock().unwrap().remove(path);
        self.revalidate(&[path.to_path_buf()])
    }

    /// Identifies this cache among others.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Run `f` and report the disk state its resolutions looked at. `None`
    /// when the state is unsafe to rely on: a resolution the cache does not
    /// track, or a drop of cache entries while `f` ran.
    pub fn record_deps<T>(&self, f: impl FnOnce() -> T) -> (T, Option<Deps>) {
        let epoch = self.epoch.load(Ordering::SeqCst);
        let mut outer = Restore(TOUCHES.with(|t| t.borrow_mut().replace(Touches::default())));
        let out = f();
        let touches = TOUCHES.with(|t| t.borrow_mut().take()).unwrap_or_default();
        // What a nested recording touched, an enclosing one touched too.
        if let Some(enclosing) = outer.0.as_mut() {
            enclosing.paths.extend(touches.paths.iter().cloned());
            enclosing.untracked |= touches.untracked;
        }
        drop(outer);
        // A drop bumps the epoch before it bumps a generation, both under the
        // write lock, so under the read lock the two agree.
        let tracked = self.tracked.read().unwrap();
        if touches.untracked || self.epoch.load(Ordering::SeqCst) != epoch {
            return (out, None);
        }
        let mut paths: Vec<(PathBuf, u64)> = touches
            .paths
            .iter()
            .flat_map(|set| set.iter())
            .map(|p| {
                let generation = tracked.generations.get(p).copied().unwrap_or(0);
                (p.clone(), generation)
            })
            .collect();
        paths.sort();
        paths.dedup();
        let deps = Deps {
            cache: self.id,
            clears: tracked.clears,
            paths,
        };
        (out, Some(deps))
    }

    /// The stored result of a module probe, if the files it read are
    /// unchanged. The recording that is active (see `record_deps`) learns of
    /// these files too.
    pub(crate) fn probe_lookup(&self, key: &str) -> Option<Arc<ModuleProbe>> {
        let (deps, probe) = self.module_probes.lock().unwrap().get(key).cloned()?;
        if !self.deps_valid(&deps) {
            return None;
        }
        let paths: Arc<[PathBuf]> = deps.paths.iter().map(|(p, _)| p.clone()).collect();
        touch(&paths);
        Some(probe)
    }

    /// `compute(source)`, kept per text of `source` and per `class`.
    pub(crate) fn class_source(
        &self,
        source: &str,
        class: &str,
        compute: impl FnOnce() -> Option<String>,
    ) -> Option<Arc<String>> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = (hasher.finish(), class.to_string());
        if let Some(found) = self.class_sources.lock().unwrap().get(&key) {
            return found.clone();
        }
        let found = compute().map(Arc::new);
        let mut sources = self.class_sources.lock().unwrap();
        if sources.len() >= MAX_MODULE_PROBES {
            sources.clear();
        }
        sources.insert(key, found.clone());
        found
    }

    /// Keep the result of a module probe that read the files in `deps`.
    pub(crate) fn probe_store(&self, key: String, deps: Deps, probe: Arc<ModuleProbe>) {
        let mut probes = self.module_probes.lock().unwrap();
        if probes.len() >= MAX_MODULE_PROBES && !probes.contains_key(&key) {
            probes.clear();
        }
        probes.insert(key, (deps, probe));
    }

    /// Whether no path in `deps` was marked stale, and the cache was not
    /// cleared, since `deps` was recorded.
    pub fn deps_valid(&self, deps: &Deps) -> bool {
        let tracked = self.tracked.read().unwrap();
        deps.cache == self.id
            && deps.clears == tracked.clears
            && deps
                .paths
                .iter()
                .all(|(p, generation)| tracked.generations.get(p).copied().unwrap_or(0) == *generation)
    }

    fn clear(&self) {
        let mut tracked = self.tracked.write().unwrap();
        self.epoch.fetch_add(1, Ordering::SeqCst);
        *tracked = Tracked {
            clears: tracked.clears + 1,
            ..Tracked::default()
        };
        self.handled_events.lock().unwrap().clear();
        self.map.write().unwrap().clear();
        self.signatures.write().unwrap().clear();
        self.callables.write().unwrap().clear();
        self.parsed.lock().unwrap().clear();
        self.module_probes.lock().unwrap().clear();
        self.class_sources.lock().unwrap().clear();
    }
}

/// Compute a stable u64 fingerprint for an ordered list of search-root paths.
/// Uses `DefaultHasher` over the `PathBuf` slice, which is deterministic
/// within a session and order-sensitive (reordering roots produces a
/// different fingerprint).
pub fn search_roots_fingerprint(search_roots: &[PathBuf]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    search_roots.hash(&mut hasher);
    hasher.finish()
}

/// Create a new empty resolution cache.
pub fn new_resolution_cache() -> Arc<ResolutionCache> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    Arc::new(ResolutionCache {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        map: std::sync::RwLock::new(HashMap::new()),
        signatures: std::sync::RwLock::new(HashMap::new()),
        callables: std::sync::RwLock::new(HashMap::new()),
        tracked: std::sync::RwLock::new(Tracked::default()),
        handled_events: std::sync::Mutex::new(HashMap::new()),
        epoch: AtomicU64::new(0),
        hits: AtomicUsize::new(0),
        misses: AtomicUsize::new(0),
        parses: AtomicUsize::new(0),
        parsed: std::sync::Mutex::new(HashMap::new()),
        module_probes: std::sync::Mutex::new(HashMap::new()),
        class_sources: std::sync::Mutex::new(HashMap::new()),
    })
}

/// Clear all entries from the resolution cache. Called when workspace
/// folders change, since site-packages roots may shift.
pub fn clear_resolution_cache(cache: &Arc<ResolutionCache>) {
    cache.clear();
}

#[cfg(test)]
pub fn resolve_python_module_on_disk(
    module: &[String],
    search_roots: &[PathBuf],
) -> Option<PathBuf> {
    resolve_module_probing(module, search_roots, &mut Vec::new())
}

/// Stat `path` and record the stamp, which is taken before any later read.
fn probe(path: &std::path::Path, probes: &mut Probes) -> bool {
    let stamp = stamp_of(path);
    let exists = stamp.is_some();
    probes.push((path.to_path_buf(), stamp));
    exists
}

fn resolve_module_probing(
    module: &[String],
    search_roots: &[PathBuf],
    probes: &mut Probes,
) -> Option<PathBuf> {
    if module.is_empty() {
        return None;
    }
    // CPython takes a package directory before a module file of one name.
    for root in search_roots {
        let module_path = root.join(module.join("/"));
        let package_path = module_path.join("__init__.py");
        if probe(&package_path, probes) {
            return Some(package_path);
        }

        let file_path = module_path.with_extension("py");
        if probe(&file_path, probes) {
            return Some(file_path);
        }
    }
    None
}

pub fn resolve_call_target(
    target: &str,
    import_map: &HashMap<String, ImportPath>,
) -> ResolvedTarget {
    let target_parts = target
        .split(".")
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>();

    let Some((first, rest)) = target_parts.split_first() else {
        return ResolvedTarget {
            dots: 0,
            parts: target_parts.iter().map(|p| p.to_string()).collect(),
        };
    };

    let imported = import_map.get(*first);

    match imported {
        Some(i) => {
            let mut parts = i.module.clone();
            parts.push(i.name.to_string());
            parts.extend(rest.iter().map(|p| p.to_string()));

            ResolvedTarget {
                dots: i.dots,
                parts,
            }
        }
        None => ResolvedTarget {
            dots: 0,
            parts: target_parts.iter().map(|p| p.to_string()).collect(),
        },
    }
}

#[cfg(test)]
pub fn resolve_target_on_disk(
    target: &ResolvedTarget,
    search_roots: &[PathBuf],
) -> Option<ResolvedModuleTarget> {
    let read = |path: &PathBuf| std::fs::read_to_string(path).ok();
    resolve_target_probing(target, search_roots, &read, None, &mut Vec::new())
}

/// The longest prefix of `target` that is a module. A submodule `a.b` is also
/// the attribute `b` of `a` that the package's `__init__.py` may bind to
/// something else. Which one wins depends on the order of imports at run
/// time, so a package that binds the name to anything but the submodule
/// gives no module.
fn resolve_target_probing<F>(
    target: &ResolvedTarget,
    search_roots: &[PathBuf],
    read_file: &F,
    cache: Option<&ResolutionCache>,
    probes: &mut Probes,
) -> Option<ResolvedModuleTarget>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    if target.dots > 0 || target.parts.is_empty() {
        return None;
    }
    // As `sys.path`: the first root with the top-level package or module owns
    // it and its submodules. A namespace package (no `__init__.py`) has no
    // owner, so every root is searched.
    let owner = search_roots.iter().find(|root| {
        let top = root.join(&target.parts[0]);
        probe(&top.join("__init__.py"), probes) || probe(&top.with_extension("py"), probes)
    });
    let search_roots = owner.map_or(search_roots, std::slice::from_ref);

    for len in (1..=target.parts.len()).rev() {
        let module_parts = &target.parts[..len];
        let symbol_parts = &target.parts[len..];

        if let Some(file_path) = resolve_module_probing(module_parts, search_roots, probes) {
            if !submodule_unshadowed(&file_path, module_parts, read_file, cache, probes) {
                return None;
            }
            return Some(ResolvedModuleTarget {
                dots: target.dots,
                module_parts: module_parts.to_vec(),
                file_path,
                symbol_parts: symbol_parts.to_vec(),
            });
        }
    }

    None
}

/// Whether no `__init__.py` above the module `file_path` binds the next part
/// of `module_parts` to anything but that submodule.
fn submodule_unshadowed<F>(
    file_path: &std::path::Path,
    module_parts: &[String],
    read_file: &F,
    cache: Option<&ResolutionCache>,
    probes: &mut Probes,
) -> bool
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut dir = file_path.parent();
    if file_path.file_name().is_some_and(|name| name == "__init__.py") {
        dir = dir.and_then(|d| d.parent());
    }
    for len in (2..=module_parts.len()).rev() {
        let Some(package_dir) = dir else {
            return false;
        };
        dir = package_dir.parent();
        let init = package_dir.join("__init__.py");
        if !probe(&init, probes) {
            continue;
        }
        let Some(text) = read_file(&init) else {
            return false;
        };
        let Ok(tree) = parse_implementation(cache, &init, &text) else {
            return false;
        };
        let name = &module_parts[len - 1];
        let package = &module_parts[..len - 1];
        match binds_name(tree.root_node(), &text, name) {
            Ok(false) => {}
            Ok(true) => match find_top_level_symbol(tree.root_node(), &text, name) {
                Ok(Some(PythonSymbol::Import { path, .. }))
                    if resolve_import_path_from_package(package, &path)
                        .is_some_and(|t| t.parts == module_parts[..len]) => {}
                _ => return false,
            },
            Err(_) => return false,
        }
    }
    true
}

pub fn resolve_import_path_from_package(
    current_package_parts: &[String],
    import_path: &ImportPath,
) -> Option<ResolvedTarget> {
    if import_path.dots == 0 {
        let mut parts = import_path.module.clone();
        parts.push(import_path.name.clone());
        return Some(ResolvedTarget { dots: 0, parts });
    }

    let up = import_path.dots - 1;

    if up >= current_package_parts.len() {
        return None;
    }

    let base_len = current_package_parts.len() - up;
    let mut parts = current_package_parts[..base_len].to_vec();

    parts.extend(import_path.module.iter().cloned());
    parts.push(import_path.name.clone());

    Some(ResolvedTarget { dots: 0, parts })
}

pub fn follow_import_symbol_once(
    current_package_parts: &[String],
    symbol: &PythonSymbol,
) -> Option<ResolvedTarget> {
    match symbol {
        PythonSymbol::Import { path, .. } => {
            resolve_import_path_from_package(current_package_parts, path)
        }
        PythonSymbol::Class { .. } | PythonSymbol::Function { .. } => None,
    }
}

/// The package that a relative import in the module of `target` starts from:
/// the module itself for an `__init__.py`, else the module's parent.
fn package_of(target: &ResolvedModuleTarget) -> &[String] {
    let parts = &target.module_parts;
    if target.file_path.file_name().is_some_and(|name| name == "__init__.py") {
        parts
    } else {
        &parts[..parts.len().saturating_sub(1)]
    }
}

/// Parse `text` as the content of `path`, reusing the cached tree when the
/// cache holds a parse of identical text.
fn parse_implementation(
    cache: Option<&ResolutionCache>,
    path: &std::path::Path,
    text: &str,
) -> Result<Tree, String> {
    if let Some(cache) = cache
        && let Some((cached_text, tree)) = cache.parsed.lock().unwrap().get(path)
        && &**cached_text == text
    {
        return Ok(tree.clone());
    }
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;
    let tree = parser
        .parse(text, None)
        .ok_or_else(|| "failed to parse file".to_string())?;
    if let Some(cache) = cache {
        cache.parses.fetch_add(1, Ordering::Relaxed);
        let mut parsed = cache.parsed.lock().unwrap();
        if parsed.len() >= MAX_PARSED_FILES && !parsed.contains_key(path) {
            parsed.clear();
        }
        parsed.insert(path.to_path_buf(), (Arc::from(text), tree.clone()));
    }
    Ok(tree)
}

pub fn resolve_implementation<F>(
    start: ResolvedTarget,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Option<ResolvedImplementation>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    if start.dots > 0 {
        let Some((start, roots)) = from_document(&start, search_roots) else {
            return Ok(None);
        };
        return resolve_implementation(start, &roots, read_file, max_depth, cache);
    }
    let epoch = cache.map_or(0, |c| c.epoch.load(Ordering::SeqCst));
    // Cache lookup for non-empty targets. A relative target of the document
    // was made absolute above, so it uses the cache too.
    if start.dots == 0
        && !start.parts.is_empty()
        && let Some(cache) = cache
    {
        let fingerprint = search_roots_fingerprint(search_roots);
        let key = (start.parts.clone(), fingerprint);
        // Lock order: `tracked`, then `map`. The paths of an entry are read
        // together with the entry.
        let tracked = recording().then(|| cache.tracked.read().unwrap());
        let cache_map = cache.map.read().unwrap();
        if let Some(cached) = cache_map.get(&key) {
            cache.hits.fetch_add(1, Ordering::Relaxed);
            match tracked.as_ref().map(|t| t.map_paths.get(&key)) {
                Some(Some(paths)) => touch(paths),
                Some(None) => touch_untracked(),
                None => {}
            }
            return Ok(cached.clone());
        }
        cache.misses.fetch_add(1, Ordering::Relaxed);
    }

    let mut current = start.clone();

    let mut visited = HashSet::new();
    let mut probes = Probes::new();

    let result = (|| {
        for _ in 0..max_depth {
            if !visited.insert(current.parts.clone()) {
                return Ok(None);
            }
            let Some(resolved) =
                resolve_target_probing(&current, search_roots, &read_file, cache, &mut probes)
            else {
                return Ok(None);
            };

            if resolved.symbol_parts.is_empty() {
                return Ok(Some(ResolvedImplementation {
                    target: resolved,
                    symbol: None,
                }));
            }

            // A file that cannot be read (not UTF-8) gives no symbol until
            // its stamp changes.
            let Some(text) = read_file(&resolved.file_path) else {
                return Ok(None);
            };

            let tree = parse_implementation(cache, &resolved.file_path, &text)?;

            let Some(symbol) = find_top_level_symbol(
                tree.root_node(),
                &text,
                &resolved.symbol_parts[0],
            )? else {
                return Ok(None);
            };

            match symbol {
                PythonSymbol::Class { .. } | PythonSymbol::Function { .. } => {
                    return Ok(Some(ResolvedImplementation {
                        target: resolved,
                        symbol: Some(symbol),
                    }));
                }
                PythonSymbol::Import { .. } => {
                    let Some(mut next) = follow_import_symbol_once(package_of(&resolved), &symbol) else {
                        return Ok(None);
                    };
                    next.parts.extend(resolved.symbol_parts.iter().skip(1).cloned());
                    current = next;
                }
            }
        }

        Ok(None)
    })();

    // Cache insert — absolute targets only, found or not found. An error is
    // not a resolution result.
    if start.dots == 0
        && !start.parts.is_empty()
        && let (Some(cache), Ok(resolved)) = (cache, &result)
    {
        let fingerprint = search_roots_fingerprint(search_roots);
        let key = (start.parts.clone(), fingerprint);
        let mut tracked = cache.tracked.write().unwrap();
        if cache.epoch.load(Ordering::SeqCst) == epoch {
            tracked.add_stamps(&probes);
            let paths: Arc<[PathBuf]> = probes.into_iter().map(|(path, _)| path).collect();
            touch(&paths);
            tracked.map_paths.insert(key.clone(), paths);
            cache.map.write().unwrap().insert(key, resolved.clone());
        } else {
            touch_untracked();
        }
    } else if start.dots == 0 && !start.parts.is_empty() && cache.is_some() {
        touch_untracked();
    }

    result
}

#[cfg(test)]
pub fn resolve_call_signature<F>(
    call: &CallInfo,
    source_text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Option<ResolvedCallSignature>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    resolve_call_signature_with_node(
        call,
        None,
        source_text,
        import_map,
        search_roots,
        read_file,
        max_depth,
        cache,
    )
}

/// Reuse a node from the source tree to locate the call's arguments. Only the
/// public text-only wrapper passes `None`, parsing the caller on demand after
/// the implementation resolves. The node and text must be from the same snapshot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_call_signature_with_node<F>(
    call: &CallInfo,
    source_node: Option<Node>,
    source_text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Option<ResolvedCallSignature>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let target = resolve_call_target(&call.target, import_map);
    let Some(implementation) =
        resolve_implementation(target, search_roots, &read_file, max_depth, cache)?
    else {
        return Ok(None);
    };
    let Some(symbol) = &implementation.symbol else {
        return Ok(None);
    };
    let Some(signature) =
        callable_signature(&implementation.target.file_path, symbol, &read_file, cache)?
    else {
        return Ok(None);
    };

    let source_tree;
    let source_node = match source_node {
        Some(node) => node,
        None => {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_python::LANGUAGE.into())
                .map_err(|e| e.to_string())?;
            source_tree = parser
                .parse(source_text, None)
                .ok_or_else(|| "failed to parse source file".to_string())?;
            source_tree.root_node()
        }
    };
    let Some(args_node) = source_node.descendant_for_byte_range(
        call.args_node_range.start_byte,
        call.args_node_range.end_byte,
    ) else {
        return Ok(None);
    };

    let arguments = extract_call_arguments(args_node, source_text)?;
    let Some(bindings) = bind_call_arguments(&signature, &arguments) else {
        return Ok(None);
    };

    Ok(Some(ResolvedCallSignature {
        implementation,
        signature,
        arguments,
        bindings,
    }))
}

/// The parameters of `symbol` in `file`, kept in the cache under the stamp
/// of the file.
fn callable_signature<F>(
    file: &std::path::Path,
    symbol: &PythonSymbol,
    read_file: &F,
    cache: Option<&ResolutionCache>,
) -> Result<Option<PythonCallableSignature>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let name = match symbol {
        PythonSymbol::Class { name } => format!("class {name}"),
        PythonSymbol::Function { name } => format!("def {name}"),
        PythonSymbol::Import { .. } => return Ok(None),
    };
    let key = (file.to_path_buf(), name);
    if let Some(cache) = cache
        && let Some(found) = cache.callables.read().unwrap().get(&key)
    {
        return Ok(found.clone());
    }
    let epoch = cache.map_or(0, |c| c.epoch.load(Ordering::SeqCst));
    let stamp = stamp_of(file);
    let Some(text) = read_file(&key.0) else {
        return Ok(None);
    };
    let tree = parse_implementation(cache, file, &text)?;
    let found = extract_callable_signature(tree.root_node(), &text, symbol)?;
    if let Some(cache) = cache {
        let mut tracked = cache.tracked.write().unwrap();
        if cache.epoch.load(Ordering::SeqCst) == epoch {
            tracked.add_stamps(&vec![(key.0.clone(), stamp)]);
            cache.callables.write().unwrap().insert(key, found.clone());
        }
    }
    Ok(found)
}

/// Resolve `target` as an imported free function and extract its jaxtyping
/// parameter + return shape annotations (cross-file return-type tracing).
///
/// Mirrors `resolve_call_signature`'s import-resolution walk, but pulls the
/// callee's `FunctionShapeScope` (via `extract_jaxtyping_shapes`) instead of
/// its bare parameter names, so the caller can feed the result straight into
/// the same `bind_and_substitute` logic already used for same-file helpers.
///
/// Returns `Ok(None)` when the target isn't found on disk, resolves to
/// something other than a function (e.g. a class), or the function has no
/// jaxtyping annotations at all.
pub fn resolve_imported_function_shape<F>(
    target: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Option<FunctionShapeScope>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let resolved_target = resolve_call_target(target, import_map);
    let Some(implementation) =
        resolve_implementation(resolved_target, search_roots, &read_file, max_depth, cache)?
    else {
        return Ok(None);
    };
    let Some(PythonSymbol::Function { name }) = &implementation.symbol else {
        return Ok(None);
    };
    // `lib.f.g` is an attribute of the function, not the function.
    if implementation.target.symbol_parts.len() != 1 {
        return Ok(None);
    }

    let epoch = cache.map_or(0, |c| c.epoch.load(Ordering::SeqCst));
    let cache_key = (implementation.target.file_path.clone(), name.clone());
    if let Some(cache) = cache
        && let Some(cached) = cache.signatures.read().unwrap().get(&cache_key)
    {
        return Ok(cached.clone());
    }

    let file_stamp = stamp_of(&implementation.target.file_path);
    let Some(text) = read_file(&implementation.target.file_path) else {
        return Ok(None);
    };

    let tree = parse_implementation(cache, &implementation.target.file_path, &text)?;

    let scopes = extract_jaxtyping_shapes(tree.root_node(), &text)?;
    let root = tree.root_node();
    let definition = root.named_children(&mut root.walk()).map(undecorated).find(|node| {
        node.kind() == "function_definition"
            && node.child_by_field_name("name").is_some_and(|n| text[n.byte_range()] == **name)
    });
    let found = definition.and_then(|definition| {
        scopes
            .iter()
            .skip(1)
            .find(|scope| scope.start_byte == definition.start_byte())
            .cloned()
    });

    if let Some(cache) = cache {
        let mut tracked = cache.tracked.write().unwrap();
        if cache.epoch.load(Ordering::SeqCst) == epoch {
            tracked.add_stamps(&vec![(cache_key.0.clone(), file_stamp)]);
            tracked
                .signature_files
                .insert(cache_key.clone(), cache_key.0.clone());
            cache.signatures.write().unwrap().insert(cache_key, found.clone());
        }
    }

    Ok(found)
}

/// The argument of each parameter that a call binds. `params` marks a bare
/// `*` and `/`, and a splat parameter with its stars. A splat in the call
/// can fill any parameter, so it gives `None`.
pub fn bind_call_arguments(
    signature: &PythonCallableSignature,
    args: &[CallArgument],
) -> Option<HashMap<String, String>> {
    let mut params = signature.params.as_slice();
    if signature.owner.is_some()
        && matches!(params.first(), Some(first) if first == "self" || first == "cls")
    {
        params = &params[1..];
    }
    let star = params.iter().position(|p| p.starts_with('*')).unwrap_or(params.len());
    let slash = params.iter().position(|p| p == "/");
    let positional: Vec<&String> = params[..star].iter().filter(|p| *p != "/").collect();
    let positional_only = &params[..slash.unwrap_or(0)];

    let mut bindings = HashMap::new();
    let mut positional_index = 0;

    for arg in args {
        match arg {
            CallArgument::Positional { value } if value.starts_with('*') => return None,
            CallArgument::Positional { value } => {
                if let Some(param) = positional.get(positional_index) {
                    bindings.insert((*param).clone(), value.clone());
                }
                positional_index += 1;
            }
            CallArgument::Keyword { name, value } => {
                if !positional_only.contains(name) {
                    bindings.insert(name.clone(), value.clone());
                }
            }
        }
    }

    Some(bindings)
}

#[cfg(test)]
mod resolve_import_path_from_package_tests {
    use super::*;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    #[test]
    fn test_absolute_import_ignores_current_package() {
        let found = resolve_import_path_from_package(
            &parts(&["equinox", "nn"]),
            &ip(0, &["jax"], "random"),
        );

        assert_eq!(found, Some(target(&["jax", "random"])));
    }

    #[test]
    fn test_relative_import_same_package_with_module() {
        let found = resolve_import_path_from_package(
            &parts(&["equinox", "nn"]),
            &ip(1, &["_linear"], "Linear"),
        );

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_relative_import_same_package_without_module() {
        let found =
            resolve_import_path_from_package(&parts(&["equinox", "nn"]), &ip(1, &[], "layers"));

        assert_eq!(found, Some(target(&["equinox", "nn", "layers"])));
    }

    #[test]
    fn test_relative_import_parent_package() {
        let found = resolve_import_path_from_package(&parts(&["pkg", "sub"]), &ip(2, &["x"], "Y"));

        assert_eq!(found, Some(target(&["pkg", "x", "Y"])));
    }

    #[test]
    fn test_relative_import_grandparent_package() {
        let found =
            resolve_import_path_from_package(&parts(&["pkg", "sub", "inner"]), &ip(3, &["x"], "Y"));

        assert_eq!(found, Some(target(&["pkg", "x", "Y"])));
    }

    #[test]
    fn test_relative_import_too_many_dots_returns_none() {
        let found = resolve_import_path_from_package(&parts(&["pkg", "sub"]), &ip(3, &["x"], "Y"));

        assert_eq!(found, None);
    }

    #[test]
    fn test_relative_import_from_empty_package_returns_none() {
        let found = resolve_import_path_from_package(&[], &ip(1, &["x"], "Y"));

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_absolute_import_name_is_kept() {
        let found = resolve_import_path_from_package(&parts(&["pkg"]), &ip(0, &[], "foo"));

        assert_eq!(found, Some(target(&["foo"])));
    }
}

#[cfg(test)]
mod resolve_implementation_tests {
    use super::*;
    use std::fs;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    fn implementation(
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
        symbol: Option<PythonSymbol>,
    ) -> ResolvedImplementation {
        ResolvedImplementation {
            target: ResolvedModuleTarget {
                dots: 0,
                module_parts: parts(module_parts),
                file_path,
                symbol_parts: parts(symbol_parts),
            },
            symbol,
        }
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    #[test]
    fn test_resolves_direct_class() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["pkg", "linear", "Linear"]), &roots, read, 5, None)
                .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["pkg", "linear"],
                tmp.path().join("pkg/linear.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_resolves_direct_function() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("jax/numpy")).unwrap();
        fs::write(
            tmp.path().join("jax/numpy/api.py"),
            "def concatenate(): pass",
        )
        .unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(
            target(&["jax", "numpy", "api", "concatenate"]),
            &roots,
            read,
            5,
            None,
        )
        .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["jax", "numpy", "api"],
                tmp.path().join("jax/numpy/api.py"),
                &["concatenate"],
                Some(PythonSymbol::Function {
                    name: "concatenate".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_resolves_module_itself_when_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/mod.py"), "VALUE = 1").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["pkg", "mod"]), &roots, read, 5, None).unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["pkg", "mod"],
                tmp.path().join("pkg/mod.py"),
                &[],
                None
            ))
        );
    }

    #[test]
    fn test_follows_reexport_then_resolves_class() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear: pass",
        )
        .unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["equinox", "nn", "Linear"]), &roots, read, 5, None)
                .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["equinox", "nn", "_linear"],
                tmp.path().join("equinox/nn/_linear.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["missing", "Linear"]), &roots, read, 5, None).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_read_failure_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["foo", "Foo"]), &roots, |_| None, 5, None).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Bar: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 5, None).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_max_depth_zero_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 0, None).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_reexport_preserves_remaining_symbol_parts_across_loop() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("realpkg")).unwrap();
        fs::write(tmp.path().join("aliaspkg.py"), "from realpkg import layers").unwrap();
        fs::write(tmp.path().join("realpkg/layers.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(
            target(&["aliaspkg", "layers", "Linear"]),
            &roots,
            read,
            10,
            None,
        )
        .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["realpkg", "layers"],
                tmp.path().join("realpkg/layers.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    /// Writes `files` to a root, resolves `target` there, and gives the file
    /// relative to the root and the symbol.
    fn resolve_in(files: &[(&str, &str)], target: &[&str]) -> Option<(String, Option<PythonSymbol>)> {
        let tmp = tempfile::tempdir().unwrap();
        for (path, text) in files {
            let path = tmp.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(self::target(target), &roots, read, 10, None).unwrap()?;
        let file = found.target.file_path.strip_prefix(tmp.path()).unwrap();
        Some((file.to_string_lossy().into_owned(), found.symbol))
    }

    #[test]
    fn test_resolves_like_python() {
        let function = |name: &str| Some(PythonSymbol::Function { name: name.to_string() });
        let class = |name: &str| Some(PythonSymbol::Class { name: name.to_string() });
        type Case<'a> = (&'a [(&'a str, &'a str)], &'a [&'a str], Option<(&'a str, Option<PythonSymbol>)>);
        let cases: &[Case] = &[
            // A relative import of a plain module starts from its package.
            (
                &[("pkg/__init__.py", ""), ("pkg/mod.py", "from .x import y"), ("pkg/x.py", "def y(): pass")],
                &["pkg", "mod", "y"],
                Some(("pkg/x.py", function("y"))),
            ),
            (
                &[("pkg/sub/mod.py", "from ..x import y"), ("pkg/x.py", "def y(): pass")],
                &["pkg", "sub", "mod", "y"],
                Some(("pkg/x.py", function("y"))),
            ),
            // A decorated definition is found.
            (&[("lib.py", "import jax\n@jax.jit\ndef f(x): pass")], &["lib", "f"], Some(("lib.py", function("f")))),
            (&[("lib.py", "@dataclass\nclass C:\n    x: int")], &["lib", "C"], Some(("lib.py", class("C")))),
            // CPython takes the package directory before the module file.
            (
                &[("foo.py", "def f(): pass"), ("foo/__init__.py", "def g(): pass")],
                &["foo", "g"],
                Some(("foo/__init__.py", function("g"))),
            ),
            (&[("foo.py", "def f(): pass"), ("foo/__init__.py", "def g(): pass")], &["foo", "f"], None),
            // A package that binds the name of a submodule to something else.
            (
                &[
                    ("pkg/__init__.py", "from .impl import attention"),
                    ("pkg/impl.py", "def attention(): pass"),
                    ("pkg/attention.py", "def f(): pass"),
                ],
                &["pkg", "attention", "f"],
                None,
            ),
            (
                &[("pkg/__init__.py", "attention = 1"), ("pkg/attention.py", "def f(): pass")],
                &["pkg", "attention"],
                None,
            ),
            (
                &[("pkg/__init__.py", "from . import attention"), ("pkg/attention.py", "def f(): pass")],
                &["pkg", "attention", "f"],
                Some(("pkg/attention.py", function("f"))),
            ),
            (
                &[("pkg/__init__.py", "from .attention import f"), ("pkg/attention.py", "def f(): pass")],
                &["pkg", "attention", "f"],
                Some(("pkg/attention.py", function("f"))),
            ),
            // `import a.b` and `import a.c` bind `a` once.
            (
                &[("lib.py", "import a.b\nimport a.c"), ("a/__init__.py", "def f(): pass")],
                &["lib", "a", "f"],
                Some(("a/__init__.py", function("f"))),
            ),
        ];
        for (files, target, expected) in cases {
            let expected = expected.clone().map(|(file, symbol)| (file.to_string(), symbol));
            assert_eq!(resolve_in(files, target), expected, "{target:?} in {files:?}");
        }
    }

    /// The first root with the top-level package or module owns it, as in
    /// `sys.path`. Its submodules come from that root only. A name with no
    /// regular package (a namespace package) is looked up in every root.
    #[test]
    fn test_first_root_owns_the_top_level_package() {
        type Case<'a> = (&'a [&'a str], &'a [&'a str], &'a [&'a str], Option<&'a str>);
        let cases: &[Case] = &[
            (&["pkg/__init__.py"], &["pkg/__init__.py", "pkg/sub.py"], &["pkg", "sub", "f"], None),
            (&["pkg.py"], &["pkg/__init__.py", "pkg/sub.py"], &["pkg", "sub", "f"], None),
            (&["pkg/__init__.py", "pkg/sub.py"], &["pkg/__init__.py", "pkg/sub.py"], &["pkg", "sub", "f"], Some("a/pkg/sub.py")),
            (&["other/__init__.py"], &["pkg/__init__.py", "pkg/sub.py"], &["pkg", "sub", "f"], Some("b/pkg/sub.py")),
            (&["ns/x.py"], &["ns/sub.py"], &["ns", "sub", "f"], Some("b/ns/sub.py")),
        ];
        for (a, b, target, expected) in cases {
            let tmp = tempfile::tempdir().unwrap();
            for (root, files) in [("a", a), ("b", b)] {
                for file in files.iter() {
                    let path = tmp.path().join(root).join(file);
                    fs::create_dir_all(path.parent().unwrap()).unwrap();
                    fs::write(path, "def f(): pass").unwrap();
                }
            }
            let roots = vec![tmp.path().join("a"), tmp.path().join("b")];
            let found = resolve_implementation(self::target(target), &roots, read, 10, None).unwrap();
            let file = found.map(|f| f.target.file_path.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned());
            assert_eq!(file.as_deref(), *expected, "{a:?} then {b:?}");
        }
    }

    #[test]
    fn test_cycle_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/__init__.py"), "from .a import X").unwrap();
        fs::write(tmp.path().join("pkg/a.py"), "from . import X").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["pkg", "X"]), &roots, read, 10, None).unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_call_target_tests {
    use super::*;

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|part| part.to_string()).collect(),
            name: name.to_string(),
        }
    }

    fn rt(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    #[test]
    fn test_unimported_single_segment_target_returns_itself() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo", &import_map);

        assert_eq!(resolved, rt(0, &["foo"]));
    }

    #[test]
    fn test_unimported_dotted_target_returns_itself() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo.bar.baz", &import_map);

        assert_eq!(resolved, rt(0, &["foo", "bar", "baz"]));
    }

    #[test]
    fn test_plain_import_alias_resolves_first_segment() {
        let import_map = HashMap::from([("jnp".to_string(), ip(0, &["jax"], "numpy"))]);

        let resolved = resolve_call_target("jnp.concatenate", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "numpy", "concatenate"]));
    }

    #[test]
    fn test_plain_import_without_alias_resolves_first_segment() {
        let import_map = HashMap::from([("jax".to_string(), ip(0, &[], "jax"))]);

        let resolved = resolve_call_target("jax.numpy.concatenate", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "numpy", "concatenate"]));
    }

    #[test]
    fn test_from_import_resolves_imported_name() {
        let import_map = HashMap::from([("random".to_string(), ip(0, &["jax"], "random"))]);

        let resolved = resolve_call_target("random.PRNGKey", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "random", "PRNGKey"]));
    }

    #[test]
    fn test_from_import_alias_resolves_alias() {
        let import_map = HashMap::from([("lin".to_string(), ip(0, &["equinox", "nn"], "Linear"))]);

        let resolved = resolve_call_target("lin", &import_map);

        assert_eq!(resolved, rt(0, &["equinox", "nn", "Linear"]));
    }

    #[test]
    fn test_deep_import_alias_resolves_first_segment_only() {
        let import_map = HashMap::from([("nn".to_string(), ip(0, &["equinox"], "nn"))]);

        let resolved = resolve_call_target("nn.Linear", &import_map);

        assert_eq!(resolved, rt(0, &["equinox", "nn", "Linear"]));
    }

    #[test]
    fn test_only_exact_first_segment_is_resolved() {
        let import_map = HashMap::from([("foo".to_string(), ip(0, &["real"], "foo"))]);

        let resolved = resolve_call_target("foobar.baz", &import_map);

        assert_eq!(resolved, rt(0, &["foobar", "baz"]));
    }

    #[test]
    fn test_empty_target_returns_empty_parts() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("", &import_map);

        assert_eq!(resolved, rt(0, &[]));
    }

    #[test]
    fn test_extra_dots_are_ignored() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo..bar.", &import_map);

        assert_eq!(resolved, rt(0, &["foo", "bar"]));
    }

    #[test]
    fn test_relative_import_preserves_one_dot() {
        let import_map = HashMap::from([("Linear".to_string(), ip(1, &["layers"], "Linear"))]);

        let resolved = resolve_call_target("Linear", &import_map);

        assert_eq!(resolved, rt(1, &["layers", "Linear"]));
    }

    #[test]
    fn test_relative_import_preserves_multiple_dots() {
        let import_map = HashMap::from([("Linear".to_string(), ip(2, &["layers"], "Linear"))]);

        let resolved = resolve_call_target("Linear", &import_map);

        assert_eq!(resolved, rt(2, &["layers", "Linear"]));
    }

    #[test]
    fn test_relative_import_preserves_dots_with_remaining_target_parts() {
        let import_map = HashMap::from([("layers".to_string(), ip(1, &[], "layers"))]);

        let resolved = resolve_call_target("layers.Linear", &import_map);

        assert_eq!(resolved, rt(1, &["layers", "Linear"]));
    }
}

#[cfg(test)]
mod resolve_python_module_on_disk_tests {
    use super::*;
    use std::fs;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    #[test]
    fn test_resolves_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar.py")));
    }

    #[test]
    fn test_resolves_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/__init__.py")));
    }

    #[test]
    fn test_package_init_wins_over_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/__init__.py")));
    }

    #[test]
    fn test_resolves_deeply_nested_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/qux.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar", "baz", "qux"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/baz/qux.py")));
    }

    #[test]
    fn test_resolves_deeply_nested_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz/qux")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/qux/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar", "baz", "qux"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/baz/qux/__init__.py")));
    }

    #[test]
    fn test_resolves_single_segment_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo.py")));
    }

    #[test]
    fn test_resolves_single_segment_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/__init__.py")));
    }

    #[test]
    fn test_searches_roots_in_order() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("foo.py"), "").unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(first.path().join("foo.py")));
    }

    #[test]
    fn test_searches_later_roots_if_missing_in_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(second.path().join("foo.py")));
    }

    #[test]
    fn test_empty_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&[], &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_search_roots_returns_none() {
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &[]);

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod follow_import_symbol_once_tests {
    use super::*;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    fn import(path: ImportPath) -> PythonSymbol {
        PythonSymbol::Import {
            name: path.name.clone(),
            path,
        }
    }

    #[test]
    fn test_follows_relative_import_with_module() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &import(ip(1, &["_linear"], "Linear")),
        );

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_follows_relative_import_without_module() {
        let found =
            follow_import_symbol_once(&parts(&["equinox", "nn"]), &import(ip(1, &[], "layers")));

        assert_eq!(found, Some(target(&["equinox", "nn", "layers"])));
    }

    #[test]
    fn test_follows_relative_parent_import() {
        let found = follow_import_symbol_once(
            &parts(&["pkg", "sub"]),
            &import(ip(2, &["layers"], "Linear")),
        );

        assert_eq!(found, Some(target(&["pkg", "layers", "Linear"])));
    }

    #[test]
    fn test_follows_absolute_import() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &import(ip(0, &["jax", "numpy"], "concatenate")),
        );

        assert_eq!(found, Some(target(&["jax", "numpy", "concatenate"])));
    }

    #[test]
    fn test_too_many_relative_dots_returns_none() {
        let found = follow_import_symbol_once(&parts(&["pkg"]), &import(ip(2, &["x"], "Y")));

        assert_eq!(found, None);
    }

    #[test]
    fn test_class_returns_none() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &PythonSymbol::Class {
                name: "Linear".to_string(),
            },
        );

        assert_eq!(found, None);
    }

    #[test]
    fn test_function_returns_none() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &PythonSymbol::Function {
                name: "linear".to_string(),
            },
        );

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolution_cache_tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    fn implementation(
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
        symbol: Option<PythonSymbol>,
    ) -> ResolvedImplementation {
        ResolvedImplementation {
            target: ResolvedModuleTarget {
                dots: 0,
                module_parts: parts(module_parts),
                file_path,
                symbol_parts: parts(symbol_parts),
            },
            symbol,
        }
    }

    #[test]
    fn test_cache_hit_avoids_disk_read() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let read_count = AtomicUsize::new(0);
        let read = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };

        let cache = new_resolution_cache();

        // First call — populates cache
        let found1 = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();
        let reads_first = read_count.load(Ordering::Relaxed);
        assert!(reads_first > 0, "first call should read files");

        assert_eq!(
            found1,
            Some(implementation(
                &["pkg", "linear"],
                tmp.path().join("pkg/linear.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string(),
                })
            ))
        );

        // Reset counter
        read_count.store(0, Ordering::Relaxed);

        // Second call — should hit cache, no disk reads
        let found2 = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();

        assert_eq!(found2, found1, "cache hit should return same result");
        assert_eq!(
            read_count.load(Ordering::Relaxed),
            0,
            "cache hit should not read files"
        );

        // Cache should have exactly 1 entry
        assert_eq!(cache.map.read().unwrap().len(), 1);
    }

    /// A second lookup of a constructor signature, or of a file that is not
    /// UTF-8, reads nothing until the file changes.
    #[test]
    fn test_cache_hit_reads_no_signature_and_no_unreadable_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/linear.py"), "class Linear:\n    def __init__(self, a): pass").unwrap();
        fs::write(tmp.path().join("pkg/binary.py"), [0xff, 0xfe, 0x00]).unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let read_count = AtomicUsize::new(0);
        let read = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };
        let code = "x = Linear(3)";
        let call = CallInfo {
            variable: "x".to_string(),
            target: "Linear".to_string(),
            args_node_range: tree_sitter::Range {
                start_byte: 10,
                end_byte: 13,
                start_point: tree_sitter::Point { row: 0, column: 10 },
                end_point: tree_sitter::Point { row: 0, column: 13 },
            },
        };
        let imports = HashMap::from([(
            "Linear".to_string(),
            ImportPath { dots: 0, module: parts(&["pkg", "linear"]), name: "Linear".to_string() },
        )]);
        let cache = new_resolution_cache();
        for round in 0..3 {
            if round == 2 {
                fs::write(tmp.path().join("pkg/binary.py"), "class B: pass").unwrap();
                cache.revalidate(&[tmp.path().join("pkg/binary.py")]);
            }
            read_count.store(0, Ordering::Relaxed);
            let found = resolve_call_signature(&call, code, &imports, &roots, read, 5, Some(&cache)).unwrap();
            assert_eq!(found.unwrap().bindings.get("a"), Some(&"3".to_string()));
            let binary = resolve_implementation(target(&["pkg", "binary", "B"]), &roots, read, 5, Some(&cache));
            assert_eq!(binary.unwrap().is_some(), round == 2);
            assert_eq!(read_count.load(Ordering::Relaxed), if round == 1 { 0 } else { 3 - round }, "round {round}");
        }
    }

    #[test]
    fn test_cache_stats_track_hits_and_misses() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();

        let read = |path: &PathBuf| fs::read_to_string(path).ok();

        // First call — cache miss
        let _ = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();
        assert_eq!(cache.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cache.misses.load(Ordering::Relaxed), 1);

        // Second call — cache hit
        let _ = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        assert_eq!(cache.misses.load(Ordering::Relaxed), 1);

        // Third call — another hit
        let _ = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();
        assert_eq!(cache.hits.load(Ordering::Relaxed), 2);
        assert_eq!(cache.misses.load(Ordering::Relaxed), 1);

        // Miss on a different target
        let _ = resolve_implementation(target(&["pkg", "missing"]), &roots, read, 5, Some(&cache))
            .unwrap();
        // Failed resolution is not cached, so this counts as a miss
        // (the miss was recorded at lookup time, before the disk walk)
        assert_eq!(cache.misses.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_cache_different_roots_is_separate() {
        let tmp1 = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp1.path().join("pkg")).unwrap();
        fs::write(tmp1.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let tmp2 = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp2.path().join("pkg")).unwrap();
        fs::write(tmp2.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let cache = new_resolution_cache();

        let read = |path: &PathBuf| fs::read_to_string(path).ok();

        // Resolve with roots1
        let _found1 = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &[tmp1.path().to_path_buf()],
            read,
            5,
            Some(&cache),
        )
        .unwrap();

        // Resolve with roots2 — different fingerprint, should not cache-hit
        let read_count = AtomicUsize::new(0);
        let read_counted = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };
        let _found2 = resolve_implementation(
            target(&["pkg", "linear", "Linear"]),
            &[tmp2.path().to_path_buf()],
            read_counted,
            5,
            Some(&cache),
        )
        .unwrap();

        assert!(
            read_count.load(Ordering::Relaxed) > 0,
            "different roots should miss cache"
        );
    }

    #[test]
    fn test_clear_resolution_cache_empties() {
        let cache = new_resolution_cache();
        cache.map.write().unwrap().insert(
            (parts(&["test"]), 0),
            Some(ResolvedImplementation {
                target: ResolvedModuleTarget {
                    dots: 0,
                    module_parts: parts(&["test"]),
                    file_path: PathBuf::from("fake.py"),
                    symbol_parts: vec![],
                },
                symbol: None,
            }),
        );
        cache.tracked.write().unwrap().generations.insert(PathBuf::from("fake.py"), 1);
        cache.handled_events.lock().unwrap().insert(PathBuf::from("fake.py"), None);
        assert_eq!(cache.map.read().unwrap().len(), 1);
        clear_resolution_cache(&cache);
        assert_eq!(cache.map.read().unwrap().len(), 0);
        assert!(cache.tracked.read().unwrap().generations.is_empty());
        assert!(cache.handled_events.lock().unwrap().is_empty());
    }

    #[test]
    fn test_search_roots_fingerprint_same_roots_same_hash() {
        let roots = vec![PathBuf::from("/a/b"), PathBuf::from("/c/d")];
        let h1 = search_roots_fingerprint(&roots);
        let h2 = search_roots_fingerprint(&roots);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_search_roots_fingerprint_different_roots_different_hash() {
        let roots1 = vec![PathBuf::from("/a/b")];
        let roots2 = vec![PathBuf::from("/x/y")];
        assert_ne!(
            search_roots_fingerprint(&roots1),
            search_roots_fingerprint(&roots2)
        );
    }

    #[test]
    fn test_search_roots_fingerprint_order_matters() {
        let roots1 = vec![PathBuf::from("/a/b"), PathBuf::from("/c/d")];
        let roots2 = vec![PathBuf::from("/c/d"), PathBuf::from("/a/b")];
        assert_ne!(
            search_roots_fingerprint(&roots1),
            search_roots_fingerprint(&roots2)
        );
    }
}

#[cfg(test)]
mod resolve_imported_function_shape_tests {
    use super::*;
    use std::fs;

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|p| p.to_string()).collect(),
            name: name.to_string(),
        }
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|d| d.to_string()).collect()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    #[test]
    fn test_resolves_function_shape_from_another_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map =
            HashMap::from([("project".to_string(), ip(0, &["mylib", "helpers"], "project"))]);

        let found =
            resolve_imported_function_shape("project", &import_map, &roots, read, 5, None)
                .unwrap()
                .unwrap();

        assert_eq!(found.function_name.as_deref(), Some("project"));
        assert_eq!(
            found.shapes.get("x"),
            Some(&shape(&["batch", "d_in"]))
        );
        assert_eq!(found.return_shape, Some(shape(&["batch", "d_out"])));
    }

    #[test]
    fn test_resolves_to_class_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("mylib.py"), "class Linear: pass").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map = HashMap::from([("Linear".to_string(), ip(0, &["mylib"], "Linear"))]);

        let found = resolve_imported_function_shape("Linear", &import_map, &roots, read, 5, None)
            .unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map =
            HashMap::from([("project".to_string(), ip(0, &["mylib", "helpers"], "project"))]);

        let found =
            resolve_imported_function_shape("project", &import_map, &roots, read, 5, None)
                .unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_function_without_annotations_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x):\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map =
            HashMap::from([("project".to_string(), ip(0, &["mylib", "helpers"], "project"))]);

        let found =
            resolve_imported_function_shape("project", &import_map, &roots, read, 5, None)
                .unwrap()
                .unwrap();

        // The scope is still returned (a function was found); it just has
        // no annotations for the caller to bind against.
        assert!(found.shapes.is_empty());
        assert_eq!(found.return_shape, None);
    }

    #[test]
    fn test_caches_signature_across_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"a\"]) -> Float[Array, \"b\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map =
            HashMap::from([("project".to_string(), ip(0, &["mylib", "helpers"], "project"))]);

        let read_count = AtomicUsize::new(0);
        let counting_read = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };

        let cache = new_resolution_cache();

        let found1 = resolve_imported_function_shape(
            "project",
            &import_map,
            &roots,
            counting_read,
            5,
            Some(&cache),
        )
        .unwrap();
        assert!(read_count.load(Ordering::Relaxed) > 0);

        read_count.store(0, Ordering::Relaxed);

        let found2 = resolve_imported_function_shape(
            "project",
            &import_map,
            &roots,
            counting_read,
            5,
            Some(&cache),
        )
        .unwrap();

        assert_eq!(found2, found1);
        assert_eq!(
            read_count.load(Ordering::Relaxed),
            0,
            "second lookup should hit both the implementation and signature caches"
        );
    }
}

#[cfg(test)]
mod cache_freshness_tests {
    use super::*;
    use std::fs;

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: parts.iter().map(|p| p.to_string()).collect(),
        }
    }

    fn resolve(roots: &[PathBuf], cache: &ResolutionCache) -> Option<ResolvedImplementation> {
        cache.revalidate(&[]);
        resolve_implementation(target(&["lib", "f"]), roots, read, 5, Some(cache)).unwrap()
    }

    fn symbol_name(found: &Option<ResolvedImplementation>) -> Option<&str> {
        match found.as_ref()?.symbol.as_ref()? {
            PythonSymbol::Function { name } | PythonSymbol::Class { name, .. } => Some(name),
            PythonSymbol::Import { .. } => None,
        }
    }

    #[test]
    fn modified_file_is_resolved_again() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        fs::write(&file, "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        assert_eq!(symbol_name(&resolve(&roots, &cache)), Some("f"));

        fs::write(&file, "x = 1\n").unwrap();
        assert!(resolve(&roots, &cache).is_none());
    }

    #[test]
    fn deleted_file_is_resolved_again() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        fs::write(&file, "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        assert!(resolve(&roots, &cache).is_some());

        fs::remove_file(&file).unwrap();
        assert!(resolve(&roots, &cache).is_none());
    }

    #[test]
    fn failed_resolution_sees_a_created_file() {
        let tmp = tempfile::tempdir().unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        assert!(resolve(&roots, &cache).is_none());
        assert!(resolve(&roots, &cache).is_none());
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1, "failure is cached");

        fs::write(tmp.path().join("lib.py"), "def f(): pass\n").unwrap();
        assert_eq!(symbol_name(&resolve(&roots, &cache)), Some("f"));
    }

    #[test]
    fn created_file_in_an_earlier_root_wins() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("lib.py"), "def f(): pass\n").unwrap();
        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let cache = new_resolution_cache();
        let before = resolve(&roots, &cache).unwrap();
        assert_eq!(before.target.file_path, second.path().join("lib.py"));

        fs::write(first.path().join("lib.py"), "def f(): pass\n").unwrap();
        let after = resolve(&roots, &cache).unwrap();
        assert_eq!(after.target.file_path, first.path().join("lib.py"));
    }

    #[test]
    fn unchanged_files_keep_the_cache() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("lib.py"), "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        resolve(&roots, &cache);
        resolve(&roots, &cache);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        assert!(!cache.revalidate(&[tmp.path().join("other.py")]));
    }

    #[test]
    fn editor_event_drops_the_cache_without_a_stamp_change() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        fs::write(&file, "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        resolve(&roots, &cache);
        assert!(cache.revalidate(&[file]));
        assert!(cache.map.read().unwrap().is_empty());
    }

    #[test]
    fn event_during_a_resolution_skips_its_insert() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        fs::write(&file, "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        let read_then_event = |path: &PathBuf| {
            let text = read(path);
            assert!(!cache.revalidate(std::slice::from_ref(path)));
            text
        };
        resolve_implementation(target(&["lib", "f"]), &roots, read_then_event, 5, Some(&cache))
            .unwrap();
        assert!(cache.map.read().unwrap().is_empty());
    }

    #[test]
    fn change_drops_only_the_entries_that_looked_at_it() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("lib.py"), "def f(): pass\n").unwrap();
        fs::write(tmp.path().join("other.py"), "def g(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        resolve(&roots, &cache);
        resolve_implementation(target(&["other", "g"]), &roots, read, 5, Some(&cache)).unwrap();

        fs::write(tmp.path().join("lib.py"), "x = 1\n").unwrap();
        assert!(cache.revalidate(&[]));
        assert_eq!(cache.map.read().unwrap().len(), 1);
        let hits = cache.hits.load(Ordering::Relaxed);
        resolve_implementation(target(&["other", "g"]), &roots, read, 5, Some(&cache)).unwrap();
        assert_eq!(cache.hits.load(Ordering::Relaxed), hits + 1);
    }

    #[test]
    fn second_event_for_the_same_change_keeps_the_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        fs::write(&file, "def f(): pass\n").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cache = new_resolution_cache();
        resolve(&roots, &cache);
        assert!(cache.revalidate(std::slice::from_ref(&file)));
        resolve(&roots, &cache);
        assert!(!cache.revalidate(std::slice::from_ref(&file)));
    }

    #[test]
    fn signature_cache_sees_a_modified_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("lib.py");
        let sig = |ret: &str| {
            format!("def f(x: Float[Array, \"a b\"]) -> Float[Array, \"{ret}\"]:\n    pass\n")
        };
        fs::write(&file, sig("a b")).unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let import_map = HashMap::from([(
            "f".to_string(),
            ImportPath {
                dots: 0,
                module: vec!["lib".to_string()],
                name: "f".to_string(),
            },
        )]);
        let cache = new_resolution_cache();
        let get = || {
            cache.revalidate(&[]);
            resolve_imported_function_shape("f", &import_map, &roots, read, 5, Some(&cache))
                .unwrap()
                .unwrap()
        };
        let before = get();
        fs::write(&file, sig("b a")).unwrap();
        assert_ne!(get().return_shape, before.return_shape);
    }
}

#[cfg(test)]
mod record_deps_tests {
    use super::*;

    #[test]
    fn a_panic_in_a_recorded_computation_ends_the_recording() {
        let cache = new_resolution_cache();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.record_deps(|| panic!("extractor failed"))
        }));
        assert!(caught.is_err());
        assert!(!recording());
    }

    #[test]
    fn a_nested_recording_counts_for_the_enclosing_one() {
        let cache = new_resolution_cache();
        let path: Arc<[PathBuf]> = Arc::from(vec![PathBuf::from("/outer.py")]);
        let (_, outer) = cache.record_deps(|| {
            cache.record_deps(|| touch(&Arc::from(vec![PathBuf::from("/inner.py")])));
            touch(&path);
        });
        let outer = outer.unwrap();
        let paths: Vec<_> = outer.paths.iter().map(|(p, _)| p.to_str().unwrap()).collect();
        assert_eq!(paths, ["/inner.py", "/outer.py"]);
        assert!(cache.deps_valid(&outer));
    }

}

#[cfg(test)]
mod relative_import_tests {
    use super::*;
    use std::fs;

    fn parse(text: &str) -> Tree {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        parser.parse(text, None).unwrap()
    }

    fn read(p: &PathBuf) -> Option<String> {
        fs::read_to_string(p).ok()
    }

    /// `<tmp>/pkg/__init__.py`, `<tmp>/pkg/sub/__init__.py`, and `files`.
    fn package(files: &[(&str, &str)]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for (name, text) in [("pkg/__init__.py", ""), ("pkg/sub/__init__.py", "")].iter().chain(files) {
            fs::create_dir_all(root.join(name).parent().unwrap()).unwrap();
            fs::write(root.join(name), text).unwrap();
        }
        (tmp, root)
    }

    /// The memo of a document sees a save of the file that a relative import
    /// of the document names, as for an absolute import.
    #[test]
    fn a_memo_sees_a_saved_relative_import() {
        let helper = |ret: &str| {
            format!("from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"{ret}\"]:\n    return x @ w\n")
        };
        let cases = [
            ("pkg/main.py", "from .helpers import proj", "proj"),
            ("pkg/main.py", "from . import helpers", "helpers.proj"),
            ("pkg/sub/main.py", "from .. import helpers", "helpers.proj"),
        ];
        for (document, import, call) in cases {
            let (_tmp, root) = package(&[("pkg/helpers.py", &helper("b k"))]);
            let path = root.join("pkg/helpers.py");
            let roots = vec![root.clone()];
            let cache = new_resolution_cache();
            let text = format!(
                "from jaxtyping import Float, Array\n{import}\n\ndef f(x: Float[Array, \"b 4\"], w: Float[Array, \"4 8\"], v: Float[Array, \"8 3\"]):\n    y = {call}(x, w) @ v\n"
            );
            let tree = parse(&text);
            let mut memo = crate::incremental::UnitMemo::default();
            let errors = |memo: &mut crate::incremental::UnitMemo| {
                with_document(Some(&root.join(document)), || {
                    crate::analysis::analyze_layer_shapes_with_memo(
                        tree.root_node(), &text, &roots, read, 8, Some(&cache), memo,
                    )
                    .unwrap()
                    .errors
                    .len()
                })
            };
            assert_eq!(errors(&mut memo), 0, "{import}");
            assert_eq!(errors(&mut memo), 0, "{import}");
            assert!(memo.stats().layer_hits > 0, "{import}: no memo reuse");
            fs::write(&path, helper("b d")).unwrap();
            cache.revalidate(std::slice::from_ref(&path));
            assert_eq!(errors(&mut memo), 1, "{import}: stale memo after a save");
        }
    }

    /// A module class of another file is analyzed without the path of the
    /// document: its relative import does not resolve next to the document.
    /// The same import written absolute gives the shape.
    #[test]
    fn a_nested_analysis_does_not_see_the_document() {
        let block = |import: &str| {
            format!("import torch.nn as nn\nfrom jaxtyping import Float, Array\nfrom {import} import proj\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        return proj(x)\n")
        };
        let helper = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"n 8\"]) -> Float[Array, \"n 5\"]:\n    return x\n";
        let main = "from jaxtyping import Float, Array\nfrom pkg.blocks import Block\n\ndef f(x: Float[Array, \"n 8\"]):\n    y = Block()(x)\n";
        for (import, want) in [("pkg.sub.helpers", Some(vec!["n".to_string(), "5".to_string()])), (".helpers", None)] {
            let (_tmp, root) = package(&[("pkg/sub/helpers.py", helper), ("pkg/blocks.py", &block(import))]);
            let roots = vec![root.clone()];
            let cache = new_resolution_cache();
            let tree = parse(main);
            let analysis = with_document(Some(&root.join("pkg/sub/main.py")), || {
                crate::analysis::analyze_layer_shapes(tree.root_node(), main, &roots, read, 8, Some(&cache)).unwrap()
            });
            let y = analysis
                .scopes
                .iter()
                .find(|s| s.function_name.as_deref() == Some("f"))
                .and_then(|s| s.shapes.get("y"))
                .cloned();
            assert_eq!(y, want, "{import}");
        }
    }
}
