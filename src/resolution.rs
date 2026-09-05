use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
};

use tree_sitter::{Node, Parser};

use crate::python_ast::{
    extract_call_arguments, extract_callable_signature, extract_jaxtyping_shapes,
    find_top_level_symbol,
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
/// imports, not intermediate re-export hops).
pub struct ResolutionCache {
    pub map: std::sync::RwLock<HashMap<(Vec<String>, u64), ResolvedImplementation>>,
    /// Session-lifetime cache for extracted jaxtyping `FunctionShapeScope`s of
    /// cross-file helper functions (cross-file return-type tracing). Keyed by
    /// the resolved implementation's file path plus function name, since one
    /// file can define several functions. `None` caches a lookup that found
    /// the function but no matching jaxtyping scope, so repeat calls to an
    /// unannotated cross-file helper don't re-parse the file.
    pub signatures: std::sync::RwLock<HashMap<(PathBuf, String), Option<FunctionShapeScope>>>,
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
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
    Arc::new(ResolutionCache {
        map: std::sync::RwLock::new(HashMap::new()),
        signatures: std::sync::RwLock::new(HashMap::new()),
        hits: AtomicUsize::new(0),
        misses: AtomicUsize::new(0),
    })
}

/// Clear all entries from the resolution cache. Called when workspace
/// folders change, since site-packages roots may shift.
pub fn clear_resolution_cache(cache: &Arc<ResolutionCache>) {
    cache.map.write().unwrap().clear();
    cache.signatures.write().unwrap().clear();
}

pub fn resolve_python_module_on_disk(
    module: &[String],
    search_roots: &[PathBuf],
) -> Option<PathBuf> {
    if module.is_empty() {
        return None;
    }
    for root in search_roots {
        let module_path = root.join(module.join("/"));
        let file_path = module_path.with_extension("py");
        if file_path.exists() {
            return Some(file_path);
        }

        let package_path = module_path.join("__init__.py");
        if package_path.exists() {
            return Some(package_path);
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

pub fn resolve_target_on_disk(
    target: &ResolvedTarget,
    search_roots: &[PathBuf],
) -> Option<ResolvedModuleTarget> {
    if target.dots > 0 || target.parts.is_empty() {
        return None;
    }

    for len in (1..=target.parts.len()).rev() {
        let module_parts = &target.parts[..len];
        let symbol_parts = &target.parts[len..];

        if let Some(file_path) = resolve_python_module_on_disk(module_parts, search_roots) {
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

pub fn resolve_reexport_once(
    resolved: &ResolvedModuleTarget,
    node: Node,
    text: &str,
) -> Result<Option<ResolvedTarget>, String> {
    let Some(first) = resolved.symbol_parts.first() else {
        return Ok(None);
    };

    let Some(symbol) = find_top_level_symbol(node, text, first)? else {
        return Ok(None);
    };

    let Some(mut next) = follow_import_symbol_once(&resolved.module_parts, &symbol) else {
        return Ok(None);
    };

    next.parts
        .extend(resolved.symbol_parts.iter().skip(1).cloned());
    Ok(Some(next))
}

pub fn resolve_terminal_symbol_once(
    resolved: &ResolvedModuleTarget,
    node: Node,
    text: &str,
) -> Result<Option<PythonSymbol>, String> {
    let Some(first) = resolved.symbol_parts.first() else {
        return Ok(None);
    };

    let Some(symbol) = find_top_level_symbol(node, text, first)? else {
        return Ok(None);
    };

    match symbol {
        PythonSymbol::Class { .. } | PythonSymbol::Function { .. } => Ok(Some(symbol)),
        PythonSymbol::Import { .. } => Ok(None),
    }
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
    // Cache lookup — only for absolute (dots==0) targets with non-empty parts.
    if start.dots == 0
        && !start.parts.is_empty()
        && let Some(cache) = cache
    {
        let fingerprint = search_roots_fingerprint(search_roots);
        let key = (start.parts.clone(), fingerprint);
        let cache_map = cache.map.read().unwrap();
        if let Some(cached) = cache_map.get(&key) {
            cache.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(cached.clone()));
        }
        cache.misses.fetch_add(1, Ordering::Relaxed);
    }

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;

    let mut current = start.clone();

    let mut visited = HashSet::new();

    let result = (|| {
        for _ in 0..max_depth {
            if !visited.insert(current.parts.clone()) {
                return Ok(None);
            }
            let Some(resolved) = resolve_target_on_disk(&current, search_roots) else {
                return Ok(None);
            };

            if resolved.symbol_parts.is_empty() {
                return Ok(Some(ResolvedImplementation {
                    target: resolved,
                    symbol: None,
                }));
            }

            let Some(text) = read_file(&resolved.file_path) else {
                return Ok(None);
            };

            let Some(tree) = parser.parse(&text, None) else {
                return Err("failed to parse file".to_string());
            };

            let root = tree.root_node();

            if let Some(symbol) = resolve_terminal_symbol_once(&resolved, root, &text)? {
                return Ok(Some(ResolvedImplementation {
                    target: resolved,
                    symbol: Some(symbol),
                }));
            }

            if let Some(next) = resolve_reexport_once(&resolved, root, &text)? {
                current = next;
                continue;
            }

            return Ok(None);
        }

        Ok(None)
    })();

    // Cache insert — only on successful resolution of absolute targets.
    if start.dots == 0
        && !start.parts.is_empty()
        && let (Some(cache), Ok(Some(impl_result))) = (cache, &result)
    {
        let fingerprint = search_roots_fingerprint(search_roots);
        let key = (start.parts.clone(), fingerprint);
        let mut cache_map = cache.map.write().unwrap();
        cache_map.insert(key, impl_result.clone());
    }

    result
}

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

    let Some(implementation_text) = read_file(&implementation.target.file_path) else {
        return Ok(None);
    };

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;

    let Some(implementation_tree) = parser.parse(&implementation_text, None) else {
        return Err("failed to parse implementation file".to_string());
    };

    let Some(signature) = extract_callable_signature(
        implementation_tree.root_node(),
        &implementation_text,
        symbol,
    )?
    else {
        return Ok(None);
    };

    let source_tree;
    let source_node = match source_node {
        Some(node) => node,
        None => {
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
    let bindings = bind_call_arguments(&signature, &arguments);

    Ok(Some(ResolvedCallSignature {
        implementation,
        signature,
        arguments,
        bindings,
    }))
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

    let cache_key = (implementation.target.file_path.clone(), name.clone());
    if let Some(cache) = cache
        && let Some(cached) = cache.signatures.read().unwrap().get(&cache_key)
    {
        return Ok(cached.clone());
    }

    let Some(text) = read_file(&implementation.target.file_path) else {
        return Ok(None);
    };

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;
    let Some(tree) = parser.parse(&text, None) else {
        return Err("failed to parse implementation file".to_string());
    };

    let scopes = extract_jaxtyping_shapes(tree.root_node(), &text)?;
    let found = scopes
        .iter()
        .enumerate()
        .filter(|(i, scope)| *i != 0 && scope.function_name.as_deref() == Some(name.as_str()))
        .min_by_key(|(_, scope)| scope.end_byte - scope.start_byte)
        .map(|(_, scope)| scope.clone());

    if let Some(cache) = cache {
        cache
            .signatures
            .write()
            .unwrap()
            .insert(cache_key, found.clone());
    }

    Ok(found)
}

pub fn bind_call_arguments(
    signature: &PythonCallableSignature,
    args: &[CallArgument],
) -> HashMap<String, String> {
    let mut params = signature.params.as_slice();
    if signature.owner.is_some()
        && matches!(params.first(), Some(first) if first == "self" || first == "cls")
    {
        params = &params[1..];
    }

    let mut bindings = HashMap::new();
    let mut positional_index = 0;

    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(param) = params.get(positional_index) {
                    bindings.insert((*param).clone(), value.clone());
                }
                positional_index += 1;
            }
            CallArgument::Keyword { name, value } => {
                bindings.insert(name.clone(), value.clone());
            }
        }
    }

    bindings
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
    fn test_module_file_wins_over_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar.py")));
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
mod resolve_reexport_once_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn resolved(module_parts: &[&str], symbol_parts: &[&str]) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots: 0,
            module_parts: parts(module_parts),
            file_path: PathBuf::from("unused.py"),
            symbol_parts: parts(symbol_parts),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    #[test]
    fn test_follows_relative_reexport() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_follows_relative_package_reexport() {
        let code = "from . import layers";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["layers", "Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["equinox", "nn", "layers", "Linear"])));
    }

    #[test]
    fn test_preserves_remaining_symbol_parts_after_reexport() {
        let code = "from . import layers";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["layers", "nn", "Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["pkg", "layers", "nn", "Linear"])));
    }

    #[test]
    fn test_follows_absolute_reexport() {
        let code = "from jax.numpy import concatenate";
        let tree = parse(code);
        let current = resolved(&["my", "pkg"], &["concatenate"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["jax", "numpy", "concatenate"])));
    }

    #[test]
    fn test_class_returns_none() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_function_returns_none() {
        let code = "def linear(): pass";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "from ._linear import Other";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_symbol_parts_returns_none() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &[]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_too_many_relative_dots_returns_none() {
        let code = "from ..x import Y";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["Y"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_terminal_symbol_once_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn resolved(module_parts: &[&str], symbol_parts: &[&str]) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots: 0,
            module_parts: parts(module_parts),
            file_path: PathBuf::from("unused.py"),
            symbol_parts: parts(symbol_parts),
        }
    }

    #[test]
    fn test_returns_class_symbol() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn", "_linear"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Class {
                name: "Linear".to_string()
            })
        );
    }

    #[test]
    fn test_returns_function_symbol() {
        let code = "def concatenate(): pass";
        let tree = parse(code);
        let current = resolved(&["jax", "numpy"], &["concatenate"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Function {
                name: "concatenate".to_string()
            })
        );
    }

    #[test]
    fn test_import_symbol_returns_none() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "class Other: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_symbol_parts_returns_none() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &[]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_uses_first_symbol_part_only() {
        let code = "class nn: pass";
        let tree = parse(code);
        let current = resolved(&["torch"], &["nn", "Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Class {
                name: "nn".to_string()
            })
        );
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
            ResolvedImplementation {
                target: ResolvedModuleTarget {
                    dots: 0,
                    module_parts: parts(&["test"]),
                    file_path: PathBuf::from("fake.py"),
                    symbol_parts: vec![],
                },
                symbol: None,
            },
        );
        assert_eq!(cache.map.read().unwrap().len(), 1);
        clear_resolution_cache(&cache);
        assert_eq!(cache.map.read().unwrap().len(), 0);
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
