use std::{collections::HashMap, path::PathBuf};

use tree_sitter::{Node, Parser, Query, QueryCursor, Range, StreamingIterator};

#[derive(Debug, PartialEq, Clone)]
pub struct ImportPath {
    pub dots: usize,
    pub module: Vec<String>,
    pub name: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct CallInfo {
    pub variable: String,
    pub target: String,
    pub args_node_range: tree_sitter::Range,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedTarget {
    pub dots: usize,
    pub parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedModuleTarget {
    pub dots: usize,
    pub module_parts: Vec<String>,
    pub file_path: PathBuf,
    pub symbol_parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PythonSymbol {
    Class { name: String },
    Function { name: String },
    Import { name: String, path: ImportPath },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedImplementation {
    pub target: ResolvedModuleTarget,
    pub symbol: Option<PythonSymbol>,
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

    Ok(follow_import_symbol_once(&resolved.module_parts, &symbol))
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
) -> Result<Option<ResolvedImplementation>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;

    let mut current = start;

    for _ in 0..max_depth {
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
}

pub fn find_top_level_symbol(
    node: Node,
    text: &str,
    name: &str,
) -> Result<Option<PythonSymbol>, String> {
    let query_string = r#"
        (module (class_definition name: (_) @cls_def))
        (module (function_definition name: (_) @fn_def))
        (module (import_statement) @import)
        (module (import_from_statement) @import)
    "#;

    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;
    let Some(class_idx) = query.capture_index_for_name("cls_def") else {
        return Err("Failed to find capture index for 'cls_def'".to_string());
    };

    let Some(fn_idx) = query.capture_index_for_name("fn_def") else {
        return Err("Failed to find capture index for 'fn_def'".to_string());
    };
    let Some(import_idx) = query.capture_index_for_name("import") else {
        return Err("Failed to find capture index for 'import'".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());
    let mut found = None;

    while let Some(match_) = matches.next() {
        match match_.pattern_index {
            0 => {
                for capture in match_.captures {
                    if capture.index == class_idx {
                        let class_name = capture
                            .node
                            .utf8_text(text.as_bytes())
                            .map_err(|e| e.to_string())?;

                        if class_name == name {
                            found = Some(PythonSymbol::Class {
                                name: class_name.to_string(),
                            });
                        }
                    }
                }
            }
            1 => {
                for capture in match_.captures {
                    if capture.index == fn_idx {
                        let fn_name = capture
                            .node
                            .utf8_text(text.as_bytes())
                            .map_err(|e| e.to_string())?;

                        if fn_name == name {
                            found = Some(PythonSymbol::Function {
                                name: fn_name.to_string(),
                            });
                        }
                    }
                }
            }
            2 | 3 => {
                for capture in match_.captures {
                    if capture.index == import_idx {
                        let import_map = build_import_map(capture.node, text)?;
                        if let Some(path) = import_map.get(name) {
                            found = Some(PythonSymbol::Import {
                                name: name.to_string(),
                                path: path.clone(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(found)
}

pub fn build_import_map(node: Node, text: &str) -> Result<HashMap<String, ImportPath>, String> {
    fn plain_import_dotted_name_to_vec(
        dotted_name: &str,
    ) -> Result<(Vec<String>, String, String), String> {
        let parts: Vec<&str> = dotted_name.split(".").collect();
        let module: Vec<String> = parts[..parts.len() - 1]
            .iter()
            .map(|p| p.to_string())
            .collect();
        let Some(name) = parts.last() else {
            return Err("Failed to fetch the last part of the import path".to_string());
        };
        let Some(first) = parts.first() else {
            return Err("Empty import path".to_string());
        };
        Ok((module, name.to_string(), first.to_string()))
    }

    let query_string = r#"
        (import_statement name: (_) @import)
        (import_from_statement module_name: (_) @module_name name: (_) @name)
    "#;
    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;

    let Some(import_statement_idx) = query.capture_index_for_name("import") else {
        return Err("Failed to capture index for name 'import'".to_string());
    };

    let Some(module_name_idx) = query.capture_index_for_name("module_name") else {
        return Err("Failed to capture index for name 'module_name'".to_string());
    };
    let Some(name_idx) = query.capture_index_for_name("name") else {
        return Err("Failed to capture index for name 'name'".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());

    let mut import_map: HashMap<String, ImportPath> = HashMap::new();

    while let Some(match_) = matches.next() {
        if match_.pattern_index == 0 {
            for capture in match_.captures {
                if capture.index == import_statement_idx {
                    match capture.node.kind() {
                        "dotted_name" => {
                            let dotted_name = capture
                                .node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let (module, name, first) =
                                plain_import_dotted_name_to_vec(dotted_name)
                                    .map_err(|e| e.to_string())?;
                            import_map.insert(
                                first.to_string(),
                                ImportPath {
                                    dots: 0,
                                    module,
                                    name: name.to_string(),
                                },
                            );
                        }
                        "aliased_import" => {
                            let Some(name_child) = capture.node.child_by_field_name("name") else {
                                return Err("aliased_import missing name".to_string());
                            };
                            let dotted_name = name_child
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let (module, name, _) = plain_import_dotted_name_to_vec(dotted_name)?;

                            let Some(alias_node) = capture.node.child_by_field_name("alias") else {
                                return Err("aliased_import missing alias".to_string());
                            };
                            let alias = alias_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();

                            import_map.insert(
                                alias,
                                ImportPath {
                                    dots: 0,
                                    module,
                                    name,
                                },
                            );
                        }
                        _ => {}
                    }
                }
            }
        } else if match_.pattern_index == 1 {
            let mut module: Vec<String> = Vec::new();
            let mut name = String::new();
            let mut alias = String::new();
            let mut dots = 0;
            for capture in match_.captures {
                if capture.index == module_name_idx {
                    let n = capture.node;
                    match n.kind() {
                        "dotted_name" => {
                            let identifier =
                                n.utf8_text(text.as_bytes()).map_err(|e| e.to_string())?;
                            module = identifier.split(".").map(|f| f.to_string()).collect();
                        }
                        "relative_import" => {
                            let mut child_cursor = n.walk();
                            for child in n.named_children(&mut child_cursor) {
                                match child.kind() {
                                    "import_prefix" => {
                                        let prefix = child
                                            .utf8_text(text.as_bytes())
                                            .map_err(|e| e.to_string())?;
                                        dots = prefix.len();
                                    }
                                    "dotted_name" => {
                                        let dotted = child
                                            .utf8_text(text.as_bytes())
                                            .map_err(|e| e.to_string())?;
                                        module = dotted.split(".").map(|f| f.to_string()).collect();
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                } else if capture.index == name_idx {
                    let n = capture.node;
                    match n.kind() {
                        "dotted_name" => {
                            name = n
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                            alias = name.clone();
                        }
                        "aliased_import" => {
                            let Some(alias_child_node) = n.child_by_field_name("alias") else {
                                return Err("aliased_import has no alias field".to_string());
                            };
                            alias = alias_child_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();

                            let Some(dotted_name_idx) = n.child_by_field_name("name") else {
                                return Err("aliased_import has no name field".to_string());
                            };
                            let dotted_name = dotted_name_idx
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                            name = dotted_name;
                        }
                        _ => {}
                    }
                }
            }

            import_map.insert(alias, ImportPath { dots, module, name });
        }
    }

    Ok(import_map)
}

pub fn extract_calls(node: Node, text: &str) -> Result<Vec<CallInfo>, String> {
    let mut result: Vec<CallInfo> = Vec::new();
    let query_string = r#"
        (assignment
          left: (identifier) @var
          right: (call
            function: [
              (identifier) @fn_id
              (attribute) @fn_attr
            ]
          )
        ) @assignment
    "#;

    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;

    let Some(var_idx) = query.capture_index_for_name("var") else {
        return Err("var not found".to_string());
    };

    let Some(fn_id_idx) = query.capture_index_for_name("fn_id") else {
        return Err("fn_id not found".to_string());
    };

    let Some(fn_attr_idx) = query.capture_index_for_name("fn_attr") else {
        return Err("fn_attr not found".to_string());
    };

    let Some(assignment_idx) = query.capture_index_for_name("assignment") else {
        return Err("assignment not found".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut variable = String::new();
        let mut target = String::new();
        let mut args_node_range: Option<Range> = None;

        for capture in match_.captures {
            if capture.index == var_idx {
                variable = capture
                    .node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string();
            } else if capture.index == assignment_idx {
                let Some(right_child_node) = capture.node.child_by_field_name("right") else {
                    return Err("right child not found".to_string());
                };

                let Some(arguments_node) = right_child_node.child_by_field_name("arguments") else {
                    return Err("arguments child not found".to_string());
                };
                args_node_range = Some(arguments_node.range());
            } else if capture.index == fn_id_idx || capture.index == fn_attr_idx {
                target = capture
                    .node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string();
            }
        }

        let args_node_range = args_node_range.ok_or("args_node_range not found".to_string())?;
        result.push(CallInfo {
            variable,
            target,
            args_node_range,
        });
    }

    Ok(result)
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
            resolve_implementation(target(&["pkg", "linear", "Linear"]), &roots, read, 5).unwrap();

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
        let found = resolve_implementation(target(&["pkg", "mod"]), &roots, read, 5).unwrap();

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
            resolve_implementation(target(&["equinox", "nn", "Linear"]), &roots, read, 5).unwrap();

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
            resolve_implementation(target(&["missing", "Linear"]), &roots, read, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_read_failure_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, |_| None, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Bar: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_max_depth_zero_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 0).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_cycle_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/__init__.py"), "from .a import X").unwrap();
        fs::write(tmp.path().join("pkg/a.py"), "from . import X").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["pkg", "X"]), &roots, read, 10).unwrap();

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

        assert_eq!(found, Some(target(&["equinox", "nn", "layers"])));
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
mod find_top_level_symbol_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|part| part.to_string()).collect(),
            name: name.to_string(),
        }
    }

    fn class(name: &str) -> PythonSymbol {
        PythonSymbol::Class {
            name: name.to_string(),
        }
    }

    fn function(name: &str) -> PythonSymbol {
        PythonSymbol::Function {
            name: name.to_string(),
        }
    }

    fn import(name: &str, path: ImportPath) -> PythonSymbol {
        PythonSymbol::Import {
            name: name.to_string(),
            path,
        }
    }

    #[test]
    fn test_finds_top_level_class() {
        let code = "class Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(class("Linear")));
    }

    #[test]
    fn test_finds_top_level_function() {
        let code = "def linear(): pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "linear").unwrap();

        assert_eq!(found, Some(function("linear")));
    }

    #[test]
    fn test_finds_plain_import() {
        let code = "import foo";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "foo").unwrap();

        assert_eq!(found, Some(import("foo", ip(0, &[], "foo"))));
    }

    #[test]
    fn test_finds_plain_import_alias() {
        let code = "import foo as bar";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "bar").unwrap();

        assert_eq!(found, Some(import("bar", ip(0, &[], "foo"))));
    }

    #[test]
    fn test_finds_deep_import_alias() {
        let code = "import foo.bar as baz";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "baz").unwrap();

        assert_eq!(found, Some(import("baz", ip(0, &["foo"], "bar"))));
    }

    #[test]
    fn test_finds_from_import() {
        let code = "from jax import random";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "random").unwrap();

        assert_eq!(found, Some(import("random", ip(0, &["jax"], "random"))));
    }

    #[test]
    fn test_finds_from_import_alias() {
        let code = "from equinox.nn import Linear as Lin";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Lin").unwrap();

        assert_eq!(
            found,
            Some(import("Lin", ip(0, &["equinox", "nn"], "Linear")))
        );
    }

    #[test]
    fn test_finds_relative_from_import() {
        let code = "from ._linear import Linear";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(import("Linear", ip(1, &["_linear"], "Linear"))));
    }

    #[test]
    fn test_finds_relative_package_import() {
        let code = "from . import layers";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "layers").unwrap();

        assert_eq!(found, Some(import("layers", ip(1, &[], "layers"))));
    }

    #[test]
    fn test_skips_nested_class() {
        let code = "def outer():\n    class Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_skips_nested_function() {
        let code = "def outer():\n    def inner(): pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "inner").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_star_import_does_not_define_named_symbol() {
        let code = "from foo import *";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "class Other: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_last_top_level_definition_wins() {
        let code = "from foo import Linear\nclass Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(class("Linear")));
    }

    #[test]
    fn test_last_top_level_import_wins() {
        let code = "class Linear: pass\nfrom foo import Linear";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(import("Linear", ip(0, &["foo"], "Linear"))));
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
mod resolve_target_on_disk_tests {
    use super::*;
    use std::fs;

    fn target(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    fn expected(
        dots: usize,
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
    ) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots,
            module_parts: module_parts.iter().map(|part| part.to_string()).collect(),
            file_path,
            symbol_parts: symbol_parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    #[test]
    fn test_resolves_exact_module_file_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_exact_package_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz/__init__.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_longest_module_prefix_and_keeps_symbol() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_falls_back_to_shorter_module_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                tmp.path().join("foo.py"),
                &["bar", "Baz"]
            ))
        );
    }

    #[test]
    fn test_prefers_longer_module_over_shorter_module() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_module_file_wins_over_package_init_for_same_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_searches_roots_in_order_for_same_prefix() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("foo.py"), "").unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(0, &["foo"], first.path().join("foo.py"), &["Bar"]))
        );
    }

    #[test]
    fn test_searches_later_roots_if_missing_in_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                second.path().join("foo.py"),
                &["Bar"]
            ))
        );
    }

    #[test]
    fn test_empty_target_parts_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &[]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_search_roots_returns_none() {
        let found = resolve_target_on_disk(&target(0, &["foo"]), &[]);

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar"]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_relative_target_returns_none_until_base_package_is_known() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("layers")).unwrap();
        fs::write(tmp.path().join("layers.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(1, &["layers", "Linear"]), &roots);

        assert_eq!(found, None);
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
mod import_map_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|s| s.to_string()).collect(),
            name: name.to_string(),
        }
    }

    #[test]
    fn test_plain_import() {
        let code = "import jax";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &[], "jax")));
    }

    #[test]
    fn test_plain_dotted_import() {
        let code = "import jax.numpy";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_plain_import_with_alias() {
        let code = "import jax.numpy as jnp";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jnp"), Some(&ip(0, &["jax"], "numpy")));
        assert_eq!(map.get("jax"), None);
    }

    #[test]
    fn test_plain_import_deeper_with_alias() {
        let code = "import equinox.nn as nn";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("nn"), Some(&ip(0, &["equinox"], "nn")));
    }

    #[test]
    fn test_from_import_single() {
        let code = "from jax import random";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
    }

    #[test]
    fn test_from_import_multiple() {
        let code = "from jaxtyping import Float, Array";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Float"), Some(&ip(0, &["jaxtyping"], "Float")));
        assert_eq!(map.get("Array"), Some(&ip(0, &["jaxtyping"], "Array")));
    }

    #[test]
    fn test_from_import_with_alias() {
        let code = "from jaxtyping import Array as Arr";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Arr"), Some(&ip(0, &["jaxtyping"], "Array")));
        assert_eq!(map.get("Array"), None);
    }

    #[test]
    fn test_from_import_multiple_mixed_aliases() {
        let code = "from mypackage import transform, MyLinear as ML, helper";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("transform"),
            Some(&ip(0, &["mypackage"], "transform"))
        );
        assert_eq!(map.get("ML"), Some(&ip(0, &["mypackage"], "MyLinear")));
        assert_eq!(map.get("helper"), Some(&ip(0, &["mypackage"], "helper")));
        assert_eq!(map.get("MyLinear"), None);
    }

    #[test]
    fn test_from_import_deeply_nested() {
        let code = "from google.cloud.storage.bucket import Bucket as GCSBucket";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("GCSBucket"),
            Some(&ip(0, &["google", "cloud", "storage", "bucket"], "Bucket"))
        );
    }

    #[test]
    fn test_relative_import_dot() {
        let code = "from . import utils";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
    }

    #[test]
    fn test_relative_import_dot_with_path() {
        let code = "from .layers import MyLinear";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("MyLinear"), Some(&ip(1, &["layers"], "MyLinear")));
    }

    #[test]
    fn test_relative_import_double_dot() {
        let code = "from ..utils import helper as h";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("h"), Some(&ip(2, &["utils"], "helper")));
    }

    #[test]
    fn test_relative_import_triple_dot() {
        let code = "from ...core.base import BaseModel";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("BaseModel"),
            Some(&ip(3, &["core", "base"], "BaseModel"))
        );
    }

    #[test]
    fn test_relative_import_dot_only_multiple() {
        let code = "from . import utils, helpers";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("helpers"), Some(&ip(1, &[], "helpers")));
    }

    #[test]
    fn test_comma_separated_plain_imports() {
        let code = "import sys, os";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("sys"), Some(&ip(0, &[], "sys")));
        assert_eq!(map.get("os"), Some(&ip(0, &[], "os")));
    }

    #[test]
    fn test_star_import_skipped() {
        let code = "from os.path import *";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_empty_file() {
        let code = "";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_no_imports() {
        let code = "x = 1\ny = 2";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_mixed_imports() {
        let code = r#"
import jax
import equinox as eqx
from jaxtyping import Float, Array
from mypackage.layers import MyLinear as ML
from . import utils
from ..core import Base
"#;
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &[], "jax")));
        assert_eq!(map.get("eqx"), Some(&ip(0, &[], "equinox")));
        assert_eq!(map.get("Float"), Some(&ip(0, &["jaxtyping"], "Float")));
        assert_eq!(map.get("Array"), Some(&ip(0, &["jaxtyping"], "Array")));
        assert_eq!(
            map.get("ML"),
            Some(&ip(0, &["mypackage", "layers"], "MyLinear"))
        );
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("Base"), Some(&ip(2, &["core"], "Base")));
        assert_eq!(map.len(), 7);
    }

    #[test]
    fn test_relative_import_with_alias() {
        let code = "from . import utils as u";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("u"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("utils"), None);
    }

    #[test]
    fn test_relative_import_with_path_and_alias() {
        let code = "from .layers import MyLinear as ML";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("ML"), Some(&ip(1, &["layers"], "MyLinear")));
        assert_eq!(map.get("MyLinear"), None);
    }

    #[test]
    fn test_deeply_nested_plain_import() {
        let code = "import a.b.c.d";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("a"), Some(&ip(0, &["a", "b", "c"], "d")));
    }

    #[test]
    fn test_comma_separated_aliased_imports() {
        let code = "import jax.numpy as jnp, equinox as eqx";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jnp"), Some(&ip(0, &["jax"], "numpy")));
        assert_eq!(map.get("eqx"), Some(&ip(0, &[], "equinox")));
    }

    #[test]
    fn test_parenthesized_from_import() {
        let code = "from jax import (\n    random,\n    numpy\n)";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
        assert_eq!(map.get("numpy"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_duplicate_import_last_wins() {
        let code = "import jax\nimport jax.numpy as jax";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_from_import_single_segment_module() {
        let code = "from os import path, getcwd, listdir";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("path"), Some(&ip(0, &["os"], "path")));
        assert_eq!(map.get("getcwd"), Some(&ip(0, &["os"], "getcwd")));
        assert_eq!(map.get("listdir"), Some(&ip(0, &["os"], "listdir")));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_backslash_continuation_import() {
        let code = "from jax \\\n    import random";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
    }

    #[test]
    fn test_multiple_relative_aliased_imports() {
        let code = "from . import utils as u, helpers as h";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("u"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("h"), Some(&ip(1, &[], "helpers")));
        assert_eq!(map.get("utils"), None);
        assert_eq!(map.get("helpers"), None);
    }
}

#[cfg(test)]
mod extract_calls_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn args_text<'a>(text: &'a str, range: &tree_sitter::Range) -> &'a str {
        &text[range.start_byte..range.end_byte]
    }

    #[test]
    fn test_simple_identifier_call() {
        let code = "x = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
        assert_eq!(args_text(code, &calls[0].args_node_range), "()");
    }

    #[test]
    fn test_identifier_call_with_args() {
        let code = "y = transform(x)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
        assert_eq!(calls[0].target, "transform");
        assert_eq!(args_text(code, &calls[0].args_node_range), "(x)");
    }

    #[test]
    fn test_attribute_call() {
        let code = "x = jnp.zeros((32, 64))";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "jnp.zeros");
        assert_eq!(args_text(code, &calls[0].args_node_range), "((32, 64))");
    }

    #[test]
    fn test_deep_attribute_call() {
        let code = "layer = eqx.nn.Linear(128, 64)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "layer");
        assert_eq!(calls[0].target, "eqx.nn.Linear");
        assert_eq!(args_text(code, &calls[0].args_node_range), "(128, 64)");
    }

    #[test]
    fn test_multiple_calls() {
        let code = "x = foo()\ny = bar(1, 2)\nz = jnp.zeros((3,))";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
        assert_eq!(calls[1].variable, "y");
        assert_eq!(calls[1].target, "bar");
        assert_eq!(calls[2].variable, "z");
        assert_eq!(calls[2].target, "jnp.zeros");
    }

    #[test]
    fn test_skips_assignment_without_call() {
        let code = "x = 5\ny = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }

    #[test]
    fn test_skips_bare_call_without_assignment() {
        let code = "foo()\nx = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
    }

    #[test]
    fn test_skips_binary_op_assignment() {
        let code = "x = a + b\ny = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }

    #[test]
    fn test_nested_call_captures_outer_only() {
        let code = "x = foo(bar())";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
    }

    #[test]
    fn test_call_with_kwargs() {
        let code = "layer = eqx.nn.Linear(128, 64, key=key)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].target, "eqx.nn.Linear");
        assert_eq!(
            args_text(code, &calls[0].args_node_range),
            "(128, 64, key=key)"
        );
    }

    #[test]
    fn test_empty_file() {
        let code = "";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_no_calls() {
        let code = "x = 5\ny = 'hello'\nz = [1, 2, 3]";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_inside_function() {
        let code = r#"
def forward(x):
    y = jnp.matmul(x, w)
    z = transform(y)
"#;
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].target, "jnp.matmul");
        assert_eq!(calls[1].target, "transform");
    }

    #[test]
    fn test_mixed_module_and_function_level() {
        let code = r#"
x = jnp.zeros((32,))
def forward(a):
    b = transform(a)
y = relu(x)
"#;
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].target, "jnp.zeros");
        assert_eq!(calls[1].target, "transform");
        assert_eq!(calls[2].target, "relu");
    }

    #[test]
    fn test_skips_tuple_unpack_assignment() {
        let code = "a, b = foo()\nx = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
    }

    #[test]
    fn test_skips_attribute_assignment() {
        let code = "self.x = foo()\ny = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }
}
