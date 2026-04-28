use std::collections::HashMap;

use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

#[derive(Debug, PartialEq, Clone)]
pub struct ImportPath {
    pub dots: usize,         // 0 = absolute, 1 = ".", 2 = "..", etc.
    pub module: Vec<String>, // ["jax", "numpy"] or ["utils"]
    pub name: String,        // "random", "Array", "MyLinear"
}

/// Builds a map from local name (what you type in code) to dotted path (what it resolves to).
///
/// Examples:
///   import jax              → {"jax": "jax"}
///   import jax.numpy as jnp → {"jnp": "jax.numpy"}
///   from jax import random  → {"random": "jax.random"}
///   from . import utils     → {"utils": ".utils"}
///   from ..core import Base → {"Base": "..core.Base"}
///
/// Star imports are skipped (can't know what's imported statically).
pub fn build_import_map(node: Node, text: &str) -> Result<HashMap<String, String>, String> {
    let mut result: HashMap<String, String> = HashMap::new();

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
                            let value = capture
                                .node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let parts: Vec<&str> = value.split(".").collect();
                            let module: Vec<String> = parts[..parts.len() - 1]
                                .iter()
                                .map(|p| p.to_string())
                                .collect();
                            let Some(name) = parts.last() else {
                                return Err(
                                    "Failed to fetch the last part of the import path".to_string()
                                );
                            };
                            let Some(first) = parts.first() else {
                                return Err("Empty import path".to_string());
                            };
                            import_map.insert(
                                first.to_string(),
                                ImportPath {
                                    dots: 0,
                                    module: module.clone(),
                                    name: name.to_string(),
                                },
                            );
                        }
                        "aliased_import" => {
                            let Some(name_child) = capture.node.child_by_field_name("name") else {
                                return Err("Failed to capture child of aliased_import".to_string());
                            };
                            let name = name_child
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let parts: Vec<&str> = name.split(".").collect();
                            let module: Vec<String> = parts[..parts.len() - 1]
                                .iter()
                                .map(|p| p.to_string())
                                .collect();
                            let Some(name) = parts.last() else {
                                return Err(
                                    "Failed to fetch the last part of the import path".to_string()
                                );
                            };

                            let Some(alias_node) = capture.node.child_by_field_name("alias") else {
                                return Err("Failed to capture child of aliased_import".to_string());
                            };
                            let alias = alias_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                            import_map.insert(
                                alias,
                                ImportPath {
                                    dots: 0,
                                    module: module.clone(),
                                    name: name.to_string(),
                                },
                            );
                        }
                        _ => {}
                    }
                }
            }
        } else if match_.pattern_index == 1 {
            let mut module_name = String::new();
            let mut name = String::new();
            for capture in match_.captures {
                if capture.index == module_name_idx {
                    let n = capture.node;
                    match n.kind() {
                        "dotted_name" => {
                            let identifier =
                                n.utf8_text(text.as_bytes()).map_err(|e| e.to_string())?;
                            module_name = identifier.to_string();
                        }
                        "relative_import" => {
                            let Some(import_prefix_node) = n.child_by_field_name("import_prefix")
                            else {
                                return Err(
                                    "Failed to get import prefix for relative import".to_string()
                                );
                            };
                            let import_prefix = import_prefix_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let mut dotted_name = String::new();
                            match n.child_by_field_name("dotted_name") {
                                Some(c) => {
                                    dotted_name = c
                                        .utf8_text(text.as_bytes())
                                        .map_err(|e| e.to_string())?
                                        .to_string();
                                }
                                None => dotted_name = "".to_string(),
                            }
                            module_name = format!("{}.{}", import_prefix, dotted_name);
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
                        }
                        "aliased_import" => {
                            let Some(alias_child_node) = n.child_by_field_name("alias") else {
                                return Err("aliased_import has no alias field".to_string());
                            };
                            name = alias_child_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(result)
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

    #[test]
    fn test_plain_import() {
        let code = "import jax";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&"jax".to_string()));
    }

    #[test]
    fn test_plain_dotted_import() {
        let code = "import jax.numpy";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&"jax".to_string()));
    }

    #[test]
    fn test_plain_import_with_alias() {
        let code = "import jax.numpy as jnp";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jnp"), Some(&"jax.numpy".to_string()));
        assert_eq!(map.get("jax"), None);
    }

    #[test]
    fn test_plain_import_deeper_with_alias() {
        let code = "import equinox.nn as nn";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("nn"), Some(&"equinox.nn".to_string()));
    }

    #[test]
    fn test_from_import_single() {
        let code = "from jax import random";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&"jax.random".to_string()));
    }

    #[test]
    fn test_from_import_multiple() {
        let code = "from jaxtyping import Float, Array";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Float"), Some(&"jaxtyping.Float".to_string()));
        assert_eq!(map.get("Array"), Some(&"jaxtyping.Array".to_string()));
    }

    #[test]
    fn test_from_import_with_alias() {
        let code = "from jaxtyping import Array as Arr";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Arr"), Some(&"jaxtyping.Array".to_string()));
        assert_eq!(map.get("Array"), None);
    }

    #[test]
    fn test_from_import_multiple_mixed_aliases() {
        let code = "from mypackage import transform, MyLinear as ML, helper";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("transform"),
            Some(&"mypackage.transform".to_string())
        );
        assert_eq!(map.get("ML"), Some(&"mypackage.MyLinear".to_string()));
        assert_eq!(map.get("helper"), Some(&"mypackage.helper".to_string()));
        assert_eq!(map.get("MyLinear"), None);
    }

    #[test]
    fn test_from_import_deeply_nested() {
        let code = "from google.cloud.storage.bucket import Bucket as GCSBucket";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("GCSBucket"),
            Some(&"google.cloud.storage.bucket.Bucket".to_string())
        );
    }

    #[test]
    fn test_relative_import_dot() {
        let code = "from . import utils";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&".utils".to_string()));
    }

    #[test]
    fn test_relative_import_dot_with_path() {
        let code = "from .layers import MyLinear";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("MyLinear"), Some(&".layers.MyLinear".to_string()));
    }

    #[test]
    fn test_relative_import_double_dot() {
        let code = "from ..utils import helper as h";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("h"), Some(&"..utils.helper".to_string()));
    }

    #[test]
    fn test_relative_import_triple_dot() {
        let code = "from ...core.base import BaseModel";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("BaseModel"),
            Some(&"...core.base.BaseModel".to_string())
        );
    }

    #[test]
    fn test_relative_import_dot_only_multiple() {
        let code = "from . import utils, helpers";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&".utils".to_string()));
        assert_eq!(map.get("helpers"), Some(&".helpers".to_string()));
    }

    #[test]
    fn test_comma_separated_plain_imports() {
        let code = "import sys, os";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("sys"), Some(&"sys".to_string()));
        assert_eq!(map.get("os"), Some(&"os".to_string()));
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
        assert_eq!(map.get("jax"), Some(&"jax".to_string()));
        assert_eq!(map.get("eqx"), Some(&"equinox".to_string()));
        assert_eq!(map.get("Float"), Some(&"jaxtyping.Float".to_string()));
        assert_eq!(map.get("Array"), Some(&"jaxtyping.Array".to_string()));
        assert_eq!(
            map.get("ML"),
            Some(&"mypackage.layers.MyLinear".to_string())
        );
        assert_eq!(map.get("utils"), Some(&".utils".to_string()));
        assert_eq!(map.get("Base"), Some(&"..core.Base".to_string()));
        assert_eq!(map.len(), 7);
    }
}
