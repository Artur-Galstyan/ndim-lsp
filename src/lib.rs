use std::collections::HashMap;

use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

#[derive(Debug, PartialEq, Clone)]
pub struct ImportPath {
    pub dots: usize,
    pub module: Vec<String>,
    pub name: String,
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
}
