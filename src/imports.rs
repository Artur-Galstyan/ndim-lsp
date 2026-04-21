use std::collections::HashMap;

use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

pub fn get_imports<'a>(node: Node<'a>, text: &str) -> Result<HashMap<String, String>, String> {
    let language = tree_sitter_python::LANGUAGE.into();

    let query_str = r#"
        (import_statement
          name: [
            (dotted_name) @import_name
            (aliased_import
              name: (dotted_name) @import_name
              alias: (identifier) @import_alias
            )
          ]
        )

        (import_from_statement
          module_name: (dotted_name) @from_module
          name: [
            (dotted_name) @from_name
            (aliased_import
              name: (dotted_name) @from_name
              alias: (identifier) @from_alias
            )
          ]
        )
    "#;

    let mut map: HashMap<String, String> = HashMap::new();
    let query = Query::new(&language, query_str).unwrap();
    let mut cursor = QueryCursor::new();

    let Some(import_name_idx) = query.capture_index_for_name("import_name") else {
        return Err("import_name capture not found".to_string());
    };
    let Some(import_alias_idx) = query.capture_index_for_name("import_alias") else {
        return Err("import_alias capture not found".to_string());
    };
    let Some(from_module_idx) = query.capture_index_for_name("from_module") else {
        return Err("from_module capture not found".to_string());
    };
    let Some(from_name_idx) = query.capture_index_for_name("from_name") else {
        return Err("from_name capture not found".to_string());
    };
    let Some(from_alias_idx) = query.capture_index_for_name("from_alias") else {
        return Err("from_alias capture not found".to_string());
    };

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut alias = None;
        let mut module = None;

        for capture in match_.captures {
            let captured_text = capture
                .node
                .utf8_text(text.as_bytes())
                .map_err(|e| e.to_string())?;
            if capture.index == import_name_idx {
                name = Some(captured_text);
            } else if capture.index == import_alias_idx {
                alias = Some(captured_text);
            } else if capture.index == from_module_idx {
                module = Some(captured_text);
            } else if capture.index == from_name_idx {
                name = Some(captured_text);
            } else if capture.index == from_alias_idx {
                alias = Some(captured_text);
            }
        }

        let key = alias.or(name).ok_or("Import name is missing".to_string())?;
        let n = name.ok_or("Import name is missing")?;
        let value = match module {
            Some(m) => format!("{}.{}", m, n),
            None => n.to_string(),
        };

        map.insert(key.to_string(), value);
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use tree_sitter::Parser;

    use super::*;

    fn get_ast(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_get_imports() {
        let code = r#"
            from jaxtyping import Array, Float
            from jaxtyping import Array as Arr
            import jax.numpy as jnp
            import equinox as eqx
            import equinox.nn as nn
            import jax.numpy as jnp
            import jax
        "#;
        let tree = get_ast(code);
        let root_node = tree.root_node();

        let imports = get_imports(root_node, code);
        println!("{:?}", imports);

        assert!(imports.is_ok());
        let map = imports.unwrap();
        assert_eq!(map.len(), 7);
        assert_eq!(map.get("Array"), Some(&"jaxtyping.Array".to_string()));
        assert_eq!(map.get("Float"), Some(&"jaxtyping.Float".to_string()));
        assert_eq!(map.get("Arr"), Some(&"jaxtyping.Array".to_string()));
        assert_eq!(map.get("jnp"), Some(&"jax.numpy".to_string()));
        assert_eq!(map.get("eqx"), Some(&"equinox".to_string()));
        assert_eq!(map.get("nn"), Some(&"equinox.nn".to_string()));
        assert_eq!(map.get("jax"), Some(&"jax".to_string()));
    }

    #[test]
    fn test_deeply_nested_imports() {
        let code = r#"
            from google.cloud.storage.bucket import Bucket as GCSBucket
            import matplotlib.pyplot as plt
        "#;
        let tree = get_ast(code);
        let imports = get_imports(tree.root_node(), code).unwrap();

        assert_eq!(
            imports.get("GCSBucket"),
            Some(&"google.cloud.storage.bucket.Bucket".to_string())
        );
        assert_eq!(imports.get("plt"), Some(&"matplotlib.pyplot".to_string()));
    }

    #[test]
    fn test_ignored_syntax() {
        let code = r#"
            from . import utils
            from ..models import User
            import sys, os
        "#;
        let tree = get_ast(code);
        let imports = get_imports(tree.root_node(), code).unwrap();

        assert_eq!(imports.get("sys"), Some(&"sys".to_string()));
        assert_eq!(imports.get("utils"), None);
    }
}
