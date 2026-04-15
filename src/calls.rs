use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

pub fn get_calls<'a>(node: Node<'a>, text: &str) -> Result<Vec<Node<'a>>, String> {
    let language = tree_sitter_python::LANGUAGE.into();

    let query_str = r#"
        [
          (call)
        ] @call
    "#;

    let mut calls: Vec<Node<'a>> = Vec::new();
    let query = Query::new(&language, query_str).unwrap();
    let mut cursor = QueryCursor::new();

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            calls.push(capture.node);
        }
    }

    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn get_ast(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_finds_simple_call() {
        let code = "print('hello')";
        let tree = get_ast(code);
        let root = tree.root_node();

        let calls = get_calls(root, code).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kind(), "call");

        let func_node = calls[0].child_by_field_name("function").unwrap();
        assert_eq!(func_node.kind(), "identifier");
        assert_eq!(func_node.utf8_text(code.as_bytes()).unwrap(), "print");
    }

    #[test]
    fn test_finds_attribute_call() {
        let code = "layer = equinox.nn.Linear(128, 64)";
        let tree = get_ast(code);
        let root = tree.root_node();

        let calls = get_calls(root, code).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kind(), "call");

        let func_node = calls[0].child_by_field_name("function").unwrap();
        assert_eq!(func_node.kind(), "attribute");
        assert_eq!(
            func_node.utf8_text(code.as_bytes()).unwrap(),
            "equinox.nn.Linear"
        );
    }

    #[test]
    fn test_finds_multiple_calls() {
        let code = r#"
layer = equinox.nn.Linear(128, 64)
x = jnp.zeros((1, 128))
out = layer(x)
        "#;
        let tree = get_ast(code);
        let root = tree.root_node();

        let calls = get_calls(root, code).unwrap();

        assert_eq!(calls.len(), 3);

        let func_names: Vec<_> = calls
            .iter()
            .map(|n| {
                n.child_by_field_name("function")
                    .unwrap()
                    .utf8_text(code.as_bytes())
                    .unwrap()
            })
            .collect();

        assert_eq!(func_names, vec!["equinox.nn.Linear", "jnp.zeros", "layer"]);
    }

    #[test]
    fn test_finds_nested_calls() {
        let code = "result = max(min(x, 10), 0)";
        let tree = get_ast(code);
        let root = tree.root_node();

        let calls = get_calls(root, code).unwrap();

        assert_eq!(calls.len(), 2);

        let func_names: Vec<_> = calls
            .iter()
            .map(|n| {
                n.child_by_field_name("function")
                    .unwrap()
                    .utf8_text(code.as_bytes())
                    .unwrap()
            })
            .collect();

        // Output order depends on AST traversal (usually pre-order, so outer then inner)
        assert!(func_names.contains(&"max"));
        assert!(func_names.contains(&"min"));
    }
}
