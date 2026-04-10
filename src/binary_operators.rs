use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

pub fn get_binary_operators<'a>(node: Node<'a>, text: &str) -> Result<Vec<Node<'a>>, String> {
    let language = tree_sitter_python::LANGUAGE.into();

    let query_str = r#"
        [
          (binary_operator)
        ] @binary_operator
    "#;

    let mut binary_operators: Vec<Node<'a>> = Vec::new();
    let query = Query::new(&language, query_str).unwrap();
    let mut cursor = QueryCursor::new();

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            binary_operators.push(capture.node);
        }
    }

    Ok(binary_operators)
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
    fn test_finds_simple_binary_operator() {
        let code = "a + b";
        let tree = get_ast(code);
        let root = tree.root_node();

        let ops = get_binary_operators(root, code).unwrap();

        // Note: This test will fail until you update your query string
        // to use `_` instead of `(_)` for the operator field!
        assert!(
            !ops.is_empty(),
            "Failed to find any nodes. Check the operator pattern."
        );

        // Note: You will also need to filter `match_.captures` so you only
        // return the `@binary_operator` node, not the `@left` and `@right` nodes.
        let has_binary_op = ops.iter().any(|n| n.kind() == "binary_operator");
        assert!(
            has_binary_op,
            "Expected result to contain a 'binary_operator' node"
        );
    }

    #[test]
    fn test_finds_multiple_binary_operators() {
        let code = "x = (a * b) - c";
        let tree = get_ast(code);
        let root = tree.root_node();

        let ops = get_binary_operators(root, code).unwrap();

        // Filter out the child captures assuming the implementation gets fixed
        let binary_ops: Vec<_> = ops
            .into_iter()
            .filter(|n| n.kind() == "binary_operator")
            .collect();
        assert_eq!(binary_ops.len(), 2, "Expected exactly 2 binary operators");
    }

    #[test]
    fn test_ignores_other_operators() {
        let code = "x = -a"; // unary operator, not binary
        let tree = get_ast(code);
        let root = tree.root_node();

        let ops = get_binary_operators(root, code).unwrap();

        let binary_ops: Vec<_> = ops
            .into_iter()
            .filter(|n| n.kind() == "binary_operator")
            .collect();
        assert_eq!(binary_ops.len(), 0, "Should not match unary operators");
    }

    #[test]
    fn test_extracts_correct_text() {
        let code = "a + b";
        let tree = get_ast(code);
        let root = tree.root_node();

        let ops = get_binary_operators(root, code).unwrap();

        if let Some(op_node) = ops.into_iter().find(|n| n.kind() == "binary_operator") {
            let start = op_node.start_byte();
            let end = op_node.end_byte();
            let text_match = &code[start..end];
            assert_eq!(text_match, "a + b");
        }
    }
}
