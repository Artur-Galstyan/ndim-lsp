use std::collections::HashMap;

use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

pub fn get_assignments<'a>(node: Node<'a>, text: &str) -> Result<Vec<Node<'a>>, String> {
    let language = tree_sitter_python::LANGUAGE.into();

    let query_str = r#"
        [
          (assignment)
          (augmented_assignment)
        ] @assign
    "#;

    let mut assignments: Vec<Node<'a>> = Vec::new();
    let query = Query::new(&language, query_str).unwrap();
    let mut cursor = QueryCursor::new();

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            assignments.push(capture.node);
        }
    }

    Ok(assignments)
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
    fn test_basic_assignments() {
        let code = r#"
def my_func():
    x = 10
    y = x + 5
    z = jnp.zeros((10, 20))
        "#;
        let tree = get_ast(code);

        let assignments = get_assignments(tree.root_node(), code).unwrap();

        assert_eq!(assignments.len(), 3, "Should find 3 assignments");

        let texts: Vec<_> = assignments
            .iter()
            .map(|n| n.utf8_text(code.as_bytes()).unwrap().trim())
            .collect();

        assert!(texts.contains(&"x = 10"));
        assert!(texts.contains(&"y = x + 5"));
        assert!(texts.contains(&"z = jnp.zeros((10, 20))"));
    }

    #[test]
    fn test_multiple_target_assignment() {
        // Python allows `a = b = 5` and `x, y = 1, 2`
        let code = r#"
    def multiple():
        a = b = 5
        x, y = get_shapes()
            "#;
        let tree = get_ast(code);
        let assignments = get_assignments(tree.root_node(), code).unwrap();
        assert_eq!(
            assignments.len(),
            3,
            "Should handle multiple target syntaxes"
        );
    }

    #[test]
    fn test_annotated_assignment() {
        // In python, `x: int = 5` is an `annotated_assignment`, NOT a regular `assignment`.
        // A robust language server should catch these too!
        let code = r#"
    def annotated():
        x: int = 5
        layer: eqx.nn.Linear = eqx.nn.Linear(10, 20)
            "#;
        let tree = get_ast(code);

        let assignments = get_assignments(tree.root_node(), code).unwrap();

        assert_eq!(assignments.len(), 2, "Should find annotated assignments");
    }

    #[test]
    fn test_augmented_assignment() {
        // `x += 5` is an `augmented_assignment` in tree-sitter.
        let code = r#"
    def augmented():
        x = 5
        x += 10
            "#;
        let tree = get_ast(code);

        let assignments = get_assignments(tree.root_node(), code).unwrap();

        // Whether you want to track augmented assignments is up to your LSP's logic,
        // but typically you do if it alters the shape/type of `x`.
        assert_eq!(
            assignments.len(),
            2,
            "Should find both regular and augmented assignments"
        );
    }
}
