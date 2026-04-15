use crate::shape_resolvers::shape_resolver::ShapeInfo;
use crate::shape_resolvers::shape_resolver::ShapeResult;
use std::collections::HashMap;

use tree_sitter::Node;
use tree_sitter::Query;
use tree_sitter::QueryCursor;
use tree_sitter::StreamingIterator;

use crate::ParamKind;

pub fn find_node_by_kind<'a>(node: Node<'a>, kind: &str, results: &mut Vec<Node<'a>>) {
    if node.kind() == kind {
        results.push(node);
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        find_node_by_kind(child, kind, results);
    }
}

pub fn get_first_matching_parent<'a>(node: Node<'a>, target_type: &str) -> Option<Node<'a>> {
    let mut parent = node.parent();

    while let Some(p) = parent {
        if p.kind() == target_type {
            return Some(p);
        } else {
            parent = p.parent();
        }
    }

    return None;
}

pub fn get_shapes_from_fn_args(
    function_node: Node<'_>,
    params: &mut HashMap<String, ParamKind>,
    text: &str,
) {
    let mut typed_params: Vec<Node<'_>> = Vec::new();
    find_node_by_kind(function_node, "typed_parameter", &mut typed_params);

    for param_node in typed_params {
        let Some(node_identifier) = param_node.child(0) else {
            return;
        };

        let param_name = node_identifier
            .utf8_text(text.as_bytes())
            .expect("Failed to get node identifier");

        let Some(node_type) = param_node.child_by_field_name("type") else {
            return;
        };

        let mut string_content_results: Vec<Node<'_>> = Vec::new();
        find_node_by_kind(node_type, "string_content", &mut string_content_results);

        let string_content_node = string_content_results
            .into_iter()
            .next()
            .expect("Expected to find at least one string_content");

        let string_content = string_content_node
            .utf8_text(&text.as_bytes())
            .expect("Failed to get string_content");

        let dims: Vec<String> = string_content
            .split_whitespace()
            .map(String::from)
            .collect();
        let shape_info = ShapeInfo {
            dims,
            line: 0,
            character: 0,
            is_inferred: false,
        };
        params.insert(param_name.to_string(), ParamKind::Shape(shape_info));
    }
}

pub fn get_arg<'a>(
    args_node: Node<'a>,
    position: usize,
    name: &str,
    text: &str,
) -> Option<Node<'a>> {
    let mut positional_index = 0;
    let mut cursor = args_node.walk();

    for child in args_node.named_children(&mut cursor) {
        if child.kind() == "keyword_argument" {
            let name_node = child.child_by_field_name("name")?;
            let kw_name = name_node.utf8_text(text.as_bytes()).ok()?;
            if kw_name == name {
                return child.child_by_field_name("value");
            }
        } else {
            if positional_index == position {
                return Some(child);
            } else {
                positional_index += 1;
            }
        }
    }

    None
}

pub fn handle_elementwise_ops(left_shape: Vec<String>, right_shape: Vec<String>) -> ShapeResult {
    if left_shape.len() != right_shape.len() {
        return ShapeResult::Error(format!(
            "Invalid shapes found: {:?} and {:?}",
            left_shape, right_shape
        ));
    }

    for i in 0..left_shape.len() {
        if left_shape[i] != right_shape[i] {
            return ShapeResult::Error(format!(
                "Invalid shapes found: {:?} and {:?}",
                left_shape, right_shape
            ));
        }
    }

    return ShapeResult::Ok(left_shape.clone());
}

pub fn parse_axis(axis_node: Node<'_>, text: &str) -> Option<usize> {
    let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
        return None;
    };
    let Ok(parsed_axis) = axis_str.parse::<usize>() else {
        return None;
    };
    Some(parsed_axis)
}

pub fn get_functions<'a>(root: Node<'a>, text: &str) -> Result<Vec<Node<'a>>, String> {
    let language = tree_sitter_python::LANGUAGE.into();
    let query = Query::new(&language, "(function_definition) @func").map_err(|e| e.to_string())?;

    let mut cursor = QueryCursor::new();
    let mut functions = Vec::new();

    let mut matches = cursor.matches(&query, root, text.as_bytes());

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            functions.push(capture.node);
        }
    }

    Ok(functions)
}
