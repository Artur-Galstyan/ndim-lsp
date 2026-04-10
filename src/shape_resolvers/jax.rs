use std::collections::HashMap;
use tree_sitter::Node;

use super::shape_resolver::{ParamKind, ShapeResult, resolve_shape};
use crate::helpers::{get_arg, parse_axis};

pub fn jax_numpy_concatenate(
    args_node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    let Some(list_node) = get_arg(args_node, 0, "arrays", text) else {
        return ShapeResult::Unknown;
    };

    let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
        return ShapeResult::Unknown;
    };

    let Ok(_) = axis_node.utf8_text(text.as_bytes()) else {
        return ShapeResult::Unknown;
    };

    let Some(parsed_axis) = parse_axis(axis_node, text) else {
        return ShapeResult::Unknown;
    };

    let mut cursor = list_node.walk();
    let mut all_shapes: Vec<Vec<String>> = Vec::new();

    for child in list_node.named_children(&mut cursor) {
        match resolve_shape(child, params, import_alias_map, text) {
            ShapeResult::Ok(dims) => all_shapes.push(dims),
            other => return other,
        }
    }

    if all_shapes.is_empty() {
        return ShapeResult::Unknown;
    }

    let ndim = all_shapes[0].len();
    for shape in &all_shapes {
        if shape.len() != ndim {
            return ShapeResult::Error(format!(
                "All inputs to concatenate must have the same number of dims"
            ));
        }
    }

    if parsed_axis >= ndim {
        return ShapeResult::Error(format!(
            "Axis {} is out of bounds for shape with {} dims",
            parsed_axis, ndim
        ));
    }

    for dim_idx in 0..ndim {
        if dim_idx == parsed_axis {
            continue;
        }
        let first_dim = &all_shapes[0][dim_idx];
        for shape in &all_shapes[1..] {
            if &shape[dim_idx] != first_dim {
                return ShapeResult::Error(format!(
                    "Dim mismatch at axis {}: '{}' vs '{}'",
                    dim_idx, first_dim, shape[dim_idx]
                ));
            }
        }
    }

    let mut result = all_shapes[0].clone();
    let concat_dim = all_shapes
        .iter()
        .map(|s| s[parsed_axis].clone())
        .collect::<Vec<_>>()
        .join("+");
    result[parsed_axis] = concat_dim;
    ShapeResult::Ok(result)
}

pub fn jax_numpy_transpose(
    args_node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    let Some(input_node) = get_arg(args_node, 0, "a", text) else {
        return ShapeResult::Unknown;
    };
    let input_shape = match resolve_shape(input_node, params, import_alias_map, text) {
        ShapeResult::Ok(dims) => dims,
        other => return other,
    };

    let Some(tuple_node) = get_arg(args_node, 1, "axes", text) else {
        return ShapeResult::Unknown;
    };

    let mut new_shape = Vec::new();
    let mut cursor = tuple_node.walk();
    for child in tuple_node.named_children(&mut cursor) {
        if child.kind() == "integer" {
            let Ok(text) = child.utf8_text(text.as_bytes()) else {
                return ShapeResult::Unknown;
            };
            let Ok(idx) = text.parse::<usize>() else {
                return ShapeResult::Unknown;
            };

            let Some(dim) = input_shape.get(idx) else {
                return ShapeResult::Error(format!(
                    "Axis {} is out of bounds for shape with {} dims",
                    idx,
                    input_shape.len()
                ));
            };

            new_shape.push(dim.clone());
        }
    }
    ShapeResult::Ok(new_shape)
}

pub fn jax_numpy_reduce(
    args_node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    let Some(input_node) = get_arg(args_node, 0, "a", text) else {
        return ShapeResult::Unknown;
    };

    let input_shape = match resolve_shape(input_node, params, import_alias_map, text) {
        ShapeResult::Ok(dims) => dims,
        other => return other,
    };

    let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
        return ShapeResult::Unknown;
    };

    let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
        return ShapeResult::Unknown;
    };

    let Ok(parsed_axis) = axis_str.parse::<usize>() else {
        return ShapeResult::Unknown;
    };

    if parsed_axis >= input_shape.len() {
        return ShapeResult::Error(format!(
            "Axis index {} is out of bounds for shape with {} dimensions",
            parsed_axis,
            input_shape.len()
        ));
    }

    let keepdims = match get_arg(args_node, 4, "keepdims", text) {
        Some(n) => n.utf8_text(text.as_bytes()).ok() == Some("True"),
        None => false,
    };
    let mut result = input_shape.clone();

    if keepdims {
        result[parsed_axis] = "1".to_string();
    } else {
        result.remove(parsed_axis);
    }
    return ShapeResult::Ok(result);
}
