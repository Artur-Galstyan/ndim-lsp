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
            return ShapeResult::Error(
                "All inputs to concatenate must have the same number of dims".to_string(),
            );
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

    let axis_node_opt = get_arg(args_node, 1, "axis", text);
    let mut axes_to_reduce = Vec::new();

    if let Some(axis_node) = axis_node_opt {
        match axis_node.kind() {
            "none" => {
                axes_to_reduce = (0..input_shape.len()).collect();
            }
            "integer" => {
                if let Some(axis) = parse_axis(axis_node, text) {
                    axes_to_reduce.push(axis);
                } else {
                    return ShapeResult::Unknown;
                }
            }
            "tuple" | "list" => {
                let mut cursor = axis_node.walk();

                for child in axis_node.named_children(&mut cursor) {
                    if child.kind() == "integer" {
                        if let Some(axis) = parse_axis(child, text) {
                            axes_to_reduce.push(axis);
                        } else {
                            return ShapeResult::Unknown;
                        }
                    }
                }
            }
            _ => return ShapeResult::Unknown,
        }
    } else {
        axes_to_reduce = (0..input_shape.len()).collect();
    }

    if axes_to_reduce.len() > input_shape.len() {
        return ShapeResult::Error(format!(
            "Too many axes specified: {} for shape with {} dimensions",
            axes_to_reduce.len(),
            input_shape.len()
        ));
    }

    // check bounds
    for &axis in &axes_to_reduce {
        if axis >= input_shape.len() {
            return ShapeResult::Error(format!(
                "Axis index {} is out of bounds for shape with {} dimensions",
                axis,
                input_shape.len()
            ));
        }
    }

    let keepdims = match get_arg(args_node, 4, "keepdims", text) {
        Some(n) => n.utf8_text(text.as_bytes()).ok() == Some("True"),
        None => false,
    };
    let mut result = input_shape.clone();

    axes_to_reduce.sort_unstable_by(|a, b| b.cmp(a));
    axes_to_reduce.dedup();

    for &axis in &axes_to_reduce {
        if keepdims {
            result[axis] = "1".to_string();
        } else {
            result.remove(axis);
        }
    }

    ShapeResult::Ok(result)
}

pub fn jax_expand_dims(
    args_node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    let Some(input_node) = get_arg(args_node, 0, "a", text) else {
        return ShapeResult::Error("Unexpected TS error: failed to get input shape".to_string());
    };

    let shape = match resolve_shape(input_node, params, import_alias_map, text) {
        ShapeResult::Ok(items) => items,
        other => return other,
    };

    let axis_node_opt = get_arg(args_node, 1, "axis", text);
    let mut axes_to_expand = Vec::new();

    if let Some(axis_node) = axis_node_opt {
        match axis_node.kind() {
            "integer" => {
                if let Some(axis) = parse_axis(axis_node, text) {
                    axes_to_expand.push(axis);
                } else {
                    return ShapeResult::Unknown;
                }
            }
            "tuple" | "list" => {
                let mut cursor = axis_node.walk();

                for child in axis_node.named_children(&mut cursor) {
                    if child.kind() == "integer" {
                        if let Some(axis) = parse_axis(child, text) {
                            axes_to_expand.push(axis);
                        } else {
                            return ShapeResult::Unknown;
                        }
                    }
                }
            }
            _ => return ShapeResult::Unknown,
        }
    } else {
        return ShapeResult::Error("Axis argument is required for expand_dims".to_string());
    }

    let mut current_dims = shape.clone();
    axes_to_expand.sort_unstable();

    for axis in axes_to_expand {
        if axis > current_dims.len() {
            return ShapeResult::Error(format!(
                "Axis {} is out of bounds for expand_dims on shape with {} dims",
                axis,
                current_dims.len()
            ));
        }
        current_dims.insert(axis, "1".to_string());
    }

    ShapeResult::Ok(current_dims)
}

pub fn jax_squeeze(
    args_node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    let Some(input_node) = get_arg(args_node, 0, "a", text) else {
        return ShapeResult::Error("Unexpected TS error: failed to get input shape".to_string());
    };
    let input_shape = match resolve_shape(input_node, params, import_alias_map, text) {
        ShapeResult::Ok(items) => items,
        other => return other,
    };

    let mut axes_to_squeeze: Vec<usize> = Vec::new();
    let axis_node_opt = get_arg(args_node, 1, "axis", text);

    if axis_node_opt.is_none_or(|n| n.kind() == "none") {
        let new_dims: Vec<String> = input_shape.into_iter().filter(|d| d != "1").collect();
        return ShapeResult::Ok(new_dims);
    }

    if let Some(axis_node) = axis_node_opt {
        match axis_node.kind() {
            "integer" => {
                if let Some(axis) = parse_axis(axis_node, text) {
                    axes_to_squeeze.push(axis);
                } else {
                    return ShapeResult::Unknown;
                }
            }
            "tuple" | "list" => {
                let mut cursor = axis_node.walk();

                for child in axis_node.named_children(&mut cursor) {
                    if child.kind() == "integer" {
                        if let Some(axis) = parse_axis(child, text) {
                            axes_to_squeeze.push(axis);
                        } else {
                            return ShapeResult::Unknown;
                        }
                    }
                }
            }
            _ => return ShapeResult::Unknown,
        }
    }
    if axes_to_squeeze.len() > input_shape.len() {
        return ShapeResult::Error(format!(
            "Cannot squeeze out more axes than are present in the shape, got shape {:?} and axes to squeeze {:?}",
            input_shape, axes_to_squeeze
        ));
    }

    let mut new_dims = input_shape.clone();

    axes_to_squeeze.sort_unstable_by(|a, b| b.cmp(a));

    for &axis in &axes_to_squeeze {
        if axis >= new_dims.len() {
            return ShapeResult::Error(format!(
                "Cannot squeeze out axis {} which is out of bounds for shape {:?}",
                axis, input_shape
            ));
        }

        if new_dims[axis] != "1" {
            return ShapeResult::Error(format!(
                "Cannot select an axis to squeeze out which has size not equal to one, got shape {:?} and axis {} is size {}",
                input_shape, axis, new_dims[axis]
            ));
        }

        new_dims.remove(axis);
    }

    ShapeResult::Ok(new_dims)
}
