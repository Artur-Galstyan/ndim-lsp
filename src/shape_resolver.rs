use std::collections::HashMap;

use tree_sitter::Node;

use crate::helpers::{get_arg, handle_elementwise_ops, parse_axis};

pub enum ParamKind {
    Shape(ShapeInfo),
    Layer(LayerInfo),
}

pub enum ShapeResult {
    Ok(Vec<String>),
    Error(String),
    Unknown,
}

pub struct LayerInfo {
    pub layer_type: String,
    pub in_features: String,
    pub out_features: String,
}

pub struct ShapeInfo {
    pub dims: Vec<String>,
    pub line: u32,
    pub character: u32,
    pub is_inferred: bool,
}

pub fn resolve_shape(
    node: Node<'_>,
    params: &HashMap<String, ParamKind>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> ShapeResult {
    match node.kind() {
        "identifier" => {
            let param_name = node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");

            match params.get(param_name) {
                Some(ParamKind::Shape(info)) => ShapeResult::Ok(info.dims.clone()),
                Some(ParamKind::Layer(_)) => ShapeResult::Unknown,
                None => ShapeResult::Unknown,
            }
        }
        "call" => {
            let Some(func_node) = node.child_by_field_name("function") else {
                return ShapeResult::Unknown;
            };
            let Some(args_node) = node.child_by_field_name("arguments") else {
                return ShapeResult::Unknown;
            };

            if func_node.kind() == "attribute" {
                let Some(obj) = func_node.child_by_field_name("object") else {
                    return ShapeResult::Unknown;
                };
                let Some(attr) = func_node.child_by_field_name("attribute") else {
                    return ShapeResult::Unknown;
                };
                let Some(obj_name) = obj.utf8_text(text.as_bytes()).ok() else {
                    return ShapeResult::Unknown;
                };
                let Some(attr_name) = attr.utf8_text(text.as_bytes()).ok() else {
                    return ShapeResult::Unknown;
                };
                let resolved_object = if let Some((prefix, rest)) = obj_name.split_once('.') {
                    match import_alias_map.get(prefix) {
                        Some(resolved) => format!("{}.{}", resolved, rest),
                        None => obj_name.to_string(),
                    }
                } else {
                    import_alias_map
                        .get(obj_name)
                        .cloned()
                        .unwrap_or(obj_name.to_string())
                };

                match (resolved_object.as_str(), attr_name) {
                    ("jax.numpy", "concatenate") => {
                        let Some(list_node) = get_arg(args_node, 0, "arrays", text) else {
                            return ShapeResult::Unknown;
                        };

                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            return ShapeResult::Unknown;
                        };

                        let Ok(axis_text) = axis_node.utf8_text(text.as_bytes()) else {
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

                    ("jax.numpy", "transpose") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Unknown;
                        };
                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
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
                    ("jax.numpy", "sum")
                    | ("jax.numpy", "mean")
                    | ("jax.numpy", "max")
                    | ("jax.numpy", "min") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Unknown;
                        };

                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
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
                    ("jax.numpy", "expand_dims") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get input shape"
                            ));
                        };
                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get axis"
                            ));
                        };

                        let shape = match resolve_shape(input_node, params, import_alias_map, text)
                        {
                            ShapeResult::Ok(items) => items,
                            other => return other,
                        };

                        let Ok(axis_str) = axis_node.utf8_text(text.as_bytes()) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get axis string"
                            ));
                        };

                        let Ok(parsed_axis) = axis_str.parse::<usize>() else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to parse axis string"
                            ));
                        };

                        let mut current_dims = shape.clone();
                        if parsed_axis > shape.len() {
                            return ShapeResult::Error(format!(
                                "Axis {} is out of bounds for expand_dims on shape with {} dims",
                                parsed_axis,
                                shape.len()
                            ));
                        }
                        current_dims.insert(parsed_axis, "1".to_string());
                        return ShapeResult::Ok(current_dims);
                    }
                    ("jax.numpy", "squeeze") => {
                        let Some(input_node) = get_arg(args_node, 0, "a", text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to get input shape"
                            ));
                        };
                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
                                ShapeResult::Ok(items) => items,
                                other => return other,
                            };

                        let Some(axis_node) = get_arg(args_node, 1, "axis", text) else {
                            let result: Vec<String> =
                                input_shape.into_iter().filter(|d| d != "1").collect();
                            return ShapeResult::Ok(result);
                        };

                        let Some(parsed_axis) = parse_axis(axis_node, text) else {
                            return ShapeResult::Error(format!(
                                "Unexpected TS error: failed to parse axis string"
                            ));
                        };

                        if parsed_axis >= input_shape.len() {
                            return ShapeResult::Error(format!(
                                "Axis {} is out of bounds for shape with {} dims",
                                parsed_axis,
                                input_shape.len()
                            ));
                        }

                        if input_shape[parsed_axis] != "1" {
                            return ShapeResult::Error(format!(
                                "Cannot squeeze axis {} with dim '{}' — only dims of size 1 can be squeezed",
                                parsed_axis, input_shape[parsed_axis]
                            ));
                        }

                        let mut new_dims = input_shape.clone();
                        new_dims.remove(parsed_axis);
                        return ShapeResult::Ok(new_dims);
                    }
                    _ => return ShapeResult::Unknown,
                }
            } else if func_node.kind() == "identifier" {
                let Ok(func_name) = func_node.utf8_text(text.as_bytes()) else {
                    return ShapeResult::Unknown;
                };

                match params.get(func_name) {
                    Some(ParamKind::Layer(layer)) => {
                        let Some(input_node) = args_node.named_child(0) else {
                            return ShapeResult::Unknown;
                        };

                        let input_shape =
                            match resolve_shape(input_node, params, import_alias_map, text) {
                                ShapeResult::Ok(items) => items,
                                other => return other,
                            };

                        match layer.layer_type.as_str() {
                            "Linear" => {
                                if input_shape.last().map(|s| s.as_str())
                                    != Some(&layer.in_features)
                                {
                                    return ShapeResult::Error(format!(
                                        "Linear layer expects last dim '{}', got '{}'",
                                        layer.in_features,
                                        input_shape.last().unwrap_or(&"?".to_string())
                                    ));
                                }
                                let mut result = input_shape.clone();
                                *result.last_mut().unwrap() = layer.out_features.clone();
                                ShapeResult::Ok(result)
                            }
                            _ => return ShapeResult::Unknown,
                        }
                    }
                    _ => return ShapeResult::Unknown,
                }
            } else {
                return ShapeResult::Unknown;
            }
        }
        "parenthesized_expression" => {
            let binary_operator_child = node
                .named_child(0)
                .expect("A parenthesized_expression always has a child");

            return resolve_shape(binary_operator_child, params, import_alias_map, text);
        }
        "attribute" => {
            let Some(attribute_identifier_node) = node.child_by_field_name("attribute") else {
                return ShapeResult::Unknown;
            };

            let attribute_name = attribute_identifier_node
                .utf8_text(text.as_bytes())
                .expect("Failed to get node identifier");

            match attribute_name {
                "T" => {
                    let Some(object_node) = node.child_by_field_name("object") else {
                        return ShapeResult::Error(format!(
                            "Unexpected TS error: Failed to get object child from attribute node"
                        ));
                    };
                    let mut shape = match resolve_shape(object_node, params, import_alias_map, text)
                    {
                        ShapeResult::Ok(shape) => shape,
                        other => return other,
                    };
                    shape.reverse();
                    return ShapeResult::Ok(shape);
                }
                _ => ShapeResult::Unknown,
            }
        }
        "binary_operator" => {
            let Some(op) = node.child_by_field_name("operator") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get operator child from binary_operator node"
                        .to_string(),
                );
            };

            let op_text = op
                .utf8_text(&text.as_bytes())
                .expect("Operator has no text");

            let Some(left_node) = node.child_by_field_name("left") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get left child from binary_operator node"
                        .to_string(),
                );
            };
            let Some(right_node) = node.child_by_field_name("right") else {
                return ShapeResult::Error(
                    "Unexpected TS error: Failed to get right child from binary_operator node"
                        .to_string(),
                );
            };

            let left_shape = match resolve_shape(left_node, params, import_alias_map, text) {
                ShapeResult::Ok(shape) => shape,
                other => return other,
            };
            let right_shape = match resolve_shape(right_node, params, import_alias_map, text) {
                ShapeResult::Ok(shape) => shape,
                other => return other,
            };

            match op_text {
                "@" => {
                    if &left_shape.last().unwrap() != &right_shape.first().unwrap() {
                        return ShapeResult::Error(format!(
                            "Invalid shapes found: {:?} and {:?}",
                            left_shape, right_shape
                        ));
                    } else {
                        return ShapeResult::Ok(vec![
                            left_shape.first().unwrap().clone(),
                            right_shape.last().unwrap().clone(),
                        ]);
                    }
                }
                "+" | "-" | "*" | "/" => handle_elementwise_ops(left_shape, right_shape),
                _ => return ShapeResult::Unknown,
            }
        }
        _ => return ShapeResult::Unknown,
    }
}
