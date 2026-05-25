use std::collections::HashMap;

use crate::types::*;

#[cfg(test)]
use crate::{build_import_map, resolve_call_target};

pub fn classify_known_function(target: &ResolvedTarget) -> Option<KnownFunction> {
    let (name, module) = target.parts.split_last()?;

    let is_jax = module == ["jax"];
    let is_jax_numpy = module == ["jax", "numpy"];
    let is_numpy = module == ["numpy"];
    let is_torch = module == ["torch"];
    let is_jax_lax = module == ["jax", "lax"];

    if is_jax {
        return match name.as_str() {
            "vmap" => Some(KnownFunction::Vmap),
            _ => None,
        };
    }

    if is_jax_numpy || is_numpy {
        return match name.as_str() {
            "concatenate" | "concat" => Some(KnownFunction::Concatenate),
            "stack" => Some(KnownFunction::Stack),
            "reshape" => Some(KnownFunction::Reshape),
            "transpose" => Some(KnownFunction::Transpose),
            "expand_dims" => Some(KnownFunction::ExpandDims),
            "squeeze" => Some(KnownFunction::Squeeze),
            "sum" => Some(KnownFunction::Sum),
            "mean" => Some(KnownFunction::Mean),
            "max" | "amax" => Some(KnownFunction::Max),
            "min" | "amin" => Some(KnownFunction::Min),
            "prod" => Some(KnownFunction::Prod),
            "std" => Some(KnownFunction::Std),
            "var" => Some(KnownFunction::Var),
            "all" => Some(KnownFunction::All),
            "any" => Some(KnownFunction::Any),
            "argmax" => Some(KnownFunction::ArgMax),
            "argmin" => Some(KnownFunction::ArgMin),
            "argsort" => Some(KnownFunction::Argsort),
            "sort" => Some(KnownFunction::Sort),
            "cumsum" => Some(KnownFunction::Cumsum),
            "cumprod" => Some(KnownFunction::Cumprod),
            "matmul" => Some(KnownFunction::Matmul),
            "dot" => Some(KnownFunction::Dot),
            "tensordot" => Some(KnownFunction::TensorDot),
            "outer" => Some(KnownFunction::Outer),
            "inner" => Some(KnownFunction::Inner),
            "vdot" => Some(KnownFunction::Vdot),
            "einsum" => Some(KnownFunction::Einsum),
            "split" => Some(KnownFunction::Split),
            "tile" => Some(KnownFunction::Tile),
            "repeat" => Some(KnownFunction::Repeat),
            "flatten" => Some(KnownFunction::Flatten),
            "ravel" => Some(KnownFunction::Ravel),
            "moveaxis" => Some(KnownFunction::MoveAxis),
            "swapaxes" => Some(KnownFunction::SwapAxes),
            "where" => Some(KnownFunction::Where),
            "zeros" => Some(KnownFunction::Zeros),
            "ones" => Some(KnownFunction::Ones),
            "full" => Some(KnownFunction::Full),
            "empty" => Some(KnownFunction::Empty),
            "zeros_like" => Some(KnownFunction::ZerosLike),
            "ones_like" => Some(KnownFunction::OnesLike),
            "full_like" => Some(KnownFunction::FullLike),
            "empty_like" => Some(KnownFunction::EmptyLike),
            "arange" => Some(KnownFunction::Arange),
            "linspace" => Some(KnownFunction::Linspace),
            "logspace" => Some(KnownFunction::Logspace),
            "eye" => Some(KnownFunction::Eye),
            "identity" => Some(KnownFunction::Identity),
            "broadcast_to" => Some(KnownFunction::BroadcastTo),
            "broadcast_arrays" => Some(KnownFunction::BroadcastArrays),
            "atleast_1d" => Some(KnownFunction::AtLeast1D),
            "atleast_2d" => Some(KnownFunction::AtLeast2D),
            "atleast_3d" => Some(KnownFunction::AtLeast3D),
            "pad" => Some(KnownFunction::Pad),
            "roll" => Some(KnownFunction::Roll),
            "flip" | "fliplr" | "flipud" => Some(KnownFunction::Flip),
            "rot90" => Some(KnownFunction::Rot90),
            "take" => Some(KnownFunction::Take),
            "diag" => Some(KnownFunction::Diag),
            "diagonal" => Some(KnownFunction::Diagonal),
            "trace" => Some(KnownFunction::Trace),
            "triu" => Some(KnownFunction::Triu),
            "tril" => Some(KnownFunction::Tril),
            "meshgrid" => Some(KnownFunction::Meshgrid),
            "vstack" | "row_stack" => Some(KnownFunction::Vstack),
            "hstack" => Some(KnownFunction::Hstack),
            "dstack" => Some(KnownFunction::Dstack),
            "column_stack" => Some(KnownFunction::ColumnStack),
            "block" => Some(KnownFunction::Block),
            "array" => Some(KnownFunction::Array),
            "asarray" => Some(KnownFunction::AsArray),
            _ => None,
        };
    }

    if is_torch {
        return match name.as_str() {
            "cat" | "concat" | "concatenate" => Some(KnownFunction::Concatenate),
            "stack" => Some(KnownFunction::Stack),
            "reshape" => Some(KnownFunction::Reshape),
            "transpose" => Some(KnownFunction::Transpose),
            "unsqueeze" => Some(KnownFunction::ExpandDims),
            "squeeze" => Some(KnownFunction::Squeeze),
            "sum" => Some(KnownFunction::Sum),
            "mean" => Some(KnownFunction::Mean),
            "max" => Some(KnownFunction::Max),
            "min" => Some(KnownFunction::Min),
            "prod" => Some(KnownFunction::Prod),
            "std" => Some(KnownFunction::Std),
            "var" => Some(KnownFunction::Var),
            "matmul" => Some(KnownFunction::Matmul),
            "dot" => Some(KnownFunction::Dot),
            "tensordot" => Some(KnownFunction::TensorDot),
            "outer" => Some(KnownFunction::Outer),
            "inner" => Some(KnownFunction::Inner),
            "einsum" => Some(KnownFunction::Einsum),
            "split" => Some(KnownFunction::Split),
            "tile" => Some(KnownFunction::Tile),
            "repeat" => Some(KnownFunction::Repeat),
            "flatten" => Some(KnownFunction::Flatten),
            "ravel" => Some(KnownFunction::Ravel),
            "where" => Some(KnownFunction::Where),
            "all" => Some(KnownFunction::All),
            "any" => Some(KnownFunction::Any),
            "argmax" => Some(KnownFunction::ArgMax),
            "argmin" => Some(KnownFunction::ArgMin),
            "argsort" => Some(KnownFunction::Argsort),
            "sort" => Some(KnownFunction::Sort),
            "cumsum" => Some(KnownFunction::Cumsum),
            "cumprod" => Some(KnownFunction::Cumprod),
            "zeros" => Some(KnownFunction::Zeros),
            "ones" => Some(KnownFunction::Ones),
            "full" => Some(KnownFunction::Full),
            "empty" => Some(KnownFunction::Empty),
            "zeros_like" => Some(KnownFunction::ZerosLike),
            "ones_like" => Some(KnownFunction::OnesLike),
            "full_like" => Some(KnownFunction::FullLike),
            "empty_like" => Some(KnownFunction::EmptyLike),
            "arange" => Some(KnownFunction::Arange),
            "linspace" => Some(KnownFunction::Linspace),
            "eye" => Some(KnownFunction::Eye),
            "broadcast_to" => Some(KnownFunction::BroadcastTo),
            "broadcast_tensors" => Some(KnownFunction::BroadcastArrays),
            "atleast_1d" => Some(KnownFunction::AtLeast1D),
            "atleast_2d" => Some(KnownFunction::AtLeast2D),
            "atleast_3d" => Some(KnownFunction::AtLeast3D),
            "roll" => Some(KnownFunction::Roll),
            "flip" | "fliplr" | "flipud" => Some(KnownFunction::Flip),
            "rot90" => Some(KnownFunction::Rot90),
            "take" => Some(KnownFunction::Take),
            "diag" => Some(KnownFunction::Diag),
            "diagonal" => Some(KnownFunction::Diagonal),
            "trace" => Some(KnownFunction::Trace),
            "triu" => Some(KnownFunction::Triu),
            "tril" => Some(KnownFunction::Tril),
            "meshgrid" => Some(KnownFunction::Meshgrid),
            "vstack" | "row_stack" => Some(KnownFunction::Vstack),
            "hstack" => Some(KnownFunction::Hstack),
            "dstack" => Some(KnownFunction::Dstack),
            "column_stack" => Some(KnownFunction::ColumnStack),
            "permute" => Some(KnownFunction::Permute),
            "tensor" => Some(KnownFunction::Array),
            "as_tensor" => Some(KnownFunction::AsArray),
            _ => None,
        };
    }

    let is_jax_numpy_linalg = module == ["jax", "numpy", "linalg"];
    let is_numpy_linalg = module == ["numpy", "linalg"];
    let is_torch_linalg = module == ["torch", "linalg"];
    let is_torch_nn_functional = module == ["torch", "nn", "functional"];

    if is_torch_nn_functional {
        return match name.as_str() {
            "pad" => Some(KnownFunction::Pad),
            _ => None,
        };
    }

    if is_jax_numpy_linalg || is_numpy_linalg || is_torch_linalg {
        return match name.as_str() {
            "inv" => Some(KnownFunction::LinalgInv),
            _ => None,
        };
    }

    if is_jax_lax {
        return match name.as_str() {
            "dot" => Some(KnownFunction::Dot),
            "dot_general" => Some(KnownFunction::Matmul),
            _ => None,
        };
    }

    None
}

fn parse_simple_sequence_names(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let names = inner
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string())
        .collect::<Vec<_>>();

    if names.is_empty() { None } else { Some(names) }
}

fn parse_axis(value: &str) -> Option<isize> {
    value.trim().parse::<isize>().ok()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim() {
        "True" | "true" => Some(true),
        "False" | "false" => Some(false),
        _ => None,
    }
}

fn parse_axis_list(value: &str) -> Option<Vec<isize>> {
    let trimmed = value.trim();
    if trimmed == "None" {
        return Some(vec![isize::MIN]);
    }
    if let Some(axis) = parse_axis(trimmed) {
        return Some(vec![axis]);
    }
    let inner = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return None;
    };
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(parse_axis)
        .collect()
}

fn first_array_arg(args: &[CallArgument]) -> Option<&str> {
    for arg in args {
        match arg {
            CallArgument::Positional { value } => return Some(value),
            CallArgument::Keyword { name, value }
                if name == "a" || name == "x" || name == "input" || name == "array" =>
            {
                return Some(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    None
}

fn parse_shape_value(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let dims = inner
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>();

    if dims.is_empty() { None } else { Some(dims) }
}

fn dim_product(dims: &[String]) -> Option<usize> {
    dims.iter()
        .map(|dim| dim.parse::<usize>())
        .try_fold(1usize, |acc, dim| dim.map(|dim| acc * dim).ok())
}

fn flattened_dim(dims: &[String]) -> String {
    if let Some(product) = dim_product(dims) {
        return product.to_string();
    }
    dims.join("*")
}

fn multiply_dim(dim: &str, factor: &str) -> String {
    if factor == "1" {
        return dim.to_string();
    }
    if dim == "1" {
        return factor.to_string();
    }
    if let (Ok(dim), Ok(factor)) = (dim.parse::<usize>(), factor.parse::<usize>()) {
        return (dim * factor).to_string();
    }
    format!("{}*{}", dim, factor)
}

fn add_to_dim(dim: &str, amount: isize) -> String {
    if amount == 0 {
        return dim.to_string();
    }
    if let Ok(dim) = dim.parse::<isize>() {
        return (dim + amount).to_string();
    }
    if amount > 0 {
        format!("{}+{}", dim, amount)
    } else {
        format!("{}{}", dim, amount)
    }
}

fn parse_ints(value: &str) -> Vec<isize> {
    let mut ints = Vec::new();
    let mut current = String::new();

    for ch in value.chars() {
        if ch.is_ascii_digit() || (ch == '-' && current.is_empty()) {
            current.push(ch);
        } else if !current.is_empty() && current != "-" {
            if let Ok(value) = current.parse::<isize>() {
                ints.push(value);
            }
            current.clear();
        } else {
            current.clear();
        }
    }

    if !current.is_empty()
        && current != "-"
        && let Ok(value) = current.parse::<isize>()
    {
        ints.push(value);
    }

    ints
}

fn broadcast_two_shapes(left: &[String], right: &[String]) -> Result<Vec<String>, String> {
    let rank = left.len().max(right.len());
    let mut result = Vec::new();

    for i in 0..rank {
        let left_dim = left
            .get(left.len().wrapping_sub(1 + i))
            .map(String::as_str)
            .unwrap_or("1");
        let right_dim = right
            .get(right.len().wrapping_sub(1 + i))
            .map(String::as_str)
            .unwrap_or("1");

        let dim = if left_dim == right_dim {
            left_dim.to_string()
        } else if left_dim == "1" {
            right_dim.to_string()
        } else if right_dim == "1" {
            left_dim.to_string()
        } else {
            return Err(format!(
                "cannot broadcast dimensions {} and {}",
                left_dim, right_dim
            ));
        };
        result.push(dim);
    }

    result.reverse();
    Ok(result)
}

fn concat_dim(dims: &[String]) -> String {
    let parsed = dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<Result<Vec<_>, _>>();

    if let Ok(values) = parsed {
        return values.iter().sum::<usize>().to_string();
    }

    dims.join("+")
}

pub fn classify_method_call(method: &str) -> Option<KnownFunction> {
    match method {
        "reshape" | "view" => Some(KnownFunction::Reshape),
        "flatten" => Some(KnownFunction::Flatten),
        "ravel" => Some(KnownFunction::Ravel),
        "transpose" => Some(KnownFunction::Transpose),
        "permute" => Some(KnownFunction::Permute),
        "swapaxes" => Some(KnownFunction::SwapAxes),
        "moveaxis" => Some(KnownFunction::MoveAxis),
        "squeeze" => Some(KnownFunction::Squeeze),
        "unsqueeze" | "expand_dims" => Some(KnownFunction::ExpandDims),
        "sum" => Some(KnownFunction::Sum),
        "mean" => Some(KnownFunction::Mean),
        "max" => Some(KnownFunction::Max),
        "min" => Some(KnownFunction::Min),
        "prod" => Some(KnownFunction::Prod),
        "std" => Some(KnownFunction::Std),
        "var" => Some(KnownFunction::Var),
        "all" => Some(KnownFunction::All),
        "any" => Some(KnownFunction::Any),
        "argmax" => Some(KnownFunction::ArgMax),
        "argmin" => Some(KnownFunction::ArgMin),
        "argsort" => Some(KnownFunction::Argsort),
        "sort" => Some(KnownFunction::Sort),
        "cumsum" => Some(KnownFunction::Cumsum),
        "cumprod" => Some(KnownFunction::Cumprod),
        _ => None,
    }
}

pub fn apply_method_call(
    method: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let synthesized = synthesize_method_args(method, receiver, args);
    apply_known_function(method, &synthesized, shapes)
}

fn synthesize_method_args(
    method: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
) -> Vec<CallArgument> {
    let positional_count = args
        .iter()
        .filter(|arg| matches!(arg, CallArgument::Positional { .. }))
        .count();

    let needs_tuple_collapse = matches!(
        method,
        KnownFunction::Reshape | KnownFunction::Permute | KnownFunction::Transpose
    ) && positional_count >= 2;

    let mut result = Vec::with_capacity(args.len() + 1);
    result.push(CallArgument::Positional {
        value: receiver.to_string(),
    });

    if needs_tuple_collapse {
        let positionals: Vec<String> = args
            .iter()
            .filter_map(|arg| match arg {
                CallArgument::Positional { value } => Some(value.clone()),
                _ => None,
            })
            .collect();
        let tuple_str = format!("({})", positionals.join(", "));
        result.push(CallArgument::Positional { value: tuple_str });
        for arg in args {
            if let CallArgument::Keyword { .. } = arg {
                result.push(arg.clone());
            }
        }
    } else {
        result.extend_from_slice(args);
    }

    result
}

pub fn apply_known_function(
    function: &KnownFunction,
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    match function {
        KnownFunction::Concatenate => apply_known_concatenate(args, shapes),
        KnownFunction::Stack => apply_known_stack(args, shapes),
        KnownFunction::Reshape => apply_known_reshape(args, shapes),
        KnownFunction::Flatten | KnownFunction::Ravel => apply_known_flatten(args, shapes),
        KnownFunction::Transpose | KnownFunction::Permute => apply_known_transpose(args, shapes),
        KnownFunction::SwapAxes => apply_known_swapaxes(args, shapes),
        KnownFunction::MoveAxis => apply_known_moveaxis(args, shapes),
        KnownFunction::ExpandDims => apply_known_expand_dims(args, shapes),
        KnownFunction::Squeeze => apply_known_squeeze(args, shapes),
        KnownFunction::AtLeast1D => apply_known_atleast(args, shapes, 1),
        KnownFunction::AtLeast2D => apply_known_atleast(args, shapes, 2),
        KnownFunction::AtLeast3D => apply_known_atleast(args, shapes, 3),
        KnownFunction::Zeros | KnownFunction::Ones | KnownFunction::Full | KnownFunction::Empty => {
            apply_known_shape_constructor(args)
        }
        KnownFunction::ZerosLike
        | KnownFunction::OnesLike
        | KnownFunction::FullLike
        | KnownFunction::EmptyLike
        | KnownFunction::Array
        | KnownFunction::AsArray
        | KnownFunction::Argsort
        | KnownFunction::Sort
        | KnownFunction::Cumsum
        | KnownFunction::Cumprod => apply_known_shape_preserving(args, shapes),
        KnownFunction::Arange => apply_known_arange(args),
        KnownFunction::Linspace | KnownFunction::Logspace => apply_known_linspace(args),
        KnownFunction::Eye | KnownFunction::Identity => apply_known_eye(args),
        KnownFunction::Matmul => apply_known_matmul(args, shapes),
        KnownFunction::Dot => apply_known_dot(args, shapes),
        KnownFunction::TensorDot => apply_known_tensordot(args, shapes),
        KnownFunction::Outer => apply_known_outer(args, shapes),
        KnownFunction::Inner => apply_known_inner(args, shapes),
        KnownFunction::Vdot => apply_known_vdot(args, shapes),
        KnownFunction::Diag => apply_known_diag(args, shapes),
        KnownFunction::Diagonal => apply_known_diagonal(args, shapes),
        KnownFunction::Trace => apply_known_trace(args, shapes),
        KnownFunction::Take => apply_known_take(args, shapes),
        KnownFunction::BroadcastTo => apply_known_broadcast_to(args, shapes),
        KnownFunction::BroadcastArrays => apply_known_broadcast_arrays(args, shapes),
        KnownFunction::Tile => apply_known_tile(args, shapes),
        KnownFunction::Repeat => apply_known_repeat(args, shapes),
        KnownFunction::Roll | KnownFunction::Flip | KnownFunction::Triu | KnownFunction::Tril => {
            apply_known_shape_preserving(args, shapes)
        }
        KnownFunction::Pad => apply_known_pad(args, shapes),
        KnownFunction::Rot90 => apply_known_rot90(args, shapes),
        KnownFunction::Vstack => apply_known_stack_family(args, shapes, "vstack"),
        KnownFunction::Hstack => apply_known_stack_family(args, shapes, "hstack"),
        KnownFunction::Dstack => apply_known_stack_family(args, shapes, "dstack"),
        KnownFunction::ColumnStack => apply_known_stack_family(args, shapes, "column_stack"),
        KnownFunction::Where => apply_known_where(args, shapes),
        KnownFunction::Sum
        | KnownFunction::Mean
        | KnownFunction::Max
        | KnownFunction::Min
        | KnownFunction::Prod
        | KnownFunction::Std
        | KnownFunction::Var
        | KnownFunction::All
        | KnownFunction::Any
        | KnownFunction::ArgMax
        | KnownFunction::ArgMin => apply_known_reduction(args, shapes),
        KnownFunction::LinalgInv => apply_known_linalg_inv(args, shapes),
        _ => Ok(None),
    }
}

fn sequence_arg_value(args: &[CallArgument]) -> Option<&str> {
    let first_arg = args.first()?;
    match first_arg {
        CallArgument::Positional { value } => Some(value),
        CallArgument::Keyword { name, value }
            if name == "arrays" || name == "tensors" || name == "arys" =>
        {
            Some(value)
        }
        CallArgument::Keyword { .. } => None,
    }
}

fn axis_arg(args: &[CallArgument], default: isize) -> isize {
    let mut axis = default;
    for arg in args.iter().skip(1) {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(parsed) = parse_axis(value) {
                    axis = parsed;
                }
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                if let Some(parsed) = parse_axis(value) {
                    axis = parsed;
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    axis
}

fn apply_known_concatenate(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };

    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let axis = axis_arg(args, 0);

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.get(input_name) else {
            return Ok(None);
        };
        input_shapes.push(shape.clone());
    }

    let Some(first_shape) = input_shapes.first() else {
        return Ok(None);
    };
    if first_shape.is_empty() {
        return Err("concatenate requires rank >= 1 inputs".to_string());
    }

    let rank = first_shape.len();
    let axis = if axis < 0 { rank as isize + axis } else { axis };
    if axis < 0 || axis as usize >= rank {
        return Err(format!(
            "concatenate axis {} out of bounds for rank {}",
            axis, rank
        ));
    }
    let axis = axis as usize;

    for shape in &input_shapes {
        if shape.len() != rank {
            return Err(format!(
                "concatenate expected all inputs to have rank {}, got rank {}",
                rank,
                shape.len()
            ));
        }
        for dim_idx in 0..rank {
            if dim_idx == axis {
                continue;
            }
            if shape[dim_idx] != first_shape[dim_idx] {
                return Err(format!(
                    "concatenate dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, first_shape[dim_idx], shape[dim_idx]
                ));
            }
        }
    }

    let mut output = first_shape.clone();
    let concat_dims = input_shapes
        .iter()
        .map(|shape| shape[axis].clone())
        .collect::<Vec<_>>();
    output[axis] = concat_dim(&concat_dims);

    Ok(Some(output))
}

fn apply_known_stack(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };
    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.get(input_name) else {
            return Ok(None);
        };
        input_shapes.push(shape.clone());
    }

    let Some(first_shape) = input_shapes.first() else {
        return Ok(None);
    };

    for shape in &input_shapes {
        if shape.len() != first_shape.len() {
            return Err(format!(
                "stack expected all inputs to have rank {}, got rank {}",
                first_shape.len(),
                shape.len()
            ));
        }
        for (dim_idx, (expected, got)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if expected != got {
                return Err(format!(
                    "stack dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, expected, got
                ));
            }
        }
    }

    let output_rank = first_shape.len() + 1;
    let axis = axis_arg(args, 0);
    let axis = if axis < 0 {
        output_rank as isize + axis
    } else {
        axis
    };
    if axis < 0 || axis as usize > first_shape.len() {
        return Err(format!(
            "stack axis {} out of bounds for output rank {}",
            axis, output_rank
        ));
    }

    let mut output = first_shape.clone();
    output.insert(axis as usize, input_shapes.len().to_string());
    Ok(Some(output))
}

fn apply_known_reshape(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut shape_value = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if shape_value.is_none() {
                    shape_value = Some(value.as_str());
                }
            }
            CallArgument::Keyword { name, value }
                if name == "shape" || name == "newshape" || name == "size" =>
            {
                shape_value = Some(value.as_str());
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let Some(shape_value) = shape_value else {
        return Ok(None);
    };
    let Some(mut output_shape) = parse_shape_value(shape_value) else {
        return Ok(None);
    };

    let minus_one_count = output_shape
        .iter()
        .filter(|dim| dim.as_str() == "-1")
        .count();
    if minus_one_count > 1 {
        return Err("reshape can only infer one -1 dimension".to_string());
    }

    if minus_one_count == 1 {
        let Some(input_product) = dim_product(input_shape) else {
            return Err("reshape cannot infer -1 dimension for symbolic input shape".to_string());
        };
        let known_dims = output_shape
            .iter()
            .filter(|dim| dim.as_str() != "-1")
            .cloned()
            .collect::<Vec<_>>();
        let Some(known_product) = dim_product(&known_dims) else {
            return Err("reshape cannot infer -1 dimension with symbolic target shape".to_string());
        };
        if known_product == 0 || input_product % known_product != 0 {
            return Err(format!(
                "reshape cannot infer -1 dimension: input size {} not divisible by {}",
                input_product, known_product
            ));
        }
        let inferred = (input_product / known_product).to_string();
        for dim in &mut output_shape {
            if dim == "-1" {
                *dim = inferred.clone();
            }
        }
    }

    if minus_one_count == 0
        && let (Some(input_product), Some(output_product)) =
            (dim_product(input_shape), dim_product(&output_shape))
        && input_product != output_product
    {
        return Err(format!(
            "reshape changes total size from {} to {}",
            input_product, output_product
        ));
    }

    Ok(Some(output_shape))
}

fn apply_known_flatten(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    Ok(Some(vec![flattened_dim(input_shape)]))
}

fn normalize_axis(axis: isize, rank: usize, context: &str) -> Result<usize, String> {
    let normalized = if axis < 0 { rank as isize + axis } else { axis };
    if normalized < 0 || normalized as usize >= rank {
        return Err(format!(
            "{} axis {} out of bounds for rank {}",
            context, normalized, rank
        ));
    }
    Ok(normalized as usize)
}

fn apply_known_transpose(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut axes = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { name, value }
                if name == "axes" || name == "axis" || name == "dims" =>
            {
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let axes = axes.unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose expected {} axes, got {}",
            rank,
            axes.len()
        ));
    }
    let mut normalized = Vec::new();
    for axis in axes {
        let axis = normalize_axis(axis, rank, "transpose")?;
        if normalized.contains(&axis) {
            return Err(format!("transpose duplicate axis {}", axis));
        }
        normalized.push(axis);
    }

    Ok(Some(
        normalized
            .iter()
            .map(|axis| input_shape[*axis].clone())
            .collect(),
    ))
}

fn apply_known_swapaxes(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut axes = Vec::new();
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if let Some(axis) = parse_axis(value) {
                    axes.push(axis);
                }
            }
            CallArgument::Keyword { name, value }
                if name == "axis1" || name == "axis2" || name == "dim0" || name == "dim1" =>
            {
                if let Some(axis) = parse_axis(value) {
                    axes.push(axis);
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    if axes.len() != 2 {
        return Ok(None);
    }

    let axis0 = normalize_axis(axes[0], rank, "swapaxes")?;
    let axis1 = normalize_axis(axes[1], rank, "swapaxes")?;
    let mut output = input_shape.clone();
    output.swap(axis0, axis1);
    Ok(Some(output))
}

fn apply_known_moveaxis(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut source = None;
    let mut destination = None;
    let mut positional = Vec::new();
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                positional.push(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "source" => {
                source = parse_axis_list(value)
            }
            CallArgument::Keyword { name, value } if name == "destination" => {
                destination = parse_axis_list(value)
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    if source.is_none() && !positional.is_empty() {
        source = parse_axis_list(positional[0]);
    }
    if destination.is_none() && positional.len() > 1 {
        destination = parse_axis_list(positional[1]);
    }
    let Some(source) = source else {
        return Ok(None);
    };
    let Some(destination) = destination else {
        return Ok(None);
    };
    if source.len() != destination.len() {
        return Err("moveaxis source and destination lengths differ".to_string());
    }

    let source = source
        .into_iter()
        .map(|axis| normalize_axis(axis, rank, "moveaxis"))
        .collect::<Result<Vec<_>, _>>()?;
    let destination = destination
        .into_iter()
        .map(|axis| normalize_axis(axis, rank, "moveaxis"))
        .collect::<Result<Vec<_>, _>>()?;

    let mut order = (0..rank)
        .filter(|axis| !source.contains(axis))
        .collect::<Vec<_>>();
    for (src, dst) in source.iter().zip(destination.iter()) {
        let insert_at = (*dst).min(order.len());
        order.insert(insert_at, *src);
    }

    Ok(Some(
        order
            .iter()
            .map(|axis| input_shape[*axis].clone())
            .collect(),
    ))
}

fn apply_known_expand_dims(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let output_rank = input_shape.len() + 1;
    let mut axis = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                axis = parse_axis(value);
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                axis = parse_axis(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    let Some(axis) = axis else {
        return Ok(None);
    };
    let axis = if axis < 0 {
        output_rank as isize + axis
    } else {
        axis
    };
    if axis < 0 || axis as usize > input_shape.len() {
        return Err(format!(
            "expand_dims axis {} out of bounds for output rank {}",
            axis, output_rank
        ));
    }
    let mut output = input_shape.clone();
    output.insert(axis as usize, "1".to_string());
    Ok(Some(output))
}

fn apply_known_squeeze(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let mut axes = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let rank = input_shape.len();
    let axes = if let Some(axes) = axes {
        axes.into_iter()
            .map(|axis| normalize_axis(axis, rank, "squeeze"))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..rank)
            .filter(|axis| input_shape[*axis] == "1")
            .collect::<Vec<_>>()
    };

    for axis in &axes {
        if input_shape[*axis] != "1" {
            return Err(format!(
                "cannot squeeze axis {} with dimension {}",
                axis, input_shape[*axis]
            ));
        }
    }

    let output = input_shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, dim)| dim.clone())
        .collect();
    Ok(Some(output))
}

fn apply_known_atleast(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
    min_rank: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    if input_shape.len() >= min_rank {
        return Ok(Some(input_shape.clone()));
    }
    let mut output = input_shape.clone();
    while output.len() < min_rank {
        output.insert(0, "1".to_string());
    }
    Ok(Some(output))
}

fn positional_arg_values(args: &[CallArgument]) -> Vec<String> {
    args.iter()
        .filter_map(|arg| match arg {
            CallArgument::Positional { value } => Some(value.clone()),
            CallArgument::Keyword { .. } => None,
        })
        .collect()
}

fn numeric_min_dim(left: &str, right: &str) -> String {
    if left == right {
        return left.to_string();
    }
    if let (Ok(left), Ok(right)) = (left.parse::<usize>(), right.parse::<usize>()) {
        return left.min(right).to_string();
    }
    format!("min({},{})", left, right)
}

fn first_two_positional_values(args: &[CallArgument]) -> Option<(String, String)> {
    let values = positional_arg_values(args);
    Some((values.first()?.clone(), values.get(1)?.clone()))
}

fn check_dim_match(left: &str, right: &str, context: &str) -> Result<(), String> {
    if left == right {
        Ok(())
    } else {
        Err(format!(
            "{} dimension mismatch: expected {}, got {}",
            context, left, right
        ))
    }
}

fn broadcast_prefix_shapes(
    left: &[String],
    right: &[String],
    context: &str,
) -> Result<Vec<String>, String> {
    broadcast_two_shapes(left, right).map_err(|err| format!("{} {}", context, err))
}

fn apply_known_shape_constructor(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(first) = args.first() else {
        return Ok(None);
    };
    let value = match first {
        CallArgument::Positional { value } => value,
        CallArgument::Keyword { name, value } if name == "shape" || name == "size" => value,
        CallArgument::Keyword { .. } => return Ok(None),
    };
    Ok(parse_shape_value(value))
}

fn apply_known_arange(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() == 1 {
        return Ok(Some(vec![values[0].clone()]));
    }
    if let (Ok(start), Ok(stop)) = (values[0].parse::<isize>(), values[1].parse::<isize>()) {
        return Ok(Some(vec![(stop - start).max(0).to_string()]));
    }
    Ok(Some(vec![format!("{}-{}", values[1], values[0])]))
}

fn apply_known_linspace(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    for arg in args {
        if let CallArgument::Keyword { name, value } = arg
            && name == "num"
        {
            return Ok(Some(vec![value.clone()]));
        }
    }
    let values = positional_arg_values(args);
    if let Some(num) = values.get(2) {
        return Ok(Some(vec![num.clone()]));
    }
    Ok(Some(vec!["50".to_string()]))
}

fn apply_known_eye(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    let Some(n) = values.first() else {
        return Ok(None);
    };
    let m = values.get(1).unwrap_or(n);
    Ok(Some(vec![n.clone(), m.clone()]))
}

fn apply_known_matmul(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    match (left.len(), right.len()) {
        (0, _) | (_, 0) => Err("matmul does not support scalar inputs".to_string()),
        (1, 1) => {
            check_dim_match(&left[0], &right[0], "matmul")?;
            Ok(Some(Vec::new()))
        }
        (1, 2) => {
            check_dim_match(&left[0], &right[0], "matmul")?;
            Ok(Some(vec![right[1].clone()]))
        }
        (2, 1) => {
            check_dim_match(&left[1], &right[0], "matmul")?;
            Ok(Some(vec![left[0].clone()]))
        }
        _ => {
            let left_batch_end = left.len().saturating_sub(2);
            let right_batch_end = right.len().saturating_sub(2);
            let left_batch = &left[..left_batch_end];
            let right_batch = &right[..right_batch_end];
            let batch = broadcast_prefix_shapes(left_batch, right_batch, "matmul")?;

            let left_m = if left.len() == 1 {
                None
            } else {
                Some(left[left.len() - 2].clone())
            };
            let left_k = left.last().unwrap();
            let right_k = if right.len() == 1 {
                right.last().unwrap()
            } else {
                &right[right.len() - 2]
            };
            let right_n = if right.len() == 1 {
                None
            } else {
                Some(right[right.len() - 1].clone())
            };
            check_dim_match(left_k, right_k, "matmul")?;

            let mut output = batch;
            if let Some(m) = left_m {
                output.push(m);
            }
            if let Some(n) = right_n {
                output.push(n);
            }
            Ok(Some(output))
        }
    }
}

fn apply_known_dot(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    if left.is_empty() || right.is_empty() {
        return Err("dot does not support scalar inputs".to_string());
    }

    if right.len() == 1 {
        check_dim_match(left.last().unwrap(), &right[0], "dot")?;
        return Ok(Some(left[..left.len() - 1].to_vec()));
    }

    check_dim_match(left.last().unwrap(), &right[right.len() - 2], "dot")?;
    let mut output = left[..left.len() - 1].to_vec();
    output.extend(right[..right.len() - 2].to_vec());
    output.push(right[right.len() - 1].clone());
    Ok(Some(output))
}

fn apply_known_tensordot(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    let axes_keyword = args.iter().find_map(|arg| match arg {
        CallArgument::Keyword { name, value } if name == "axes" || name == "dims" => {
            Some(value.as_str())
        }
        _ => None,
    });
    let third_positional = positional_arg_values(args).get(2).cloned();
    let axes_value = axes_keyword
        .map(str::to_string)
        .or(third_positional)
        .unwrap_or_else(|| "2".to_string());

    let n = match parse_axis(&axes_value) {
        Some(n) if n >= 0 => n as usize,
        Some(_) => return Err("tensordot axes must be non-negative".to_string()),
        None => return Ok(None),
    };

    if n > left.len() {
        return Err(format!(
            "tensordot axes {} exceeds left rank {}",
            n,
            left.len()
        ));
    }
    if n > right.len() {
        return Err(format!(
            "tensordot axes {} exceeds right rank {}",
            n,
            right.len()
        ));
    }

    let left_keep = &left[..left.len() - n];
    let left_contract = &left[left.len() - n..];
    let right_contract = &right[..n];
    let right_keep = &right[n..];

    for (l, r) in left_contract.iter().zip(right_contract.iter()) {
        check_dim_match(l, r, "tensordot")?;
    }

    let mut output = left_keep.to_vec();
    output.extend(right_keep.iter().cloned());
    Ok(Some(output))
}

fn apply_known_outer(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    if left.is_empty() || right.is_empty() {
        return Err("outer does not support scalar inputs".to_string());
    }

    let left_dim = if left.len() == 1 {
        left[0].clone()
    } else {
        flattened_dim(left)
    };
    let right_dim = if right.len() == 1 {
        right[0].clone()
    } else {
        flattened_dim(right)
    };

    Ok(Some(vec![left_dim, right_dim]))
}

fn apply_known_inner(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    if left.is_empty() || right.is_empty() {
        return Err("inner does not support scalar inputs".to_string());
    }

    check_dim_match(left.last().unwrap(), right.last().unwrap(), "inner")?;

    let mut output = left[..left.len() - 1].to_vec();
    output.extend(right[..right.len() - 1].iter().cloned());
    Ok(Some(output))
}

fn apply_known_vdot(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some((left_name, right_name)) = first_two_positional_values(args) else {
        return Ok(None);
    };
    let Some(_left) = shapes.get(&left_name) else {
        return Ok(None);
    };
    let Some(_right) = shapes.get(&right_name) else {
        return Ok(None);
    };

    Ok(Some(Vec::new()))
}

fn apply_known_diag(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    match input_shape.as_slice() {
        [n] => Ok(Some(vec![n.clone(), n.clone()])),
        [m, n] => Ok(Some(vec![numeric_min_dim(m, n)])),
        _ => Err(format!(
            "diag expects rank 1 or 2, got rank {}",
            input_shape.len()
        )),
    }
}

fn apply_known_diagonal(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "diagonal expects rank >= 2, got rank {}",
            input_shape.len()
        ));
    }
    let mut output = input_shape[..input_shape.len() - 2].to_vec();
    output.push(numeric_min_dim(
        &input_shape[input_shape.len() - 2],
        &input_shape[input_shape.len() - 1],
    ));
    Ok(Some(output))
}

fn apply_known_trace(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "trace expects rank >= 2, got rank {}",
            input_shape.len()
        ));
    }
    Ok(Some(input_shape[..input_shape.len() - 2].to_vec()))
}

fn apply_known_take(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 2 {
        return Ok(None);
    }
    let Some(input_shape) = shapes.get(&values[0]) else {
        return Ok(None);
    };
    let Some(indices_shape) = shapes.get(&values[1]) else {
        return Ok(None);
    };
    let mut axis = None;
    for arg in args.iter().skip(2) {
        match arg {
            CallArgument::Positional { value } => axis = parse_axis(value),
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                axis = parse_axis(value)
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    let Some(axis) = axis else {
        return Ok(Some(indices_shape.clone()));
    };
    let axis = normalize_axis(axis, input_shape.len(), "take")?;
    let mut output = input_shape[..axis].to_vec();
    output.extend(indices_shape.clone());
    output.extend(input_shape[axis + 1..].to_vec());
    Ok(Some(output))
}

fn apply_known_broadcast_to(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    if !shapes.contains_key(input_name) {
        return Ok(None);
    }

    let mut shape_value = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                shape_value = Some(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "shape" => shape_value = Some(value),
            CallArgument::Keyword { .. } => {}
        }
    }

    Ok(shape_value.and_then(parse_shape_value))
}

fn apply_known_broadcast_arrays(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    let input_names = if values.len() == 1 {
        parse_simple_sequence_names(&values[0]).unwrap_or(values)
    } else {
        values
    };
    if input_names.is_empty() {
        return Ok(None);
    }

    let mut output = Vec::new();
    for input_name in input_names {
        let Some(shape) = shapes.get(&input_name) else {
            return Ok(None);
        };
        output = broadcast_two_shapes(&output, shape)?;
    }
    Ok(Some(output))
}

fn apply_known_tile(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut reps_value = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                reps_value = Some(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "reps" || name == "dims" => {
                reps_value = Some(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    let Some(reps) = reps_value.and_then(parse_shape_value) else {
        return Ok(None);
    };

    let rank = input_shape.len().max(reps.len());
    let mut shape = vec!["1".to_string(); rank - input_shape.len()];
    shape.extend(input_shape.clone());
    let mut reps_padded = vec!["1".to_string(); rank - reps.len()];
    reps_padded.extend(reps);

    Ok(Some(
        shape
            .iter()
            .zip(reps_padded.iter())
            .map(|(dim, rep)| multiply_dim(dim, rep))
            .collect(),
    ))
}

fn apply_known_repeat(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut repeats = None;
    let mut axis = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if repeats.is_none() {
                    repeats = Some(value.as_str());
                } else if axis.is_none() {
                    axis = parse_axis(value);
                }
            }
            CallArgument::Keyword { name, value } if name == "repeats" => repeats = Some(value),
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                axis = parse_axis(value)
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    let Some(repeats) = repeats else {
        return Ok(None);
    };

    if let Some(axis) = axis {
        let axis = normalize_axis(axis, input_shape.len(), "repeat")?;
        let mut output = input_shape.clone();
        output[axis] = multiply_dim(&output[axis], repeats);
        return Ok(Some(output));
    }

    Ok(Some(vec![multiply_dim(
        &flattened_dim(input_shape),
        repeats,
    )]))
}

fn apply_known_shape_preserving(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    Ok(shapes.get(input_name).cloned())
}

fn concat_shapes_along_axis(
    input_shapes: &[Vec<String>],
    axis: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_shape) = input_shapes.first() else {
        return Ok(None);
    };
    let rank = first_shape.len();
    if axis >= rank {
        return Err(format!(
            "concat axis {} out of bounds for rank {}",
            axis, rank
        ));
    }

    for shape in input_shapes {
        if shape.len() != rank {
            return Err(format!(
                "concat expected rank {}, got rank {}",
                rank,
                shape.len()
            ));
        }
        for dim_idx in 0..rank {
            if dim_idx == axis {
                continue;
            }
            if shape[dim_idx] != first_shape[dim_idx] {
                return Err(format!(
                    "concat dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, first_shape[dim_idx], shape[dim_idx]
                ));
            }
        }
    }

    let mut output = first_shape.clone();
    output[axis] = concat_dim(
        &input_shapes
            .iter()
            .map(|shape| shape[axis].clone())
            .collect::<Vec<_>>(),
    );
    Ok(Some(output))
}

fn stack_family_shape(kind: &str, shape: &[String]) -> Vec<String> {
    match kind {
        "vstack" => {
            if shape.len() == 1 {
                vec!["1".to_string(), shape[0].clone()]
            } else {
                shape.to_vec()
            }
        }
        "hstack" => shape.to_vec(),
        "dstack" => match shape.len() {
            0 => vec!["1".to_string(), "1".to_string(), "1".to_string()],
            1 => vec!["1".to_string(), shape[0].clone(), "1".to_string()],
            2 => vec![shape[0].clone(), shape[1].clone(), "1".to_string()],
            _ => shape.to_vec(),
        },
        "column_stack" => {
            if shape.len() == 1 {
                vec![shape[0].clone(), "1".to_string()]
            } else {
                shape.to_vec()
            }
        }
        _ => shape.to_vec(),
    }
}

fn apply_known_stack_family(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
    kind: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };
    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let mut input_shapes = Vec::new();
    for input_name in input_names {
        let Some(shape) = shapes.get(&input_name) else {
            return Ok(None);
        };
        input_shapes.push(stack_family_shape(kind, shape));
    }

    let axis = match kind {
        "vstack" => 0,
        "hstack" => {
            if input_shapes.first().map(|shape| shape.len()) == Some(1) {
                0
            } else {
                1
            }
        }
        "dstack" => 2,
        "column_stack" => 1,
        _ => 0,
    };

    concat_shapes_along_axis(&input_shapes, axis)
}

fn apply_known_rot90(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "rot90 expects rank >= 2, got rank {}",
            input_shape.len()
        ));
    }

    let mut k = 1;
    let mut axes = vec![0, 1];
    let mut seen_first_positional = false;
    let mut positional_after_input = Vec::new();
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                positional_after_input.push(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "k" => {
                if let Some(parsed) = parse_axis(value) {
                    k = parsed;
                }
            }
            CallArgument::Keyword { name, value } if name == "axes" => {
                if let Some(parsed) = parse_axis_list(value) {
                    axes = parsed;
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    if let Some(value) = positional_after_input.first()
        && let Some(parsed) = parse_axis(value)
    {
        k = parsed;
    }
    if let Some(value) = positional_after_input.get(1)
        && let Some(parsed) = parse_axis_list(value)
    {
        axes = parsed;
    }
    if axes.len() != 2 {
        return Err("rot90 expects exactly two axes".to_string());
    }
    let axis0 = normalize_axis(axes[0], input_shape.len(), "rot90")?;
    let axis1 = normalize_axis(axes[1], input_shape.len(), "rot90")?;
    if axis0 == axis1 {
        return Err("rot90 axes must be different".to_string());
    }

    let mut output = input_shape.clone();
    if k.rem_euclid(2) == 1 {
        output.swap(axis0, axis1);
    }
    Ok(Some(output))
}

fn apply_known_pad(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut pad_width = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                pad_width = Some(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "pad_width" || name == "pad" => {
                pad_width = Some(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let Some(pad_width) = pad_width else {
        return Ok(None);
    };
    let values = parse_ints(pad_width);
    if values.is_empty() {
        return Ok(None);
    }

    let additions = if values.len() == 1 {
        vec![values[0] * 2; input_shape.len()]
    } else if values.len() == 2 {
        vec![values[0] + values[1]; input_shape.len()]
    } else if values.len() == input_shape.len() * 2 {
        values
            .chunks_exact(2)
            .map(|pair| pair[0] + pair[1])
            .collect::<Vec<_>>()
    } else {
        return Ok(None);
    };

    Ok(Some(
        input_shape
            .iter()
            .zip(additions.iter())
            .map(|(dim, addition)| add_to_dim(dim, *addition))
            .collect(),
    ))
}

fn apply_known_where(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 3 {
        return Ok(None);
    }
    let mut output = Vec::new();
    for value in values.iter().take(3) {
        let Some(shape) = shapes.get(value) else {
            return Ok(None);
        };
        output = broadcast_two_shapes(&output, shape)?;
    }
    Ok(Some(output))
}

fn apply_known_reduction(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut axes: Option<Vec<isize>> = None;
    let mut invalid_axis = false;
    let mut keepdims = false;
    let mut seen_first_positional = false;

    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if axes.is_none() {
                    match parse_axis_list(value) {
                        Some(parsed) => axes = Some(parsed),
                        None => invalid_axis = true,
                    }
                }
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                match parse_axis_list(value) {
                    Some(parsed) => axes = Some(parsed),
                    None => invalid_axis = true,
                }
            }
            CallArgument::Keyword { name, value } if name == "keepdims" || name == "keepdim" => {
                if let Some(parsed) = parse_bool(value) {
                    keepdims = parsed;
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    if invalid_axis {
        return Ok(None);
    }

    let rank = input_shape.len();
    let Some(axes) = axes else {
        if keepdims {
            return Ok(Some(vec!["1".to_string(); rank]));
        }
        return Ok(Some(Vec::new()));
    };

    if axes.is_empty() {
        return Ok(Some(input_shape.clone()));
    }

    let axes = if axes.contains(&isize::MIN) {
        (0..rank).map(|axis| axis as isize).collect::<Vec<_>>()
    } else {
        axes
    };

    let mut normalized = Vec::new();
    for axis in axes {
        let axis = if axis < 0 { rank as isize + axis } else { axis };
        if axis < 0 || axis as usize >= rank {
            return Err(format!(
                "reduction axis {} out of bounds for rank {}",
                axis, rank
            ));
        }
        let axis = axis as usize;
        if normalized.contains(&axis) {
            return Err(format!("duplicate reduction axis {}", axis));
        }
        normalized.push(axis);
    }
    normalized.sort_unstable();

    if keepdims {
        let mut output = input_shape.clone();
        for axis in normalized {
            output[axis] = "1".to_string();
        }
        return Ok(Some(output));
    }

    let output = input_shape
        .iter()
        .enumerate()
        .filter(|(idx, _)| !normalized.contains(idx))
        .map(|(_, dim)| dim.clone())
        .collect();

    Ok(Some(output))
}

fn apply_known_linalg_inv(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    if input_shape.len() < 2 {
        return Err(format!(
            "linalg.inv requires rank >= 2, got rank {}",
            input_shape.len()
        ));
    }

    let last = &input_shape[input_shape.len() - 1];
    let second_last = &input_shape[input_shape.len() - 2];
    if last != second_last {
        return Err(format!(
            "linalg.inv requires last two dimensions to match, got {} and {}",
            second_last, last
        ));
    }

    Ok(Some(input_shape.clone()))
}


#[cfg(test)]
mod known_function_shape_rule_tests {
    use super::*;

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn pos(value: &str) -> CallArgument {
        CallArgument::Positional {
            value: value.to_string(),
        }
    }

    fn kw(name: &str, value: &str) -> CallArgument {
        CallArgument::Keyword {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    #[test]
    fn test_concatenate_axis_0_numeric_dims() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["2", "features"])),
            ("b".to_string(), shape(&["3", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["5", "features"])));
    }

    #[test]
    fn test_concatenate_axis_1_numeric_dims() {
        let args = vec![pos("[a, b]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
    }

    #[test]
    fn test_concatenate_negative_axis() {
        let args = vec![pos("[a, b]"), kw("axis", "-1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
    }

    #[test]
    fn test_concatenate_torch_dim_keyword() {
        let args = vec![pos("[a, b]"), kw("dim", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "m"])),
            ("b".to_string(), shape(&["batch", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "m+n"])));
    }

    #[test]
    fn test_concatenate_arrays_keyword() {
        let args = vec![kw("arrays", "[a, b]"), kw("axis", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_tensors_keyword() {
        let args = vec![kw("tensors", "[a, b]"), kw("dim", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_tuple_inputs() {
        let args = vec![pos("(a, b)")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_single_input_returns_same_shape() {
        let args = vec![pos("[a]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_concatenate_missing_input_returns_none() {
        let args = vec![pos("[a, missing]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_concatenate_no_args_returns_none() {
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Concatenate, &[], &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_concatenate_rank_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["features"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("rank"));
    }

    #[test]
    fn test_concatenate_non_axis_dim_mismatch_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["other", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
        assert!(error.contains("expected batch"));
        assert!(error.contains("got other"));
    }

    #[test]
    fn test_concatenate_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "2")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_concatenate_negative_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "-3")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_concatenate_scalar_input_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([("a".to_string(), Vec::new()), ("b".to_string(), Vec::new())]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("rank >= 1"));
    }

    #[test]
    fn test_stack_default_axis() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch", "features"])));
    }

    #[test]
    fn test_stack_axis_1() {
        let args = vec![pos("[a, b, c]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
            ("c".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "3", "features"])));
    }

    #[test]
    fn test_stack_negative_axis() {
        let args = vec![pos("[a, b]"), kw("axis", "-1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features", "2"])));
    }

    #[test]
    fn test_stack_torch_dim_keyword() {
        let args = vec![pos("[a, b]"), kw("dim", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "2", "features"])));
    }

    #[test]
    fn test_stack_arys_keyword() {
        let args = vec![kw("arys", "[a, b]"), kw("axis", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch"])));
    }

    #[test]
    fn test_stack_tuple_inputs() {
        let args = vec![pos("(a, b)")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch"])));
    }

    #[test]
    fn test_stack_missing_input_returns_none() {
        let args = vec![pos("[a, missing]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch"]))]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_stack_no_args_returns_none() {
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Stack, &[], &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_stack_rank_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("rank"));
    }

    #[test]
    fn test_stack_dim_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "other"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
        assert!(error.contains("expected features"));
        assert!(error.contains("got other"));
    }

    #[test]
    fn test_stack_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "3")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_reshape_positional_shape() {
        let args = vec![pos("x"), pos("(2, 6)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_keyword_shape() {
        let args = vec![pos("x"), kw("shape", "[2, 6]")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_infers_minus_one() {
        let args = vec![pos("x"), pos("(2, -1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_symbolic_shape_allowed_without_size_check() {
        let args = vec![pos("x"), pos("(batch, features)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_reshape_size_mismatch_errors() {
        let args = vec![pos("x"), pos("(5, 5)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let error = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap_err();

        assert!(error.contains("changes total size"));
    }

    #[test]
    fn test_reshape_multiple_minus_one_errors() {
        let args = vec![pos("x"), pos("(-1, -1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let error = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap_err();

        assert!(error.contains("only infer one -1"));
    }

    #[test]
    fn test_flatten_numeric_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Flatten, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["24"])));
    }

    #[test]
    fn test_flatten_symbolic_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Flatten, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch*features"])));
    }

    #[test]
    fn test_ravel_uses_flatten_rule() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);

        let output = apply_known_function(&KnownFunction::Ravel, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["6"])));
    }

    #[test]
    fn test_transpose_default_reverses_axes() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_transpose_axes_keyword() {
        let args = vec![pos("x"), kw("axes", "(1, 0, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_permute_dims_keyword() {
        let args = vec![pos("x"), kw("dims", "(2, 0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Permute, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "a", "b"])));
    }

    #[test]
    fn test_transpose_negative_axes() {
        let args = vec![pos("x"), kw("axes", "(-1, -2, -3)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_transpose_wrong_number_of_axes_errors() {
        let args = vec![pos("x"), kw("axes", "(0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap_err();

        assert!(error.contains("expected 3 axes"));
    }

    #[test]
    fn test_transpose_duplicate_axes_errors() {
        let args = vec![pos("x"), kw("axes", "(0, 0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap_err();

        assert!(error.contains("duplicate"));
    }

    #[test]
    fn test_swapaxes_positional() {
        let args = vec![pos("x"), pos("0"), pos("2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_swapaxes_torch_dim_keywords() {
        let args = vec![pos("x"), kw("dim0", "0"), kw("dim1", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_swapaxes_missing_axis_returns_none() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_moveaxis_single_axis() {
        let args = vec![pos("x"), pos("0"), pos("2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "c", "a"])));
    }

    #[test]
    fn test_moveaxis_keyword_axes() {
        let args = vec![pos("x"), kw("source", "0"), kw("destination", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_moveaxis_length_mismatch_errors() {
        let args = vec![pos("x"), kw("source", "(0, 1)"), kw("destination", "2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap_err();

        assert!(error.contains("lengths differ"));
    }

    #[test]
    fn test_expand_dims_axis_0() {
        let args = vec![pos("x"), kw("axis", "0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ExpandDims, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "batch", "features"])));
    }

    #[test]
    fn test_expand_dims_negative_axis() {
        let args = vec![pos("x"), kw("axis", "-1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ExpandDims, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features", "1"])));
    }

    #[test]
    fn test_expand_dims_missing_axis_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ExpandDims, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_squeeze_all_unit_dims() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["1", "batch", "1", "features"]))]);

        let output = apply_known_function(&KnownFunction::Squeeze, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let args = vec![pos("x"), kw("axis", "0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["1", "batch", "1"]))]);

        let output = apply_known_function(&KnownFunction::Squeeze, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "1"])));
    }

    #[test]
    fn test_squeeze_non_unit_axis_errors() {
        let args = vec![pos("x"), kw("axis", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["1", "batch", "1"]))]);

        let error = apply_known_function(&KnownFunction::Squeeze, &args, &shapes).unwrap_err();

        assert!(error.contains("cannot squeeze"));
    }

    #[test]
    fn test_atleast_1d_scalar() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), Vec::new())]);

        let output = apply_known_function(&KnownFunction::AtLeast1D, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1"])));
    }

    #[test]
    fn test_atleast_2d_vector() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["features"]))]);

        let output = apply_known_function(&KnownFunction::AtLeast2D, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "features"])));
    }

    #[test]
    fn test_atleast_3d_matrix() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::AtLeast3D, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "batch", "features"])));
    }

    #[test]
    fn test_atleast_preserves_sufficient_rank() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::AtLeast2D, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["a", "b", "c"])));
    }

    #[test]
    fn test_matmul_vector_vector_scalar() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["k"])),
            ("b".to_string(), shape(&["k"])),
        ]);

        let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_matmul_matrix_matrix() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_matmul_batched_matrix_matrix() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "m", "k"])),
            ("b".to_string(), shape(&["batch", "k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "m", "n"])));
    }

    #[test]
    fn test_matmul_broadcasts_batch_dims() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "m", "k"])),
            ("b".to_string(), shape(&["1", "k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "m", "n"])));
    }

    #[test]
    fn test_matmul_inner_dim_mismatch_errors() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["other", "n"])),
        ]);

        let error = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
    }

    #[test]
    fn test_dot_matrix_vector() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["k"])),
        ]);

        let output = apply_known_function(&KnownFunction::Dot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m"])));
    }

    #[test]
    fn test_dot_matrix_matrix() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Dot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_dot_high_rank() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["b1", "m", "k"])),
            ("b".to_string(), shape(&["b2", "k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Dot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b1", "m", "b2", "n"])));
    }

    #[test]
    fn test_tensordot_default_axes_two() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "n", "k1", "k2"])),
            ("b".to_string(), shape(&["k1", "k2", "p", "q"])),
        ]);

        let output = apply_known_function(&KnownFunction::TensorDot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n", "p", "q"])));
    }

    #[test]
    fn test_tensordot_int_axes_one() {
        let args = vec![pos("a"), pos("b"), pos("1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["k", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::TensorDot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_tensordot_axes_keyword() {
        let args = vec![pos("a"), pos("b"), kw("axes", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["features", "out"])),
        ]);

        let output = apply_known_function(&KnownFunction::TensorDot, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "out"])));
    }

    #[test]
    fn test_tensordot_contraction_mismatch_errors() {
        let args = vec![pos("a"), pos("b"), pos("1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["other", "n"])),
        ]);

        let error = apply_known_function(&KnownFunction::TensorDot, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
    }

    #[test]
    fn test_outer_1d_1d() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m"])),
            ("b".to_string(), shape(&["n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Outer, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_outer_higher_rank_flattens() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["2", "3"])),
            ("b".to_string(), shape(&["4", "5"])),
        ]);

        let output = apply_known_function(&KnownFunction::Outer, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["6", "20"])));
    }

    #[test]
    fn test_inner_2d_2d_match() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["n", "k"])),
        ]);

        let output = apply_known_function(&KnownFunction::Inner, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_inner_last_dim_mismatch_errors() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "k"])),
            ("b".to_string(), shape(&["n", "other"])),
        ]);

        let error = apply_known_function(&KnownFunction::Inner, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
    }

    #[test]
    fn test_vdot_returns_scalar() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["2", "3"])),
            ("b".to_string(), shape(&["6"])),
        ]);

        let output = apply_known_function(&KnownFunction::Vdot, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_zeros_shape_constructor() {
        let args = vec![pos("(batch, features)")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Zeros, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_ones_shape_keyword() {
        let args = vec![kw("shape", "(2, 3)")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Ones, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "3"])));
    }

    #[test]
    fn test_arange_single_stop() {
        let args = vec![pos("10")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Arange, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["10"])));
    }

    #[test]
    fn test_arange_start_stop() {
        let args = vec![pos("2"), pos("10")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Arange, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["8"])));
    }

    #[test]
    fn test_eye_square() {
        let args = vec![pos("n")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Eye, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["n", "n"])));
    }

    #[test]
    fn test_eye_rectangular() {
        let args = vec![pos("n"), pos("m")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Eye, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["n", "m"])));
    }

    #[test]
    fn test_diag_vector_to_matrix() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["n"]))]);

        let output = apply_known_function(&KnownFunction::Diag, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["n", "n"])));
    }

    #[test]
    fn test_diag_matrix_to_vector() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["m", "n"]))]);

        let output = apply_known_function(&KnownFunction::Diag, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["min(m,n)"])));
    }

    #[test]
    fn test_diagonal_batch_matrix() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "m", "n"]))]);

        let output = apply_known_function(&KnownFunction::Diagonal, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "min(m,n)"])));
    }

    #[test]
    fn test_trace_batch_matrix() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "m", "n"]))]);

        let output = apply_known_function(&KnownFunction::Trace, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_take_without_axis_uses_indices_shape() {
        let args = vec![pos("x"), pos("idx")];
        let shapes = HashMap::from([
            ("x".to_string(), shape(&["batch", "features"])),
            ("idx".to_string(), shape(&["k"])),
        ]);

        let output = apply_known_function(&KnownFunction::Take, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["k"])));
    }

    #[test]
    fn test_take_with_axis_inserts_indices_shape() {
        let args = vec![pos("x"), pos("idx"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("x".to_string(), shape(&["batch", "features"])),
            ("idx".to_string(), shape(&["k"])),
        ]);

        let output = apply_known_function(&KnownFunction::Take, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "k"])));
    }

    #[test]
    fn test_pad_single_width_all_axes() {
        let args = vec![pos("x"), pos("1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["10", "20"]))]);

        let output = apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["12", "22"])));
    }

    #[test]
    fn test_pad_before_after_all_axes() {
        let args = vec![pos("x"), pos("(1, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["h", "w"]))]);

        let output = apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["h+3", "w+3"])));
    }

    #[test]
    fn test_pad_per_axis_widths() {
        let args = vec![pos("x"), pos("((1, 2), (3, 4))")];
        let shapes = HashMap::from([("x".to_string(), shape(&["10", "20"]))]);

        let output = apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["13", "27"])));
    }

    #[test]
    fn test_pad_symbolic_per_axis_widths() {
        let args = vec![pos("x"), kw("pad_width", "((1, 2), (3, 4))")];
        let shapes = HashMap::from([("x".to_string(), shape(&["h", "w"]))]);

        let output = apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["h+3", "w+7"])));
    }

    #[test]
    fn test_pad_dynamic_width_returns_none() {
        let args = vec![pos("x"), pos("pad_width")];
        let shapes = HashMap::from([("x".to_string(), shape(&["h", "w"]))]);

        let output = apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_rot90_default_swaps_first_two_axes() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["h", "w", "c"]))]);

        let output = apply_known_function(&KnownFunction::Rot90, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["w", "h", "c"])));
    }

    #[test]
    fn test_rot90_even_k_preserves_shape() {
        let args = vec![pos("x"), kw("k", "2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["h", "w", "c"]))]);

        let output = apply_known_function(&KnownFunction::Rot90, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["h", "w", "c"])));
    }

    #[test]
    fn test_rot90_custom_axes() {
        let args = vec![pos("x"), kw("axes", "(1, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Rot90, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["a", "c", "b"])));
    }

    #[test]
    fn test_vstack_vectors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["features"])),
            ("b".to_string(), shape(&["features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Vstack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "features"])));
    }

    #[test]
    fn test_vstack_matrices() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Vstack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_hstack_vectors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m"])),
            ("b".to_string(), shape(&["n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Hstack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n"])));
    }

    #[test]
    fn test_hstack_matrices() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "m"])),
            ("b".to_string(), shape(&["batch", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Hstack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "m+n"])));
    }

    #[test]
    fn test_dstack_vectors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["features"])),
            ("b".to_string(), shape(&["features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Dstack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "features", "2"])));
    }

    #[test]
    fn test_column_stack_vectors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["rows"])),
            ("b".to_string(), shape(&["rows"])),
        ]);

        let output = apply_known_function(&KnownFunction::ColumnStack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["rows", "2"])));
    }

    #[test]
    fn test_broadcast_to_shape() {
        let args = vec![pos("x"), kw("shape", "(batch, features)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["features"]))]);

        let output = apply_known_function(&KnownFunction::BroadcastTo, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_broadcast_arrays_two_inputs() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "1"])),
            ("b".to_string(), shape(&["1", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::BroadcastArrays, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_broadcast_arrays_mismatch_errors() {
        let args = vec![pos("a"), pos("b")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "x"])),
            ("b".to_string(), shape(&["batch", "y"])),
        ]);

        let error =
            apply_known_function(&KnownFunction::BroadcastArrays, &args, &shapes).unwrap_err();

        assert!(error.contains("cannot broadcast"));
    }

    #[test]
    fn test_tile_repeats_dims() {
        let args = vec![pos("x"), pos("(2, 3)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Tile, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch*2", "features*3"])));
    }

    #[test]
    fn test_tile_numeric_dims() {
        let args = vec![pos("x"), pos("(2, 3)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["4", "5"]))]);

        let output = apply_known_function(&KnownFunction::Tile, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["8", "15"])));
    }

    #[test]
    fn test_repeat_axis() {
        let args = vec![pos("x"), kw("repeats", "3"), kw("axis", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Repeat, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features*3"])));
    }

    #[test]
    fn test_repeat_without_axis_flattens() {
        let args = vec![pos("x"), pos("2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);

        let output = apply_known_function(&KnownFunction::Repeat, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["12"])));
    }

    #[test]
    fn test_shape_preserving_roll() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Roll, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    // ── *_like shape-preserving tests ──

    #[test]
    fn test_zeros_like_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output =
            apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_ones_like_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output =
            apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_full_like_preserves_shape() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output =
            apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_empty_like_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output =
            apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_zeros_like_missing_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output =
            apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_ones_like_missing_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output =
            apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_full_like_missing_input_returns_none() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::new();

        let output =
            apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_empty_like_missing_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output =
            apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_zeros_like_keyword_x() {
        let args = vec![kw("x", "arr")];
        let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

        let output =
            apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_ones_like_keyword_input() {
        let args = vec![kw("input", "arr")];
        let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

        let output =
            apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_full_like_keyword_x() {
        let args = vec![kw("x", "arr"), pos("0")];
        let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

        let output =
            apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_empty_like_keyword_input() {
        let args = vec![kw("input", "arr")];
        let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

        let output =
            apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "n"])));
    }

    #[test]
    fn test_zeros_like_no_args_returns_none() {
        let shapes = HashMap::new();

        let output =
            apply_known_function(&KnownFunction::ZerosLike, &[], &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_full_like_unrecognized_keyword_returns_none() {
        let args = vec![kw("template", "arr")];
        let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

        let output =
            apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

        // 'template' is not a recognized keyword for first_array_arg
        assert_eq!(output, None);
    }

    #[test]
    fn test_where_broadcasts_three_inputs() {
        let args = vec![pos("cond"), pos("x"), pos("y")];
        let shapes = HashMap::from([
            ("cond".to_string(), shape(&["batch", "1"])),
            ("x".to_string(), shape(&["batch", "features"])),
            ("y".to_string(), shape(&["1", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Where, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_where_broadcast_mismatch_errors() {
        let args = vec![pos("cond"), pos("x"), pos("y")];
        let shapes = HashMap::from([
            ("cond".to_string(), shape(&["batch", "1"])),
            ("x".to_string(), shape(&["batch", "x"])),
            ("y".to_string(), shape(&["batch", "y"])),
        ]);

        let error = apply_known_function(&KnownFunction::Where, &args, &shapes).unwrap_err();

        assert!(error.contains("cannot broadcast"));
    }

    #[test]
    fn test_reduction_no_axis_returns_scalar_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_reduction_no_axis_keepdims() {
        let args = vec![pos("x"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Mean, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "1"])));
    }

    #[test]
    fn test_reduction_axis_keyword_removes_axis() {
        let args = vec![pos("x"), kw("axis", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features", "hidden"]))]);

        let output = apply_known_function(&KnownFunction::Max, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_reduction_dim_keyword_removes_axis() {
        let args = vec![pos("x"), kw("dim", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features", "hidden"]))]);

        let output = apply_known_function(&KnownFunction::Min, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_reduction_positional_axis() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Prod, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["features"])));
    }

    #[test]
    fn test_reduction_negative_axis() {
        let args = vec![pos("x"), kw("axis", "-1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Std, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_reduction_multiple_axes_tuple() {
        let args = vec![pos("x"), kw("axis", "(0, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "time", "features"]))]);

        let output = apply_known_function(&KnownFunction::Var, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["time"])));
    }

    #[test]
    fn test_reduction_multiple_axes_keepdims() {
        let args = vec![pos("x"), kw("axis", "(0, 2)"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "time", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "time", "1"])));
    }

    #[test]
    fn test_reduction_axis_none_reduces_all_axes() {
        let args = vec![pos("x"), kw("axis", "None")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_reduction_missing_input_returns_none() {
        let args = vec![pos("x"), kw("axis", "0")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_reduction_axis_out_of_bounds_errors() {
        let args = vec![pos("x"), kw("axis", "2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_reduction_duplicate_axes_errors() {
        let args = vec![pos("x"), kw("axis", "(1, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap_err();

        assert!(error.contains("duplicate"));
    }

    #[test]
    fn test_reshape_missing_input_returns_none() {
        let args = vec![pos("x"), pos("(2, 6)")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_reshape_missing_shape_arg_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "6"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_reshape_symbolic_minus_one_errors() {
        let args = vec![pos("x"), pos("(batch, -1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap_err();

        assert!(error.contains("symbolic input"));
    }

    #[test]
    fn test_flatten_scalar_shape_returns_one_dim_one() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), Vec::new())]);

        let output = apply_known_function(&KnownFunction::Flatten, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1"])));
    }

    #[test]
    fn test_reduction_invalid_dynamic_axis_returns_none() {
        let args = vec![pos("x"), kw("axis", "axis")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_reduction_empty_axis_tuple_preserves_shape() {
        let args = vec![pos("x"), kw("axis", "()")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_reduction_axis_none_keepdims_all_ones() {
        let args = vec![pos("x"), kw("axis", "None"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "1"])));
    }

    #[test]
    fn test_transpose_missing_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_transpose_axis_out_of_bounds_errors() {
        let args = vec![pos("x"), kw("axes", "(0, 1, 3)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_swapaxes_negative_axes() {
        let args = vec![pos("x"), pos("-1"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_swapaxes_axis_out_of_bounds_errors() {
        let args = vec![pos("x"), pos("0"), pos("3")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_moveaxis_missing_destination_returns_none() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_non_implemented_known_function_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Tile, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    // ── Reduction shape-rule tests for All / Any / ArgMax / ArgMin ──

    #[test]
    fn test_all_axis_1_reduces_axis() {
        let args = vec![pos("x"), kw("axis", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::All, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_any_axis_0_keepdims_true() {
        let args = vec![pos("x"), kw("axis", "0"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Any, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "features"])));
    }

    #[test]
    fn test_argmax_axis_negative_removes_last_axis() {
        let args = vec![pos("x"), kw("axis", "-1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ArgMax, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_argmin_no_axis_reduces_all() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ArgMin, &args, &shapes).unwrap();

        // No axis = reduce all axes → scalar shape
        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_all_unknown_input_returns_none() {
        let args = vec![pos("x"), kw("axis", "1")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::All, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_any_invalid_axis_errors() {
        let args = vec![pos("x"), kw("axis", "3")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Any, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    // ── Shape-preserving tests for Argsort / Sort / Cumsum / Cumprod ──

    #[test]
    fn test_argsort_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Argsort, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_sort_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sort, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_cumsum_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Cumsum, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_cumprod_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Cumprod, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_argsort_unknown_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Argsort, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_argmax_dim_keyword_torch_style() {
        let args = vec![pos("x"), kw("dim", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::ArgMax, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_all_axis_none_reduces_all() {
        let args = vec![pos("x"), kw("axis", "None")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::All, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_any_keepdim_keyword_torch_style() {
        let args = vec![pos("x"), kw("dim", "1"), kw("keepdim", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Any, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "1"])));
    }

    // ── linalg.inv shape rule tests ──

    #[test]
    fn test_linalg_inv_square_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["n", "n"]))]);

        let output = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["n", "n"])));
    }

    #[test]
    fn test_linalg_inv_batched_preserves_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "n", "n"]))]);

        let output = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "n", "n"])));
    }

    #[test]
    fn test_linalg_inv_symbolic_square_dims_preserve() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["m", "m"]))]);

        let output = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m", "m"])));
    }

    #[test]
    fn test_linalg_inv_non_square_errors() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["m", "n"]))]);

        let error = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap_err();

        assert!(error.contains("last two dimensions to match"));
        assert!(error.contains("m"));
        assert!(error.contains("n"));
    }

    #[test]
    fn test_linalg_inv_rank_1_errors() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["n"]))]);

        let error = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap_err();

        assert!(error.contains("rank >= 2"));
    }

    #[test]
    fn test_linalg_inv_unknown_input_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::LinalgInv, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }
}

#[cfg(test)]
mod known_function_tests {
    use super::*;
    use tree_sitter::Parser;

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    macro_rules! known_case {
        ($name:ident, [$($part:expr),*], $kind:expr) => {
            #[test]
            fn $name() {
                assert_eq!(classify_known_function(&target(&[$($part),*])), $kind);
            }
        };
    }

    known_case!(
        jnp_concatenate,
        ["jax", "numpy", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        jnp_concat,
        ["jax", "numpy", "concat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        jnp_stack,
        ["jax", "numpy", "stack"],
        Some(KnownFunction::Stack)
    );
    known_case!(
        jnp_reshape,
        ["jax", "numpy", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        jnp_transpose,
        ["jax", "numpy", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        jnp_expand_dims,
        ["jax", "numpy", "expand_dims"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        jnp_squeeze,
        ["jax", "numpy", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(jnp_sum, ["jax", "numpy", "sum"], Some(KnownFunction::Sum));
    known_case!(
        jnp_mean,
        ["jax", "numpy", "mean"],
        Some(KnownFunction::Mean)
    );
    known_case!(jnp_max, ["jax", "numpy", "max"], Some(KnownFunction::Max));
    known_case!(jnp_amax, ["jax", "numpy", "amax"], Some(KnownFunction::Max));
    known_case!(jnp_min, ["jax", "numpy", "min"], Some(KnownFunction::Min));
    known_case!(jnp_amin, ["jax", "numpy", "amin"], Some(KnownFunction::Min));
    known_case!(
        jnp_prod,
        ["jax", "numpy", "prod"],
        Some(KnownFunction::Prod)
    );
    known_case!(jnp_std, ["jax", "numpy", "std"], Some(KnownFunction::Std));
    known_case!(jnp_var, ["jax", "numpy", "var"], Some(KnownFunction::Var));
    known_case!(
        jnp_matmul,
        ["jax", "numpy", "matmul"],
        Some(KnownFunction::Matmul)
    );
    known_case!(jnp_dot, ["jax", "numpy", "dot"], Some(KnownFunction::Dot));
    known_case!(
        jnp_einsum,
        ["jax", "numpy", "einsum"],
        Some(KnownFunction::Einsum)
    );
    known_case!(
        jnp_split,
        ["jax", "numpy", "split"],
        Some(KnownFunction::Split)
    );
    known_case!(
        jnp_tile,
        ["jax", "numpy", "tile"],
        Some(KnownFunction::Tile)
    );
    known_case!(
        jnp_repeat,
        ["jax", "numpy", "repeat"],
        Some(KnownFunction::Repeat)
    );
    known_case!(
        jnp_flatten,
        ["jax", "numpy", "flatten"],
        Some(KnownFunction::Flatten)
    );
    known_case!(
        jnp_ravel,
        ["jax", "numpy", "ravel"],
        Some(KnownFunction::Ravel)
    );
    known_case!(
        jnp_moveaxis,
        ["jax", "numpy", "moveaxis"],
        Some(KnownFunction::MoveAxis)
    );
    known_case!(
        jnp_swapaxes,
        ["jax", "numpy", "swapaxes"],
        Some(KnownFunction::SwapAxes)
    );
    known_case!(
        jnp_where,
        ["jax", "numpy", "where"],
        Some(KnownFunction::Where)
    );
    known_case!(
        jnp_zeros,
        ["jax", "numpy", "zeros"],
        Some(KnownFunction::Zeros)
    );
    known_case!(
        jnp_ones,
        ["jax", "numpy", "ones"],
        Some(KnownFunction::Ones)
    );
    known_case!(
        jnp_full,
        ["jax", "numpy", "full"],
        Some(KnownFunction::Full)
    );
    known_case!(
        jnp_arange,
        ["jax", "numpy", "arange"],
        Some(KnownFunction::Arange)
    );
    known_case!(jnp_eye, ["jax", "numpy", "eye"], Some(KnownFunction::Eye));
    known_case!(
        jnp_broadcast_to,
        ["jax", "numpy", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        jnp_broadcast_arrays,
        ["jax", "numpy", "broadcast_arrays"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        jnp_atleast_1d,
        ["jax", "numpy", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        jnp_atleast_2d,
        ["jax", "numpy", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        jnp_atleast_3d,
        ["jax", "numpy", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(jnp_pad, ["jax", "numpy", "pad"], Some(KnownFunction::Pad));
    known_case!(
        jnp_roll,
        ["jax", "numpy", "roll"],
        Some(KnownFunction::Roll)
    );
    known_case!(
        jnp_flip,
        ["jax", "numpy", "flip"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_fliplr,
        ["jax", "numpy", "fliplr"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_flipud,
        ["jax", "numpy", "flipud"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_rot90,
        ["jax", "numpy", "rot90"],
        Some(KnownFunction::Rot90)
    );
    known_case!(
        jnp_take,
        ["jax", "numpy", "take"],
        Some(KnownFunction::Take)
    );
    known_case!(
        jnp_diag,
        ["jax", "numpy", "diag"],
        Some(KnownFunction::Diag)
    );
    known_case!(
        jnp_diagonal,
        ["jax", "numpy", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(
        jnp_trace,
        ["jax", "numpy", "trace"],
        Some(KnownFunction::Trace)
    );
    known_case!(
        jnp_triu,
        ["jax", "numpy", "triu"],
        Some(KnownFunction::Triu)
    );
    known_case!(
        jnp_tril,
        ["jax", "numpy", "tril"],
        Some(KnownFunction::Tril)
    );
    known_case!(
        jnp_meshgrid,
        ["jax", "numpy", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(
        jnp_vstack,
        ["jax", "numpy", "vstack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        jnp_row_stack,
        ["jax", "numpy", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        jnp_hstack,
        ["jax", "numpy", "hstack"],
        Some(KnownFunction::Hstack)
    );
    known_case!(
        jnp_dstack,
        ["jax", "numpy", "dstack"],
        Some(KnownFunction::Dstack)
    );
    known_case!(
        jnp_column_stack,
        ["jax", "numpy", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(
        jnp_block,
        ["jax", "numpy", "block"],
        Some(KnownFunction::Block)
    );
    known_case!(
        jnp_zeros_like,
        ["jax", "numpy", "zeros_like"],
        Some(KnownFunction::ZerosLike)
    );
    known_case!(
        jnp_ones_like,
        ["jax", "numpy", "ones_like"],
        Some(KnownFunction::OnesLike)
    );
    known_case!(
        jnp_full_like,
        ["jax", "numpy", "full_like"],
        Some(KnownFunction::FullLike)
    );
    known_case!(
        jnp_empty_like,
        ["jax", "numpy", "empty_like"],
        Some(KnownFunction::EmptyLike)
    );
    known_case!(
        jnp_array,
        ["jax", "numpy", "array"],
        Some(KnownFunction::Array)
    );
    known_case!(
        jnp_asarray,
        ["jax", "numpy", "asarray"],
        Some(KnownFunction::AsArray)
    );

    known_case!(
        np_concatenate,
        ["numpy", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(np_stack, ["numpy", "stack"], Some(KnownFunction::Stack));
    known_case!(
        np_reshape,
        ["numpy", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        np_transpose,
        ["numpy", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        np_expand_dims,
        ["numpy", "expand_dims"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        np_squeeze,
        ["numpy", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(np_sum, ["numpy", "sum"], Some(KnownFunction::Sum));
    known_case!(np_mean, ["numpy", "mean"], Some(KnownFunction::Mean));
    known_case!(np_amax, ["numpy", "amax"], Some(KnownFunction::Max));
    known_case!(np_amin, ["numpy", "amin"], Some(KnownFunction::Min));
    known_case!(np_prod, ["numpy", "prod"], Some(KnownFunction::Prod));
    known_case!(np_std, ["numpy", "std"], Some(KnownFunction::Std));
    known_case!(np_var, ["numpy", "var"], Some(KnownFunction::Var));
    known_case!(np_matmul, ["numpy", "matmul"], Some(KnownFunction::Matmul));
    known_case!(np_dot, ["numpy", "dot"], Some(KnownFunction::Dot));
    known_case!(np_einsum, ["numpy", "einsum"], Some(KnownFunction::Einsum));
    known_case!(np_split, ["numpy", "split"], Some(KnownFunction::Split));
    known_case!(np_tile, ["numpy", "tile"], Some(KnownFunction::Tile));
    known_case!(np_repeat, ["numpy", "repeat"], Some(KnownFunction::Repeat));
    known_case!(np_ravel, ["numpy", "ravel"], Some(KnownFunction::Ravel));
    known_case!(
        np_moveaxis,
        ["numpy", "moveaxis"],
        Some(KnownFunction::MoveAxis)
    );
    known_case!(
        np_swapaxes,
        ["numpy", "swapaxes"],
        Some(KnownFunction::SwapAxes)
    );
    known_case!(np_where, ["numpy", "where"], Some(KnownFunction::Where));
    known_case!(np_zeros, ["numpy", "zeros"], Some(KnownFunction::Zeros));
    known_case!(np_ones, ["numpy", "ones"], Some(KnownFunction::Ones));
    known_case!(np_full, ["numpy", "full"], Some(KnownFunction::Full));
    known_case!(np_arange, ["numpy", "arange"], Some(KnownFunction::Arange));
    known_case!(np_eye, ["numpy", "eye"], Some(KnownFunction::Eye));
    known_case!(
        np_broadcast_to,
        ["numpy", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        np_broadcast_arrays,
        ["numpy", "broadcast_arrays"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        np_atleast_1d,
        ["numpy", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        np_atleast_2d,
        ["numpy", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        np_atleast_3d,
        ["numpy", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(np_pad, ["numpy", "pad"], Some(KnownFunction::Pad));
    known_case!(np_roll, ["numpy", "roll"], Some(KnownFunction::Roll));
    known_case!(np_flip, ["numpy", "flip"], Some(KnownFunction::Flip));
    known_case!(np_fliplr, ["numpy", "fliplr"], Some(KnownFunction::Flip));
    known_case!(np_flipud, ["numpy", "flipud"], Some(KnownFunction::Flip));
    known_case!(np_rot90, ["numpy", "rot90"], Some(KnownFunction::Rot90));
    known_case!(np_take, ["numpy", "take"], Some(KnownFunction::Take));
    known_case!(np_diag, ["numpy", "diag"], Some(KnownFunction::Diag));
    known_case!(
        np_diagonal,
        ["numpy", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(np_trace, ["numpy", "trace"], Some(KnownFunction::Trace));
    known_case!(np_triu, ["numpy", "triu"], Some(KnownFunction::Triu));
    known_case!(np_tril, ["numpy", "tril"], Some(KnownFunction::Tril));
    known_case!(
        np_meshgrid,
        ["numpy", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(np_vstack, ["numpy", "vstack"], Some(KnownFunction::Vstack));
    known_case!(
        np_row_stack,
        ["numpy", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(np_hstack, ["numpy", "hstack"], Some(KnownFunction::Hstack));
    known_case!(np_dstack, ["numpy", "dstack"], Some(KnownFunction::Dstack));
    known_case!(
        np_column_stack,
        ["numpy", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(np_block, ["numpy", "block"], Some(KnownFunction::Block));
    known_case!(
        np_zeros_like,
        ["numpy", "zeros_like"],
        Some(KnownFunction::ZerosLike)
    );
    known_case!(
        np_ones_like,
        ["numpy", "ones_like"],
        Some(KnownFunction::OnesLike)
    );
    known_case!(
        np_full_like,
        ["numpy", "full_like"],
        Some(KnownFunction::FullLike)
    );
    known_case!(
        np_empty_like,
        ["numpy", "empty_like"],
        Some(KnownFunction::EmptyLike)
    );
    known_case!(np_array, ["numpy", "array"], Some(KnownFunction::Array));
    known_case!(
        np_asarray,
        ["numpy", "asarray"],
        Some(KnownFunction::AsArray)
    );

    known_case!(
        torch_cat,
        ["torch", "cat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        torch_concat,
        ["torch", "concat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        torch_concatenate,
        ["torch", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(torch_stack, ["torch", "stack"], Some(KnownFunction::Stack));
    known_case!(
        torch_reshape,
        ["torch", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        torch_transpose,
        ["torch", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        torch_unsqueeze,
        ["torch", "unsqueeze"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        torch_squeeze,
        ["torch", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(torch_sum, ["torch", "sum"], Some(KnownFunction::Sum));
    known_case!(torch_mean, ["torch", "mean"], Some(KnownFunction::Mean));
    known_case!(torch_max, ["torch", "max"], Some(KnownFunction::Max));
    known_case!(torch_min, ["torch", "min"], Some(KnownFunction::Min));
    known_case!(torch_prod, ["torch", "prod"], Some(KnownFunction::Prod));
    known_case!(torch_std, ["torch", "std"], Some(KnownFunction::Std));
    known_case!(torch_var, ["torch", "var"], Some(KnownFunction::Var));
    known_case!(
        torch_matmul,
        ["torch", "matmul"],
        Some(KnownFunction::Matmul)
    );
    known_case!(torch_dot, ["torch", "dot"], Some(KnownFunction::Dot));
    known_case!(
        torch_einsum,
        ["torch", "einsum"],
        Some(KnownFunction::Einsum)
    );
    known_case!(torch_split, ["torch", "split"], Some(KnownFunction::Split));
    known_case!(torch_tile, ["torch", "tile"], Some(KnownFunction::Tile));
    known_case!(
        torch_repeat,
        ["torch", "repeat"],
        Some(KnownFunction::Repeat)
    );
    known_case!(
        torch_flatten,
        ["torch", "flatten"],
        Some(KnownFunction::Flatten)
    );
    known_case!(torch_ravel, ["torch", "ravel"], Some(KnownFunction::Ravel));
    known_case!(torch_where, ["torch", "where"], Some(KnownFunction::Where));
    known_case!(torch_zeros, ["torch", "zeros"], Some(KnownFunction::Zeros));
    known_case!(torch_ones, ["torch", "ones"], Some(KnownFunction::Ones));
    known_case!(torch_full, ["torch", "full"], Some(KnownFunction::Full));
    known_case!(
        torch_zeros_like,
        ["torch", "zeros_like"],
        Some(KnownFunction::ZerosLike)
    );
    known_case!(
        torch_ones_like,
        ["torch", "ones_like"],
        Some(KnownFunction::OnesLike)
    );
    known_case!(
        torch_full_like,
        ["torch", "full_like"],
        Some(KnownFunction::FullLike)
    );
    known_case!(
        torch_empty_like,
        ["torch", "empty_like"],
        Some(KnownFunction::EmptyLike)
    );
    known_case!(
        torch_arange,
        ["torch", "arange"],
        Some(KnownFunction::Arange)
    );
    known_case!(torch_eye, ["torch", "eye"], Some(KnownFunction::Eye));
    known_case!(
        torch_broadcast_to,
        ["torch", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        torch_broadcast_tensors,
        ["torch", "broadcast_tensors"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        torch_atleast_1d,
        ["torch", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        torch_atleast_2d,
        ["torch", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        torch_atleast_3d,
        ["torch", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(torch_roll, ["torch", "roll"], Some(KnownFunction::Roll));
    known_case!(torch_flip, ["torch", "flip"], Some(KnownFunction::Flip));
    known_case!(torch_fliplr, ["torch", "fliplr"], Some(KnownFunction::Flip));
    known_case!(torch_flipud, ["torch", "flipud"], Some(KnownFunction::Flip));
    known_case!(torch_rot90, ["torch", "rot90"], Some(KnownFunction::Rot90));
    known_case!(torch_take, ["torch", "take"], Some(KnownFunction::Take));
    known_case!(torch_diag, ["torch", "diag"], Some(KnownFunction::Diag));
    known_case!(
        torch_diagonal,
        ["torch", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(torch_trace, ["torch", "trace"], Some(KnownFunction::Trace));
    known_case!(torch_triu, ["torch", "triu"], Some(KnownFunction::Triu));
    known_case!(torch_tril, ["torch", "tril"], Some(KnownFunction::Tril));
    known_case!(
        torch_meshgrid,
        ["torch", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(
        torch_vstack,
        ["torch", "vstack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        torch_row_stack,
        ["torch", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        torch_hstack,
        ["torch", "hstack"],
        Some(KnownFunction::Hstack)
    );
    known_case!(
        torch_dstack,
        ["torch", "dstack"],
        Some(KnownFunction::Dstack)
    );
    known_case!(
        torch_column_stack,
        ["torch", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(
        torch_permute,
        ["torch", "permute"],
        Some(KnownFunction::Permute)
    );
    known_case!(
        torch_tensor,
        ["torch", "tensor"],
        Some(KnownFunction::Array)
    );
    known_case!(
        torch_as_tensor,
        ["torch", "as_tensor"],
        Some(KnownFunction::AsArray)
    );
    known_case!(torch_pad_rejected_for_now, ["torch", "pad"], None);
    known_case!(torch_block_rejected_for_now, ["torch", "block"], None);

    // torch.nn.functional.* — deep module path classification
    known_case!(
        torch_nn_functional_pad,
        ["torch", "nn", "functional", "pad"],
        Some(KnownFunction::Pad)
    );
    known_case!(
        torch_nn_functional_relu_not_classified,
        ["torch", "nn", "functional", "relu"],
        None
    );
    known_case!(
        torch_nn_functional_unknown_not_classified,
        ["torch", "nn", "functional", "unknown_func"],
        None
    );

    known_case!(jax_vmap, ["jax", "vmap"], Some(KnownFunction::Vmap));
    known_case!(jax_lax_dot, ["jax", "lax", "dot"], Some(KnownFunction::Dot));
    known_case!(
        jax_lax_dot_general,
        ["jax", "lax", "dot_general"],
        Some(KnownFunction::Matmul)
    );

    known_case!(unknown_module_rejected, ["foo", "concatenate"], None);
    known_case!(
        deep_unknown_module_rejected,
        ["jax", "random", "split"],
        None
    );
    known_case!(too_short_target_rejected, ["concatenate"], None);
    known_case!(empty_target_rejected, [], None);
    known_case!(
        case_sensitive_function_rejected,
        ["jax", "numpy", "Concatenate"],
        None
    );
    known_case!(
        case_sensitive_module_rejected,
        ["JAX", "numpy", "concatenate"],
        None
    );
    known_case!(method_like_known_function_rejected, ["x", "reshape"], None);
    // ── Classification tests for all/any/argmax/argmin/argsort/sort/cumsum/cumprod ──

    known_case!(jnp_all, ["jax", "numpy", "all"], Some(KnownFunction::All));
    known_case!(np_all, ["numpy", "all"], Some(KnownFunction::All));
    known_case!(torch_all, ["torch", "all"], Some(KnownFunction::All));

    known_case!(jnp_any, ["jax", "numpy", "any"], Some(KnownFunction::Any));
    known_case!(np_any, ["numpy", "any"], Some(KnownFunction::Any));
    known_case!(torch_any, ["torch", "any"], Some(KnownFunction::Any));

    known_case!(jnp_argmax, ["jax", "numpy", "argmax"], Some(KnownFunction::ArgMax));
    known_case!(np_argmax, ["numpy", "argmax"], Some(KnownFunction::ArgMax));
    known_case!(torch_argmax, ["torch", "argmax"], Some(KnownFunction::ArgMax));

    known_case!(jnp_argmin, ["jax", "numpy", "argmin"], Some(KnownFunction::ArgMin));
    known_case!(np_argmin, ["numpy", "argmin"], Some(KnownFunction::ArgMin));
    known_case!(torch_argmin, ["torch", "argmin"], Some(KnownFunction::ArgMin));

    known_case!(jnp_argsort, ["jax", "numpy", "argsort"], Some(KnownFunction::Argsort));
    known_case!(np_argsort, ["numpy", "argsort"], Some(KnownFunction::Argsort));
    known_case!(torch_argsort, ["torch", "argsort"], Some(KnownFunction::Argsort));

    known_case!(jnp_sort, ["jax", "numpy", "sort"], Some(KnownFunction::Sort));
    known_case!(np_sort, ["numpy", "sort"], Some(KnownFunction::Sort));
    known_case!(torch_sort, ["torch", "sort"], Some(KnownFunction::Sort));

    known_case!(jnp_cumsum, ["jax", "numpy", "cumsum"], Some(KnownFunction::Cumsum));
    known_case!(np_cumsum, ["numpy", "cumsum"], Some(KnownFunction::Cumsum));
    known_case!(torch_cumsum, ["torch", "cumsum"], Some(KnownFunction::Cumsum));

    known_case!(jnp_cumprod, ["jax", "numpy", "cumprod"], Some(KnownFunction::Cumprod));
    known_case!(np_cumprod, ["numpy", "cumprod"], Some(KnownFunction::Cumprod));
    known_case!(torch_cumprod, ["torch", "cumprod"], Some(KnownFunction::Cumprod));

    known_case!(jax_numpy_vmap_rejected, ["jax", "numpy", "vmap"], None);
    known_case!(numpy_vmap_rejected, ["numpy", "vmap"], None);
    known_case!(torch_vmap_rejected_for_now, ["torch", "vmap"], None);
    known_case!(torch_nn_function_rejected, ["torch", "nn", "Linear"], None);
    known_case!(
        numpy_linalg_dot_rejected_for_now,
        ["numpy", "linalg", "dot"],
        None
    );
    known_case!(
        jax_numpy_linalg_norm_rejected_for_now,
        ["jax", "numpy", "linalg", "norm"],
        None
    );

    // ── linalg.inv classification tests ──

    known_case!(
        jnp_linalg_inv,
        ["jax", "numpy", "linalg", "inv"],
        Some(KnownFunction::LinalgInv)
    );
    known_case!(
        np_linalg_inv,
        ["numpy", "linalg", "inv"],
        Some(KnownFunction::LinalgInv)
    );
    known_case!(
        torch_linalg_inv,
        ["torch", "linalg", "inv"],
        Some(KnownFunction::LinalgInv)
    );
    known_case!(
        jax_linalg_inv_unsupported,
        ["jax", "linalg", "inv"],
        None
    );

    macro_rules! alias_case {
        ($name:ident, $code:expr, $target:expr, $kind:expr) => {
            #[test]
            fn $name() {
                let tree = parse($code);
                let import_map = build_import_map(tree.root_node(), $code).unwrap();
                let resolved = resolve_call_target($target, &import_map);
                assert_eq!(classify_known_function(&resolved), $kind);
            }
        };
    }

    alias_case!(
        alias_jnp_concatenate,
        "import jax.numpy as jnp",
        "jnp.concatenate",
        Some(KnownFunction::Concatenate)
    );
    alias_case!(
        alias_np_stack,
        "import numpy as np",
        "np.stack",
        Some(KnownFunction::Stack)
    );
    alias_case!(
        alias_torch_cat,
        "import torch as th",
        "th.cat",
        Some(KnownFunction::Concatenate)
    );
    alias_case!(
        alias_jax_vmap,
        "import jax",
        "jax.vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        alias_jax_as_jx_vmap,
        "import jax as jx",
        "jx.vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        alias_jax_lax_dot,
        "import jax.lax as lax",
        "lax.dot",
        Some(KnownFunction::Dot)
    );
    alias_case!(
        from_import_jax_vmap,
        "from jax import vmap",
        "vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        from_import_jax_vmap_alias,
        "from jax import vmap as vm",
        "vm",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        from_import_jnp_reshape,
        "from jax.numpy import reshape",
        "reshape",
        Some(KnownFunction::Reshape)
    );
    alias_case!(
        from_import_np_transpose_alias,
        "from numpy import transpose as T",
        "T",
        Some(KnownFunction::Transpose)
    );
    alias_case!(
        from_import_torch_stack_alias,
        "from torch import stack as stk",
        "stk",
        Some(KnownFunction::Stack)
    );
    alias_case!(
        alias_jnp_broadcast_to,
        "import jax.numpy as jnp",
        "jnp.broadcast_to",
        Some(KnownFunction::BroadcastTo)
    );
    alias_case!(
        alias_np_vstack,
        "import numpy as np",
        "np.vstack",
        Some(KnownFunction::Vstack)
    );
    alias_case!(
        alias_torch_permute,
        "import torch as th",
        "th.permute",
        Some(KnownFunction::Permute)
    );
    alias_case!(
        from_import_np_meshgrid,
        "from numpy import meshgrid",
        "meshgrid",
        Some(KnownFunction::Meshgrid)
    );
    alias_case!(
        from_import_jnp_asarray_alias,
        "from jax.numpy import asarray as arr",
        "arr",
        Some(KnownFunction::AsArray)
    );
    alias_case!(
        from_import_torch_as_tensor,
        "from torch import as_tensor",
        "as_tensor",
        Some(KnownFunction::AsArray)
    );
    alias_case!(
        from_import_unknown_rejected,
        "from foo import concatenate",
        "concatenate",
        None
    );
    alias_case!(unimported_known_name_rejected, "", "concatenate", None);
    alias_case!(
        alias_exact_matching_rejected,
        "import numpy as np",
        "npx.concatenate",
        None
    );

    // torch.nn.functional alias/import tests
    alias_case!(
        alias_torch_nn_functional_imported_as_f_pad,
        "import torch.nn.functional as F",
        "F.pad",
        Some(KnownFunction::Pad)
    );
    alias_case!(
        from_import_torch_nn_functional_pad,
        "from torch.nn.functional import pad",
        "pad",
        Some(KnownFunction::Pad)
    );
    alias_case!(
        from_import_torch_nn_functional_pad_alias,
        "from torch.nn.functional import pad as F_pad",
        "F_pad",
        Some(KnownFunction::Pad)
    );
    alias_case!(
        alias_torch_nn_functional_imported_as_f_relu_not_classified,
        "import torch.nn.functional as F",
        "F.relu",
        None
    );
}

#[cfg(test)]
mod method_call_tests {
    use super::*;

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn pos(value: &str) -> CallArgument {
        CallArgument::Positional {
            value: value.to_string(),
        }
    }

    fn kw(name: &str, value: &str) -> CallArgument {
        CallArgument::Keyword {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    #[test]
    fn test_classify_reshape_view() {
        assert_eq!(
            classify_method_call("reshape"),
            Some(KnownFunction::Reshape)
        );
        assert_eq!(classify_method_call("view"), Some(KnownFunction::Reshape));
    }

    #[test]
    fn test_classify_reductions() {
        assert_eq!(classify_method_call("sum"), Some(KnownFunction::Sum));
        assert_eq!(classify_method_call("mean"), Some(KnownFunction::Mean));
        assert_eq!(classify_method_call("argmax"), Some(KnownFunction::ArgMax));
    }

    #[test]
    fn test_classify_unsqueeze_aliases_expand_dims() {
        assert_eq!(
            classify_method_call("unsqueeze"),
            Some(KnownFunction::ExpandDims)
        );
        assert_eq!(
            classify_method_call("expand_dims"),
            Some(KnownFunction::ExpandDims)
        );
    }

    #[test]
    fn test_classify_unknown_returns_none() {
        assert_eq!(classify_method_call("frobnicate"), None);
        assert_eq!(classify_method_call("to"), None);
        assert_eq!(classify_method_call("clone"), None);
    }

    #[test]
    fn test_apply_reshape_multi_positional_collapses_to_tuple() {
        let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
        let args = vec![pos("3"), pos("8")];

        let output = apply_method_call(&KnownFunction::Reshape, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["3", "8"])));
    }

    #[test]
    fn test_apply_reshape_single_tuple_arg() {
        let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
        let args = vec![pos("(3, 8)")];

        let output = apply_method_call(&KnownFunction::Reshape, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["3", "8"])));
    }

    #[test]
    fn test_apply_view_with_minus_one() {
        let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
        let args = vec![pos("3"), pos("-1")];

        let output = apply_method_call(&KnownFunction::Reshape, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["3", "8"])));
    }

    #[test]
    fn test_apply_reshape_size_mismatch_errors() {
        let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
        let args = vec![pos("3"), pos("9")];

        let result = apply_method_call(&KnownFunction::Reshape, "x", &args, &shapes);

        assert!(result.is_err());
    }

    #[test]
    fn test_apply_flatten_no_args() {
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3", "4"]))]);
        let args: Vec<CallArgument> = vec![];

        let output = apply_method_call(&KnownFunction::Flatten, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["24"])));
    }

    #[test]
    fn test_apply_sum_no_args_reduces_to_scalar() {
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b"]))]);
        let args: Vec<CallArgument> = vec![];

        let output = apply_method_call(&KnownFunction::Sum, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::<String>::new()));
    }

    #[test]
    fn test_apply_sum_keyword_axis() {
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
        let args = vec![kw("axis", "0")];

        let output = apply_method_call(&KnownFunction::Sum, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["features"])));
    }

    #[test]
    fn test_apply_mean_keyword_dim_torch_style() {
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
        let args = vec![kw("dim", "1"), kw("keepdim", "True")];

        let output = apply_method_call(&KnownFunction::Mean, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "1"])));
    }

    #[test]
    fn test_apply_squeeze_axis() {
        let shapes = HashMap::from([("x".to_string(), shape(&["1", "batch", "1", "features"]))]);
        let args = vec![pos("0")];

        let output = apply_method_call(&KnownFunction::Squeeze, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "1", "features"])));
    }

    #[test]
    fn test_apply_unsqueeze_dim() {
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
        let args = vec![pos("1")];

        let output = apply_method_call(&KnownFunction::ExpandDims, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "1", "features"])));
    }

    #[test]
    fn test_apply_permute_multi_positional() {
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);
        let args = vec![pos("2"), pos("0"), pos("1")];

        let output = apply_method_call(&KnownFunction::Permute, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "a", "b"])));
    }

    #[test]
    fn test_apply_transpose_no_args_reverses() {
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);
        let args: Vec<CallArgument> = vec![];

        let output = apply_method_call(&KnownFunction::Transpose, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_apply_swapaxes_two_positional() {
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);
        let args = vec![pos("0"), pos("2")];

        let output = apply_method_call(&KnownFunction::SwapAxes, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_apply_argmax_axis() {
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "classes"]))]);
        let args = vec![kw("axis", "1")];

        let output = apply_method_call(&KnownFunction::ArgMax, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_apply_cumsum_shape_preserving() {
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
        let args: Vec<CallArgument> = vec![];

        let output = apply_method_call(&KnownFunction::Cumsum, "x", &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_apply_unknown_receiver_returns_none() {
        let shapes: HashMap<String, Vec<String>> = HashMap::new();
        let args: Vec<CallArgument> = vec![];

        let output = apply_method_call(&KnownFunction::Sum, "x", &args, &shapes).unwrap();

        assert_eq!(output, None);
    }
}
