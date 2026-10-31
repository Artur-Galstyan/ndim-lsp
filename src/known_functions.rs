use std::collections::HashMap;

use crate::types::*;

mod jax;
pub(crate) use jax::{compute_qr_multiply_shapes, compute_top_k_shape};

pub fn classify_known_function(target: &ResolvedTarget) -> Option<KnownFunction> {
    let (name, module) = target.parts.split_last()?;

    if let Some(function) = jax::classify(module, name) {
        return Some(function);
    }

    let is_jax = module == ["jax"];
    let is_jax_numpy = module == ["jax", "numpy"];
    let is_numpy = module == ["numpy"];
    let is_torch = module == ["torch"];
    let is_equinox = module == ["equinox"];
    if is_equinox {
        return match name.as_str() {
            "filter_vmap" => Some(KnownFunction::Vmap),
            _ => None,
        };
    }

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
            "minmax" if is_jax_numpy => Some(KnownFunction::MinMax),
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
            "array_split" => Some(KnownFunction::ArraySplit),
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
            "pad" if is_numpy => Some(KnownFunction::NumpyPad),
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
            "diagflat" => Some(KnownFunction::Diagflat),
            "tri" => Some(KnownFunction::Tri),
            "indices" => Some(KnownFunction::Indices),
            "bincount" => Some(KnownFunction::BinCount),
            "unique" => Some(KnownFunction::Unique),
            "select" => Some(KnownFunction::Select),
            "rollaxis" => Some(KnownFunction::RollAxis),
            "resize" => Some(KnownFunction::Resize),
            "insert" => Some(KnownFunction::Insert),
            "delete" => Some(KnownFunction::Delete),
            "append" => Some(KnownFunction::Append),
            "hsplit" => Some(KnownFunction::HSplit),
            "vsplit" => Some(KnownFunction::VSplit),
            "dsplit" => Some(KnownFunction::DSplit),
            "kron" => Some(KnownFunction::Kron),
            "take_along_axis" => Some(KnownFunction::TakeAlongAxis),
            "put_along_axis" => Some(KnownFunction::PutAlongAxis),
            "nonzero" => Some(KnownFunction::Nonzero),
            "argwhere" => Some(KnownFunction::Argwhere),
            "searchsorted" => Some(KnownFunction::SearchSorted),
            "extract" => Some(KnownFunction::Extract),
            "compress" => Some(KnownFunction::Compress),
            "histogram" => Some(KnownFunction::Histogram),
            // Same axis/keepdims mechanics as the reduction they alias to —
            // reused directly rather than adding near-duplicate variants.
            "median" | "quantile" | "percentile" => Some(KnownFunction::Mean),
            "count_nonzero" => Some(KnownFunction::Sum),
            "ptp" => Some(KnownFunction::Max),
            "cross" => Some(KnownFunction::Cross),
            _ => None,
        };
    }

    if is_torch {
        return match name.as_str() {
            "cat" | "concat" | "concatenate" => Some(KnownFunction::Concatenate),
            "stack" => Some(KnownFunction::Stack),
            "reshape" => Some(KnownFunction::Reshape),
            "transpose" | "swapaxes" | "swapdims" => Some(KnownFunction::SwapAxes),
            "unsqueeze" => Some(KnownFunction::ExpandDims),
            "squeeze" => Some(KnownFunction::Squeeze),
            "sum" => Some(KnownFunction::Sum),
            "mean" => Some(KnownFunction::Mean),
            "max" | "min" => Some(KnownFunction::TorchMaxMin),
            "prod" => Some(KnownFunction::Prod),
            "std" => Some(KnownFunction::Std),
            "var" => Some(KnownFunction::Var),
            "matmul" => Some(KnownFunction::Matmul),
            "dot" => Some(KnownFunction::Dot),
            "tensordot" => Some(KnownFunction::TensorDot),
            "outer" => Some(KnownFunction::Outer),
            "inner" => Some(KnownFunction::Inner),
            "einsum" => Some(KnownFunction::Einsum),
            // Real torch `split` semantics: 2nd arg is a chunk *size*, not a
            // section count. `tensor_split` keeps numpy/jnp `array_split`
            // semantics (verified against torch docs).
            "split" => Some(KnownFunction::TorchSplit),
            "tensor_split" => Some(KnownFunction::ArraySplit),
            "tile" => Some(KnownFunction::Tile),
            "repeat" => Some(KnownFunction::Repeat),
            "flatten" => Some(KnownFunction::TorchFlatten),
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
            "zeros" | "ones" | "empty" | "rand" | "randn" => {
                Some(KnownFunction::TorchSizeConstructor)
            }
            "full" => Some(KnownFunction::Full),
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
            "cross" => Some(KnownFunction::TorchCross),
            "gather" => Some(KnownFunction::Gather),
            "scatter" | "scatter_add" => Some(KnownFunction::Scatter),
            "take_along_dim" => Some(KnownFunction::TakeAlongAxis),
            "topk" => Some(KnownFunction::TopK),
            "unbind" => Some(KnownFunction::Unbind),
            "chunk" => Some(KnownFunction::Chunk),
            "narrow" => Some(KnownFunction::Narrow),
            "select" => Some(KnownFunction::SelectDim),
            "masked_select" => Some(KnownFunction::MaskedSelect),
            "index_select" => Some(KnownFunction::IndexSelect),
            "kthvalue" => Some(KnownFunction::KthValue),
            "median" | "mode" => Some(KnownFunction::MedianDim),
            "unique" => Some(KnownFunction::Unique),
            "combinations" => Some(KnownFunction::Combinations),
            "cartesian_prod" => Some(KnownFunction::CartesianProd),
            "block_diag" => Some(KnownFunction::BlockDiag),
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
            "interpolate" => Some(KnownFunction::Interpolate),
            "conv1d" => Some(KnownFunction::FunctionalConv1d),
            "conv2d" => Some(KnownFunction::FunctionalConv2d),
            "conv3d" => Some(KnownFunction::FunctionalConv3d),
            "max_pool1d" => Some(KnownFunction::FunctionalMaxPool1d),
            "max_pool2d" => Some(KnownFunction::FunctionalMaxPool2d),
            "max_pool3d" => Some(KnownFunction::FunctionalMaxPool3d),
            "avg_pool1d" => Some(KnownFunction::FunctionalAvgPool1d),
            "avg_pool2d" => Some(KnownFunction::FunctionalAvgPool2d),
            "avg_pool3d" => Some(KnownFunction::FunctionalAvgPool3d),
            "softmax" | "log_softmax" | "normalize" => Some(KnownFunction::Copy),
            "one_hot" => Some(KnownFunction::OneHot),
            "embedding" => Some(KnownFunction::FunctionalEmbedding),
            // Full activation family: all shape-preserving (elementwise),
            // same rule as `softmax`/`log_softmax`/`normalize` above.
            "relu" | "relu6" | "gelu" | "silu" | "sigmoid" | "tanh" | "softplus"
            | "softsign" | "selu" | "celu" | "elu" | "leaky_relu" | "hardtanh"
            | "hardswish" | "hardsigmoid" | "mish" => Some(KnownFunction::Copy),
            // Dropout family: shape-preserving (a no-op at inference / same
            // rule as `torch.nn.Dropout*` layers).
            "dropout" | "dropout2d" | "dropout3d" => Some(KnownFunction::Copy),
            // `glu` halves a dim rather than preserving shape.
            "glu" => Some(KnownFunction::FunctionalGlu),
            _ => None,
        };
    }

    if module == ["torch", "nn", "utils", "rnn"] {
        return match name.as_str() {
            "pad_sequence" => Some(KnownFunction::PadSequence),
            _ => None,
        };
    }

    if is_jax_numpy_linalg || is_numpy_linalg || is_torch_linalg {
        return match name.as_str() {
            "inv" => Some(KnownFunction::LinalgInv),
            "det" => Some(KnownFunction::LinalgDet),
            "svd" => Some(KnownFunction::LinalgSvd),
            "qr" => Some(KnownFunction::LinalgQr),
            "eig" | "eigh" => Some(KnownFunction::LinalgEig),
            // Axis/keepdims mechanics identical to a plain reduction.
            "norm" => Some(KnownFunction::Sum),
            "solve" => Some(KnownFunction::LinalgSolve),
            // Shape-preserving on a square matrix — same rule as `inv`.
            "cholesky" => Some(KnownFunction::LinalgInv),
            "lstsq" => Some(KnownFunction::LinalgLstsq),
            "pinv" => Some(KnownFunction::LinalgPinv),
            "matrix_rank" => Some(KnownFunction::LinalgMatrixRank),
            "matrix_sqrth" if is_torch_linalg => Some(KnownFunction::LinalgMatrixSqrth),
            _ => None,
        };
    }

    if module == ["jax", "numpy", "fft"] || module == ["numpy", "fft"] || module == ["torch", "fft"]
    {
        return match name.as_str() {
            "fft" | "ifft" | "fft2" | "ifft2" | "fftn" | "ifftn" | "fftshift" | "ifftshift" => {
                Some(KnownFunction::Copy) // shape-preserving; reuse the generic pass-through rule
            }
            _ => None,
        };
    }

    if is_jax_lax {
        return match name.as_str() {
            "dot" => Some(KnownFunction::Dot),
            "dot_general" => Some(KnownFunction::Matmul),
            "scan" => Some(KnownFunction::Scan),
            "map" => Some(KnownFunction::LaxMap),
            "cond" => Some(KnownFunction::LaxCond),
            "switch" => Some(KnownFunction::LaxSwitch),
            "while_loop" => Some(KnownFunction::LaxWhileLoop),
            "fori_loop" => Some(KnownFunction::LaxForiLoop),
            "conv_general_dilated" => Some(KnownFunction::LaxConvGeneralDilated),
            "gather" => Some(KnownFunction::LaxGather),
            "scatter" | "scatter_add" | "scatter_mul" | "scatter_min" | "scatter_max"
            | "scatter_apply" => Some(KnownFunction::LaxScatter),
            "reduce_window" => Some(KnownFunction::LaxReduceWindow),
            "top_k" => Some(KnownFunction::LaxTopK),
            "sort" => Some(KnownFunction::LaxSort),
            "sort_key_val" => Some(KnownFunction::LaxSortKeyVal),
            "pad" => Some(KnownFunction::LaxPad),
            "broadcast" => Some(KnownFunction::LaxBroadcast),
            "broadcast_in_dim" => Some(KnownFunction::LaxBroadcastInDim),
            "slice" => Some(KnownFunction::LaxSlice),
            "dynamic_slice" => Some(KnownFunction::LaxDynamicSlice),
            "dynamic_update_slice" => Some(KnownFunction::LaxDynamicUpdateSlice),
            "concatenate" => Some(KnownFunction::Concatenate),
            "rev" => Some(KnownFunction::Flip),
            "squeeze" => Some(KnownFunction::Squeeze),
            "expand_dims" => Some(KnownFunction::ExpandDims),
            "transpose" => Some(KnownFunction::Transpose),
            "associative_scan" => Some(KnownFunction::LaxAssociativeScan),
            "log2" | "one_minus_square" => Some(KnownFunction::Copy),
            "polynomial" => Some(KnownFunction::LaxPolynomial),
            _ => None,
        };
    }

    if module == ["jax", "random"] && name == "multinomial" {
        return Some(KnownFunction::RandomMultinomial);
    }

    if module == ["jax", "scipy", "special"] && name == "erfcinv" {
        return Some(KnownFunction::Copy);
    }

    if module == ["flax", "linen"] {
        return match name.as_str() {
            "avg_pool" | "max_pool" => Some(KnownFunction::FlaxPool),
            _ => None,
        };
    }

    if module == ["jax", "nn"] {
        return match name.as_str() {
            "one_hot" => Some(KnownFunction::OneHot),
            "dot_product_attention" => Some(KnownFunction::DotProductAttention),
            // Same axis/keepdims mechanics as a plain sum reduction.
            "logsumexp" => Some(KnownFunction::Sum),
            _ => None,
        };
    }

    if module == ["einops"] {
        return match name.as_str() {
            "rearrange" => Some(KnownFunction::EinopsRearrange),
            "reduce" => Some(KnownFunction::EinopsReduce),
            "repeat" => Some(KnownFunction::EinopsRepeat),
            "einsum" => Some(KnownFunction::EinopsEinsum),
            "pack" => Some(KnownFunction::EinopsPack),
            "unpack" => Some(KnownFunction::EinopsUnpack),
            "parse_shape" => Some(KnownFunction::EinopsParseShape),
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
                if name == "a"
                    || name == "x"
                    || name == "input"
                    || name == "array"
                    || name == "y" =>
            {
                return Some(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    None
}

/// Resolve the first array argument to its name and known shape, or `None`
/// if the argument or its shape isn't known. Used by the many `apply_known_*`
/// functions that bail out early when the input's shape is unavailable.
fn first_array_arg_shape<'a, 'b>(
    args: &'a [CallArgument],
    shapes: &'b dyn ShapeLookup,
) -> Option<(&'a str, &'b Vec<String>)> {
    let input_name = first_array_arg(args)?;
    let input_shape = shapes.shape(input_name)?;
    Some((input_name, input_shape))
}

/// After skipping the first `skip` positional arguments, return the value of
/// the next positional argument, or the value of a keyword argument whose
/// name is in `keywords` (whichever comes later wins, matching Python's
/// left-to-right evaluation order).
fn nth_positional_or_keyword<'a>(
    args: &'a [CallArgument],
    skip: usize,
    keywords: &[&str],
) -> Option<&'a str> {
    let mut value = None;
    let mut positional_seen = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value: v } => {
                if positional_seen < skip {
                    positional_seen += 1;
                    continue;
                }
                positional_seen += 1;
                if value.is_none() {
                    value = Some(v.as_str());
                }
            }
            CallArgument::Keyword { name, value: v } if keywords.contains(&name.as_str()) => {
                value = Some(v.as_str());
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    value
}

/// Two-slot variant of [`nth_positional_or_keyword`]: after skipping the
/// first `skip` positional arguments, the next two positional arguments fill
/// `a` and `b` in order, unless overridden by a keyword argument matching
/// `keywords_a` / `keywords_b` respectively.
fn nth_two_positional_or_keywords<'a>(
    args: &'a [CallArgument],
    skip: usize,
    keywords_a: &[&str],
    keywords_b: &[&str],
) -> (Option<&'a str>, Option<&'a str>) {
    let mut a = None;
    let mut b = None;
    let mut positional_seen = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if positional_seen < skip {
                    positional_seen += 1;
                    continue;
                }
                positional_seen += 1;
                if a.is_none() {
                    a = Some(value.as_str());
                } else if b.is_none() {
                    b = Some(value.as_str());
                }
            }
            CallArgument::Keyword { name, value } if keywords_a.contains(&name.as_str()) => {
                a = Some(value.as_str());
            }
            CallArgument::Keyword { name, value } if keywords_b.contains(&name.as_str()) => {
                b = Some(value.as_str());
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    (a, b)
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

        let dim = if dims_canonically_equal(left_dim, right_dim) {
            left_dim.to_string()
        } else if left_dim == "1" {
            right_dim.to_string()
        } else if right_dim == "1" {
            left_dim.to_string()
        } else {
            return Err(format!(
                "cannot broadcast dim {} with {} (shapes [{}] and [{}])",
                left_dim,
                right_dim,
                left.join(", "),
                right.join(", ")
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
        "swapaxes" | "swapdims" => Some(KnownFunction::SwapAxes),
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
        "tile" => Some(KnownFunction::Tile),
        "repeat" | "repeat_interleave" => Some(KnownFunction::Repeat),
        "expand" | "broadcast_to" => Some(KnownFunction::BroadcastTo),
        "astype" => Some(KnownFunction::Astype),
        "copy" | "byteswap" => Some(KnownFunction::Copy),
        "detach" => Some(KnownFunction::Detach),
        "contiguous" => Some(KnownFunction::Contiguous),
        "to" => Some(KnownFunction::To),
        "gather" => Some(KnownFunction::Gather),
        "scatter" => Some(KnownFunction::Scatter),
        "masked_select" => Some(KnownFunction::MaskedSelect),
        "masked_fill" => Some(KnownFunction::MaskedFill),
        "index_select" => Some(KnownFunction::IndexSelect),
        "narrow" => Some(KnownFunction::Narrow),
        "select" => Some(KnownFunction::SelectDim),
        "topk" => Some(KnownFunction::TopK),
        "unfold" => Some(KnownFunction::Unfold),
        "view_as" | "reshape_as" | "expand_as" => Some(KnownFunction::ShapeAs),
        "flip" => Some(KnownFunction::Flip),
        "roll" => Some(KnownFunction::Roll),
        "item" => Some(KnownFunction::Item),
        "new_zeros" | "new_ones" | "new_empty" => Some(KnownFunction::NewSizeConstructor),
        "new_full" => Some(KnownFunction::NewConstructor),
        "clone" => Some(KnownFunction::Copy),
        "cpu" | "cuda" => Some(KnownFunction::Copy),
        "float" | "long" | "int" | "bool" | "double" | "half" => Some(KnownFunction::Copy),
        "clamp" | "clip" => Some(KnownFunction::Copy),
        "softmax" => Some(KnownFunction::Copy),
        "norm" => Some(KnownFunction::Sum),
        "diagonal" => Some(KnownFunction::Diagonal),
        "tril" => Some(KnownFunction::Tril),
        "triu" => Some(KnownFunction::Triu),
        "chunk" => Some(KnownFunction::Chunk),
        "unbind" => Some(KnownFunction::Unbind),
        // Tensor method form: `x.split(size, dim=0)` — real torch semantics
        // (chunk size, not section count); see `KnownFunction::TorchSplit`.
        "split" => Some(KnownFunction::TorchSplit),
        "kthvalue" => Some(KnownFunction::KthValue),
        "median" | "mode" => Some(KnownFunction::MedianDim),
        _ => None,
    }
}

pub fn apply_method_call(
    method: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    // torch `x.repeat(2, 3)` tiles per-dim; numpy `a.repeat(n[, axis])`
    // repeats elements. ≥2 positional args → torch tile semantics.
    let positional_count = args
        .iter()
        .filter(|arg| matches!(arg, CallArgument::Positional { .. }))
        .count();
    let method = if matches!(method, KnownFunction::Repeat) && positional_count >= 2 {
        &KnownFunction::Tile
    } else if matches!(method, KnownFunction::Flatten) && is_torch_flatten_call(args) {
        &KnownFunction::TorchFlatten
    } else {
        method
    };
    let synthesized = synthesize_method_args(method, receiver, args);
    apply_known_function(method, &synthesized, shapes)
}

/// `ndarray.flatten(order)` takes a string order; a positional non-string
/// argument or a `start_dim`/`end_dim` keyword can only be `Tensor.flatten`.
fn is_torch_flatten_call(args: &[CallArgument]) -> bool {
    args.iter().any(|arg| match arg {
        CallArgument::Positional { value } => !value.trim().starts_with(['\'', '"']),
        CallArgument::Keyword { name, .. } => name == "start_dim" || name == "end_dim",
    })
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
        KnownFunction::Reshape
            | KnownFunction::Permute
            | KnownFunction::Transpose
            | KnownFunction::Tile
            | KnownFunction::BroadcastTo
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    match function {
        KnownFunction::Concatenate => apply_known_concatenate(args, shapes),
        KnownFunction::Stack => apply_known_stack(args, shapes),
        KnownFunction::Reshape => apply_known_reshape(args, shapes),
        KnownFunction::Flatten | KnownFunction::Ravel => apply_known_flatten(args, shapes),
        KnownFunction::TorchFlatten => apply_known_torch_flatten(args, shapes),
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
        KnownFunction::Roll
        | KnownFunction::Flip
        | KnownFunction::Triu
        | KnownFunction::Tril
        | KnownFunction::Astype
        | KnownFunction::Copy
        | KnownFunction::Detach
        | KnownFunction::Contiguous
        | KnownFunction::To => apply_known_shape_preserving(args, shapes),
        KnownFunction::Pad => apply_known_pad(args, shapes),
        KnownFunction::NumpyPad => apply_known_numpy_pad(args, shapes),
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
        KnownFunction::TorchMaxMin => apply_known_torch_max_min(args, shapes),
        KnownFunction::LinalgInv => apply_known_linalg_inv(args, shapes),
        KnownFunction::LinalgMatrixSqrth => {
            let Some(input_shape) =
                nth_positional_or_keyword(args, 0, &["A"]).and_then(|name| shapes.shape(name))
            else {
                return Ok(None);
            };
            require_square_matrix(input_shape, "linalg.matrix_sqrth")?;
            Ok(Some(input_shape.clone()))
        }
        KnownFunction::LinalgDet => apply_known_linalg_det(args, shapes),
        KnownFunction::Einsum => apply_known_einsum(args, shapes),
        KnownFunction::Split => apply_known_split(args, shapes),
        KnownFunction::ArraySplit => {
            let _ = compute_array_split_shapes(args, shapes)?;
            Ok(None)
        }
        KnownFunction::TorchSplit => {
            // Same "validate, but return None" reasoning as `Split`: a single
            // (non-tuple) LHS can't hold the real tuple-of-chunks return.
            let _ = compute_torch_split_shapes(args, shapes, None)?;
            Ok(None)
        }
        KnownFunction::FunctionalGlu => apply_known_functional_glu(args, shapes),
        KnownFunction::FlaxPool => apply_known_flax_pool(args, shapes),
        KnownFunction::EinopsRearrange | KnownFunction::EinopsReduce
        | KnownFunction::EinopsRepeat => apply_known_einops(args, shapes),
        KnownFunction::EinopsEinsum => apply_known_einops_einsum(args, shapes),
        KnownFunction::EinopsParseShape => {
            // Returns a dict of axis-name -> size, not an array shape — never
            // meaningful to store as a tensor shape.
            Ok(None)
        }
        KnownFunction::LaxScatter | KnownFunction::LaxDynamicUpdateSlice | KnownFunction::LaxSort => {
            apply_known_shape_preserving(args, shapes)
        }
        KnownFunction::LaxWhileLoop => apply_known_lax_carry(args, shapes, 2, "init_val"),
        KnownFunction::LaxForiLoop => apply_known_lax_carry(args, shapes, 3, "init_val"),
        KnownFunction::LaxAssociativeScan => apply_known_lax_carry(args, shapes, 1, "elems"),
        KnownFunction::LaxConvGeneralDilated => {
            apply_known_lax_conv_general_dilated(args, shapes)
        }
        KnownFunction::LaxReduceWindow => apply_known_lax_reduce_window(args, shapes),
        KnownFunction::LaxPad => apply_known_lax_pad(args, shapes),
        KnownFunction::LaxBroadcast => apply_known_lax_broadcast(args, shapes),
        KnownFunction::LaxBroadcastInDim => apply_known_lax_broadcast_in_dim(args),
        KnownFunction::LaxSlice => apply_known_lax_slice(args, shapes),
        KnownFunction::LaxDynamicSlice => apply_known_lax_dynamic_slice(args, shapes),
        KnownFunction::LaxGather => apply_known_lax_gather(args, shapes),
        KnownFunction::LaxPolynomial => apply_known_lax_polynomial(args, shapes),
        KnownFunction::RandomMultinomial => apply_known_multinomial(args, shapes),
        KnownFunction::Diagflat => apply_known_diagflat(args, shapes),
        KnownFunction::Tri => apply_known_tri(args),
        KnownFunction::Indices => apply_known_indices(args),
        KnownFunction::Select => apply_known_select(args, shapes),
        KnownFunction::RollAxis => apply_known_rollaxis(args, shapes),
        KnownFunction::Resize => apply_known_resize(args),
        KnownFunction::Insert => apply_known_insert(args, shapes),
        KnownFunction::Delete => apply_known_delete(args, shapes),
        KnownFunction::Append => apply_known_append(args, shapes),
        KnownFunction::Kron => apply_known_kron(args, shapes),
        KnownFunction::Block => apply_known_block(args, shapes),
        KnownFunction::TakeAlongAxis => apply_known_take_along_axis(args, shapes),
        KnownFunction::PutAlongAxis => apply_known_shape_preserving(args, shapes),
        KnownFunction::Argwhere => apply_known_argwhere(args, shapes),
        KnownFunction::SearchSorted => apply_known_searchsorted(args, shapes),
        KnownFunction::Histogram => apply_known_histogram(args),
        KnownFunction::Cross => apply_known_cross(args, shapes),
        KnownFunction::TorchCross => apply_known_torch_cross(args, shapes),
        KnownFunction::LinalgSolve => apply_known_linalg_solve(args, shapes),
        KnownFunction::LinalgPinv => apply_known_linalg_pinv(args, shapes),
        KnownFunction::LinalgMatrixRank => apply_known_linalg_matrix_rank(args, shapes),
        KnownFunction::OneHot => apply_known_one_hot(args, shapes),
        KnownFunction::DotProductAttention => apply_known_dot_product_attention(args, shapes),
        KnownFunction::Gather => apply_known_gather(args, shapes),
        KnownFunction::Scatter | KnownFunction::MaskedFill => {
            apply_known_shape_preserving(args, shapes)
        }
        KnownFunction::IndexSelect => apply_known_index_select(args, shapes),
        KnownFunction::Narrow => apply_known_narrow(args, shapes),
        KnownFunction::SelectDim => apply_known_select_dim(args, shapes),
        KnownFunction::MaskedSelect => Ok(None),
        KnownFunction::Unfold => apply_known_unfold(args, shapes),
        KnownFunction::ShapeAs => apply_known_shape_as(args, shapes),
        KnownFunction::Item => Ok(Some(Vec::new())),
        KnownFunction::NewConstructor => apply_known_new_constructor(args),
        KnownFunction::NewSizeConstructor => apply_known_torch_size(&args[args.len().min(1)..]),
        KnownFunction::TorchSizeConstructor => apply_known_torch_size(args),
        KnownFunction::Chunk => {
            // Same "validate, but return None" reasoning as `Split`: the real
            // return is a tuple of N tensors, not expressible as a single
            // shape here — see `compute_chunk_shapes` (used from
            // `analysis.rs`'s tuple-unpacking dispatch) for the real math.
            let _ = compute_chunk_shapes(args, shapes)?;
            Ok(None)
        }
        KnownFunction::TopK
        | KnownFunction::MinMax
        | KnownFunction::Unbind
        | KnownFunction::KthValue
        | KnownFunction::MedianDim => Ok(None),
        KnownFunction::Combinations => apply_known_combinations(args, shapes),
        KnownFunction::CartesianProd => apply_known_cartesian_prod(args, shapes),
        KnownFunction::BlockDiag => apply_known_block_diag(args, shapes),
        KnownFunction::Interpolate => apply_known_interpolate(args, shapes),
        KnownFunction::FunctionalConv1d => apply_known_functional_conv(args, shapes, 1),
        KnownFunction::FunctionalConv2d => apply_known_functional_conv(args, shapes, 2),
        KnownFunction::FunctionalConv3d => apply_known_functional_conv(args, shapes, 3),
        KnownFunction::FunctionalMaxPool1d | KnownFunction::FunctionalAvgPool1d => {
            apply_known_functional_pool(args, shapes, 1)
        }
        KnownFunction::FunctionalMaxPool2d | KnownFunction::FunctionalAvgPool2d => {
            apply_known_functional_pool(args, shapes, 2)
        }
        KnownFunction::FunctionalMaxPool3d | KnownFunction::FunctionalAvgPool3d => {
            apply_known_functional_pool(args, shapes, 3)
        }
        KnownFunction::FunctionalEmbedding => apply_known_functional_embedding(args, shapes),
        KnownFunction::PadSequence => apply_known_pad_sequence(args, shapes),
        KnownFunction::Elementwise { parameters, rank_promotion } => {
            jax::apply_elementwise(args, shapes, parameters, *rank_promotion)
        }
        KnownFunction::BroadcastLike => jax::apply_broadcast_like(args, shapes),
        KnownFunction::Hadamard | KnownFunction::Dft | KnownFunction::InvHilbert
        | KnownFunction::InvPascal | KnownFunction::Helmert | KnownFunction::Circulant
        | KnownFunction::Fiedler | KnownFunction::Companion | KnownFunction::FiedlerCompanion
        | KnownFunction::Leslie | KnownFunction::ConvolutionMatrix => {
            jax::apply_matrix(function, args, shapes)
        }
        // Validate tuple-returning JAX calls without assigning a tensor shape
        // to the tuple itself. Tuple unpacking uses these same helpers.
        KnownFunction::LaxTopK => {
            compute_top_k_shape(args, shapes)?;
            Ok(None)
        }
        KnownFunction::QrMultiply => {
            compute_qr_multiply_shapes(args, shapes)?;
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Parse one side of an einops pattern into axis groups:
/// `"c (h p1) (w p2)"` → `[[c], [h, p1], [w, p2]]`. Nested parens or
/// non-identifier tokens reject the pattern.
fn parse_einops_groups(side: &str) -> Option<Vec<Vec<String>>> {
    let spaced = side.replace('(', " ( ").replace(')', " ) ");
    let mut groups = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for token in spaced.split_whitespace() {
        match token {
            "(" => {
                if current.is_some() {
                    return None;
                }
                current = Some(Vec::new());
            }
            ")" => groups.push(current.take()?),
            "..." => {
                // Ellipsis is only supported as a standalone group (not
                // nested inside a composite `(... h)` group).
                if current.is_some() {
                    return None;
                }
                groups.push(vec!["...".to_string()]);
            }
            name => {
                if !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    return None;
                }
                match &mut current {
                    Some(group) => group.push(name.to_string()),
                    None => groups.push(vec![name.to_string()]),
                }
            }
        }
    }
    if current.is_some() {
        return None;
    }
    Some(groups)
}

/// `einops.rearrange / reduce / repeat` — the pattern string carries the
/// full shape algebra. LHS groups bind axis names against the input dims
/// (composite groups solve one unknown factor by division, concrete dims
/// only); RHS groups multiply bound axes back together. Keyword arguments
/// (`p1=16`, `n=4`) pre-seed the bindings.
///
/// `...` (ellipsis) is supported as a standalone group appearing on both
/// sides: it binds to however many leading/unmatched dims remain once the
/// named groups are accounted for, and passes them through unchanged and
/// in order on the output side. Ellipsis nested inside a composite group
/// (e.g. `(... h)`), or appearing on only one side, isn't modelled.
pub(crate) fn apply_known_einops(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let positional = positional_arg_values(args);
    let Some(input_name) = positional.first() else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.shape(input_name.as_str()) else {
        return Ok(None);
    };
    let Some(pattern_raw) = positional.get(1) else {
        return Ok(None);
    };

    let trimmed = pattern_raw.trim();
    let pattern = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return Ok(None);
    };
    let Some((lhs, rhs)) = pattern.split_once("->") else {
        return Ok(None);
    };
    let Some(mut lhs_groups) = parse_einops_groups(lhs) else {
        return Ok(None);
    };
    let Some(mut rhs_groups) = parse_einops_groups(rhs) else {
        return Ok(None);
    };

    // Axis bindings, pre-seeded from keyword args (p1=16, n=4, …) before
    // rank checks/ellipsis expansion so both can see them.
    let mut bound: HashMap<String, String> = args
        .iter()
        .filter_map(|arg| match arg {
            CallArgument::Keyword { name, value } => Some((name.clone(), value.clone())),
            _ => None,
        })
        .collect();

    let lhs_ellipsis = lhs_groups.iter().position(|g| g.as_slice() == ["..."]);
    let rhs_ellipsis = rhs_groups.iter().position(|g| g.as_slice() == ["..."]);
    if lhs_ellipsis.is_some() != rhs_ellipsis.is_some() {
        // Ellipsis on only one side isn't modelled — skip rather than guess.
        return Ok(None);
    }
    if let Some(lhs_idx) = lhs_ellipsis {
        let n_named = lhs_groups.len() - 1;
        if input_shape.len() < n_named {
            return Err(format!(
                "einops pattern '{}' expects at least rank {}, got rank {} for '{}'",
                pattern,
                n_named,
                input_shape.len(),
                input_name
            ));
        }
        let n_ellipsis_dims = input_shape.len() - n_named;
        let ellipsis_names: Vec<String> =
            (0..n_ellipsis_dims).map(|i| format!("__ellipsis{i}__")).collect();
        for (name, dim) in ellipsis_names
            .iter()
            .zip(&input_shape[lhs_idx..lhs_idx + n_ellipsis_dims])
        {
            bound.insert(name.clone(), dim.clone());
        }
        let ellipsis_groups: Vec<Vec<String>> =
            ellipsis_names.iter().map(|n| vec![n.clone()]).collect();
        lhs_groups.splice(lhs_idx..=lhs_idx, ellipsis_groups.clone());
        let rhs_idx = rhs_ellipsis.unwrap();
        rhs_groups.splice(rhs_idx..=rhs_idx, ellipsis_groups);
    }

    if lhs_groups.len() != input_shape.len() {
        return Err(format!(
            "einops pattern '{}' expects rank {}, got rank {} for '{}'",
            pattern,
            lhs_groups.len(),
            input_shape.len(),
            input_name
        ));
    }

    for (group, dim) in lhs_groups.iter().zip(input_shape.iter()) {
        if let [name] = group.as_slice() {
            if name != "1" {
                bound.entry(name.clone()).or_insert_with(|| dim.clone());
            }
            continue;
        }
        // Composite group: solve at most one unknown factor by division.
        let mut unknown: Option<&String> = None;
        let mut known_product: usize = 1;
        let mut knowns_concrete = true;
        for factor in group {
            let value = if factor == "1" {
                Some("1".to_string())
            } else {
                bound.get(factor).cloned()
            };
            match value {
                Some(v) => match v.parse::<usize>() {
                    Ok(n) => known_product *= n,
                    Err(_) => knowns_concrete = false,
                },
                None if unknown.is_none() => unknown = Some(factor),
                None => return Ok(None),
            }
        }
        if let Some(factor) = unknown {
            if !knowns_concrete || known_product == 0 {
                return Ok(None);
            }
            let Ok(d) = dim.parse::<usize>() else {
                return Ok(None);
            };
            if d % known_product != 0 {
                return Err(format!(
                    "einops: dim {} is not divisible by the known factors ({}) of axis '{}'",
                    d, known_product, factor
                ));
            }
            bound.insert(factor.clone(), (d / known_product).to_string());
        }
    }

    let mut output = Vec::with_capacity(rhs_groups.len());
    for group in &rhs_groups {
        let mut dim = "1".to_string();
        for factor in group {
            let value = if factor == "1" {
                "1".to_string()
            } else {
                match bound.get(factor) {
                    Some(v) => v.clone(),
                    None => return Ok(None),
                }
            };
            dim = multiply_dim(&dim, &value);
        }
        output.push(dim);
    }
    Ok(Some(output))
}

/// `flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))` — channels-
/// LAST: the window applies to the dims immediately before the trailing
/// channel dim. Default strides = 1, default padding VALID:
/// out = (d - w)/s + 1. Concrete dims only in v1; symbolic inputs skip.
fn apply_known_flax_pool(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let (window, strides) =
        nth_two_positional_or_keywords(args, 1, &["window_shape"], &["strides"]);
    let Some(window) = window.and_then(parse_shape_value) else {
        return Ok(None);
    };
    let strides = strides
        .and_then(parse_shape_value)
        .unwrap_or_else(|| vec!["1".to_string(); window.len()]);
    if strides.len() != window.len() {
        return Ok(None);
    }

    let spatial_rank = window.len();
    if input_shape.len() < spatial_rank + 1 {
        return Err(format!(
            "flax pool requires input with at least {} dims, got {} for '{}'",
            spatial_rank + 1,
            input_shape.len(),
            input_name
        ));
    }
    let start = input_shape.len() - spatial_rank - 1;
    let mut output = input_shape.clone();
    for i in 0..spatial_rank {
        let dim = &input_shape[start + i];
        let (Ok(d), Ok(w), Ok(s)) = (
            dim.parse::<isize>(),
            window[i].parse::<isize>(),
            strides[i].parse::<isize>(),
        ) else {
            return Ok(None);
        };
        if s <= 0 {
            return Ok(None);
        }
        output[start + i] = ((d - w) / s + 1).to_string();
    }
    Ok(Some(output))
}

fn sequence_arg_value(args: &[CallArgument]) -> Option<&str> {
    nth_positional_or_keyword(args, 0, &["arrays", "tensors", "arys", "operands"])
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

fn apply_known_einsum(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let positional_values = positional_arg_values(args);
    let Some(equation) = positional_values.first() else {
        return Ok(None);
    };

    let equation_trimmed = equation.trim();

    // Remove quotes if present
    let equation_str = if (equation_trimmed.starts_with('"') && equation_trimmed.ends_with('"'))
        || (equation_trimmed.starts_with('\'') && equation_trimmed.ends_with('\''))
    {
        &equation_trimmed[1..equation_trimmed.len() - 1]
    } else {
        // Not a string literal
        return Ok(None);
    };

    // If equation contains ellipsis, return Ok(None) for v1
    if equation_str.contains("...") {
        return Ok(None);
    }

    // Split on "->" for explicit output
    let Some((inputs_part, output_part)) = equation_str.split_once("->") else {
        // Implicit-mode (no "->"), return Ok(None) for v1
        return Ok(None);
    };

    let input_specs: Vec<&str> = inputs_part.split(',').map(str::trim).collect();
    let output_spec = output_part.trim();

    // Collect operand shapes (reuse positional_values from above)
    let operand_names = &positional_values[1..]; // Skip equation string

    if operand_names.len() != input_specs.len() {
        return Err(format!(
            "einsum equation has {} input specs but got {} operands",
            input_specs.len(),
            operand_names.len()
        ));
    }

    // Build label→dim map
    let mut label_map: HashMap<char, String> = HashMap::new();

    for (spec, operand_name) in input_specs.iter().zip(operand_names.iter()) {
        let Some(shape) = shapes.shape(operand_name.as_str()) else {
            return Ok(None);
        };

        // Note: einsum subscripts are ASCII letters in practice, so chars().count() == byte len.
        // Using chars().count() is more correct for the rank comparison.
        let spec_label_count = spec.chars().count();
        if shape.len() != spec_label_count {
            return Err(format!(
                "einsum operand '{}' has rank {} but subscript '{}' has length {}",
                operand_name,
                shape.len(),
                spec,
                spec_label_count
            ));
        }

        for (label_char, dim) in spec.chars().zip(shape.iter()) {
            if let Some(existing_dim) = label_map.get(&label_char) {
                check_dim_match(existing_dim, dim, &format!("einsum label '{}'", label_char))?;
            } else {
                label_map.insert(label_char, dim.clone());
            }
        }
    }

    // Build output shape
    let mut output_shape = Vec::new();
    for label_char in output_spec.chars() {
        let Some(dim) = label_map.get(&label_char) else {
            return Err(format!(
                "einsum output label '{}' not found in input subscripts",
                label_char
            ));
        };
        output_shape.push(dim.clone());
    }

    Ok(Some(output_shape))
}

fn apply_known_concatenate(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };

    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let Some(axis) = parse_axis(
        nth_positional_or_keyword(args, 1, &["axis", "dim", "dimension"]).unwrap_or("0"),
    ) else {
        return Ok(None);
    };

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.shape(input_name) else {
            return Ok(None);
        };
        input_shapes.push(shape.clone());
    }

    // torch.cat skips a 1-D empty tensor of any rank, so the ranks may differ.
    let legacy_empty = |shape: &Vec<String>| shape.len() == 1 && shape[0] == "0";
    if input_shapes.iter().any(legacy_empty)
        && input_shapes.iter().any(|shape| shape.len() != 1)
    {
        return Ok(None);
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
            if !dims_canonically_equal(&shape[dim_idx], &first_shape[dim_idx]) {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };
    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.shape(input_name) else {
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
            if !dims_canonically_equal(expected, got) {
                return Err(format!(
                    "stack dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, expected, got
                ));
            }
        }
    }

    let output_rank = first_shape.len() + 1;
    let Some(axis) = parse_axis(
        nth_positional_or_keyword(args, 1, &["axis", "dim"]).unwrap_or("0"),
    ) else {
        return Ok(None);
    };
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let Some(shape_value) = nth_positional_or_keyword(args, 1, &["shape", "newshape", "size"])
    else {
        return Ok(None);
    };
    let Some(mut output_shape) = parse_shape_value(shape_value) else {
        return Ok(None);
    };

    for dim in &mut output_shape {
        if let Some(resolved) = resolve_shape_index(dim, shapes) {
            *dim = resolved;
        }
    }

    let minus_one_count = output_shape
        .iter()
        .filter(|dim| dim.as_str() == "-1")
        .count();
    if minus_one_count > 1 {
        return Err("reshape can only infer one -1 dimension".to_string());
    }

    if minus_one_count == 1 {
        let known_dims: Vec<String> = output_shape
            .iter()
            .filter(|dim| dim.as_str() != "-1")
            .cloned()
            .collect();

        let inferred = match (dim_product(input_shape), dim_product(&known_dims)) {
            (Some(input_product), Some(known_product)) => {
                if known_product == 0 || input_product % known_product != 0 {
                    return Err(format!(
                        "reshape cannot infer -1 dimension: input size {} not divisible by {}",
                        input_product, known_product
                    ));
                }
                Some((input_product / known_product).to_string())
            }
            _ => infer_symbolic_minus_one(input_shape, &known_dims),
        };

        let Some(inferred) = inferred else {
            return Ok(None);
        };
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

fn resolve_shape_index(dim: &str, shapes: &dyn ShapeLookup) -> Option<String> {
    let dim = dim.trim();
    let suffix_start = dim.find(".shape[")?;
    let (ident, rest) = dim.split_at(suffix_start);
    let ident = ident.trim();
    if ident.is_empty() || !ident.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let idx_part = rest.strip_prefix(".shape[")?.strip_suffix(']')?;
    let idx: isize = idx_part.trim().parse().ok()?;
    let shape = shapes.shape(ident)?;
    let resolved = if idx >= 0 {
        shape.get(idx as usize)?
    } else {
        let from_end = (-idx) as usize;
        shape.get(shape.len().checked_sub(from_end)?)?
    };
    Some(resolved.clone())
}

fn infer_symbolic_minus_one(input_shape: &[String], known_dims: &[String]) -> Option<String> {
    let mut remaining: Vec<String> = input_shape.to_vec();
    for known in known_dims {
        let pos = remaining.iter().position(|d| d == known)?;
        remaining.remove(pos);
    }
    if remaining.is_empty() {
        Some("1".to_string())
    } else {
        Some(flattened_dim(&remaining))
    }
}

fn apply_known_flatten(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    Ok(Some(vec![flattened_dim(input_shape)]))
}

fn apply_known_torch_flatten(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let dim_arg = |skip: usize, keyword: &str, default: isize| -> Option<isize> {
        match nth_positional_or_keyword(args, skip, &[keyword]) {
            Some(value) => parse_axis(value),
            None => Some(default),
        }
    };
    let (Some(start), Some(end)) = (dim_arg(1, "start_dim", 0), dim_arg(2, "end_dim", -1)) else {
        return Ok(None);
    };

    // torch treats a scalar as rank 1 for dim wrapping and returns shape [1].
    if input_shape.is_empty() {
        return Ok(if matches!(start, 0 | -1) && matches!(end, 0 | -1) {
            Some(vec!["1".to_string()])
        } else {
            None
        });
    }

    let rank = input_shape.len() as isize;
    let wrap = |dim: isize| (-rank..rank).contains(&dim).then(|| dim.rem_euclid(rank) as usize);
    let (Some(start), Some(end)) = (wrap(start), wrap(end)) else {
        return Ok(None);
    };
    if start > end {
        return Ok(None);
    }

    let mut output = input_shape[..start].to_vec();
    output.push(flattened_dim(&input_shape[start..=end]));
    output.extend_from_slice(&input_shape[end + 1..]);
    Ok(Some(output))
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let axes = nth_positional_or_keyword(args, 1, &["axes", "axis", "dims", "permutation"])
        .and_then(parse_axis_list)
        .unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose of '{}' (rank {}) expected {} axes, got {}",
            input_name,
            rank,
            rank,
            axes.len()
        ));
    }
    let mut normalized = Vec::new();
    for axis in axes {
        let axis = normalize_axis(axis, rank, "transpose")?;
        if normalized.contains(&axis) {
            return Err(format!(
                "transpose of '{}': axis {} given more than once",
                input_name, axis
            ));
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let (source, destination) =
        nth_two_positional_or_keywords(args, 1, &["source"], &["destination"]);
    let Some(source) = source.and_then(parse_axis_list) else {
        return Ok(None);
    };
    let Some(destination) = destination.and_then(parse_axis_list) else {
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

/// Handles both a single axis (`jnp.expand_dims(x, 1)`,
/// `x.unsqueeze(0)`) and a sequence of axes (`jax.lax.expand_dims(x, (1,
/// 2))`, `np.expand_dims(x, (0, -1))`) — each entry names a position in the
/// *final* (post-insertion) shape.
fn apply_known_expand_dims(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(axes) = nth_positional_or_keyword(args, 1, &["axis", "dim", "dimensions"])
        .and_then(parse_axis_list)
    else {
        return Ok(None);
    };
    if axes.contains(&isize::MIN) {
        // axis=None isn't meaningful for expand_dims.
        return Ok(None);
    }
    let output_rank = input_shape.len() + axes.len();
    let mut normalized: Vec<usize> = Vec::with_capacity(axes.len());
    for axis in axes {
        let axis = if axis < 0 {
            output_rank as isize + axis
        } else {
            axis
        };
        if axis < 0 || axis as usize >= output_rank {
            return Err(format!(
                "expand_dims axis {} out of bounds for output rank {}",
                axis, output_rank
            ));
        }
        normalized.push(axis as usize);
    }
    normalized.sort_unstable();
    let mut output = input_shape.clone();
    for axis in normalized {
        output.insert(axis.min(output.len()), "1".to_string());
    }
    Ok(Some(output))
}

fn apply_known_squeeze(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let axes =
        nth_positional_or_keyword(args, 1, &["axis", "dim", "dimensions"]).and_then(parse_axis_list);

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
    shapes: &dyn ShapeLookup,
    min_rank: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() >= min_rank {
        return Ok(Some(input_shape.clone()));
    }
    let mut output = input_shape.clone();
    if min_rank == 3 && !output.is_empty() {
        output.push("1".to_string());
    }
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

/// Resolve the first two positional arguments to their names and known
/// shapes, or `None` if either argument or its shape isn't known. Used by
/// the binary shape functions (matmul, dot, tensordot, outer, inner, vdot).
fn resolve_binary_shapes<'a>(
    args: &[CallArgument],
    shapes: &'a dyn ShapeLookup,
) -> Option<(String, &'a Vec<String>, String, &'a Vec<String>)> {
    let (left_name, right_name) = first_two_positional_values(args)?;
    let left = shapes.shape(&left_name)?;
    let right = shapes.shape(&right_name)?;
    Some((left_name, left, right_name, right))
}

/// Validate that `shape` has rank >= 2 and its last two dimensions match,
/// erroring with `context` (e.g. `"linalg.inv"`) if not.
fn require_square_matrix(shape: &[String], context: &str) -> Result<(), String> {
    if shape.len() < 2 {
        return Err(format!(
            "{} requires rank >= 2, got rank {}",
            context,
            shape.len()
        ));
    }
    let last = &shape[shape.len() - 1];
    let second_last = &shape[shape.len() - 2];
    if last != second_last {
        return Err(format!(
            "{} requires last two dimensions to match, got {} and {}",
            context, second_last, last
        ));
    }
    Ok(())
}

fn check_dim_match(left: &str, right: &str, context: &str) -> Result<(), String> {
    if dims_canonically_equal(left, right) {
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

/// Torch size constructors take the dims as varargs (`torch.zeros(4, 5)`)
/// or as one tuple/list (`torch.zeros((4, 5))`, `size=(4, 5)`). Anything
/// else (star args, shape attributes, mixed forms) stays unknown.
fn apply_known_torch_size(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let positional = positional_arg_values(args);
    let size_keyword = args.iter().find_map(|arg| match arg {
        CallArgument::Keyword { name, value } if name == "size" => Some(value),
        _ => None,
    });
    let is_literal = |value: &String| value.starts_with(['(', '[']);
    let dims = match (positional.as_slice(), size_keyword) {
        ([], Some(value)) if is_pure_literal(value) => parse_shape_value(value),
        ([value], None) if is_pure_literal(value) => parse_shape_value(value),
        ([value], None) if value.parse::<usize>().is_ok() => Some(vec![value.clone()]),
        ([_], None) => None,
        (values, None) if !values.is_empty() => Some(values.to_vec()),
        _ => None,
    };
    Ok(dims.filter(|dims| {
        dims.iter().all(|dim| {
            !dim.starts_with('*') && is_balanced(dim) && (dims.len() == 1 || !is_literal(dim))
        })
    }))
}

/// True when the text is one complete `(...)` or `[...]` group, so it is
/// not an expression such as `(B, S) + (3,)`.
fn is_pure_literal(value: &str) -> bool {
    let value = value.trim();
    if !value.starts_with(['(', '[']) {
        return false;
    }
    let mut depth = 0usize;
    for (index, ch) in value.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth = match depth.checked_sub(1) {
                    Some(depth) => depth,
                    None => return false,
                };
                if depth == 0 && index + 1 != value.len() {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn is_balanced(value: &str) -> bool {
    let mut depth = 0usize;
    for ch in value.chars() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => match depth.checked_sub(1) {
                Some(next) => depth = next,
                None => return false,
            },
            _ => {}
        }
    }
    depth == 0
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
            && (name == "num" || name == "steps")
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, left, _, right)) = resolve_binary_shapes(args, shapes) else {
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
            // Invariant: the `match` arms above handle (0, _), (_, 0), (1, 1), (1, 2),
            // and (2, 1). In the `_` arm both left.len() and right.len() are >= 1.
            let left_k = left
                .last()
                .expect("invariant: left.len() >= 1 in the `_` arm of matmul");
            let right_k = if right.len() == 1 {
                right
                    .last()
                    .expect("invariant: right.len() >= 1 in the `_` arm of matmul")
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, left, _, right)) = resolve_binary_shapes(args, shapes) else {
        return Ok(None);
    };

    if left.is_empty() || right.is_empty() {
        return Err("dot does not support scalar inputs".to_string());
    }

    if right.len() == 1 {
        // Invariant: `left.is_empty()` guard above ensures left.last() is Some.
        let left_last = left
            .last()
            .expect("invariant: left non-empty checked above");
        check_dim_match(left_last, &right[0], "dot")?;
        return Ok(Some(left[..left.len() - 1].to_vec()));
    }

    // Invariant: `left.is_empty()` guard above ensures left.last() is Some.
    let left_last = left
        .last()
        .expect("invariant: left non-empty checked above");
    check_dim_match(left_last, &right[right.len() - 2], "dot")?;
    let mut output = left[..left.len() - 1].to_vec();
    output.extend(right[..right.len() - 2].to_vec());
    output.push(right[right.len() - 1].clone());
    Ok(Some(output))
}

fn apply_known_tensordot(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, left, _, right)) = resolve_binary_shapes(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, left, _, right)) = resolve_binary_shapes(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, left, _, right)) = resolve_binary_shapes(args, shapes) else {
        return Ok(None);
    };

    if left.is_empty() || right.is_empty() {
        return Err("inner does not support scalar inputs".to_string());
    }

    // Invariant: `left.is_empty() || right.is_empty()` guard above ensures both are non-empty.
    let left_last = left
        .last()
        .expect("invariant: left non-empty checked above");
    let right_last = right
        .last()
        .expect("invariant: right non-empty checked above");
    check_dim_match(left_last, right_last, "inner")?;

    let mut output = left[..left.len() - 1].to_vec();
    output.extend(right[..right.len() - 1].iter().cloned());
    Ok(Some(output))
}

fn apply_known_vdot(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if resolve_binary_shapes(args, shapes).is_none() {
        return Ok(None);
    }

    Ok(Some(Vec::new()))
}

fn apply_known_diag(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 2 {
        return Ok(None);
    }
    let Some(input_shape) = shapes.shape(&values[0]) else {
        return Ok(None);
    };
    let Some(indices_shape) = shapes.shape(&values[1]) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let Some(mut target) = nth_positional_or_keyword(args, 1, &["shape"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    // torch `x.expand(...)`: -1 means "keep this dim". Right-align against
    // the input shape (expand may prepend new leading dims).
    let offset = target.len().saturating_sub(input_shape.len());
    for (i, dim) in target.iter_mut().enumerate() {
        if dim == "-1" && i >= offset {
            dim.clone_from(&input_shape[i - offset]);
        }
    }
    Ok(Some(target))
}

fn apply_known_broadcast_arrays(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
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
        let Some(shape) = shapes.shape(&input_name) else {
            return Ok(None);
        };
        output = broadcast_two_shapes(&output, shape)?;
    }
    Ok(Some(output))
}

fn apply_known_tile(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let Some(reps) =
        nth_positional_or_keyword(args, 1, &["reps", "dims"]).and_then(parse_shape_value)
    else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let (repeats, axis) = nth_two_positional_or_keywords(args, 1, &["repeats"], &["axis", "dim"]);
    let axis = axis.and_then(parse_axis);
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    Ok(shapes.shape(input_name).cloned())
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
            if !dims_canonically_equal(&shape[dim_idx], &first_shape[dim_idx]) {
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
    shapes: &dyn ShapeLookup,
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
        let Some(shape) = shapes.shape(&input_name) else {
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "rot90 expects rank >= 2, got rank {}",
            input_shape.len()
        ));
    }

    let (k_value, axes_value) = nth_two_positional_or_keywords(args, 1, &["k"], &["axes"]);
    let k = k_value.and_then(parse_axis).unwrap_or(1);
    let axes = axes_value
        .and_then(parse_axis_list)
        .unwrap_or_else(|| vec![0, 1]);
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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let Some(pad_width) = nth_positional_or_keyword(args, 1, &["pad_width", "pad"]) else {
        return Ok(None);
    };
    if pad_width.contains(['{', '}']) {
        return Ok(None);
    }
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
            .as_chunks::<2>()
            .0
            .iter()
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

fn apply_known_numpy_pad(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(width) = nth_positional_or_keyword(args, 1, &["pad_width"]) else {
        return Ok(None);
    };
    let width = width.trim();
    if !width.starts_with('{') || !width.ends_with('}') {
        return if width.contains(['{', '}']) {
            Ok(None)
        } else {
            apply_known_pad(args, shapes)
        };
    }
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let mut additions = vec![0isize; input_shape.len()];
    let mut seen = vec![false; input_shape.len()];
    let inner = &width[1..width.len() - 1];
    let mut start = 0;
    let mut depth = 0;
    for (i, ch) in inner
        .char_indices()
        .chain(std::iter::once((inner.len(), ',')))
    {
        match ch {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            ',' if depth == 0 => {
                let entry = inner[start..i].trim();
                if !entry.is_empty() {
                    let Some((axis, value)) = entry.split_once(':') else {
                        return Ok(None);
                    };
                    let Some(axis) = parse_axis(axis) else {
                        return Ok(None);
                    };
                    let axis = normalize_axis(axis, input_shape.len(), "pad")?;
                    if seen[axis] {
                        return Ok(None);
                    }
                    seen[axis] = true;
                    let value = value.trim();
                    let pair = value.strip_prefix('(').and_then(|v| v.strip_suffix(')'));
                    let amount = if let Some(pair) = pair {
                        let Some((before, after)) = pair.split_once(',') else {
                            return Ok(None);
                        };
                        let (Ok(before), Ok(after)) = (
                            before.trim().parse::<isize>(),
                            after.trim().parse::<isize>(),
                        ) else {
                            return Ok(None);
                        };
                        if before < 0 || after < 0 {
                            return Ok(None);
                        }
                        before.checked_add(after)
                    } else {
                        value
                            .parse::<isize>()
                            .ok()
                            .filter(|v| *v >= 0)
                            .and_then(|v| v.checked_mul(2))
                    };
                    let Some(amount) = amount else {
                        return Ok(None);
                    };
                    additions[axis] = amount;
                } else if i != inner.len() {
                    return Ok(None);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Ok(None);
    }
    Ok(Some(
        input_shape
            .iter()
            .zip(additions)
            .map(|(dim, add)| add_to_dim(dim, add))
            .collect(),
    ))
}

fn apply_known_where(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 3 {
        return Ok(None);
    }
    let mut output = Vec::new();
    for value in values.iter().take(3) {
        let Some(shape) = shapes.shape(value) else {
            return Ok(None);
        };
        output = broadcast_two_shapes(&output, shape)?;
    }
    Ok(Some(output))
}

/// `torch.max(a, b)` broadcasts elementwise. A `dim` returns a
/// `(values, indices)` namedtuple, which has no single shape. A broadcast
/// mismatch stays unknown, like every other torch elementwise binary op.
fn apply_known_torch_max_min(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let has_dim = args.iter().any(|arg| {
        matches!(arg, CallArgument::Keyword { name, .. } if name == "dim" || name == "axis")
    });
    match nth_positional_or_keyword(args, 1, &["other"]) {
        Some(other) if !has_dim => {
            let (Some((_, input)), Some(other)) = (first_array_arg_shape(args, shapes), shapes.shape(other))
            else {
                return Ok(None);
            };
            // A scalar may be an int `dim`, which returns a tuple.
            if other.is_empty() {
                return Ok(None);
            }
            Ok(broadcast_two_shapes(input, other).ok())
        }
        Some(_) => Ok(None),
        None if has_dim => Ok(None),
        None => apply_known_reduction(args, shapes),
    }
}

pub fn apply_known_reduction(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let axis_raw = nth_positional_or_keyword(args, 1, &["axis", "dim"]);
    let mut keepdims = false;
    for arg in args {
        if let CallArgument::Keyword { name, value } = arg
            && (name == "keepdims" || name == "keepdim")
            && let Some(parsed) = parse_bool(value)
        {
            keepdims = parsed;
        }
    }

    let axes = match axis_raw {
        None => None,
        Some(raw) => match parse_axis_list(raw) {
            Some(parsed) => Some(parsed),
            None => return Ok(None),
        },
    };

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
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    require_square_matrix(input_shape, "linalg.inv")?;

    Ok(Some(input_shape.clone()))
}

fn apply_known_lax_polynomial(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(x) = nth_positional_or_keyword(args, 0, &["x"]).and_then(|x| shapes.shape(x)) else {
        return Ok(None);
    };
    let Some(coeffs) = nth_positional_or_keyword(args, 1, &["coeffs"]) else {
        return Ok(None);
    };
    let names = if let Some(sequence) = coeffs
        .trim()
        .strip_prefix(['[', '('])
        .and_then(|value| value.strip_suffix([']', ')']))
    {
        parse_simple_sequence_names(sequence).unwrap_or_default()
    } else if let Some(shape) = shapes.shape(coeffs) {
        let Some(first) = shape.first().and_then(|dim| dim.parse::<usize>().ok()) else {
            return Ok(None);
        };
        if first == 0 {
            return Ok(Some(x.clone()));
        }
        let coeff = &shape[1..];
        if !coeff.is_empty() && coeff.len() != x.len() {
            return Err("lax.polynomial coefficients must have the same rank as x".to_string());
        }
        return broadcast_two_shapes(x, coeff).map(Some);
    } else {
        return Ok(None);
    };
    let mut result = x.clone();
    for name in names {
        let coeff = if let Some(shape) = shapes.shape(&name) {
            shape.as_slice()
        } else if name.parse::<f64>().is_ok() || parse_bool(&name).is_some() {
            &[]
        } else {
            return Ok(None);
        };
        if !coeff.is_empty() && coeff.len() != x.len() {
            return Err("lax.polynomial coefficients must have the same rank as x".to_string());
        }
        result = broadcast_two_shapes(&result, coeff)?;
    }
    Ok(Some(result))
}

fn apply_known_multinomial(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(p) = nth_positional_or_keyword(args, 2, &["p"]).and_then(|p| shapes.shape(p)) else {
        return Ok(None);
    };
    let Some((categories, p_batch)) = p.split_last() else {
        return Err("random.multinomial requires p with a category axis".to_string());
    };
    let batch = match nth_positional_or_keyword(args, 3, &["shape"]) {
        None | Some("None") => p_batch.to_vec(),
        Some(value) => {
            let Some(inner) = value
                .trim()
                .strip_prefix('(')
                .and_then(|v| v.strip_suffix(')'))
            else {
                return Ok(None);
            };
            if inner.trim().is_empty() {
                Vec::new()
            } else {
                let mut dims = Vec::new();
                let parts = inner.strip_suffix(',').unwrap_or(inner);
                for part in parts.split(',') {
                    let Ok(dim) = part.trim().parse::<usize>() else {
                        return Ok(None);
                    };
                    dims.push(dim.to_string());
                }
                dims
            }
        }
    };
    if broadcast_two_shapes(&batch, p_batch)? != batch {
        return Err("random.multinomial shape cannot broadcast p batch dimensions".to_string());
    }
    if let Some(n) = nth_positional_or_keyword(args, 1, &["n"]).and_then(|n| shapes.shape(n))
        && broadcast_two_shapes(&batch, n)? != batch
    {
        return Err("random.multinomial n cannot broadcast to the batch shape".to_string());
    }
    let mut output = batch;
    output.push(categories.clone());
    Ok(Some(output))
}

fn apply_known_linalg_det(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    require_square_matrix(input_shape, "linalg.det")?;

    Ok(Some(input_shape[..input_shape.len() - 2].to_vec()))
}

/// A single assignment cannot store the multiple arrays returned by split.
/// Validate here; tuple unpacking calls `compute_split_shapes` directly.
fn apply_known_split(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    // Validate — surface errors to the user but don’t store a shape.
    let _ = compute_split_shapes(args, shapes)?;
    Ok(None)
}

/// Compute the per-element output shapes for a `split`-family call.
///
/// Returns `Ok(Some(shapes))` where `shapes` is a `Vec` of length `N`, one
/// element per output array. Returns `Ok(None)` when the split specification
/// is not a recognised literal (CASE 3: out of scope — a variable or
/// non-literal expression).
///
/// ## Semantics
///
/// ### CASE 1 — integer literal `N`
///
/// `split(x, N, axis=k)` divides axis `k` into `N` equal parts.
///
/// * If the axis dimension is **numeric** and divisible by `N`: each output
///   gets `axis_dim / N` along the split axis.
/// * If the axis dimension is **symbolic**: each output gets a synthetic
///   dimension named `"split(<axis_dim>, <N>)"`.
/// * If the axis dimension is numeric but **not** divisible by `N`: error.
///
/// ### CASE 2 — list literal `[i₁, i₂, …, iₖ]`
///
/// Each index is a Python slice boundary: negative indices resolve from
/// the end, out-of-range indices clip, and reversed slices have size zero.
///
/// ### CASE 3 — non-literal (out of scope)
///
/// Returns `Ok(None)`.
/// Simplify `dim / n` by cancelling a literal factor `n` from a flat product,
/// e.g. `"d_model * 3" / 3 → "d_model"`, `"3 * a * b" / 3 → "a * b"`. Returns
/// `None` when `dim` isn't a plain product or has no matching literal factor.
/// Common for fused projections: `Linear(d, 3*d)` then `split(qkv, 3)`.
fn cancel_product_factor(dim: &str, n: usize) -> Option<String> {
    if dim.contains(['+', '-', '/', '(', ')']) {
        return None;
    }
    let factors: Vec<&str> = dim.split('*').map(|f| f.trim()).collect();
    if factors.len() < 2 {
        return None;
    }
    let mut cancelled = false;
    let mut remaining: Vec<&str> = Vec::new();
    for f in factors {
        if !cancelled && f.parse::<usize>() == Ok(n) {
            cancelled = true;
            continue;
        }
        remaining.push(f);
    }
    if cancelled && !remaining.is_empty() {
        Some(remaining.join(" * "))
    } else {
        None
    }
}

pub fn compute_split_shapes(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<Vec<String>>>, String> {
    compute_split_shapes_impl(args, shapes, false)
}

pub fn compute_array_split_shapes(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<Vec<String>>>, String> {
    compute_split_shapes_impl(args, shapes, true)
}

fn compute_split_shapes_impl(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    array_split: bool,
) -> Result<Option<Vec<Vec<String>>>, String> {
    // ── input array ──────────────────────────────────────────────────
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.is_empty() {
        return Err("split requires rank >= 1 input".to_string());
    }
    let rank = input_shape.len();

    // ── axis ─────────────────────────────────────────────────────────
    let axis_raw = nth_positional_or_keyword(args, 2, &["axis", "dim"]).unwrap_or("0");
    let Some(axis) = parse_axis(axis_raw) else {
        return Ok(None);
    };
    let axis = normalize_axis(axis, rank, "split")?;

    // ── split specification (second positional or keyword) ───────────
    let Some(split_spec) =
        nth_positional_or_keyword(args, 1, &["indices_or_sections", "sections"])
    else {
        return Ok(None);
    };
    let trimmed_spec = split_spec.trim();

    // ── CASE 1: integer literal N ────────────────────────────────────
    if let Ok(n) = trimmed_spec.parse::<usize>() {
        if n == 0 {
            return Err("split requires N > 0".to_string());
        }
        let axis_dim = &input_shape[axis];
        if array_split {
            if let Ok(axis_size) = axis_dim.parse::<usize>() {
                let small = axis_size / n;
                let large = axis_size % n;
                return Ok(Some(
                    (0..n)
                        .map(|i| {
                            let mut out = input_shape.clone();
                            out[axis] = (small + usize::from(i < large)).to_string();
                            out
                        })
                        .collect(),
                ));
            }
            return if n == 1 {
                Ok(Some(vec![input_shape.clone()]))
            } else {
                Ok(None)
            };
        }

        if let Ok(axis_size) = axis_dim.parse::<usize>()
            && axis_size % n != 0
        {
            return Err(format!(
                "split cannot divide axis size {} evenly into {} parts",
                axis_size, n
            ));
        }

        // Synthetic dimension naming convention:
        //   numeric         → axis_size / N            (e.g. "2")
        //   symbolic "X*N"  → X via factor cancellation (e.g. "d_model*3"/3 → "d_model")
        //   other symbolic  → "split(<dim>, <N>)"      (e.g. "split(n, 2)")
        let chunk_dim = if let Ok(axis_size) = axis_dim.parse::<usize>() {
            (axis_size / n).to_string()
        } else if let Some(simplified) = cancel_product_factor(axis_dim, n) {
            simplified
        } else {
            format!("split({}, {})", axis_dim, n)
        };

        let mut output_shapes = Vec::with_capacity(n);
        for _ in 0..n {
            let mut out = input_shape.clone();
            out[axis] = chunk_dim.clone();
            output_shapes.push(out);
        }
        return Ok(Some(output_shapes));
    }

    // ── CASE 2: list literal of indices ──────────────────────────────
    if (trimmed_spec.starts_with('[') && trimmed_spec.ends_with(']'))
        || (trimmed_spec.starts_with('(') && trimmed_spec.ends_with(')'))
    {
        let indices = parse_simple_sequence_names(trimmed_spec).unwrap_or_default();
        if indices.is_empty() {
            return Ok(Some(vec![input_shape.clone()]));
        }

        let axis_dim = &input_shape[axis];
        let Some(total) = axis_dim.parse::<usize>().ok() else {
            // Approximate symbolic bounds as positive, ordered, and in-bounds.
            let difference = |left: &str, right: &str| {
                if right.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.') {
                    format!("{}-{}", left, right)
                } else {
                    format!("{}-({})", left, right)
                }
            };
            let mut previous = "0";
            let mut output = Vec::with_capacity(indices.len() + 1);
            for index in &indices {
                if index.starts_with('-') {
                    return Ok(None);
                }
                let mut part = input_shape.clone();
                part[axis] = match (previous.parse::<usize>(), index.parse::<usize>()) {
                    (Ok(start), Ok(end)) => end.saturating_sub(start).to_string(),
                    _ if previous == "0" => index.clone(),
                    _ => difference(index, previous),
                };
                output.push(part);
                previous = index;
            }
            let mut last = input_shape.clone();
            last[axis] = if previous == "0" {
                axis_dim.clone()
            } else {
                difference(axis_dim, previous)
            };
            output.push(last);
            return Ok(Some(output));
        };
        let Some(total_int) = i128::try_from(total).ok() else {
            return Ok(None);
        };
        let mut previous = 0usize;
        let mut output = Vec::with_capacity(indices.len() + 1);
        for index in indices {
            let Ok(index) = index.parse::<i128>() else {
                return Ok(None);
            };
            let boundary = if index < 0 {
                index.saturating_add(total_int)
            } else {
                index
            };
            let boundary = boundary.clamp(0, total_int) as usize;
            let mut part = input_shape.clone();
            part[axis] = boundary.saturating_sub(previous).to_string();
            output.push(part);
            previous = boundary;
        }
        let mut last = input_shape.clone();
        last[axis] = total.saturating_sub(previous).to_string();
        output.push(last);
        return Ok(Some(output));
    }

    // ── CASE 3: non-literal → out of scope ───────────────────────────
    Ok(None)
}

/// Apply real torch `split`/`Tensor.split` semantics — see
/// [`KnownFunction::TorchSplit`]. Unlike [`compute_array_split_shapes`]
/// (`np.array_split`/`torch.tensor_split` section-*count* semantics), torch's
/// `split_size_or_sections` is either:
///
/// * an integer literal `size`: the axis is cut into `ceil(axis_dim / size)`
///   chunks of `size`, with a smaller trailing remainder chunk when it
///   doesn't divide evenly (mirrors [`compute_chunk_shapes`]'s remainder
///   handling, just with the roles of "count" and "size" swapped); or
/// * a list literal of explicit per-chunk sizes (torch allows at most one
///   `-1` entry meaning "infer the remainder").
///
/// `n_targets` is the LHS tuple arity when known (`Some(k)` from `a, b, c =
/// x.split(...)`). It's only consulted for the literal-size /
/// symbolic-axis-dim combination: the chunk *count* can't be derived from an
/// unknown axis size, but it *is* implied by how many names the caller
/// unpacked into. Without that hint (`None`, e.g. the single-LHS validate
/// path), a symbolic axis dim with a literal size returns `Ok(None)`
/// (unknown count). A non-literal, non-list split spec is always `Ok(None)`
/// regardless of `n_targets` — the per-chunk size itself is unknown, so
/// nothing can be bound.
pub fn compute_torch_split_shapes(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    n_targets: Option<usize>,
) -> Result<Option<Vec<Vec<String>>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.is_empty() {
        return Err("split requires rank >= 1 input".to_string());
    }
    let rank = input_shape.len();

    // torch's real signature is `(input, split_size_or_sections, dim=0)` —
    // unlike `axis_arg` (shared by functions where the *only* other
    // positional slot is the axis), split has a meaningful positional at
    // index 1 (the size spec) before `dim`, so we look for `dim` starting
    // after position 2, not position 0.
    let axis_raw = nth_positional_or_keyword(args, 2, &["dim"]).unwrap_or("0");
    let Some(axis) = parse_axis(axis_raw) else {
        return Ok(None);
    };
    let axis = normalize_axis(axis, rank, "split")?;
    let axis_dim = &input_shape[axis];

    let Some(split_spec) =
        nth_positional_or_keyword(args, 1, &["split_size_or_sections", "split_size"])
    else {
        return Ok(None);
    };
    let trimmed_spec = split_spec.trim();

    // ── literal integer size ─────────────────────────────────────────
    if let Ok(size) = trimmed_spec.parse::<usize>() {
        if size == 0 {
            return Err("split requires split_size > 0".to_string());
        }

        if let Ok(axis_size) = axis_dim.parse::<usize>() {
            let count = axis_size.div_ceil(size).max(1);
            let mut output_shapes = Vec::with_capacity(count);
            let mut remaining = axis_size;
            for _ in 0..count {
                let this_size = remaining.min(size);
                let mut out = input_shape.clone();
                out[axis] = this_size.to_string();
                output_shapes.push(out);
                remaining -= this_size;
            }
            return Ok(Some(output_shapes));
        }

        // Symbolic axis dim: the chunk count can't be derived from the
        // shape alone — fall back to the LHS tuple arity, if known.
        let Some(k) = n_targets.filter(|k| *k > 0) else {
            return Ok(None);
        };
        let mut output_shapes = Vec::with_capacity(k);
        for _ in 0..k.saturating_sub(1) {
            let mut out = input_shape.clone();
            out[axis] = size.to_string();
            output_shapes.push(out);
        }
        let consumed = size * (k - 1);
        let mut last = input_shape.clone();
        last[axis] = if consumed == 0 {
            axis_dim.clone()
        } else {
            format!("{}-{}", axis_dim, consumed)
        };
        output_shapes.push(last);
        return Ok(Some(output_shapes));
    }

    // ── list literal of explicit per-chunk sizes ─────────────────────
    // Guarded by an explicit bracket check (unlike `compute_split_shapes`'s
    // CASE 2): `parse_simple_sequence_names` treats any comma-free string as
    // a one-element list, which would otherwise swallow a bare symbolic
    // split-size identifier (e.g. `x.split(k, dim=0)`) that should fall
    // through to "unknown" instead.
    let looks_like_sequence =
        trimmed_spec.starts_with('[') || trimmed_spec.starts_with('(');
    if looks_like_sequence
        && let Some(size_names) = parse_simple_sequence_names(trimmed_spec)
    {
        if size_names.is_empty() {
            return Ok(Some(vec![input_shape.clone()]));
        }

        let parsed: Vec<Option<isize>> = size_names
            .iter()
            .map(|s| s.trim().parse::<isize>().ok())
            .collect();
        let neg_one_count = parsed.iter().filter(|v| **v == Some(-1)).count();
        if neg_one_count > 1 {
            return Err("split accepts at most one -1 size entry".to_string());
        }

        let axis_total: Option<usize> = axis_dim.parse().ok();
        let known_sum: Option<usize> = if neg_one_count == 1 {
            parsed
                .iter()
                .filter(|v| **v != Some(-1))
                .try_fold(0usize, |acc, v| -> Option<usize> {
                    Some(acc + usize::try_from((*v)?).ok()?)
                })
        } else {
            None
        };

        let mut output_shapes = Vec::with_capacity(size_names.len());
        for (name, value) in size_names.iter().zip(parsed.iter()) {
            let mut out = input_shape.clone();
            out[axis] = match *value {
                Some(-1) => match (axis_total, known_sum) {
                    (Some(total), Some(sum)) if total >= sum => (total - sum).to_string(),
                    _ => format!("remainder({})", axis_dim),
                },
                Some(v) if v >= 0 => v.to_string(),
                _ => name.trim().to_string(),
            };
            output_shapes.push(out);
        }

        if neg_one_count == 0
            && let Some(total) = axis_total
            && parsed.iter().all(|v| v.is_some())
        {
            let sum: isize = parsed.iter().filter_map(|v| *v).sum();
            if sum != total as isize {
                return Err(format!(
                    "split sizes {:?} sum to {} but axis size is {}",
                    size_names, sum, total
                ));
            }
        }

        return Ok(Some(output_shapes));
    }

    // ── non-literal, non-list → out of scope ──────────────────────────
    Ok(None)
}

/// Force a fixed split axis for `hsplit`/`vsplit`/`dsplit` (numpy always
/// picks the axis from the function name, not from an argument) by
/// synthesizing an explicit `axis` keyword and delegating to
/// [`compute_split_shapes`].
pub fn compute_fixed_axis_split_shapes(
    kind: &KnownFunction,
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<Vec<String>>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let axis = match kind {
        KnownFunction::HSplit => {
            if rank <= 1 {
                0
            } else {
                1
            }
        }
        KnownFunction::VSplit => 0,
        KnownFunction::DSplit => {
            if rank < 3 {
                return Err(format!("dsplit requires rank >= 3, got rank {}", rank));
            }
            2
        }
        _ => return Ok(None),
    };
    let mut new_args = args.to_vec();
    new_args.push(CallArgument::Keyword {
        name: "axis".to_string(),
        value: axis.to_string(),
    });
    compute_split_shapes(&new_args, shapes)
}

/// Parse a literal like `"((1, 1, 0), (2, 2, 1))"` into a list of
/// same-shape int tuples (used by `jax.lax.pad`'s per-axis `(low, high,
/// interior)` triples and `jax.lax.reduce_window`'s per-axis `(low, high)`
/// padding pairs). Splits on top-level commas only, respecting nested
/// paren/bracket depth.
fn parse_nested_int_tuples(value: &str) -> Option<Vec<Vec<isize>>> {
    let trimmed = value.trim();
    let inner = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return None;
    };
    let mut groups = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in inner.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                if !current.trim().is_empty() {
                    groups.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        groups.push(current.trim().to_string());
    }
    if groups.is_empty() {
        return None;
    }
    Some(groups.iter().map(|g| parse_ints(g)).collect())
}

/// Shared "output equals the shape of one named/positional carry argument"
/// rule for `jax.lax.while_loop`/`fori_loop` (carry invariant across
/// iterations) and `jax.lax.associative_scan` (shape-preserving scan
/// variant).
fn apply_known_lax_carry(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    skip: usize,
    keyword: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(name) = nth_positional_or_keyword(args, skip, &[keyword]) else {
        return Ok(None);
    };
    Ok(shapes.shape(name).cloned())
}

/// `jax.lax.broadcast(operand, sizes)` — prepends `sizes` as new leading
/// dims.
fn apply_known_lax_broadcast(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(sizes) = nth_positional_or_keyword(args, 1, &["sizes"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    let mut output = sizes;
    output.extend(input_shape.clone());
    Ok(Some(output))
}

/// `jax.lax.broadcast_in_dim(operand, shape, broadcast_dimensions)` — the
/// output is the explicit target `shape` argument; `operand`'s own shape
/// isn't needed to determine it.
fn apply_known_lax_broadcast_in_dim(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(target) = nth_positional_or_keyword(args, 1, &["shape"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    Ok(Some(target))
}

/// `jax.lax.slice(operand, start_indices, limit_indices, strides=None)`.
/// Concrete (literal) indices/strides only.
fn apply_known_lax_slice(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let (start, limit) =
        nth_two_positional_or_keywords(args, 1, &["start_indices"], &["limit_indices"]);
    let Some(start) = start.map(parse_ints) else {
        return Ok(None);
    };
    let Some(limit) = limit.map(parse_ints) else {
        return Ok(None);
    };
    if start.len() != input_shape.len() || limit.len() != input_shape.len() {
        return Ok(None);
    }
    let strides = nth_positional_or_keyword(args, 3, &["strides"]).map(parse_ints);
    let strides = match strides {
        Some(s) if s.len() == input_shape.len() => s,
        Some(_) => return Ok(None),
        None => vec![1; input_shape.len()],
    };
    let mut output = Vec::with_capacity(input_shape.len());
    for i in 0..input_shape.len() {
        let size = limit[i] - start[i];
        if size < 0 {
            return Err(format!(
                "lax.slice: limit_indices[{}]={} < start_indices[{}]={}",
                i, limit[i], i, start[i]
            ));
        }
        let stride = strides[i].max(1);
        output.push((((size - 1) / stride) + 1).to_string());
    }
    Ok(Some(output))
}

/// `jax.lax.dynamic_slice(operand, start_indices, slice_sizes)` — the
/// output shape is exactly `slice_sizes`.
fn apply_known_lax_dynamic_slice(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(sizes) =
        nth_positional_or_keyword(args, 2, &["slice_sizes"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    if sizes.len() != input_shape.len() {
        return Err(format!(
            "lax.dynamic_slice: slice_sizes length {} does not match operand rank {}",
            sizes.len(),
            input_shape.len()
        ));
    }
    Ok(Some(sizes))
}

/// `jax.lax.pad(operand, padding_value, padding_config)` — `padding_config`
/// is a per-axis `(low, high, interior)` triple; `interior` inserts values
/// *between* existing elements, so the growth is
/// `low + high + interior * (dim - 1)` for a numeric dim (numeric `dim` is
/// required whenever `interior != 0`; symbolic dims with `interior == 0`
/// still resolve via a plain additive expression).
fn apply_known_lax_pad(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(config_raw) = nth_positional_or_keyword(args, 2, &["padding_config"]) else {
        return Ok(None);
    };
    let Some(config) = parse_nested_int_tuples(config_raw) else {
        return Ok(None);
    };
    if config.len() != input_shape.len() {
        return Err(format!(
            "lax.pad: padding_config length {} does not match operand rank {}",
            config.len(),
            input_shape.len()
        ));
    }
    let mut output = Vec::with_capacity(input_shape.len());
    for (dim, triple) in input_shape.iter().zip(config.iter()) {
        let [lo, hi, interior] = triple.as_slice() else {
            return Ok(None);
        };
        if let Ok(d) = dim.parse::<isize>() {
            let total = d + lo + hi + interior * (d - 1).max(0);
            output.push(total.to_string());
        } else if *interior == 0 {
            output.push(add_to_dim(dim, lo + hi));
        } else {
            return Ok(None);
        }
    }
    Ok(Some(output))
}

/// `jax.lax.reduce_window(operand, init_value, computation,
/// window_dimensions, window_strides, padding)` — pooling-style output
/// shape, concrete window/strides/padding only. `'VALID'` and `'SAME'`
/// string padding are recognised; explicit per-axis `(low, high)` pairs are
/// parsed too. Other padding modes bail.
fn apply_known_lax_reduce_window(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(window) = nth_positional_or_keyword(args, 3, &["window_dimensions"]).map(parse_ints)
    else {
        return Ok(None);
    };
    let Some(strides) = nth_positional_or_keyword(args, 4, &["window_strides"]).map(parse_ints)
    else {
        return Ok(None);
    };
    if window.len() != input_shape.len() || strides.len() != input_shape.len() {
        return Ok(None);
    }
    let padding_raw = nth_positional_or_keyword(args, 5, &["padding"]);
    let padding: Vec<(isize, isize)> = match padding_raw.map(str::trim) {
        Some("'VALID'") | Some("\"VALID\"") | None => vec![(0, 0); input_shape.len()],
        Some(raw) => match parse_nested_int_tuples(raw) {
            Some(p) if p.len() == input_shape.len() && p.iter().all(|t| t.len() == 2) => {
                p.iter().map(|t| (t[0], t[1])).collect()
            }
            _ => return Ok(None), // e.g. 'SAME' or an unrecognised spec
        },
    };
    let mut output = Vec::with_capacity(input_shape.len());
    for i in 0..input_shape.len() {
        let Ok(d) = input_shape[i].parse::<isize>() else {
            return Ok(None);
        };
        let (lo, hi) = padding[i];
        let w = window[i];
        let s = strides[i].max(1);
        let padded = d + lo + hi;
        if padded < w {
            return Err(format!(
                "reduce_window: padded dim {} smaller than window {}",
                padded, w
            ));
        }
        output.push((((padded - w) / s) + 1).to_string());
    }
    Ok(Some(output))
}

/// `jax.lax.conv_general_dilated(lhs, rhs, window_strides, padding,
/// lhs_dilation=None, rhs_dilation=None, dimension_numbers=None, ...)`.
/// Only the *default* dimension_numbers case is modelled: `lhs` is
/// `(batch, in_channel, *spatial)`, `rhs` is `(out_channel, in_channel,
/// *spatial)`, output is `(batch, out_channel, *spatial)` — i.e. any
/// explicit `dimension_numbers` bails. `rhs_dilation` (atrous conv) is
/// supported; `lhs_dilation` (transposed-conv input dilation) is not.
fn apply_known_lax_conv_general_dilated(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if args
        .iter()
        .any(|a| matches!(a, CallArgument::Keyword { name, .. } if name == "dimension_numbers"))
    {
        return Ok(None);
    }
    let Some(lhs_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(lhs_shape) = shapes.shape(lhs_name) else {
        return Ok(None);
    };
    let Some(rhs_name) = nth_positional_or_keyword(args, 1, &["rhs"]) else {
        return Ok(None);
    };
    let Some(rhs_shape) = shapes.shape(rhs_name) else {
        return Ok(None);
    };
    if lhs_shape.len() < 3 || rhs_shape.len() != lhs_shape.len() {
        return Ok(None);
    }
    let spatial_rank = lhs_shape.len() - 2;

    let Some(strides) = nth_positional_or_keyword(args, 2, &["window_strides"]).map(parse_ints)
    else {
        return Ok(None);
    };
    if strides.len() != spatial_rank {
        return Ok(None);
    }

    if let Some(d) = nth_positional_or_keyword(args, 4, &["lhs_dilation"])
        && parse_ints(d).iter().any(|v| *v != 1)
    {
        return Ok(None); // input dilation (transposed conv) not modelled
    }
    let dilation = match nth_positional_or_keyword(args, 5, &["rhs_dilation"]).map(parse_ints) {
        Some(d) if d.len() == spatial_rank => d,
        Some(_) => return Ok(None),
        None => vec![1; spatial_rank],
    };

    let padding_raw = nth_positional_or_keyword(args, 3, &["padding"]);
    let same = matches!(padding_raw.map(str::trim), Some("'SAME'") | Some("\"SAME\""));
    let padding: Option<Vec<(isize, isize)>> = if same {
        None
    } else {
        match padding_raw.and_then(parse_nested_int_tuples) {
            Some(p) if p.len() == spatial_rank && p.iter().all(|t| t.len() == 2) => {
                Some(p.iter().map(|t| (t[0], t[1])).collect())
            }
            _ if matches!(padding_raw.map(str::trim), Some("'VALID'") | Some("\"VALID\"")) => {
                Some(vec![(0, 0); spatial_rank])
            }
            _ => return Ok(None),
        }
    };

    let mut output = Vec::with_capacity(lhs_shape.len());
    output.push(lhs_shape[0].clone()); // batch
    output.push(rhs_shape[0].clone()); // out channels
    for i in 0..spatial_rank {
        let Ok(d) = lhs_shape[2 + i].parse::<isize>() else {
            return Ok(None);
        };
        let Ok(k) = rhs_shape[2 + i].parse::<isize>() else {
            return Ok(None);
        };
        let eff_k = (k - 1) * dilation[i] + 1;
        let s = strides[i].max(1);
        let out = match &padding {
            Some(pads) => {
                let (lo, hi) = pads[i];
                ((d + lo + hi - eff_k) / s) + 1
            }
            None => (d + s - 1) / s, // 'SAME': ceil(d / s)
        };
        output.push(out.to_string());
    }
    Ok(Some(output))
}

/// Extract the parenthesized/bracketed literal-int list that follows
/// `keyword=` inside `text` (e.g. `"...offset_dims=(1, 2)..."` -> `Some([1,
/// 2])`). Returns `None` if `keyword=` isn't found, or isn't immediately
/// followed (after whitespace) by a balanced `(...)`/`[...]` group — a bare
/// identifier/expression there means the value isn't a literal we can read
/// statically. `keyword` must not be a substring of another accepted keyword
/// in the same caller (true for `offset_dims`/`collapsed_slice_dims`/
/// `start_index_map`), since this does a plain substring search.
fn extract_paren_ints_after(text: &str, keyword: &str) -> Option<Vec<isize>> {
    let marker = format!("{keyword}=");
    let after = text.find(&marker)? + marker.len();
    let rest = text[after..].trim_start();
    let (open, close) = match rest.chars().next()? {
        '(' => ('(', ')'),
        '[' => ('[', ']'),
        _ => return None,
    };
    let mut depth = 0i32;
    let mut end = None;
    for (i, ch) in rest.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                end = Some(i);
                break;
            }
        }
    }
    Some(parse_ints(&rest[..=end?]))
}

/// Compute `jax.lax.gather`'s output shape per the XLA gather spec, for the
/// common statically-derivable case: `offset_dims`, `collapsed_slice_dims`,
/// and `slice_sizes` are known integer lists (from inline literal tuples at
/// the call site — see `apply_known_lax_gather`). `start_index_map` doesn't
/// affect the output *shape* (it only says which operand axis each index
/// component targets), so it's required to be present as a literal (see the
/// caller) but not threaded through here.
///
/// - `output_rank = len(offset_dims) + (start_indices.rank - 1)`: the batch
///   rank is everything but `start_indices`' trailing "index vector" axis.
/// - Output axes listed in `offset_dims` take, in order, the `slice_sizes`
///   entries whose operand axis isn't in `collapsed_slice_dims`.
/// - Every other output axis takes, in order, one of `start_indices`'
///   leading (batch) dims.
fn compute_lax_gather_shape(
    operand_shape: &[String],
    indices_shape: &[String],
    offset_dims: &[isize],
    collapsed_slice_dims: &[isize],
    slice_sizes: &[String],
) -> Option<Vec<String>> {
    if slice_sizes.len() != operand_shape.len() {
        return None;
    }
    let indices_rank = indices_shape.len();
    if indices_rank == 0 {
        return None;
    }
    let batch_rank = indices_rank - 1;
    let output_rank = offset_dims.len() + batch_rank;

    let kept_slice_sizes: Vec<String> = slice_sizes
        .iter()
        .enumerate()
        .filter(|(i, _)| !collapsed_slice_dims.contains(&(*i as isize)))
        .map(|(_, s)| s.clone())
        .collect();
    if kept_slice_sizes.len() != offset_dims.len() {
        return None;
    }

    let batch_dims_shape = &indices_shape[..batch_rank];
    let mut output = Vec::with_capacity(output_rank);
    let mut batch_cursor = 0usize;
    for p in 0..output_rank {
        if let Some(k) = offset_dims.iter().position(|&d| d == p as isize) {
            output.push(kept_slice_sizes[k].clone());
        } else {
            output.push(batch_dims_shape.get(batch_cursor)?.clone());
            batch_cursor += 1;
        }
    }
    Some(output)
}

/// `jax.lax.gather(operand, start_indices, dimension_numbers, slice_sizes,
/// ...)`. Only the case where `dimension_numbers` is an inline
/// `jax.lax.GatherDimensionNumbers(offset_dims=(...),
/// collapsed_slice_dims=(...), start_index_map=(...))` literal (all three
/// fields literal int tuples) and `slice_sizes` is itself an inline literal
/// int tuple is modelled; anything else (a variable, a partially-literal
/// tuple, `operand_batching_dims`/`start_indices_batching_dims`) falls back
/// to `Ok(None)`. See `compute_lax_gather_shape` for the shape formula.
fn apply_known_lax_gather(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(operand_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(operand_shape) = shapes.shape(operand_name) else {
        return Ok(None);
    };
    let Some(indices_name) = nth_positional_or_keyword(args, 1, &["start_indices"]) else {
        return Ok(None);
    };
    let Some(indices_shape) = shapes.shape(indices_name) else {
        return Ok(None);
    };
    let Some(dimension_numbers_raw) =
        nth_positional_or_keyword(args, 2, &["dimension_numbers"])
    else {
        return Ok(None);
    };
    let Some(offset_dims) = extract_paren_ints_after(dimension_numbers_raw, "offset_dims") else {
        return Ok(None);
    };
    let Some(collapsed_slice_dims) =
        extract_paren_ints_after(dimension_numbers_raw, "collapsed_slice_dims")
    else {
        return Ok(None);
    };
    // Required to be present as an inline literal (part of the "fully
    // statically-specified" gather form we accept) even though its values
    // don't feed the shape formula.
    if extract_paren_ints_after(dimension_numbers_raw, "start_index_map").is_none() {
        return Ok(None);
    }

    let Some(slice_sizes_raw) = nth_positional_or_keyword(args, 3, &["slice_sizes"]) else {
        return Ok(None);
    };
    let trimmed = slice_sizes_raw.trim();
    if !((trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
    {
        return Ok(None);
    }
    let slice_sizes_ints = parse_ints(slice_sizes_raw);
    if slice_sizes_ints.len() != operand_shape.len() {
        return Ok(None);
    }
    let slice_sizes: Vec<String> = slice_sizes_ints.iter().map(isize::to_string).collect();

    Ok(compute_lax_gather_shape(
        operand_shape,
        indices_shape,
        &offset_dims,
        &collapsed_slice_dims,
        &slice_sizes,
    ))
}

/// `jnp.diagflat(v, k=0)` — flattens `v` then builds a square diagonal
/// matrix. `k` (off-diagonal offset) is ignored (v1: `k=0` sizing only).
fn apply_known_diagflat(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let n = flattened_dim(input_shape);
    Ok(Some(vec![n.clone(), n]))
}

/// `jnp.tri(N, M=None, k=0)` — shape `(N, M or N)`.
fn apply_known_tri(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(n) = nth_positional_or_keyword(args, 0, &["N"]) else {
        return Ok(None);
    };
    let m = nth_positional_or_keyword(args, 1, &["M"]).unwrap_or(n);
    Ok(Some(vec![n.to_string(), m.to_string()]))
}

/// `jnp.indices(dimensions)` — prepends a rank dimension:
/// `indices((2, 3)).shape == (2, 2, 3)`.
fn apply_known_indices(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(dims_raw) = nth_positional_or_keyword(args, 0, &["dimensions"]) else {
        return Ok(None);
    };
    let Some(dims) = parse_shape_value(dims_raw) else {
        return Ok(None);
    };
    let mut output = vec![dims.len().to_string()];
    output.extend(dims);
    Ok(Some(output))
}

/// `jnp.select(condlist, choicelist, default=0)` — approximated as the
/// shape of `choicelist`'s first array (all choices are expected to share
/// a broadcast-compatible shape).
fn apply_known_select(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(choicelist_raw) = nth_positional_or_keyword(args, 1, &["choicelist"]) else {
        return Ok(None);
    };
    let Some(names) = parse_simple_sequence_names(choicelist_raw) else {
        return Ok(None);
    };
    let Some(first) = names.first() else {
        return Ok(None);
    };
    Ok(shapes.shape(first).cloned())
}

/// `jnp.rollaxis(a, axis, start=0)` — rolls `axis` backward until it lies
/// before position `start`.
fn apply_known_rollaxis(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let Some(axis) = nth_positional_or_keyword(args, 1, &["axis"]).and_then(parse_axis) else {
        return Ok(None);
    };
    let start = nth_positional_or_keyword(args, 2, &["start"])
        .and_then(parse_axis)
        .unwrap_or(0);
    let axis = normalize_axis(axis, rank, "rollaxis")?;
    let start_norm = if start < 0 {
        (rank as isize + start).max(0) as usize
    } else {
        (start as usize).min(rank)
    };
    let dest = if axis < start_norm {
        start_norm - 1
    } else {
        start_norm
    };
    let mut order: Vec<usize> = (0..rank).filter(|&i| i != axis).collect();
    let dest = dest.min(order.len());
    order.insert(dest, axis);
    Ok(Some(
        order.iter().map(|&i| input_shape[i].clone()).collect(),
    ))
}

/// `jnp.resize(a, new_shape)` — output is exactly `new_shape` (tiling or
/// truncating the data as needed doesn't change the *shape* rule).
fn apply_known_resize(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(target) = nth_positional_or_keyword(args, 1, &["new_shape"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    Ok(Some(target))
}

/// `np.insert(arr, obj, values, axis=None)` — axis length grows by the
/// number of inserted values. Only the `axis` given + concrete insertion
/// count case is modelled (a scalar literal inserts 1; a known array/list
/// contributes its own axis-length/element-count); `axis=None` (flattening)
/// is data-shape-dependent and bails.
fn apply_known_insert(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(axis_raw) = nth_positional_or_keyword(args, 3, &["axis"]) else {
        return Ok(None);
    };
    let Some(axis) = parse_axis(axis_raw) else {
        return Ok(None);
    };
    let axis = normalize_axis(axis, input_shape.len(), "insert")?;
    let Some(values_raw) = nth_positional_or_keyword(args, 2, &["values"]) else {
        return Ok(None);
    };
    let count: isize = if let Some(shape) = shapes.shape(values_raw) {
        match shape.get(axis).and_then(|d| d.parse::<isize>().ok()) {
            Some(n) => n,
            None => return Ok(None),
        }
    } else if values_raw.trim().parse::<f64>().is_ok() {
        1
    } else if let Some(items) = parse_simple_sequence_names(values_raw) {
        items.len() as isize
    } else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output[axis] = add_to_dim(&output[axis], count);
    Ok(Some(output))
}

/// `np.delete(arr, obj, axis=None)` — axis length shrinks by the number of
/// removed indices. Only a single-int `obj` (removes 1) or a literal list
/// of indices (removes its length) is modelled; mask-based deletion is
/// data-dependent and bails.
fn apply_known_delete(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(axis_raw) = nth_positional_or_keyword(args, 2, &["axis"]) else {
        return Ok(None);
    };
    let Some(axis) = parse_axis(axis_raw) else {
        return Ok(None);
    };
    let axis = normalize_axis(axis, input_shape.len(), "delete")?;
    let Some(obj_raw) = nth_positional_or_keyword(args, 1, &["obj"]) else {
        return Ok(None);
    };
    let trimmed = obj_raw.trim();
    let count: isize = if trimmed.parse::<isize>().is_ok() {
        1
    } else if let Some(items) = parse_simple_sequence_names(trimmed) {
        items.len() as isize
    } else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output[axis] = add_to_dim(&output[axis], -count);
    Ok(Some(output))
}

/// `jnp.append(arr, values, axis=None)` — concatenates like
/// `jnp.concatenate` when `axis` is given; flattens both operands to 1D and
/// sums their lengths when `axis` is omitted.
fn apply_known_append(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 2 {
        return Ok(None);
    }
    let Some(arr_shape) = shapes.shape(&values[0]) else {
        return Ok(None);
    };
    let Some(val_shape) = shapes.shape(&values[1]) else {
        return Ok(None);
    };
    let axis_raw = nth_positional_or_keyword(args, 2, &["axis"]);
    match axis_raw {
        None => {
            let a = flattened_dim(arr_shape);
            let b = flattened_dim(val_shape);
            Ok(Some(vec![concat_dim(&[a, b])]))
        }
        Some(raw) => {
            let Some(axis) = parse_axis(raw) else {
                return Ok(None);
            };
            let rank = arr_shape.len();
            let axis = if axis < 0 { rank as isize + axis } else { axis };
            if axis < 0 || axis as usize >= rank {
                return Err(format!("append axis {} out of bounds for rank {}", axis, rank));
            }
            concat_shapes_along_axis(&[arr_shape.clone(), val_shape.clone()], axis as usize)
        }
    }
}

/// `jnp.kron(a, b)` — Kronecker product: elementwise product of dims after
/// left-padding the shorter shape with size-1 axes.
fn apply_known_kron(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, a_shape, _, b_shape)) = resolve_binary_shapes(args, shapes) else {
        return Ok(None);
    };
    let rank = a_shape.len().max(b_shape.len());
    let mut a = vec!["1".to_string(); rank - a_shape.len()];
    a.extend(a_shape.clone());
    let mut b = vec!["1".to_string(); rank - b_shape.len()];
    b.extend(b_shape.clone());
    Ok(Some(
        a.iter().zip(b.iter()).map(|(x, y)| multiply_dim(x, y)).collect(),
    ))
}

/// A leaf array name, or a nested list of blocks, parsed from `np.block`'s
/// nested-list literal argument.
enum BlockItem {
    Leaf(String),
    List(Vec<BlockItem>),
}

fn parse_block_item(value: &str) -> Option<BlockItem> {
    let trimmed = value.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut items = Vec::new();
        let mut depth = 0i32;
        let mut current = String::new();
        for ch in inner.chars() {
            match ch {
                '[' => {
                    depth += 1;
                    current.push(ch);
                }
                ']' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        items.push(current.trim().to_string());
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.trim().is_empty() {
            items.push(current.trim().to_string());
        }
        if items.is_empty() {
            return None;
        }
        let parsed: Option<Vec<BlockItem>> = items.iter().map(|s| parse_block_item(s)).collect();
        Some(BlockItem::List(parsed?))
    } else if !trimmed.is_empty()
        && trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        Some(BlockItem::Leaf(trimmed.to_string()))
    } else {
        None
    }
}

/// Recursively resolves a `BlockItem`, returning its shape and the nesting
/// depth (list-levels) below it (`0` for a leaf). numpy concatenates the
/// innermost list level along the last axis, the next level up along the
/// second-to-last axis, and so on — i.e. axis = `rank - this_depth`.
fn block_item_shape(
    item: &BlockItem,
    shapes: &dyn ShapeLookup,
) -> Result<(Option<Vec<String>>, usize), String> {
    match item {
        BlockItem::Leaf(name) => Ok((shapes.shape(name).cloned(), 0)),
        BlockItem::List(children) => {
            let mut child_shapes = Vec::with_capacity(children.len());
            let mut depth = 0usize;
            for child in children {
                let (shape, d) = block_item_shape(child, shapes)?;
                depth = depth.max(d);
                match shape {
                    Some(s) => child_shapes.push(s),
                    None => return Ok((None, depth + 1)),
                }
            }
            let Some(first) = child_shapes.first() else {
                return Ok((None, depth + 1));
            };
            let rank = first.len();
            let this_depth = depth + 1;
            if this_depth > rank {
                return Err(format!(
                    "block: nesting depth {} exceeds operand rank {}",
                    this_depth, rank
                ));
            }
            let axis = rank - this_depth;
            let combined = concat_shapes_along_axis(&child_shapes, axis)?;
            Ok((combined, this_depth))
        }
    }
}

fn apply_known_block(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(raw) = positional_arg_values(args).into_iter().next() else {
        return Ok(None);
    };
    let Some(item) = parse_block_item(&raw) else {
        return Ok(None);
    };
    let (shape, _depth) = block_item_shape(&item, shapes)?;
    Ok(shape)
}

/// `jnp.take_along_axis(arr, indices, axis)` — output shape matches
/// `indices` exactly.
fn apply_known_take_along_axis(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    let Some(idx_name) = values.get(1) else {
        return Ok(None);
    };
    Ok(shapes.shape(idx_name).cloned())
}

/// `np.argwhere(a)` — shape `(N, ndim)`: the element count `N` is
/// data-dependent (an opaque symbolic dim), but `ndim` is known statically
/// from `a`'s rank.
fn apply_known_argwhere(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    Ok(Some(vec![
        format!("nonzero({input_name})"),
        input_shape.len().to_string(),
    ]))
}

/// `jnp.searchsorted(a, v, ...)` — output shape follows `v` (the values
/// being searched for), not `a`.
fn apply_known_searchsorted(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    let Some(v_name) = values.get(1) else {
        return Ok(None);
    };
    Ok(shapes.shape(v_name).cloned())
}

/// `jnp.histogram(a, bins=10, ...)` — 1D output of length `bins` when
/// `bins` is an integer literal, or `len(edges) - 1` when `bins` is a
/// literal list of bin edges. Otherwise unresolvable statically.
fn apply_known_histogram(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    let Some(bins_raw) = nth_positional_or_keyword(args, 1, &["bins"]) else {
        return Ok(Some(vec!["10".to_string()])); // numpy/jax default
    };
    let trimmed = bins_raw.trim();
    if let Ok(n) = trimmed.parse::<usize>() {
        return Ok(Some(vec![n.to_string()]));
    }
    let looks_like_list = (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'));
    if looks_like_list
        && let Some(edges) = parse_simple_sequence_names(trimmed)
        && edges.len() >= 2
    {
        return Ok(Some(vec![(edges.len() - 1).to_string()]));
    }
    Ok(None)
}

fn apply_known_cross(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(a) = nth_positional_or_keyword(args, 0, &["a"]).and_then(|name| shapes.shape(name))
    else {
        return Ok(None);
    };
    let Some(b) = nth_positional_or_keyword(args, 1, &["b"]).and_then(|name| shapes.shape(name))
    else {
        return Ok(None);
    };
    let axis = nth_positional_or_keyword(args, 5, &["axis"]);
    let axis_override = match axis.filter(|axis| *axis != "None") {
        Some(value) => match parse_axis(value) {
            Some(axis) => Some(axis),
            None => return Ok(None),
        },
        None => None,
    };
    let mut vector_axes = Vec::new();
    for (shape, position, keyword) in [(a, 2, "axisa"), (b, 3, "axisb")] {
        let raw = nth_positional_or_keyword(args, position, &[keyword]).unwrap_or("-1");
        let Some(axis) = axis_override.or_else(|| parse_axis(raw)) else {
            return Ok(None);
        };
        let axis = normalize_axis(axis, shape.len(), "cross")?;
        match shape[axis].parse::<usize>() {
            Ok(3) => vector_axes.push(axis),
            Ok(_) => return Err("cross requires vector dimensions of size 3".to_string()),
            Err(_) => return Ok(None),
        }
    }
    let mut a_batch = a.clone();
    let mut b_batch = b.clone();
    a_batch.remove(vector_axes[0]);
    b_batch.remove(vector_axes[1]);
    let mut result = broadcast_two_shapes(&a_batch, &b_batch)?;
    let raw = nth_positional_or_keyword(args, 4, &["axisc"]).unwrap_or("-1");
    let Some(output_axis) = axis_override.or_else(|| parse_axis(raw)) else {
        return Ok(None);
    };
    let output_axis = normalize_axis(output_axis, result.len() + 1, "cross")?;
    result.insert(output_axis, "3".to_string());
    Ok(Some(result))
}

fn apply_known_torch_cross(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(a) =
        nth_positional_or_keyword(args, 0, &["input"]).and_then(|name| shapes.shape(name))
    else {
        return Ok(None);
    };
    let Some(b) =
        nth_positional_or_keyword(args, 1, &["other"]).and_then(|name| shapes.shape(name))
    else {
        return Ok(None);
    };
    if a.len() != b.len() {
        return Err("torch.cross inputs must have the same number of dimensions".to_string());
    }
    let axis = match nth_positional_or_keyword(args, 2, &["dim"]) {
        None | Some("None") => match a.iter().position(|dim| dim == "3") {
            Some(axis) if a[..axis].iter().all(|dim| dim.parse::<usize>().is_ok()) => axis,
            _ if a.iter().any(|dim| dim.parse::<usize>().is_err()) => return Ok(None),
            _ => return Err("torch.cross requires vector dimensions of size 3".to_string()),
        },
        Some(value) => match parse_axis(value) {
            Some(axis) => normalize_axis(axis, a.len(), "torch.cross")?,
            None => return Ok(None),
        },
    };
    for dim in [&a[axis], &b[axis]] {
        match dim.parse::<usize>() {
            Ok(3) => {}
            Ok(_) => return Err("torch.cross requires vector dimensions of size 3".to_string()),
            Err(_) => return Ok(None),
        }
    }
    broadcast_two_shapes(a, b).map(Some)
}

/// `jnp.linalg.solve(a, b)` — output shape follows `b` (the right-hand
/// side); `a` is validated as square.
fn apply_known_linalg_solve(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, a_shape, _, b_shape)) = resolve_binary_shapes(args, shapes) else {
        return Ok(None);
    };
    require_square_matrix(a_shape, "linalg.solve")?;
    Ok(Some(b_shape.clone()))
}

/// `torch.linalg.pinv(A)` — pseudo-inverse swaps the last two dims:
/// `(..., m, n) -> (..., n, m)`.
fn apply_known_linalg_pinv(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "linalg.pinv requires rank >= 2, got rank {}",
            input_shape.len()
        ));
    }
    let mut output = input_shape.clone();
    let len = output.len();
    output.swap(len - 1, len - 2);
    Ok(Some(output))
}

/// `torch.linalg.matrix_rank(A)` — batched: reduces the last two dims to a
/// scalar per batch element, `(..., m, n) -> (...,)`. No square requirement
/// (rank is defined for non-square matrices too).
fn apply_known_linalg_matrix_rank(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() < 2 {
        return Err(format!(
            "linalg.matrix_rank requires rank >= 2, got rank {}",
            input_shape.len()
        ));
    }
    Ok(Some(input_shape[..input_shape.len() - 2].to_vec()))
}

/// `torch.linalg.lstsq(A, B)` — solution shape is `A`'s batch dims + `[n,
/// (k)]` where `n = A`'s last dim and `k = B`'s last dim (only present when
/// `B` is a matrix rhs, i.e. same rank as `A`; a vector rhs — one rank
/// lower — yields a vector solution).
pub fn apply_known_linalg_lstsq_solution(
    a_shape: &[String],
    b_shape: &[String],
) -> Option<Vec<String>> {
    if a_shape.len() < 2 || b_shape.is_empty() {
        return None;
    }
    let n = a_shape[a_shape.len() - 1].clone();
    let mut solution = a_shape[..a_shape.len() - 2].to_vec();
    if b_shape.len() == a_shape.len() {
        solution.push(n);
        solution.push(b_shape.last()?.clone());
    } else if b_shape.len() + 1 == a_shape.len() {
        solution.push(n);
    } else {
        return None;
    }
    Some(solution)
}

/// `einops.einsum(*operands, pattern)` — same equation-based semantics as
/// `torch.einsum`/`jnp.einsum`, but the pattern string is the *last*
/// positional argument rather than the first.
fn apply_known_einops_einsum(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let positional_values = positional_arg_values(args);
    if positional_values.len() < 2 {
        return Ok(None);
    }
    let (equation, operand_names) = positional_values.split_last().unwrap();

    let trimmed = equation.trim();
    let equation_str = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return Ok(None);
    };
    if equation_str.contains("...") {
        return Ok(None);
    }
    let Some((inputs_part, output_part)) = equation_str.split_once("->") else {
        return Ok(None);
    };

    let input_specs: Vec<Vec<&str>> = inputs_part
        .split(',')
        .map(|s| s.split_whitespace().collect())
        .collect();
    let output_spec: Vec<&str> = output_part.split_whitespace().collect();

    if operand_names.len() != input_specs.len() {
        return Err(format!(
            "einops.einsum equation has {} input specs but got {} operands",
            input_specs.len(),
            operand_names.len()
        ));
    }

    let mut label_map: HashMap<&str, String> = HashMap::new();
    for (spec, operand_name) in input_specs.iter().zip(operand_names.iter()) {
        let Some(shape) = shapes.shape(operand_name.as_str()) else {
            return Ok(None);
        };
        if shape.len() != spec.len() {
            return Err(format!(
                "einops.einsum operand '{}' has rank {} but subscript '{}' has {} axes",
                operand_name,
                shape.len(),
                spec.join(" "),
                spec.len()
            ));
        }
        for (label, dim) in spec.iter().zip(shape.iter()) {
            if let Some(existing_dim) = label_map.get(label) {
                check_dim_match(
                    existing_dim,
                    dim,
                    &format!("einops.einsum label '{}'", label),
                )?;
            } else {
                label_map.insert(label, dim.clone());
            }
        }
    }

    let mut output_shape = Vec::with_capacity(output_spec.len());
    for label in output_spec {
        let Some(dim) = label_map.get(label) else {
            return Err(format!(
                "einops.einsum output label '{}' not found in input subscripts",
                label
            ));
        };
        output_shape.push(dim.clone());
    }
    Ok(Some(output_shape))
}

/// `einops.pack(tensors, pattern)` — restricted to patterns where `*`
/// stands for exactly one axis per tensor (the common case: a single
/// variable batch axis). Under that restriction, packing is exactly
/// `concatenate` along the `*` axis position; each tensor's rank must
/// match the pattern's token count. Patterns with more than one `*`, or
/// where `*` spans a variable number of axes per tensor, aren't modelled.
pub fn compute_einops_pack_shape(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let positional = positional_arg_values(args);
    let Some(tensors_raw) = positional.first() else {
        return Ok(None);
    };
    let Some(pattern_raw) = positional.get(1) else {
        return Ok(None);
    };
    let trimmed = pattern_raw.trim();
    let pattern = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return Ok(None);
    };
    let tokens: Vec<&str> = pattern.split_whitespace().collect();
    if tokens.iter().filter(|t| **t == "*").count() != 1 {
        return Ok(None);
    }
    let star_pos = tokens.iter().position(|t| *t == "*").unwrap();

    let Some(names) = parse_simple_sequence_names(tensors_raw) else {
        return Ok(None);
    };
    if names.is_empty() {
        return Ok(None);
    }

    let mut input_shapes = Vec::with_capacity(names.len());
    for name in &names {
        let Some(shape) = shapes.shape(name) else {
            return Ok(None);
        };
        if shape.len() != tokens.len() {
            return Ok(None);
        }
        input_shapes.push(shape.clone());
    }
    concat_shapes_along_axis(&input_shapes, star_pos)
}

/// `jax.nn.one_hot(x, num_classes, ...)` — appends `num_classes` to `x`'s
/// shape.
fn apply_known_one_hot(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(num_classes) = nth_positional_or_keyword(args, 1, &["num_classes"]) else {
        return Ok(None);
    };
    // torch.nn.functional.one_hot's default/sentinel `num_classes=-1` means
    // "infer from the max index at runtime" — data-dependent, not a literal.
    if num_classes.trim() == "-1" {
        return Ok(None);
    }
    let mut output = input_shape.clone();
    output.push(num_classes.to_string());
    Ok(Some(output))
}

/// `jax.nn.dot_product_attention(query, key, value, ...)` — output has
/// `query`'s shape with the last (head) dim replaced by `value`'s head dim.
fn apply_known_dot_product_attention(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let values = positional_arg_values(args);
    if values.len() < 3 {
        return Ok(None);
    }
    let Some(query_shape) = shapes.shape(&values[0]) else {
        return Ok(None);
    };
    let Some(value_shape) = shapes.shape(&values[2]) else {
        return Ok(None);
    };
    if query_shape.is_empty() || value_shape.is_empty() {
        return Ok(None);
    }
    let mut output = query_shape.clone();
    let last = output.len() - 1;
    output[last] = value_shape.last().unwrap().clone();
    Ok(Some(output))
}

// ── torch tensor indexing / selection methods ──────────────────────────

/// `torch.gather(input, dim, index)` / `x.gather(dim, index)` — output
/// shape matches `index` exactly (`index` is always the last positional
/// argument in both the free-function and method forms).
fn apply_known_gather(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if let Some(idx) = args.iter().find_map(|a| match a {
        CallArgument::Keyword { name, value } if name == "index" => Some(value.as_str()),
        _ => None,
    }) {
        return Ok(shapes.shape(idx).cloned());
    }
    let positionals = positional_arg_values(args);
    let Some(idx_name) = positionals.last() else {
        return Ok(None);
    };
    Ok(shapes.shape(idx_name).cloned())
}

/// `torch.index_select(input, dim, index)` / `x.index_select(dim, index)` —
/// `dim`'s length becomes `len(index)` (`index` is always 1D).
fn apply_known_index_select(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let Some(dim_raw) = nth_positional_or_keyword(args, 1, &["dim"]) else {
        return Ok(None);
    };
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "index_select")?;
    let Some(index_name) = nth_positional_or_keyword(args, 2, &["index"]) else {
        return Ok(None);
    };
    let Some(index_shape) = shapes.shape(index_name) else {
        return Ok(None);
    };
    let Some(len) = index_shape.first() else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output[dim] = len.clone();
    Ok(Some(output))
}

/// `torch.narrow(input, dim, start, length)` / `x.narrow(dim, start,
/// length)` — slices `dim` down to `length`.
fn apply_known_narrow(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let Some(dim_raw) = nth_positional_or_keyword(args, 1, &["dim"]) else {
        return Ok(None);
    };
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "narrow")?;
    let Some(length) = nth_positional_or_keyword(args, 3, &["length"]) else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output[dim] = length.to_string();
    Ok(Some(output))
}

/// `torch.select(input, dim, index)` / `x.select(dim, index)` — indexes into
/// `dim` with a single integer, removing that axis. Distinct from
/// `jnp.select`/`np.select` (`KnownFunction::Select`, choicelist-based).
fn apply_known_select_dim(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let Some(dim_raw) = nth_positional_or_keyword(args, 1, &["dim"]) else {
        return Ok(None);
    };
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "select")?;
    let mut output = input_shape.clone();
    output.remove(dim);
    Ok(Some(output))
}

/// `x.unfold(dimension, size, step)` — replaces `dimension`'s length with
/// the number of windows and appends a new trailing axis of length `size`.
/// Concrete integer dims only; symbolic geometry is unresolvable.
fn apply_known_unfold(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    let Some(dim_raw) = nth_positional_or_keyword(args, 1, &["dimension", "dim"]) else {
        return Ok(None);
    };
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "unfold")?;
    let Some(size_raw) = nth_positional_or_keyword(args, 2, &["size"]) else {
        return Ok(None);
    };
    let Some(step_raw) = nth_positional_or_keyword(args, 3, &["step"]) else {
        return Ok(None);
    };
    let (Ok(d), Ok(sz), Ok(st)) = (
        input_shape[dim].parse::<isize>(),
        size_raw.trim().parse::<isize>(),
        step_raw.trim().parse::<isize>(),
    ) else {
        return Ok(None);
    };
    if st <= 0 || sz <= 0 || d < sz {
        return Err(format!(
            "unfold: invalid size {} / step {} for dim {} of length {}",
            sz, st, dim, d
        ));
    }
    let mut output = input_shape.clone();
    output[dim] = ((d - sz) / st + 1).to_string();
    output.push(sz.to_string());
    Ok(Some(output))
}

/// `x.view_as(other)` / `x.reshape_as(other)` / `x.expand_as(other)` — the
/// output shape is simply `other`'s shape.
fn apply_known_shape_as(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(other_name) = nth_positional_or_keyword(args, 1, &["other"]) else {
        return Ok(None);
    };
    Ok(shapes.shape(other_name).cloned())
}

/// `x.new_zeros(size)` / `x.new_ones(size)` / `x.new_full(size, value)` /
/// `x.new_empty(size)` — output shape from `size` (the receiver is only a
/// dtype/device template, positional[0] after synthesis; the shape spec is
/// the next positional/keyword arg).
fn apply_known_new_constructor(args: &[CallArgument]) -> Result<Option<Vec<String>>, String> {
    match nth_positional_or_keyword(args, 1, &["size", "shape"]) {
        Some(v) => Ok(parse_shape_value(v)),
        None => Ok(None),
    }
}

// ── torch tuple-output methods ──────────────────────────────────────────

/// `torch.topk(input, k, dim=-1, ...)` / `x.topk(k, dim=-1, ...)` — shared
/// shape math for the `(values, indices)` tuple: both replace `dim` with
/// `k`. Used from `analysis.rs`'s tuple-unpacking dispatch; the real return
/// is always a 2-tuple, so `apply_known_function`'s single-shape dispatch
/// for `KnownFunction::TopK` is conservatively `Ok(None)`.
pub fn apply_known_topk_shape(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    if rank == 0 {
        return Err("topk requires rank >= 1 input".to_string());
    }
    let Some(k) = nth_positional_or_keyword(args, 1, &["k"]) else {
        return Ok(None);
    };
    let dim_raw = nth_positional_or_keyword(args, 2, &["dim", "axis"]).unwrap_or("-1");
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "topk")?;
    let k_dim = if let Ok(value) = k.parse::<isize>() {
        if value < 0
            || input_shape[dim]
                .parse::<usize>()
                .is_ok_and(|size| value as usize > size)
        {
            return Err("topk requires 0 <= k <= axis size".to_string());
        }
        value.to_string()
    } else if k
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        k.to_string()
    } else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output[dim] = k_dim;
    Ok(Some(output))
}

/// `torch.kthvalue(input, k, dim=-1, keepdim=False)` / `x.kthvalue(k,
/// dim=-1, keepdim=False)` — `values`/`indices` share the shape of a plain
/// single-axis reduction over `dim` (like `max`/`min`); `k` itself only
/// picks *which* element, not the shape. Re-synthesizes `(input, dim=..,
/// keepdim=..)` — dropping `k` — and reuses [`apply_known_reduction`],
/// since `k` sits at the position a normal reduction call would expect the
/// axis to be.
pub fn apply_known_kthvalue_shape(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let mut synthesized = vec![CallArgument::Positional {
        value: input_name.to_string(),
    }];
    if let Some(dim) = nth_positional_or_keyword(args, 2, &["dim"]) {
        synthesized.push(CallArgument::Keyword {
            name: "dim".to_string(),
            value: dim.to_string(),
        });
    }
    for arg in args {
        if let CallArgument::Keyword { name, value } = arg
            && (name == "keepdim" || name == "keepdims")
        {
            synthesized.push(CallArgument::Keyword {
                name: name.clone(),
                value: value.clone(),
            });
        }
    }
    apply_known_reduction(&synthesized, shapes)
}

/// `torch.unbind(input, dim=0)` / `x.unbind(dim=0)` — removes `dim`,
/// producing one output per element along that axis. `n_targets` is
/// already fixed by the tuple-unpacking LHS; when the axis is a literal, it
/// must agree with `n_targets` (mismatch is a genuine contradiction, hence
/// `Err`). A symbolic axis size is conservatively unknown (`Ok(None)`).
pub fn compute_unbind_shape(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    n_targets: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    if rank == 0 {
        return Err("unbind requires rank >= 1 input".to_string());
    }
    let dim_raw = nth_positional_or_keyword(args, 1, &["dim", "axis"]).unwrap_or("0");
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "unbind")?;
    let Ok(size) = input_shape[dim].parse::<usize>() else {
        return Ok(None);
    };
    if size != n_targets {
        return Err(format!(
            "unbind along dim {} produces {} outputs, but {} were unpacked",
            dim, size, n_targets
        ));
    }
    let mut out = input_shape.clone();
    out.remove(dim);
    Ok(Some(out))
}

/// `torch.chunk(input, chunks, dim=0)` / `x.chunk(chunks, dim=0)` — splits
/// `dim` into at most `chunks` pieces of `ceil(size/chunks)` each (the last
/// piece may be smaller); unlike `split`'s `N`-equal-sections semantics,
/// `chunks` need not evenly divide the axis. A symbolic axis size falls
/// back to `chunks` same-named synthetic pieces, mirroring `split`'s
/// symbolic-axis convention.
pub fn compute_chunk_shapes(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<Vec<String>>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.is_empty() {
        return Err("chunk requires rank >= 1 input".to_string());
    }
    let rank = input_shape.len();
    let Some(chunks_raw) = nth_positional_or_keyword(args, 1, &["chunks"]) else {
        return Ok(None);
    };
    let Ok(n) = chunks_raw.trim().parse::<usize>() else {
        return Ok(None);
    };
    if n == 0 {
        return Err("chunk requires chunks > 0".to_string());
    }
    let dim_raw = nth_positional_or_keyword(args, 2, &["dim"]).unwrap_or("0");
    let Some(dim) = parse_axis(dim_raw) else {
        return Ok(None);
    };
    let dim = normalize_axis(dim, rank, "chunk")?;
    let axis_dim = &input_shape[dim];

    let Ok(axis_size) = axis_dim.parse::<usize>() else {
        let chunk_dim = format!("chunk({}, {})", axis_dim, n);
        let mut output_shapes = Vec::with_capacity(n);
        for _ in 0..n {
            let mut out = input_shape.clone();
            out[dim] = chunk_dim.clone();
            output_shapes.push(out);
        }
        return Ok(Some(output_shapes));
    };

    let chunk_size = axis_size.div_ceil(n).max(1);
    let mut output_shapes = Vec::new();
    let mut remaining = axis_size;
    for _ in 0..n {
        if remaining == 0 {
            break;
        }
        let this_size = remaining.min(chunk_size);
        let mut out = input_shape.clone();
        out[dim] = this_size.to_string();
        output_shapes.push(out);
        remaining -= this_size;
    }
    Ok(Some(output_shapes))
}

// ── torch combinatorics ──────────────────────────────────────────────────

fn n_choose_r(n: u64, r: u64) -> u64 {
    if r > n {
        return 0;
    }
    let r = r.min(n - r);
    let mut result: u64 = 1;
    for i in 0..r {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// `torch.combinations(input, r=2, with_replacement=False)` — `input` is
/// 1D of length `n`; output is `(nCr(n, r), r)`. `with_replacement` isn't
/// modelled (changes the count formula but not this rule's structure).
fn apply_known_combinations(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() != 1 {
        return Ok(None);
    }
    let Ok(n) = input_shape[0].parse::<u64>() else {
        return Ok(None);
    };
    let r = nth_positional_or_keyword(args, 1, &["r"])
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(2);
    Ok(Some(vec![n_choose_r(n, r).to_string(), r.to_string()]))
}

/// `torch.cartesian_prod(*tensors)` — each tensor is 1D; output is
/// `(prod(lengths), num_tensors)`, or just the tensor itself when only one
/// is given (torch's degenerate single-input case).
fn apply_known_cartesian_prod(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let names = positional_arg_values(args);
    if names.is_empty() {
        return Ok(None);
    }
    if names.len() == 1 {
        return Ok(shapes.shape(&names[0]).cloned());
    }
    let mut lens = Vec::with_capacity(names.len());
    for name in &names {
        let Some(shape) = shapes.shape(name) else {
            return Ok(None);
        };
        let [len] = shape.as_slice() else {
            return Ok(None);
        };
        lens.push(len.clone());
    }
    let total = match lens
        .iter()
        .map(|s| s.parse::<usize>().ok())
        .collect::<Option<Vec<_>>>()
    {
        Some(values) => values.iter().product::<usize>().to_string(),
        None => lens.join("*"),
    };
    Ok(Some(vec![total, names.len().to_string()]))
}

/// `torch.block_diag(*tensors)` — sums each block's row/col dims onto the
/// diagonal: `(sum(rows), sum(cols))`. 1D tensors are treated as a single
/// row (torch promotes them before block-diagonalizing); anything else
/// (rank >= 3) isn't modelled.
fn apply_known_block_diag(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let names = positional_arg_values(args);
    if names.is_empty() {
        return Ok(None);
    }
    let mut rows = Vec::with_capacity(names.len());
    let mut cols = Vec::with_capacity(names.len());
    for name in &names {
        let Some(shape) = shapes.shape(name) else {
            return Ok(None);
        };
        match shape.as_slice() {
            [r, c] => {
                rows.push(r.clone());
                cols.push(c.clone());
            }
            [n] => {
                rows.push("1".to_string());
                cols.push(n.clone());
            }
            _ => return Ok(None),
        }
    }
    Ok(Some(vec![concat_dim(&rows), concat_dim(&cols)]))
}

// ── torch.nn.functional ──────────────────────────────────────────────────

/// Parse a `stride`/`padding`/`dilation`-style argument: a bare int
/// (broadcast to every spatial axis) or an explicit per-axis tuple/list of
/// length `rank`.
fn parse_int_seq_arg(raw: Option<&str>, rank: usize, default: isize) -> Option<Vec<isize>> {
    match raw {
        None => Some(vec![default; rank]),
        Some(v) => {
            let trimmed = v.trim();
            if let Ok(n) = trimmed.parse::<isize>() {
                return Some(vec![n; rank]);
            }
            let list = parse_shape_value(trimmed)?;
            let ints: Vec<isize> = list.iter().filter_map(|s| s.trim().parse().ok()).collect();
            if ints.len() == rank { Some(ints) } else { None }
        }
    }
}

/// `torch.nn.functional.conv1d/2d/3d(input, weight, bias=None, stride=1,
/// padding=0, dilation=1, groups=1)` — channels-first: `input` is `(N,
/// C_in, *spatial)`, `weight` is `(C_out, C_in/groups, *kernel)`. Output is
/// `(N, C_out, *spatial_out)` via the standard conv formula. Channel-count
/// validation is skipped (grouped convs make `C_in/groups` legitimately
/// differ from `C_in`), matching `jax.lax.conv_general_dilated`'s existing
/// approximation. `'same'`/`'valid'` string padding and symbolic spatial
/// dims aren't modelled.
fn apply_known_functional_conv(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    spatial_rank: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() != spatial_rank + 2 {
        return Ok(None);
    }
    let Some(weight_name) = nth_positional_or_keyword(args, 1, &["weight"]) else {
        return Ok(None);
    };
    let Some(weight_shape) = shapes.shape(weight_name) else {
        return Ok(None);
    };
    if weight_shape.len() != spatial_rank + 2 {
        return Ok(None);
    }

    let padding_raw = nth_positional_or_keyword(args, 4, &["padding"]);
    if matches!(
        padding_raw.map(str::trim),
        Some("'same'") | Some("\"same\"") | Some("'valid'") | Some("\"valid\"")
    ) {
        return Ok(None);
    }
    let Some(stride) = parse_int_seq_arg(
        nth_positional_or_keyword(args, 3, &["stride"]),
        spatial_rank,
        1,
    ) else {
        return Ok(None);
    };
    let Some(padding) = parse_int_seq_arg(padding_raw, spatial_rank, 0) else {
        return Ok(None);
    };
    let Some(dilation) = parse_int_seq_arg(
        nth_positional_or_keyword(args, 5, &["dilation"]),
        spatial_rank,
        1,
    ) else {
        return Ok(None);
    };

    let mut output = input_shape.clone();
    output[1] = weight_shape[0].clone();
    for i in 0..spatial_rank {
        let idx = 2 + i;
        let Ok(d) = input_shape[idx].parse::<isize>() else {
            return Ok(None);
        };
        let Ok(k) = weight_shape[idx].parse::<isize>() else {
            return Ok(None);
        };
        let s = stride[i].max(1);
        let eff_k = (k - 1) * dilation[i] + 1;
        output[idx] = ((d + 2 * padding[i] - eff_k) / s + 1).to_string();
    }
    Ok(Some(output))
}

/// `torch.nn.functional.max_pool1d/2d/3d` / `avg_pool1d/2d/3d(input,
/// kernel_size, stride=None, padding=0, ...)` — channels-first; `stride`
/// defaults to `kernel_size` (torch convention) when omitted. `dilation`
/// (max_pool only) isn't modelled (assumed 1), matching the approximation
/// already used for conv layers with `dilation != 1`.
fn apply_known_functional_pool(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    spatial_rank: usize,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() != spatial_rank + 2 {
        return Ok(None);
    }
    let stride_raw = nth_positional_or_keyword(args, 2, &["stride"]);
    let Some(kernel) = nth_positional_or_keyword(args, 1, &["kernel_size"])
        .and_then(|v| parse_int_seq_arg(Some(v), spatial_rank, 0))
    else {
        return Ok(None);
    };
    let stride = match stride_raw {
        None => kernel.clone(),
        Some(_) => match parse_int_seq_arg(stride_raw, spatial_rank, 1) {
            Some(s) => s,
            None => return Ok(None),
        },
    };
    let Some(padding) = parse_int_seq_arg(
        nth_positional_or_keyword(args, 3, &["padding"]),
        spatial_rank,
        0,
    ) else {
        return Ok(None);
    };

    let mut output = input_shape.clone();
    for i in 0..spatial_rank {
        let idx = 2 + i;
        let Ok(d) = input_shape[idx].parse::<isize>() else {
            return Ok(None);
        };
        let s = stride[i].max(1);
        output[idx] = ((d + 2 * padding[i] - kernel[i]) / s + 1).to_string();
    }
    Ok(Some(output))
}

/// `torch.nn.functional.interpolate(input, size=None, scale_factor=None,
/// ...)` — sets the trailing spatial dims (everything after the leading
/// batch+channel pair) directly to `size`, or scales them by
/// `scale_factor` (floor-rounded, matching torch's output-size formula).
fn apply_known_interpolate(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.len() < 3 {
        return Ok(None);
    }
    let spatial_rank = input_shape.len() - 2;
    let mut output = input_shape.clone();

    if let Some(size_raw) = nth_positional_or_keyword(args, 1, &["size"]) {
        if let Ok(n) = size_raw.trim().parse::<i64>() {
            for i in 0..spatial_rank {
                output[2 + i] = n.to_string();
            }
            return Ok(Some(output));
        }
        let Some(list) = parse_shape_value(size_raw) else {
            return Ok(None);
        };
        if list.len() != spatial_rank {
            return Ok(None);
        }
        output[2..2 + spatial_rank].clone_from_slice(&list);
        return Ok(Some(output));
    }

    if let Some(scale_raw) = nth_positional_or_keyword(args, 2, &["scale_factor"]) {
        let factors: Vec<f64> = if let Ok(f) = scale_raw.trim().parse::<f64>() {
            vec![f; spatial_rank]
        } else {
            let Some(list) = parse_shape_value(scale_raw) else {
                return Ok(None);
            };
            let Some(parsed) = list
                .iter()
                .map(|s| s.trim().parse::<f64>().ok())
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            if parsed.len() != spatial_rank {
                return Ok(None);
            }
            parsed
        };
        for i in 0..spatial_rank {
            let Ok(d) = input_shape[2 + i].parse::<f64>() else {
                return Ok(None);
            };
            output[2 + i] = ((d * factors[i]).floor() as i64).to_string();
        }
        return Ok(Some(output));
    }

    Ok(None)
}

/// `torch.nn.functional.embedding(input, weight, ...)` — appends `weight`'s
/// embedding-dim (its last axis) to `input`'s shape, same rule as
/// `torch.nn.Embedding`/`equinox.nn.Embedding`.
fn apply_known_functional_embedding(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(weight_name) = nth_positional_or_keyword(args, 1, &["weight"]) else {
        return Ok(None);
    };
    let Some(weight_shape) = shapes.shape(weight_name) else {
        return Ok(None);
    };
    let Some(embed_dim) = weight_shape.last() else {
        return Ok(None);
    };
    let mut output = input_shape.clone();
    output.push(embed_dim.clone());
    Ok(Some(output))
}

/// `torch.nn.functional.glu(input, dim=-1)` — splits `input` in half along
/// `dim` and gates one half with the other (`a * sigmoid(b)`); the halved
/// dim is the only shape change. Same numeric/factor-cancellation/opaque
/// naming convention as `compute_split_shapes`'s literal-`N` case.
fn apply_known_functional_glu(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if input_shape.is_empty() {
        return Err("glu requires rank >= 1 input".to_string());
    }
    let rank = input_shape.len();
    let axis = normalize_axis(axis_arg(args, -1), rank, "glu")?;

    let axis_dim = &input_shape[axis];
    let half_dim = if let Ok(axis_size) = axis_dim.parse::<usize>() {
        if axis_size % 2 != 0 {
            return Err(format!("glu requires an even-sized dim, got {}", axis_size));
        }
        (axis_size / 2).to_string()
    } else if let Some(simplified) = cancel_product_factor(axis_dim, 2) {
        simplified
    } else {
        format!("glu({})", axis_dim)
    };

    let mut output = input_shape.clone();
    output[axis] = half_dim;
    Ok(Some(output))
}

// ── torch.nn.utils.rnn ───────────────────────────────────────────────────

/// `torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, ...)` —
/// `sequences` must be a literal list of known-shape tensors; the batch
/// count and trailing (non-length) dims are then known statically, but the
/// padded sequence length is genuinely data-dependent (the max length
/// across inputs), so it's emitted as an opaque symbolic dim.
fn apply_known_pad_sequence(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(list_arg) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(names) = parse_simple_sequence_names(list_arg) else {
        return Ok(None);
    };
    if names.is_empty() {
        return Ok(None);
    }
    let mut trailing: Option<Vec<String>> = None;
    for name in &names {
        let Some(shape) = shapes.shape(name) else {
            return Ok(None);
        };
        if shape.is_empty() {
            return Ok(None);
        }
        let t = shape[1..].to_vec();
        match &trailing {
            None => trailing = Some(t),
            Some(existing) if *existing != t => {
                return Err("pad_sequence: mismatched trailing dims across sequences".to_string());
            }
            _ => {}
        }
    }
    let trailing = trailing.unwrap();
    let batch_first = args.iter().any(|a| {
        matches!(a, CallArgument::Keyword { name, value } if name == "batch_first" && value.trim() == "True")
    });
    let max_len = "pad_len".to_string();
    let mut output = if batch_first {
        vec![names.len().to_string(), max_len]
    } else {
        vec![max_len, names.len().to_string()]
    };
    output.extend(trailing);
    Ok(Some(output))
}

#[cfg(test)]
mod known_function_shape_rule_tests;

#[cfg(test)]
mod known_function_tests;

#[cfg(test)]
mod method_call_tests;
