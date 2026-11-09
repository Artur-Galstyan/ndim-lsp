use std::collections::HashMap;

use crate::layers::{UNPROVEN_DIM, string_text};
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
            "transpose" => Some(KnownFunction::Transpose),
            "expand_dims" => Some(KnownFunction::ExpandDims),
            "max" | "amax" => Some(KnownFunction::Max),
            "min" | "amin" => Some(KnownFunction::Min),
            "minmax" if is_jax_numpy => Some(KnownFunction::MinMax),
            "vdot" => Some(KnownFunction::Vdot),
            "split" => Some(KnownFunction::Split),
            "array_split" => Some(KnownFunction::ArraySplit),
            "flatten" => Some(KnownFunction::Flatten),
            "moveaxis" => Some(KnownFunction::MoveAxis),
            "zeros" => Some(KnownFunction::Zeros),
            "ones" => Some(KnownFunction::Ones),
            "empty" => Some(KnownFunction::Empty),
            "logspace" | "geomspace" => Some(KnownFunction::Logspace),
            "identity" => Some(KnownFunction::Identity),
            "broadcast_arrays" => Some(KnownFunction::BroadcastArrays),
            "pad" if is_numpy => Some(KnownFunction::NumpyPad),
            "pad" => Some(KnownFunction::Pad),
            "block" => Some(KnownFunction::Block),
            "array" => Some(KnownFunction::Array),
            "asarray" => Some(KnownFunction::AsArray),
            "diagflat" => Some(KnownFunction::Diagflat),
            "tri" => Some(KnownFunction::Tri),
            "indices" => Some(KnownFunction::Indices),
            "bincount" => Some(KnownFunction::BinCount),
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
            "median" => Some(KnownFunction::Reduction { axis: 1, keepdims: Some(4), q: None }),
            "quantile" | "percentile" => {
                Some(KnownFunction::Reduction { axis: 2, keepdims: Some(6), q: Some(1) })
            }
            "count_nonzero" => Some(KnownFunction::Reduction { axis: 1, keepdims: None, q: None }),
            "ptp" => Some(KnownFunction::Max),
            "cross" => Some(KnownFunction::Cross),
            _ => classify_array_function(name),
        };
    }

    if is_torch {
        return match name.as_str() {
            "cat" => Some(KnownFunction::Concatenate),
            "transpose" | "swapdims" => Some(KnownFunction::SwapAxes),
            "unsqueeze" => Some(KnownFunction::ExpandDims),
            "max" | "min" => Some(KnownFunction::TorchMaxMin),
            "sum" | "mean" | "prod" | "all" | "any" | "argmax" | "argmin" => Some(TORCH_REDUCTION),
            "std" | "var" => Some(TORCH_STD_VAR),
            "quantile" => Some(KnownFunction::Reduction { axis: 2, keepdims: Some(3), q: Some(1) }),
            "norm" => Some(NORM),
            "squeeze" => Some(KnownFunction::TorchSqueeze),
            "repeat_interleave" => Some(KnownFunction::Repeat),
            "bmm" => Some(KnownFunction::Bmm),
            "mm" => Some(KnownFunction::Mm),
            "mv" => Some(KnownFunction::Mv),
            "addmm" => Some(KnownFunction::Addmm),
            "baddbmm" => Some(KnownFunction::Baddbmm),
            "bincount" => Some(KnownFunction::BinCount),
            // Real torch `split` semantics: 2nd arg is a chunk *size*, not a
            // section count. `tensor_split` keeps numpy/jnp `array_split`
            // semantics (verified against torch docs).
            "split" => Some(KnownFunction::TorchSplit),
            "tensor_split" => Some(KnownFunction::ArraySplit),
            "flatten" => Some(KnownFunction::TorchFlatten),
            "clamp" => Some(KnownFunction::Clamp),
            "zeros" | "ones" | "empty" | "rand" | "randn" => {
                Some(KnownFunction::TorchSizeConstructor)
            }
            "broadcast_tensors" => Some(KnownFunction::BroadcastArrays),
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
            "sort" => Some(KnownFunction::TorchSort),
            "combinations" => Some(KnownFunction::Combinations),
            "cartesian_prod" => Some(KnownFunction::CartesianProd),
            "block_diag" => Some(KnownFunction::BlockDiag),
            "linspace" | "logspace" => Some(KnownFunction::TorchLinspace),
            _ => classify_array_function(name),
        };
    }

    let is_jax_numpy_linalg = module == ["jax", "numpy", "linalg"];
    let is_numpy_linalg = module == ["numpy", "linalg"];
    let is_torch_linalg = module == ["torch", "linalg"];
    let is_torch_nn_functional = module == ["torch", "nn", "functional"];

    if is_torch_nn_functional {
        return match name.as_str() {
            "pad" => Some(KnownFunction::TorchPad),
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
            "one_hot" => Some(KnownFunction::OneHot),
            "linear" => Some(KnownFunction::Linear),
            "scaled_dot_product_attention" => Some(KnownFunction::ScaledDotProductAttention),
            "mse_loss" => Some(functional_loss("MSELoss", 4, Some((2, 3)))),
            "l1_loss" => Some(functional_loss("L1Loss", 4, Some((2, 3)))),
            "smooth_l1_loss" => Some(functional_loss("SmoothL1Loss", 4, Some((2, 3)))),
            "huber_loss" => Some(functional_loss("HuberLoss", 2, None)),
            "binary_cross_entropy" => Some(functional_loss("BCELoss", 5, Some((3, 4)))),
            "binary_cross_entropy_with_logits" => {
                Some(functional_loss("BCEWithLogitsLoss", 5, Some((3, 4))))
            }
            "cross_entropy" => Some(KnownFunction::CrossEntropy),
            "nll_loss" => Some(KnownFunction::NllLoss),
            "embedding" => Some(KnownFunction::FunctionalEmbedding),
            // Activations, normalization, and dropout keep the shape.
            "softmax" | "log_softmax" | "normalize" | "relu" | "relu6" | "gelu" | "silu"
            | "sigmoid" | "tanh" | "softplus" | "softsign" | "selu" | "celu" | "elu"
            | "leaky_relu" | "hardtanh" | "hardswish" | "hardsigmoid" | "mish" | "dropout"
            | "dropout1d" | "dropout2d" | "dropout3d" | "alpha_dropout" | "feature_alpha_dropout"
            | "layer_norm" | "batch_norm" | "group_norm" | "instance_norm" => Some(KnownFunction::Copy),
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
            "norm" => Some(NORM),
            "solve" => Some(KnownFunction::LinalgSolve),
            // Shape-preserving on a square matrix — same rule as `inv`.
            "cholesky" => Some(KnownFunction::LinalgInv),
            "lstsq" => Some(KnownFunction::LinalgLstsq),
            "pinv" => Some(KnownFunction::LinalgPinv),
            "matrix_rank" => Some(KnownFunction::LinalgMatrixRank),
            "matrix_sqrth" if is_torch_linalg => Some(KnownFunction::LinalgMatrixSqrth),
            "eigvals" | "eigvalsh" => Some(KnownFunction::LinalgEigvals),
            "matrix_power" => Some(KnownFunction::LinalgMatrixPower),
            "multi_dot" => Some(KnownFunction::LinalgMultiDot),
            "slogdet" => Some(KnownFunction::LinalgSlogdet),
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
            "dot_general" => Some(KnownFunction::DotGeneral),
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
            "logsumexp" => Some(KnownFunction::Reduction { axis: 1, keepdims: Some(3), q: None }),
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

/// The names with one rule in `numpy`, `jax.numpy`, and `torch`.
fn classify_array_function(name: &str) -> Option<KnownFunction> {
    match name {
        "concatenate" | "concat" => Some(KnownFunction::Concatenate),
        "swapaxes" => Some(KnownFunction::SwapAxes),
        "clip" => Some(KnownFunction::Clamp),
        "stack" => Some(KnownFunction::Stack),
        "reshape" => Some(KnownFunction::Reshape),
        "squeeze" => Some(KnownFunction::Squeeze),
        "sum" => Some(KnownFunction::Sum),
        "mean" => Some(KnownFunction::Mean),
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
        "einsum" => Some(KnownFunction::Einsum),
        "tile" => Some(KnownFunction::Tile),
        "repeat" => Some(KnownFunction::Repeat),
        "ravel" => Some(KnownFunction::Ravel),
        "where" => Some(KnownFunction::Where),
        "full" => Some(KnownFunction::Full),
        "zeros_like" => Some(KnownFunction::ZerosLike),
        "ones_like" => Some(KnownFunction::OnesLike),
        "full_like" => Some(KnownFunction::FullLike),
        "empty_like" => Some(KnownFunction::EmptyLike),
        "arange" => Some(KnownFunction::Arange),
        "linspace" => Some(KnownFunction::Linspace),
        "eye" => Some(KnownFunction::Eye),
        "broadcast_to" => Some(KnownFunction::BroadcastTo),
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
        "unique" => Some(KnownFunction::Unique),
        _ => None,
    }
}

fn parse_simple_sequence_names(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = bracket_inner(trimmed).unwrap_or(trimmed);
    let names = split_top_level(inner)
        .into_iter()
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

/// Whether the flag at positional `at` or keyword `name` is absent or a
/// literal `False`.
fn flag_unset(args: &[CallArgument], at: usize, name: &str) -> bool {
    nth_positional_or_keyword(args, at, &[name]).is_none_or(|value| parse_bool(value) == Some(false))
}

fn parse_axis_list(value: &str) -> Option<Vec<isize>> {
    let trimmed = value.trim();
    if trimmed == "None" {
        return Some(vec![isize::MIN]);
    }
    if let Some(axis) = parse_axis(trimmed) {
        return Some(vec![axis]);
    }
    let inner = bracket_inner(trimmed)?;
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

/// The shape of the array that [`nth_positional_or_keyword`] finds.
fn shape_arg<'a>(
    args: &[CallArgument],
    skip: usize,
    keywords: &[&str],
    shapes: &'a dyn ShapeLookup,
) -> Option<&'a Vec<String>> {
    shapes.shape(nth_positional_or_keyword(args, skip, keywords)?)
}

/// The axis that [`nth_positional_or_keyword`] finds, or `default`, as an
/// index for `rank`. `Ok(None)` when there is no value or it is not an int.
fn axis_at(
    args: &[CallArgument],
    skip: usize,
    keywords: &[&str],
    default: Option<&str>,
    rank: usize,
    context: &str,
) -> Result<Option<usize>, String> {
    let axis = nth_positional_or_keyword(args, skip, keywords).or(default).and_then(parse_axis);
    axis.map(|axis| normalize_axis(axis, rank, context)).transpose()
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

/// The text between a first `(` or `[` and a last matching `)` or `]`.
fn bracket_inner(text: &str) -> Option<&str> {
    text.strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
        .or_else(|| text.strip_prefix('[')?.strip_suffix(']'))
}

/// The text inside the brackets of a sequence. `(a)*(b)` has no outer pair.
fn sequence_inner(text: &str) -> Option<&str> {
    let inner = bracket_inner(text)?;
    let mut depth = 0usize;
    for (at, c) in text.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && at + 1 < text.len() {
                    return None;
                }
            }
            _ => {}
        }
    }
    Some(inner)
}

/// Whether `text` calls something: `get(1)`, but not `(n-1)*2`.
fn has_call(text: &str) -> bool {
    text.char_indices().any(|(at, c)| {
        c == '(' && text[..at].ends_with(|before: char| before.is_alphanumeric() || "_.)]".contains(before))
    })
}

/// Whether `text`, which is not a sequence, is proven to be one integer. A plain
/// name or an arithmetic on names stays, and a call, a subscript, or `x.shape`
/// can be a sequence. A free dim is a value of the analysis.
fn is_one_size(text: &str) -> bool {
    if text.contains(FREE_DIM) {
        return true;
    }
    let shape_index = text.split_once(".shape[").is_some_and(|(_, index)| {
        index.strip_suffix(']').is_some_and(|index| index.parse::<i64>().is_ok())
    });
    shape_index
        || !(text.contains([',', '[', '{'])
            || text.ends_with(".shape")
            || has_call(text))
}

/// Whether `dim`, which has a comma, is one call or one subscript: `min(k, 2)`.
/// The comma of `d for d in (2, 3)` or of `(2, 3)` is not in one size.
fn is_call_with_comma(dim: &str) -> bool {
    let mut depth = 0usize;
    dim.starts_with(|c: char| c.is_alphabetic() || c == '_')
        && dim.char_indices().all(|(_, c)| {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = depth.saturating_sub(1),
                _ => {}
            }
            depth > 0 || !c.is_whitespace()
        })
}

/// The dims of a size argument: a tuple, a list, or one size. A dim with a
/// comma, such as `min(k, 2)`, has a value the analysis does not compute, so it
/// is free. Any other expression has no proven rank, so `None`.
pub(crate) fn parse_shape_value(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = match sequence_inner(trimmed) {
        Some(inner) if trimmed.starts_with('(') && split_top_level(inner).len() == 1 => {
            return parse_shape_value(inner);
        }
        Some(inner) => inner,
        None if is_one_size(trimmed) => trimmed,
        None => return None,
    };

    let dims = split_top_level(inner)
        .into_iter()
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| {
            if !dim.contains(',') || dim.contains(FREE_DIM) {
                Some(dim.to_string())
            } else {
                is_call_with_comma(dim).then(|| format!("{FREE_DIM}{dim}"))
            }
        })
        .collect::<Option<Vec<_>>>()?;

    let spaced = |dim: &String| !dim.contains(FREE_DIM) && dim.contains(char::is_whitespace);
    let invalid = |dim: &String| dim.starts_with("__synth_") || dim.starts_with('*') || spaced(dim);
    if dims.is_empty() || dims.iter().any(invalid) {
        None
    } else {
        Some(dims)
    }
}

fn dim_product(dims: &[String]) -> Option<usize> {
    dims.iter()
        .map(|dim| dim.parse::<usize>())
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim.ok()?))
}

fn flattened_dim(dims: &[String]) -> String {
    if let Some(product) = dim_product(dims) {
        return product.to_string();
    }
    match dims {
        [dim] => dim.clone(),
        _ => joined_dim(&dims.iter().map(|dim| factor_text(dim)).collect::<Vec<_>>(), "*"),
    }
}

pub(crate) fn multiply_dim(dim: &str, factor: &str) -> String {
    if factor == "1" {
        return dim.to_string();
    }
    if dim == "1" {
        return factor.to_string();
    }
    if let (Ok(dim), Ok(factor)) = (dim.parse::<usize>(), factor.parse::<usize>()) {
        return dim.checked_mul(factor).map_or(UNKNOWN_DIM.to_string(), |dim| dim.to_string());
    }
    joined_dim(&[factor_text(dim), factor_text(factor)], "*")
}

fn add_to_dim(dim: &str, amount: isize) -> String {
    if amount == 0 {
        return dim.to_string();
    }
    if let Ok(dim) = dim.parse::<isize>() {
        return dim.checked_add(amount).map_or(UNKNOWN_DIM.to_string(), |dim| dim.to_string());
    }
    let Some(negated) = (amount as i64).checked_neg() else {
        return UNKNOWN_DIM.to_string();
    };
    offset_dim(dim, negated).unwrap_or_else(|| format!("{dim}{amount:+}"))
}

/// The integers of a literal. A value with a name has none: `min(1, 2)` is
/// one item, not two.
fn parse_ints(value: &str) -> Vec<isize> {
    let mut ints = Vec::new();
    if value.chars().any(char::is_alphabetic) {
        return ints;
    }
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
    let mut aligned = true;
    let mut conflicts = Conflicts::default();

    for i in 0..rank {
        let left_dim = left
            .get(left.len().wrapping_sub(1 + i))
            .map(String::as_str)
            .unwrap_or("1");
        let right_dim = right
            .get(right.len().wrapping_sub(1 + i))
            .map(String::as_str)
            .unwrap_or("1");

        // Past a variadic dim, the two shapes have no common axis numbers.
        aligned &= !is_variadic_dim(left_dim) && !is_variadic_dim(right_dim);

        let dim = if right_dim == "1" {
            left_dim.to_string()
        } else if left_dim == "1" {
            right_dim.to_string()
        } else if !aligned || !dims_conflict(left_dim, right_dim) {
            unified_dim(left_dim, right_dim)
        } else {
            let message = format!(
                "cannot broadcast dim {} with {} (shapes [{}] and [{}])",
                left_dim,
                right_dim,
                left.join(", "),
                right.join(", ")
            );
            conflicts.add(left_dim, right_dim, message)?;
            left_dim.to_string()
        };
        result.push(dim);
    }

    conflicts.finish()?;
    result.reverse();
    Ok(result)
}

/// Whether `other` cannot broadcast to `target`.
fn broadcast_changes(target: &[String], other: &[String]) -> Result<bool, String> {
    let broadcast = broadcast_two_shapes(target, other)?;
    Ok(broadcast.len() != target.len()
        || broadcast.iter().zip(target).any(|(a, b)| dims_conflict(a, b)))
}

fn has_variadic_dim(shape: &[String]) -> bool {
    shape.iter().any(|dim| is_variadic_dim(dim))
}

/// The index in `shape` of the axis `axis`, when no variadic dim is between it
/// and the end that `axis` counts from.
fn fixed_axis(shape: &[String], axis: isize) -> Option<usize> {
    let at = if axis < 0 { shape.len().checked_add_signed(axis)? } else { axis as usize };
    match shape.iter().position(|dim| is_variadic_dim(dim)) {
        Some(variadic) if axis < 0 => (at > variadic).then_some(at),
        Some(variadic) => (at < variadic).then_some(at),
        None => (at < shape.len()).then_some(at),
    }
}

/// The index in `shape` of the new axis of a stack at `axis`, on the same terms
/// as `fixed_axis`.
fn stack_axis(shape: &[String], axis: isize) -> Option<usize> {
    let variadic = shape.iter().position(|dim| is_variadic_dim(dim))?;
    if axis < 0 {
        (shape.len() + 1).checked_add_signed(axis).filter(|at| *at > variadic)
    } else {
        Some(axis as usize).filter(|at| *at <= variadic)
    }
}

/// Whether the dim at index `at` of `shape` can be the axis `axis` for some
/// count of axes of the variadic dim.
fn can_be_axis(shape: &[String], at: usize, axis: isize) -> bool {
    let from_end = |index: usize| shape.len() - 1 - index;
    let variadic = shape.iter().position(|dim| is_variadic_dim(dim));
    let (at, axis, variadic) = if axis < 0 {
        (from_end(at), axis.unsigned_abs() - 1, variadic.map(from_end))
    } else {
        (at, axis as usize, variadic)
    };
    match variadic {
        Some(variadic) if at > variadic => axis >= at - 1,
        _ => at == axis,
    }
}

/// Compare `dim` with the dim that the shapes before it have at index `at`
/// from one end. Gives the earlier dim when the two conflict.
fn merge_aligned_dim(seen: &mut Vec<String>, at: usize, dim: &str) -> Option<String> {
    if seen.len() <= at {
        seen.resize(at + 1, UNKNOWN_DIM.to_string());
    }
    if dims_conflict(&seen[at], dim) {
        return Some(seen[at].clone());
    }
    seen[at] = unified_dim(&seen[at], dim);
    None
}

/// The error that `shapes`, of which at least one has a variadic dim, prove for
/// `rule`. A variadic dim gives a lower bound of the rank. The dims between it
/// and an end align from that end. `join_axis` is the axis of a concatenate,
/// which may differ in size.
fn variadic_join_conflict(rule: &str, shapes: &[Vec<String>], join_axis: Option<isize>) -> Result<(), String> {
    let lowest = |shape: &Vec<String>| shape.len() - usize::from(has_variadic_dim(shape));
    let lower_bound = shapes.iter().map(lowest).max().unwrap_or(0);
    if let Some(short) = shapes.iter().find(|shape| !has_variadic_dim(shape) && shape.len() < lower_bound) {
        return Err(format!(
            "{rule} expected all inputs to have rank {lower_bound} or more, got rank {}",
            short.len()
        ));
    }

    let first_rank = shapes.first().map_or(0, Vec::len);
    let (mut from_start, mut from_end) = (Vec::new(), Vec::new());
    let mut conflicts = Conflicts::default();
    for shape in shapes {
        let variadic = shape.iter().position(|dim| is_variadic_dim(dim));
        let mismatch = |axis: usize, expected: &str, got: &str| {
            format!("{rule} dimension mismatch at axis {axis}: expected {expected}, got {got}")
        };
        let joined = |at: usize| join_axis.is_some_and(|axis| can_be_axis(shape, at, axis));
        for at in (0..variadic.unwrap_or(shape.len())).filter(|at| !joined(*at)) {
            if let Some(expected) = merge_aligned_dim(&mut from_start, at, &shape[at]) {
                conflicts.add(&expected, &shape[at], mismatch(at, &expected, &shape[at]))?;
            }
        }
        for at in (variadic.map_or(0, |at| at + 1)..shape.len()).rev().filter(|at| !joined(*at)) {
            let back = shape.len() - 1 - at;
            if let Some(expected) = merge_aligned_dim(&mut from_end, back, &shape[at]) {
                let message = mismatch(first_rank.saturating_sub(back + 1), &expected, &shape[at]);
                conflicts.add(&expected, &shape[at], message)?;
            }
        }
    }
    conflicts.finish()
}

/// Whether a `Concatenate` or a `Stack` call reads a shape with a variadic dim.
/// The error of such a call is proven, see `variadic_join_conflict`.
pub(crate) fn joins_variadic_shape(known: &KnownFunction, args: &[CallArgument], shapes: &dyn ShapeLookup) -> bool {
    matches!(known, KnownFunction::Concatenate | KnownFunction::Stack)
        && sequence_arg_value(args).and_then(parse_simple_sequence_names).is_some_and(|names| {
            names.iter().any(|name| shapes.shape(name).is_some_and(|shape| has_variadic_dim(shape)))
        })
}

fn concat_dim(dims: &[String]) -> String {
    let parsed = dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<Result<Vec<_>, _>>();

    if let Ok(values) = parsed {
        let sum = values.iter().try_fold(0usize, |sum, value| sum.checked_add(*value));
        return sum.map_or(UNKNOWN_DIM.to_string(), |sum| sum.to_string());
    }

    joined_dim(dims, "+")
}

fn functional_loss(layer: &'static str, reduction: usize, deprecated: Option<(usize, usize)>) -> KnownFunction {
    KnownFunction::FunctionalLoss { layer, reduction, deprecated }
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
        "repeat" => Some(KnownFunction::Repeat),
        "repeat_interleave" => Some(KnownFunction::RepeatInterleave),
        "expand" | "broadcast_to" => Some(KnownFunction::BroadcastTo),
        "astype" => Some(KnownFunction::Astype),
        "copy" | "byteswap" | "clone" | "cpu" | "cuda" | "float" | "long" | "int" | "bool"
        | "double" | "half" | "softmax" | "log_softmax" | "type_as" | "exp" | "log" | "sqrt"
        | "rsqrt" | "abs" | "neg" | "sigmoid" | "tanh" | "relu" | "sin" | "cos" | "square"
        | "floor" | "ceil" | "round" | "sign" | "exp_" | "log_" | "sqrt_" | "rsqrt_" | "abs_"
        | "neg_" | "sigmoid_" | "tanh_" | "relu_" | "sin_" | "cos_" | "square_" | "floor_"
        | "ceil_" | "round_" | "sign_" | "zero_" | "fill_" | "add_" | "sub_" | "mul_" | "div_"
        | "pow_" | "clamp_" | "copy_" | "uniform_" | "normal_" => Some(KnownFunction::Copy),
        "detach" => Some(KnownFunction::Detach),
        "contiguous" => Some(KnownFunction::Contiguous),
        "to" => Some(KnownFunction::To),
        "gather" => Some(KnownFunction::Gather),
        "scatter" => Some(KnownFunction::Scatter),
        "masked_select" => Some(KnownFunction::MaskedSelect),
        "masked_fill" | "masked_fill_" => Some(KnownFunction::MaskedFill),
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
        "clamp" | "clip" => Some(KnownFunction::Clamp),
        "norm" => Some(NORM),
        "diagonal" => Some(KnownFunction::Diagonal),
        "tril" => Some(KnownFunction::Tril),
        "triu" => Some(KnownFunction::Triu),
        "chunk" => Some(KnownFunction::Chunk),
        "unbind" => Some(KnownFunction::Unbind),
        "bmm" => Some(KnownFunction::Bmm),
        "mm" => Some(KnownFunction::Mm),
        "mv" => Some(KnownFunction::Mv),
        "matmul" => Some(KnownFunction::Matmul),
        "t" => Some(KnownFunction::TorchT),
        // Tensor method form: `x.split(size, dim=0)` — real torch semantics
        // (chunk size, not section count); see `KnownFunction::TorchSplit`.
        "split" => Some(KnownFunction::TorchSplit),
        "kthvalue" => Some(KnownFunction::KthValue),
        "median" | "mode" => Some(KnownFunction::MedianDim),
        _ => None,
    }
}

/// The torch rule of a method whose numpy rule is `method`, when the two
/// libraries read the arguments in different ways.
pub fn torch_method_reading(method: &KnownFunction) -> Option<KnownFunction> {
    match method {
        KnownFunction::Sum
        | KnownFunction::Mean
        | KnownFunction::Prod
        | KnownFunction::All
        | KnownFunction::Any
        | KnownFunction::ArgMax
        | KnownFunction::ArgMin => Some(TORCH_REDUCTION),
        KnownFunction::Std | KnownFunction::Var => Some(TORCH_STD_VAR),
        KnownFunction::Max | KnownFunction::Min => Some(KnownFunction::TorchMaxMin),
        KnownFunction::Repeat => Some(KnownFunction::TorchRepeat),
        KnownFunction::Squeeze => Some(KnownFunction::TorchSqueeze),
        KnownFunction::Sort => Some(KnownFunction::TorchSort),
        _ => None,
    }
}

pub fn apply_method_call(
    method: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let method = if matches!(method, KnownFunction::Flatten) && is_torch_flatten_call(args) {
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
            | KnownFunction::TorchRepeat
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
    let result = apply_known_rule(function, args, shapes);
    match &result {
        Ok(Some(shape)) if shape.iter().any(|dim| dim.contains(UNPROVEN_DIM)) => Ok(None),
        _ => result,
    }
}

fn apply_known_rule(
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
        KnownFunction::Squeeze => apply_known_squeeze(args, shapes, false),
        KnownFunction::TorchSqueeze => apply_known_squeeze(args, shapes, true),
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
        | KnownFunction::Cumprod
        | KnownFunction::Roll
        | KnownFunction::Flip
        | KnownFunction::Triu
        | KnownFunction::Tril
        | KnownFunction::Astype
        | KnownFunction::Copy
        | KnownFunction::Detach
        | KnownFunction::Contiguous
        | KnownFunction::To
        | KnownFunction::LaxScatter
        | KnownFunction::LaxDynamicUpdateSlice
        | KnownFunction::LaxSort
        | KnownFunction::PutAlongAxis
        | KnownFunction::Scatter
        | KnownFunction::MaskedFill => apply_known_shape_preserving(args, shapes),
        KnownFunction::Arange => apply_known_arange(args),
        KnownFunction::Linspace | KnownFunction::Logspace | KnownFunction::TorchLinspace => {
            apply_known_linspace(function, args, shapes)
        }
        KnownFunction::Eye | KnownFunction::Identity => apply_known_eye(args),
        KnownFunction::Matmul => apply_known_matmul(args, shapes),
        KnownFunction::DotGeneral => apply_known_dot_general(args, shapes),
        KnownFunction::Bmm => apply_known_bmm(args, shapes),
        KnownFunction::Mm => apply_known_mm(args, shapes, 2),
        KnownFunction::Mv => apply_known_mm(args, shapes, 1),
        KnownFunction::Addmm => apply_known_add_product(args, shapes, KnownFunction::Mm),
        KnownFunction::Baddbmm => apply_known_add_product(args, shapes, KnownFunction::Bmm),
        KnownFunction::TorchT => apply_known_torch_t(args, shapes),
        KnownFunction::Linear => apply_known_linear(args, shapes),
        KnownFunction::ScaledDotProductAttention => apply_known_scaled_dot_product_attention(args, shapes),
        KnownFunction::FunctionalLoss { layer, reduction, deprecated } => {
            apply_known_functional_loss(args, shapes, layer, *reduction, *deprecated)
        }
        KnownFunction::LinalgEigvals => {
            let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
                return Ok(None);
            };
            require_square_matrix(input_shape, "linalg.eigvals")?;
            Ok(Some(input_shape[..input_shape.len() - 1].to_vec()))
        }
        KnownFunction::LinalgMatrixPower => {
            let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
                return Ok(None);
            };
            require_square_matrix(input_shape, "linalg.matrix_power")?;
            Ok(Some(input_shape.clone()))
        }
        KnownFunction::LinalgSlogdet => {
            if let Some((_, input_shape)) = first_array_arg_shape(args, shapes) {
                require_square_matrix(input_shape, "linalg.slogdet")?;
            }
            Ok(None)
        }
        KnownFunction::LinalgMultiDot => apply_known_multi_dot(args, shapes),
        KnownFunction::Unique => apply_known_unique(args, shapes),
        KnownFunction::MaskedSelect | KnownFunction::BinCount => {
            Ok(first_array_arg_shape(args, shapes).map(|_| vec![UNKNOWN_DIM.to_string()]))
        }
        KnownFunction::Extract => {
            Ok(shape_arg(args, 1, &["arr"], shapes).map(|_| vec![UNKNOWN_DIM.to_string()]))
        }
        KnownFunction::Compress => {
            let axis = nth_positional_or_keyword(args, 2, &["axis"]).filter(|axis| axis.trim() != "None");
            let input = shape_arg(args, 1, &["a"], shapes);
            Ok(input.filter(|_| axis.is_none()).map(|_| vec![UNKNOWN_DIM.to_string()]))
        }
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
        KnownFunction::Tile => apply_known_tile(args, shapes, false),
        KnownFunction::TorchRepeat => apply_known_tile(args, shapes, true),
        KnownFunction::Repeat | KnownFunction::RepeatInterleave => apply_known_repeat(args, shapes),
        KnownFunction::Clamp => apply_known_clamp(args, shapes),
        KnownFunction::Pad => apply_known_pad(args, shapes),
        KnownFunction::TorchPad => apply_known_torch_pad(args, shapes),
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
        | KnownFunction::ArgMin => {
            let (axis, keepdims, q) = numpy_reduction(function);
            apply_reduction(args, shapes, axis, keepdims, q)
        }
        KnownFunction::Reduction { axis, keepdims, q } => apply_reduction(args, shapes, *axis, *keepdims, *q),
        KnownFunction::TorchMaxMin => apply_known_torch_max_min(args, shapes),
        KnownFunction::LinalgInv => apply_known_linalg_inv(args, shapes),
        KnownFunction::LinalgMatrixSqrth => {
            let Some(input_shape) = shape_arg(args, 0, &["A"], shapes) else {
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
        // The carry of a loop and the elems of a scan keep their shape.
        KnownFunction::LaxWhileLoop => Ok(shape_arg(args, 2, &["init_val"], shapes).cloned()),
        KnownFunction::LaxForiLoop => Ok(shape_arg(args, 3, &["init_val"], shapes).cloned()),
        KnownFunction::LaxAssociativeScan => Ok(shape_arg(args, 1, &["elems"], shapes).cloned()),
        KnownFunction::LaxConvGeneralDilated => {
            apply_known_lax_conv_general_dilated(args, shapes)
        }
        KnownFunction::LaxReduceWindow => apply_known_lax_reduce_window(args, shapes),
        KnownFunction::LaxPad => apply_known_lax_pad(args, shapes),
        KnownFunction::LaxBroadcast => apply_known_lax_broadcast(args, shapes),
        KnownFunction::LaxBroadcastInDim => {
            Ok(nth_positional_or_keyword(args, 1, &["shape"]).and_then(parse_shape_value))
        }
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
        KnownFunction::Resize => {
            Ok(nth_positional_or_keyword(args, 1, &["new_shape"]).and_then(parse_shape_value))
        }
        KnownFunction::Insert => apply_known_insert(args, shapes),
        KnownFunction::Delete => apply_known_delete(args, shapes),
        KnownFunction::Append => apply_known_append(args, shapes),
        KnownFunction::Kron => apply_known_kron(args, shapes),
        KnownFunction::Block => apply_known_block(args, shapes),
        // The result has the shape of `indices` or of the searched values `v`.
        KnownFunction::TakeAlongAxis | KnownFunction::SearchSorted => {
            Ok(positional_arg_values(args).get(1).and_then(|name| shapes.shape(name)).cloned())
        }
        KnownFunction::Argwhere => apply_known_argwhere(args, shapes),
        // `(hist, bin_edges)`: a tuple, not one array.
        KnownFunction::Histogram | KnownFunction::TorchSort => Ok(None),
        KnownFunction::Cross => apply_known_cross(args, shapes),
        KnownFunction::TorchCross => apply_known_torch_cross(args, shapes),
        KnownFunction::LinalgSolve => apply_known_linalg_solve(args, shapes),
        KnownFunction::LinalgPinv => apply_known_linalg_pinv(args, shapes),
        KnownFunction::LinalgMatrixRank => apply_known_linalg_matrix_rank(args, shapes),
        KnownFunction::OneHot => apply_known_one_hot(args, shapes),
        KnownFunction::CrossEntropy => apply_known_cross_entropy(args, shapes, true),
        KnownFunction::NllLoss => apply_known_cross_entropy(args, shapes, false),
        KnownFunction::DotProductAttention => apply_known_dot_product_attention(args, shapes),
        KnownFunction::Gather => apply_known_gather(args, shapes),
        KnownFunction::IndexSelect => apply_known_index_select(args, shapes),
        KnownFunction::Narrow => apply_known_narrow(args, shapes),
        KnownFunction::SelectDim => apply_known_select_dim(args, shapes),
        KnownFunction::Unfold => apply_known_unfold(args, shapes),
        KnownFunction::ShapeAs => Ok(shape_arg(args, 1, &["other"], shapes).cloned()),
        KnownFunction::Item => Ok(Some(Vec::new())),
        KnownFunction::NewConstructor => {
            Ok(nth_positional_or_keyword(args, 1, &["size", "shape"]).and_then(parse_shape_value))
        }
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
        // A dict of axis sizes, a data-dependent size, or a tuple of arrays.
        KnownFunction::EinopsParseShape
        | KnownFunction::TopK
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
        KnownFunction::FunctionalMaxPool1d => apply_known_functional_pool(args, shapes, 1, true),
        KnownFunction::FunctionalMaxPool2d => apply_known_functional_pool(args, shapes, 2, true),
        KnownFunction::FunctionalMaxPool3d => apply_known_functional_pool(args, shapes, 3, true),
        KnownFunction::FunctionalAvgPool1d => apply_known_functional_pool(args, shapes, 1, false),
        KnownFunction::FunctionalAvgPool2d => apply_known_functional_pool(args, shapes, 2, false),
        KnownFunction::FunctionalAvgPool3d => apply_known_functional_pool(args, shapes, 3, false),
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
    let Some(pattern) = positional.get(1).and_then(|raw| string_text(raw)) else {
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

    let is_ellipsis = |g: &Vec<String>| g.as_slice() == ["..."];
    if lhs_groups.iter().filter(|g| is_ellipsis(g)).count() > 1
        || rhs_groups.iter().filter(|g| is_ellipsis(g)).count() > 1
    {
        return Ok(None);
    }
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
/// channel dim. Default strides = 1, default padding VALID: out = (d - w)/s + 1,
/// or 0 when the window is larger than `d`. SAME: out = ceil(d / s). Any other
/// padding is unknown. Concrete dims only in v1; symbolic inputs skip.
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

    let same = match nth_positional_or_keyword(args, 3, &["padding"]) {
        None => false,
        Some(padding) => match string_text(padding).map(str::to_ascii_uppercase).as_deref() {
            Some("VALID") => false,
            Some("SAME" | "SAME_LOWER") => true,
            _ => return Ok(None),
        },
    };

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
        let count = if same { same_size(d, s) } else { Some(window_count(d, w, s).unwrap_or(0)) };
        let Some(count) = count else {
            return Ok(None);
        };
        output[start + i] = count.to_string();
    }
    Ok(Some(output))
}

/// The count of windows of size `window` at steps of `stride` in `padded`
/// elements, or `None` when one window is larger than `padded`.
fn window_count(padded: isize, window: isize, stride: isize) -> Option<isize> {
    if padded < window {
        return None;
    }
    padded.checked_sub(window)?.checked_div(stride)?.checked_add(1)
}

/// `(kernel - 1) * dilation + 1`, the extent of a dilated window.
fn dilated_window(kernel: isize, dilation: isize) -> Option<isize> {
    kernel.checked_sub(1)?.checked_mul(dilation)?.checked_add(1)
}

/// `size + low + high`, the size of a padded axis.
fn padded_size(size: isize, low: isize, high: isize) -> Option<isize> {
    size.checked_add(low)?.checked_add(high)
}

/// `ceil(size / stride)`, the size of a `SAME` window axis.
fn same_size(size: isize, stride: isize) -> Option<isize> {
    size.checked_add(stride)?.checked_sub(1)?.checked_div(stride)
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
    let Some(equation_str) = positional_values.first().and_then(|equation| string_text(equation))
    else {
        return Ok(None);
    };
    // NumPy, torch and JAX skip spaces; each other subscript is one ASCII letter.
    let equation: String = equation_str.chars().filter(|c| *c != ' ').collect();
    let letters = |spec: &str| -> Option<Vec<char>> {
        spec.chars().map(|c| c.is_ascii_alphabetic().then_some(c)).collect()
    };
    let (inputs_part, output_part) = match equation.split_once("->") {
        Some((inputs, output)) => (inputs, Some(output)),
        None => (equation.as_str(), None),
    };
    let Some(input_specs) = inputs_part.split(',').map(letters).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let output_spec = match output_part {
        Some(output) => match letters(output) {
            Some(labels) => labels,
            None => return Ok(None),
        },
        None => {
            let mut once: Vec<char> = input_specs
                .iter()
                .flatten()
                .copied()
                .filter(|label| input_specs.iter().flatten().filter(|other| *other == label).count() == 1)
                .collect();
            once.sort_unstable();
            once
        }
    };
    einsum_output_shape("einsum", &input_specs, &output_spec, &positional_values[1..], shapes)
}

/// The shape of an einsum with one label list per operand. `None` when an
/// operand is a star argument or has no known shape, or the output repeats a label. A dim of `1`
/// broadcasts against the other dims of its label.
fn einsum_output_shape<L: Copy + Eq + std::hash::Hash + std::fmt::Display>(
    context: &str,
    input_specs: &[Vec<L>],
    output_spec: &[L],
    operand_names: &[String],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if operand_names.iter().any(|name| name.starts_with('*')) {
        return Ok(None);
    }
    if operand_names.len() != input_specs.len() {
        return Err(format!(
            "{} equation has {} input specs but got {} operands",
            context,
            input_specs.len(),
            operand_names.len()
        ));
    }
    let Some(operand_shapes) = operand_names
        .iter()
        .map(|name| shapes.shape(name.as_str()))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let letters = context == "einsum";
    let spec_text = |spec: &[L]| {
        let text = spec.iter().map(ToString::to_string).collect::<Vec<_>>();
        match letters {
            true => format!("'{}' has length {}", text.concat(), spec.len()),
            false => format!("'{}' has {} axes", text.join(" "), spec.len()),
        }
    };
    let mut label_map: HashMap<L, String> = HashMap::new();
    for ((spec, operand_name), shape) in input_specs.iter().zip(operand_names).zip(operand_shapes) {
        if shape.len() != spec.len() {
            return Err(format!(
                "{} operand '{}' has rank {} but subscript {}",
                context,
                operand_name,
                shape.len(),
                spec_text(spec)
            ));
        }
        for (label, dim) in spec.iter().zip(shape.iter()) {
            match label_map.get(label) {
                Some(existing) if existing == "1" => {
                    label_map.insert(*label, dim.clone());
                }
                Some(_) if dim == "1" => {}
                Some(existing) => {
                    check_dim_match(existing, dim, &format!("{} label '{}'", context, label))?;
                }
                None => {
                    label_map.insert(*label, dim.clone());
                }
            }
        }
    }
    let mut output_shape = Vec::with_capacity(output_spec.len());
    for (index, label) in output_spec.iter().enumerate() {
        if output_spec[..index].contains(label) {
            return Ok(None);
        }
        let Some(dim) = label_map.get(label) else {
            return Err(format!(
                "{} output label '{}' not found in input subscripts",
                context, label
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

    if let Some(variadic) = input_shapes.iter().find(|shape| has_variadic_dim(shape)) {
        variadic_join_conflict("concatenate", &input_shapes, Some(axis))?;
        let Some(at) = fixed_axis(variadic, axis) else {
            return Ok(None);
        };
        let dims = input_shapes.iter().map(|shape| Some(shape[fixed_axis(shape, axis)?].clone()));
        let Some(dims) = dims.collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let mut output = variadic.clone();
        output[at] = concat_dim(&dims);
        return Ok(Some(output));
    }

    let rank = first_shape.len();
    let axis = normalize_axis(axis, rank, "concatenate")?;

    let mut output = first_shape.clone();
    let mut conflicts = Conflicts::default();
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
            if dims_conflict(&shape[dim_idx], &output[dim_idx]) {
                let message = format!(
                    "concatenate dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, output[dim_idx], shape[dim_idx]
                );
                conflicts.add(&output[dim_idx], &shape[dim_idx], message)?;
                continue;
            }
            output[dim_idx] = unified_dim(&output[dim_idx], &shape[dim_idx]);
        }
    }
    conflicts.finish()?;

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

    if let Some(variadic) = input_shapes.iter().find(|shape| has_variadic_dim(shape)) {
        variadic_join_conflict("stack", &input_shapes, None)?;
        let axis = nth_positional_or_keyword(args, 1, &["axis", "dim"]).map_or(Some(0), parse_axis);
        let Some(axis) = axis.and_then(|axis| stack_axis(variadic, axis)) else {
            return Ok(None);
        };
        let mut output = variadic.clone();
        output.insert(axis, input_shapes.len().to_string());
        return Ok(Some(output));
    }

    let mut stacked = first_shape.clone();
    let mut conflicts = Conflicts::default();
    for shape in &input_shapes {
        if shape.len() != first_shape.len() {
            return Err(format!(
                "stack expected all inputs to have rank {}, got rank {}",
                first_shape.len(),
                shape.len()
            ));
        }
        for (dim_idx, got) in shape.iter().enumerate() {
            if dims_conflict(&stacked[dim_idx], got) {
                let message = format!(
                    "stack dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, stacked[dim_idx], got
                );
                conflicts.add(&stacked[dim_idx], got, message)?;
                continue;
            }
            stacked[dim_idx] = unified_dim(&stacked[dim_idx], got);
        }
    }
    conflicts.finish()?;

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

    let mut output = stacked;
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

/// The integer that `dim` is proven to be.
fn proven_size(dim: &str) -> Option<i64> {
    dim.parse().ok().or_else(|| literal_dim(dim)?.parse().ok())
}

/// numpy and jax refuse to squeeze an axis whose size is not 1. torch keeps
/// such an axis. An axis of a size that is not proven gives no shape.
fn apply_known_squeeze(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    torch: bool,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let axes_raw = nth_positional_or_keyword(args, 1, &["axis", "dim", "dimensions"]);
    let axes = match axes_raw.map(parse_axis_list) {
        Some(None) => return Ok(None),
        Some(Some(axes)) if axes.contains(&isize::MIN) => None,
        Some(axes) => axes,
        None => None,
    };

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

    let mut removed = Vec::new();
    for axis in axes {
        match proven_size(&input_shape[axis]) {
            Some(1) => removed.push(axis),
            Some(_) if torch => {}
            Some(_) => {
                return Err(format!(
                    "cannot squeeze axis {} with dimension {}",
                    axis, input_shape[axis]
                ));
            }
            None => return Ok(None),
        }
    }
    let axes = removed;

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
    if dims_canonically_equal(left, right) {
        return left.to_string();
    }
    if let (Ok(left), Ok(right)) = (left.parse::<usize>(), right.parse::<usize>()) {
        return left.min(right).to_string();
    }
    format!("{FREE_DIM}min({},{})", left, right)
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
    if dims_conflict(last, second_last) {
        let message = format!(
            "{} requires last two dimensions to match, got {} and {}",
            context, second_last, last
        );
        let mut conflicts = Conflicts::default();
        conflicts.add(second_last, last, message)?;
        return conflicts.finish();
    }
    Ok(())
}

fn check_dim_match(left: &str, right: &str, context: &str) -> Result<(), String> {
    if !dims_conflict(left, right) {
        return Ok(());
    }
    let message = format!("{} dimension mismatch: expected {}, got {}", context, left, right);
    let mut conflicts = Conflicts::default();
    conflicts.add(left, right, message)?;
    conflicts.finish()
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
        return Ok(stop.checked_sub(start).map(|size| vec![size.max(0).to_string()]));
    }
    Ok(Some(vec![format!("{}-{}", values[1], values[0])]))
}

/// `linspace(start, stop, num)`: numpy and jax give `[num, *broadcast(start,
/// stop)]` for array endpoints. A number is a scalar endpoint (the analysis
/// writes `0` for a provable scalar). torch needs scalar endpoints and `steps`.
fn apply_known_linspace(
    function: &KnownFunction,
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let torch = matches!(function, KnownFunction::TorchLinspace);
    // `retstep=True` gives a `(samples, step)` tuple.
    if matches!(function, KnownFunction::Linspace) && !flag_unset(args, 4, "retstep") {
        return Ok(None);
    }
    let endpoint = |value: Option<&str>| -> Option<Vec<String>> {
        let value = value?.trim();
        if value.parse::<f64>().is_ok() {
            return Some(Vec::new());
        }
        shapes.shape(value).cloned()
    };
    let start = endpoint(nth_positional_or_keyword(args, 0, &["start"]));
    let stop = endpoint(nth_positional_or_keyword(args, 1, &["stop", "end"]));
    let (Some(start), Some(stop)) = (start, stop) else {
        return Ok(None);
    };
    let keywords: &[&str] = if torch { &["steps"] } else { &["num"] };
    let num = match nth_positional_or_keyword(args, 2, keywords) {
        Some(num) => num.to_string(),
        None if torch => return Ok(None),
        None => "50".to_string(),
    };
    if torch {
        return Ok((start.is_empty() && stop.is_empty()).then(|| vec![num]));
    }
    let tail = broadcast_two_shapes(&start, &stop)?;
    let axis_set = positional_arg_values(args).len() > 5
        || args.iter().any(|arg| matches!(arg, CallArgument::Keyword { name, value } if name == "axis" && value.trim() != "0"));
    if !tail.is_empty() && axis_set {
        return Ok(None);
    }
    let mut output = vec![num];
    output.extend(tail);
    Ok(Some(output))
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

/// The two items of a literal pair `(a, b)`.
fn literal_pair(text: &str) -> Option<(&str, &str)> {
    match split_top_level(sequence_inner(text.trim())?).as_slice() {
        [a, b] => Some((a, b)),
        _ => None,
    }
}

/// The axes of a literal tuple of non-negative integers.
fn literal_axes(text: &str) -> Option<Vec<usize>> {
    split_top_level(sequence_inner(text.trim())?)
        .into_iter()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.parse().ok())
        .collect()
}

/// `jax.lax.dot_general(lhs, rhs, ((lc, rc), (lb, rb)))`: the batch dims, the
/// free dims of `lhs`, then the free dims of `rhs`. Only literal axes give a
/// shape.
fn apply_known_dot_general(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let (Some(lhs), Some(rhs)) = (shape_arg(args, 0, &["lhs"], shapes), shape_arg(args, 1, &["rhs"], shapes)) else {
        return Ok(None);
    };
    let numbers = nth_positional_or_keyword(args, 2, &["dimension_numbers"]).and_then(|numbers| {
        let (contracting, batch) = literal_pair(numbers)?;
        let (lc, rc) = literal_pair(contracting)?;
        let (lb, rb) = literal_pair(batch)?;
        Some([literal_axes(lc)?, literal_axes(rc)?, literal_axes(lb)?, literal_axes(rb)?])
    });
    let Some([lc, rc, lb, rb]) = numbers else {
        return Ok(None);
    };
    let valid = |shape: &[String], contracting: &[usize], batch: &[usize]| {
        let mut axes = [contracting, batch].concat();
        axes.sort_unstable();
        axes.dedup();
        !has_variadic_dim(shape)
            && axes.len() == contracting.len() + batch.len()
            && axes.iter().all(|axis| *axis < shape.len())
    };
    if lc.len() != rc.len() || lb.len() != rb.len() || !valid(lhs, &lc, &lb) || !valid(rhs, &rc, &rb) {
        return Ok(None);
    }
    for (l, r) in lb.iter().zip(&rb).chain(lc.iter().zip(&rc)) {
        check_dim_match(&lhs[*l], &rhs[*r], "dot_general")?;
    }
    let free = |shape: &[String], contracting: &[usize], batch: &[usize]| {
        (0..shape.len())
            .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
            .map(|axis| shape[axis].clone())
            .collect::<Vec<_>>()
    };
    let batch = lb.iter().zip(&rb).map(|(l, r)| unified_dim(&lhs[*l], &rhs[*r]));
    Ok(Some(batch.chain(free(lhs, &lc, &lb)).chain(free(rhs, &rc, &rb)).collect()))
}

/// `torch.bmm(input, mat2)`: both operands rank 3, equal batch dim (unlike
/// `matmul`, a batch of 1 does not broadcast), equal contracted dim.
fn apply_known_bmm(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(left) = shape_arg(args, 0, &["input"], shapes) else {
        return Ok(None);
    };
    let Some(right) = shape_arg(args, 1, &["mat2"], shapes) else {
        return Ok(None);
    };
    for (operand, dims) in [("input", left), ("mat2", right)] {
        if dims.len() != 3 {
            return Err(format!(
                "bmm requires {} to be a 3D tensor, got rank {}",
                operand,
                dims.len()
            ));
        }
    }
    check_dim_match(&left[0], &right[0], "bmm batch")?;
    check_dim_match(&left[2], &right[1], "bmm")?;
    Ok(Some(vec![left[0].clone(), left[1].clone(), right[2].clone()]))
}

/// `torch.mm(input, mat2)` with a rank-2 `mat2`, or `torch.mv(input, vec)`
/// with a rank-1 `vec`. `input` has rank 2. No operand broadcasts.
fn apply_known_mm(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    right_rank: usize,
) -> Result<Option<Vec<String>>, String> {
    let name = if right_rank == 2 { "mm" } else { "mv" };
    let (Some(left), Some(right)) = (
        shape_arg(args, 0, &["input"], shapes),
        shape_arg(args, 1, &["mat2", "vec"], shapes),
    ) else {
        return Ok(None);
    };
    if has_variadic_dim(left) || has_variadic_dim(right) {
        return Ok(None);
    }
    for (operand, dims, rank) in [("input", left, 2), ("the second operand", right, right_rank)] {
        if dims.len() != rank {
            return Err(format!("{name} requires {operand} to have rank {rank}, got rank {}", dims.len()));
        }
    }
    check_dim_match(&left[1], &right[0], name)?;
    Ok(Some([&left[..1], &right[1..]].concat()))
}

/// `torch.addmm(input, mat1, mat2)` and `torch.baddbmm(input, batch1,
/// batch2)`: the product of the two matrices, to which `input` broadcasts.
fn apply_known_add_product(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    product: KnownFunction,
) -> Result<Option<Vec<String>>, String> {
    let (Some(left), Some(right)) = (
        nth_positional_or_keyword(args, 1, &["mat1", "batch1"]),
        nth_positional_or_keyword(args, 2, &["mat2", "batch2"]),
    ) else {
        return Ok(None);
    };
    let operands = [CallArgument::Positional { value: left.to_string() }, CallArgument::Positional { value: right.to_string() }];
    let Some(output) = apply_known_rule(&product, &operands, shapes)? else {
        return Ok(None);
    };
    let Some(input) = shape_arg(args, 0, &["input"], shapes) else {
        return Ok(None);
    };
    if has_variadic_dim(input) || broadcast_changes(&output, input)? {
        return Ok(None);
    }
    Ok(Some(output))
}

/// `Tensor.t()` swaps the two axes of a matrix and keeps a tensor of rank 0 or 1.
fn apply_known_torch_t(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    if has_variadic_dim(input_shape) {
        return Ok(None);
    }
    if input_shape.len() > 2 {
        return Err(format!("t() expects a tensor with rank 2 or less, got rank {}", input_shape.len()));
    }
    Ok(Some(input_shape.iter().rev().cloned().collect()))
}

/// `F.linear(input, weight)`: the last dim of `input` is the `in` dim of
/// `weight`. A `[out, in]` weight gives `[..., out]`, a `[in]` weight `[...]`.
fn apply_known_linear(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let (Some(input), Some(weight)) = (
        shape_arg(args, 0, &["input"], shapes),
        shape_arg(args, 1, &["weight"], shapes),
    ) else {
        return Ok(None);
    };
    let Some(features) = input.last().filter(|dim| !is_variadic_dim(dim)) else {
        return Ok(None);
    };
    if has_variadic_dim(weight) {
        return Ok(None);
    }
    let out = match weight.as_slice() {
        [out, inner] => {
            check_dim_match(features, inner, "linear")?;
            Some(out)
        }
        [inner] => {
            check_dim_match(features, inner, "linear")?;
            None
        }
        _ => return Err(format!("linear requires weight to have rank 1 or 2, got rank {}", weight.len())),
    };
    Ok(Some(input[..input.len() - 1].iter().chain(out).cloned().collect()))
}

/// `F.scaled_dot_product_attention(query, key, value)`: `[..., L, E]`,
/// `[..., S, E]`, and `[..., S, Ev]` give `[..., L, Ev]`. The head dims can
/// differ (`enable_gqa`), so only the ranks and the last two dims are read.
fn apply_known_scaled_dot_product_attention(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let (Some(query), Some(key), Some(value)) = (
        shape_arg(args, 0, &["query"], shapes),
        shape_arg(args, 1, &["key"], shapes),
        shape_arg(args, 2, &["value"], shapes),
    ) else {
        return Ok(None);
    };
    let rank = query.len();
    if rank < 2 || key.len() != rank || value.len() != rank || [query, key, value].iter().any(|shape| has_variadic_dim(shape)) {
        return Ok(None);
    }
    check_dim_match(&query[rank - 1], &key[rank - 1], "scaled_dot_product_attention key")?;
    check_dim_match(&key[rank - 2], &value[rank - 2], "scaled_dot_product_attention value")?;
    Ok(Some(query[..rank - 1].iter().chain(&value[rank - 1..]).cloned().collect()))
}

/// A `torch.nn.functional` loss by the rule of its module, see `apply_loss_layer`.
fn apply_known_functional_loss(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    layer: &str,
    reduction: usize,
    deprecated: Option<(usize, usize)>,
) -> Result<Option<Vec<String>>, String> {
    let (Some(input), Some(target)) = (
        nth_positional_or_keyword(args, 0, &["input"]),
        nth_positional_or_keyword(args, 1, &["target"]),
    ) else {
        return Ok(None);
    };
    let given = |position, name| {
        nth_positional_or_keyword(args, position, &[name]).is_some_and(|value| value != "None")
    };
    let reduction = match deprecated {
        Some((size_average, reduce)) if given(size_average, "size_average") || given(reduce, "reduce") => None,
        _ => Some(nth_positional_or_keyword(args, reduction, &["reduction"]).unwrap_or("'mean'")),
    };
    apply_loss_layer(layer, reduction, input, target, shapes)
}

/// `linalg.multi_dot(arrays)`: a chain of matrix products. The first array
/// can be a row and the last a column, all others have rank 2.
fn apply_known_multi_dot(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(names) = sequence_arg_value(args).and_then(parse_simple_sequence_names) else {
        return Ok(None);
    };
    let Some(operands) = names.iter().map(|name| shapes.shape(name)).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    if operands.len() < 2 || operands.iter().any(|shape| has_variadic_dim(shape)) {
        return Ok(None);
    }
    let last = operands.len() - 1;
    for (at, shape) in operands.iter().enumerate() {
        let allowed = shape.len() == 2 || (shape.len() == 1 && (at == 0 || at == last));
        if !allowed {
            return Err(format!("multi_dot operand {at} must have rank 2, got rank {}", shape.len()));
        }
    }
    let mut output = operands[0].clone();
    for shape in &operands[1..] {
        let inner = output.pop().unwrap_or_default();
        check_dim_match(&inner, &shape[0], "multi_dot")?;
        output.extend(shape[1..].iter().cloned());
    }
    Ok(Some(output))
}

/// `unique` without options: the distinct values of the flat input, so one
/// axis of a size the file does not prove. An option that changes the result
/// type or rank (`return_*`, `axis`, `dim`, `size`) gives no shape.
fn apply_known_unique(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if first_array_arg_shape(args, shapes).is_none() {
        return Ok(None);
    }
    let unset = |value: &str| matches!(value.trim(), "False" | "None");
    let plain = args.iter().skip(1).all(|arg| match arg {
        CallArgument::Positional { value } => unset(value),
        CallArgument::Keyword { name, value } => {
            matches!(name.as_str(), "sorted" | "equal_nan") || unset(value)
        }
    });
    Ok(plain.then(|| vec![UNKNOWN_DIM.to_string()]))
}

/// `F.cross_entropy` / `F.nll_loss`. The class dim is 0 for an unbatched
/// input and 1 otherwise. A target of the input's own shape holds class
/// probabilities (cross_entropy only); a target without the class dim holds
/// class indices. `reduction='none'` returns the input shape without the
/// class dim, any other valid reduction a scalar.
fn apply_known_cross_entropy(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    probabilities: bool,
) -> Result<Option<Vec<String>>, String> {
    cross_entropy_with_reduction(args, shapes, probabilities, loss_reduction(args))
}

fn cross_entropy_with_reduction(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    probabilities: bool,
    reduction: Result<Option<&str>, String>,
) -> Result<Option<Vec<String>>, String> {
    let name = if probabilities { "cross_entropy" } else { "nll_loss" };
    let Some(input) = shape_arg(args, 0, &["input"], shapes) else {
        return Ok(None);
    };
    let Some(target) = shape_arg(args, 1, &["target"], shapes) else {
        return Ok(None);
    };
    if input.is_empty() {
        return Err(format!("{} requires input with at least 1 dimension", name));
    }
    let class_axis = usize::from(input.len() > 1);
    let per_item: Vec<String> = input
        .iter()
        .enumerate()
        .filter(|&(axis, _)| axis != class_axis)
        .map(|(_, dim)| dim.clone())
        .collect();

    let reduction = reduction?;
    let scalar_or_items = |reduction: Option<&str>| match reduction {
        Some("none") => Some(per_item.clone()),
        Some(_) => Some(Vec::new()),
        None => None,
    };

    if let Some(weight) = nth_positional_or_keyword(args, 2, &["weight"])
        .filter(|&w| w != "None")
        .and_then(|w| shapes.shape(w))
    {
        if weight.len() != 1 {
            return Err(format!(
                "{} weight must be a 1D tensor with one entry per class, got rank {}",
                name,
                weight.len()
            ));
        }
        check_dim_match(&input[class_axis], &weight[0], &format!("{} weight", name))?;
    }

    // Unbatched input: a rank-0 target is a class index, a rank-1 target of
    // the input's shape holds probabilities, and a rank-1 target of size 1 is
    // accepted as one class index (checked against torch 2.x).
    if input.len() == 1 && target.len() == 1 {
        if probabilities && dims_canonically_equal(&input[0], &target[0]) {
            return Ok(scalar_or_items(reduction));
        }
        if target[0] == "1" {
            return Ok(scalar_or_items(reduction));
        }
        if input[0].parse::<usize>().is_ok() && target[0].parse::<usize>().is_ok() {
            return Err(format!(
                "{} for a 1D input needs a target of size 1, got {}",
                name, target[0]
            ));
        }
        return Ok(None);
    }

    if probabilities && target.len() == input.len() {
        for (input_dim, target_dim) in input.iter().zip(target) {
            check_dim_match(input_dim, target_dim, &format!("{} target", name))?;
        }
        return Ok(scalar_or_items(reduction));
    }
    if target.len() + 1 != input.len() {
        return Err(format!(
            "{} target must have rank {} for input of rank {}, got rank {}",
            name,
            input.len() - 1,
            input.len(),
            target.len()
        ));
    }
    for (input_dim, target_dim) in per_item.iter().zip(target) {
        check_dim_match(input_dim, target_dim, &format!("{} target", name))?;
    }
    Ok(scalar_or_items(reduction))
}

/// The literal `reduction` of a loss call: `Some("mean" | "sum" | "none")`,
/// or `None` when it depends on a value (a variable, or the deprecated
/// `size_average` / `reduce` arguments, which override it). An invalid
/// string literal is an error, as in torch.
fn loss_reduction(args: &[CallArgument]) -> Result<Option<&str>, String> {
    let given = |position, name| {
        nth_positional_or_keyword(args, position, &[name]).is_some_and(|value| value != "None")
    };
    if given(3, "size_average") || given(5, "reduce") {
        return Ok(None);
    }
    literal_reduction(nth_positional_or_keyword(args, 6, &["reduction"]))
}

/// The reduction a loss was given as source text: `None` text is the
/// default `'mean'`.
fn literal_reduction(text: Option<&str>) -> Result<Option<&str>, String> {
    let Some(value) = text else {
        return Ok(Some("mean"));
    };
    let value = value.trim();
    let literal = [('\'', '\''), ('"', '"')]
        .iter()
        .find_map(|&(open, close)| value.strip_prefix(open)?.strip_suffix(close));
    match literal {
        Some(reduction @ ("mean" | "sum" | "none")) => Ok(Some(reduction)),
        Some(other) => Err(format!("{} is not a valid value for reduction", other)),
        None => Ok(None),
    }
}

/// A `torch.nn` loss module applied to `input` and `target`. The classification
/// losses follow the rule of the matching function, with the reduction of the
/// constructor. `MSELoss`, `L1Loss`, `SmoothL1Loss`, and `HuberLoss` broadcast
/// the two operands (torch warns, and does not raise). `BCELoss` and
/// `BCEWithLogitsLoss` need equal sizes. The constructor `weight` and the other
/// arguments do not change the shape, so they are not read. An unknown operand
/// gives an unknown result.
pub(crate) fn apply_loss_layer(
    name: &str,
    reduction: Option<&str>,
    input: &str,
    target: &str,
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let (Some(input_shape), Some(target_shape)) = (shapes.shape(input), shapes.shape(target))
    else {
        return Ok(None);
    };
    let reduction = match reduction {
        Some(text) => literal_reduction(Some(text)),
        None => Ok(None),
    };
    if matches!(name, "CrossEntropyLoss" | "NLLLoss") {
        let args = [
            CallArgument::Positional { value: input.to_string() },
            CallArgument::Positional { value: target.to_string() },
        ];
        return cross_entropy_with_reduction(&args, shapes, name == "CrossEntropyLoss", reduction);
    }
    let reduction = reduction?;
    let reduced = |items: Vec<String>| match reduction {
        Some("none") => Some(items),
        Some(_) => Some(Vec::new()),
        None => None,
    };
    if matches!(name, "BCELoss" | "BCEWithLogitsLoss") {
        if input_shape.len() != target_shape.len() {
            return Err(format!(
                "{name} target shape [{}] must match input shape [{}]",
                target_shape.join(", "),
                input_shape.join(", ")
            ));
        }
        for (want, got) in input_shape.iter().zip(target_shape) {
            if provably_different(want, got) {
                return Err(format!(
                    "{name} target shape [{}] must match input shape [{}]",
                    target_shape.join(", "),
                    input_shape.join(", ")
                ));
            }
        }
        return Ok(reduced(input_shape.clone()));
    }
    let mut broadcast = Some(Vec::new());
    for i in 1..=input_shape.len().max(target_shape.len()) {
        let left = input_shape.len().checked_sub(i).map(|at| &input_shape[at]);
        let right = target_shape.len().checked_sub(i).map(|at| &target_shape[at]);
        let dim = match (left, right) {
            (Some(dim), None) | (None, Some(dim)) => Some(dim.clone()),
            (Some(a), Some(b)) if dims_canonically_equal(a, b) => Some(a.clone()),
            (Some(a), Some(b)) if a == "1" => Some(b.clone()),
            (Some(a), Some(b)) if b == "1" => Some(a.clone()),
            (Some(a), Some(b)) if provably_different(a, b) => {
                return Err(format!(
                    "{name} cannot broadcast input shape [{}] with target shape [{}]: dim {a} vs {b}",
                    input_shape.join(", "),
                    target_shape.join(", ")
                ));
            }
            _ => None,
        };
        match (&mut broadcast, dim) {
            (Some(dims), Some(dim)) => dims.insert(0, dim),
            _ => broadcast = None,
        }
    }
    Ok(match (reduction, broadcast) {
        (Some("none"), broadcast) => broadcast,
        (Some(_), _) => Some(Vec::new()),
        (None, _) => None,
    })
}

/// Two dims that are different integers.
fn provably_different(a: &str, b: &str) -> bool {
    matches!((a.parse::<u64>(), b.parse::<u64>()), (Ok(a), Ok(b)) if a != b)
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

/// `np.tile` pads `reps` with ones. torch `x.repeat(*sizes)` refuses fewer
/// sizes than dims, and one size can be a sequence, so it gives no shape then.
fn apply_known_tile(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    torch: bool,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };

    let Some(reps) =
        nth_positional_or_keyword(args, 1, &["reps", "dims", "repeats"]).and_then(parse_shape_value)
    else {
        return Ok(None);
    };
    if torch && reps.len() < input_shape.len() {
        return Ok(None);
    }

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
    let Some(repeats) = repeats else {
        return Ok(None);
    };
    // An array of counts gives a size the file does not prove.
    let array_repeats = shapes.shape(repeats).is_some_and(|shape| !shape.is_empty());
    if array_repeats || !is_one_size(repeats) {
        return Ok(None);
    }
    let axis = match axis.map(str::trim) {
        None | Some("None") => None,
        Some(axis) => match parse_axis(axis) {
            Some(axis) => Some(axis),
            None => return Ok(None),
        },
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

/// `clip` / `clamp` broadcast the input with every array bound, so
/// `jnp.clip(x, lo, hi)` with a `lo` of a larger shape is bigger than `x`.
/// The caller removes bounds that are scalars; a bound with no known shape
/// leaves the result unknown, and so does a mismatch.
fn apply_known_clamp(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(CallArgument::Positional { value: input }) = args.first() else {
        return Ok(None);
    };
    let Some(mut output) = shapes.shape(input).cloned() else {
        return Ok(None);
    };
    for arg in &args[1..] {
        let value = match arg {
            CallArgument::Positional { value } => value,
            CallArgument::Keyword { name, value }
                if matches!(name.as_str(), "min" | "max" | "a_min" | "a_max") =>
            {
                value
            }
            CallArgument::Keyword { .. } => continue,
        };
        let Some(bound) = shapes.shape(value) else {
            return Ok(None);
        };
        match broadcast_two_shapes(&output, bound) {
            Ok(broadcast) => output = broadcast,
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(output))
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

    let mut output = first_shape.clone();
    let mut conflicts = Conflicts::default();
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
            if dims_conflict(&shape[dim_idx], &output[dim_idx]) {
                let message = format!(
                    "concat dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, output[dim_idx], shape[dim_idx]
                );
                conflicts.add(&output[dim_idx], &shape[dim_idx], message)?;
                continue;
            }
            output[dim_idx] = unified_dim(&output[dim_idx], &shape[dim_idx]);
        }
    }
    conflicts.finish()?;

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

/// The items of a tuple or list literal. `None` for other text.
fn literal_items(text: &str) -> Option<Vec<&str>> {
    let inner = sequence_inner(text.trim())?;
    let mut items: Vec<&str> = split_top_level(inner).into_iter().map(str::trim).collect();
    if items.last() == Some(&"") {
        items.pop();
    }
    (!items.contains(&"")).then_some(items)
}

/// `np.pad`/`jnp.pad`: `pad_width` broadcasts to one `(before, after)` pair
/// per axis, in axis order. Negative or other widths stay unknown.
fn numpy_pad_pairs(width: &str, rank: usize) -> Option<Vec<(isize, isize)>> {
    let int = |text: &str| text.trim().parse::<isize>().ok().filter(|value| *value >= 0);
    let pair = |text: &str| match literal_items(text)?.as_slice() {
        [value] => int(value).map(|value| (value, value)),
        [before, after] => Some((int(before)?, int(after)?)),
        _ => None,
    };
    if let Some(value) = int(width) {
        return Some(vec![(value, value); rank]);
    }
    if let Some(pair) = pair(width) {
        return Some(vec![pair; rank]);
    }
    let pairs = literal_items(width)?.into_iter().map(pair).collect::<Option<Vec<_>>>()?;
    match pairs.len() {
        1 => Some(vec![pairs[0]; rank]),
        len if len == rank => Some(pairs),
        _ => None,
    }
}

fn apply_known_pad(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(pad_width) = nth_positional_or_keyword(args, 1, &["pad_width"]) else {
        return Ok(None);
    };
    let Some(pairs) = numpy_pad_pairs(pad_width, input_shape.len()) else {
        return Ok(None);
    };
    let Some(additions) = pairs
        .iter()
        .map(|(before, after)| before.checked_add(*after))
        .collect::<Option<Vec<_>>>()
    else {
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

/// `F.pad(x, pad)`: `pad` holds `(left, right)` pairs for the last dims, the
/// last dim first. A pad that is not an integer gives that dim no size.
fn apply_known_torch_pad(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(pad) = nth_positional_or_keyword(args, 1, &["pad"]) else {
        return Ok(None);
    };
    let Some(items) = literal_items(pad) else {
        return Ok(None);
    };
    if items.iter().any(|item| item.starts_with('*') || bracket_inner(item).is_some()) {
        return Ok(None);
    }
    if items.len() % 2 != 0 {
        return Err(format!("pad length must be even, got {}", items.len()));
    }
    if items.len() > 2 * input_shape.len() {
        return Err(format!(
            "pad has {} values but the input has rank {}, at most {} values",
            items.len(),
            input_shape.len(),
            2 * input_shape.len()
        ));
    }
    let mut output = input_shape.clone();
    for (pair, dim) in items.chunks(2).zip(output.iter_mut().rev()) {
        let (Ok(left), Ok(right)) = (pair[0].parse::<isize>(), pair[1].parse::<isize>()) else {
            *dim = UNKNOWN_DIM.to_string();
            continue;
        };
        let Some(addition) = left.checked_add(right) else {
            return Ok(None);
        };
        *dim = add_to_dim(dim, addition);
        if dim.parse::<isize>().is_ok_and(|size| size < 0) {
            return Ok(None);
        }
    }
    Ok(Some(output))
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

/// `where(condition, x, y)` broadcasts its three operands. Two operands with
/// shapes that do not broadcast are a conflict even when the third has no
/// shape. One argument gives a tuple of index arrays.
fn apply_known_where(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let operands = [
        nth_positional_or_keyword(args, 0, &["condition"]),
        nth_positional_or_keyword(args, 1, &["x", "input"]),
        nth_positional_or_keyword(args, 2, &["y", "other"]),
    ];
    if operands[1].is_none() || operands[2].is_none() {
        return Ok(None);
    }
    let known: Vec<Option<Vec<String>>> = operands
        .into_iter()
        .map(|value| {
            let value = value?.trim();
            if value.parse::<f64>().is_ok() {
                return Some(Vec::new());
            }
            shapes.shape(value).cloned()
        })
        .collect();
    let mut output = Vec::new();
    for shape in known.iter().flatten() {
        output = broadcast_two_shapes(&output, shape)?;
    }
    Ok(known.iter().all(Option::is_some).then_some(output))
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

/// The signature of torch `sum`, `mean`, `prod`, `all`, `any`, `argmax`, `argmin`.
const TORCH_REDUCTION: KnownFunction = KnownFunction::Reduction { axis: 1, keepdims: Some(2), q: None };
/// torch `std` and `var`: `(input, dim, unbiased, keepdim)`.
const TORCH_STD_VAR: KnownFunction = KnownFunction::Reduction { axis: 1, keepdims: Some(3), q: None };
/// `norm(x, ord, axis, keepdims)` in numpy, jax, and torch.
const NORM: KnownFunction = KnownFunction::Reduction { axis: 2, keepdims: Some(3), q: None };

/// A torch reduction: `(input, dim, keepdim)`.
pub fn apply_known_reduction(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    apply_known_rule(&TORCH_REDUCTION, args, shapes)
}

/// The numpy signature of a reduction: `(a, axis, dtype, out, keepdims)` for
/// `sum`, `std` with `ddof` before `keepdims`, and `(a, axis, out, keepdims)`.
fn numpy_reduction(function: &KnownFunction) -> (usize, Option<usize>, Option<usize>) {
    match function {
        KnownFunction::Sum | KnownFunction::Mean | KnownFunction::Prod => (1, Some(4), None),
        KnownFunction::Std | KnownFunction::Var => (1, Some(5), None),
        _ => (1, Some(3), None),
    }
}

/// The shape of a `q` argument: a number is a scalar, a list of numbers has
/// one axis, and a name has its shape.
fn quantile_shape(q: &str, shapes: &dyn ShapeLookup) -> Option<Vec<String>> {
    let q = q.trim();
    if q.parse::<f64>().is_ok() {
        return Some(Vec::new());
    }
    if let Some(inner) = sequence_inner(q) {
        let items = split_top_level(inner).into_iter().map(str::trim).filter(|item| !item.is_empty()).collect::<Vec<_>>();
        return items.iter().all(|item| item.parse::<f64>().is_ok()).then(|| vec![items.len().to_string()]);
    }
    shapes.shape(q).cloned()
}

fn apply_reduction(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    axis_at: usize,
    keepdims_at: Option<usize>,
    q_at: Option<usize>,
) -> Result<Option<Vec<String>>, String> {
    let Some((_input_name, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    // `logsumexp(a, axis, b, keepdims, return_sign)` gives a tuple with the sign.
    let sign_at = if (axis_at, keepdims_at, q_at) == (1, Some(3), None) { 4 } else { usize::MAX };
    if !flag_unset(args, sign_at, "return_sign") {
        return Ok(None);
    }

    let leading = match q_at {
        Some(at) => match nth_positional_or_keyword(args, at, &["q"]).and_then(|q| quantile_shape(q, shapes)) {
            Some(leading) => leading,
            None => return Ok(None),
        },
        None => Vec::new(),
    };
    let axis_raw = nth_positional_or_keyword(args, axis_at, &["axis", "dim"]);
    let keepdims_raw = match keepdims_at {
        Some(at) => nth_positional_or_keyword(args, at, &["keepdims", "keepdim"]),
        None => args.iter().rev().find_map(|arg| match arg {
            CallArgument::Keyword { name, value } if name == "keepdims" || name == "keepdim" => Some(value.as_str()),
            _ => None,
        }),
    };
    let Some(keepdims) = keepdims_raw.map_or(Some(false), parse_bool) else {
        return Ok(None);
    };
    let reduced = reduced_shape(input_shape, axis_raw, keepdims)?;
    Ok(reduced.map(|reduced| [leading, reduced].concat()))
}

fn reduced_shape(input_shape: &[String], axis_raw: Option<&str>, keepdims: bool) -> Result<Option<Vec<String>>, String> {
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
        return Ok(Some(input_shape.to_vec()));
    }

    let axes = if axes.contains(&isize::MIN) {
        (0..rank).map(|axis| axis as isize).collect::<Vec<_>>()
    } else {
        axes
    };

    let mut normalized = Vec::new();
    for axis in axes {
        let axis = normalize_axis(axis, rank, "reduction")?;
        if normalized.contains(&axis) {
            return Err(format!("duplicate reduction axis {}", axis));
        }
        normalized.push(axis);
    }
    normalized.sort_unstable();

    if keepdims {
        let mut output = input_shape.to_vec();
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
    let Some(x) = shape_arg(args, 0, &["x"], shapes) else {
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
    let Some(p) = shape_arg(args, 2, &["p"], shapes) else {
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
    if broadcast_changes(&batch, p_batch)? {
        return Err("random.multinomial shape cannot broadcast p batch dimensions".to_string());
    }
    if let Some(n) = shape_arg(args, 1, &["n"], shapes)
        && broadcast_changes(&batch, n)?
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
/// The most pieces that a split, chunk or array_split gives a shape to. A
/// call with more pieces has no shape, so that a large literal count does not
/// allocate one shape per piece.
const MAX_PIECES: usize = 256;

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
    let Some(axis) = axis_at(args, 2, &["axis", "dim"], Some("0"), rank, "split")? else {
        return Ok(None);
    };

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
            if n > MAX_PIECES {
                return Ok(None);
            }
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
        if n > MAX_PIECES {
            return Ok(None);
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

        let mut out = input_shape.clone();
        out[axis] = chunk_dim;
        return Ok(Some(vec![out; n]));
    }

    // ── CASE 2: list literal of indices ──────────────────────────────
    if bracket_inner(trimmed_spec).is_some() {
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
    let Some(axis) = axis_at(args, 2, &["dim"], Some("0"), rank, "split")? else {
        return Ok(None);
    };
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
            if count > MAX_PIECES {
                return Ok(None);
            }
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
        let mut out = input_shape.clone();
        out[axis] = size.to_string();
        let mut output_shapes = vec![out; k - 1];
        let Some(consumed) = size.checked_mul(k - 1) else {
            return Ok(None);
        };
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
                    acc.checked_add(usize::try_from((*v)?).ok()?)
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
    let inner = bracket_inner(value.trim())?;
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
        let Some(size) = limit[i].checked_sub(start[i]) else {
            return Ok(None);
        };
        if size < 0 {
            return Err(format!(
                "lax.slice: limit_indices[{}]={} < start_indices[{}]={}",
                i, limit[i], i, start[i]
            ));
        }
        let stride = strides[i].max(1);
        let count = if size == 0 { 0 } else { (size - 1) / stride + 1 };
        output.push(count.to_string());
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
            let total = padded_size(d, *lo, *hi)
                .zip(interior.checked_mul((d - 1).max(0)))
                .and_then(|(padded, inserted)| padded.checked_add(inserted))
                .filter(|total| *total >= 0);
            let Some(total) = total else {
                return Ok(None);
            };
            output.push(total.to_string());
        } else if *interior == 0 {
            let Some(added) = lo.checked_add(*hi) else {
                return Ok(None);
            };
            output.push(add_to_dim(dim, added));
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
        let Some(padded) = padded_size(d, lo, hi) else {
            return Ok(None);
        };
        if padded < w {
            return Err(format!(
                "reduce_window: padded dim {} smaller than window {}",
                padded, w
            ));
        }
        let Some(count) = window_count(padded, w, s) else {
            return Ok(None);
        };
        output.push(count.to_string());
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
    if nth_positional_or_keyword(args, 6, &["dimension_numbers"]).is_some_and(|numbers| numbers.trim() != "None") {
        return Ok(None);
    }
    let Some((_, lhs_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(rhs_shape) = shape_arg(args, 1, &["rhs"], shapes) else {
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
        let s = strides[i].max(1);
        let out = match &padding {
            Some(pads) => {
                let (lo, hi) = pads[i];
                dilated_window(k, dilation[i])
                    .zip(padded_size(d, lo, hi))
                    .and_then(|(eff_k, padded)| window_count(padded, eff_k, s))
            }
            None => same_size(d, s),
        };
        let Some(out) = out else {
            return Ok(None);
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
    let Some((_, operand_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let Some(indices_shape) = shape_arg(args, 1, &["start_indices"], shapes) else {
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
    if bracket_inner(slice_sizes_raw.trim()).is_none() {
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
    // `sparse=True` gives a tuple of arrays.
    if !flag_unset(args, 2, "sparse") {
        return Ok(None);
    }
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
    let Some(axis) = axis_at(args, 1, &["axis"], None, rank, "rollaxis")? else {
        return Ok(None);
    };
    let start = nth_positional_or_keyword(args, 2, &["start"])
        .and_then(parse_axis)
        .unwrap_or(0);
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
    let Some(axis) = axis_at(args, 3, &["axis"], None, input_shape.len(), "insert")? else {
        return Ok(None);
    };
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
    let Some(axis) = axis_at(args, 2, &["axis"], None, input_shape.len(), "delete")? else {
        return Ok(None);
    };
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
            let axis = normalize_axis(axis, arr_shape.len(), "append")?;
            concat_shapes_along_axis(&[arr_shape.clone(), val_shape.clone()], axis)
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

fn apply_known_cross(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(a) = shape_arg(args, 0, &["a"], shapes) else {
        return Ok(None);
    };
    let Some(b) = shape_arg(args, 1, &["b"], shapes) else {
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
    let Some(a) = shape_arg(args, 0, &["input"], shapes) else {
        return Ok(None);
    };
    let Some(b) = shape_arg(args, 1, &["other"], shapes) else {
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
    let Some(equation_str) = string_text(equation) else {
        return Ok(None);
    };
    let Some((inputs_part, output_part)) = equation_str.split_once("->") else {
        return Ok(None);
    };
    // Space-separated axis names. `...`, `(...)` and anonymous axes stay unknown.
    let Some(input_specs) = inputs_part.split(',').map(einops_axis_names).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let Some(output_spec) = einops_axis_names(output_part) else {
        return Ok(None);
    };
    einsum_output_shape("einops.einsum", &input_specs, &output_spec, operand_names, shapes)
}

/// The space-separated axis names of one einops.einsum operand. `None` for
/// `...`, `(...)`, anonymous axes and repeated names.
fn einops_axis_names(spec: &str) -> Option<Vec<&str>> {
    let names: Vec<&str> = spec.split_whitespace().collect();
    let valid = |name: &str| {
        name.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
    };
    let unique = names.iter().enumerate().all(|(i, name)| !names[..i].contains(name));
    (names.iter().all(|name| valid(name)) && unique).then_some(names)
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
    let Some(pattern) = positional.get(1).and_then(|raw| string_text(raw)) else {
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
    let Some(dim) = axis_at(args, 1, &["dim"], None, input_shape.len(), "index_select")? else {
        return Ok(None);
    };
    let Some(index_shape) = shape_arg(args, 2, &["index"], shapes) else {
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
    let Some(dim) = axis_at(args, 1, &["dim"], None, input_shape.len(), "narrow")? else {
        return Ok(None);
    };
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
    let Some(dim) = axis_at(args, 1, &["dim"], None, input_shape.len(), "select")? else {
        return Ok(None);
    };
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
    let Some(dim) = axis_at(args, 1, &["dimension", "dim"], None, input_shape.len(), "unfold")? else {
        return Ok(None);
    };
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
    let Some(dim) = axis_at(args, 2, &["dim", "axis"], Some("-1"), rank, "topk")? else {
        return Ok(None);
    };
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
    synthesized.extend(args.iter().filter(|arg| {
        matches!(arg, CallArgument::Keyword { name, .. } if name == "keepdim" || name == "keepdims")
    }).cloned());
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
    star: bool,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let rank = input_shape.len();
    if rank == 0 {
        return Err("unbind requires rank >= 1 input".to_string());
    }
    let Some(dim) = axis_at(args, 1, &["dim", "axis"], Some("0"), rank, "unbind")? else {
        return Ok(None);
    };
    let Ok(size) = input_shape[dim].parse::<usize>() else {
        return Ok(None);
    };
    if star && size + 1 < n_targets {
        return Err(format!(
            "unbind along dim {} produces {} outputs, but at least {} were unpacked",
            dim, size, n_targets - 1
        ));
    }
    if !star && size != n_targets {
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
    let Some(dim) = axis_at(args, 2, &["dim"], Some("0"), rank, "chunk")? else {
        return Ok(None);
    };
    let axis_dim = &input_shape[dim];

    // A symbolic size gives `n` equal pieces only when `n` divides it as a
    // factor. Else torch may give fewer pieces, and a smaller last piece.
    let Ok(axis_size) = axis_dim.parse::<usize>() else {
        if n > MAX_PIECES {
            return Ok(None);
        }
        let mut out = input_shape.clone();
        out[dim] = cancel_product_factor(axis_dim, n).unwrap_or_else(|| UNKNOWN_DIM.to_string());
        return Ok(Some(vec![out; n]));
    };

    let chunk_size = axis_size.div_ceil(n).max(1);
    if axis_size.div_ceil(chunk_size) > MAX_PIECES {
        return Ok(None);
    }
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

fn n_choose_r(n: u64, r: u64) -> Option<u64> {
    if r > n {
        return Some(0);
    }
    let r = r.min(n - r);
    let mut result: u64 = 1;
    for i in 0..r {
        result = result.checked_mul(n - i)? / (i + 1);
    }
    Some(result)
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
    Ok(n_choose_r(n, r).map(|count| vec![count.to_string(), r.to_string()]))
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
    let total = if lens.iter().all(|len| len.parse::<usize>().is_ok()) {
        dim_product(&lens).map_or(UNKNOWN_DIM.to_string(), |total| total.to_string())
    } else {
        joined_dim(&lens, "*")
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
    let Some(weight_shape) = shape_arg(args, 1, &["weight"], shapes) else {
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
        let count = dilated_window(k, dilation[i])
            .zip(padded_size(d, padding[i], padding[i]))
            .and_then(|(eff_k, padded)| window_count(padded, eff_k, s));
        let Some(count) = count else {
            return Ok(None);
        };
        output[idx] = count.to_string();
    }
    Ok(Some(output))
}

/// `torch.nn.functional.max_pool1d/2d/3d` / `avg_pool1d/2d/3d(input,
/// kernel_size, stride=None, padding=0, ...)` — channels-first; `stride`
/// defaults to `kernel_size` (torch convention) when omitted. `ceil_mode` and
/// max_pool's `return_indices` (a tuple) give unknown. A window larger than
/// the input is unknown, as torch refuses it.
fn apply_known_functional_pool(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    spatial_rank: usize,
    max: bool,
) -> Result<Option<Vec<String>>, String> {
    let Some((_, input_shape)) = first_array_arg_shape(args, shapes) else {
        return Ok(None);
    };
    let ceil_at = if max { 5 } else { 4 };
    if !flag_unset(args, ceil_at, "ceil_mode") || (max && !flag_unset(args, 6, "return_indices")) {
        return Ok(None);
    }
    let dilation = if max { nth_positional_or_keyword(args, 4, &["dilation"]) } else { None };
    let Some(dilation) = parse_int_seq_arg(dilation, spatial_rank, 1).filter(|d| d.iter().all(|d| *d >= 1)) else {
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
        let Some(window) = kernel[i].checked_sub(1).and_then(|k| k.checked_mul(dilation[i])).and_then(|k| k.checked_add(1)) else {
            return Ok(None);
        };
        let count = padded_size(d, padding[i], padding[i]).and_then(|padded| window_count(padded, window, s));
        let Some(count) = count else {
            return Ok(None);
        };
        output[idx] = count.to_string();
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
    let Some(weight_shape) = shape_arg(args, 1, &["weight"], shapes) else {
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

#[cfg(test)]
mod signature_tests;
