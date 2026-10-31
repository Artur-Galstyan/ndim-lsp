use super::*;
use crate::{build_import_map, resolve_call_target};
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
    jnp_empty,
    ["jax", "numpy", "empty"],
    Some(KnownFunction::Empty)
);
known_case!(
    jnp_linspace,
    ["jax", "numpy", "linspace"],
    Some(KnownFunction::Linspace)
);
known_case!(
    jnp_logspace,
    ["jax", "numpy", "logspace"],
    Some(KnownFunction::Logspace)
);
known_case!(
    jnp_identity,
    ["jax", "numpy", "identity"],
    Some(KnownFunction::Identity)
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
known_case!(
    np_array_split,
    ["numpy", "array_split"],
    Some(KnownFunction::ArraySplit)
);
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
known_case!(np_empty, ["numpy", "empty"], Some(KnownFunction::Empty));
known_case!(
    np_linspace,
    ["numpy", "linspace"],
    Some(KnownFunction::Linspace)
);
known_case!(
    np_logspace,
    ["numpy", "logspace"],
    Some(KnownFunction::Logspace)
);
known_case!(
    np_identity,
    ["numpy", "identity"],
    Some(KnownFunction::Identity)
);
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
known_case!(np_pad, ["numpy", "pad"], Some(KnownFunction::NumpyPad));
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
    Some(KnownFunction::SwapAxes)
);
known_case!(
    torch_swapdims,
    ["torch", "swapdims"],
    Some(KnownFunction::SwapAxes)
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
known_case!(
    torch_max,
    ["torch", "max"],
    Some(KnownFunction::TorchMaxMin)
);
known_case!(
    torch_min,
    ["torch", "min"],
    Some(KnownFunction::TorchMaxMin)
);
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
known_case!(
    torch_split,
    ["torch", "split"],
    Some(KnownFunction::TorchSplit)
);
known_case!(
    torch_tensor_split,
    ["torch", "tensor_split"],
    Some(KnownFunction::ArraySplit)
);
known_case!(torch_tile, ["torch", "tile"], Some(KnownFunction::Tile));
known_case!(
    torch_repeat,
    ["torch", "repeat"],
    Some(KnownFunction::Repeat)
);
known_case!(
    torch_flatten,
    ["torch", "flatten"],
    Some(KnownFunction::TorchFlatten)
);
known_case!(torch_ravel, ["torch", "ravel"], Some(KnownFunction::Ravel));
known_case!(torch_where, ["torch", "where"], Some(KnownFunction::Where));
known_case!(
    torch_zeros,
    ["torch", "zeros"],
    Some(KnownFunction::TorchSizeConstructor)
);
known_case!(
    torch_ones,
    ["torch", "ones"],
    Some(KnownFunction::TorchSizeConstructor)
);
known_case!(
    torch_rand,
    ["torch", "rand"],
    Some(KnownFunction::TorchSizeConstructor)
);
known_case!(
    torch_randn,
    ["torch", "randn"],
    Some(KnownFunction::TorchSizeConstructor)
);
known_case!(torch_full, ["torch", "full"], Some(KnownFunction::Full));
known_case!(
    torch_empty,
    ["torch", "empty"],
    Some(KnownFunction::TorchSizeConstructor)
);
known_case!(
    torch_linspace,
    ["torch", "linspace"],
    Some(KnownFunction::Linspace)
);
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
// Was a locking test for an unclassified gap (`F.relu` fell through to
// `None`); the full activation family is now classified as
// shape-preserving (`KnownFunction::Copy`), same rule as
// `softmax`/`log_softmax`/`normalize` right above.
known_case!(
    torch_nn_functional_relu_classified_shape_preserving,
    ["torch", "nn", "functional", "relu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_gelu_classified_shape_preserving,
    ["torch", "nn", "functional", "gelu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_silu_classified_shape_preserving,
    ["torch", "nn", "functional", "silu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_sigmoid_classified_shape_preserving,
    ["torch", "nn", "functional", "sigmoid"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_tanh_classified_shape_preserving,
    ["torch", "nn", "functional", "tanh"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_softplus_classified_shape_preserving,
    ["torch", "nn", "functional", "softplus"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_softsign_classified_shape_preserving,
    ["torch", "nn", "functional", "softsign"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_selu_classified_shape_preserving,
    ["torch", "nn", "functional", "selu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_celu_classified_shape_preserving,
    ["torch", "nn", "functional", "celu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_elu_classified_shape_preserving,
    ["torch", "nn", "functional", "elu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_leaky_relu_classified_shape_preserving,
    ["torch", "nn", "functional", "leaky_relu"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_hardtanh_classified_shape_preserving,
    ["torch", "nn", "functional", "hardtanh"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_hardswish_classified_shape_preserving,
    ["torch", "nn", "functional", "hardswish"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_hardsigmoid_classified_shape_preserving,
    ["torch", "nn", "functional", "hardsigmoid"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_mish_classified_shape_preserving,
    ["torch", "nn", "functional", "mish"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_relu6_classified_shape_preserving,
    ["torch", "nn", "functional", "relu6"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_dropout_classified_shape_preserving,
    ["torch", "nn", "functional", "dropout"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_dropout2d_classified_shape_preserving,
    ["torch", "nn", "functional", "dropout2d"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_dropout3d_classified_shape_preserving,
    ["torch", "nn", "functional", "dropout3d"],
    Some(KnownFunction::Copy)
);
// `glu` halves a dim rather than preserving shape — not `Copy`.
known_case!(
    torch_nn_functional_glu_classified,
    ["torch", "nn", "functional", "glu"],
    Some(KnownFunction::FunctionalGlu)
);
known_case!(
    torch_nn_functional_unknown_not_classified,
    ["torch", "nn", "functional", "unknown_func"],
    None
);

known_case!(jax_vmap, ["jax", "vmap"], Some(KnownFunction::Vmap));
known_case!(
    equinox_filter_vmap,
    ["equinox", "filter_vmap"],
    Some(KnownFunction::Vmap)
);
// equinox.nn.filter_vmap is NOT a known function — wrong module path
known_case!(
    equinox_nn_filter_vmap_rejected,
    ["equinox", "nn", "filter_vmap"],
    None
);
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

known_case!(
    jnp_argmax,
    ["jax", "numpy", "argmax"],
    Some(KnownFunction::ArgMax)
);
known_case!(np_argmax, ["numpy", "argmax"], Some(KnownFunction::ArgMax));
known_case!(
    torch_argmax,
    ["torch", "argmax"],
    Some(KnownFunction::ArgMax)
);

known_case!(
    jnp_argmin,
    ["jax", "numpy", "argmin"],
    Some(KnownFunction::ArgMin)
);
known_case!(np_argmin, ["numpy", "argmin"], Some(KnownFunction::ArgMin));
known_case!(
    torch_argmin,
    ["torch", "argmin"],
    Some(KnownFunction::ArgMin)
);

known_case!(
    jnp_argsort,
    ["jax", "numpy", "argsort"],
    Some(KnownFunction::Argsort)
);
known_case!(
    np_argsort,
    ["numpy", "argsort"],
    Some(KnownFunction::Argsort)
);
known_case!(
    torch_argsort,
    ["torch", "argsort"],
    Some(KnownFunction::Argsort)
);

known_case!(
    jnp_sort,
    ["jax", "numpy", "sort"],
    Some(KnownFunction::Sort)
);
known_case!(np_sort, ["numpy", "sort"], Some(KnownFunction::Sort));
known_case!(torch_sort, ["torch", "sort"], Some(KnownFunction::Sort));

known_case!(
    jnp_cumsum,
    ["jax", "numpy", "cumsum"],
    Some(KnownFunction::Cumsum)
);
known_case!(np_cumsum, ["numpy", "cumsum"], Some(KnownFunction::Cumsum));
known_case!(
    torch_cumsum,
    ["torch", "cumsum"],
    Some(KnownFunction::Cumsum)
);

known_case!(
    jnp_cumprod,
    ["jax", "numpy", "cumprod"],
    Some(KnownFunction::Cumprod)
);
known_case!(
    np_cumprod,
    ["numpy", "cumprod"],
    Some(KnownFunction::Cumprod)
);
known_case!(
    torch_cumprod,
    ["torch", "cumprod"],
    Some(KnownFunction::Cumprod)
);

known_case!(jax_numpy_vmap_rejected, ["jax", "numpy", "vmap"], None);
known_case!(numpy_vmap_rejected, ["numpy", "vmap"], None);
known_case!(torch_vmap_rejected_for_now, ["torch", "vmap"], None);
known_case!(torch_nn_function_rejected, ["torch", "nn", "Linear"], None);
known_case!(
    numpy_linalg_dot_rejected_for_now,
    ["numpy", "linalg", "dot"],
    None
);
// `linalg.norm` reuses the plain-reduction shape rule (axis/keepdims
// mechanics are identical); see `linalg_norm_reduction_tests` below.
known_case!(
    jax_numpy_linalg_norm_classified,
    ["jax", "numpy", "linalg", "norm"],
    Some(KnownFunction::Sum)
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
known_case!(jax_linalg_inv_unsupported, ["jax", "linalg", "inv"], None);

// ── linalg.det classification tests ──

known_case!(
    jnp_linalg_det,
    ["jax", "numpy", "linalg", "det"],
    Some(KnownFunction::LinalgDet)
);
known_case!(
    np_linalg_det,
    ["numpy", "linalg", "det"],
    Some(KnownFunction::LinalgDet)
);
known_case!(
    torch_linalg_det,
    ["torch", "linalg", "det"],
    Some(KnownFunction::LinalgDet)
);
known_case!(jax_linalg_det_unsupported, ["jax", "linalg", "det"], None);

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
// equinox.filter_vmap alias tests
alias_case!(
    alias_equinox_filter_vmap,
    "import equinox",
    "equinox.filter_vmap",
    Some(KnownFunction::Vmap)
);
alias_case!(
    alias_equinox_as_eqx_filter_vmap,
    "import equinox as eqx",
    "eqx.filter_vmap",
    Some(KnownFunction::Vmap)
);
alias_case!(
    from_import_equinox_filter_vmap,
    "from equinox import filter_vmap",
    "filter_vmap",
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
    alias_torch_nn_functional_imported_as_f_relu_classified_shape_preserving,
    "import torch.nn.functional as F",
    "F.relu",
    Some(KnownFunction::Copy)
);

// ── jax.lax higher-order / structured op classification ──

known_case!(
    jax_lax_map,
    ["jax", "lax", "map"],
    Some(KnownFunction::LaxMap)
);
known_case!(
    jax_lax_cond,
    ["jax", "lax", "cond"],
    Some(KnownFunction::LaxCond)
);
known_case!(
    jax_lax_switch,
    ["jax", "lax", "switch"],
    Some(KnownFunction::LaxSwitch)
);
known_case!(
    jax_lax_while_loop,
    ["jax", "lax", "while_loop"],
    Some(KnownFunction::LaxWhileLoop)
);
known_case!(
    jax_lax_fori_loop,
    ["jax", "lax", "fori_loop"],
    Some(KnownFunction::LaxForiLoop)
);
known_case!(
    jax_lax_conv_general_dilated,
    ["jax", "lax", "conv_general_dilated"],
    Some(KnownFunction::LaxConvGeneralDilated)
);
known_case!(
    jax_lax_gather,
    ["jax", "lax", "gather"],
    Some(KnownFunction::LaxGather)
);
known_case!(
    jax_lax_scatter,
    ["jax", "lax", "scatter"],
    Some(KnownFunction::LaxScatter)
);
known_case!(
    jax_lax_scatter_add,
    ["jax", "lax", "scatter_add"],
    Some(KnownFunction::LaxScatter)
);
known_case!(
    jax_lax_reduce_window,
    ["jax", "lax", "reduce_window"],
    Some(KnownFunction::LaxReduceWindow)
);
known_case!(
    jax_lax_top_k,
    ["jax", "lax", "top_k"],
    Some(KnownFunction::LaxTopK)
);
known_case!(
    jax_lax_sort,
    ["jax", "lax", "sort"],
    Some(KnownFunction::LaxSort)
);
known_case!(
    jax_lax_sort_key_val,
    ["jax", "lax", "sort_key_val"],
    Some(KnownFunction::LaxSortKeyVal)
);
known_case!(
    jax_lax_pad,
    ["jax", "lax", "pad"],
    Some(KnownFunction::LaxPad)
);
known_case!(
    jax_lax_broadcast,
    ["jax", "lax", "broadcast"],
    Some(KnownFunction::LaxBroadcast)
);
known_case!(
    jax_lax_broadcast_in_dim,
    ["jax", "lax", "broadcast_in_dim"],
    Some(KnownFunction::LaxBroadcastInDim)
);
known_case!(
    jax_lax_slice,
    ["jax", "lax", "slice"],
    Some(KnownFunction::LaxSlice)
);
known_case!(
    jax_lax_dynamic_slice,
    ["jax", "lax", "dynamic_slice"],
    Some(KnownFunction::LaxDynamicSlice)
);
known_case!(
    jax_lax_dynamic_update_slice,
    ["jax", "lax", "dynamic_update_slice"],
    Some(KnownFunction::LaxDynamicUpdateSlice)
);
known_case!(
    jax_lax_concatenate,
    ["jax", "lax", "concatenate"],
    Some(KnownFunction::Concatenate)
);
known_case!(
    jax_lax_rev,
    ["jax", "lax", "rev"],
    Some(KnownFunction::Flip)
);
known_case!(
    jax_lax_squeeze,
    ["jax", "lax", "squeeze"],
    Some(KnownFunction::Squeeze)
);
known_case!(
    jax_lax_expand_dims,
    ["jax", "lax", "expand_dims"],
    Some(KnownFunction::ExpandDims)
);
known_case!(
    jax_lax_transpose,
    ["jax", "lax", "transpose"],
    Some(KnownFunction::Transpose)
);
known_case!(
    jax_lax_associative_scan,
    ["jax", "lax", "associative_scan"],
    Some(KnownFunction::LaxAssociativeScan)
);
known_case!(jax_lax_psum_not_classified, ["jax", "lax", "psum"], None);
known_case!(jax_jit_not_classified, ["jax", "jit"], None);
known_case!(jax_grad_not_classified, ["jax", "grad"], None);

// ── jax.numpy / numpy array-creation classification ──

known_case!(
    jnp_diagflat,
    ["jax", "numpy", "diagflat"],
    Some(KnownFunction::Diagflat)
);
known_case!(jnp_tri, ["jax", "numpy", "tri"], Some(KnownFunction::Tri));
known_case!(
    jnp_indices,
    ["jax", "numpy", "indices"],
    Some(KnownFunction::Indices)
);
known_case!(
    jnp_bincount,
    ["jax", "numpy", "bincount"],
    Some(KnownFunction::BinCount)
);
known_case!(
    jnp_unique,
    ["jax", "numpy", "unique"],
    Some(KnownFunction::Unique)
);
known_case!(
    jnp_select,
    ["jax", "numpy", "select"],
    Some(KnownFunction::Select)
);

// ── jax.numpy / numpy shape-transform classification ──

known_case!(
    jnp_rollaxis,
    ["jax", "numpy", "rollaxis"],
    Some(KnownFunction::RollAxis)
);
known_case!(
    jnp_resize,
    ["jax", "numpy", "resize"],
    Some(KnownFunction::Resize)
);
known_case!(
    jnp_insert,
    ["jax", "numpy", "insert"],
    Some(KnownFunction::Insert)
);
known_case!(
    jnp_delete,
    ["jax", "numpy", "delete"],
    Some(KnownFunction::Delete)
);
known_case!(
    jnp_append,
    ["jax", "numpy", "append"],
    Some(KnownFunction::Append)
);
known_case!(
    jnp_broadcast_shapes_not_classified,
    ["jax", "numpy", "broadcast_shapes"],
    None
);

// ── jax.numpy / numpy joining-and-splitting classification ──

known_case!(
    jnp_hsplit,
    ["jax", "numpy", "hsplit"],
    Some(KnownFunction::HSplit)
);
known_case!(
    jnp_vsplit,
    ["jax", "numpy", "vsplit"],
    Some(KnownFunction::VSplit)
);
known_case!(
    jnp_dsplit,
    ["jax", "numpy", "dsplit"],
    Some(KnownFunction::DSplit)
);
known_case!(
    jnp_kron,
    ["jax", "numpy", "kron"],
    Some(KnownFunction::Kron)
);

// ── jax.numpy / numpy indexing-and-selection classification ──

known_case!(
    jnp_take_along_axis,
    ["jax", "numpy", "take_along_axis"],
    Some(KnownFunction::TakeAlongAxis)
);
known_case!(
    jnp_put_along_axis,
    ["jax", "numpy", "put_along_axis"],
    Some(KnownFunction::PutAlongAxis)
);
known_case!(
    jnp_nonzero,
    ["jax", "numpy", "nonzero"],
    Some(KnownFunction::Nonzero)
);
known_case!(
    jnp_argwhere,
    ["jax", "numpy", "argwhere"],
    Some(KnownFunction::Argwhere)
);
known_case!(
    jnp_searchsorted,
    ["jax", "numpy", "searchsorted"],
    Some(KnownFunction::SearchSorted)
);
known_case!(
    jnp_extract,
    ["jax", "numpy", "extract"],
    Some(KnownFunction::Extract)
);
known_case!(
    jnp_compress,
    ["jax", "numpy", "compress"],
    Some(KnownFunction::Compress)
);
known_case!(
    jnp_histogram,
    ["jax", "numpy", "histogram"],
    Some(KnownFunction::Histogram)
);

// ── jax.numpy / numpy reduction-alias classification ──

known_case!(
    jnp_median_reuses_mean,
    ["jax", "numpy", "median"],
    Some(KnownFunction::Mean)
);
known_case!(
    jnp_quantile_reuses_mean,
    ["jax", "numpy", "quantile"],
    Some(KnownFunction::Mean)
);
known_case!(
    jnp_count_nonzero_reuses_sum,
    ["jax", "numpy", "count_nonzero"],
    Some(KnownFunction::Sum)
);
known_case!(
    jnp_ptp_reuses_max,
    ["jax", "numpy", "ptp"],
    Some(KnownFunction::Max)
);
known_case!(
    jax_numpy_fft_fft_classified,
    ["jax", "numpy", "fft", "fft"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_fft_rfft_not_classified,
    ["torch", "fft", "rfft"],
    None
);

// ── linear algebra classification ──

known_case!(
    jnp_cross,
    ["jax", "numpy", "cross"],
    Some(KnownFunction::Cross)
);
known_case!(
    torch_cross,
    ["torch", "cross"],
    Some(KnownFunction::TorchCross)
);
known_case!(
    jnp_linalg_solve,
    ["jax", "numpy", "linalg", "solve"],
    Some(KnownFunction::LinalgSolve)
);
known_case!(
    jnp_linalg_cholesky_reuses_inv,
    ["jax", "numpy", "linalg", "cholesky"],
    Some(KnownFunction::LinalgInv)
);
known_case!(
    torch_linalg_lstsq,
    ["torch", "linalg", "lstsq"],
    Some(KnownFunction::LinalgLstsq)
);
known_case!(
    torch_linalg_pinv,
    ["torch", "linalg", "pinv"],
    Some(KnownFunction::LinalgPinv)
);
known_case!(
    torch_linalg_matrix_rank,
    ["torch", "linalg", "matrix_rank"],
    Some(KnownFunction::LinalgMatrixRank)
);

// ── einops classification ──

known_case!(
    einops_einsum,
    ["einops", "einsum"],
    Some(KnownFunction::EinopsEinsum)
);
known_case!(
    einops_pack,
    ["einops", "pack"],
    Some(KnownFunction::EinopsPack)
);
known_case!(
    einops_unpack,
    ["einops", "unpack"],
    Some(KnownFunction::EinopsUnpack)
);
known_case!(
    einops_parse_shape,
    ["einops", "parse_shape"],
    Some(KnownFunction::EinopsParseShape)
);

// ── jax.nn classification ──

known_case!(
    jax_nn_one_hot,
    ["jax", "nn", "one_hot"],
    Some(KnownFunction::OneHot)
);
known_case!(
    jax_nn_dot_product_attention,
    ["jax", "nn", "dot_product_attention"],
    Some(KnownFunction::DotProductAttention)
);
known_case!(
    jax_nn_logsumexp_reuses_sum,
    ["jax", "nn", "logsumexp"],
    Some(KnownFunction::Sum)
);
known_case!(jax_nn_relu_not_classified, ["jax", "nn", "relu"], None);

known_case!(
    torch_gather,
    ["torch", "gather"],
    Some(KnownFunction::Gather)
);
known_case!(
    torch_scatter,
    ["torch", "scatter"],
    Some(KnownFunction::Scatter)
);
known_case!(
    torch_scatter_add,
    ["torch", "scatter_add"],
    Some(KnownFunction::Scatter)
);
known_case!(
    torch_take_along_dim,
    ["torch", "take_along_dim"],
    Some(KnownFunction::TakeAlongAxis)
);
known_case!(torch_topk, ["torch", "topk"], Some(KnownFunction::TopK));
known_case!(
    torch_unbind,
    ["torch", "unbind"],
    Some(KnownFunction::Unbind)
);
known_case!(torch_chunk, ["torch", "chunk"], Some(KnownFunction::Chunk));
known_case!(
    torch_narrow,
    ["torch", "narrow"],
    Some(KnownFunction::Narrow)
);
known_case!(
    torch_select,
    ["torch", "select"],
    Some(KnownFunction::SelectDim)
);
known_case!(
    torch_masked_select,
    ["torch", "masked_select"],
    Some(KnownFunction::MaskedSelect)
);
known_case!(
    torch_index_select,
    ["torch", "index_select"],
    Some(KnownFunction::IndexSelect)
);
known_case!(
    torch_kthvalue,
    ["torch", "kthvalue"],
    Some(KnownFunction::KthValue)
);
known_case!(
    torch_median,
    ["torch", "median"],
    Some(KnownFunction::MedianDim)
);
known_case!(
    torch_mode,
    ["torch", "mode"],
    Some(KnownFunction::MedianDim)
);
known_case!(
    torch_unique,
    ["torch", "unique"],
    Some(KnownFunction::Unique)
);
known_case!(
    torch_combinations,
    ["torch", "combinations"],
    Some(KnownFunction::Combinations)
);
known_case!(
    torch_cartesian_prod,
    ["torch", "cartesian_prod"],
    Some(KnownFunction::CartesianProd)
);
known_case!(
    torch_block_diag,
    ["torch", "block_diag"],
    Some(KnownFunction::BlockDiag)
);
known_case!(
    torch_nn_functional_interpolate,
    ["torch", "nn", "functional", "interpolate"],
    Some(KnownFunction::Interpolate)
);
known_case!(
    torch_nn_functional_conv1d,
    ["torch", "nn", "functional", "conv1d"],
    Some(KnownFunction::FunctionalConv1d)
);
known_case!(
    torch_nn_functional_conv2d,
    ["torch", "nn", "functional", "conv2d"],
    Some(KnownFunction::FunctionalConv2d)
);
known_case!(
    torch_nn_functional_conv3d,
    ["torch", "nn", "functional", "conv3d"],
    Some(KnownFunction::FunctionalConv3d)
);
known_case!(
    torch_nn_functional_max_pool1d,
    ["torch", "nn", "functional", "max_pool1d"],
    Some(KnownFunction::FunctionalMaxPool1d)
);
known_case!(
    torch_nn_functional_max_pool2d,
    ["torch", "nn", "functional", "max_pool2d"],
    Some(KnownFunction::FunctionalMaxPool2d)
);
known_case!(
    torch_nn_functional_max_pool3d,
    ["torch", "nn", "functional", "max_pool3d"],
    Some(KnownFunction::FunctionalMaxPool3d)
);
known_case!(
    torch_nn_functional_avg_pool1d,
    ["torch", "nn", "functional", "avg_pool1d"],
    Some(KnownFunction::FunctionalAvgPool1d)
);
known_case!(
    torch_nn_functional_avg_pool2d,
    ["torch", "nn", "functional", "avg_pool2d"],
    Some(KnownFunction::FunctionalAvgPool2d)
);
known_case!(
    torch_nn_functional_avg_pool3d,
    ["torch", "nn", "functional", "avg_pool3d"],
    Some(KnownFunction::FunctionalAvgPool3d)
);
known_case!(
    torch_nn_functional_softmax,
    ["torch", "nn", "functional", "softmax"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_log_softmax,
    ["torch", "nn", "functional", "log_softmax"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_normalize,
    ["torch", "nn", "functional", "normalize"],
    Some(KnownFunction::Copy)
);
known_case!(
    torch_nn_functional_one_hot,
    ["torch", "nn", "functional", "one_hot"],
    Some(KnownFunction::OneHot)
);
known_case!(
    torch_nn_functional_embedding,
    ["torch", "nn", "functional", "embedding"],
    Some(KnownFunction::FunctionalEmbedding)
);
known_case!(
    torch_nn_utils_rnn_pad_sequence,
    ["torch", "nn", "utils", "rnn", "pad_sequence"],
    Some(KnownFunction::PadSequence)
);
