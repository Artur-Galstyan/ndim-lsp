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

macro_rules! known_cases {
    ($($name:ident, [$($part:expr),*], $kind:expr;)*) => {
        $(
            #[test]
            fn $name() {
                assert_eq!(classify_known_function(&target(&[$($part),*])), $kind);
            }
        )*
    };
}

known_cases! {
    jnp_concatenate, ["jax", "numpy", "concatenate"], Some(KnownFunction::Concatenate);
    jnp_concat, ["jax", "numpy", "concat"], Some(KnownFunction::Concatenate);
    jnp_stack, ["jax", "numpy", "stack"], Some(KnownFunction::Stack);
    jnp_reshape, ["jax", "numpy", "reshape"], Some(KnownFunction::Reshape);
    jnp_transpose, ["jax", "numpy", "transpose"], Some(KnownFunction::Transpose);
    jnp_expand_dims, ["jax", "numpy", "expand_dims"], Some(KnownFunction::ExpandDims);
    jnp_squeeze, ["jax", "numpy", "squeeze"], Some(KnownFunction::Squeeze);
    jnp_sum, ["jax", "numpy", "sum"], Some(KnownFunction::Sum);
    jnp_mean, ["jax", "numpy", "mean"], Some(KnownFunction::Mean);
    jnp_max, ["jax", "numpy", "max"], Some(KnownFunction::Max);
    jnp_amax, ["jax", "numpy", "amax"], Some(KnownFunction::Max);
    jnp_min, ["jax", "numpy", "min"], Some(KnownFunction::Min);
    jnp_amin, ["jax", "numpy", "amin"], Some(KnownFunction::Min);
    jnp_prod, ["jax", "numpy", "prod"], Some(KnownFunction::Prod);
    jnp_std, ["jax", "numpy", "std"], Some(KnownFunction::Std);
    jnp_var, ["jax", "numpy", "var"], Some(KnownFunction::Var);
    jnp_matmul, ["jax", "numpy", "matmul"], Some(KnownFunction::Matmul);
    jnp_dot, ["jax", "numpy", "dot"], Some(KnownFunction::Dot);
    jnp_einsum, ["jax", "numpy", "einsum"], Some(KnownFunction::Einsum);
    jnp_split, ["jax", "numpy", "split"], Some(KnownFunction::Split);
    jnp_tile, ["jax", "numpy", "tile"], Some(KnownFunction::Tile);
    jnp_repeat, ["jax", "numpy", "repeat"], Some(KnownFunction::Repeat);
    jnp_flatten, ["jax", "numpy", "flatten"], Some(KnownFunction::Flatten);
    jnp_ravel, ["jax", "numpy", "ravel"], Some(KnownFunction::Ravel);
    jnp_moveaxis, ["jax", "numpy", "moveaxis"], Some(KnownFunction::MoveAxis);
    jnp_swapaxes, ["jax", "numpy", "swapaxes"], Some(KnownFunction::SwapAxes);
    jnp_where, ["jax", "numpy", "where"], Some(KnownFunction::Where);
    jnp_zeros, ["jax", "numpy", "zeros"], Some(KnownFunction::Zeros);
    jnp_ones, ["jax", "numpy", "ones"], Some(KnownFunction::Ones);
    jnp_full, ["jax", "numpy", "full"], Some(KnownFunction::Full);
    jnp_empty, ["jax", "numpy", "empty"], Some(KnownFunction::Empty);
    jnp_linspace, ["jax", "numpy", "linspace"], Some(KnownFunction::Linspace);
    jnp_logspace, ["jax", "numpy", "logspace"], Some(KnownFunction::Logspace);
    jnp_identity, ["jax", "numpy", "identity"], Some(KnownFunction::Identity);
    jnp_arange, ["jax", "numpy", "arange"], Some(KnownFunction::Arange);
    jnp_eye, ["jax", "numpy", "eye"], Some(KnownFunction::Eye);
    jnp_broadcast_to, ["jax", "numpy", "broadcast_to"], Some(KnownFunction::BroadcastTo);
    jnp_broadcast_arrays, ["jax", "numpy", "broadcast_arrays"], Some(KnownFunction::BroadcastArrays);
    jnp_atleast_1d, ["jax", "numpy", "atleast_1d"], Some(KnownFunction::AtLeast1D);
    jnp_atleast_2d, ["jax", "numpy", "atleast_2d"], Some(KnownFunction::AtLeast2D);
    jnp_atleast_3d, ["jax", "numpy", "atleast_3d"], Some(KnownFunction::AtLeast3D);
    jnp_pad, ["jax", "numpy", "pad"], Some(KnownFunction::Pad);
    jnp_roll, ["jax", "numpy", "roll"], Some(KnownFunction::Roll);
    jnp_flip, ["jax", "numpy", "flip"], Some(KnownFunction::Flip);
    jnp_fliplr, ["jax", "numpy", "fliplr"], Some(KnownFunction::Flip);
    jnp_flipud, ["jax", "numpy", "flipud"], Some(KnownFunction::Flip);
    jnp_rot90, ["jax", "numpy", "rot90"], Some(KnownFunction::Rot90);
    jnp_take, ["jax", "numpy", "take"], Some(KnownFunction::Take);
    jnp_diag, ["jax", "numpy", "diag"], Some(KnownFunction::Diag);
    jnp_diagonal, ["jax", "numpy", "diagonal"], Some(KnownFunction::Diagonal);
    jnp_trace, ["jax", "numpy", "trace"], Some(KnownFunction::Trace);
    jnp_triu, ["jax", "numpy", "triu"], Some(KnownFunction::Triu);
    jnp_tril, ["jax", "numpy", "tril"], Some(KnownFunction::Tril);
    jnp_meshgrid, ["jax", "numpy", "meshgrid"], Some(KnownFunction::Meshgrid);
    jnp_vstack, ["jax", "numpy", "vstack"], Some(KnownFunction::Vstack);
    jnp_row_stack, ["jax", "numpy", "row_stack"], Some(KnownFunction::Vstack);
    jnp_hstack, ["jax", "numpy", "hstack"], Some(KnownFunction::Hstack);
    jnp_dstack, ["jax", "numpy", "dstack"], Some(KnownFunction::Dstack);
    jnp_column_stack, ["jax", "numpy", "column_stack"], Some(KnownFunction::ColumnStack);
    jnp_block, ["jax", "numpy", "block"], Some(KnownFunction::Block);
    jnp_zeros_like, ["jax", "numpy", "zeros_like"], Some(KnownFunction::ZerosLike);
    jnp_ones_like, ["jax", "numpy", "ones_like"], Some(KnownFunction::OnesLike);
    jnp_full_like, ["jax", "numpy", "full_like"], Some(KnownFunction::FullLike);
    jnp_empty_like, ["jax", "numpy", "empty_like"], Some(KnownFunction::EmptyLike);
    jnp_array, ["jax", "numpy", "array"], Some(KnownFunction::Array);
    jnp_asarray, ["jax", "numpy", "asarray"], Some(KnownFunction::AsArray);

    np_concatenate, ["numpy", "concatenate"], Some(KnownFunction::Concatenate);
    np_stack, ["numpy", "stack"], Some(KnownFunction::Stack);
    np_reshape, ["numpy", "reshape"], Some(KnownFunction::Reshape);
    np_transpose, ["numpy", "transpose"], Some(KnownFunction::Transpose);
    np_expand_dims, ["numpy", "expand_dims"], Some(KnownFunction::ExpandDims);
    np_squeeze, ["numpy", "squeeze"], Some(KnownFunction::Squeeze);
    np_sum, ["numpy", "sum"], Some(KnownFunction::Sum);
    np_mean, ["numpy", "mean"], Some(KnownFunction::Mean);
    np_amax, ["numpy", "amax"], Some(KnownFunction::Max);
    np_amin, ["numpy", "amin"], Some(KnownFunction::Min);
    np_prod, ["numpy", "prod"], Some(KnownFunction::Prod);
    np_std, ["numpy", "std"], Some(KnownFunction::Std);
    np_var, ["numpy", "var"], Some(KnownFunction::Var);
    np_matmul, ["numpy", "matmul"], Some(KnownFunction::Matmul);
    np_dot, ["numpy", "dot"], Some(KnownFunction::Dot);
    np_einsum, ["numpy", "einsum"], Some(KnownFunction::Einsum);
    np_split, ["numpy", "split"], Some(KnownFunction::Split);
    np_array_split, ["numpy", "array_split"], Some(KnownFunction::ArraySplit);
    np_tile, ["numpy", "tile"], Some(KnownFunction::Tile);
    np_repeat, ["numpy", "repeat"], Some(KnownFunction::Repeat);
    np_ravel, ["numpy", "ravel"], Some(KnownFunction::Ravel);
    np_moveaxis, ["numpy", "moveaxis"], Some(KnownFunction::MoveAxis);
    np_swapaxes, ["numpy", "swapaxes"], Some(KnownFunction::SwapAxes);
    np_where, ["numpy", "where"], Some(KnownFunction::Where);
    np_zeros, ["numpy", "zeros"], Some(KnownFunction::Zeros);
    np_ones, ["numpy", "ones"], Some(KnownFunction::Ones);
    np_full, ["numpy", "full"], Some(KnownFunction::Full);
    np_empty, ["numpy", "empty"], Some(KnownFunction::Empty);
    np_linspace, ["numpy", "linspace"], Some(KnownFunction::Linspace);
    np_logspace, ["numpy", "logspace"], Some(KnownFunction::Logspace);
    np_identity, ["numpy", "identity"], Some(KnownFunction::Identity);
    np_arange, ["numpy", "arange"], Some(KnownFunction::Arange);
    np_eye, ["numpy", "eye"], Some(KnownFunction::Eye);
    np_broadcast_to, ["numpy", "broadcast_to"], Some(KnownFunction::BroadcastTo);
    np_broadcast_arrays, ["numpy", "broadcast_arrays"], Some(KnownFunction::BroadcastArrays);
    np_atleast_1d, ["numpy", "atleast_1d"], Some(KnownFunction::AtLeast1D);
    np_atleast_2d, ["numpy", "atleast_2d"], Some(KnownFunction::AtLeast2D);
    np_atleast_3d, ["numpy", "atleast_3d"], Some(KnownFunction::AtLeast3D);
    np_pad, ["numpy", "pad"], Some(KnownFunction::NumpyPad);
    np_roll, ["numpy", "roll"], Some(KnownFunction::Roll);
    np_flip, ["numpy", "flip"], Some(KnownFunction::Flip);
    np_fliplr, ["numpy", "fliplr"], Some(KnownFunction::Flip);
    np_flipud, ["numpy", "flipud"], Some(KnownFunction::Flip);
    np_rot90, ["numpy", "rot90"], Some(KnownFunction::Rot90);
    np_take, ["numpy", "take"], Some(KnownFunction::Take);
    np_diag, ["numpy", "diag"], Some(KnownFunction::Diag);
    np_diagonal, ["numpy", "diagonal"], Some(KnownFunction::Diagonal);
    np_trace, ["numpy", "trace"], Some(KnownFunction::Trace);
    np_triu, ["numpy", "triu"], Some(KnownFunction::Triu);
    np_tril, ["numpy", "tril"], Some(KnownFunction::Tril);
    np_meshgrid, ["numpy", "meshgrid"], Some(KnownFunction::Meshgrid);
    np_vstack, ["numpy", "vstack"], Some(KnownFunction::Vstack);
    np_row_stack, ["numpy", "row_stack"], Some(KnownFunction::Vstack);
    np_hstack, ["numpy", "hstack"], Some(KnownFunction::Hstack);
    np_dstack, ["numpy", "dstack"], Some(KnownFunction::Dstack);
    np_column_stack, ["numpy", "column_stack"], Some(KnownFunction::ColumnStack);
    np_block, ["numpy", "block"], Some(KnownFunction::Block);
    np_zeros_like, ["numpy", "zeros_like"], Some(KnownFunction::ZerosLike);
    np_ones_like, ["numpy", "ones_like"], Some(KnownFunction::OnesLike);
    np_full_like, ["numpy", "full_like"], Some(KnownFunction::FullLike);
    np_empty_like, ["numpy", "empty_like"], Some(KnownFunction::EmptyLike);
    np_array, ["numpy", "array"], Some(KnownFunction::Array);
    np_asarray, ["numpy", "asarray"], Some(KnownFunction::AsArray);

    torch_cat, ["torch", "cat"], Some(KnownFunction::Concatenate);
    torch_concat, ["torch", "concat"], Some(KnownFunction::Concatenate);
    torch_concatenate, ["torch", "concatenate"], Some(KnownFunction::Concatenate);
    torch_stack, ["torch", "stack"], Some(KnownFunction::Stack);
    torch_reshape, ["torch", "reshape"], Some(KnownFunction::Reshape);
    torch_transpose, ["torch", "transpose"], Some(KnownFunction::SwapAxes);
    torch_swapdims, ["torch", "swapdims"], Some(KnownFunction::SwapAxes);
    torch_unsqueeze, ["torch", "unsqueeze"], Some(KnownFunction::ExpandDims);
    torch_squeeze, ["torch", "squeeze"], Some(KnownFunction::Squeeze);
    torch_sum, ["torch", "sum"], Some(KnownFunction::Sum);
    torch_mean, ["torch", "mean"], Some(KnownFunction::Mean);
    torch_max, ["torch", "max"], Some(KnownFunction::TorchMaxMin);
    torch_min, ["torch", "min"], Some(KnownFunction::TorchMaxMin);
    torch_prod, ["torch", "prod"], Some(KnownFunction::Prod);
    torch_std, ["torch", "std"], Some(KnownFunction::Std);
    torch_var, ["torch", "var"], Some(KnownFunction::Var);
    torch_matmul, ["torch", "matmul"], Some(KnownFunction::Matmul);
    torch_dot, ["torch", "dot"], Some(KnownFunction::Dot);
    torch_einsum, ["torch", "einsum"], Some(KnownFunction::Einsum);
    torch_split, ["torch", "split"], Some(KnownFunction::TorchSplit);
    torch_tensor_split, ["torch", "tensor_split"], Some(KnownFunction::ArraySplit);
    torch_tile, ["torch", "tile"], Some(KnownFunction::Tile);
    torch_repeat, ["torch", "repeat"], Some(KnownFunction::Repeat);
    torch_flatten, ["torch", "flatten"], Some(KnownFunction::TorchFlatten);
    torch_ravel, ["torch", "ravel"], Some(KnownFunction::Ravel);
    torch_where, ["torch", "where"], Some(KnownFunction::Where);
    torch_zeros, ["torch", "zeros"], Some(KnownFunction::TorchSizeConstructor);
    torch_ones, ["torch", "ones"], Some(KnownFunction::TorchSizeConstructor);
    torch_rand, ["torch", "rand"], Some(KnownFunction::TorchSizeConstructor);
    torch_randn, ["torch", "randn"], Some(KnownFunction::TorchSizeConstructor);
    torch_full, ["torch", "full"], Some(KnownFunction::Full);
    torch_empty, ["torch", "empty"], Some(KnownFunction::TorchSizeConstructor);
    torch_linspace, ["torch", "linspace"], Some(KnownFunction::Linspace);
    torch_zeros_like, ["torch", "zeros_like"], Some(KnownFunction::ZerosLike);
    torch_ones_like, ["torch", "ones_like"], Some(KnownFunction::OnesLike);
    torch_full_like, ["torch", "full_like"], Some(KnownFunction::FullLike);
    torch_empty_like, ["torch", "empty_like"], Some(KnownFunction::EmptyLike);
    torch_arange, ["torch", "arange"], Some(KnownFunction::Arange);
    torch_eye, ["torch", "eye"], Some(KnownFunction::Eye);
    torch_broadcast_to, ["torch", "broadcast_to"], Some(KnownFunction::BroadcastTo);
    torch_broadcast_tensors, ["torch", "broadcast_tensors"], Some(KnownFunction::BroadcastArrays);
    torch_atleast_1d, ["torch", "atleast_1d"], Some(KnownFunction::AtLeast1D);
    torch_atleast_2d, ["torch", "atleast_2d"], Some(KnownFunction::AtLeast2D);
    torch_atleast_3d, ["torch", "atleast_3d"], Some(KnownFunction::AtLeast3D);
    torch_roll, ["torch", "roll"], Some(KnownFunction::Roll);
    torch_flip, ["torch", "flip"], Some(KnownFunction::Flip);
    torch_fliplr, ["torch", "fliplr"], Some(KnownFunction::Flip);
    torch_flipud, ["torch", "flipud"], Some(KnownFunction::Flip);
    torch_rot90, ["torch", "rot90"], Some(KnownFunction::Rot90);
    torch_take, ["torch", "take"], Some(KnownFunction::Take);
    torch_diag, ["torch", "diag"], Some(KnownFunction::Diag);
    torch_diagonal, ["torch", "diagonal"], Some(KnownFunction::Diagonal);
    torch_trace, ["torch", "trace"], Some(KnownFunction::Trace);
    torch_triu, ["torch", "triu"], Some(KnownFunction::Triu);
    torch_tril, ["torch", "tril"], Some(KnownFunction::Tril);
    torch_meshgrid, ["torch", "meshgrid"], Some(KnownFunction::Meshgrid);
    torch_vstack, ["torch", "vstack"], Some(KnownFunction::Vstack);
    torch_row_stack, ["torch", "row_stack"], Some(KnownFunction::Vstack);
    torch_hstack, ["torch", "hstack"], Some(KnownFunction::Hstack);
    torch_dstack, ["torch", "dstack"], Some(KnownFunction::Dstack);
    torch_column_stack, ["torch", "column_stack"], Some(KnownFunction::ColumnStack);
    torch_permute, ["torch", "permute"], Some(KnownFunction::Permute);
    torch_tensor, ["torch", "tensor"], Some(KnownFunction::Array);
    torch_as_tensor, ["torch", "as_tensor"], Some(KnownFunction::AsArray);
    torch_pad_rejected_for_now, ["torch", "pad"], None;
    torch_block_rejected_for_now, ["torch", "block"], None;

    // torch.nn.functional.* — deep module path classification
    torch_nn_functional_pad, ["torch", "nn", "functional", "pad"], Some(KnownFunction::Pad);
    // Was a locking test for an unclassified gap (`F.relu` fell through to
    // `None`); the full activation family is now classified as
    // shape-preserving (`KnownFunction::Copy`), same rule as
    // `softmax`/`log_softmax`/`normalize` right above.
    torch_nn_functional_relu_classified_shape_preserving, ["torch", "nn", "functional", "relu"], Some(KnownFunction::Copy);
    torch_nn_functional_gelu_classified_shape_preserving, ["torch", "nn", "functional", "gelu"], Some(KnownFunction::Copy);
    torch_nn_functional_silu_classified_shape_preserving, ["torch", "nn", "functional", "silu"], Some(KnownFunction::Copy);
    torch_nn_functional_sigmoid_classified_shape_preserving, ["torch", "nn", "functional", "sigmoid"], Some(KnownFunction::Copy);
    torch_nn_functional_tanh_classified_shape_preserving, ["torch", "nn", "functional", "tanh"], Some(KnownFunction::Copy);
    torch_nn_functional_softplus_classified_shape_preserving, ["torch", "nn", "functional", "softplus"], Some(KnownFunction::Copy);
    torch_nn_functional_softsign_classified_shape_preserving, ["torch", "nn", "functional", "softsign"], Some(KnownFunction::Copy);
    torch_nn_functional_selu_classified_shape_preserving, ["torch", "nn", "functional", "selu"], Some(KnownFunction::Copy);
    torch_nn_functional_celu_classified_shape_preserving, ["torch", "nn", "functional", "celu"], Some(KnownFunction::Copy);
    torch_nn_functional_elu_classified_shape_preserving, ["torch", "nn", "functional", "elu"], Some(KnownFunction::Copy);
    torch_nn_functional_leaky_relu_classified_shape_preserving, ["torch", "nn", "functional", "leaky_relu"], Some(KnownFunction::Copy);
    torch_nn_functional_hardtanh_classified_shape_preserving, ["torch", "nn", "functional", "hardtanh"], Some(KnownFunction::Copy);
    torch_nn_functional_hardswish_classified_shape_preserving, ["torch", "nn", "functional", "hardswish"], Some(KnownFunction::Copy);
    torch_nn_functional_hardsigmoid_classified_shape_preserving, ["torch", "nn", "functional", "hardsigmoid"], Some(KnownFunction::Copy);
    torch_nn_functional_mish_classified_shape_preserving, ["torch", "nn", "functional", "mish"], Some(KnownFunction::Copy);
    torch_nn_functional_relu6_classified_shape_preserving, ["torch", "nn", "functional", "relu6"], Some(KnownFunction::Copy);
    torch_nn_functional_dropout_classified_shape_preserving, ["torch", "nn", "functional", "dropout"], Some(KnownFunction::Copy);
    torch_nn_functional_dropout2d_classified_shape_preserving, ["torch", "nn", "functional", "dropout2d"], Some(KnownFunction::Copy);
    torch_nn_functional_dropout3d_classified_shape_preserving, ["torch", "nn", "functional", "dropout3d"], Some(KnownFunction::Copy);
    // `glu` halves a dim rather than preserving shape — not `Copy`.
    torch_nn_functional_glu_classified, ["torch", "nn", "functional", "glu"], Some(KnownFunction::FunctionalGlu);
    torch_nn_functional_unknown_not_classified, ["torch", "nn", "functional", "unknown_func"], None;

    jax_vmap, ["jax", "vmap"], Some(KnownFunction::Vmap);
    equinox_filter_vmap, ["equinox", "filter_vmap"], Some(KnownFunction::Vmap);
    // equinox.nn.filter_vmap is NOT a known function — wrong module path
    equinox_nn_filter_vmap_rejected, ["equinox", "nn", "filter_vmap"], None;
    jax_lax_dot, ["jax", "lax", "dot"], Some(KnownFunction::Dot);
    jax_lax_dot_general, ["jax", "lax", "dot_general"], Some(KnownFunction::Matmul);

    unknown_module_rejected, ["foo", "concatenate"], None;
    deep_unknown_module_rejected, ["jax", "random", "split"], None;
    too_short_target_rejected, ["concatenate"], None;
    empty_target_rejected, [], None;
    case_sensitive_function_rejected, ["jax", "numpy", "Concatenate"], None;
    case_sensitive_module_rejected, ["JAX", "numpy", "concatenate"], None;
    method_like_known_function_rejected, ["x", "reshape"], None;
    // ── Classification tests for all/any/argmax/argmin/argsort/sort/cumsum/cumprod ──

    jnp_all, ["jax", "numpy", "all"], Some(KnownFunction::All);
    np_all, ["numpy", "all"], Some(KnownFunction::All);
    torch_all, ["torch", "all"], Some(KnownFunction::All);

    jnp_any, ["jax", "numpy", "any"], Some(KnownFunction::Any);
    np_any, ["numpy", "any"], Some(KnownFunction::Any);
    torch_any, ["torch", "any"], Some(KnownFunction::Any);

    jnp_argmax, ["jax", "numpy", "argmax"], Some(KnownFunction::ArgMax);
    np_argmax, ["numpy", "argmax"], Some(KnownFunction::ArgMax);
    torch_argmax, ["torch", "argmax"], Some(KnownFunction::ArgMax);

    jnp_argmin, ["jax", "numpy", "argmin"], Some(KnownFunction::ArgMin);
    np_argmin, ["numpy", "argmin"], Some(KnownFunction::ArgMin);
    torch_argmin, ["torch", "argmin"], Some(KnownFunction::ArgMin);

    jnp_argsort, ["jax", "numpy", "argsort"], Some(KnownFunction::Argsort);
    np_argsort, ["numpy", "argsort"], Some(KnownFunction::Argsort);
    torch_argsort, ["torch", "argsort"], Some(KnownFunction::Argsort);

    jnp_sort, ["jax", "numpy", "sort"], Some(KnownFunction::Sort);
    np_sort, ["numpy", "sort"], Some(KnownFunction::Sort);
    torch_sort, ["torch", "sort"], Some(KnownFunction::Sort);

    jnp_cumsum, ["jax", "numpy", "cumsum"], Some(KnownFunction::Cumsum);
    np_cumsum, ["numpy", "cumsum"], Some(KnownFunction::Cumsum);
    torch_cumsum, ["torch", "cumsum"], Some(KnownFunction::Cumsum);

    jnp_cumprod, ["jax", "numpy", "cumprod"], Some(KnownFunction::Cumprod);
    np_cumprod, ["numpy", "cumprod"], Some(KnownFunction::Cumprod);
    torch_cumprod, ["torch", "cumprod"], Some(KnownFunction::Cumprod);

    jax_numpy_vmap_rejected, ["jax", "numpy", "vmap"], None;
    numpy_vmap_rejected, ["numpy", "vmap"], None;
    torch_vmap_rejected_for_now, ["torch", "vmap"], None;
    torch_nn_function_rejected, ["torch", "nn", "Linear"], None;
    numpy_linalg_dot_rejected_for_now, ["numpy", "linalg", "dot"], None;
    // `linalg.norm` reuses the plain-reduction shape rule (axis/keepdims
    // mechanics are identical); see `linalg_norm_reduction_tests` below.
    jax_numpy_linalg_norm_classified, ["jax", "numpy", "linalg", "norm"], Some(KnownFunction::Sum);

    // ── linalg.inv classification tests ──

    jnp_linalg_inv, ["jax", "numpy", "linalg", "inv"], Some(KnownFunction::LinalgInv);
    np_linalg_inv, ["numpy", "linalg", "inv"], Some(KnownFunction::LinalgInv);
    torch_linalg_inv, ["torch", "linalg", "inv"], Some(KnownFunction::LinalgInv);
    jax_linalg_inv_unsupported, ["jax", "linalg", "inv"], None;

    // ── linalg.det classification tests ──

    jnp_linalg_det, ["jax", "numpy", "linalg", "det"], Some(KnownFunction::LinalgDet);
    np_linalg_det, ["numpy", "linalg", "det"], Some(KnownFunction::LinalgDet);
    torch_linalg_det, ["torch", "linalg", "det"], Some(KnownFunction::LinalgDet);
    jax_linalg_det_unsupported, ["jax", "linalg", "det"], None;
}

macro_rules! alias_cases {
    ($($name:ident, $code:expr, $target:expr, $kind:expr;)*) => {
        $(
            #[test]
            fn $name() {
                let tree = parse($code);
                let import_map = build_import_map(tree.root_node(), $code).unwrap();
                let resolved = resolve_call_target($target, &import_map);
                assert_eq!(classify_known_function(&resolved), $kind);
            }
        )*
    };
}

alias_cases! {
    alias_jnp_concatenate, "import jax.numpy as jnp", "jnp.concatenate", Some(KnownFunction::Concatenate);
    alias_np_stack, "import numpy as np", "np.stack", Some(KnownFunction::Stack);
    alias_torch_cat, "import torch as th", "th.cat", Some(KnownFunction::Concatenate);
    alias_jax_vmap, "import jax", "jax.vmap", Some(KnownFunction::Vmap);
    alias_jax_as_jx_vmap, "import jax as jx", "jx.vmap", Some(KnownFunction::Vmap);
    alias_jax_lax_dot, "import jax.lax as lax", "lax.dot", Some(KnownFunction::Dot);
    from_import_jax_vmap, "from jax import vmap", "vmap", Some(KnownFunction::Vmap);
    from_import_jax_vmap_alias, "from jax import vmap as vm", "vm", Some(KnownFunction::Vmap);
    // equinox.filter_vmap alias tests
    alias_equinox_filter_vmap, "import equinox", "equinox.filter_vmap", Some(KnownFunction::Vmap);
    alias_equinox_as_eqx_filter_vmap, "import equinox as eqx", "eqx.filter_vmap", Some(KnownFunction::Vmap);
    from_import_equinox_filter_vmap, "from equinox import filter_vmap", "filter_vmap", Some(KnownFunction::Vmap);
    from_import_jnp_reshape, "from jax.numpy import reshape", "reshape", Some(KnownFunction::Reshape);
    from_import_np_transpose_alias, "from numpy import transpose as T", "T", Some(KnownFunction::Transpose);
    from_import_torch_stack_alias, "from torch import stack as stk", "stk", Some(KnownFunction::Stack);
    alias_jnp_broadcast_to, "import jax.numpy as jnp", "jnp.broadcast_to", Some(KnownFunction::BroadcastTo);
    alias_np_vstack, "import numpy as np", "np.vstack", Some(KnownFunction::Vstack);
    alias_torch_permute, "import torch as th", "th.permute", Some(KnownFunction::Permute);
    from_import_np_meshgrid, "from numpy import meshgrid", "meshgrid", Some(KnownFunction::Meshgrid);
    from_import_jnp_asarray_alias, "from jax.numpy import asarray as arr", "arr", Some(KnownFunction::AsArray);
    from_import_torch_as_tensor, "from torch import as_tensor", "as_tensor", Some(KnownFunction::AsArray);
    from_import_unknown_rejected, "from foo import concatenate", "concatenate", None;
    unimported_known_name_rejected, "", "concatenate", None;
    alias_exact_matching_rejected, "import numpy as np", "npx.concatenate", None;

    // torch.nn.functional alias/import tests
    alias_torch_nn_functional_imported_as_f_pad, "import torch.nn.functional as F", "F.pad", Some(KnownFunction::Pad);
    from_import_torch_nn_functional_pad, "from torch.nn.functional import pad", "pad", Some(KnownFunction::Pad);
    from_import_torch_nn_functional_pad_alias, "from torch.nn.functional import pad as F_pad", "F_pad", Some(KnownFunction::Pad);
    alias_torch_nn_functional_imported_as_f_relu_classified_shape_preserving, "import torch.nn.functional as F", "F.relu", Some(KnownFunction::Copy);
}

// ── jax.lax higher-order / structured op classification ──

known_cases! {
    jax_lax_map, ["jax", "lax", "map"], Some(KnownFunction::LaxMap);
    jax_lax_cond, ["jax", "lax", "cond"], Some(KnownFunction::LaxCond);
    jax_lax_switch, ["jax", "lax", "switch"], Some(KnownFunction::LaxSwitch);
    jax_lax_while_loop, ["jax", "lax", "while_loop"], Some(KnownFunction::LaxWhileLoop);
    jax_lax_fori_loop, ["jax", "lax", "fori_loop"], Some(KnownFunction::LaxForiLoop);
    jax_lax_conv_general_dilated, ["jax", "lax", "conv_general_dilated"], Some(KnownFunction::LaxConvGeneralDilated);
    jax_lax_gather, ["jax", "lax", "gather"], Some(KnownFunction::LaxGather);
    jax_lax_scatter, ["jax", "lax", "scatter"], Some(KnownFunction::LaxScatter);
    jax_lax_scatter_add, ["jax", "lax", "scatter_add"], Some(KnownFunction::LaxScatter);
    jax_lax_reduce_window, ["jax", "lax", "reduce_window"], Some(KnownFunction::LaxReduceWindow);
    jax_lax_top_k, ["jax", "lax", "top_k"], Some(KnownFunction::LaxTopK);
    jax_lax_sort, ["jax", "lax", "sort"], Some(KnownFunction::LaxSort);
    jax_lax_sort_key_val, ["jax", "lax", "sort_key_val"], Some(KnownFunction::LaxSortKeyVal);
    jax_lax_pad, ["jax", "lax", "pad"], Some(KnownFunction::LaxPad);
    jax_lax_broadcast, ["jax", "lax", "broadcast"], Some(KnownFunction::LaxBroadcast);
    jax_lax_broadcast_in_dim, ["jax", "lax", "broadcast_in_dim"], Some(KnownFunction::LaxBroadcastInDim);
    jax_lax_slice, ["jax", "lax", "slice"], Some(KnownFunction::LaxSlice);
    jax_lax_dynamic_slice, ["jax", "lax", "dynamic_slice"], Some(KnownFunction::LaxDynamicSlice);
    jax_lax_dynamic_update_slice, ["jax", "lax", "dynamic_update_slice"], Some(KnownFunction::LaxDynamicUpdateSlice);
    jax_lax_concatenate, ["jax", "lax", "concatenate"], Some(KnownFunction::Concatenate);
    jax_lax_rev, ["jax", "lax", "rev"], Some(KnownFunction::Flip);
    jax_lax_squeeze, ["jax", "lax", "squeeze"], Some(KnownFunction::Squeeze);
    jax_lax_expand_dims, ["jax", "lax", "expand_dims"], Some(KnownFunction::ExpandDims);
    jax_lax_transpose, ["jax", "lax", "transpose"], Some(KnownFunction::Transpose);
    jax_lax_associative_scan, ["jax", "lax", "associative_scan"], Some(KnownFunction::LaxAssociativeScan);
    jax_lax_psum_not_classified, ["jax", "lax", "psum"], None;
    jax_jit_not_classified, ["jax", "jit"], None;
    jax_grad_not_classified, ["jax", "grad"], None;

    // ── jax.numpy / numpy array-creation classification ──

    jnp_diagflat, ["jax", "numpy", "diagflat"], Some(KnownFunction::Diagflat);
    jnp_tri, ["jax", "numpy", "tri"], Some(KnownFunction::Tri);
    jnp_indices, ["jax", "numpy", "indices"], Some(KnownFunction::Indices);
    jnp_bincount, ["jax", "numpy", "bincount"], Some(KnownFunction::BinCount);
    jnp_unique, ["jax", "numpy", "unique"], Some(KnownFunction::Unique);
    jnp_select, ["jax", "numpy", "select"], Some(KnownFunction::Select);

    // ── jax.numpy / numpy shape-transform classification ──

    jnp_rollaxis, ["jax", "numpy", "rollaxis"], Some(KnownFunction::RollAxis);
    jnp_resize, ["jax", "numpy", "resize"], Some(KnownFunction::Resize);
    jnp_insert, ["jax", "numpy", "insert"], Some(KnownFunction::Insert);
    jnp_delete, ["jax", "numpy", "delete"], Some(KnownFunction::Delete);
    jnp_append, ["jax", "numpy", "append"], Some(KnownFunction::Append);
    jnp_broadcast_shapes_not_classified, ["jax", "numpy", "broadcast_shapes"], None;

    // ── jax.numpy / numpy joining-and-splitting classification ──

    jnp_hsplit, ["jax", "numpy", "hsplit"], Some(KnownFunction::HSplit);
    jnp_vsplit, ["jax", "numpy", "vsplit"], Some(KnownFunction::VSplit);
    jnp_dsplit, ["jax", "numpy", "dsplit"], Some(KnownFunction::DSplit);
    jnp_kron, ["jax", "numpy", "kron"], Some(KnownFunction::Kron);

    // ── jax.numpy / numpy indexing-and-selection classification ──

    jnp_take_along_axis, ["jax", "numpy", "take_along_axis"], Some(KnownFunction::TakeAlongAxis);
    jnp_put_along_axis, ["jax", "numpy", "put_along_axis"], Some(KnownFunction::PutAlongAxis);
    jnp_nonzero, ["jax", "numpy", "nonzero"], Some(KnownFunction::Nonzero);
    jnp_argwhere, ["jax", "numpy", "argwhere"], Some(KnownFunction::Argwhere);
    jnp_searchsorted, ["jax", "numpy", "searchsorted"], Some(KnownFunction::SearchSorted);
    jnp_extract, ["jax", "numpy", "extract"], Some(KnownFunction::Extract);
    jnp_compress, ["jax", "numpy", "compress"], Some(KnownFunction::Compress);
    jnp_histogram, ["jax", "numpy", "histogram"], Some(KnownFunction::Histogram);

    // ── jax.numpy / numpy reduction-alias classification ──

    jnp_median_reuses_mean, ["jax", "numpy", "median"], Some(KnownFunction::Mean);
    jnp_quantile_reuses_mean, ["jax", "numpy", "quantile"], Some(KnownFunction::Mean);
    jnp_count_nonzero_reuses_sum, ["jax", "numpy", "count_nonzero"], Some(KnownFunction::Sum);
    jnp_ptp_reuses_max, ["jax", "numpy", "ptp"], Some(KnownFunction::Max);
    jax_numpy_fft_fft_classified, ["jax", "numpy", "fft", "fft"], Some(KnownFunction::Copy);
    torch_fft_rfft_not_classified, ["torch", "fft", "rfft"], None;

    // ── linear algebra classification ──

    jnp_cross, ["jax", "numpy", "cross"], Some(KnownFunction::Cross);
    torch_cross, ["torch", "cross"], Some(KnownFunction::TorchCross);
    jnp_linalg_solve, ["jax", "numpy", "linalg", "solve"], Some(KnownFunction::LinalgSolve);
    jnp_linalg_cholesky_reuses_inv, ["jax", "numpy", "linalg", "cholesky"], Some(KnownFunction::LinalgInv);
    torch_linalg_lstsq, ["torch", "linalg", "lstsq"], Some(KnownFunction::LinalgLstsq);
    torch_linalg_pinv, ["torch", "linalg", "pinv"], Some(KnownFunction::LinalgPinv);
    torch_linalg_matrix_rank, ["torch", "linalg", "matrix_rank"], Some(KnownFunction::LinalgMatrixRank);

    // ── einops classification ──

    einops_einsum, ["einops", "einsum"], Some(KnownFunction::EinopsEinsum);
    einops_pack, ["einops", "pack"], Some(KnownFunction::EinopsPack);
    einops_unpack, ["einops", "unpack"], Some(KnownFunction::EinopsUnpack);
    einops_parse_shape, ["einops", "parse_shape"], Some(KnownFunction::EinopsParseShape);

    // ── jax.nn classification ──

    jax_nn_one_hot, ["jax", "nn", "one_hot"], Some(KnownFunction::OneHot);
    jax_nn_dot_product_attention, ["jax", "nn", "dot_product_attention"], Some(KnownFunction::DotProductAttention);
    jax_nn_logsumexp_reuses_sum, ["jax", "nn", "logsumexp"], Some(KnownFunction::Sum);
    jax_nn_relu_not_classified, ["jax", "nn", "relu"], None;

    torch_gather, ["torch", "gather"], Some(KnownFunction::Gather);
    torch_scatter, ["torch", "scatter"], Some(KnownFunction::Scatter);
    torch_scatter_add, ["torch", "scatter_add"], Some(KnownFunction::Scatter);
    torch_take_along_dim, ["torch", "take_along_dim"], Some(KnownFunction::TakeAlongAxis);
    torch_topk, ["torch", "topk"], Some(KnownFunction::TopK);
    torch_unbind, ["torch", "unbind"], Some(KnownFunction::Unbind);
    torch_chunk, ["torch", "chunk"], Some(KnownFunction::Chunk);
    torch_narrow, ["torch", "narrow"], Some(KnownFunction::Narrow);
    torch_select, ["torch", "select"], Some(KnownFunction::SelectDim);
    torch_masked_select, ["torch", "masked_select"], Some(KnownFunction::MaskedSelect);
    torch_index_select, ["torch", "index_select"], Some(KnownFunction::IndexSelect);
    torch_kthvalue, ["torch", "kthvalue"], Some(KnownFunction::KthValue);
    torch_median, ["torch", "median"], Some(KnownFunction::MedianDim);
    torch_mode, ["torch", "mode"], Some(KnownFunction::MedianDim);
    torch_unique, ["torch", "unique"], Some(KnownFunction::Unique);
    torch_combinations, ["torch", "combinations"], Some(KnownFunction::Combinations);
    torch_cartesian_prod, ["torch", "cartesian_prod"], Some(KnownFunction::CartesianProd);
    torch_block_diag, ["torch", "block_diag"], Some(KnownFunction::BlockDiag);
    torch_nn_functional_interpolate, ["torch", "nn", "functional", "interpolate"], Some(KnownFunction::Interpolate);
    torch_nn_functional_conv1d, ["torch", "nn", "functional", "conv1d"], Some(KnownFunction::FunctionalConv1d);
    torch_nn_functional_conv2d, ["torch", "nn", "functional", "conv2d"], Some(KnownFunction::FunctionalConv2d);
    torch_nn_functional_conv3d, ["torch", "nn", "functional", "conv3d"], Some(KnownFunction::FunctionalConv3d);
    torch_nn_functional_max_pool1d, ["torch", "nn", "functional", "max_pool1d"], Some(KnownFunction::FunctionalMaxPool1d);
    torch_nn_functional_max_pool2d, ["torch", "nn", "functional", "max_pool2d"], Some(KnownFunction::FunctionalMaxPool2d);
    torch_nn_functional_max_pool3d, ["torch", "nn", "functional", "max_pool3d"], Some(KnownFunction::FunctionalMaxPool3d);
    torch_nn_functional_avg_pool1d, ["torch", "nn", "functional", "avg_pool1d"], Some(KnownFunction::FunctionalAvgPool1d);
    torch_nn_functional_avg_pool2d, ["torch", "nn", "functional", "avg_pool2d"], Some(KnownFunction::FunctionalAvgPool2d);
    torch_nn_functional_avg_pool3d, ["torch", "nn", "functional", "avg_pool3d"], Some(KnownFunction::FunctionalAvgPool3d);
    torch_nn_functional_softmax, ["torch", "nn", "functional", "softmax"], Some(KnownFunction::Copy);
    torch_nn_functional_log_softmax, ["torch", "nn", "functional", "log_softmax"], Some(KnownFunction::Copy);
    torch_nn_functional_normalize, ["torch", "nn", "functional", "normalize"], Some(KnownFunction::Copy);
    torch_nn_functional_one_hot, ["torch", "nn", "functional", "one_hot"], Some(KnownFunction::OneHot);
    torch_nn_functional_embedding, ["torch", "nn", "functional", "embedding"], Some(KnownFunction::FunctionalEmbedding);
    torch_nn_utils_rnn_pad_sequence, ["torch", "nn", "utils", "rnn", "pad_sequence"], Some(KnownFunction::PadSequence);

    torch_bmm, ["torch", "bmm"], Some(KnownFunction::Bmm);
    numpy_bmm_is_not_known, ["numpy", "bmm"], None;
    torch_nn_functional_cross_entropy, ["torch", "nn", "functional", "cross_entropy"], Some(KnownFunction::CrossEntropy);
    torch_nn_functional_nll_loss, ["torch", "nn", "functional", "nll_loss"], Some(KnownFunction::NllLoss);
    torch_cross_entropy_is_not_known, ["torch", "cross_entropy"], None;
}
