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

type Case = (
    &'static str,
    Vec<CallArgument>,
    &'static [(&'static str, &'static [&'static str])],
    Result<Option<&'static [&'static str]>, &'static [&'static str]>,
);

/// Each case is `(name, args, input shapes, expected)`; an `Err` lists substrings the
/// error must contain. A method call runs when `receiver` is set. Reports every failing case.
pub(super) fn check(function: KnownFunction, receiver: Option<&str>, cases: &[Case]) {
    let mut failures = Vec::new();
    for (name, args, inputs, expected) in cases {
        let shapes: HashMap<String, Vec<String>> = inputs
            .iter()
            .map(|(input, dims)| (input.to_string(), shape(dims)))
            .collect();
        let output = match receiver {
            Some(receiver) => apply_method_call(&function, receiver, args, &shapes),
            None => apply_known_function(&function, args, &shapes),
        };
        let passed = match (&output, expected) {
            (Ok(output), Ok(expected)) => *output == expected.map(shape),
            (Err(error), Err(parts)) => parts.iter().all(|part| error.contains(part)),
            _ => false,
        };
        if !passed {
            failures.push(format!("{name}: expected {expected:?}, got {output:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{function:?} cases failed:\n{}",
        failures.join("\n")
    );
}

#[test]
#[rustfmt::skip]
fn test_concatenate_cases() {
    check(KnownFunction::Concatenate, None, &[
        ("concatenate_axis_0_numeric_dims", vec![pos("[a, b]")], &[("a", &["2", "features"]), ("b", &["3", "features"])], Ok(Some(&["5", "features"]))),
        ("concatenate_axis_1_numeric_dims", vec![pos("[a, b]"), kw("axis", "1")], &[("a", &["batch", "2"]), ("b", &["batch", "3"])], Ok(Some(&["batch", "5"]))),
        ("concatenate_negative_axis", vec![pos("[a, b]"), kw("axis", "-1")], &[("a", &["batch", "2"]), ("b", &["batch", "3"])], Ok(Some(&["batch", "5"]))),
        ("concatenate_torch_dim_keyword", vec![pos("[a, b]"), kw("dim", "1")], &[("a", &["batch", "m"]), ("b", &["batch", "n"])], Ok(Some(&["batch", "m+n"]))),
        ("concatenate_arrays_keyword", vec![kw("arrays", "[a, b]"), kw("axis", "0")], &[("a", &["m", "features"]), ("b", &["n", "features"])], Ok(Some(&["m+n", "features"]))),
        ("concatenate_tensors_keyword", vec![kw("tensors", "[a, b]"), kw("dim", "0")], &[("a", &["m", "features"]), ("b", &["n", "features"])], Ok(Some(&["m+n", "features"]))),
        ("concatenate_tuple_inputs", vec![pos("(a, b)")], &[("a", &["m", "features"]), ("b", &["n", "features"])], Ok(Some(&["m+n", "features"]))),
        ("concatenate_single_input_returns_same_shape", vec![pos("[a]")], &[("a", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("concatenate_missing_input_returns_none", vec![pos("[a, missing]")], &[("a", &["batch", "features"])], Ok(None)),
        ("concatenate_no_args_returns_none", vec![], &[], Ok(None)),
        ("concatenate_rank_mismatch_errors", vec![pos("[a, b]")], &[("a", &["batch", "features"]), ("b", &["features"])], Err(&["rank"])),
        ("concatenate_non_axis_dim_mismatch_errors", vec![pos("[a, b]"), kw("axis", "1")], &[("a", &["batch", "2"]), ("b", &["other", "3"])], Err(&["dimension mismatch", "expected batch", "got other"])),
        // "d*2" and "2*d" are the same dim under canonicalization — must not
        // be a false "dimension mismatch" (issue: dims were compared as raw
        // strings).
        ("concatenate_commutative_symbolic_dims_do_not_mismatch", vec![pos("[a, b]"), kw("axis", "1")], &[("a", &["d*2", "3"]), ("b", &["2*d", "5"])], Ok(Some(&["d*2", "8"]))),
        ("concatenate_axis_out_of_bounds_errors", vec![pos("[a, b]"), kw("axis", "2")], &[("a", &["batch", "2"]), ("b", &["batch", "3"])], Err(&["out of bounds"])),
        ("concatenate_negative_axis_out_of_bounds_errors", vec![pos("[a, b]"), kw("axis", "-3")], &[("a", &["batch", "2"]), ("b", &["batch", "3"])], Err(&["out of bounds"])),
        ("concatenate_scalar_input_errors", vec![pos("[a, b]")], &[("a", &[]), ("b", &[])], Err(&["rank >= 1"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_stack_cases() {
    check(KnownFunction::Stack, None, &[
        ("stack_default_axis", vec![pos("[a, b]")], &[("a", &["batch", "features"]), ("b", &["batch", "features"])], Ok(Some(&["2", "batch", "features"]))),
        ("stack_axis_1", vec![pos("[a, b, c]"), kw("axis", "1")], &[("a", &["batch", "features"]), ("b", &["batch", "features"]), ("c", &["batch", "features"])], Ok(Some(&["batch", "3", "features"]))),
        ("stack_negative_axis", vec![pos("[a, b]"), kw("axis", "-1")], &[("a", &["batch", "features"]), ("b", &["batch", "features"])], Ok(Some(&["batch", "features", "2"]))),
        ("stack_torch_dim_keyword", vec![pos("[a, b]"), kw("dim", "1")], &[("a", &["batch", "features"]), ("b", &["batch", "features"])], Ok(Some(&["batch", "2", "features"]))),
        ("stack_arys_keyword", vec![kw("arys", "[a, b]"), kw("axis", "0")], &[("a", &["batch"]), ("b", &["batch"])], Ok(Some(&["2", "batch"]))),
        ("stack_tuple_inputs", vec![pos("(a, b)")], &[("a", &["batch"]), ("b", &["batch"])], Ok(Some(&["2", "batch"]))),
        ("stack_missing_input_returns_none", vec![pos("[a, missing]")], &[("a", &["batch"])], Ok(None)),
        ("stack_no_args_returns_none", vec![], &[], Ok(None)),
        ("stack_rank_mismatch_errors", vec![pos("[a, b]")], &[("a", &["batch", "features"]), ("b", &["batch"])], Err(&["rank"])),
        ("stack_dim_mismatch_errors", vec![pos("[a, b]")], &[("a", &["batch", "features"]), ("b", &["batch", "other"])], Err(&["dimension mismatch", "expected features", "got other"])),
        ("stack_commutative_symbolic_dims_do_not_mismatch", vec![pos("[a, b]")], &[("a", &["batch", "d_state*2"]), ("b", &["batch", "2*d_state"])], Ok(Some(&["2", "batch", "d_state*2"]))),
        ("stack_axis_out_of_bounds_errors", vec![pos("[a, b]"), kw("axis", "3")], &[("a", &["batch", "features"]), ("b", &["batch", "features"])], Err(&["out of bounds"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_reshape_cases() {
    check(KnownFunction::Reshape, None, &[
        ("reshape_positional_shape", vec![pos("x"), pos("(2, 6)")], &[("x", &["3", "4"])], Ok(Some(&["2", "6"]))),
        ("reshape_keyword_shape", vec![pos("x"), kw("shape", "[2, 6]")], &[("x", &["3", "4"])], Ok(Some(&["2", "6"]))),
        ("reshape_infers_minus_one", vec![pos("x"), pos("(2, -1)")], &[("x", &["3", "4"])], Ok(Some(&["2", "6"]))),
        ("reshape_symbolic_shape_allowed_without_size_check", vec![pos("x"), pos("(batch, features)")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("reshape_size_mismatch_errors", vec![pos("x"), pos("(5, 5)")], &[("x", &["3", "4"])], Err(&["changes total size"])),
        ("reshape_multiple_minus_one_errors", vec![pos("x"), pos("(-1, -1)")], &[("x", &["3", "4"])], Err(&["only infer one -1"])),
        ("reshape_missing_input_returns_none", vec![pos("x"), pos("(2, 6)")], &[], Ok(None)),
        ("reshape_missing_shape_arg_returns_none", vec![pos("x")], &[("x", &["2", "6"])], Ok(None)),
        ("reshape_symbolic_minus_one_cancels_known_dim", vec![pos("x"), pos("(batch, -1)")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("reshape_symbolic_minus_one_flattens_remaining_dims", vec![pos("x"), pos("(c, -1)")], &[("x", &["c", "h", "w"])], Ok(Some(&["c", "h*w"]))),
        ("reshape_shape_index_resolves_to_input_dim", vec![pos("x"), pos("(x.shape[0], -1)")], &[("x", &["c", "h", "w"])], Ok(Some(&["c", "h*w"]))),
        ("reshape_shape_index_negative_resolves", vec![pos("x"), pos("(x.shape[-1], -1)")], &[("x", &["c", "h", "w"])], Ok(Some(&["w", "c*h"]))),
        ("reshape_symbolic_minus_one_unmatched_known_dim_returns_none", vec![pos("x"), pos("(z, -1)")], &[("x", &["a", "b"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_flatten_cases() {
    check(KnownFunction::Flatten, None, &[
        ("flatten_numeric_shape", vec![pos("x")], &[("x", &["2", "3", "4"])], Ok(Some(&["24"]))),
        ("flatten_symbolic_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch*features"]))),
        ("flatten_scalar_shape_returns_one_dim_one", vec![pos("x")], &[("x", &[])], Ok(Some(&["1"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_ravel_cases() {
    check(KnownFunction::Ravel, None, &[
        ("ravel_uses_flatten_rule", vec![pos("x")], &[("x", &["2", "3"])], Ok(Some(&["6"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_transpose_cases() {
    check(KnownFunction::Transpose, None, &[
        ("transpose_default_reverses_axes", vec![pos("x")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "b", "a"]))),
        ("transpose_axes_keyword", vec![pos("x"), kw("axes", "(1, 0, 2)")], &[("x", &["a", "b", "c"])], Ok(Some(&["b", "a", "c"]))),
        ("transpose_negative_axes", vec![pos("x"), kw("axes", "(-1, -2, -3)")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "b", "a"]))),
        ("transpose_wrong_number_of_axes_errors", vec![pos("x"), kw("axes", "(0, 1)")], &[("x", &["a", "b", "c"])], Err(&["expected 3 axes"])),
        ("transpose_duplicate_axes_errors", vec![pos("x"), kw("axes", "(0, 0, 1)")], &[("x", &["a", "b", "c"])], Err(&["axis 0 given more than once"])),
        ("transpose_missing_input_returns_none", vec![pos("x")], &[], Ok(None)),
        ("transpose_axis_out_of_bounds_errors", vec![pos("x"), kw("axes", "(0, 1, 3)")], &[("x", &["a", "b", "c"])], Err(&["out of bounds"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_permute_cases() {
    check(KnownFunction::Permute, None, &[
        ("permute_dims_keyword", vec![pos("x"), kw("dims", "(2, 0, 1)")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "a", "b"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_swap_axes_cases() {
    check(KnownFunction::SwapAxes, None, &[
        ("swapaxes_positional", vec![pos("x"), pos("0"), pos("2")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "b", "a"]))),
        ("swapaxes_torch_dim_keywords", vec![pos("x"), kw("dim0", "0"), kw("dim1", "1")], &[("x", &["a", "b", "c"])], Ok(Some(&["b", "a", "c"]))),
        ("swapaxes_missing_axis_returns_none", vec![pos("x"), pos("0")], &[("x", &["a", "b", "c"])], Ok(None)),
        ("swapaxes_negative_axes", vec![pos("x"), pos("-1"), pos("0")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "b", "a"]))),
        ("swapaxes_axis_out_of_bounds_errors", vec![pos("x"), pos("0"), pos("3")], &[("x", &["a", "b", "c"])], Err(&["out of bounds"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_move_axis_cases() {
    check(KnownFunction::MoveAxis, None, &[
        ("moveaxis_single_axis", vec![pos("x"), pos("0"), pos("2")], &[("x", &["a", "b", "c"])], Ok(Some(&["b", "c", "a"]))),
        ("moveaxis_keyword_axes", vec![pos("x"), kw("source", "0"), kw("destination", "1")], &[("x", &["a", "b", "c"])], Ok(Some(&["b", "a", "c"]))),
        ("moveaxis_length_mismatch_errors", vec![pos("x"), kw("source", "(0, 1)"), kw("destination", "2")], &[("x", &["a", "b", "c"])], Err(&["lengths differ"])),
        ("moveaxis_missing_destination_returns_none", vec![pos("x"), pos("0")], &[("x", &["a", "b", "c"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_expand_dims_cases() {
    check(KnownFunction::ExpandDims, None, &[
        ("expand_dims_axis_0", vec![pos("x"), kw("axis", "0")], &[("x", &["batch", "features"])], Ok(Some(&["1", "batch", "features"]))),
        ("expand_dims_negative_axis", vec![pos("x"), kw("axis", "-1")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features", "1"]))),
        ("expand_dims_missing_axis_returns_none", vec![pos("x")], &[("x", &["batch", "features"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_squeeze_cases() {
    check(KnownFunction::Squeeze, None, &[
        ("squeeze_all_unit_dims", vec![pos("x")], &[("x", &["1", "batch", "1", "features"])], Ok(Some(&["batch", "features"]))),
        ("squeeze_specific_axis", vec![pos("x"), kw("axis", "0")], &[("x", &["1", "batch", "1"])], Ok(Some(&["batch", "1"]))),
        ("squeeze_non_unit_axis_errors", vec![pos("x"), kw("axis", "1")], &[("x", &["1", "batch", "1"])], Err(&["cannot squeeze"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_at_least1_d_cases() {
    check(KnownFunction::AtLeast1D, None, &[
        ("atleast_1d_scalar", vec![pos("x")], &[("x", &[])], Ok(Some(&["1"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_at_least2_d_cases() {
    check(KnownFunction::AtLeast2D, None, &[
        ("atleast_2d_vector", vec![pos("x")], &[("x", &["features"])], Ok(Some(&["1", "features"]))),
        ("atleast_preserves_sufficient_rank", vec![pos("x")], &[("x", &["a", "b", "c"])], Ok(Some(&["a", "b", "c"]))),
    ]);
}

#[test]
fn test_atleast_3d_matrix() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

    let output = apply_known_function(&KnownFunction::AtLeast3D, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features", "1"])));
    let vector_shapes = HashMap::from([("x".to_string(), shape(&["features"]))]);
    assert_eq!(
        apply_known_function(&KnownFunction::AtLeast3D, &args, &vector_shapes).unwrap(),
        Some(shape(&["1", "features", "1"]))
    );
    let scalar_shapes = HashMap::from([("x".to_string(), vec![])]);
    assert_eq!(
        apply_known_function(&KnownFunction::AtLeast3D, &args, &scalar_shapes).unwrap(),
        Some(shape(&["1", "1", "1"]))
    );
}

#[test]
#[rustfmt::skip]
fn test_matmul_cases() {
    check(KnownFunction::Matmul, None, &[
        ("matmul_vector_vector_scalar", vec![pos("a"), pos("b")], &[("a", &["k"]), ("b", &["k"])], Ok(Some(&[]))),
        ("matmul_matrix_matrix", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["k", "n"])], Ok(Some(&["m", "n"]))),
        ("matmul_batched_matrix_matrix", vec![pos("a"), pos("b")], &[("a", &["batch", "m", "k"]), ("b", &["batch", "k", "n"])], Ok(Some(&["batch", "m", "n"]))),
        ("matmul_broadcasts_batch_dims", vec![pos("a"), pos("b")], &[("a", &["batch", "m", "k"]), ("b", &["1", "k", "n"])], Ok(Some(&["batch", "m", "n"]))),
        ("matmul_inner_dim_mismatch_errors", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["other", "n"])], Err(&["dimension mismatch"])),
        // "k*2" and "2*k" are the same contracted dim under canonicalization.
        ("matmul_commutative_symbolic_inner_dim_does_not_mismatch", vec![pos("a"), pos("b")], &[("a", &["m", "k*2"]), ("b", &["2*k", "n"])], Ok(Some(&["m", "n"]))),
        ("matmul_commutative_symbolic_batch_dims_broadcast", vec![pos("a"), pos("b")], &[("a", &["b_state*2", "m", "k"]), ("b", &["2*b_state", "k", "n"])], Ok(Some(&["b_state*2", "m", "n"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_dot_cases() {
    check(KnownFunction::Dot, None, &[
        ("dot_matrix_vector", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["k"])], Ok(Some(&["m"]))),
        ("dot_matrix_matrix", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["k", "n"])], Ok(Some(&["m", "n"]))),
        ("dot_high_rank", vec![pos("a"), pos("b")], &[("a", &["b1", "m", "k"]), ("b", &["b2", "k", "n"])], Ok(Some(&["b1", "m", "b2", "n"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_tensor_dot_cases() {
    check(KnownFunction::TensorDot, None, &[
        ("tensordot_default_axes_two", vec![pos("a"), pos("b")], &[("a", &["m", "n", "k1", "k2"]), ("b", &["k1", "k2", "p", "q"])], Ok(Some(&["m", "n", "p", "q"]))),
        ("tensordot_int_axes_one", vec![pos("a"), pos("b"), pos("1")], &[("a", &["m", "k"]), ("b", &["k", "n"])], Ok(Some(&["m", "n"]))),
        ("tensordot_axes_keyword", vec![pos("a"), pos("b"), kw("axes", "1")], &[("a", &["batch", "features"]), ("b", &["features", "out"])], Ok(Some(&["batch", "out"]))),
        ("tensordot_contraction_mismatch_errors", vec![pos("a"), pos("b"), pos("1")], &[("a", &["m", "k"]), ("b", &["other", "n"])], Err(&["dimension mismatch"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_outer_cases() {
    check(KnownFunction::Outer, None, &[
        ("outer_1d_1d", vec![pos("a"), pos("b")], &[("a", &["m"]), ("b", &["n"])], Ok(Some(&["m", "n"]))),
        ("outer_higher_rank_flattens", vec![pos("a"), pos("b")], &[("a", &["2", "3"]), ("b", &["4", "5"])], Ok(Some(&["6", "20"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_inner_cases() {
    check(KnownFunction::Inner, None, &[
        ("inner_2d_2d_match", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["n", "k"])], Ok(Some(&["m", "n"]))),
        ("inner_last_dim_mismatch_errors", vec![pos("a"), pos("b")], &[("a", &["m", "k"]), ("b", &["n", "other"])], Err(&["dimension mismatch"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_vdot_cases() {
    check(KnownFunction::Vdot, None, &[
        ("vdot_returns_scalar", vec![pos("a"), pos("b")], &[("a", &["2", "3"]), ("b", &["6"])], Ok(Some(&[]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_zeros_cases() {
    check(KnownFunction::Zeros, None, &[
        ("zeros_shape_constructor", vec![pos("(batch, features)")], &[], Ok(Some(&["batch", "features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_ones_cases() {
    check(KnownFunction::Ones, None, &[
        ("ones_shape_keyword", vec![kw("shape", "(2, 3)")], &[], Ok(Some(&["2", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_arange_cases() {
    check(KnownFunction::Arange, None, &[
        ("arange_single_stop", vec![pos("10")], &[], Ok(Some(&["10"]))),
        ("arange_start_stop", vec![pos("2"), pos("10")], &[], Ok(Some(&["8"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_eye_cases() {
    check(KnownFunction::Eye, None, &[
        ("eye_square", vec![pos("n")], &[], Ok(Some(&["n", "n"]))),
        ("eye_rectangular", vec![pos("n"), pos("m")], &[], Ok(Some(&["n", "m"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_diag_cases() {
    check(KnownFunction::Diag, None, &[
        ("diag_vector_to_matrix", vec![pos("x")], &[("x", &["n"])], Ok(Some(&["n", "n"]))),
        ("diag_matrix_to_vector", vec![pos("x")], &[("x", &["m", "n"])], Ok(Some(&["__free_min(m,n)"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_diagonal_cases() {
    check(KnownFunction::Diagonal, None, &[
        ("diagonal_batch_matrix", vec![pos("x")], &[("x", &["batch", "m", "n"])], Ok(Some(&["batch", "__free_min(m,n)"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_trace_cases() {
    check(KnownFunction::Trace, None, &[
        ("trace_batch_matrix", vec![pos("x")], &[("x", &["batch", "m", "n"])], Ok(Some(&["batch"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_take_cases() {
    check(KnownFunction::Take, None, &[
        ("take_without_axis_uses_indices_shape", vec![pos("x"), pos("idx")], &[("x", &["batch", "features"]), ("idx", &["k"])], Ok(Some(&["k"]))),
        ("take_with_axis_inserts_indices_shape", vec![pos("x"), pos("idx"), kw("axis", "1")], &[("x", &["batch", "features"]), ("idx", &["k"])], Ok(Some(&["batch", "k"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_pad_cases() {
    check(KnownFunction::Pad, None, &[
        ("pad_single_width_all_axes", vec![pos("x"), pos("1")], &[("x", &["10", "20"])], Ok(Some(&["12", "22"]))),
        ("pad_before_after_all_axes", vec![pos("x"), pos("(1, 2)")], &[("x", &["h", "w"])], Ok(Some(&["h+3", "w+3"]))),
        ("pad_per_axis_widths", vec![pos("x"), pos("((1, 2), (3, 4))")], &[("x", &["10", "20"])], Ok(Some(&["13", "27"]))),
        ("pad_symbolic_per_axis_widths", vec![pos("x"), kw("pad_width", "((1, 2), (3, 4))")], &[("x", &["h", "w"])], Ok(Some(&["h+3", "w+7"]))),
        ("pad_dynamic_width_returns_none", vec![pos("x"), pos("pad_width")], &[("x", &["h", "w"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_rot90_cases() {
    check(KnownFunction::Rot90, None, &[
        ("rot90_default_swaps_first_two_axes", vec![pos("x")], &[("x", &["h", "w", "c"])], Ok(Some(&["w", "h", "c"]))),
        ("rot90_even_k_preserves_shape", vec![pos("x"), kw("k", "2")], &[("x", &["h", "w", "c"])], Ok(Some(&["h", "w", "c"]))),
        ("rot90_custom_axes", vec![pos("x"), kw("axes", "(1, 2)")], &[("x", &["a", "b", "c"])], Ok(Some(&["a", "c", "b"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_vstack_cases() {
    check(KnownFunction::Vstack, None, &[
        ("vstack_vectors", vec![pos("[a, b]")], &[("a", &["features"]), ("b", &["features"])], Ok(Some(&["2", "features"]))),
        ("vstack_matrices", vec![pos("[a, b]")], &[("a", &["m", "features"]), ("b", &["n", "features"])], Ok(Some(&["m+n", "features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_hstack_cases() {
    check(KnownFunction::Hstack, None, &[
        ("hstack_vectors", vec![pos("[a, b]")], &[("a", &["m"]), ("b", &["n"])], Ok(Some(&["m+n"]))),
        ("hstack_matrices", vec![pos("[a, b]")], &[("a", &["batch", "m"]), ("b", &["batch", "n"])], Ok(Some(&["batch", "m+n"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_dstack_cases() {
    check(KnownFunction::Dstack, None, &[
        ("dstack_vectors", vec![pos("[a, b]")], &[("a", &["features"]), ("b", &["features"])], Ok(Some(&["1", "features", "2"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_column_stack_cases() {
    check(KnownFunction::ColumnStack, None, &[
        ("column_stack_vectors", vec![pos("[a, b]")], &[("a", &["rows"]), ("b", &["rows"])], Ok(Some(&["rows", "2"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_broadcast_to_cases() {
    check(KnownFunction::BroadcastTo, None, &[
        ("broadcast_to_shape", vec![pos("x"), kw("shape", "(batch, features)")], &[("x", &["features"])], Ok(Some(&["batch", "features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_broadcast_arrays_cases() {
    check(KnownFunction::BroadcastArrays, None, &[
        ("broadcast_arrays_two_inputs", vec![pos("a"), pos("b")], &[("a", &["batch", "1"]), ("b", &["1", "features"])], Ok(Some(&["batch", "features"]))),
        ("broadcast_arrays_mismatch_errors", vec![pos("a"), pos("b")], &[("a", &["batch", "x"]), ("b", &["batch", "y"])], Err(&["cannot broadcast"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_tile_cases() {
    check(KnownFunction::Tile, None, &[
        ("tile_repeats_dims", vec![pos("x"), pos("(2, 3)")], &[("x", &["batch", "features"])], Ok(Some(&["batch*2", "features*3"]))),
        ("tile_numeric_dims", vec![pos("x"), pos("(2, 3)")], &[("x", &["4", "5"])], Ok(Some(&["8", "15"]))),
        ("non_implemented_known_function_returns_none", vec![pos("x")], &[("x", &["batch", "features"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_repeat_cases() {
    check(KnownFunction::Repeat, None, &[
        ("repeat_axis", vec![pos("x"), kw("repeats", "3"), kw("axis", "1")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features*3"]))),
        ("repeat_without_axis_flattens", vec![pos("x"), pos("2")], &[("x", &["2", "3"])], Ok(Some(&["12"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_roll_cases() {
    check(KnownFunction::Roll, None, &[
        ("shape_preserving_roll", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
    ]);
}

// ── *_like shape-preserving tests ──

#[test]
#[rustfmt::skip]
fn test_zeros_like_cases() {
    check(KnownFunction::ZerosLike, None, &[
        ("zeros_like_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("zeros_like_missing_input_returns_none", vec![pos("x")], &[], Ok(None)),
        ("zeros_like_keyword_x", vec![kw("x", "arr")], &[("arr", &["m", "n"])], Ok(Some(&["m", "n"]))),
        ("zeros_like_no_args_returns_none", vec![], &[], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_ones_like_cases() {
    check(KnownFunction::OnesLike, None, &[
        ("ones_like_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("ones_like_missing_input_returns_none", vec![pos("x")], &[], Ok(None)),
        ("ones_like_keyword_input", vec![kw("input", "arr")], &[("arr", &["m", "n"])], Ok(Some(&["m", "n"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_full_like_cases() {
    check(KnownFunction::FullLike, None, &[
        ("full_like_preserves_shape", vec![pos("x"), pos("0")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("full_like_missing_input_returns_none", vec![pos("x"), pos("0")], &[], Ok(None)),
        ("full_like_keyword_x", vec![kw("x", "arr"), pos("0")], &[("arr", &["m", "n"])], Ok(Some(&["m", "n"]))),
        // 'template' is not a recognized keyword for first_array_arg
        ("full_like_unrecognized_keyword_returns_none", vec![kw("template", "arr")], &[("arr", &["m", "n"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_empty_like_cases() {
    check(KnownFunction::EmptyLike, None, &[
        ("empty_like_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("empty_like_missing_input_returns_none", vec![pos("x")], &[], Ok(None)),
        ("empty_like_keyword_input", vec![kw("input", "arr")], &[("arr", &["m", "n"])], Ok(Some(&["m", "n"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_where_cases() {
    check(KnownFunction::Where, None, &[
        ("where_broadcasts_three_inputs", vec![pos("cond"), pos("x"), pos("y")], &[("cond", &["batch", "1"]), ("x", &["batch", "features"]), ("y", &["1", "features"])], Ok(Some(&["batch", "features"]))),
        ("where_broadcast_mismatch_errors", vec![pos("cond"), pos("x"), pos("y")], &[("cond", &["batch", "1"]), ("x", &["batch", "x"]), ("y", &["batch", "y"])], Err(&["cannot broadcast"])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_sum_cases() {
    check(KnownFunction::Sum, None, &[
        ("reduction_no_axis_returns_scalar_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&[]))),
        ("reduction_multiple_axes_keepdims", vec![pos("x"), kw("axis", "(0, 2)"), kw("keepdims", "True")], &[("x", &["batch", "time", "features"])], Ok(Some(&["1", "time", "1"]))),
        ("reduction_axis_none_reduces_all_axes", vec![pos("x"), kw("axis", "None")], &[("x", &["batch", "features"])], Ok(Some(&[]))),
        ("reduction_missing_input_returns_none", vec![pos("x"), kw("axis", "0")], &[], Ok(None)),
        ("reduction_axis_out_of_bounds_errors", vec![pos("x"), kw("axis", "2")], &[("x", &["batch", "features"])], Err(&["out of bounds"])),
        ("reduction_duplicate_axes_errors", vec![pos("x"), kw("axis", "(1, 1)")], &[("x", &["batch", "features"])], Err(&["duplicate"])),
        ("reduction_invalid_dynamic_axis_returns_none", vec![pos("x"), kw("axis", "axis")], &[("x", &["batch", "features"])], Ok(None)),
        ("reduction_empty_axis_tuple_preserves_shape", vec![pos("x"), kw("axis", "()")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("reduction_axis_none_keepdims_all_ones", vec![pos("x"), kw("axis", "None"), kw("keepdims", "True")], &[("x", &["batch", "features"])], Ok(Some(&["1", "1"]))),
        // linalg.norm dispatches through KnownFunction::Sum's reduction rule.
        ("norm_reduction_axis_semantics", vec![pos("x"), kw("axis", "1")], &[("x", &["b", "d"])], Ok(Some(&["b"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_mean_cases() {
    check(KnownFunction::Mean, None, &[
        ("reduction_no_axis_keepdims", vec![pos("x"), kw("keepdims", "True")], &[("x", &["batch", "features"])], Ok(Some(&["1", "1"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_max_cases() {
    check(KnownFunction::Max, None, &[
        ("reduction_axis_keyword_removes_axis", vec![pos("x"), kw("axis", "1")], &[("x", &["batch", "features", "hidden"])], Ok(Some(&["batch", "hidden"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_min_cases() {
    check(KnownFunction::Min, None, &[
        ("reduction_dim_keyword_removes_axis", vec![pos("x"), kw("dim", "1")], &[("x", &["batch", "features", "hidden"])], Ok(Some(&["batch", "hidden"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_prod_cases() {
    check(KnownFunction::Prod, None, &[
        ("reduction_positional_axis", vec![pos("x"), pos("0")], &[("x", &["batch", "features"])], Ok(Some(&["features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_std_cases() {
    check(KnownFunction::Std, None, &[
        ("reduction_negative_axis", vec![pos("x"), kw("axis", "-1")], &[("x", &["batch", "features"])], Ok(Some(&["batch"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_var_cases() {
    check(KnownFunction::Var, None, &[
        ("reduction_multiple_axes_tuple", vec![pos("x"), kw("axis", "(0, 2)")], &[("x", &["batch", "time", "features"])], Ok(Some(&["time"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_reshape_method_cases() {
    check(KnownFunction::Reshape, Some("x"), &[
        ("reshape_method_form_with_shape_index_and_minus_one", vec![pos("x.shape[0]"), pos("-1")], &[("x", &["c", "h", "w"])], Ok(Some(&["c", "h*w"]))),
    ]);
}

// ── Reduction shape-rule tests for All / Any / ArgMax / ArgMin ──

#[test]
#[rustfmt::skip]
fn test_all_cases() {
    check(KnownFunction::All, None, &[
        ("all_axis_1_reduces_axis", vec![pos("x"), kw("axis", "1")], &[("x", &["batch", "features"])], Ok(Some(&["batch"]))),
        ("all_unknown_input_returns_none", vec![pos("x"), kw("axis", "1")], &[], Ok(None)),
        ("all_axis_none_reduces_all", vec![pos("x"), kw("axis", "None")], &[("x", &["batch", "features"])], Ok(Some(&[]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_any_cases() {
    check(KnownFunction::Any, None, &[
        ("any_axis_0_keepdims_true", vec![pos("x"), kw("axis", "0"), kw("keepdims", "True")], &[("x", &["batch", "features"])], Ok(Some(&["1", "features"]))),
        ("any_invalid_axis_errors", vec![pos("x"), kw("axis", "3")], &[("x", &["batch", "features"])], Err(&["out of bounds"])),
        ("any_negative_axis_too_negative_errors", vec![pos("x"), kw("axis", "-5")], &[("x", &["batch", "features"])], Err(&["out of bounds"])),
        ("any_keepdim_keyword_torch_style", vec![pos("x"), kw("dim", "1"), kw("keepdim", "True")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "1"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_arg_max_cases() {
    check(KnownFunction::ArgMax, None, &[
        ("argmax_axis_negative_removes_last_axis", vec![pos("x"), kw("axis", "-1")], &[("x", &["batch", "features"])], Ok(Some(&["batch"]))),
        ("argmax_dim_keyword_torch_style", vec![pos("x"), kw("dim", "1")], &[("x", &["batch", "features"])], Ok(Some(&["batch"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_arg_min_cases() {
    check(KnownFunction::ArgMin, None, &[
        // No axis = reduce all axes → scalar shape
        ("argmin_no_axis_reduces_all", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&[]))),
    ]);
}

// ── Shape-preserving tests for Argsort / Sort / Cumsum / Cumprod ──

#[test]
#[rustfmt::skip]
fn test_argsort_cases() {
    check(KnownFunction::Argsort, None, &[
        ("argsort_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
        ("argsort_unknown_input_returns_none", vec![pos("x")], &[], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_sort_cases() {
    check(KnownFunction::Sort, None, &[
        ("sort_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_cumsum_cases() {
    check(KnownFunction::Cumsum, None, &[
        ("cumsum_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_cumprod_cases() {
    check(KnownFunction::Cumprod, None, &[
        ("cumprod_preserves_shape", vec![pos("x")], &[("x", &["batch", "features"])], Ok(Some(&["batch", "features"]))),
    ]);
}

// ── linalg.inv shape rule tests ──

#[test]
#[rustfmt::skip]
fn test_linalg_inv_cases() {
    check(KnownFunction::LinalgInv, None, &[
        ("linalg_inv_square_preserves_shape", vec![pos("x")], &[("x", &["n", "n"])], Ok(Some(&["n", "n"]))),
        ("linalg_inv_batched_preserves_shape", vec![pos("x")], &[("x", &["batch", "n", "n"])], Ok(Some(&["batch", "n", "n"]))),
        ("linalg_inv_symbolic_square_dims_preserve", vec![pos("x")], &[("x", &["m", "m"])], Ok(Some(&["m", "m"]))),
        ("linalg_inv_non_square_errors", vec![pos("x")], &[("x", &["m", "n"])], Err(&["last two dimensions to match", "m", "n"])),
        ("linalg_inv_rank_1_errors", vec![pos("x")], &[("x", &["n"])], Err(&["rank >= 2"])),
        ("linalg_inv_unknown_input_returns_none", vec![pos("x")], &[], Ok(None)),
    ]);
}

// ── linalg.det shape rule tests ──

#[test]
#[rustfmt::skip]
fn test_linalg_det_cases() {
    check(KnownFunction::LinalgDet, None, &[
        ("linalg_det_2d_square_returns_scalar", vec![pos("x")], &[("x", &["n", "n"])], Ok(Some(&[]))),
        ("linalg_det_batched_square_returns_batch_prefix", vec![pos("x")], &[("x", &["batch", "n", "n"])], Ok(Some(&["batch"]))),
        ("linalg_det_multi_batch_returns_all_prefix_dims", vec![pos("x")], &[("x", &["b", "t", "n", "n"])], Ok(Some(&["b", "t"]))),
        ("linalg_det_symbolic_square_dims_work", vec![pos("x")], &[("x", &["m", "m"])], Ok(Some(&[]))),
        ("linalg_det_non_square_errors", vec![pos("x")], &[("x", &["m", "n"])], Err(&["last two dimensions to match", "m", "n"])),
        ("linalg_det_rank_1_errors", vec![pos("x")], &[("x", &["n"])], Err(&["rank >= 2"])),
        ("linalg_det_unknown_input_returns_none", vec![pos("x")], &[], Ok(None)),
    ]);
}

// ── Constructor coverage: Empty, Linspace, Logspace, Identity shape-rule tests ──

// Empty uses apply_known_shape_constructor (same as Zeros/Ones/Full)

#[test]
#[rustfmt::skip]
fn test_empty_cases() {
    check(KnownFunction::Empty, None, &[
        ("empty_positional_tuple_shape", vec![pos("(batch, features)")], &[], Ok(Some(&["batch", "features"]))),
        ("empty_keyword_shape", vec![kw("shape", "(2, 3)")], &[], Ok(Some(&["2", "3"]))),
        ("empty_no_args_returns_none", vec![], &[], Ok(None)),
    ]);
}

// Linspace uses apply_known_linspace

#[test]
#[rustfmt::skip]
fn test_linspace_cases() {
    check(KnownFunction::Linspace, None, &[
        ("linspace_positional_num", vec![pos("0"), pos("1"), pos("100")], &[], Ok(Some(&["100"]))),
        ("linspace_keyword_num", vec![pos("0"), pos("1"), kw("num", "steps")], &[], Ok(Some(&["steps"]))),
        ("linspace_default_num_is_50", vec![pos("0"), pos("1")], &[], Ok(Some(&["50"]))),
        ("torch_linspace_keyword_steps", vec![pos("0"), pos("1"), kw("steps", "steps")], &[], Ok(Some(&["steps"]))),
        ("torch_linspace_positional_steps", vec![pos("0"), pos("1"), pos("200")], &[], Ok(Some(&["200"]))),
    ]);
}

// Logspace uses apply_known_linspace (same helper)

#[test]
#[rustfmt::skip]
fn test_logspace_cases() {
    check(KnownFunction::Logspace, None, &[
        ("logspace_positional_num", vec![pos("0"), pos("3"), pos("n")], &[], Ok(Some(&["n"]))),
        ("logspace_keyword_num", vec![pos("0"), pos("3"), kw("num", "count")], &[], Ok(Some(&["count"]))),
        ("logspace_default_num_is_50", vec![pos("0"), pos("3")], &[], Ok(Some(&["50"]))),
    ]);
}

// Identity uses apply_known_eye (same helper as Eye)

#[test]
#[rustfmt::skip]
fn test_identity_cases() {
    check(KnownFunction::Identity, None, &[
        ("identity_square", vec![pos("n")], &[], Ok(Some(&["n", "n"]))),
        ("identity_no_args_returns_none", vec![], &[], Ok(None)),
    ]);
}

// ── einsum shape rule tests ──

#[test]
#[rustfmt::skip]
fn test_einsum_cases() {
    check(KnownFunction::Einsum, None, &[
        ("einsum_matmul_2d", vec![pos("\"ij,jk->ik\""), pos("a"), pos("b")], &[("a", &["m", "n"]), ("b", &["n", "p"])], Ok(Some(&["m", "p"]))),
        ("einsum_batched_matmul", vec![pos("\"bij,bjk->bik\""), pos("a"), pos("b")], &[("a", &["b", "m", "n"]), ("b", &["b", "n", "p"])], Ok(Some(&["b", "m", "p"]))),
        ("einsum_transpose", vec![pos("\"ij->ji\""), pos("x")], &[("x", &["a", "b"])], Ok(Some(&["b", "a"]))),
        ("einsum_sum_over_axis", vec![pos("\"ij->i\""), pos("x")], &[("x", &["a", "b"])], Ok(Some(&["a"]))),
        ("einsum_dim_mismatch_errors", vec![pos("\"ij,jk->ik\""), pos("a"), pos("b")], &[("a", &["m", "n"]), ("b", &["x", "p"])], Err(&[])),
        ("einsum_non_literal_equation_returns_none", vec![pos("equation"), pos("a"), pos("b")], &[("a", &["m", "n"]), ("b", &["n", "p"])], Ok(None)),
        ("einsum_ellipsis_returns_none", vec![pos("\"...ij,...jk->...ik\""), pos("a"), pos("b")], &[("a", &["b", "m", "n"]), ("b", &["b", "n", "p"])], Ok(None)),
        // "z" appears in output but not in any input subscript
        ("einsum_output_label_not_in_inputs_errors", vec![pos("\"ij->z\""), pos("a")], &[("a", &["m", "n"])], Err(&["output label 'z' not found"])),
        // "ijk" has 3 labels but operand has rank 2
        ("einsum_rank_mismatch_errors", vec![pos("\"ijk->ij\""), pos("a")], &[("a", &["m", "n"])], Err(&["rank 2 but subscript 'ijk' has length 3"])),
    ]);
}

// ── jax.lax higher-order / structured op shape rules ──

#[test]
#[rustfmt::skip]
fn test_lax_while_loop_cases() {
    check(KnownFunction::LaxWhileLoop, None, &[
        ("lax_while_loop_carry_invariant", vec![pos("cond_fn"), pos("body_fn"), pos("state")], &[("state", &["b", "d"])], Ok(Some(&["b", "d"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_fori_loop_cases() {
    check(KnownFunction::LaxForiLoop, None, &[
        ("lax_fori_loop_carry_invariant", vec![pos("0"), pos("10"), pos("body_fn"), pos("state")], &[("state", &["b", "d"])], Ok(Some(&["b", "d"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_associative_scan_cases() {
    check(KnownFunction::LaxAssociativeScan, None, &[
        ("lax_associative_scan_carry_invariant", vec![pos("combine_fn"), pos("elems")], &[("elems", &["n", "d"])], Ok(Some(&["n", "d"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_scatter_cases() {
    check(KnownFunction::LaxScatter, None, &[
        ("lax_scatter_shape_preserving_on_operand", vec![pos("operand"), pos("indices"), pos("updates")], &[("operand", &["b", "d"])], Ok(Some(&["b", "d"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_broadcast_cases() {
    check(KnownFunction::LaxBroadcast, None, &[
        ("lax_broadcast_prepends_sizes", vec![pos("x"), pos("(4, 5)")], &[("x", &["d"])], Ok(Some(&["4", "5", "d"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_broadcast_in_dim_cases() {
    check(KnownFunction::LaxBroadcastInDim, None, &[
        ("lax_broadcast_in_dim_uses_target_shape", vec![pos("x"), pos("(2, 3, 4)"), pos("(1,)")], &[("x", &["3"])], Ok(Some(&["2", "3", "4"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_slice_cases() {
    check(KnownFunction::LaxSlice, None, &[
        ("lax_slice_concrete_indices", vec![pos("x"), pos("(0, 1)"), pos("(4, 5)")], &[("x", &["8", "8"])], Ok(Some(&["4", "4"]))),
        ("lax_slice_with_strides", vec![pos("x"), pos("(0,)"), pos("(10,)"), pos("(2,)")], &[("x", &["10"])], Ok(Some(&["5"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_dynamic_slice_cases() {
    check(KnownFunction::LaxDynamicSlice, None, &[
        ("lax_dynamic_slice_uses_slice_sizes", vec![pos("x"), pos("(i, j)"), pos("(2, 3)")], &[("x", &["8", "8"])], Ok(Some(&["2", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_dynamic_update_slice_cases() {
    check(KnownFunction::LaxDynamicUpdateSlice, None, &[
        ("lax_dynamic_update_slice_shape_preserving", vec![pos("x"), pos("update"), pos("(0, 0)")], &[("x", &["8", "8"])], Ok(Some(&["8", "8"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_pad_cases() {
    check(KnownFunction::LaxPad, None, &[
        // dim0: 4 + 1 + 1 = 6; dim1: 3 + 1*(3-1) = 5
        ("lax_pad_low_high_interior", vec![pos("x"), pos("0.0"), pos("((1, 1, 0), (0, 0, 1))")], &[("x", &["4", "3"])], Ok(Some(&["6", "5"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_reduce_window_cases() {
    check(KnownFunction::LaxReduceWindow, None, &[
        ("lax_reduce_window_valid_padding_pooling_formula", vec![pos("x"), pos("-inf"), pos("max_fn"), pos("(1, 2, 2, 1)"), pos("(1, 2, 2, 1)"), pos("'VALID'")], &[("x", &["1", "8", "8", "3"])], Ok(Some(&["1", "4", "4", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_conv_general_dilated_cases() {
    check(KnownFunction::LaxConvGeneralDilated, None, &[
        ("lax_conv_general_dilated_default_dimension_numbers_valid", vec![pos("x"), pos("w"), pos("(1, 1)"), pos("'VALID'")], &[("x", &["1", "3", "8", "8"]), ("w", &["16", "3", "3", "3"])], Ok(Some(&["1", "16", "6", "6"]))),
        ("lax_conv_general_dilated_explicit_dimension_numbers_skips", vec![pos("x"), pos("w"), pos("(1, 1)"), pos("'VALID'"), kw("dimension_numbers", "('NHWC', 'HWIO', 'NHWC')")], &[("x", &["1", "8", "8", "3"]), ("w", &["3", "3", "3", "16"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_lax_gather_cases() {
    check(KnownFunction::LaxGather, None, &[
        // operand (2, 2), start_indices (2, 1): pick one row per index,
        // keeping the row's 2 columns. offset_dims=(1,) (output axis 1 is
        // the kept slice axis), collapsed_slice_dims=(0,) (the picked row
        // axis is dropped), slice_sizes=(1, 2).
        ("lax_gather_row_selection", vec![pos("operand"), pos("indices"), pos("jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))"), pos("(1, 2)")], &[("operand", &["2", "2"]), ("indices", &["2", "1"])], Ok(Some(&["2", "2"]))),
        // Batch dim comes from start_indices' leading (non-index-vector)
        // axis and can be symbolic; the offset (kept slice) dim is literal.
        ("lax_gather_symbolic_batch_dim", vec![pos("operand"), pos("indices"), pos("jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))"), pos("(1, 2)")], &[("operand", &["n", "2"]), ("indices", &["batch", "1"])], Ok(Some(&["batch", "2"]))),
        ("lax_gather_keyword_dimension_numbers_and_slice_sizes", vec![pos("operand"), pos("indices"), kw("dimension_numbers", "jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))"), kw("slice_sizes", "(1, 2)")], &[("operand", &["2", "2"]), ("indices", &["2", "1"])], Ok(Some(&["2", "2"]))),
        ("lax_gather_non_literal_dimension_numbers_skips", vec![pos("operand"), pos("indices"), pos("dnums"), pos("(1, 2)")], &[("operand", &["2", "2"]), ("indices", &["2", "1"])], Ok(None)),
        ("lax_gather_non_literal_slice_sizes_skips", vec![pos("operand"), pos("indices"), pos("jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))"), pos("sizes")], &[("operand", &["2", "2"]), ("indices", &["2", "1"])], Ok(None)),
    ]);
}

// ── jax.numpy / numpy array-creation shape rules ──

#[test]
#[rustfmt::skip]
fn test_diagflat_cases() {
    check(KnownFunction::Diagflat, None, &[
        ("diagflat_flattens_then_squares", vec![pos("x")], &[("x", &["2", "3"])], Ok(Some(&["6", "6"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_tri_cases() {
    check(KnownFunction::Tri, None, &[
        ("tri_default_square", vec![pos("4")], &[], Ok(Some(&["4", "4"]))),
        ("tri_rectangular", vec![pos("4"), pos("6")], &[], Ok(Some(&["4", "6"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_indices_cases() {
    check(KnownFunction::Indices, None, &[
        ("indices_prepends_rank", vec![pos("(2, 3)")], &[], Ok(Some(&["2", "2", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_select_cases() {
    check(KnownFunction::Select, None, &[
        ("select_approximates_first_choice_shape", vec![pos("[c1, c2]"), pos("[a, b]")], &[("a", &["n"]), ("b", &["n"])], Ok(Some(&["n"]))),
    ]);
}

// ── jax.numpy / numpy shape-transform shape rules ──

#[test]
#[rustfmt::skip]
fn test_roll_axis_cases() {
    check(KnownFunction::RollAxis, None, &[
        ("rollaxis_moves_axis_before_start", vec![pos("x"), pos("2"), pos("0")], &[("x", &["a", "b", "c"])], Ok(Some(&["c", "a", "b"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_resize_cases() {
    check(KnownFunction::Resize, None, &[
        ("resize_uses_target_shape", vec![pos("x"), pos("(3, 3)")], &[("x", &["2"])], Ok(Some(&["3", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_insert_cases() {
    check(KnownFunction::Insert, None, &[
        ("insert_scalar_grows_axis_by_one", vec![pos("x"), pos("1"), pos("9"), kw("axis", "0")], &[("x", &["4"])], Ok(Some(&["5"]))),
        ("insert_list_grows_axis_by_len", vec![pos("x"), pos("1"), pos("[9, 10, 11]"), kw("axis", "0")], &[("x", &["4"])], Ok(Some(&["7"]))),
        ("insert_axis_none_skips", vec![pos("x"), pos("1"), pos("9")], &[("x", &["4"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_delete_cases() {
    check(KnownFunction::Delete, None, &[
        ("delete_single_index_shrinks_axis_by_one", vec![pos("x"), pos("2"), kw("axis", "0")], &[("x", &["5"])], Ok(Some(&["4"]))),
        ("delete_list_shrinks_axis_by_len", vec![pos("x"), pos("[1, 2]"), kw("axis", "0")], &[("x", &["5"])], Ok(Some(&["3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_append_cases() {
    check(KnownFunction::Append, None, &[
        ("append_with_axis_like_concatenate", vec![pos("a"), pos("b"), kw("axis", "0")], &[("a", &["2", "d"]), ("b", &["3", "d"])], Ok(Some(&["5", "d"]))),
        ("append_axis_none_flattens_and_sums", vec![pos("a"), pos("b")], &[("a", &["2", "3"]), ("b", &["4"])], Ok(Some(&["10"]))),
    ]);
}

// ── jax.numpy / numpy joining-and-splitting shape rules ──

#[test]
fn test_hsplit_forces_axis_1_for_rank_2() {
    let args = vec![pos("x"), pos("2")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "6"]))]);
    let output = compute_fixed_axis_split_shapes(&KnownFunction::HSplit, &args, &shapes).unwrap();
    assert_eq!(output, Some(vec![shape(&["4", "3"]), shape(&["4", "3"])]));
}

#[test]
fn test_hsplit_forces_axis_0_for_rank_1() {
    let args = vec![pos("x"), pos("2")];
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let output = compute_fixed_axis_split_shapes(&KnownFunction::HSplit, &args, &shapes).unwrap();
    assert_eq!(output, Some(vec![shape(&["3"]), shape(&["3"])]));
}

#[test]
fn test_vsplit_uses_axis_0() {
    let args = vec![pos("x"), pos("2")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "6"]))]);
    let output = compute_fixed_axis_split_shapes(&KnownFunction::VSplit, &args, &shapes).unwrap();
    assert_eq!(output, Some(vec![shape(&["2", "6"]), shape(&["2", "6"])]));
}

#[test]
fn test_dsplit_uses_axis_2() {
    let args = vec![pos("x"), pos("2")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "6", "8"]))]);
    let output = compute_fixed_axis_split_shapes(&KnownFunction::DSplit, &args, &shapes).unwrap();
    assert_eq!(
        output,
        Some(vec![shape(&["4", "6", "4"]), shape(&["4", "6", "4"])])
    );
}

#[test]
fn test_dsplit_requires_rank_3_errors() {
    let args = vec![pos("x"), pos("2")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "6"]))]);
    let result = compute_fixed_axis_split_shapes(&KnownFunction::DSplit, &args, &shapes);
    assert!(result.is_err());
}

#[test]
#[rustfmt::skip]
fn test_kron_cases() {
    check(KnownFunction::Kron, None, &[
        ("kron_elementwise_product_of_dims", vec![pos("a"), pos("b")], &[("a", &["2", "3"]), ("b", &["4", "5"])], Ok(Some(&["8", "15"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_block_cases() {
    check(KnownFunction::Block, None, &[
        ("block_nested_2x2_assembly", vec![pos("[[a, b], [c, d]]")], &[("a", &["2", "3"]), ("b", &["2", "4"]), ("c", &["5", "3"]), ("d", &["5", "4"])], Ok(Some(&["7", "7"]))),
    ]);
}

// ── jax.numpy / numpy indexing-and-selection shape rules ──

#[test]
#[rustfmt::skip]
fn test_take_along_axis_cases() {
    check(KnownFunction::TakeAlongAxis, None, &[
        ("take_along_axis_matches_indices_shape", vec![pos("a"), pos("idx"), kw("axis", "1")], &[("a", &["4", "8"]), ("idx", &["4", "1"])], Ok(Some(&["4", "1"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_put_along_axis_cases() {
    check(KnownFunction::PutAlongAxis, None, &[
        ("put_along_axis_shape_preserving_on_arr", vec![pos("a"), pos("idx"), pos("values"), kw("axis", "1")], &[("a", &["4", "8"])], Ok(Some(&["4", "8"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_argwhere_cases() {
    check(KnownFunction::Argwhere, None, &[
        ("argwhere_known_rank_unknown_count", vec![pos("mask")], &[("mask", &["4", "8"])], Ok(Some(&["nonzero(mask)", "2"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_search_sorted_cases() {
    check(KnownFunction::SearchSorted, None, &[
        ("searchsorted_follows_values_shape", vec![pos("a"), pos("v")], &[("a", &["100"]), ("v", &["10"])], Ok(Some(&["10"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_histogram_cases() {
    check(KnownFunction::Histogram, None, &[
        ("histogram_default_bins", vec![pos("x")], &[], Ok(Some(&["10"]))),
        ("histogram_literal_bin_count", vec![pos("x"), pos("20")], &[], Ok(Some(&["20"]))),
        ("histogram_literal_edges_list", vec![pos("x"), pos("[0, 1, 2, 3]")], &[], Ok(Some(&["3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_bin_count_cases() {
    check(KnownFunction::BinCount, None, &[
        ("bincount_conservatively_unknown", vec![pos("x")], &[("x", &["100"])], Ok(None)),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_unique_cases() {
    check(KnownFunction::Unique, None, &[
        ("unique_conservatively_unknown", vec![pos("x")], &[("x", &["100"])], Ok(None)),
    ]);
}

// ── linear algebra shape rules ──

#[test]
#[rustfmt::skip]
fn test_cross_cases() {
    check(KnownFunction::Cross, None, &[
        ("cross_broadcasts_batch_dims", vec![pos("a"), pos("b")], &[("a", &["n", "3"]), ("b", &["3"])], Ok(Some(&["n", "3"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_linalg_solve_cases() {
    check(KnownFunction::LinalgSolve, None, &[
        ("linalg_solve_follows_rhs_shape", vec![pos("a"), pos("b")], &[("a", &["n", "n"]), ("b", &["n", "k"])], Ok(Some(&["n", "k"]))),
        ("linalg_solve_non_square_errors", vec![pos("a"), pos("b")], &[("a", &["m", "n"]), ("b", &["n", "k"])], Err(&[])),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_linalg_pinv_cases() {
    check(KnownFunction::LinalgPinv, None, &[
        ("linalg_pinv_swaps_last_two_dims", vec![pos("a")], &[("a", &["m", "n"])], Ok(Some(&["n", "m"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_linalg_matrix_rank_cases() {
    check(KnownFunction::LinalgMatrixRank, None, &[
        ("linalg_matrix_rank_drops_last_two_dims", vec![pos("a")], &[("a", &["b", "m", "n"])], Ok(Some(&["b"]))),
    ]);
}

#[test]
fn test_linalg_lstsq_solution_matrix_rhs() {
    let a = shape(&["m", "n"]);
    let b = shape(&["m", "k"]);
    assert_eq!(
        apply_known_linalg_lstsq_solution(&a, &b),
        Some(shape(&["n", "k"]))
    );
}

#[test]
fn test_linalg_lstsq_solution_vector_rhs() {
    let a = shape(&["m", "n"]);
    let b = shape(&["m"]);
    assert_eq!(
        apply_known_linalg_lstsq_solution(&a, &b),
        Some(shape(&["n"]))
    );
}

// ── einops shape rules ──

#[test]
#[rustfmt::skip]
fn test_einops_einsum_cases() {
    check(KnownFunction::EinopsEinsum, None, &[
        ("einops_einsum_pattern_is_last_positional", vec![pos("a"), pos("b"), pos("\"i j, j k -> i k\"")], &[("a", &["m", "n"]), ("b", &["n", "p"])], Ok(Some(&["m", "p"]))),
    ]);
}

#[test]
fn test_einops_pack_single_star_axis_like_concatenate() {
    let args = vec![pos("[a, b]"), pos("\"* d\"")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "d"])),
        ("b".to_string(), shape(&["3", "d"])),
    ]);
    let output = compute_einops_pack_shape(&args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["5", "d"])));
}

#[test]
#[rustfmt::skip]
fn test_einops_parse_shape_cases() {
    check(KnownFunction::EinopsParseShape, None, &[
        ("einops_parse_shape_always_none", vec![pos("x"), pos("\"h w\"")], &[("x", &["4", "5"])], Ok(None)),
    ]);
}

fn einops(
    func: KnownFunction,
    x_shape: &[&str],
    pattern: &str,
    extra: Vec<CallArgument>,
) -> Result<Option<Vec<String>>, String> {
    let mut args = vec![pos("x"), pos(&format!("\"{pattern}\""))];
    args.extend(extra);
    let shapes = HashMap::from([("x".to_string(), shape(x_shape))]);
    apply_known_function(&func, &args, &shapes)
}

#[test]
fn test_einops_ellipsis_rearrange_trailing_and_leading() {
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["b", "s", "h", "d"],
        "... h d -> ... (h d)",
        vec![],
    );
    assert_eq!(out, Ok(Some(shape(&["b", "s", "h*d"]))));
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["2", "3", "4", "5"],
        "b ... -> ... b",
        vec![],
    );
    assert_eq!(out, Ok(Some(shape(&["3", "4", "5", "2"]))));
}

#[test]
fn test_einops_ellipsis_middle_position() {
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["b", "2", "3", "d"],
        "b ... d -> d b ...",
        vec![],
    );
    assert_eq!(out, Ok(Some(shape(&["d", "b", "2", "3"]))));
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["b", "d"],
        "b ... d -> d b ...",
        vec![],
    );
    assert_eq!(out, Ok(Some(shape(&["d", "b"]))));
}

#[test]
fn test_einops_ellipsis_reduce_and_repeat() {
    let out = einops(
        KnownFunction::EinopsReduce,
        &["b", "h", "w", "c"],
        "b ... c -> b c",
        vec![pos("\"mean\"")],
    );
    assert_eq!(out, Ok(None));
    let out = einops(
        KnownFunction::EinopsReduce,
        &["b", "h", "w", "c"],
        "b ... c -> b ... ",
        vec![pos("\"max\"")],
    );
    assert_eq!(out, Ok(Some(shape(&["b", "h", "w"]))));
    let out = einops(
        KnownFunction::EinopsRepeat,
        &["b", "h", "w"],
        "b ... -> b ... c",
        vec![kw("c", "3")],
    );
    assert_eq!(out, Ok(Some(shape(&["b", "h", "w", "3"]))));
    let out = einops(
        KnownFunction::EinopsRepeat,
        &["b", "h", "w"],
        "b ... -> ... (n b)",
        vec![kw("n", "2")],
    );
    assert_eq!(out, Ok(Some(shape(&["h", "w", "2*b"]))));
}

#[test]
fn test_einops_ellipsis_rank_too_small_errors() {
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["b"],
        "b ... d -> d b ...",
        vec![],
    );
    assert!(out.is_err());
}

#[test]
fn test_einops_two_ellipses_unknown() {
    let out = einops(
        KnownFunction::EinopsRearrange,
        &["a", "b", "c"],
        "... b ... -> ... b",
        vec![],
    );
    assert_eq!(out, Ok(None));
}

// ── jax.nn shape rules ──

#[test]
#[rustfmt::skip]
fn test_one_hot_cases() {
    check(KnownFunction::OneHot, None, &[
        ("one_hot_appends_num_classes", vec![pos("x"), pos("10")], &[("x", &["batch"])], Ok(Some(&["batch", "10"]))),
        ("one_hot_symbolic_num_classes", vec![pos("x"), pos("n_classes")], &[("x", &["batch", "seq"])], Ok(Some(&["batch", "seq", "n_classes"]))),
    ]);
}

#[test]
#[rustfmt::skip]
fn test_dot_product_attention_cases() {
    check(KnownFunction::DotProductAttention, None, &[
        ("dot_product_attention_uses_query_and_value_head_dim", vec![pos("q"), pos("k"), pos("v")], &[("q", &["b", "sq", "h", "dk"]), ("k", &["b", "sk", "h", "dk"]), ("v", &["b", "sk", "h", "dv"])], Ok(Some(&["b", "sq", "h", "dv"]))),
    ]);
}

#[test]
fn test_scoped_new_classifications() {
    let classify = |parts: &[&str]| {
        classify_known_function(&ResolvedTarget {
            dots: 0,
            parts: parts.iter().map(|part| (*part).to_string()).collect(),
        })
    };
    for (path, kind) in [
        (&["jax", "lax", "log2"][..], KnownFunction::Copy),
        (&["jax", "lax", "one_minus_square"], KnownFunction::Copy),
        (&["jax", "scipy", "special", "erfcinv"], KnownFunction::Copy),
        (&["jax", "lax", "polynomial"], KnownFunction::LaxPolynomial),
        (
            &["jax", "random", "multinomial"],
            KnownFunction::RandomMultinomial,
        ),
        (&["jax", "numpy", "minmax"], KnownFunction::MinMax),
        (&["jax", "numpy", "top_k"], KnownFunction::LaxTopK),
        (&["jax", "numpy", "array_split"], KnownFunction::ArraySplit),
        (&["numpy", "pad"], KnownFunction::NumpyPad),
        (&["torch", "cross"], KnownFunction::TorchCross),
        (
            &["torch", "linalg", "matrix_sqrth"],
            KnownFunction::LinalgMatrixSqrth,
        ),
    ] {
        assert_eq!(classify(path), Some(kind), "{path:?}");
    }
    assert_eq!(classify(&["jax", "numpy", "pad"]), Some(KnownFunction::Pad));
    assert_eq!(classify(&["scipy", "special", "erfcinv"]), None);
    assert_eq!(classify(&["numpy", "minmax"]), None);
    assert_eq!(classify(&["numpy", "top_k"]), None);
    assert_eq!(classify(&["jax", "numpy", "linalg", "matrix_sqrth"]), None);
}

#[test]
fn test_unary_new_rules_keep_shape_with_documented_keywords() {
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "4"]))]);
    for args in [vec![kw("x", "x")], vec![kw("y", "x")]] {
        assert_eq!(
            apply_known_function(&KnownFunction::Copy, &args, &shapes).unwrap(),
            Some(shape(&["n", "4"]))
        );
    }
}

#[test]
fn test_polynomial_broadcasts_only_known_coefficients() {
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["1", "4"])),
        ("a".to_string(), shape(&["3", "1"])),
        ("coeffs".to_string(), shape(&["5", "3", "1"])),
        ("empty".to_string(), shape(&["0", "3", "1"])),
        ("dynamic".to_string(), shape(&["n", "3", "1"])),
        ("bad".to_string(), shape(&["2", "5"])),
        ("rank".to_string(), shape(&["4"])),
    ]);
    let run = |coeff| {
        apply_known_function(
            &KnownFunction::LaxPolynomial,
            &[pos("x"), kw("coeffs", coeff)],
            &shapes,
        )
    };
    assert_eq!(run("[1., a, 3]").unwrap(), Some(shape(&["3", "4"])));
    assert_eq!(run("coeffs").unwrap(), Some(shape(&["3", "4"])));
    assert_eq!(run("[missing]").unwrap(), None);
    assert_eq!(run("dynamic").unwrap(), None);
    assert_eq!(run("[]").unwrap(), Some(shape(&["1", "4"])));
    assert_eq!(run("empty").unwrap(), Some(shape(&["1", "4"])));
    assert!(run("[rank]").is_err());
    assert!(run("[bad]").is_err());
}

#[test]
fn test_multinomial_batch_shape_and_validation() {
    let shapes = HashMap::from([
        ("p".to_string(), shape(&["2", "1", "7"])),
        ("single".to_string(), shape(&["7"])),
        ("n".to_string(), shape(&["1", "3"])),
    ]);
    let run = |p, n, extra: Vec<CallArgument>| {
        let mut args = vec![pos("key"), pos(n), pos(p)];
        args.extend(extra);
        apply_known_function(&KnownFunction::RandomMultinomial, &args, &shapes)
    };
    assert_eq!(
        run("p", "3", vec![]).unwrap(),
        Some(shape(&["2", "1", "7"]))
    );
    assert_eq!(
        run("p", "n", vec![kw("shape", "(2, 3)")]).unwrap(),
        Some(shape(&["2", "3", "7"]))
    );
    assert_eq!(
        run("single", "3", vec![kw("shape", "()")]).unwrap(),
        Some(shape(&["7"]))
    );
    assert_eq!(run("p", "3", vec![kw("shape", "dynamic")]).unwrap(), None);
    assert!(run("p", "n", vec![]).is_err());
    assert!(run("p", "3", vec![kw("shape", "(4, 3)")]).is_err());
    assert!(run("p", "3", vec![kw("shape", "()")]).is_err());
}

#[test]
fn test_numpy_pad_dictionary_is_not_shared_with_jax_pad() {
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "4"]))]);
    let args = [pos("x"), kw("pad_width", "{0: (1, 2), -1: 3}")];
    assert_eq!(
        apply_known_function(&KnownFunction::NumpyPad, &args, &shapes).unwrap(),
        Some(shape(&["n+3", "10"]))
    );
    assert_eq!(
        apply_known_function(&KnownFunction::Pad, &args, &shapes).unwrap(),
        None
    );
    for width in ["{0: dynamic}", "{0: (1, -2)}", "{axis: 2}", "widths"] {
        assert_eq!(
            apply_known_function(
                &KnownFunction::NumpyPad,
                &[pos("x"), kw("pad_width", width)],
                &shapes
            )
            .unwrap(),
            None
        );
    }
}

#[test]
fn test_cross_vector_axes_and_torch_default() {
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["3", "1"])),
        ("b".to_string(), shape(&["3", "5"])),
        ("c".to_string(), shape(&["2", "3"])),
        ("v".to_string(), shape(&["3"])),
        ("same_rank".to_string(), shape(&["1", "3"])),
        ("uncertain".to_string(), shape(&["n", "3"])),
        ("wrong".to_string(), shape(&["2", "5"])),
        ("symbolic".to_string(), shape(&["n", "5"])),
    ]);
    let run = |kind, args: Vec<CallArgument>| apply_known_function(&kind, &args, &shapes);
    assert_eq!(
        run(
            KnownFunction::Cross,
            vec![pos("a"), pos("b"), kw("axis", "0")]
        )
        .unwrap(),
        Some(shape(&["3", "5"]))
    );
    assert_eq!(
        run(
            KnownFunction::Cross,
            vec![pos("c"), pos("v"), kw("axisa", "1"), kw("axisc", "0")]
        )
        .unwrap(),
        Some(shape(&["3", "2"]))
    );
    assert_eq!(
        run(
            KnownFunction::Cross,
            vec![pos("c"), pos("v"), kw("axis", "dynamic")]
        )
        .unwrap(),
        None
    );
    assert!(run(
        KnownFunction::Cross,
        vec![pos("wrong"), pos("b"), kw("axis", "0")]
    )
    .is_err());
    assert_eq!(
        run(
            KnownFunction::Cross,
            vec![pos("symbolic"), pos("b"), kw("axis", "0")]
        )
        .unwrap(),
        None
    );
    assert!(run(KnownFunction::TorchCross, vec![pos("c"), pos("v")]).is_err());
    assert!(run(KnownFunction::TorchCross, vec![pos("wrong"), pos("wrong")]).is_err());
    assert_eq!(
        run(KnownFunction::TorchCross, vec![pos("c"), pos("same_rank")]).unwrap(),
        Some(shape(&["2", "3"]))
    );
    assert_eq!(
        run(
            KnownFunction::TorchCross,
            vec![pos("uncertain"), pos("uncertain")]
        )
        .unwrap(),
        None
    );
    assert_eq!(
        run(
            KnownFunction::TorchCross,
            vec![pos("uncertain"), pos("uncertain"), kw("dim", "1")]
        )
        .unwrap(),
        Some(shape(&["n", "3"]))
    );
    assert_eq!(
        run(
            KnownFunction::TorchCross,
            vec![pos("a"), pos("b"), kw("dim", "0")]
        )
        .unwrap(),
        Some(shape(&["3", "5"]))
    );
    assert!(run(
        KnownFunction::TorchCross,
        vec![pos("a"), pos("b"), kw("dim", "1")]
    )
    .is_err());
}

#[test]
fn test_matrix_sqrth_keeps_only_square_matrix_shapes() {
    let shapes = HashMap::from([
        ("square".to_string(), shape(&["b", "3", "3"])),
        ("rect".to_string(), shape(&["b", "3", "4"])),
    ]);
    let run = |name| apply_known_function(&KnownFunction::LinalgMatrixSqrth, &[pos(name)], &shapes);
    assert_eq!(run("square").unwrap(), Some(shape(&["b", "3", "3"])));
    assert_eq!(
        apply_known_function(
            &KnownFunction::LinalgMatrixSqrth,
            &[kw("A", "square")],
            &shapes
        )
        .unwrap(),
        Some(shape(&["b", "3", "3"]))
    );
    assert_eq!(
        apply_known_function(
            &KnownFunction::LinalgMatrixSqrth,
            &[kw("a", "square")],
            &shapes
        )
        .unwrap(),
        None
    );
    assert!(run("rect").unwrap_err().contains("linalg.matrix_sqrth"));
    assert_eq!(run("unknown").unwrap(), None);
}

fn run_bmm(args: Vec<CallArgument>, shapes: &[(&str, &[&str])]) -> Result<Option<Vec<String>>, String> {
    let shapes: HashMap<String, Vec<String>> = shapes
        .iter()
        .map(|(name, dims)| (name.to_string(), shape(dims)))
        .collect();
    apply_known_function(&KnownFunction::Bmm, &args, &shapes)
}

#[test]
fn test_bmm_rank3_batch_and_inner_dims() {
    let operands: &[(&str, &[&str])] = &[("a", &["b", "n", "m"]), ("c", &["b", "m", "p"])];
    assert_eq!(
        run_bmm(vec![pos("a"), pos("c")], operands).unwrap(),
        Some(shape(&["b", "n", "p"]))
    );
    assert_eq!(
        run_bmm(vec![kw("input", "a"), kw("mat2", "c")], operands).unwrap(),
        Some(shape(&["b", "n", "p"]))
    );
    let numeric: &[(&str, &[&str])] = &[("a", &["4", "3", "0"]), ("c", &["4", "0", "5"])];
    assert_eq!(
        run_bmm(vec![pos("a"), pos("c")], numeric).unwrap(),
        Some(shape(&["4", "3", "5"]))
    );
}

#[test]
fn test_bmm_never_broadcasts_the_batch_dim() {
    // torch.bmm(randn(4,3,4), randn(1,4,5)) raises, unlike torch.matmul.
    let operands: &[(&str, &[&str])] = &[("a", &["4", "3", "4"]), ("c", &["1", "4", "5"])];
    let error = run_bmm(vec![pos("a"), pos("c")], operands).unwrap_err();
    assert!(error.contains("bmm batch"), "{error}");
    let reversed: &[(&str, &[&str])] = &[("a", &["1", "3", "4"]), ("c", &["b", "4", "5"])];
    assert!(run_bmm(vec![pos("a"), pos("c")], reversed).is_err());
}

#[test]
fn test_bmm_inner_dim_mismatch_errors() {
    let operands: &[(&str, &[&str])] = &[("a", &["b", "n", "m"]), ("c", &["b", "k", "p"])];
    let error = run_bmm(vec![pos("a"), pos("c")], operands).unwrap_err();
    assert!(error.contains("bmm") && error.contains("dimension mismatch"), "{error}");
}

#[test]
fn test_bmm_requires_rank_3_operands() {
    for (left, right) in [
        (&["n", "m"][..], &["m", "p"][..]),
        (&["b", "n", "m"][..], &["m", "p"][..]),
        (&["b", "n", "m"][..], &["m"][..]),
        (&["a", "b", "n", "m"][..], &["a", "b", "m", "p"][..]),
        (&["m"][..], &["m"][..]),
    ] {
        let operands: &[(&str, &[&str])] = &[("a", left), ("c", right)];
        let error = run_bmm(vec![pos("a"), pos("c")], operands).unwrap_err();
        assert!(error.contains("bmm") && error.contains("3"), "{left:?} {right:?}: {error}");
    }
}

#[test]
fn test_bmm_unknown_operand_stays_unknown() {
    let operands: &[(&str, &[&str])] = &[("a", &["b", "n", "m"])];
    assert_eq!(run_bmm(vec![pos("a"), pos("c")], operands).unwrap(), None);
    assert_eq!(run_bmm(vec![pos("a")], operands).unwrap(), None);
}

fn run_loss(
    function: KnownFunction,
    args: Vec<CallArgument>,
    shapes: &[(&str, &[&str])],
) -> Result<Option<Vec<String>>, String> {
    let shapes: HashMap<String, Vec<String>> = shapes
        .iter()
        .map(|(name, dims)| (name.to_string(), shape(dims)))
        .collect();
    apply_known_function(&function, &args, &shapes)
}

#[test]
fn test_cross_entropy_class_index_targets_per_reduction() {
    let ce = |args: Vec<CallArgument>, shapes: &[(&str, &[&str])]| {
        run_loss(KnownFunction::CrossEntropy, args, shapes)
    };
    let batched: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"])];
    let unbatched: &[(&str, &[&str])] = &[("x", &["c"]), ("y", &[])];
    let spatial: &[(&str, &[&str])] = &[("x", &["n", "c", "h", "w"]), ("y", &["n", "h", "w"])];
    for (shapes, none) in [
        (batched, shape(&["n"])),
        (unbatched, shape(&[])),
        (spatial, shape(&["n", "h", "w"])),
    ] {
        assert_eq!(ce(vec![pos("x"), pos("y")], shapes).unwrap(), Some(shape(&[])));
        for reduction in ["'mean'", "\"sum\""] {
            let args = vec![pos("x"), pos("y"), kw("reduction", reduction)];
            assert_eq!(ce(args, shapes).unwrap(), Some(shape(&[])));
        }
        let args = vec![pos("x"), pos("y"), kw("reduction", "'none'")];
        assert_eq!(ce(args, shapes).unwrap(), Some(none));
    }
    let keywords = vec![kw("input", "x"), kw("target", "y"), kw("reduction", "'none'")];
    assert_eq!(ce(keywords, batched).unwrap(), Some(shape(&["n"])));
}

#[test]
fn test_cross_entropy_probability_targets_per_reduction() {
    let ce = |args: Vec<CallArgument>, shapes: &[(&str, &[&str])]| {
        run_loss(KnownFunction::CrossEntropy, args, shapes)
    };
    let batched: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n", "c"])];
    let unbatched: &[(&str, &[&str])] = &[("x", &["c"]), ("y", &["c"])];
    let spatial: &[(&str, &[&str])] = &[("x", &["n", "c", "l"]), ("y", &["n", "c", "l"])];
    assert_eq!(ce(vec![pos("x"), pos("y")], batched).unwrap(), Some(shape(&[])));
    let none = |shapes| ce(vec![pos("x"), pos("y"), kw("reduction", "'none'")], shapes);
    assert_eq!(none(batched).unwrap(), Some(shape(&["n"])));
    assert_eq!(none(unbatched).unwrap(), Some(shape(&[])));
    assert_eq!(none(spatial).unwrap(), Some(shape(&["n", "l"])));
}

#[test]
fn test_cross_entropy_positional_reduction_and_label_smoothing() {
    let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"])];
    let positional = vec![
        pos("x"), pos("y"), pos("None"), pos("None"), pos("-100"), pos("None"), pos("'none'"),
    ];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, positional, shapes).unwrap(),
        Some(shape(&["n"]))
    );
    let smoothed = vec![pos("x"), pos("y"), kw("reduction", "'none'"), kw("label_smoothing", "0.1")];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, smoothed, shapes).unwrap(),
        Some(shape(&["n"]))
    );
}

#[test]
fn test_cross_entropy_value_dependent_reduction_stays_unknown() {
    let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"])];
    for extra in [
        kw("reduction", "mode"),
        kw("reduction", "None"),
        kw("reduce", "False"),
        kw("size_average", "False"),
    ] {
        let args = vec![pos("x"), pos("y"), extra];
        assert_eq!(run_loss(KnownFunction::CrossEntropy, args, shapes).unwrap(), None);
    }
    // A legacy argument overrides `reduction`, even an invalid one.
    let args = vec![pos("x"), pos("y"), kw("reduce", "False"), kw("reduction", "'Mean'")];
    assert_eq!(run_loss(KnownFunction::CrossEntropy, args, shapes).unwrap(), None);
    // Legacy arguments given as None change nothing.
    let args = vec![
        pos("x"), pos("y"), kw("size_average", "None"), kw("reduce", "None"), kw("reduction", "'none'"),
    ];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, args, shapes).unwrap(),
        Some(shape(&["n"]))
    );
}

#[test]
fn test_cross_entropy_invalid_reduction_string_errors() {
    let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"])];
    let args = vec![pos("x"), pos("y"), kw("reduction", "'Mean'")];
    let error = run_loss(KnownFunction::CrossEntropy, args, shapes).unwrap_err();
    assert!(error.contains("reduction"), "{error}");
}

#[test]
fn test_cross_entropy_target_shape_errors() {
    let bad: &[(&[&str], &[&str])] = &[
        (&["n", "c"], &["m"]),
        (&["n", "c"], &[]),
        (&["n", "c"], &["n", "1"]),
        (&["n", "c"], &["n", "c", "2"]),
        (&["n", "c", "l"], &["n"]),
        (&["n", "c", "l"], &["l"]),
        (&["n", "c", "l"], &["n", "c", "l", "2"]),
        (&["c"], &["2", "2"]),
        (&["4", "c"], &["3", "c"]),
        (&[], &[]),
    ];
    for function in [KnownFunction::CrossEntropy, KnownFunction::NllLoss] {
        for (input, target) in bad {
            let shapes: &[(&str, &[&str])] = &[("x", input), ("y", target)];
            let result = run_loss(function.clone(), vec![pos("x"), pos("y")], shapes);
            assert!(result.is_err(), "{function:?} {input:?} {target:?}: {result:?}");
        }
    }
}

#[test]
fn test_unbatched_rank1_target_of_size_one_is_a_scalar_loss() {
    // F.cross_entropy(randn(C), zeros(1, long), reduction='none') -> ().
    let shapes: &[(&str, &[&str])] = &[("x", &["c"]), ("y", &["1"])];
    let args = vec![pos("x"), pos("y"), kw("reduction", "'none'")];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, args.clone(), shapes).unwrap(),
        Some(shape(&[]))
    );
    assert_eq!(run_loss(KnownFunction::NllLoss, args, shapes).unwrap(), Some(shape(&[])));
    // Whether a symbolic size can be 1 depends on values.
    let symbolic: &[(&str, &[&str])] = &[("x", &["c"]), ("y", &["k"])];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, vec![pos("x"), pos("y")], symbolic).unwrap(),
        None
    );
}

#[test]
fn test_cross_entropy_class_weight_must_cover_every_class() {
    let ok: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"]), ("w", &["c"])];
    let weighted = vec![pos("x"), pos("y"), kw("weight", "w")];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, weighted.clone(), ok).unwrap(),
        Some(shape(&[]))
    );
    for weight in [&["k"][..], &["c", "1"][..], &[][..]] {
        let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"]), ("w", weight)];
        let error = run_loss(KnownFunction::CrossEntropy, weighted.clone(), shapes).unwrap_err();
        assert!(error.contains("weight"), "{weight:?}: {error}");
        assert!(run_loss(KnownFunction::NllLoss, weighted.clone(), shapes).is_err());
    }
    let positional = vec![pos("x"), pos("y"), pos("w")];
    let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"]), ("w", &["k"])];
    assert!(run_loss(KnownFunction::CrossEntropy, positional, shapes).is_err());
    // An unknown weight stays unknown, and None is the default.
    let none_weight = vec![pos("x"), pos("y"), kw("weight", "None")];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, none_weight, ok).unwrap(),
        Some(shape(&[]))
    );
    let unknown = vec![pos("x"), pos("y"), kw("weight", "mystery")];
    assert_eq!(
        run_loss(KnownFunction::CrossEntropy, unknown, ok).unwrap(),
        Some(shape(&[]))
    );
}

#[test]
fn test_cross_entropy_unknown_operand_stays_unknown() {
    let shapes: &[(&str, &[&str])] = &[("x", &["n", "c"])];
    for function in [KnownFunction::CrossEntropy, KnownFunction::NllLoss] {
        assert_eq!(run_loss(function.clone(), vec![pos("x"), pos("y")], shapes).unwrap(), None);
        assert_eq!(run_loss(function.clone(), vec![pos("z"), pos("x")], shapes).unwrap(), None);
        assert_eq!(run_loss(function, vec![pos("x")], shapes).unwrap(), None);
    }
}

#[test]
fn test_nll_loss_takes_class_indices_only() {
    let nll = |args: Vec<CallArgument>, shapes: &[(&str, &[&str])]| {
        run_loss(KnownFunction::NllLoss, args, shapes)
    };
    let batched: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n"])];
    let spatial: &[(&str, &[&str])] = &[("x", &["n", "c", "h"]), ("y", &["n", "h"])];
    let unbatched: &[(&str, &[&str])] = &[("x", &["c"]), ("y", &[])];
    let none = |shapes| nll(vec![pos("x"), pos("y"), kw("reduction", "'none'")], shapes);
    assert_eq!(nll(vec![pos("x"), pos("y")], batched).unwrap(), Some(shape(&[])));
    assert_eq!(none(batched).unwrap(), Some(shape(&["n"])));
    assert_eq!(none(spatial).unwrap(), Some(shape(&["n", "h"])));
    assert_eq!(none(unbatched).unwrap(), Some(shape(&[])));
    // A target shaped like the input is a probability target: nll_loss rejects it.
    let same: &[(&str, &[&str])] = &[("x", &["n", "c"]), ("y", &["n", "c"])];
    assert!(nll(vec![pos("x"), pos("y")], same).is_err());
}
