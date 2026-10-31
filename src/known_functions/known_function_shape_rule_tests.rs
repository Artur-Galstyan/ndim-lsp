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
fn test_concatenate_commutative_symbolic_dims_do_not_mismatch() {
    // "d*2" and "2*d" are the same dim under canonicalization — must not
    // be a false "dimension mismatch" (issue: dims were compared as raw
    // strings).
    let args = vec![pos("[a, b]"), kw("axis", "1")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["d*2", "3"])),
        ("b".to_string(), shape(&["2*d", "5"])),
    ]);

    let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["d*2", "8"])));
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
fn test_stack_commutative_symbolic_dims_do_not_mismatch() {
    let args = vec![pos("[a, b]")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["batch", "d_state*2"])),
        ("b".to_string(), shape(&["batch", "2*d_state"])),
    ]);

    let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "batch", "d_state*2"])));
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

    assert!(error.contains("axis 0 given more than once"), "{error}");
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
fn test_matmul_commutative_symbolic_inner_dim_does_not_mismatch() {
    // "k*2" and "2*k" are the same contracted dim under canonicalization.
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "k*2"])),
        ("b".to_string(), shape(&["2*k", "n"])),
    ]);

    let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "n"])));
}

#[test]
fn test_matmul_commutative_symbolic_batch_dims_broadcast() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["b_state*2", "m", "k"])),
        ("b".to_string(), shape(&["2*b_state", "k", "n"])),
    ]);

    let output = apply_known_function(&KnownFunction::Matmul, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["b_state*2", "m", "n"])));
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

    let error = apply_known_function(&KnownFunction::BroadcastArrays, &args, &shapes).unwrap_err();

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

    let output = apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_ones_like_preserves_shape() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

    let output = apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_full_like_preserves_shape() {
    let args = vec![pos("x"), pos("0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

    let output = apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_empty_like_preserves_shape() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

    let output = apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_zeros_like_missing_input_returns_none() {
    let args = vec![pos("x")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_ones_like_missing_input_returns_none() {
    let args = vec![pos("x")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_full_like_missing_input_returns_none() {
    let args = vec![pos("x"), pos("0")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_empty_like_missing_input_returns_none() {
    let args = vec![pos("x")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_zeros_like_keyword_x() {
    let args = vec![kw("x", "arr")];
    let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

    let output = apply_known_function(&KnownFunction::ZerosLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "n"])));
}

#[test]
fn test_ones_like_keyword_input() {
    let args = vec![kw("input", "arr")];
    let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

    let output = apply_known_function(&KnownFunction::OnesLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "n"])));
}

#[test]
fn test_full_like_keyword_x() {
    let args = vec![kw("x", "arr"), pos("0")];
    let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

    let output = apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "n"])));
}

#[test]
fn test_empty_like_keyword_input() {
    let args = vec![kw("input", "arr")];
    let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

    let output = apply_known_function(&KnownFunction::EmptyLike, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "n"])));
}

#[test]
fn test_zeros_like_no_args_returns_none() {
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::ZerosLike, &[], &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_full_like_unrecognized_keyword_returns_none() {
    let args = vec![kw("template", "arr")];
    let shapes = HashMap::from([("arr".to_string(), shape(&["m", "n"]))]);

    let output = apply_known_function(&KnownFunction::FullLike, &args, &shapes).unwrap();

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
fn test_reshape_symbolic_minus_one_cancels_known_dim() {
    let args = vec![pos("x"), pos("(batch, -1)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

    let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_reshape_symbolic_minus_one_flattens_remaining_dims() {
    let args = vec![pos("x"), pos("(c, -1)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["c", "h", "w"]))]);

    let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["c", "h*w"])));
}

#[test]
fn test_reshape_shape_index_resolves_to_input_dim() {
    let args = vec![pos("x"), pos("(x.shape[0], -1)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["c", "h", "w"]))]);

    let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["c", "h*w"])));
}

#[test]
fn test_reshape_shape_index_negative_resolves() {
    let args = vec![pos("x"), pos("(x.shape[-1], -1)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["c", "h", "w"]))]);

    let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["w", "c*h"])));
}

#[test]
fn test_reshape_symbolic_minus_one_unmatched_known_dim_returns_none() {
    let args = vec![pos("x"), pos("(z, -1)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["a", "b"]))]);

    let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_reshape_method_form_with_shape_index_and_minus_one() {
    let args = vec![pos("x.shape[0]"), pos("-1")];
    let shapes = HashMap::from([("x".to_string(), shape(&["c", "h", "w"]))]);

    let output = apply_method_call(&KnownFunction::Reshape, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["c", "h*w"])));
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

#[test]
fn test_any_negative_axis_too_negative_errors() {
    let args = vec![pos("x"), kw("axis", "-5")];
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

// ── linalg.det shape rule tests ──

#[test]
fn test_linalg_det_2d_square_returns_scalar() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "n"]))]);

    let output = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap();

    assert_eq!(output, Some(Vec::new()));
}

#[test]
fn test_linalg_det_batched_square_returns_batch_prefix() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "n", "n"]))]);

    let output = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch"])));
}

#[test]
fn test_linalg_det_multi_batch_returns_all_prefix_dims() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["b", "t", "n", "n"]))]);

    let output = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["b", "t"])));
}

#[test]
fn test_linalg_det_symbolic_square_dims_work() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["m", "m"]))]);

    let output = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap();

    assert_eq!(output, Some(Vec::new()));
}

#[test]
fn test_linalg_det_non_square_errors() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["m", "n"]))]);

    let error = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap_err();

    assert!(error.contains("last two dimensions to match"));
    assert!(error.contains("m"));
    assert!(error.contains("n"));
}

#[test]
fn test_linalg_det_rank_1_errors() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["n"]))]);

    let error = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap_err();

    assert!(error.contains("rank >= 2"));
}

#[test]
fn test_linalg_det_unknown_input_returns_none() {
    let args = vec![pos("x")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::LinalgDet, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

// ── Constructor coverage: Empty, Linspace, Logspace, Identity shape-rule tests ──

// Empty uses apply_known_shape_constructor (same as Zeros/Ones/Full)

#[test]
fn test_empty_positional_tuple_shape() {
    let args = vec![pos("(batch, features)")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Empty, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_empty_keyword_shape() {
    let args = vec![kw("shape", "(2, 3)")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Empty, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "3"])));
}

#[test]
fn test_empty_no_args_returns_none() {
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Empty, &[], &shapes).unwrap();

    assert_eq!(output, None);
}

// Linspace uses apply_known_linspace

#[test]
fn test_linspace_positional_num() {
    let args = vec![pos("0"), pos("1"), pos("100")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Linspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["100"])));
}

#[test]
fn test_linspace_keyword_num() {
    let args = vec![pos("0"), pos("1"), kw("num", "steps")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Linspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["steps"])));
}

#[test]
fn test_linspace_default_num_is_50() {
    let args = vec![pos("0"), pos("1")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Linspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["50"])));
}

#[test]
fn test_torch_linspace_keyword_steps() {
    let args = vec![pos("0"), pos("1"), kw("steps", "steps")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Linspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["steps"])));
}

#[test]
fn test_torch_linspace_positional_steps() {
    let args = vec![pos("0"), pos("1"), pos("200")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Linspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["200"])));
}

// Logspace uses apply_known_linspace (same helper)

#[test]
fn test_logspace_positional_num() {
    let args = vec![pos("0"), pos("3"), pos("n")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Logspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["n"])));
}

#[test]
fn test_logspace_keyword_num() {
    let args = vec![pos("0"), pos("3"), kw("num", "count")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Logspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["count"])));
}

#[test]
fn test_logspace_default_num_is_50() {
    let args = vec![pos("0"), pos("3")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Logspace, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["50"])));
}

// Identity uses apply_known_eye (same helper as Eye)

#[test]
fn test_identity_square() {
    let args = vec![pos("n")];
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Identity, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["n", "n"])));
}

#[test]
fn test_identity_no_args_returns_none() {
    let shapes = HashMap::new();

    let output = apply_known_function(&KnownFunction::Identity, &[], &shapes).unwrap();

    assert_eq!(output, None);
}

// ── einsum shape rule tests ──

#[test]
fn test_einsum_matmul_2d() {
    let args = vec![pos("\"ij,jk->ik\""), pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "n"])),
        ("b".to_string(), shape(&["n", "p"])),
    ]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["m", "p"])));
}

#[test]
fn test_einsum_batched_matmul() {
    let args = vec![pos("\"bij,bjk->bik\""), pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["b", "m", "n"])),
        ("b".to_string(), shape(&["b", "n", "p"])),
    ]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["b", "m", "p"])));
}

#[test]
fn test_einsum_transpose() {
    let args = vec![pos("\"ij->ji\""), pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["a", "b"]))]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["b", "a"])));
}

#[test]
fn test_einsum_sum_over_axis() {
    let args = vec![pos("\"ij->i\""), pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["a", "b"]))]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["a"])));
}

#[test]
fn test_einsum_dim_mismatch_errors() {
    let args = vec![pos("\"ij,jk->ik\""), pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "n"])),
        ("b".to_string(), shape(&["x", "p"])),
    ]);

    let result = apply_known_function(&KnownFunction::Einsum, &args, &shapes);

    assert!(result.is_err());
}

#[test]
fn test_einsum_non_literal_equation_returns_none() {
    let args = vec![pos("equation"), pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "n"])),
        ("b".to_string(), shape(&["n", "p"])),
    ]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_einsum_ellipsis_returns_none() {
    let args = vec![pos("\"...ij,...jk->...ik\""), pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["b", "m", "n"])),
        ("b".to_string(), shape(&["b", "n", "p"])),
    ]);

    let output = apply_known_function(&KnownFunction::Einsum, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_einsum_output_label_not_in_inputs_errors() {
    // "z" appears in output but not in any input subscript
    let args = vec![pos("\"ij->z\""), pos("a")];
    let shapes = HashMap::from([("a".to_string(), shape(&["m", "n"]))]);

    let result = apply_known_function(&KnownFunction::Einsum, &args, &shapes);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("output label 'z' not found"));
}

#[test]
fn test_einsum_rank_mismatch_errors() {
    // "ijk" has 3 labels but operand has rank 2
    let args = vec![pos("\"ijk->ij\""), pos("a")];
    let shapes = HashMap::from([("a".to_string(), shape(&["m", "n"]))]);

    let result = apply_known_function(&KnownFunction::Einsum, &args, &shapes);

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("rank 2 but subscript 'ijk' has length 3"));
}

// ── jax.lax higher-order / structured op shape rules ──

#[test]
fn test_lax_while_loop_carry_invariant() {
    let args = vec![pos("cond_fn"), pos("body_fn"), pos("state")];
    let shapes = HashMap::from([("state".to_string(), shape(&["b", "d"]))]);
    let output = apply_known_function(&KnownFunction::LaxWhileLoop, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b", "d"])));
}

#[test]
fn test_lax_fori_loop_carry_invariant() {
    let args = vec![pos("0"), pos("10"), pos("body_fn"), pos("state")];
    let shapes = HashMap::from([("state".to_string(), shape(&["b", "d"]))]);
    let output = apply_known_function(&KnownFunction::LaxForiLoop, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b", "d"])));
}

#[test]
fn test_lax_associative_scan_carry_invariant() {
    let args = vec![pos("combine_fn"), pos("elems")];
    let shapes = HashMap::from([("elems".to_string(), shape(&["n", "d"]))]);
    let output = apply_known_function(&KnownFunction::LaxAssociativeScan, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["n", "d"])));
}

#[test]
fn test_lax_scatter_shape_preserving_on_operand() {
    let args = vec![pos("operand"), pos("indices"), pos("updates")];
    let shapes = HashMap::from([("operand".to_string(), shape(&["b", "d"]))]);
    let output = apply_known_function(&KnownFunction::LaxScatter, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b", "d"])));
}

#[test]
fn test_lax_broadcast_prepends_sizes() {
    let args = vec![pos("x"), pos("(4, 5)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["d"]))]);
    let output = apply_known_function(&KnownFunction::LaxBroadcast, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "5", "d"])));
}

#[test]
fn test_lax_broadcast_in_dim_uses_target_shape() {
    let args = vec![pos("x"), pos("(2, 3, 4)"), pos("(1,)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["3"]))]);
    let output = apply_known_function(&KnownFunction::LaxBroadcastInDim, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["2", "3", "4"])));
}

#[test]
fn test_lax_slice_concrete_indices() {
    let args = vec![pos("x"), pos("(0, 1)"), pos("(4, 5)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "8"]))]);
    let output = apply_known_function(&KnownFunction::LaxSlice, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "4"])));
}

#[test]
fn test_lax_slice_with_strides() {
    let args = vec![pos("x"), pos("(0,)"), pos("(10,)"), pos("(2,)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["10"]))]);
    let output = apply_known_function(&KnownFunction::LaxSlice, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["5"])));
}

#[test]
fn test_lax_dynamic_slice_uses_slice_sizes() {
    let args = vec![pos("x"), pos("(i, j)"), pos("(2, 3)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "8"]))]);
    let output = apply_known_function(&KnownFunction::LaxDynamicSlice, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["2", "3"])));
}

#[test]
fn test_lax_dynamic_update_slice_shape_preserving() {
    let args = vec![pos("x"), pos("update"), pos("(0, 0)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "8"]))]);
    let output =
        apply_known_function(&KnownFunction::LaxDynamicUpdateSlice, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["8", "8"])));
}

#[test]
fn test_lax_pad_low_high_interior() {
    let args = vec![pos("x"), pos("0.0"), pos("((1, 1, 0), (0, 0, 1))")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let output = apply_known_function(&KnownFunction::LaxPad, &args, &shapes).unwrap();
    // dim0: 4 + 1 + 1 = 6; dim1: 3 + 1*(3-1) = 5
    assert_eq!(output, Some(shape(&["6", "5"])));
}

#[test]
fn test_lax_reduce_window_valid_padding_pooling_formula() {
    let args = vec![
        pos("x"),
        pos("-inf"),
        pos("max_fn"),
        pos("(1, 2, 2, 1)"),
        pos("(1, 2, 2, 1)"),
        pos("'VALID'"),
    ];
    let shapes = HashMap::from([("x".to_string(), shape(&["1", "8", "8", "3"]))]);
    let output = apply_known_function(&KnownFunction::LaxReduceWindow, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["1", "4", "4", "3"])));
}

#[test]
fn test_lax_conv_general_dilated_default_dimension_numbers_valid() {
    let args = vec![pos("x"), pos("w"), pos("(1, 1)"), pos("'VALID'")];
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["1", "3", "8", "8"])),
        ("w".to_string(), shape(&["16", "3", "3", "3"])),
    ]);
    let output =
        apply_known_function(&KnownFunction::LaxConvGeneralDilated, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["1", "16", "6", "6"])));
}

#[test]
fn test_lax_conv_general_dilated_explicit_dimension_numbers_skips() {
    let args = vec![
        pos("x"),
        pos("w"),
        pos("(1, 1)"),
        pos("'VALID'"),
        kw("dimension_numbers", "('NHWC', 'HWIO', 'NHWC')"),
    ];
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["1", "8", "8", "3"])),
        ("w".to_string(), shape(&["3", "3", "3", "16"])),
    ]);
    let output =
        apply_known_function(&KnownFunction::LaxConvGeneralDilated, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

#[test]
fn test_lax_gather_row_selection() {
    // operand (2, 2), start_indices (2, 1): pick one row per index,
    // keeping the row's 2 columns. offset_dims=(1,) (output axis 1 is
    // the kept slice axis), collapsed_slice_dims=(0,) (the picked row
    // axis is dropped), slice_sizes=(1, 2).
    let args = vec![
        pos("operand"),
        pos("indices"),
        pos(
            "jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))",
        ),
        pos("(1, 2)"),
    ];
    let shapes = HashMap::from([
        ("operand".to_string(), shape(&["2", "2"])),
        ("indices".to_string(), shape(&["2", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::LaxGather, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["2", "2"])));
}

#[test]
fn test_lax_gather_symbolic_batch_dim() {
    // Batch dim comes from start_indices' leading (non-index-vector)
    // axis and can be symbolic; the offset (kept slice) dim is literal.
    let args = vec![
        pos("operand"),
        pos("indices"),
        pos(
            "jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))",
        ),
        pos("(1, 2)"),
    ];
    let shapes = HashMap::from([
        ("operand".to_string(), shape(&["n", "2"])),
        ("indices".to_string(), shape(&["batch", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::LaxGather, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["batch", "2"])));
}

#[test]
fn test_lax_gather_keyword_dimension_numbers_and_slice_sizes() {
    let args = vec![
        pos("operand"),
        pos("indices"),
        kw(
            "dimension_numbers",
            "jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))",
        ),
        kw("slice_sizes", "(1, 2)"),
    ];
    let shapes = HashMap::from([
        ("operand".to_string(), shape(&["2", "2"])),
        ("indices".to_string(), shape(&["2", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::LaxGather, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["2", "2"])));
}

#[test]
fn test_lax_gather_non_literal_dimension_numbers_skips() {
    let args = vec![pos("operand"), pos("indices"), pos("dnums"), pos("(1, 2)")];
    let shapes = HashMap::from([
        ("operand".to_string(), shape(&["2", "2"])),
        ("indices".to_string(), shape(&["2", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::LaxGather, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

#[test]
fn test_lax_gather_non_literal_slice_sizes_skips() {
    let args = vec![
        pos("operand"),
        pos("indices"),
        pos(
            "jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))",
        ),
        pos("sizes"),
    ];
    let shapes = HashMap::from([
        ("operand".to_string(), shape(&["2", "2"])),
        ("indices".to_string(), shape(&["2", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::LaxGather, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

// ── jax.numpy / numpy array-creation shape rules ──

#[test]
fn test_diagflat_flattens_then_squares() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);
    let output = apply_known_function(&KnownFunction::Diagflat, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["6", "6"])));
}

#[test]
fn test_tri_default_square() {
    let args = vec![pos("4")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Tri, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "4"])));
}

#[test]
fn test_tri_rectangular() {
    let args = vec![pos("4"), pos("6")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Tri, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "6"])));
}

#[test]
fn test_indices_prepends_rank() {
    let args = vec![pos("(2, 3)")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Indices, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["2", "2", "3"])));
}

#[test]
fn test_select_approximates_first_choice_shape() {
    let args = vec![pos("[c1, c2]"), pos("[a, b]")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["n"])),
        ("b".to_string(), shape(&["n"])),
    ]);
    let output = apply_known_function(&KnownFunction::Select, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["n"])));
}

// ── jax.numpy / numpy shape-transform shape rules ──

#[test]
fn test_rollaxis_moves_axis_before_start() {
    let args = vec![pos("x"), pos("2"), pos("0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);
    let output = apply_known_function(&KnownFunction::RollAxis, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["c", "a", "b"])));
}

#[test]
fn test_resize_uses_target_shape() {
    let args = vec![pos("x"), pos("(3, 3)")];
    let shapes = HashMap::from([("x".to_string(), shape(&["2"]))]);
    let output = apply_known_function(&KnownFunction::Resize, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["3", "3"])));
}

#[test]
fn test_insert_scalar_grows_axis_by_one() {
    let args = vec![pos("x"), pos("1"), pos("9"), kw("axis", "0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4"]))]);
    let output = apply_known_function(&KnownFunction::Insert, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["5"])));
}

#[test]
fn test_insert_list_grows_axis_by_len() {
    let args = vec![pos("x"), pos("1"), pos("[9, 10, 11]"), kw("axis", "0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4"]))]);
    let output = apply_known_function(&KnownFunction::Insert, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["7"])));
}

#[test]
fn test_insert_axis_none_skips() {
    let args = vec![pos("x"), pos("1"), pos("9")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4"]))]);
    let output = apply_known_function(&KnownFunction::Insert, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

#[test]
fn test_delete_single_index_shrinks_axis_by_one() {
    let args = vec![pos("x"), pos("2"), kw("axis", "0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["5"]))]);
    let output = apply_known_function(&KnownFunction::Delete, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4"])));
}

#[test]
fn test_delete_list_shrinks_axis_by_len() {
    let args = vec![pos("x"), pos("[1, 2]"), kw("axis", "0")];
    let shapes = HashMap::from([("x".to_string(), shape(&["5"]))]);
    let output = apply_known_function(&KnownFunction::Delete, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["3"])));
}

#[test]
fn test_append_with_axis_like_concatenate() {
    let args = vec![pos("a"), pos("b"), kw("axis", "0")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "d"])),
        ("b".to_string(), shape(&["3", "d"])),
    ]);
    let output = apply_known_function(&KnownFunction::Append, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["5", "d"])));
}

#[test]
fn test_append_axis_none_flattens_and_sums() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "3"])),
        ("b".to_string(), shape(&["4"])),
    ]);
    let output = apply_known_function(&KnownFunction::Append, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["10"])));
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
fn test_kron_elementwise_product_of_dims() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "3"])),
        ("b".to_string(), shape(&["4", "5"])),
    ]);
    let output = apply_known_function(&KnownFunction::Kron, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["8", "15"])));
}

#[test]
fn test_block_nested_2x2_assembly() {
    let args = vec![pos("[[a, b], [c, d]]")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "3"])),
        ("b".to_string(), shape(&["2", "4"])),
        ("c".to_string(), shape(&["5", "3"])),
        ("d".to_string(), shape(&["5", "4"])),
    ]);
    let output = apply_known_function(&KnownFunction::Block, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["7", "7"])));
}

// ── jax.numpy / numpy indexing-and-selection shape rules ──

#[test]
fn test_take_along_axis_matches_indices_shape() {
    let args = vec![pos("a"), pos("idx"), kw("axis", "1")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["4", "8"])),
        ("idx".to_string(), shape(&["4", "1"])),
    ]);
    let output = apply_known_function(&KnownFunction::TakeAlongAxis, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "1"])));
}

#[test]
fn test_put_along_axis_shape_preserving_on_arr() {
    let args = vec![pos("a"), pos("idx"), pos("values"), kw("axis", "1")];
    let shapes = HashMap::from([("a".to_string(), shape(&["4", "8"]))]);
    let output = apply_known_function(&KnownFunction::PutAlongAxis, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["4", "8"])));
}

#[test]
fn test_argwhere_known_rank_unknown_count() {
    let args = vec![pos("mask")];
    let shapes = HashMap::from([("mask".to_string(), shape(&["4", "8"]))]);
    let output = apply_known_function(&KnownFunction::Argwhere, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["nonzero(mask)", "2"])));
}

#[test]
fn test_searchsorted_follows_values_shape() {
    let args = vec![pos("a"), pos("v")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["100"])),
        ("v".to_string(), shape(&["10"])),
    ]);
    let output = apply_known_function(&KnownFunction::SearchSorted, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["10"])));
}

#[test]
fn test_histogram_default_bins() {
    let args = vec![pos("x")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Histogram, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["10"])));
}

#[test]
fn test_histogram_literal_bin_count() {
    let args = vec![pos("x"), pos("20")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Histogram, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["20"])));
}

#[test]
fn test_histogram_literal_edges_list() {
    let args = vec![pos("x"), pos("[0, 1, 2, 3]")];
    let shapes = HashMap::new();
    let output = apply_known_function(&KnownFunction::Histogram, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["3"])));
}

#[test]
fn test_bincount_conservatively_unknown() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["100"]))]);
    let output = apply_known_function(&KnownFunction::BinCount, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

#[test]
fn test_unique_conservatively_unknown() {
    let args = vec![pos("x")];
    let shapes = HashMap::from([("x".to_string(), shape(&["100"]))]);
    let output = apply_known_function(&KnownFunction::Unique, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

// ── linear algebra shape rules ──

#[test]
fn test_cross_broadcasts_batch_dims() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["n", "3"])),
        ("b".to_string(), shape(&["3"])),
    ]);
    let output = apply_known_function(&KnownFunction::Cross, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["n", "3"])));
}

#[test]
fn test_linalg_solve_follows_rhs_shape() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["n", "n"])),
        ("b".to_string(), shape(&["n", "k"])),
    ]);
    let output = apply_known_function(&KnownFunction::LinalgSolve, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["n", "k"])));
}

#[test]
fn test_linalg_solve_non_square_errors() {
    let args = vec![pos("a"), pos("b")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "n"])),
        ("b".to_string(), shape(&["n", "k"])),
    ]);
    let result = apply_known_function(&KnownFunction::LinalgSolve, &args, &shapes);
    assert!(result.is_err());
}

#[test]
fn test_linalg_pinv_swaps_last_two_dims() {
    let args = vec![pos("a")];
    let shapes = HashMap::from([("a".to_string(), shape(&["m", "n"]))]);
    let output = apply_known_function(&KnownFunction::LinalgPinv, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["n", "m"])));
}

#[test]
fn test_linalg_matrix_rank_drops_last_two_dims() {
    let args = vec![pos("a")];
    let shapes = HashMap::from([("a".to_string(), shape(&["b", "m", "n"]))]);
    let output = apply_known_function(&KnownFunction::LinalgMatrixRank, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b"])));
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

#[test]
fn test_norm_reduction_axis_semantics() {
    let args = vec![pos("x"), kw("axis", "1")];
    let shapes = HashMap::from([("x".to_string(), shape(&["b", "d"]))]);
    // linalg.norm dispatches through KnownFunction::Sum's reduction rule.
    let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b"])));
}

// ── einops shape rules ──

#[test]
fn test_einops_einsum_pattern_is_last_positional() {
    let args = vec![pos("a"), pos("b"), pos("\"i j, j k -> i k\"")];
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["m", "n"])),
        ("b".to_string(), shape(&["n", "p"])),
    ]);
    let output = apply_known_function(&KnownFunction::EinopsEinsum, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["m", "p"])));
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
fn test_einops_parse_shape_always_none() {
    let args = vec![pos("x"), pos("\"h w\"")];
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "5"]))]);
    let output = apply_known_function(&KnownFunction::EinopsParseShape, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

// ── jax.nn shape rules ──

#[test]
fn test_one_hot_appends_num_classes() {
    let args = vec![pos("x"), pos("10")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch"]))]);
    let output = apply_known_function(&KnownFunction::OneHot, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["batch", "10"])));
}

#[test]
fn test_one_hot_symbolic_num_classes() {
    let args = vec![pos("x"), pos("n_classes")];
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "seq"]))]);
    let output = apply_known_function(&KnownFunction::OneHot, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["batch", "seq", "n_classes"])));
}

#[test]
fn test_dot_product_attention_uses_query_and_value_head_dim() {
    let args = vec![pos("q"), pos("k"), pos("v")];
    let shapes = HashMap::from([
        ("q".to_string(), shape(&["b", "sq", "h", "dk"])),
        ("k".to_string(), shape(&["b", "sk", "h", "dk"])),
        ("v".to_string(), shape(&["b", "sk", "h", "dv"])),
    ]);
    let output = apply_known_function(&KnownFunction::DotProductAttention, &args, &shapes).unwrap();
    assert_eq!(output, Some(shape(&["b", "sq", "h", "dv"])));
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
