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
    assert_eq!(classify_method_call("backward"), None);
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
fn test_classify_expand_repeat_tile() {
    assert_eq!(
        classify_method_call("expand"),
        Some(KnownFunction::BroadcastTo)
    );
    assert_eq!(classify_method_call("repeat"), Some(KnownFunction::Repeat));
    assert_eq!(
        classify_method_call("repeat_interleave"),
        Some(KnownFunction::Repeat)
    );
    assert_eq!(classify_method_call("tile"), Some(KnownFunction::Tile));
}

#[test]
fn test_apply_torch_repeat_multi_positional_tiles() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);
    let args = vec![pos("4"), pos("5")];

    let output = apply_method_call(&KnownFunction::Repeat, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "15"])));
}

#[test]
fn test_apply_numpy_repeat_single_arg_with_axis() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);
    let args = vec![pos("4"), kw("axis", "0")];

    let output = apply_method_call(&KnownFunction::Repeat, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "3"])));
}

#[test]
fn test_apply_expand_with_minus_one_keeps_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["1", "d"]))]);
    let args = vec![pos("batch"), pos("-1")];

    let output = apply_method_call(&KnownFunction::BroadcastTo, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "d"])));
}

#[test]
fn test_apply_expand_prepends_leading_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["d"]))]);
    let args = vec![pos("batch"), pos("-1")];

    let output = apply_method_call(&KnownFunction::BroadcastTo, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "d"])));
}

#[test]
fn test_apply_flatten_no_args() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3", "4"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Flatten, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["24"])));
}

fn torch_flatten(args: Vec<CallArgument>, dims: &[&str]) -> Result<Option<Vec<String>>, String> {
    let shapes = HashMap::from([("x".to_string(), shape(dims))]);
    apply_known_function(&KnownFunction::TorchFlatten, &args, &shapes)
}

#[test]
fn test_torch_flatten_defaults_collapse_all() {
    let out = torch_flatten(vec![pos("x")], &["2", "3", "4"]).unwrap();
    assert_eq!(out, Some(shape(&["24"])));
}

#[test]
fn test_torch_flatten_start_dim_positional_and_keyword() {
    let out = torch_flatten(vec![pos("x"), pos("1")], &["B", "C", "H"]).unwrap();
    assert_eq!(out, Some(shape(&["B", "C*H"])));
    let out = torch_flatten(vec![pos("x"), kw("start_dim", "1")], &["B", "C", "H"]).unwrap();
    assert_eq!(out, Some(shape(&["B", "C*H"])));
}

#[test]
fn test_torch_flatten_start_and_end_dim() {
    let out = torch_flatten(vec![pos("x"), pos("1"), pos("2")], &["2", "3", "4", "5"]).unwrap();
    assert_eq!(out, Some(shape(&["2", "12", "5"])));
    let out = torch_flatten(vec![pos("x"), kw("end_dim", "1")], &["2", "3", "4"]).unwrap();
    assert_eq!(out, Some(shape(&["6", "4"])));
}

#[test]
fn test_torch_flatten_negative_dims() {
    let out = torch_flatten(vec![pos("x"), pos("-2"), pos("-1")], &["2", "3", "4"]).unwrap();
    assert_eq!(out, Some(shape(&["2", "12"])));
    let out = torch_flatten(vec![pos("x"), pos("0"), pos("-2")], &["2", "3", "4"]).unwrap();
    assert_eq!(out, Some(shape(&["6", "4"])));
}

#[test]
fn test_torch_flatten_single_dim_range_keeps_shape() {
    let out = torch_flatten(vec![pos("x"), pos("1"), pos("1")], &["2", "3", "4"]).unwrap();
    assert_eq!(out, Some(shape(&["2", "3", "4"])));
}

#[test]
fn test_torch_flatten_invalid_dims_are_unknown() {
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("3")], &["2", "3", "4"]),
        Ok(None)
    );
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("-4")], &["2", "3", "4"]),
        Ok(None)
    );
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("0"), pos("5")], &["2", "3", "4"]),
        Ok(None)
    );
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("2"), pos("1")], &["2", "3", "4"]),
        Ok(None)
    );
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("d")], &["2", "3", "4"]),
        Ok(None)
    );
}

#[test]
fn test_torch_flatten_scalar() {
    assert_eq!(torch_flatten(vec![pos("x")], &[]), Ok(Some(shape(&["1"]))));
    assert_eq!(
        torch_flatten(vec![pos("x"), pos("0"), pos("-1")], &[]),
        Ok(Some(shape(&["1"])))
    );
    assert_eq!(torch_flatten(vec![pos("x"), pos("1")], &[]), Ok(None));
}

#[test]
fn test_method_flatten_start_dim_is_torch() {
    let shapes = HashMap::from([("x".to_string(), shape(&["B", "C", "H"]))]);
    let out = apply_method_call(&KnownFunction::Flatten, "x", &[pos("1")], &shapes).unwrap();
    assert_eq!(out, Some(shape(&["B", "C*H"])));
    let out = apply_method_call(
        &KnownFunction::Flatten,
        "x",
        &[pos("0"), pos("-2")],
        &shapes,
    )
    .unwrap();
    assert_eq!(out, Some(shape(&["B*C", "H"])));
    let out = apply_method_call(
        &KnownFunction::Flatten,
        "x",
        &[kw("start_dim", "1")],
        &shapes,
    )
    .unwrap();
    assert_eq!(out, Some(shape(&["B", "C*H"])));
    let out = apply_method_call(&KnownFunction::Flatten, "x", &[pos("d")], &shapes).unwrap();
    assert_eq!(out, None);
}

#[test]
fn test_method_flatten_order_is_not_a_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3", "4"]))]);
    for args in [
        vec![pos("'F'")],
        vec![pos("\"C\"")],
        vec![kw("order", "'F'")],
    ] {
        let out = apply_method_call(&KnownFunction::Flatten, "x", &args, &shapes).unwrap();
        assert_eq!(out, Some(shape(&["24"])));
    }
}

#[test]
fn test_ravel_order_is_not_a_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);
    let args = [pos("x"), kw("order", "'F'")];
    let out = apply_known_function(&KnownFunction::Ravel, &args, &shapes).unwrap();
    assert_eq!(out, Some(shape(&["6"])));
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

// ── split shape rule tests ──────────────────────────────────────────

#[test]
fn test_split_int_numeric_axis() {
    // split [6, 4] into 3 along axis 0 → each output [2, 4]
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("axis", "0")];

    let result = compute_split_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["2", "4"]),
            shape(&["2", "4"]),
            shape(&["2", "4"]),
        ])
    );
}

#[test]
fn test_split_indices_list() {
    // numpy semantics: split [10] at indices [2, 5] along axis 0
    // → sections [0..2], [2..5], [5..10] → sizes 2, 3, 5
    let shapes = HashMap::from([("x".to_string(), shape(&["10"]))]);
    let args = vec![pos("x"), pos("[2, 5]"), kw("axis", "0")];

    let result = compute_split_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![shape(&["2"]), shape(&["3"]), shape(&["5"]),])
    );
}

#[test]
fn test_split_symbolic_axis() {
    // split [batch, n] into 2 along axis 1 → each output [batch, "split(n, 2)"]
    //
    // Synthetic naming convention: when the axis dimension is symbolic,
    // the chunk size is emitted as "split(<axis_dim>, <N>)".
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "n"]))]);
    let args = vec![pos("x"), pos("2"), kw("axis", "1")];

    let result = compute_split_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["batch", "split(n, 2)"]),
            shape(&["batch", "split(n, 2)"]),
        ])
    );
}

#[test]
fn test_cancel_product_factor() {
    assert_eq!(
        cancel_product_factor("d_model * 3", 3).as_deref(),
        Some("d_model")
    );
    assert_eq!(
        cancel_product_factor("3 * d_model", 3).as_deref(),
        Some("d_model")
    );
    assert_eq!(
        cancel_product_factor("a * 3 * b", 3).as_deref(),
        Some("a * b")
    );
    // No matching literal factor.
    assert_eq!(cancel_product_factor("d_model * 6", 3), None);
    assert_eq!(cancel_product_factor("d_model", 3), None);
    // Not a flat product.
    assert_eq!(cancel_product_factor("d_model + 3", 3), None);
    assert_eq!(cancel_product_factor("(d_model) * 3", 3), None);
}

#[test]
fn test_split_symbolic_product_cancels() {
    // split [seq, "d * 3"] into 3 → each output [seq, "d"] (fused-QKV pattern).
    let shapes = HashMap::from([("x".to_string(), shape(&["seq", "d * 3"]))]);
    let args = vec![pos("x"), pos("3"), kw("axis", "1")];

    let result = compute_split_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["seq", "d"]),
            shape(&["seq", "d"]),
            shape(&["seq", "d"]),
        ])
    );
}

#[test]
fn test_split_non_divisible_numeric_errors() {
    // split [7] into 2 → Err (7 % 2 != 0)
    let shapes = HashMap::from([("x".to_string(), shape(&["7"]))]);
    let args = vec![pos("x"), pos("2"), kw("axis", "0")];

    let result = compute_split_shapes(&args, &shapes);

    assert!(result.is_err());
}

#[test]
fn test_split_index_exceeds_axis_size_clips() {
    let shapes = HashMap::from([("x".to_string(), shape(&["5"]))]);
    let args = vec![pos("x"), pos("[2, 7]"), kw("axis", "0")];
    assert_eq!(
        compute_split_shapes(&args, &shapes).unwrap(),
        Some(vec![shape(&["2"]), shape(&["3"]), shape(&["0"])])
    );
}

#[test]
fn test_split_indices_use_independent_slices() {
    let shapes = HashMap::from([("x".to_string(), shape(&["5"]))]);
    let args = [pos("x"), pos("[4, 2, -1, -50, 50]")];
    assert_eq!(
        compute_split_shapes(&args, &shapes).unwrap(),
        Some(vec![
            shape(&["4"]),
            shape(&["0"]),
            shape(&["2"]),
            shape(&["0"]),
            shape(&["5"]),
            shape(&["0"])
        ])
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("indices")], &shapes).unwrap(),
        None
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[]")], &shapes).unwrap(),
        Some(vec![shape(&["5"])])
    );
}

#[test]
fn test_split_symbolic_bounds_use_in_bounds_approximation() {
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["n"])),
        ("fixed".to_string(), shape(&["5"])),
    ]);
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[-1]")], &shapes).unwrap(),
        None
    );
    assert_eq!(
        compute_split_shapes(&[pos("fixed"), pos("[point]")], &shapes).unwrap(),
        None
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[point]")], &shapes).unwrap(),
        Some(vec![shape(&["point"]), shape(&["n-point"])])
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[first, first+width]")], &shapes).unwrap(),
        Some(vec![
            shape(&["first"]),
            shape(&["first+width-first"]),
            shape(&["n-(first+width)"])
        ])
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[2, 5]")], &shapes).unwrap(),
        Some(vec![shape(&["2"]), shape(&["3"]), shape(&["n-5"])])
    );
    assert_eq!(
        compute_split_shapes(&[pos("x"), pos("[]")], &shapes).unwrap(),
        Some(vec![shape(&["n"])])
    );
}

#[test]
fn test_split_and_array_split_counts_and_axes() {
    for (size, sections, expected) in [
        ("7", "3", &["3", "2", "2"][..]),
        ("2", "4", &["1", "1", "0", "0"][..]),
    ] {
        let shapes = HashMap::from([("x".to_string(), shape(&[size]))]);
        let args = [pos("x"), pos(sections)];
        assert!(compute_split_shapes(&args, &shapes)
            .unwrap_err()
            .contains("evenly"));
        assert_eq!(
            compute_array_split_shapes(&args, &shapes).unwrap(),
            Some(expected.iter().map(|dim| shape(&[dim])).collect())
        );
        assert_eq!(
            apply_known_function(&KnownFunction::ArraySplit, &args, &shapes).unwrap(),
            None
        );
    }
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let default = [pos("x"), pos("2")];
    let along_second = [pos("x"), pos("2"), pos("1")];
    assert_eq!(
        compute_split_shapes(&default, &shapes).unwrap(),
        Some(vec![shape(&["3", "4"]); 2])
    );
    assert_eq!(
        compute_split_shapes(&along_second, &shapes).unwrap(),
        Some(vec![shape(&["6", "2"]); 2])
    );
    assert_eq!(
        compute_array_split_shapes(&[pos("x"), pos("3"), kw("axis", "1")], &shapes).unwrap(),
        Some(vec![
            shape(&["6", "2"]),
            shape(&["6", "1"]),
            shape(&["6", "1"])
        ])
    );
    assert_eq!(
        compute_array_split_shapes(&[pos("x"), pos("2"), kw("dim", "1")], &shapes).unwrap(),
        Some(vec![shape(&["6", "2"]); 2])
    );
    let dynamic_axis = [pos("x"), pos("2"), kw("axis", "dynamic")];
    assert_eq!(compute_split_shapes(&dynamic_axis, &shapes).unwrap(), None);
    assert_eq!(
        compute_array_split_shapes(&dynamic_axis, &shapes).unwrap(),
        None
    );
    assert!(
        apply_known_function(&KnownFunction::ArraySplit, &[pos("x"), pos("0")], &shapes).is_err()
    );
    let symbolic = HashMap::from([("x".to_string(), shape(&["n"]))]);
    assert_eq!(
        compute_array_split_shapes(&[pos("x"), pos("1")], &symbolic).unwrap(),
        Some(vec![shape(&["n"])])
    );
    assert_eq!(
        compute_array_split_shapes(&[pos("x"), pos("2")], &symbolic).unwrap(),
        None
    );
}

/// apply_known_function dispatch returns Ok(None) for Split
/// because the current return type cannot express a tuple of shapes.
/// This is a documented blocker — once tuple returns are wired through,
/// the dispatch should delegate to compute_split_shapes directly.
#[test]
fn test_split_dispatch_returns_none_blocker() {
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("axis", "0")];

    let output = apply_known_function(&KnownFunction::Split, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

/// Validation errors from compute_split_shapes must propagate through
/// apply_known_function dispatch. This is the only reason the PR delivers
/// user value before tuple-returns land.
#[test]
fn test_split_dispatch_surfaces_validation_errors() {
    // [7] split into 2 → non-divisible → Err
    let shapes = HashMap::from([("x".to_string(), shape(&["7"]))]);
    let args = vec![pos("x"), pos("2"), kw("axis", "0")];

    let result = apply_known_function(&KnownFunction::Split, &args, &shapes);
    assert!(result.is_err());
}

// ── real torch `split` semantics (KnownFunction::TorchSplit) ───────────
// Unlike `Split` (jnp/np/tensor_split: 2nd arg is a section *count*),
// torch's `split_size_or_sections` is a chunk *size* — count is derived
// as `ceil(axis_size / size)`, with a smaller remainder chunk when it
// doesn't divide evenly.

#[test]
fn test_torch_split_size_divides_evenly() {
    // [6, 4] split_size=3, dim=0 → ceil(6/3)=2 chunks of size 3.
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(result, Some(vec![shape(&["3", "4"]), shape(&["3", "4"])]));
}

#[test]
fn test_torch_split_size_leaves_remainder_chunk() {
    // [10] split_size=3 → ceil(10/3)=4 chunks: 3, 3, 3, 1.
    let shapes = HashMap::from([("x".to_string(), shape(&["10"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["3"]),
            shape(&["3"]),
            shape(&["3"]),
            shape(&["1"]),
        ])
    );
}

#[test]
fn test_torch_split_size_larger_than_axis_yields_one_chunk() {
    // [4] split_size=10 → ceil(4/10)=1 chunk of size 4 (the whole axis).
    let shapes = HashMap::from([("x".to_string(), shape(&["4"]))]);
    let args = vec![pos("x"), pos("10"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(result, Some(vec![shape(&["4"])]));
}

#[test]
fn test_torch_split_symbolic_axis_literal_size_unknown_without_arity() {
    // Symbolic axis dim + literal size, no LHS arity hint → count is
    // genuinely unknown.
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(result, None);
}

#[test]
fn test_torch_split_symbolic_axis_literal_size_uses_lhs_arity() {
    // Symbolic axis dim + literal size, LHS arity k=3 known → first k-1
    // chunks get the literal size, last gets the symbolic remainder.
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, Some(3)).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["3", "4"]),
            shape(&["3", "4"]),
            shape(&["n-6", "4"]),
        ])
    );
}

#[test]
fn test_torch_split_symbolic_axis_literal_size_single_target_arity() {
    // k=1: the single chunk is just the whole (symbolic) axis, no
    // subtraction needed.
    let shapes = HashMap::from([("x".to_string(), shape(&["n"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, Some(1)).unwrap();

    assert_eq!(result, Some(vec![shape(&["n"])]));
}

#[test]
fn test_torch_split_non_literal_size_always_unknown() {
    // Non-literal, non-list split spec (a variable) is out of scope
    // regardless of arity — the per-chunk size itself is unknown.
    let shapes = HashMap::from([("x".to_string(), shape(&["n"]))]);
    let args = vec![pos("x"), pos("k"), kw("dim", "0")];

    assert_eq!(
        compute_torch_split_shapes(&args, &shapes, None).unwrap(),
        None
    );
    assert_eq!(
        compute_torch_split_shapes(&args, &shapes, Some(3)).unwrap(),
        None
    );
}

#[test]
fn test_torch_split_list_of_sizes() {
    // [6] split into explicit sizes [2, 3, 1].
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let args = vec![pos("x"), pos("[2, 3, 1]"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(
        result,
        Some(vec![shape(&["2"]), shape(&["3"]), shape(&["1"])])
    );
}

#[test]
fn test_torch_split_list_of_sizes_infers_negative_one() {
    // [6] split into [2, -1] → second chunk infers the remainder (4).
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let args = vec![pos("x"), pos("[2, -1]"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None).unwrap();

    assert_eq!(result, Some(vec![shape(&["2"]), shape(&["4"])]));
}

#[test]
fn test_torch_split_list_of_sizes_mismatched_sum_errors() {
    // [6] split into [2, 5] → sums to 7, not 6 → Err.
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let args = vec![pos("x"), pos("[2, 5]"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None);
    assert!(result.is_err());
}

#[test]
fn test_torch_split_size_zero_errors() {
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let args = vec![pos("x"), pos("0"), kw("dim", "0")];

    let result = compute_torch_split_shapes(&args, &shapes, None);
    assert!(result.is_err());
}

#[test]
fn test_torch_split_dispatch_returns_none_for_single_lhs() {
    // Same "validate, no single shape to store" reasoning as `Split`.
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let output = apply_known_function(&KnownFunction::TorchSplit, &args, &shapes).unwrap();
    assert_eq!(output, None);
}

#[test]
fn test_torch_split_dispatch_surfaces_validation_errors() {
    let shapes = HashMap::from([("x".to_string(), shape(&["6"]))]);
    let args = vec![pos("x"), pos("0"), kw("dim", "0")];

    let result = apply_known_function(&KnownFunction::TorchSplit, &args, &shapes);
    assert!(result.is_err());
}

// ── shape passthrough method tests (astype/copy/detach/contiguous/to) ──

#[test]
fn test_classify_astype() {
    assert_eq!(classify_method_call("astype"), Some(KnownFunction::Astype));
}

#[test]
fn test_classify_copy() {
    assert_eq!(classify_method_call("copy"), Some(KnownFunction::Copy));
}

#[test]
fn test_classify_detach() {
    assert_eq!(classify_method_call("detach"), Some(KnownFunction::Detach));
}

#[test]
fn test_classify_contiguous() {
    assert_eq!(
        classify_method_call("contiguous"),
        Some(KnownFunction::Contiguous)
    );
}

#[test]
fn test_classify_to() {
    assert_eq!(classify_method_call("to"), Some(KnownFunction::To));
}

#[test]
fn test_apply_astype_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
    let args = vec![pos("jnp.float32")];

    let output = apply_method_call(&KnownFunction::Astype, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_apply_astype_with_dtype_kwarg() {
    let shapes = HashMap::from([("x".to_string(), shape(&["n", "m"]))]);
    let args = vec![kw("dtype", "jnp.float32")];

    let output = apply_method_call(&KnownFunction::Astype, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["n", "m"])));
}

#[test]
fn test_apply_astype_no_args_preserves_shape() {
    // numpy astype with no dtype arg still returns same shape
    let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Astype, "x", &args, &shapes).unwrap();

    // shape-preserving: even without known dtype, shape passes through
    // but first_array_arg needs a positional arg — with no args, it falls
    // back to the receiver via synthesize_method_args
    assert_eq!(output, Some(shape(&["3", "4"])));
}

#[test]
fn test_apply_copy_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Copy, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "features"])));
}

#[test]
fn test_apply_detach_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "seq", "hidden"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Detach, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "seq", "hidden"])));
}

#[test]
fn test_apply_contiguous_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["c", "h", "w"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Contiguous, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["c", "h", "w"])));
}

#[test]
fn test_apply_to_device_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "n"]))]);
    let args = vec![pos("cuda")];

    let output = apply_method_call(&KnownFunction::To, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "n"])));
}

#[test]
fn test_apply_to_dtype_kwarg_preserves_shape() {
    let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);
    let args = vec![kw("dtype", "torch.float32")];

    let output = apply_method_call(&KnownFunction::To, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "3"])));
}

#[test]
fn test_apply_astype_unknown_receiver_returns_none() {
    let shapes: HashMap<String, Vec<String>> = HashMap::new();
    let args = vec![pos("jnp.float32")];

    let output = apply_method_call(&KnownFunction::Astype, "unknown", &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_apply_to_unknown_receiver_returns_none() {
    let shapes: HashMap<String, Vec<String>> = HashMap::new();
    let args = vec![pos("cuda")];

    let output = apply_method_call(&KnownFunction::To, "unknown", &args, &shapes).unwrap();

    assert_eq!(output, None);
}

// ── new torch method classify wiring ────────────────────────────────

#[test]
fn test_classify_new_indexing_methods() {
    assert_eq!(classify_method_call("gather"), Some(KnownFunction::Gather));
    assert_eq!(
        classify_method_call("scatter"),
        Some(KnownFunction::Scatter)
    );
    assert_eq!(
        classify_method_call("masked_select"),
        Some(KnownFunction::MaskedSelect)
    );
    assert_eq!(
        classify_method_call("masked_fill"),
        Some(KnownFunction::MaskedFill)
    );
    assert_eq!(
        classify_method_call("index_select"),
        Some(KnownFunction::IndexSelect)
    );
    assert_eq!(classify_method_call("narrow"), Some(KnownFunction::Narrow));
    assert_eq!(
        classify_method_call("select"),
        Some(KnownFunction::SelectDim)
    );
    assert_eq!(classify_method_call("topk"), Some(KnownFunction::TopK));
    assert_eq!(classify_method_call("unfold"), Some(KnownFunction::Unfold));
    assert_eq!(
        classify_method_call("view_as"),
        Some(KnownFunction::ShapeAs)
    );
    assert_eq!(
        classify_method_call("reshape_as"),
        Some(KnownFunction::ShapeAs)
    );
    assert_eq!(
        classify_method_call("expand_as"),
        Some(KnownFunction::ShapeAs)
    );
    assert_eq!(classify_method_call("flip"), Some(KnownFunction::Flip));
    assert_eq!(classify_method_call("roll"), Some(KnownFunction::Roll));
    assert_eq!(classify_method_call("chunk"), Some(KnownFunction::Chunk));
    assert_eq!(classify_method_call("unbind"), Some(KnownFunction::Unbind));
    assert_eq!(
        classify_method_call("split"),
        Some(KnownFunction::TorchSplit)
    );
    assert_eq!(
        classify_method_call("kthvalue"),
        Some(KnownFunction::KthValue)
    );
    assert_eq!(
        classify_method_call("median"),
        Some(KnownFunction::MedianDim)
    );
    assert_eq!(classify_method_call("mode"), Some(KnownFunction::MedianDim));
}

#[test]
fn test_classify_new_misc_methods() {
    assert_eq!(classify_method_call("item"), Some(KnownFunction::Item));
    assert_eq!(
        classify_method_call("new_zeros"),
        Some(KnownFunction::NewSizeConstructor)
    );
    assert_eq!(
        classify_method_call("new_ones"),
        Some(KnownFunction::NewSizeConstructor)
    );
    assert_eq!(
        classify_method_call("new_full"),
        Some(KnownFunction::NewConstructor)
    );
    assert_eq!(
        classify_method_call("new_empty"),
        Some(KnownFunction::NewSizeConstructor)
    );
    assert_eq!(classify_method_call("clone"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("cpu"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("cuda"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("float"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("long"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("int"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("bool"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("double"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("half"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("clamp"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("clip"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("softmax"), Some(KnownFunction::Copy));
    assert_eq!(classify_method_call("norm"), Some(KnownFunction::Sum));
    assert_eq!(
        classify_method_call("diagonal"),
        Some(KnownFunction::Diagonal)
    );
    assert_eq!(classify_method_call("tril"), Some(KnownFunction::Tril));
    assert_eq!(classify_method_call("triu"), Some(KnownFunction::Triu));
}

// ── new apply_method_call shape rules ────────────────────────────────

#[test]
fn test_apply_gather_matches_index_shape() {
    let shapes = HashMap::from([("idx".to_string(), shape(&["4", "3"]))]);
    let args = vec![pos("1"), pos("idx")];

    let output = apply_method_call(&KnownFunction::Gather, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "3"])));
}

#[test]
fn test_apply_scatter_shape_preserving() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let args = vec![pos("1"), pos("idx"), pos("src")];

    let output = apply_method_call(&KnownFunction::Scatter, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "3"])));
}

#[test]
fn test_apply_masked_select_is_conservatively_unknown() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let args = vec![pos("mask")];

    let output = apply_method_call(&KnownFunction::MaskedSelect, "x", &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_apply_masked_fill_shape_preserving() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let args = vec![pos("mask"), pos("0.0")];

    let output = apply_method_call(&KnownFunction::MaskedFill, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "3"])));
}

#[test]
fn test_apply_index_select_dim_length_from_index() {
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["4", "3"])),
        ("idx".to_string(), shape(&["2"])),
    ]);
    let args = vec![kw("dim", "1"), kw("index", "idx")];

    let output = apply_method_call(&KnownFunction::IndexSelect, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "2"])));
}

#[test]
fn test_apply_narrow_replaces_dim_with_length() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("1"), pos("2"), pos("5")];

    let output = apply_method_call(&KnownFunction::Narrow, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "5"])));
}

#[test]
fn test_apply_select_dim_removes_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10", "3"]))]);
    let args = vec![pos("1"), pos("0")];

    let output = apply_method_call(&KnownFunction::SelectDim, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "3"])));
}

#[test]
fn test_apply_unfold_appends_window_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["10"]))]);
    // dimension=0, size=2, step=1 → (10-2)/1+1 = 9 windows of size 2
    let args = vec![pos("0"), pos("2"), pos("1")];

    let output = apply_method_call(&KnownFunction::Unfold, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["9", "2"])));
}

#[test]
fn test_apply_view_as_takes_other_shape() {
    let shapes = HashMap::from([("other".to_string(), shape(&["2", "6"]))]);
    let args = vec![pos("other")];

    let output = apply_method_call(&KnownFunction::ShapeAs, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "6"])));
}

#[test]
fn test_apply_item_is_scalar() {
    let shapes: HashMap<String, Vec<String>> = HashMap::new();
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Item, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(Vec::new()));
}

#[test]
fn test_apply_new_zeros_shape_from_arg() {
    let shapes: HashMap<String, Vec<String>> = HashMap::new();
    let args = vec![pos("(3, 4)")];

    let output = apply_method_call(&KnownFunction::NewConstructor, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["3", "4"])));
}

#[test]
fn test_apply_clone_shape_preserving() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "n"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Copy, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "n"])));
}

#[test]
fn test_apply_norm_no_dim_reduces_to_scalar() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Sum, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(Vec::new()));
}

#[test]
fn test_apply_norm_with_dim_reduces_one_axis() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "3"]))]);
    let args = vec![kw("dim", "1")];

    let output = apply_method_call(&KnownFunction::Sum, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4"])));
}

#[test]
fn test_apply_diagonal_method_form() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "4"]))]);
    let args: Vec<CallArgument> = vec![];

    let output = apply_method_call(&KnownFunction::Diagonal, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4"])));
}

#[test]
fn test_apply_triu_method_form_shape_preserving() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "4"]))]);
    let args = vec![pos("1")];

    let output = apply_method_call(&KnownFunction::Triu, "x", &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "4"])));
}

// ── torch tuple-output helper functions (used from analysis.rs) ─────

#[test]
fn test_apply_known_topk_shape_replaces_dim_with_k() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("x"), pos("3")];

    let output = apply_known_topk_shape(&args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "3"])));
}

#[test]
fn test_apply_known_topk_shape_explicit_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let output = apply_known_topk_shape(&args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["3", "10"])));
}

#[test]
fn test_topk_axis_validation_and_symbolic_k() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let run = |k, axis| apply_known_topk_shape(&[pos("x"), pos(k), kw("axis", axis)], &shapes);
    assert_eq!(run("0", "0").unwrap(), Some(shape(&["0", "10"])));
    assert_eq!(run("4", "0").unwrap(), Some(shape(&["4", "10"])));
    assert!(run("5", "0").is_err());
    assert!(run("-1", "0").is_err());
    assert_eq!(run("2", "dynamic").unwrap(), None);
    assert_eq!(run("k", "0").unwrap(), Some(shape(&["k", "10"])));
    assert_eq!(run("_k2", "0").unwrap(), Some(shape(&["_k2", "10"])));
    assert_eq!(run("k + 1", "0").unwrap(), None);
    assert_eq!(run("jnp.sum(x)", "0").unwrap(), None);
    assert_eq!(
        apply_known_function(&KnownFunction::TopK, &[pos("x"), pos("2")], &shapes).unwrap(),
        None
    );
    assert_eq!(
        apply_known_function(&KnownFunction::MinMax, &[pos("x")], &shapes).unwrap(),
        None
    );
}

#[test]
fn test_apply_known_kthvalue_shape_reduces_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("x"), pos("2"), kw("dim", "1")];

    let output = apply_known_kthvalue_shape(&args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4"])));
}

#[test]
fn test_apply_known_kthvalue_shape_keepdim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("x"), pos("2"), kw("dim", "1"), kw("keepdim", "True")];

    let output = apply_known_kthvalue_shape(&args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["4", "1"])));
}

#[test]
fn test_compute_unbind_shape_removes_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["3", "4", "5"]))]);
    let args = vec![pos("x"), kw("dim", "0")];

    let output = compute_unbind_shape(&args, &shapes, 3).unwrap();

    assert_eq!(output, Some(shape(&["4", "5"])));
}

#[test]
fn test_compute_unbind_shape_mismatched_count_errors() {
    let shapes = HashMap::from([("x".to_string(), shape(&["3", "4", "5"]))]);
    let args = vec![pos("x"), kw("dim", "0")];

    let result = compute_unbind_shape(&args, &shapes, 2);

    assert!(result.is_err());
}

#[test]
fn test_compute_chunk_shapes_evenly_divisible() {
    let shapes = HashMap::from([("x".to_string(), shape(&["6", "4"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_chunk_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![
            shape(&["2", "4"]),
            shape(&["2", "4"]),
            shape(&["2", "4"])
        ])
    );
}

#[test]
fn test_compute_chunk_shapes_uneven_last_chunk_smaller() {
    // 7 split into at most 3 chunks of ceil(7/3)=3 → sizes [3, 3, 1]
    let shapes = HashMap::from([("x".to_string(), shape(&["7"]))]);
    let args = vec![pos("x"), pos("3"), kw("dim", "0")];

    let result = compute_chunk_shapes(&args, &shapes).unwrap();

    assert_eq!(
        result,
        Some(vec![shape(&["3"]), shape(&["3"]), shape(&["1"])])
    );
}

// ── torch combinatorics ──────────────────────────────────────────────

#[test]
fn test_apply_combinations_n_choose_r() {
    let shapes = HashMap::from([("x".to_string(), shape(&["5"]))]);
    let args = vec![pos("x")];

    let output = apply_known_function(&KnownFunction::Combinations, &args, &shapes).unwrap();

    // 5 choose 2 = 10
    assert_eq!(output, Some(shape(&["10", "2"])));
}

#[test]
fn test_apply_cartesian_prod_multiplies_lengths() {
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2"])),
        ("b".to_string(), shape(&["3"])),
    ]);
    let args = vec![pos("a"), pos("b")];

    let output = apply_known_function(&KnownFunction::CartesianProd, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["6", "2"])));
}

#[test]
fn test_apply_block_diag_sums_block_dims() {
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["2", "3"])),
        ("b".to_string(), shape(&["4", "5"])),
    ]);
    let args = vec![pos("a"), pos("b")];

    let output = apply_known_function(&KnownFunction::BlockDiag, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["6", "8"])));
}

// ── torch.nn.functional ──────────────────────────────────────────────

#[test]
fn test_apply_functional_conv2d_output_shape() {
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["8", "3", "32", "32"])),
        ("w".to_string(), shape(&["16", "3", "3", "3"])),
    ]);
    let args = vec![pos("x"), pos("w"), kw("padding", "1")];

    let output = apply_known_function(&KnownFunction::FunctionalConv2d, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "16", "32", "32"])));
}

#[test]
fn test_apply_functional_max_pool2d_default_stride_equals_kernel() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "3", "32", "32"]))]);
    let args = vec![pos("x"), pos("2")];

    let output = apply_known_function(&KnownFunction::FunctionalMaxPool2d, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "3", "16", "16"])));
}

#[test]
fn test_apply_functional_avg_pool1d() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "3", "10"]))]);
    let args = vec![pos("x"), pos("2"), pos("2")];

    let output = apply_known_function(&KnownFunction::FunctionalAvgPool1d, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "3", "5"])));
}

#[test]
fn test_apply_interpolate_scale_factor() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "3", "16", "16"]))]);
    let args = vec![pos("x"), kw("scale_factor", "2")];

    let output = apply_known_function(&KnownFunction::Interpolate, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "3", "32", "32"])));
}

#[test]
fn test_apply_interpolate_size() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "3", "16", "16"]))]);
    let args = vec![pos("x"), kw("size", "(8, 8)")];

    let output = apply_known_function(&KnownFunction::Interpolate, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "3", "8", "8"])));
}

#[test]
fn test_apply_functional_embedding_appends_embed_dim() {
    let shapes = HashMap::from([
        ("x".to_string(), shape(&["batch", "seq"])),
        ("weight".to_string(), shape(&["1000", "64"])),
    ]);
    let args = vec![pos("x"), pos("weight")];

    let output = apply_known_function(&KnownFunction::FunctionalEmbedding, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "seq", "64"])));
}

#[test]
fn test_apply_functional_glu_numeric_last_axis() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "16"]))]);
    let args = vec![pos("x")];

    let output = apply_known_function(&KnownFunction::FunctionalGlu, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["8", "8"])));
}

#[test]
fn test_apply_functional_glu_explicit_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["4", "10"]))]);
    let args = vec![pos("x"), kw("dim", "0")];

    let output = apply_known_function(&KnownFunction::FunctionalGlu, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "10"])));
}

#[test]
fn test_apply_functional_glu_symbolic_factor_cancels() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "d_model * 2"]))]);
    let args = vec![pos("x")];

    let output = apply_known_function(&KnownFunction::FunctionalGlu, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "d_model"])));
}

#[test]
fn test_apply_functional_glu_opaque_symbolic_dim() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch", "hidden"]))]);
    let args = vec![pos("x")];

    let output = apply_known_function(&KnownFunction::FunctionalGlu, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["batch", "glu(hidden)"])));
}

#[test]
fn test_apply_functional_glu_odd_numeric_dim_errors() {
    let shapes = HashMap::from([("x".to_string(), shape(&["8", "5"]))]);
    let args = vec![pos("x")];

    let err = apply_known_function(&KnownFunction::FunctionalGlu, &args, &shapes).unwrap_err();

    assert!(err.contains("even"));
}

#[test]
fn test_apply_functional_one_hot_num_classes_minus_one_is_unknown() {
    let shapes = HashMap::from([("x".to_string(), shape(&["batch"]))]);
    let args = vec![pos("x"), kw("num_classes", "-1")];

    let output = apply_known_function(&KnownFunction::OneHot, &args, &shapes).unwrap();

    assert_eq!(output, None);
}

#[test]
fn test_apply_pad_sequence_batch_first() {
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["3", "8"])),
        ("b".to_string(), shape(&["5", "8"])),
    ]);
    let args = vec![pos("[a, b]"), kw("batch_first", "True")];

    let output = apply_known_function(&KnownFunction::PadSequence, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["2", "pad_len", "8"])));
}

#[test]
fn test_apply_pad_sequence_default_seq_first() {
    let shapes = HashMap::from([
        ("a".to_string(), shape(&["3", "8"])),
        ("b".to_string(), shape(&["5", "8"])),
    ]);
    let args = vec![pos("[a, b]")];

    let output = apply_known_function(&KnownFunction::PadSequence, &args, &shapes).unwrap();

    assert_eq!(output, Some(shape(&["pad_len", "2", "8"])));
}
