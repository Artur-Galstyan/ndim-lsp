use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Range;

use crate::*;

#[cfg(test)]
fn find_shape<'a>(analysis: &'a LayerShapeAnalysis, var: &str) -> Option<&'a Vec<String>> {
    analysis.scopes.iter().find_map(|s| s.shapes.get(var))
}

#[cfg(test)]
fn has_shape(analysis: &LayerShapeAnalysis, var: &str) -> bool {
    analysis.scopes.iter().any(|s| s.shapes.contains_key(var))
}

#[cfg(test)]
fn shapes_empty(analysis: &LayerShapeAnalysis) -> bool {
    analysis.scopes.iter().all(|s| s.shapes.is_empty())
}

#[cfg(test)]
mod analyze_layer_shapes_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_analyzes_single_layer_shape_success() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "3"])));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_resolution_cache_shared_across_files() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A user-defined symbol on the search root. Unlike equinox.nn.*, it is
        // not in the hardcoded catalog, so resolving it actually walks disk
        // through resolve_call_signature — the path the ResolutionCache guards.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("mymod.py"),
            "class Block:\n    def __init__(self, a, b): pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let read_count = AtomicUsize::new(0);
        let counting_read = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };

        // Two distinct source files (different document URIs in the LSP) that
        // both resolve the same shared symbol `mymod.Block`.
        let a_py = "import mymod\ndef f():\n    layer = mymod.Block(3, 5)";
        let b_py = "from mymod import Block\ndef g():\n    layer = Block(3, 5)";

        let cache = new_resolution_cache();

        let tree_a = parse(a_py);
        analyze_layer_shapes(tree_a.root_node(), a_py, &roots, counting_read, 5, Some(&cache))
            .unwrap();
        let reads_a = read_count.load(Ordering::Relaxed);
        assert!(reads_a > 0, "first file should hit disk");

        read_count.store(0, Ordering::Relaxed);

        let tree_b = parse(b_py);
        analyze_layer_shapes(tree_b.root_node(), b_py, &roots, counting_read, 5, Some(&cache))
            .unwrap();
        let reads_b = read_count.load(Ordering::Relaxed);

        // The second file reuses the cached ResolvedImplementation for the
        // shared symbol, so it reads strictly less than the first (the only
        // residual read is signature extraction, which #38's cache doesn't
        // cover). A regression that built a fresh cache per analyze call would
        // re-read everything and not advance the hit counter.
        assert!(
            reads_b < reads_a,
            "second file should read less than the first (cache reuse): \
             reads_a={reads_a} reads_b={reads_b}"
        );
        assert!(
            cache.hits.load(Ordering::Relaxed) >= 1,
            "expected at least one cache hit after analyzing the second file"
        );
    }

    #[test]
    fn test_analyzes_chained_layer_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    y = l1(x)\n    z = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_reports_layer_shape_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 4\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("expected input last dim 3")
        );
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "4"])));
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_missing_input_shape_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code =
            "import equinox as eqx\ndef f(x):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(shapes_empty(&analysis));
        assert_eq!(analysis.layers.len(), 1);
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_supports_from_import_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "from equinox.nn import Linear as Lin\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = Lin(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
    }

    /// `import a.b` binds `a`, and a splat argument can fill any parameter.
    #[test]
    fn test_dotted_imports_and_splat_constructor_arguments() {
        let cases: &[(&str, &str, Option<&[&str]>)] = &[
            ("import torch.nn\ndef f():\n    x = torch.zeros(3, 8)\n    y = torch.nn.Linear(8, 4)(x)", "y", Some(&["3", "4"])),
            ("import torch.nn\ndef f():\n    y = torch.zeros(3, 8)", "y", Some(&["3", "8"])),
            ("import jax.numpy\ndef f():\n    y = jax.numpy.zeros((3, 4))", "y", Some(&["3", "4"])),
            ("import jax.numpy\ndef f():\n    a = jax.numpy.zeros((3, 4))\n    y = jax.nn.relu(a)", "y", Some(&["3", "4"])),
            ("import jax.numpy\nimport jax.random\ndef f():\n    y = jax.numpy.zeros((3, 4))", "y", Some(&["3", "4"])),
            ("import torch\nimport torch.nn as nn\ndef f(a):\n    x = torch.zeros(3, 8)\n    y = nn.Linear(8, *a)(x)", "y", None),
            ("import torch\nimport torch.nn as nn\ndef f(cfg):\n    y = nn.Flatten(**cfg)(torch.zeros(2, 3, 4))", "y", None),
        ];
        for (code, name, expected) in cases {
            let tree = parse(code);
            let analysis = analyze_layer_shapes(tree.root_node(), code, &[], read, 5, None).unwrap();
            assert_eq!(find_shape(&analysis, name), expected.map(shape).as_ref(), "{code}");
        }
    }

    #[test]
    fn test_non_layer_calls_are_ignored_in_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        fs::write(tmp.path().join("helpers.py"), "def transform(x): pass").unwrap();
        let code = "import equinox as eqx\nimport helpers\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    a = helpers.transform(x)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.layers.len(), 1);
        assert_eq!(analysis.applications.len(), 1);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(!has_shape(&analysis, "a"));
    }

    #[test]
    fn test_missing_layer_implementation_keeps_only_annotation_shapes() {
        // Uses a user-defined module path (not in the built-in catalog) so
        // missing disk implementation still yields an empty layer map.
        let tmp = tempfile::tempdir().unwrap();
        let code = "import my_layers\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = my_layers.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(analysis.layers.is_empty());
        assert!(analysis.applications.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "3"])));
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_analysis_continues_after_one_application_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"], a: Float[Array, \"batch 4\"]):\n    bad_layer = eqx.nn.Linear(3, 5)\n    good_layer = eqx.nn.Linear(4, 6)\n    bad = bad_layer(a)\n    good = good_layer(a)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "bad");
        assert!(analysis.errors[0].message.contains("bad_layer"));
        assert!(!has_shape(&analysis, "bad"));
        assert_eq!(find_shape(&analysis, "good"), Some(&shape(&["batch", "6"])));
    }

    #[test]
    fn test_analysis_collects_multiple_structured_errors() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(analysis.errors[0].variable, "a");
        assert!(analysis.errors[0].message.contains("l1"));
        assert_eq!(analysis.errors[1].variable, "b");
        assert!(analysis.errors[1].message.contains("l2"));
        assert!(!has_shape(&analysis, "a"));
        assert!(!has_shape(&analysis, "b"));
    }

    #[test]
    fn test_analysis_error_order_with_success_between_errors() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    good_layer = eqx.nn.Linear(3, 9)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    good = good_layer(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(analysis.errors[0].variable, "a");
        assert_eq!(analysis.errors[1].variable, "b");
        assert_eq!(find_shape(&analysis, "good"), Some(&shape(&["batch", "9"])));
    }

    #[test]
    fn test_analysis_failed_assignment_evicts_stale_shape() {
        // Issue #46: after a reassignment whose RHS can't be shaped (here, a
        // failing layer application), the old binding must not survive —
        // downstream uses would reason from stale data and emit false
        // diagnostics.
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(y: Float[Array, \"old shape\"], x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert_eq!(find_shape(&analysis, "y"), None);
    }

    #[test]
    fn test_unshapeable_reassignment_evicts_and_suppresses_downstream_errors() {
        // Issue #46 repro shape: an unclassified callable reassigns `h`; the
        // stale conv-output shape must not leak into the later reshape and
        // produce a false size-mismatch error.
        let code = "def f(h: Float[Array, \"16 32 32\"]):\n    h = unknown_pool(h)\n    flat = h.reshape(32)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "h"), None);
        assert_eq!(find_shape(&analysis, "flat"), None);
    }

    #[test]
    fn test_analysis_error_range_covers_failing_call_arguments() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x)");
    }

    #[test]
    fn test_analysis_multiple_error_ranges_cover_each_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(
            &code[analysis.errors[0].range.start_byte..analysis.errors[0].range.end_byte],
            "(x)"
        );
        assert_eq!(
            &code[analysis.errors[1].range.start_byte..analysis.errors[1].range.end_byte],
            "(x)"
        );
        assert_ne!(
            analysis.errors[0].range.start_byte,
            analysis.errors[1].range.start_byte
        );
    }

    #[test]
    fn test_scalar_annotated_input_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "x"));
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_keyword_constructor_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    layer = eqx.nn.Linear(out_features=hidden, in_features=features)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "hidden"]))
        );
    }

    #[test]
    fn test_duplicate_layer_assignment_affects_later_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 7\"]):\n    layer = eqx.nn.Linear(3, 5)\n    layer = eqx.nn.Linear(7, 11)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "11"])));
    }

    #[test]
    fn test_symbolic_dims_propagate_through_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    layer = eqx.nn.Linear(features, hidden)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "hidden"]))
        );
    }

    #[test]
    fn test_application_source_order_is_respected() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    z = l2(y)\n    y = l1(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(!has_shape(&analysis, "z"));
    }

    #[test]
    fn test_missing_input_does_not_block_later_valid_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    missing_out = layer(missing)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "missing_out"));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_output_reassignment_shape_last_wins() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    y = l1(x)\n    y = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_file_with_only_annotations_returns_initial_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"batch features\"], y: Float[Array, \"batch\"]):\n    pass";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(analysis.layers.is_empty());
        assert!(analysis.applications.is_empty());
        assert_eq!(
            find_shape(&analysis, "x"),
            Some(&shape(&["batch", "features"]))
        );
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_same_param_name_in_two_functions_does_not_collide() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"a b\"]):\n    pass\ndef g(x: Float[Array, \"c d\"]):\n    pass";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        let f_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("f"))
            .expect("f scope");
        let g_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("g"))
            .expect("g scope");

        assert_eq!(f_scope.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert_eq!(g_scope.shapes.get("x"), Some(&shape(&["c", "d"])));
    }

    #[test]
    fn test_layer_in_one_function_uses_its_own_x() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l_f = eqx.nn.Linear(3, 5)\n    y = l_f(x)\ndef g(x: Float[Array, \"batch 7\"]):\n    l_g = eqx.nn.Linear(7, 9)\n    y = l_g(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        let f_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("f"))
            .expect("f scope");
        let g_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("g"))
            .expect("g scope");
        assert_eq!(f_scope.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(g_scope.shapes.get("y"), Some(&shape(&["batch", "9"])));
    }

    #[test]
    fn test_layer_name_collision_across_functions_is_scoped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        // Both functions define a local `layer`, but with different
        // in_features. The second function's `layer(x)` must use *its* layer,
        // not the one from the first function.
        let code = "\
import equinox as eqx
def f(x: Float[Array, \"batch 64\"]):
    layer = eqx.nn.Linear(64, 128)
    y = layer(x)
def g(x: Float[Array, \"batch 128\"]):
    layer = eqx.nn.Linear(128, 256)
    y = layer(x)
";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        let f_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("f"))
            .expect("f scope");
        let g_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("g"))
            .expect("g scope");
        assert_eq!(f_scope.shapes.get("y"), Some(&shape(&["batch", "128"])));
        assert_eq!(g_scope.shapes.get("y"), Some(&shape(&["batch", "256"])));
    }

    #[test]
    fn test_layer_name_collision_different_layer_kinds_is_scoped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear\nfrom ._conv import Conv2d",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_conv.py"),
            "class Conv2d:\n    def __init__(self, in_channels, out_channels, kernel_size): pass",
        )
        .unwrap();
        // First function uses `layer` as a Linear, second uses `layer` as a
        // Conv2d. Without scoping the Linear definition leaks into g and
        // causes a spurious last-dim mismatch.
        let code = "\
import equinox as eqx
def f(x: Float[Array, \"batch 64\"]):
    layer = eqx.nn.Linear(64, 128)
    y = layer(x)
def g(x: Float[Array, \"batch 3 32 32\"]):
    layer = eqx.nn.Conv2d(3, 16, 3)
    y = layer(x)
";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        let g_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("g"))
            .expect("g scope");
        assert_eq!(
            g_scope.shapes.get("y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }
}

#[cfg(test)]
mod resolve_call_signature_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir, init_source: &str) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(tmp.path().join("equinox/nn/_linear.py"), init_source).unwrap();
    }

    fn resolve_first_call(source: &str, roots: &[PathBuf]) -> Option<ResolvedCallSignature> {
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let cache = new_resolution_cache();
        let expected =
            resolve_call_signature(&calls[0], source, &import_map, roots, read, 5, Some(&cache));
        let actual = crate::resolution::resolve_call_signature_with_node(
            &calls[0],
            Some(tree.root_node()),
            source,
            &import_map,
            roots,
            read,
            5,
            Some(&cache),
        );
        assert_eq!(actual, expected, "text-only and existing-tree paths differ");
        actual.unwrap()
    }

    #[test]
    fn test_resolves_call_signature_through_reexport() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(3, out_features=5)";
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_first_call(source, &roots).unwrap();

        assert_eq!(found.signature.owner, Some("Linear".to_string()));
        assert_eq!(found.signature.name, "__init__");
        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(found.bindings.get("self"), None);
    }

    #[test]
    fn test_classifies_real_resolved_equinox_linear_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(features, hidden)";
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_first_call(source, &roots).unwrap();

        assert_eq!(
            classify_layer_call(&found),
            Some(LayerKind::Linear {
                in_features: "features".to_string(),
                out_features: "hidden".to_string()
            })
        );
    }

    #[test]
    fn test_resolves_from_imported_linear_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        );

        let source = "from equinox.nn import Linear as Lin\nlayer = Lin(3, 5)";
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_first_call(source, &roots).unwrap();

        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(
            classify_layer_call(&found),
            Some(LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_classmethod_cls_param_is_skipped_for_bindings() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(cls, in_features, out_features): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)";
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_first_call(source, &roots).unwrap();

        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(found.bindings.get("cls"), None);
    }

    #[test]
    fn test_resolves_top_level_function_call_signature() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("jax/numpy")).unwrap();
        fs::write(
            tmp.path().join("jax/numpy/__init__.py"),
            "def concatenate(arrays, axis=0): pass",
        )
        .unwrap();

        let source = "# π😀\nimport jax.numpy as jnp\ndef f():\n    out = jnp.concatenate(\n        xs, axis=1\n    )";
        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_first_call(source, &roots).unwrap();

        assert_eq!(found.signature.owner, None);
        assert_eq!(found.signature.name, "concatenate");
        assert_eq!(found.bindings.get("arrays"), Some(&"xs".to_string()));
        assert_eq!(found.bindings.get("axis"), Some(&"1".to_string()));
    }

    #[test]
    fn test_returns_none_when_implementation_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let source = "import missing\nx = missing.Foo()";
        let roots = vec![tmp.path().to_path_buf()];

        assert_eq!(resolve_first_call(source, &roots), None);
    }
    fn call_site(source: &str) -> (tree_sitter::Tree, HashMap<String, ImportPath>, Vec<CallInfo>) {
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        (tree, import_map, calls)
    }

    #[test]
    fn implementation_file_is_parsed_once_for_many_calls() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp, "class Linear:\n    def __init__(self, in_features, out_features): pass\n");
        let roots = vec![tmp.path().to_path_buf()];
        let source = "from equinox.nn import Linear\na = Linear(1, 2)\nb = Linear(3, 4)\nc = Linear(5, 6)\n";
        let (tree, import_map, calls) = call_site(source);
        let cache = new_resolution_cache();
        for call in &calls {
            let resolved = crate::resolution::resolve_call_signature_with_node(
                call, Some(tree.root_node()), source, &import_map, &roots, read, 5, Some(&cache),
            )
            .unwrap();
            assert!(resolved.is_some());
        }
        assert_eq!(cache.parses.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn modified_implementation_file_is_parsed_again() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp, "class Linear:\n    def __init__(self, in_features, out_features): pass\n");
        let roots = vec![tmp.path().to_path_buf()];
        let source = "from equinox.nn import Linear\na = Linear(1, 2)\n";
        let (tree, import_map, calls) = call_site(source);
        let cache = new_resolution_cache();
        let resolve = || {
            cache.revalidate(&[]);
            crate::resolution::resolve_call_signature_with_node(
                &calls[0], Some(tree.root_node()), source, &import_map, &roots, read, 5, Some(&cache),
            )
            .unwrap()
            .unwrap()
        };
        assert_eq!(resolve().signature.params, ["self", "in_features", "out_features"]);
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, a, b, c): pass\n",
        )
        .unwrap();
        assert_eq!(resolve().signature.params, ["self", "a", "b", "c"]);
    }

    #[test]
    fn edit_with_an_editor_event_is_not_stale() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp, "class Linear:\n    def __init__(self, ab): pass\n");
        let roots = vec![tmp.path().to_path_buf()];
        let source = "from equinox.nn import Linear\na = Linear(1)\n";
        let (tree, import_map, calls) = call_site(source);
        let cache = new_resolution_cache();
        let resolve = || {
            crate::resolution::resolve_call_signature_with_node(
                &calls[0], Some(tree.root_node()), source, &import_map, &roots, read, 5, Some(&cache),
            )
            .unwrap()
            .unwrap()
        };
        assert_eq!(resolve().signature.params, ["self", "ab"]);
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, cd): pass\n",
        )
        .unwrap();
        // Same size, maybe the same modification time: only the event tells.
        cache.revalidate(&[tmp.path().join("equinox/nn/_linear.py")]);
        assert_eq!(resolve().signature.params, ["self", "cd"]);
    }

}

#[cfg(test)]
mod resolve_target_on_disk_tests {
    use super::*;
    use std::fs;

    fn target(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    fn expected(
        dots: usize,
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
    ) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots,
            module_parts: module_parts.iter().map(|part| part.to_string()).collect(),
            file_path,
            symbol_parts: symbol_parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    #[test]
    fn test_resolves_exact_module_file_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_exact_package_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz/__init__.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_longest_module_prefix_and_keeps_symbol() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_falls_back_to_shorter_module_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                tmp.path().join("foo.py"),
                &["bar", "Baz"]
            ))
        );
    }

    #[test]
    fn test_prefers_longer_module_over_shorter_module() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_package_init_wins_over_module_file_for_same_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar/__init__.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_searches_roots_in_order_for_same_prefix() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("foo.py"), "").unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(0, &["foo"], first.path().join("foo.py"), &["Bar"]))
        );
    }

    #[test]
    fn test_searches_later_roots_if_missing_in_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                second.path().join("foo.py"),
                &["Bar"]
            ))
        );
    }

    #[test]
    fn test_empty_target_parts_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &[]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_search_roots_returns_none() {
        let found = resolve_target_on_disk(&target(0, &["foo"]), &[]);

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar"]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_relative_target_returns_none_until_base_package_is_known() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("layers")).unwrap();
        fs::write(tmp.path().join("layers.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(1, &["layers", "Linear"]), &roots);

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn rt(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: self::parts(parts),
        }
    }

    fn sig(owner: Option<&str>, params: &[&str]) -> PythonCallableSignature {
        PythonCallableSignature {
            owner: owner.map(|owner| owner.to_string()),
            name: "f".to_string(),
            params: parts(params),
        }
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

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear {
            in_features: in_features.to_string(),
            out_features: out_features.to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "out".to_string(),
            layer: "layer".to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
            extra_inputs: Vec::new(),
        }
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        parts(dims)
    }

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    macro_rules! resolve_call_target_case {
        ($name:ident, [$(($alias:expr, $path:expr)),*], $target:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let import_map = HashMap::from([
                    $(($alias.to_string(), $path),)*
                ]);
                assert_eq!(resolve_call_target($target, &import_map), $expected);
            }
        };
    }

    resolve_call_target_case!(
        call_target_alias_with_two_suffixes,
        [("np", ip(0, &["numpy"], "random"))],
        "np.default_rng.seed",
        rt(0, &["numpy", "random", "default_rng", "seed"])
    );
    resolve_call_target_case!(
        call_target_alias_exact_only,
        [("np", ip(0, &["numpy"], "random"))],
        "npx.foo",
        rt(0, &["npx", "foo"])
    );
    resolve_call_target_case!(
        call_target_from_import_with_suffix_chain,
        [("Linear", ip(0, &["equinox", "nn"], "Linear"))],
        "Linear.init.extra",
        rt(0, &["equinox", "nn", "Linear", "init", "extra"])
    );
    resolve_call_target_case!(
        call_target_relative_import_with_suffix_chain,
        [("layers", ip(1, &[], "layers"))],
        "layers.Linear.forward",
        rt(1, &["layers", "Linear", "forward"])
    );
    resolve_call_target_case!(
        call_target_empty_dots_with_import_map_still_empty,
        [("x", ip(0, &["pkg"], "x"))],
        "...",
        rt(0, &[])
    );
    resolve_call_target_case!(
        call_target_leading_dot_ignored_for_unimported,
        [],
        ".foo.bar",
        rt(0, &["foo", "bar"])
    );
    resolve_call_target_case!(
        call_target_trailing_dot_ignored_for_imported,
        [("foo", ip(0, &["pkg"], "foo"))],
        "foo.",
        rt(0, &["pkg", "foo"])
    );
    resolve_call_target_case!(
        call_target_many_empty_segments_ignored,
        [],
        "foo...bar..baz",
        rt(0, &["foo", "bar", "baz"])
    );

    macro_rules! import_path_case {
        ($name:ident, $current:expr, $path:expr, $expected:expr) => {
            #[test]
            fn $name() {
                assert_eq!(
                    resolve_import_path_from_package(&parts($current), &$path),
                    $expected
                );
            }
        };
    }

    import_path_case!(
        import_abs_deep_module,
        &["pkg"],
        ip(0, &["a", "b", "c"], "D"),
        Some(rt(0, &["a", "b", "c", "D"]))
    );
    import_path_case!(
        import_abs_single_name,
        &["pkg", "sub"],
        ip(0, &[], "D"),
        Some(rt(0, &["D"]))
    );
    import_path_case!(
        import_rel_same_deep_module,
        &["pkg", "sub"],
        ip(1, &["a", "b"], "C"),
        Some(rt(0, &["pkg", "sub", "a", "b", "C"]))
    );
    import_path_case!(
        import_rel_parent_no_module,
        &["pkg", "sub"],
        ip(2, &[], "C"),
        Some(rt(0, &["pkg", "C"]))
    );
    import_path_case!(
        import_rel_exact_top_too_far,
        &["pkg"],
        ip(2, &[], "C"),
        None
    );
    import_path_case!(import_rel_empty_current_too_far, &[], ip(1, &[], "C"), None);
    import_path_case!(
        import_rel_three_dots_from_four_parts,
        &["a", "b", "c", "d"],
        ip(3, &["x"], "Y"),
        Some(rt(0, &["a", "b", "x", "Y"]))
    );
    import_path_case!(
        import_rel_four_dots_from_four_parts,
        &["a", "b", "c", "d"],
        ip(4, &["x"], "Y"),
        Some(rt(0, &["a", "x", "Y"]))
    );

    macro_rules! bind_case {
        ($name:ident, $sig:expr, [$($arg:expr),*], [$(($param:expr, $value:expr)),*]) => {
            #[test]
            fn $name() {
                let bindings = bind_call_arguments(&$sig, &[$($arg),*]);
                let expected = HashMap::from([$(($param.to_string(), $value.to_string()),)*]);
                assert_eq!(bindings, Some(expected));
            }
        };
        ($name:ident, $sig:expr, [$($arg:expr),*], None) => {
            #[test]
            fn $name() {
                assert_eq!(bind_call_arguments(&$sig, &[$($arg),*]), None);
            }
        };
    }

    bind_case!(bind_no_args_empty, sig(None, &["x"]), [], []);
    bind_case!(
        bind_too_many_positionals_ignored,
        sig(None, &["x"]),
        [pos("a"), pos("b")],
        [("x", "a")]
    );
    bind_case!(
        bind_unknown_keyword_preserved,
        sig(None, &["x"]),
        [kw("extra", "1")],
        [("extra", "1")]
    );
    bind_case!(
        bind_keyword_before_positional_does_not_shift_positionals,
        sig(None, &["x", "y"]),
        [kw("y", "2"), pos("1")],
        [("x", "1"), ("y", "2")]
    );
    bind_case!(
        bind_class_skips_cls,
        sig(Some("C"), &["cls", "x", "y"]),
        [pos("1"), pos("2")],
        [("x", "1"), ("y", "2")]
    );
    bind_case!(
        bind_class_does_not_skip_this,
        sig(Some("C"), &["this", "x"]),
        [pos("a"), pos("b")],
        [("this", "a"), ("x", "b")]
    );
    bind_case!(
        bind_function_does_not_skip_self,
        sig(None, &["self", "x"]),
        [pos("a"), pos("b")],
        [("self", "a"), ("x", "b")]
    );
    bind_case!(
        bind_keyword_overwrites_unknown_then_known_order,
        sig(None, &["x"]),
        [kw("x", "1"), kw("x", "2")],
        [("x", "2")]
    );
    bind_case!(
        bind_positional_then_keyword_override,
        sig(None, &["x"]),
        [pos("1"), kw("x", "2")],
        [("x", "2")]
    );
    bind_case!(
        bind_keyword_then_positional_override_by_position,
        sig(None, &["x"]),
        [kw("x", "1"), pos("2")],
        [("x", "2")]
    );
    bind_case!(
        bind_star_args_takes_extra_positionals,
        sig(None, &["x", "*args", "y"]),
        [pos("1"), pos("2"), pos("3")],
        [("x", "1")]
    );
    bind_case!(
        bind_bare_star_makes_keyword_only,
        sig(None, &["x", "*", "y", "**kw"]),
        [pos("1"), pos("2"), kw("y", "3")],
        [("x", "1"), ("y", "3")]
    );
    bind_case!(
        bind_positional_only_ignores_keyword,
        sig(None, &["x", "/", "y"]),
        [pos("1"), pos("2"), kw("x", "3")],
        [("x", "1"), ("y", "2")]
    );
    bind_case!(bind_list_splat_gives_none, sig(None, &["x", "y"]), [pos("1"), pos("*rest")], None);
    bind_case!(bind_dict_splat_gives_none, sig(None, &["x", "y"]), [pos("**cfg")], None);

    macro_rules! apply_ok_case {
        ($name:ident, $input:expr, $layer:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let app = app("x", $layer);
                let shapes = HashMap::from([("x".to_string(), shape($input))]);
                assert_eq!(
                    apply_layer_application(&app, &shapes).unwrap(),
                    Some(shape($expected))
                );
            }
        };
    }

    apply_ok_case!(
        apply_rank_three_numeric,
        &["a", "b", "3"],
        linear("3", "5"),
        &["a", "b", "5"]
    );
    apply_ok_case!(
        apply_rank_four_symbolic,
        &["t", "b", "h", "features"],
        linear("features", "out"),
        &["t", "b", "h", "out"]
    );
    apply_ok_case!(
        apply_single_dim_symbolic,
        &["features"],
        linear("features", "out"),
        &["out"]
    );
    apply_ok_case!(
        apply_numeric_no_leading_dims,
        &["10"],
        linear("10", "20"),
        &["20"]
    );
    apply_ok_case!(
        apply_preserves_duplicate_leading_dims,
        &["batch", "batch", "features"],
        linear("features", "out"),
        &["batch", "batch", "out"]
    );

    macro_rules! apply_err_case {
        ($name:ident, $input:expr, $layer:expr, $needle:expr) => {
            #[test]
            fn $name() {
                let app = app("x", $layer);
                let shapes = HashMap::from([("x".to_string(), shape($input))]);
                let error = apply_layer_application(&app, &shapes).unwrap_err();
                assert!(error.contains($needle));
            }
        };
    }

    apply_err_case!(
        apply_mismatch_rank_three,
        &["a", "b", "4"],
        linear("3", "5"),
        "got 4"
    );
    // Symbolic/unprovable mismatches propagate instead of erroring (#47);
    // whitespace and commutative +/* spelling are normalized before compare.
    apply_ok_case!(
        apply_case_sensitive_symbols_propagate,
        &["Batch", "Features"],
        linear("features", "out"),
        &["Batch", "out"]
    );
    apply_ok_case!(
        apply_whitespace_normalized_in_dims,
        &["features "],
        linear("features", "out"),
        &["out"]
    );
    apply_ok_case!(
        apply_expression_dim_whitespace_normalized,
        &["hidden * 2"],
        linear("hidden*2", "out"),
        &["out"]
    );
    apply_ok_case!(
        apply_commutative_sum_dims_match,
        &["hidden+features"],
        linear("features + hidden", "out"),
        &["out"]
    );
    apply_ok_case!(
        apply_empty_last_dim_propagates,
        &[""],
        linear("features", "out"),
        &["out"]
    );

    macro_rules! annotation_case {
        ($name:ident, $code:expr, [$(($param:expr, [$($dim:expr),*])),*]) => {
            #[test]
            fn $name() {
                let tree = parse($code);
                let scopes = extract_jaxtyping_shapes(tree.root_node(), $code).unwrap();
                let mut shapes: HashMap<String, Vec<String>> = HashMap::new();
                for scope in scopes {
                    for (k, v) in scope.shapes {
                        shapes.insert(k, v);
                    }
                }
                let expected = HashMap::from([
                    $(($param.to_string(), vec![$($dim.to_string()),*]),)*
                ]);
                assert_eq!(shapes, expected);
            }
        };
    }

    annotation_case!(
        ann_bool_array_shape,
        "def f(x: Bool[Array, \"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_complex_array_shape,
        "def f(x: Complex[Array, \"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_lowercase_array_rejected,
        "def f(x: Float[array, \"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_numpy_style_ndarray_shape,
        "def f(x: NDArray[Shape[\"a, b\"], Float]): pass",
        [("x", ["a", "b"])]
    );
    annotation_case!(
        ann_dtype_only_ndarray_rejected,
        "def f(x: NDArray[Float]): pass",
        []
    );
    annotation_case!(
        ann_numpy_ndarray_without_shape_rejected,
        "def f(x: Float[np.ndarray, \"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_list_wrapped_array_shape,
        "def f(x: list[Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_union_pipe_array_shape,
        "def f(x: Float[Array, \"b d\"] | None): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_two_string_literals_uses_first,
        "def f(x: Annotated[Float[Array, \"b d\"], \"meta\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_shape_comma_separates_dimensions,
        "def f(x: Float[Array, \"b,d\"]): pass",
        [("x", ["b", "d"])]
    );

    #[test]
    fn ann_qualified_numpy_style_ndarray_return_shape() {
        let code = "def f(x: nptyping.NDArray[nptyping.Shape[\"a, b\"], nptyping.Float]) -> nptyping.NDArray[nptyping.Shape[\"a, c\"], nptyping.Float]: pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        let scope = &scopes[1];
        assert_eq!(scope.shapes["x"], vec!["a", "b"]);
        assert_eq!(scope.return_shape, Some(vec!["a".to_string(), "c".to_string()]));
    }
    annotation_case!(
        ann_newline_inside_shape_splits_whitespace,
        "def f(x: Float[Array, \"b\\nd\"]): pass",
        [("x", ["b\\nd"])]
    );
    annotation_case!(
        ann_tab_inside_shape_splits_whitespace,
        "def f(x: Float[Array, \"b\td\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_uppercase_raw_prefix,
        "def f(x: Float[Array, R\"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_unicode_raw_prefix,
        "def f(x: Float[Array, ur\"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_capital_f_string_rejected,
        "def f(x: Float[Array, F\"b {d}\"]): pass",
        []
    );
    annotation_case!(
        ann_capital_b_string_rejected,
        "def f(x: Float[Array, B\"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_self_annotated_is_extracted_if_annotated,
        "class C:\n    def f(self: Float[Array, \"b d\"]): pass",
        [("self", ["b", "d"])]
    );
    annotation_case!(
        ann_cls_annotated_is_extracted_if_annotated,
        "class C:\n    def f(cls: Float[Array, \"b d\"]): pass",
        [("cls", ["b", "d"])]
    );
    annotation_case!(
        ann_tuple_wrapped_array_shape,
        "def f(x: tuple[Float[Array, \"b d\"], int]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_dict_wrapped_array_shape,
        "def f(x: dict[str, Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_typing_optional_wrapped_array_shape,
        "def f(x: typing.Optional[Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_union_uses_first_shape_string,
        "def f(x: Union[Float[Array, \"b d\"], Float[Array, \"b e\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_underscore_dimension_name,
        "def f(x: Float[Array, \"batch _features\"]): pass",
        [("x", ["batch", "_features"])]
    );
    annotation_case!(
        ann_question_mark_dimension_name_preserved,
        "def f(x: Float[Array, \"batch features?\"]): pass",
        [("x", ["batch", "features?"])]
    );
    annotation_case!(
        ann_ellipsis_dimension_name_preserved,
        "def f(x: Float[Array, \"batch ...\"]): pass",
        [("x", ["batch", "..."])]
    );
    annotation_case!(
        ann_colon_dimension_name_preserved,
        "def f(x: Float[Array, \"batch time:2\"]): pass",
        [("x", ["batch", "time:2"])]
    );

    apply_ok_case!(
        apply_rank_five_symbolic,
        &["a", "b", "c", "d", "features"],
        linear("features", "out"),
        &["a", "b", "c", "d", "out"]
    );
    apply_ok_case!(
        apply_comma_dimension_exact_match,
        &["batch", "features,"],
        linear("features,", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_question_mark_dimension_exact_match,
        &["batch", "features?"],
        linear("features?", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_ellipsis_dimension_exact_match,
        &["batch", "..."],
        linear("...", "out"),
        &["batch", "out"]
    );
    // Symbolic mismatches at the layer boundary are unprovable (ctor-arg
    // vocabulary vs annotation vocabulary, issue #47): the layer output is
    // trusted and propagated instead of erroring.
    apply_ok_case!(
        apply_comma_dimension_mismatch_propagates,
        &["batch", "features"],
        linear("features,", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_question_mark_dimension_mismatch_propagates,
        &["batch", "features"],
        linear("features?", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_ellipsis_dimension_mismatch_propagates,
        &["batch", "features"],
        linear("...", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_colon_dimension_mismatch_propagates,
        &["batch", "time"],
        linear("time:2", "out"),
        &["batch", "out"]
    );

    macro_rules! classify_case {
        ($name:ident, $module:expr, $owner:expr, $call_name:expr, [$(($key:expr, $value:expr)),*], $expected:expr) => {
            #[test]
            fn $name() {
                let bindings = HashMap::from([
                    $(($key.to_string(), $value.to_string()),)*
                ]);
                let call = ResolvedCallSignature {
                    implementation: ResolvedImplementation {
                        target: ResolvedModuleTarget {
                            dots: 0,
                            module_parts: parts($module),
                            file_path: PathBuf::from("unused.py"),
                            symbol_parts: Vec::new(),
                        },
                        symbol: $owner.map(|owner| PythonSymbol::Class {
                            name: owner.to_string(),
                        }),
                    },
                    signature: PythonCallableSignature {
                        owner: $owner.map(|owner| owner.to_string()),
                        name: $call_name.to_string(),
                        params: Vec::new(),
                    },
                    arguments: Vec::new(),
                    bindings,
                };
                assert_eq!(classify_layer_call(&call), $expected);
            }
        };
    }

    classify_case!(
        classify_exact_equinox_nn_init,
        &["equinox", "nn"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        Some(linear("1", "2"))
    );
    classify_case!(
        classify_equinox_nn_deep_init,
        &["equinox", "nn", "_linear"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        Some(linear("1", "2"))
    );
    classify_case!(
        classify_case_sensitive_module_rejected,
        &["Equinox", "nn"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
    classify_case!(
        classify_case_sensitive_owner_rejected,
        &["equinox", "nn"],
        Some("linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
    classify_case!(
        classify_missing_bindings_rejected,
        &["equinox", "nn"],
        Some("Linear"),
        "__init__",
        [],
        None
    );
    classify_case!(
        classify_wrong_function_name_rejected,
        &["equinox", "nn"],
        Some("Linear"),
        "__call__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
}

#[cfg(test)]
mod call_propagation_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_propagates_free_jnp_sum() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch features\"]):\n    y = jnp.sum(x, axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_propagates_free_np_reshape_tuple() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "import numpy as np\ndef f(x: Float[Array, \"6 4\"]):\n    y = np.reshape(x, (3, 8))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "8"])));
    }

    #[test]
    fn test_free_call_reshape_size_mismatch_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "import numpy as np\ndef f(x: Float[Array, \"6 4\"]):\n    y = np.reshape(x, (3, 9))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_propagates_method_flatten() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"2 3 4\"]):\n    y = x.flatten()";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["24"])));
    }

    #[test]
    fn test_torch_flatten_start_dim_function_and_method() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"B C H\"]):\n    a = torch.flatten(x, 1)\n    b = x.flatten(1)\n    c = torch.flatten(x, start_dim=0, end_dim=-2)\n    d = x.flatten(end_dim=1)\n    e = x.flatten('F')";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["B", "C*H"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["B", "C*H"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["B*C", "H"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["B*C", "H"])));
        assert_eq!(find_shape(&analysis, "e"), Some(&shape(&["B*C*H"])));
    }

    #[test]
    fn test_propagates_method_sum_axis_kwarg() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = x.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_propagates_method_reshape_multi_positional() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"6 4\"]):\n    y = x.reshape(3, 8)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "8"])));
    }

    #[test]
    fn test_chained_method_calls_in_source_order() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"2 3 4\"]):\n    y = x.flatten()\n    z = y.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["24"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&Vec::<String>::new()));
    }

    #[test]
    fn test_method_call_on_layer_output() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)\n    z = y.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["5"])));
    }

    #[test]
    fn test_free_and_method_call_same_function_name() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch features\"]):\n    y = jnp.sum(x, axis=0)\n    z = x.sum(axis=1)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_unknown_method_silently_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = x.frobnicate()";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_free_call_unknown_module_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = helpers.transform(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_method_call_error_range_covers_args() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"6 4\"]):\n    y = x.reshape(3, 9)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(3, 9)");
    }

    #[test]
    fn test_method_squeeze_axis() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"1 batch features\"]):\n    y = x.squeeze(0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_method_unsqueeze_axis() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = x.unsqueeze(1)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "1", "features"]))
        );
    }

    #[test]
    fn test_method_permute_multi_positional() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"a b c\"]):\n    y = x.permute(2, 0, 1)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["c", "a", "b"])));
    }
}

#[cfg(test)]
mod torch_nn_linear_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn write_fake_torch_nn(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("torch/nn")).unwrap();
        fs::write(tmp.path().join("torch/__init__.py"), "from . import nn").unwrap();
        fs::write(
            tmp.path().join("torch/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("torch/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, bias=True): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_single_torch_nn_linear_flow() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = torch.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "3"])));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_chained_torch_nn_linear_layers() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = torch.nn.Linear(3, 5)\n    l2 = torch.nn.Linear(5, 7)\n    y = l1(x)\n    z = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_from_torch_nn_import_linear_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "from torch.nn import Linear\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
    }

    #[test]
    fn test_torch_nn_linear_keyword_arguments() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = torch.nn.Linear(in_features=3, out_features=5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
    }
}

#[cfg(test)]
mod like_constructors_propagation_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_jnp_zeros_like_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch features\"]):\n    y = jnp.zeros_like(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_np_ones_like_preserves_symbolic_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.ones_like(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_torch_empty_like_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.empty_like(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_torch_chained_zeros_like_ones_like() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.zeros_like(x)\n    z = torch.ones_like(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
        assert_eq!(
            find_shape(&analysis, "z"),
            Some(&shape(&["batch", "features"]))
        );
    }
}

mod binary_op_propagation_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_matmul_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"batch k\"], b: Float[Array, \"k n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "n"])));
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"batch 3\"], b: Float[Array, \"5 n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("matmul dimension mismatch")
        );
    }

    #[test]
    fn test_matmul_batch_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"2 3 4\"], b: Float[Array, \"5 4 6\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("matmul batch dimension mismatch")
        );
    }

    #[test]
    fn test_matmul_matching_batch_dims() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"b m k\"], b: Float[Array, \"b k n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "m", "n"])));
    }

    #[test]
    fn test_return_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    return x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
        // variable is empty for return-statement ops
        assert_eq!(analysis.errors[0].variable, "");
        // The primary range selects the exact conflicting left dimension.
        assert_eq!(&code[analysis.errors[0].range.start_byte..analysis.errors[0].range.end_byte], "b");
    }

    #[test]
    fn test_return_matmul_compatible() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"b c\"]):\n    return x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "unexpected errors: {:?}", analysis.errors);
    }

    #[test]
    fn test_return_elementwise_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"c d\"]):\n    return x + y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("elementwise"));
        assert_eq!(analysis.errors[0].variable, "");
    }

    #[test]
    fn test_return_parenthesized_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    return (x @ y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_yield_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    yield x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_assert_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    assert x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_return_triple_paren_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    return (((x @ y)))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_return_tuple_paren_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    return x @ y, (a + b)";
        // x @ y mismatches (b != d); a + b is elementwise but a/b not annotated so silently skipped
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "expected exactly one error, got {:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_return_unary_negated_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    return -(x @ y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "got {:?}", analysis.errors);
    }

    #[test]
    fn test_yield_from_matmul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    yield from x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "got {:?}", analysis.errors);
    }

    #[test]
    fn test_floor_division_not_checked() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    z = x // y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 0, "got {:?}", analysis.errors);
    }

    #[test]
    fn test_tuple_unpack_lhs_matmul() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"a b\"], y: Float[Array, \"d e\"]):\n    p, q = x @ y";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 0, "got {:?}", analysis.errors);
    }

    #[test]
    fn test_elementwise_add_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"batch features\"], b: Float[Array, \"batch features\"]):\n    y = a + b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_elementwise_add_broadcast_bias() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch f\"], b: Float[Array, \"f\"]):\n    y = x + b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "f"])));
    }

    #[test]
    fn test_elementwise_mul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"a b\"], b: Float[Array, \"a c\"]):\n    y = a * b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("elementwise *"));
        assert!(analysis.errors[0].message.contains("a, b"));
        assert!(analysis.errors[0].message.contains("a, c"));
    }

    #[test]
    fn test_elementwise_sub_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"b d\"], b: Float[Array, \"b d\"]):\n    y = a - b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_elementwise_sub_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"x y\"], b: Float[Array, \"x z\"]):\n    y = a - b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("elementwise -"));
    }

    #[test]
    fn test_elementwise_div_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"b d\"], b: Float[Array, \"b d\"]):\n    y = a / b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_elementwise_div_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"p q\"], b: Float[Array, \"p r\"]):\n    y = a / b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("elementwise /"));
    }

    #[test]
    fn test_binary_op_interleaved_with_method_call() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"batch k\"], b: Float[Array, \"k n\"]):\n    y = a @ b\n    z = y.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "n"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n"])));
    }

    #[test]
    fn test_binary_op_inside_function_uses_own_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(a: Float[Array, \"b k\"], b: Float[Array, \"k n\"]):\n    y = a @ b\ndef g(a: Float[Array, \"x y\"], b: Float[Array, \"y z\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        let f_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("f"))
            .expect("f scope");
        let g_scope = analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some("g"))
            .expect("g scope");
        assert_eq!(f_scope.shapes.get("y"), Some(&shape(&["b", "n"])));
        assert_eq!(g_scope.shapes.get("y"), Some(&shape(&["x", "z"])));
    }
}
mod conv_layer_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// Write fake equinox.nn package with Conv1d, Conv2d, Conv3d, Linear
    fn write_fake_equinox_nn(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(tmp.path().join("equinox/__init__.py"), "from . import nn").unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._conv import Conv1d, Conv2d, Conv3d\nfrom ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_conv.py"),
            "class Conv1d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass\nclass Conv2d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass\nclass Conv3d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    /// Write fake torch.nn package with Conv1d, Conv2d, Conv3d, Linear
    fn write_fake_torch_nn(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("torch/nn")).unwrap();
        fs::write(tmp.path().join("torch/__init__.py"), "from . import nn").unwrap();
        fs::write(
            tmp.path().join("torch/nn/__init__.py"),
            "from ._conv import Conv1d, Conv2d, Conv3d\nfrom ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("torch/nn/_conv.py"),
            "class Conv1d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass\nclass Conv2d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1): pass\nclass Conv3d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass",
        )
        .unwrap();
        fs::write(
            tmp.path().join("torch/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, bias=True): pass",
        )
        .unwrap();
    }

    // ── Equinox Conv2d concrete dims ──

    #[test]
    fn test_equinox_conv2d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }

    // ── Torch Conv2d same ──

    #[test]
    fn test_torch_conv2d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }

    // ── in_channels mismatch -> ShapeError ──

    #[test]
    fn test_equinox_conv2d_channels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 4 32 32\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("expected 3 input channels")
        );
        assert!(analysis.errors[0].message.contains("got 4"));
    }

    #[test]
    fn test_torch_conv2d_channels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 5 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert!(
            analysis.errors[0]
                .message
                .contains("expected 3 input channels")
        );
        assert!(analysis.errors[0].message.contains("got 5"));
    }

    // ── Conv1d ──

    #[test]
    fn test_equinox_conv1d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 64\"]):\n    layer = eqx.nn.Conv1d(3, 16, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        // L_out = floor((64 + 2*0 - 5)/1) + 1 = 60
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "60"]))
        );
    }

    #[test]
    fn test_torch_conv1d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 64\"]):\n    layer = torch.nn.Conv1d(3, 16, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "60"]))
        );
    }

    // ── Conv3d ──

    #[test]
    fn test_equinox_conv3d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32 32\"]):\n    layer = eqx.nn.Conv3d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30", "30"]))
        );
    }

    #[test]
    fn test_torch_conv3d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32 32\"]):\n    layer = torch.nn.Conv3d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30", "30"]))
        );
    }

    // ── Conv2d with stride=2, padding=1 ──

    #[test]
    fn test_equinox_conv2d_stride2_padding1() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3, stride=2, padding=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        // H_out = floor((32 + 2*1 - 3)/2) + 1 = floor(31/2) + 1 = 15 + 1 = 16
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "16", "16"]))
        );
    }

    #[test]
    fn test_torch_conv2d_stride2_padding1() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "16", "16"]))
        );
    }

    // ── Symbolic input: in_channels matches as symbol ──

    #[test]
    fn test_equinox_conv2d_symbolic_in_channels() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch in_c H W\"]):\n    layer = eqx.nn.Conv2d(in_c, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        let y_shape = find_shape(&analysis, "y").unwrap();
        assert_eq!(y_shape[0], "batch");
        assert_eq!(y_shape[1], "16");
        // H and W are symbolic: (H+0-3)/1+1 and (W+0-3)/1+1
        // Since padding=0, stride=1 the formula simplifies:
        // inner = H, subtract 3 => H-3, divide by 1 => (H-3), add 1 => (H-3)+1
        assert_eq!(y_shape[2], "H-2");
        assert_eq!(y_shape[3], "W-2");
    }

    #[test]
    fn test_equinox_conv2d_symbolic_spatial_with_stride_padding() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 H W\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3, stride=2, padding=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        let y_shape = find_shape(&analysis, "y").unwrap();
        assert_eq!(y_shape[0], "batch");
        assert_eq!(y_shape[1], "16");
        // (H-1)/2+1
        assert_eq!(y_shape[2], "(H-1)/2+1");
        assert_eq!(y_shape[3], "(W-1)/2+1");
    }

    // ── Keyword constructor form ──

    #[test]
    fn test_torch_conv2d_keyword_constructor() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }

    #[test]
    fn test_equinox_conv2d_keyword_constructor() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = eqx.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }

    // ── from-import alias form ──

    #[test]
    fn test_from_torch_nn_import_conv2d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "from torch.nn import Conv2d\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "30", "30"]))
        );
    }

    // ── Direct apply_layer_application unit tests ──

    fn conv1d(
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
    ) -> LayerKind {
        LayerKind::Conv {
            spatial_rank: 1,
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
            dilation: "1".to_string(),
        }
    }

    fn conv2d(
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
    ) -> LayerKind {
        LayerKind::Conv {
            spatial_rank: 2,
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
            dilation: "1".to_string(),
        }
    }

    fn conv3d(
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
    ) -> LayerKind {
        LayerKind::Conv {
            spatial_rank: 3,
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
            dilation: "1".to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn layer_app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "y".to_string(),
            layer: "conv".to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
            extra_inputs: Vec::new(),
        }
    }

    #[test]
    fn test_conv1d_concrete_apply() {
        let app = layer_app("x", conv1d("3", "16", "5", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "64"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["batch", "16", "60"])));
    }

    #[test]
    fn test_conv2d_concrete_apply() {
        let app = layer_app("x", conv2d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "32", "32"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["batch", "16", "30", "30"])));
    }

    #[test]
    fn test_conv3d_concrete_apply() {
        let app = layer_app("x", conv3d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "32", "32", "32"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["batch", "16", "30", "30", "30"])));
    }

    #[test]
    fn test_conv2d_stride2_padding1_apply() {
        let app = layer_app("x", conv2d("3", "16", "3", "2", "1"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "32", "32"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        // (32 + 2*1 - 3)/2 + 1 = 16
        assert_eq!(output, Some(shape(&["batch", "16", "16", "16"])));
    }

    #[test]
    fn test_conv2d_channels_mismatch_error() {
        let app = layer_app("x", conv2d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "5", "32", "32"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("expected 3 input channels"));
        assert!(err.contains("got 5"));
    }

    #[test]
    fn test_conv1d_too_few_dims_error() {
        let app = layer_app("x", conv1d("3", "16", "5", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["3"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("at least 2 dims"));
    }

    #[test]
    fn test_conv2d_too_few_dims_error() {
        let app = layer_app("x", conv2d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "32"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("at least 3 dims"));
    }

    #[test]
    fn test_conv3d_too_few_dims_error() {
        let app = layer_app("x", conv3d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "32", "32"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("at least 4 dims"));
    }

    #[test]
    fn test_conv2d_symbolic_spatial_dims() {
        let app = layer_app("x", conv2d("3", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        let out = output.unwrap();
        assert_eq!(out[0], "batch");
        assert_eq!(out[1], "16");
        assert_eq!(out[2], "H-2");
        assert_eq!(out[3], "W-2");
    }

    #[test]
    fn test_conv2d_symbolic_channels_match() {
        let app = layer_app("x", conv2d("in_c", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "in_c", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        let out = output.unwrap();
        assert_eq!(out[1], "16");
    }

    #[test]
    fn test_conv2d_symbolic_channels_mismatch_propagates() {
        // Issue #47: symbolic ctor channels vs symbolic annotation dim is an
        // unprovable mismatch — trust the conv and propagate its output.
        let app = layer_app("x", conv2d("in_c", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "other", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap().unwrap();
        assert_eq!(output[1], "16");
    }

    #[test]
    fn test_conv1d_symbolic_spatial_dim() {
        let app = layer_app("x", conv1d("3", "16", "5", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "L"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        let out = output.unwrap();
        assert_eq!(out[2], "L-4");
    }

    #[test]
    fn test_conv1d_stride2_padding1_concrete() {
        let app = layer_app("x", conv1d("3", "16", "3", "2", "1"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "32"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        // (32 + 2*1 - 3)/2 + 1 = 16
        assert_eq!(output, Some(shape(&["batch", "16", "16"])));
    }

    #[test]
    fn test_conv2d_missing_input_returns_none() {
        let app = layer_app("x", conv2d("3", "16", "3", "1", "0"));
        let shapes = HashMap::new();
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, None);
    }

    // ── Tuple kernel_size not yet supported — must not classify ──

    #[test]
    fn test_torch_conv2d_tuple_kernel_size_not_classified() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\nlayer = torch.nn.Conv2d(3, 16, (3, 5))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        // Tuple kernel_size should not be classified — avoids garbage symbolic output
        assert!(!layers.contains_key("layer"));
    }

    #[test]
    fn test_torch_conv2d_tuple_stride_not_classified() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\nlayer = torch.nn.Conv2d(3, 16, 3, stride=(2, 1))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(!layers.contains_key("layer"));
    }

    // ── Fully symbolic kernel_size ──

    #[test]
    fn test_conv2d_symbolic_kernel_size() {
        let app = layer_app("x", conv2d("3", "16", "k", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        let out = output.unwrap();
        assert_eq!(out[0], "batch");
        assert_eq!(out[1], "16");
        // symbolic: H - k + 1  (stride=1, padding=0, symbolic k)
        assert!(out[2].contains('H'));
        assert!(out[2].contains('k'));
        assert!(out[2].contains('+'));
        assert!(out[3].contains('W'));
        assert!(out[3].contains('k'));
    }

    #[test]
    fn test_conv1d_symbolic_kernel_size_stride1() {
        let app = layer_app("x", conv1d("3", "16", "k", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3", "L"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        let out = output.unwrap();
        // symbolic: L - k + 1
        assert!(out[2].contains('L'));
        assert!(out[2].contains('k'));
    }

    // ── Extra kwargs (dilation, groups) silently ignored — must not crash ──

    #[test]
    fn test_torch_conv2d_with_extra_kwargs_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // dilation is not modeled; shape rule is approximate (dilation≠1 is wrong)
        // but the layer should still be classified without crashing
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3, dilation=2, groups=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // dilation=2 gives wrong spatial output, but no crash and no false error
        assert!(analysis.layers.contains_key("layer"));
        assert!(analysis.errors.is_empty());
        // Output shape is computed (though incorrect for dilation=2 — v1 limitation)
        assert!(has_shape(&analysis, "y"));
    }
}

#[cfg(test)]
mod shape_preserving_layer_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// Write fake torch.nn package with shape-preserving layers + Linear
    fn write_fake_torch_nn(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("torch/nn")).unwrap();
        fs::write(tmp.path().join("torch/__init__.py"), "from . import nn").unwrap();
        fs::write(
            tmp.path().join("torch/nn/__init__.py"),
            "from ._layers import Dropout, Dropout1d, Dropout2d, Dropout3d\nfrom ._layers import BatchNorm1d, BatchNorm2d, BatchNorm3d\nfrom ._layers import LayerNorm, GroupNorm\nfrom ._layers import ReLU, GELU, Sigmoid, Tanh, Softmax\nfrom ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("torch/nn/_layers.py"),
            "class Dropout:\n    def __init__(self, p=0.5): pass\nclass Dropout1d:\n    def __init__(self, p=0.5): pass\nclass Dropout2d:\n    def __init__(self, p=0.5): pass\nclass Dropout3d:\n    def __init__(self, p=0.5): pass\nclass BatchNorm1d:\n    def __init__(self, num_features): pass\nclass BatchNorm2d:\n    def __init__(self, num_features): pass\nclass BatchNorm3d:\n    def __init__(self, num_features): pass\nclass LayerNorm:\n    def __init__(self, normalized_shape): pass\nclass GroupNorm:\n    def __init__(self, num_groups, num_channels): pass\nclass ReLU:\n    def __init__(self): pass\nclass GELU:\n    def __init__(self): pass\nclass Sigmoid:\n    def __init__(self): pass\nclass Tanh:\n    def __init__(self): pass\nclass Softmax:\n    def __init__(self, dim=None): pass",
        )
        .unwrap();
        fs::write(
            tmp.path().join("torch/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, bias=True): pass",
        )
        .unwrap();
    }

    /// Write fake equinox.nn package with shape-preserving layers + Linear
    fn write_fake_equinox_nn(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(tmp.path().join("equinox/__init__.py"), "from . import nn").unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._layers import BatchNorm, LayerNorm, GroupNorm, PReLU\nfrom ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_layers.py"),
            "class BatchNorm:\n    def __init__(self, input_shape, axis_name): pass\nclass LayerNorm:\n    def __init__(self, normalized_shape): pass\nclass GroupNorm:\n    def __init__(self, groups, channels): pass\nclass PReLU:\n    def __init__(self): pass",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    // ── torch.nn.Dropout on 2D input ──

    #[test]
    fn test_torch_dropout_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    drop = torch.nn.Dropout(0.1)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "x"),
            Some(&shape(&["batch", "features"]))
        );
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.BatchNorm2d on (batch, 16, H, W) ──

    #[test]
    fn test_torch_batchnorm2d_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "H", "W"]))
        );
    }

    // ── torch.nn.LayerNorm on (batch, 16) ──

    #[test]
    fn test_torch_layernorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16\"]):\n    ln = torch.nn.LayerNorm([16])\n    y = ln(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16"])));
    }

    // ── equinox.nn.LayerNorm ──

    #[test]
    fn test_equinox_layernorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    ln = eqx.nn.LayerNorm(features)\n    y = ln(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.ReLU ──

    #[test]
    fn test_torch_relu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    relu = torch.nn.ReLU()\n    y = relu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "3", "32", "32"]))
        );
    }

    // ── Chained: batchnorm then relu ──

    #[test]
    fn test_chained_batchnorm_relu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    relu = torch.nn.ReLU()\n    y = bn(x)\n    z = relu(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "H", "W"]))
        );
        assert_eq!(
            find_shape(&analysis, "z"),
            Some(&shape(&["batch", "16", "H", "W"]))
        );
    }

    // ── Symbolic input shape ──

    #[test]
    fn test_symbolic_input_shape_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"a b c\"]):\n    drop = torch.nn.Dropout(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["a", "b", "c"])));
    }

    // ── equinox.nn.BatchNorm ──

    #[test]
    fn test_equinox_batchnorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    bn = eqx.nn.BatchNorm(input_shape, axis_name)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── equinox.nn.GroupNorm ──

    #[test]
    fn test_equinox_groupnorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch channels H W\"]):\n    gn = eqx.nn.GroupNorm(4, channels)\n    y = gn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "channels", "H", "W"]))
        );
    }

    // ── equinox.nn.PReLU ──

    #[test]
    fn test_equinox_prelu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    prelu = eqx.nn.PReLU()\n    y = prelu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.GELU ──

    #[test]
    fn test_torch_gelu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    gelu = torch.nn.GELU()\n    y = gelu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.Sigmoid ──

    #[test]
    fn test_torch_sigmoid_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    sig = torch.nn.Sigmoid()\n    y = sig(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.Tanh ──

    #[test]
    fn test_torch_tanh_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    tanh = torch.nn.Tanh()\n    y = tanh(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.Softmax ──

    #[test]
    fn test_torch_softmax_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    sm = torch.nn.Softmax(dim=1)\n    y = sm(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    // ── torch.nn.GroupNorm ──

    #[test]
    fn test_torch_groupnorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    gn = torch.nn.GroupNorm(4, 16)\n    y = gn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "16", "H", "W"]))
        );
    }

    // ── Chain with Linear: shape-preserving layer after Linear ──

    #[test]
    fn test_shape_preserving_after_linear() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    linear = torch.nn.Linear(3, 5)\n    relu = torch.nn.ReLU()\n    y = linear(x)\n    z = relu(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch", "5"])));
    }

    // ── Unit test: apply_layer_application with ShapePreserving variant ──

    #[test]
    fn test_apply_shape_preserving_variant() {
        let app = LayerApplication {
            variable: "y".to_string(),
            layer: "drop".to_string(),
            input: "x".to_string(),
            kind: LayerKind::ShapePreserving {
                name: "Dropout".to_string(),
            },
            range: Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
            extra_inputs: Vec::new(),
        };
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    // ── Unit test: ShapePreserving with missing input returns None ──

    #[test]
    fn test_apply_shape_preserving_missing_input() {
        let app = LayerApplication {
            variable: "y".to_string(),
            layer: "drop".to_string(),
            input: "x".to_string(),
            kind: LayerKind::ShapePreserving {
                name: "Dropout".to_string(),
            },
            range: Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
            extra_inputs: Vec::new(),
        };
        let shapes = HashMap::new();

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, None);
    }

    // ── Rank validation: under-rank input produces ShapeError ──
    //
    // Convention matches Conv layers: channels-first without requiring a batch
    // dimension. BatchNorm2d min_rank=3 (C, H, W), BatchNorm3d min_rank=4 (C, D, H, W).

    #[test]
    fn test_batchnorm2d_under_rank_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm2d requires at least 3D (C, H, W); 2D input is under-rank
        let code = "import torch\ndef f(x: Float[Array, \"batch 16\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("BatchNorm2d"));
        assert!(analysis.errors[0].message.contains("at least 3 dims"));
        assert!(analysis.errors[0].message.contains("got 2"));
    }

    #[test]
    fn test_batchnorm3d_under_rank_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm3d requires at least 4D (C, D, H, W); 3D input is under-rank
        let code = "import torch\ndef f(x: Float[Array, \"8 H W\"]):\n    bn = torch.nn.BatchNorm3d(8)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("BatchNorm3d"));
        assert!(analysis.errors[0].message.contains("at least 4 dims"));
        assert!(analysis.errors[0].message.contains("got 3"));
    }

    #[test]
    fn test_dropout2d_under_rank_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout2d requires at least 3D; 1D input is under-rank
        let code = "import torch\ndef f(x: Float[Array, \"features\"]):\n    drop = torch.nn.Dropout2d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("Dropout2d"));
        assert!(analysis.errors[0].message.contains("at least 3 dims"));
    }

    #[test]
    fn test_dropout1d_under_rank_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout1d requires at least 2D (C, L); 1D input is under-rank
        let code = "import torch\ndef f(x: Float[Array, \"features\"]):\n    drop = torch.nn.Dropout1d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("Dropout1d"));
        assert!(analysis.errors[0].message.contains("at least 2 dims"));
    }

    #[test]
    fn test_dropout3d_under_rank_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout3d requires at least 4D (C, D, H, W); 2D input is under-rank
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    drop = torch.nn.Dropout3d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("Dropout3d"));
        assert!(analysis.errors[0].message.contains("at least 4 dims"));
    }

    // ── Boundary tests: unbatched inputs accepted at min rank ──

    #[test]
    fn test_batchnorm2d_unbatched_accepts_3d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm2d on (C, H, W) — exactly at min rank 3, no batch dim
        let code = "import torch\ndef f(x: Float[Array, \"16 H W\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["16", "H", "W"])));
    }

    #[test]
    fn test_batchnorm3d_unbatched_accepts_4d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm3d on (C, D, H, W) — exactly at min rank 4, no batch dim
        let code = "import torch\ndef f(x: Float[Array, \"8 D H W\"]):\n    bn = torch.nn.BatchNorm3d(8)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "D", "H", "W"]))
        );
    }

    #[test]
    fn test_dropout2d_unbatched_accepts_3d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout2d on (C, H, W) — exactly at min rank 3
        let code = "import torch\ndef f(x: Float[Array, \"16 H W\"]):\n    drop = torch.nn.Dropout2d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["16", "H", "W"])));
    }

    #[test]
    fn test_dropout3d_unbatched_accepts_4d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout3d on (C, D, H, W) — exactly at min rank 4
        let code = "import torch\ndef f(x: Float[Array, \"8 D H W\"]):\n    drop = torch.nn.Dropout3d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "D", "H", "W"]))
        );
    }

    #[test]
    fn test_batchnorm1d_accepts_2d_input() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm1d on (C, L) — exactly at min rank 2
        let code = "import torch\ndef f(x: Float[Array, \"16 L\"]):\n    bn = torch.nn.BatchNorm1d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["16", "L"])));
    }

    #[test]
    fn test_dropout1d_accepts_2d_input() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout1d on (C, L) — exactly at min rank 2
        let code = "import torch\ndef f(x: Float[Array, \"16 L\"]):\n    drop = torch.nn.Dropout1d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["16", "L"])));
    }

    #[test]
    fn test_dropout_accepts_any_rank() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Plain Dropout accepts any rank, including 1D
        let code = "import torch\ndef f(x: Float[Array, \"features\"]):\n    drop = torch.nn.Dropout(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_equinox_batchnorm_accepts_any_rank() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        // Equinox BatchNorm is rank-agnostic
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"features\"]):\n    bn = eqx.nn.BatchNorm(input_shape, axis_name)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    // ── Unit tests for min_rank_for_shape_preserving via apply_layer_application ──

    fn sp_app(name: &str, input: &str) -> LayerApplication {
        LayerApplication {
            variable: "y".to_string(),
            layer: "layer".to_string(),
            input: input.to_string(),
            kind: LayerKind::ShapePreserving {
                name: name.to_string(),
            },
            range: Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
            extra_inputs: Vec::new(),
        }
    }

    #[test]
    fn test_unit_batchnorm2d_2d_input_rejected() {
        let app = sp_app("BatchNorm2d", "x");
        let shapes = HashMap::from([("x".to_string(), shape(&["16", "L"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("BatchNorm2d"));
        assert!(err.contains("at least 3 dims"));
        assert!(err.contains("got 2"));
    }

    #[test]
    fn test_unit_batchnorm2d_3d_input_accepted() {
        let app = sp_app("BatchNorm2d", "x");
        let shapes = HashMap::from([("x".to_string(), shape(&["16", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["16", "H", "W"])));
    }

    #[test]
    fn test_unit_batchnorm3d_3d_input_rejected() {
        let app = sp_app("BatchNorm3d", "x");
        let shapes = HashMap::from([("x".to_string(), shape(&["8", "H", "W"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("BatchNorm3d"));
        assert!(err.contains("at least 4 dims"));
        assert!(err.contains("got 3"));
    }

    #[test]
    fn test_unit_batchnorm3d_4d_input_accepted() {
        let app = sp_app("BatchNorm3d", "x");
        let shapes = HashMap::from([("x".to_string(), shape(&["8", "D", "H", "W"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["8", "D", "H", "W"])));
    }

    #[test]
    fn test_unit_relu_1d_input_accepted() {
        let app = sp_app("ReLU", "x");
        let shapes = HashMap::from([("x".to_string(), shape(&["features"]))]);
        let output = apply_layer_application(&app, &shapes).unwrap();
        assert_eq!(output, Some(shape(&["features"])));
    }

    #[test]
    fn test_unit_layernorm_scalar_input_rejected() {
        let app = sp_app("LayerNorm", "x");
        let shapes = HashMap::from([("x".to_string(), Vec::new())]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("LayerNorm"));
        assert!(err.contains("at least 1 dims"));
        assert!(err.contains("got 0"));
    }

    #[test]
    fn test_unit_groupnorm_scalar_input_rejected() {
        let app = sp_app("GroupNorm", "x");
        let shapes = HashMap::from([("x".to_string(), Vec::new())]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("GroupNorm"));
        assert!(err.contains("at least 1 dims"));
    }

    // ── Layer applied to a nested inline expression ──
    // Verifies the layer pre-check now routes its input through
    // resolve_call_args, so `y = drop(jnp.exp(x))` resolves a shape.
    #[test]
    fn test_layer_with_nested_call_input_resolves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\nimport jax.numpy as jnp\ndef f(x: Float[Array, \"batch features\"]):\n    drop = torch.nn.Dropout(0.1)\n    y = drop(jnp.exp(x))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "unexpected errors: {:?}", analysis.errors);
        // jnp.exp(x) preserves shape, Dropout preserves shape → y is [batch, features]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }
}

// ── Integration tests for torch.nn.functional.pad ──

#[cfg(test)]
mod torch_nn_functional_pad_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_f_pad_1d_symbolic() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"n\"]):\n    y = F.pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n+3"])));
    }

    #[test]
    fn test_torch_nn_functional_pad_2d_symbolic() {
        // apply_known_pad applies pad pairs in dimension order (dim 0 first, dim 1 second),
        // unlike PyTorch's reverse-axis convention. We test against the existing parser semantics.
        let code = "import torch\ndef f(x: Float[Array, \"h w\"]):\n    y = torch.nn.functional.pad(x, (1, 2, 3, 4))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["h+3", "w+7"])));
    }

    #[test]
    fn test_f_pad_preserves_symbolic_with_addition() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"height width\"]):\n    y = F.pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["height+3", "width+3"]))
        );
    }

    #[test]
    fn test_f_pad_dynamic_pad_variable_returns_none() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"h w\"]):\n    y = F.pad(x, pad_width)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // Dynamic pad width variable cannot be statically parsed; should return no shape, not error
        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_f_pad_invalid_pad_does_not_crash() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"h w\"]):\n    y = F.pad(x, \"invalid\")";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // Invalid/unparseable pad does not crash; returns no shape
        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_from_import_pad() {
        let code = "from torch.nn.functional import pad\ndef f(x: Float[Array, \"n\"]):\n    y = pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n+3"])));
    }

    #[test]
    fn test_from_import_pad_alias() {
        let code = "from torch.nn.functional import pad as F_pad\ndef f(x: Float[Array, \"n\"]):\n    y = F_pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n+3"])));
    }

    #[test]
    fn test_f_pad_per_axis_numeric() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"10 20\"]):\n    y = F.pad(x, ((1, 2), (3, 4)))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["13", "27"])));
    }
}

// ── Integration tests for free-function reductions & shape-preserving functions ──

#[cfg(test)]
mod free_reduction_shape_preserving_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_jnp_all_axis_1_gives_batch() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch features\"]):\n    y = jnp.all(x, axis=1)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_np_argmax_axis_0_gives_features() {
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.argmax(x, axis=0)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_torch_argsort_preserves_batch_features() {
        let code =
            "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.argsort(x)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_torch_cumsum_dim_1_preserves_batch_features() {
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.cumsum(x, dim=1)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_chained_np_sort_then_any_propagates() {
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.sort(x)\n    z = np.any(y, axis=-1)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch"])));
    }
}

#[cfg(test)]
mod linalg_inv_integration_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_jnp_linalg_inv_batched_square_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch n n\"]):\n    y = jnp.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "n", "n"]))
        );
    }

    #[test]
    fn test_np_linalg_inv_2d_square_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"n n\"]):\n    y = np.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn test_torch_linalg_inv_batched_square_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"b n n\"]):\n    y = torch.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "n", "n"])));
    }

    #[test]
    fn test_linalg_inv_non_square_reports_error_no_output_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"m n\"]):\n    y = np.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("last two dimensions to match")
        );
        assert!(!has_shape(&analysis, "y"));
    }
}

#[cfg(test)]
mod builtin_layer_catalog_end_to_end_tests {
    //! End-to-end tests that mirror the layer-mismatch lines in
    //! `test_python.py`, exercising `analyze_layer_shapes` with empty
    //! `search_roots` and a `read_file` that always returns `None`. These
    //! verify the built-in layer catalog short-circuits disk resolution so
    //! the analyzer still reports `ShapeError`s for equinox.nn / torch.nn
    //! layer mismatches even when the frameworks aren't reachable on disk.

    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn no_read(_: &PathBuf) -> Option<String> {
        None
    }

    fn empty_roots() -> Vec<PathBuf> {
        Vec::new()
    }

    #[test]
    fn test_equinox_linear_mismatch_without_disk_reports_error() {
        let code = "import equinox as eqx\nfrom jaxtyping import Float, Array\n\
                    def f(x: Float[Array, \"batch 32\"]):\n\
                    \x20   layer = eqx.nn.Linear(64, 128)\n\
                    \x20   y = layer(x)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.errors.len(), 1);
        let err = &analysis.errors[0];
        assert_eq!(err.variable, "y");
        assert!(err.message.contains("Linear"));
        assert!(err.message.contains("layer"));
        assert!(err.message.contains("expected input last dim"));
        assert!(err.message.contains("64"));
        assert!(err.message.contains("32"));
    }

    #[test]
    fn test_torch_linear_mismatch_without_disk_reports_error() {
        let code = "import torch\nfrom jaxtyping import Float, Array\n\
                    def f(x: Float[Array, \"batch 64\"]):\n\
                    \x20   layer = torch.nn.Linear(128, 256)\n\
                    \x20   y = layer(x)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.errors.len(), 1);
        let err = &analysis.errors[0];
        assert_eq!(err.variable, "y");
        assert!(err.message.contains("Linear"));
        assert!(err.message.contains("layer"));
        assert!(err.message.contains("expected input last dim"));
        assert!(err.message.contains("128"));
        assert!(err.message.contains("64"));
    }

    #[test]
    fn test_equinox_conv2d_channels_mismatch_without_disk_reports_error() {
        let code = "import equinox as eqx\nfrom jaxtyping import Float, Array\n\
                    def f(x: Float[Array, \"batch 1 32 32\"]):\n\
                    \x20   layer = eqx.nn.Conv2d(3, 16, 3)\n\
                    \x20   y = layer(x)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.errors.len(), 1);
        let err = &analysis.errors[0];
        assert_eq!(err.variable, "y");
        assert!(err.message.contains("Conv2d"));
        assert!(err.message.contains("expected"));
        assert!(err.message.contains("input channels"));
        assert!(err.message.contains("1"));
    }

    #[test]
    fn test_torch_conv2d_channels_mismatch_without_disk_reports_error() {
        let code = "import torch\nfrom jaxtyping import Float, Array\n\
                    def f(x: Float[Array, \"batch 8 32 32\"]):\n\
                    \x20   layer = torch.nn.Conv2d(3, 16, 3)\n\
                    \x20   y = layer(x)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.errors.len(), 1);
        let err = &analysis.errors[0];
        assert_eq!(err.variable, "y");
        assert!(err.message.contains("Conv2d"));
        assert!(err.message.contains("expected"));
        assert!(err.message.contains("input channels"));
        assert!(err.message.contains("8"));
    }

    #[test]
    fn test_equinox_linear_success_without_disk_propagates_output_shape() {
        let code = "import equinox as eqx\nfrom jaxtyping import Float, Array\n\
                    def f(x: Float[Array, \"batch 64\"]):\n\
                    \x20   layer = eqx.nn.Linear(64, 128)\n\
                    \x20   y = layer(x)";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&vec!["batch".to_string(), "128".to_string()])
        );
    }

    #[test]
    fn test_combined_test_python_mismatches_without_disk_match_expected_count() {
        // Mirror of the four layer-mismatch lines in test_python.py
        // (Linear in-dim mismatch L44/L58, Conv2d channel mismatch L72/L92).
        // Layer names are distinct because the layer map is flat across
        // functions — same-named locals would collide last-wins.
        let code = "import equinox as eqx\nimport torch\n\
                    from jaxtyping import Float, Array\n\
                    \n\
                    def eqx_linear_mismatch(x: Float[Array, \"batch 32\"]):\n\
                    \x20   eqx_lin = eqx.nn.Linear(64, 128)\n\
                    \x20   y = eqx_lin(x)\n\
                    \n\
                    def torch_linear_mismatch(x: Float[Array, \"batch 64\"]):\n\
                    \x20   torch_lin = torch.nn.Linear(128, 256)\n\
                    \x20   y = torch_lin(x)\n\
                    \n\
                    def eqx_conv2d_channels_mismatch(x: Float[Array, \"batch 1 32 32\"]):\n\
                    \x20   eqx_conv = eqx.nn.Conv2d(3, 16, 3)\n\
                    \x20   y = eqx_conv(x)\n\
                    \n\
                    def torch_conv2d_channels_mismatch(x: Float[Array, \"batch 8 32 32\"]):\n\
                    \x20   torch_conv = torch.nn.Conv2d(3, 16, 3)\n\
                    \x20   y = torch_conv(x)\n";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &empty_roots(), no_read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            4,
            "expected one ShapeError per mismatch line, got {:?}",
            analysis.errors
        );

        let messages: Vec<&str> = analysis.errors.iter().map(|e| e.message.as_str()).collect();
        let linear_errs: Vec<_> = messages.iter().filter(|m| m.contains("Linear")).collect();
        let conv_errs: Vec<_> = messages.iter().filter(|m| m.contains("Conv2d")).collect();
        assert_eq!(linear_errs.len(), 2);
        assert_eq!(conv_errs.len(), 2);
    }
}

#[cfg(test)]
mod linalg_det_integration_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_jnp_linalg_det_batched_square_returns_batch_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch n n\"]):\n    y = jnp.linalg.det(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_np_linalg_det_2d_square_returns_scalar_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"n n\"]):\n    y = np.linalg.det(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&Vec::<String>::new()));
    }

    #[test]
    fn test_torch_linalg_det_multi_batch_returns_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "import torch\ndef f(x: Float[Array, \"b t n n\"]):\n    y = torch.linalg.det(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "t"])));
    }

    #[test]
    fn test_linalg_det_non_square_reports_error_no_output_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"m n\"]):\n    y = np.linalg.det(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("last two dimensions to match")
        );
        assert!(!has_shape(&analysis, "y"));
    }
}

#[cfg(test)]
mod constructor_coverage_integration_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_jnp_empty_tuple_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f():\n    y = jnp.empty((batch, features))";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_np_identity_square() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f():\n    y = np.identity(n)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn test_jnp_linspace_keyword_num() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f():\n    y = jnp.linspace(0, 1, num=steps)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["steps"])));
    }

    #[test]
    fn test_torch_linspace_keyword_steps() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f():\n    y = torch.linspace(0, 1, steps=steps)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["steps"])));
    }

    #[test]
    fn test_np_logspace_keyword_num() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f():\n    y = np.logspace(0, 3, num=n)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n"])));
    }
}

mod torch_varargs_size_tests {
    use super::*;
    use tree_sitter::Parser;

    fn y_shape(code: &str) -> Option<Vec<String>> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();
        assert!(analysis.errors.is_empty());
        find_shape(&analysis, "y").cloned()
    }

    fn dims(dims: &[&str]) -> Option<Vec<String>> {
        Some(dims.iter().map(|dim| dim.to_string()).collect())
    }

    fn torch(call: &str) -> Option<Vec<String>> {
        y_shape(&format!("import torch\ndef f(B, x):\n    y = torch.{call}"))
    }

    fn numeric(call: &str) -> Option<Vec<String>> {
        y_shape(&format!(
            "import numpy as np\nimport jax.numpy as jnp\ndef f(B, x):\n    y = {call}"
        ))
    }

    #[test]
    fn varargs_form_gives_every_dim() {
        assert_eq!(torch("zeros(4, 5)"), dims(&["4", "5"]));
        assert_eq!(torch("ones(2, 3, 4)"), dims(&["2", "3", "4"]));
        assert_eq!(torch("empty(3, 4)"), dims(&["3", "4"]));
        assert_eq!(torch("rand(2, 3)"), dims(&["2", "3"]));
        assert_eq!(torch("randn(2, 3, 4)"), dims(&["2", "3", "4"]));
        assert_eq!(torch("zeros(4)"), dims(&["4"]));
    }

    #[test]
    fn varargs_form_ignores_keywords() {
        assert_eq!(
            torch("zeros(4, 5, dtype=torch.float32)"),
            dims(&["4", "5"])
        );
        assert_eq!(torch("randn(2, 3, requires_grad=True)"), dims(&["2", "3"]));
    }

    #[test]
    fn tuple_list_and_size_keyword_forms() {
        assert_eq!(torch("zeros((4, 5))"), dims(&["4", "5"]));
        assert_eq!(torch("rand([2, 3])"), dims(&["2", "3"]));
        assert_eq!(torch("zeros((4, 5), dtype=torch.float32)"), dims(&["4", "5"]));
        assert_eq!(torch("zeros(size=(4, 5))"), dims(&["4", "5"]));
        assert_eq!(torch("full((2, 3), 1.5)"), dims(&["2", "3"]));
    }

    #[test]
    fn symbolic_sizes_match_tuple_form() {
        assert_eq!(torch("zeros(B, 3)"), dims(&["B", "3"]));
        assert_eq!(torch("zeros((B, 3))"), dims(&["B", "3"]));
        assert_eq!(torch("randn(B, 3, 4)"), dims(&["B", "3", "4"]));
    }

    #[test]
    fn unclear_sizes_stay_unknown() {
        assert_eq!(torch("zeros(*x)"), None);
        assert_eq!(torch("zeros(x.shape)"), None);
        assert_eq!(torch("zeros(4, (5, 6))"), None);
        assert_eq!(torch("zeros()"), None);
    }

    #[test]
    fn size_expressions_and_single_names_stay_unknown() {
        assert_eq!(torch("zeros(size=shp)"), None);
        assert_eq!(torch("zeros(shp)"), None);
        assert_eq!(torch("zeros((B, S) + (3,))"), dims(&["B", "S", "3"]));
        assert_eq!(torch("zeros((B, S) * 2)"), None);
        assert_eq!(torch("zeros((f(a, b), 3))"), dims(&["f(a, b)", "3"]));
        assert_eq!(torch("zeros(size=(B, S) + (3,))"), dims(&["B", "S", "3"]));
        assert_eq!(torch("zeros(4)"), dims(&["4"]));
        let method = |call: &str| {
            y_shape(&format!("def f(x, shp):\n    y = x.{call}"))
        };
        assert_eq!(method("new_zeros(size=shp)"), None);
        assert_eq!(method("new_zeros(shp)"), None);
    }

    #[test]
    fn new_constructors_use_varargs() {
        let method = |call: &str| {
            y_shape(&format!(
                "def f(x: Float[Array, \"4\"]):\n    y = x.{call}"
            ))
        };
        assert_eq!(method("new_zeros(3, 4)"), dims(&["3", "4"]));
        assert_eq!(method("new_ones(3, 4)"), dims(&["3", "4"]));
        assert_eq!(method("new_empty(3, 4)"), dims(&["3", "4"]));
        assert_eq!(method("new_zeros((3, 4))"), dims(&["3", "4"]));
        assert_eq!(method("new_full((3, 4), 1.0)"), dims(&["3", "4"]));
    }

    #[test]
    fn numpy_and_jax_second_positional_is_dtype() {
        assert_eq!(numeric("np.zeros((4, 5), np.float32)"), dims(&["4", "5"]));
        assert_eq!(numeric("jnp.ones((2, 3), jnp.float32)"), dims(&["2", "3"]));
        assert_eq!(numeric("np.empty((3, 4), float)"), dims(&["3", "4"]));
        assert_eq!(numeric("np.zeros(4, float)"), dims(&["4"]));
        assert_eq!(numeric("jnp.zeros(B)"), None);
        assert_eq!(numeric("np.full((2, 3), 1.5)"), dims(&["2", "3"]));
    }
}


/// Regression guard for venv-discovery wiring: given a site-packages
/// directory in `search_roots`, layer-mismatch diagnostics fire through the
/// on-disk resolution path. This complements the in-process builtin layer
/// catalog by exercising the disk resolver against a fake equinox install
/// laid out exactly the way a real venv would store it.
#[cfg(test)]
mod site_packages_resolution_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    #[test]
    fn resolves_equinox_linear_through_fake_site_packages() {
        let tmp = tempfile::tempdir().unwrap();
        let site = tmp
            .path()
            .join(".venv")
            .join("lib")
            .join("python3.11")
            .join("site-packages");
        fs::create_dir_all(site.join("equinox").join("nn")).unwrap();
        fs::write(site.join("equinox").join("__init__.py"), "from . import nn").unwrap();
        fs::write(
            site.join("equinox").join("nn").join("__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            site.join("equinox").join("nn").join("_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        )
        .unwrap();

        let code = "from jaxtyping import Float, Array\nimport equinox as eqx\n\ndef f(x: Float[Array, \"batch 32\"]):\n    layer = eqx.nn.Linear(64, 128)\n    y = layer(x)\n    return y\n";
        let tree = parse(code);
        let roots = vec![site];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 8, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected exactly one ShapeError, got {:?}",
            analysis.errors
        );
        let msg = &analysis.errors[0].message;
        assert!(msg.contains("Linear"), "missing 'Linear' in {:?}", msg);
        assert!(msg.contains("64"), "missing '64' in {:?}", msg);
        assert!(msg.contains("32"), "missing '32' in {:?}", msg);
    }
}

/// Phase 2: Cross-function shape propagation tests.
/// When a user-defined function has jaxtyping annotations, calls to it
/// propagate the return shape (with dim binding) to the assignment target.
#[cfg(test)]
mod user_function_propagation_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// Helper: find a shape for `var` in the scope named `scope_name`.
    fn find_shape_in_scope<'a>(
        analysis: &'a LayerShapeAnalysis,
        scope_name: &str,
        var: &str,
    ) -> Option<&'a Vec<String>> {
        analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some(scope_name))
            .and_then(|s| s.shapes.get(var))
    }

    /// Test 1: basic return-shape propagation with symbolic dims.
    /// `z = f(y)` where f returns Float[Array, "m"] — z gets ["m"].
    #[test]
    fn test_user_function_return_shape_propagates_to_caller() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["m"]))
        );
    }

    /// Test 2: param dims bind into the return shape.
    /// f(x: Float[Array, "a b"]) -> Float[Array, "a c"]
    /// caller passes x: Float[Array, "batch features"]
    /// z = f(x) → z has shape ["batch", "c"] (a→batch, c is fresh)
    #[test]
    fn test_user_function_param_dim_binds_into_return_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "a b"]) -> Float[Array, "a c"]:
    pass

def caller(x: Float[Array, "batch features"]) -> None:
    z = f(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["batch", "c"]))
        );
    }

    /// Test 3: rank mismatch between param and arg emits a diagnostic.
    /// f expects rank 2, caller passes rank 1.
    #[test]
    fn test_user_function_arg_rank_mismatch_emits_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n m"]) -> Float[Array, "n"]:
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected 1 error, got {:?}",
            analysis.errors
        );
        assert_eq!(analysis.errors[0].variable, "z");
        assert!(
            analysis.errors[0].message.contains("rank"),
            "error should mention rank: {:?}",
            analysis.errors[0].message
        );
        // No shape recorded for z when there's a mismatch.
        assert_eq!(find_shape_in_scope(&analysis, "caller", "z"), None);
    }

    /// Test 4: concrete dim mismatch emits a diagnostic.
    /// f expects dim 3, caller passes dim 5.
    #[test]
    fn test_user_function_concrete_dim_mismatch_emits_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "3 features"]) -> Float[Array, "features"]:
    pass

def caller(y: Float[Array, "5 features"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected 1 error, got {:?}",
            analysis.errors
        );
        assert_eq!(analysis.errors[0].variable, "z");
        assert!(
            analysis.errors[0].message.contains('3') && analysis.errors[0].message.contains('5'),
            "error should mention 3 vs 5: {:?}",
            analysis.errors[0].message
        );
    }

    /// Test 5: function without return annotation — no propagation, no error.
    #[test]
    fn test_user_function_without_return_annotation_skips_propagation() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n"]):
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(find_shape_in_scope(&analysis, "caller", "z"), None);
    }

    /// Test 6: user function call at module scope.
    #[test]
    fn test_user_function_call_in_module_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

y: Float[Array, "batch"] = 0
z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        // Module scope (index 0) should have z → ["m"]
        assert_eq!(analysis.scopes[0].shapes.get("z"), Some(&shape(&["m"])));
    }

    /// Test 7: a bare name that also exists as a known function (e.g. `reshape`)
    /// is resolved as a user function, because `classify_known_function`
    /// only matches module-qualified names like `np.reshape`.
    #[test]
    fn test_bare_name_matching_known_function_resolves_as_user_function() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def reshape(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = reshape(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // Bare `reshape` is not classified by `classify_known_function`, so the
        // user-function branch handles it: z gets ["m"].
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["m"]))
        );
    }

    /// Test 8: repeated dim name in param must bind consistently.
    /// f(x: Float[Array, "n n"]) — passing ["a", "b"] where a ≠ b is an error.
    #[test]
    fn test_user_function_repeated_dim_name_must_bind_consistently() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n n"]) -> Float[Array, "n"]:
    pass

def caller(y: Float[Array, "a b"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected 1 error, got {:?}",
            analysis.errors
        );
        assert_eq!(analysis.errors[0].variable, "z");
        assert!(
            analysis.errors[0].message.contains("cannot be both"),
            "error should mention conflicting binding: {:?}",
            analysis.errors[0].message
        );
        // No shape for z when binding is inconsistent.
        assert_eq!(find_shape_in_scope(&analysis, "caller", "z"), None);
    }

    /// Test 9: keyword argument matching a declared param name is honoured.
    /// z = f(x=y) should resolve the same as z = f(y).
    #[test]
    fn test_user_function_keyword_arg_resolves_by_param_name() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "a b"]) -> Float[Array, "a c"]:
    pass

def caller(y: Float[Array, "batch features"]) -> None:
    z = f(x=y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["batch", "c"]))
        );
    }

    /// Test 10: self-recursive call is skipped (no panic, no infinite loop).
    /// The byte-range exclusion prevents a function from resolving to itself.
    #[test]
    fn test_user_function_self_recursive_call_does_not_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    z = f(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // Self-call is silently skipped — the user-function branch can't
        // resolve f (its scope contains the call byte), so it falls through.
        // No shape recorded for z, no error.
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(find_shape_in_scope(&analysis, "f", "z"), None);
    }

    /// Test 11: inner-scope preference when same function name appears at
    /// two different lexical depths. The innermost (smallest byte-range)
    /// scope wins.
    #[test]
    fn test_user_function_inner_scope_preferred_on_name_collision() {
        let tmp = tempfile::tempdir().unwrap();
        // Two functions named `g` at different depths.
        // The outer `g` returns ["outer_dim"], the inner returns ["inner_dim"].
        // Call from `caller` should pick the inner one (smaller scope),
        // but since both are at the same nesting level as the call, the
        // one whose scope does NOT contain the call byte wins. If both
        // don't contain it, smallest scope wins.
        let code = r#"
def g(x: Float[Array, "n"]) -> Float[Array, "outer_dim"]:
    pass

def h():
    def g(x: Float[Array, "n"]) -> Float[Array, "inner_dim"]:
        pass
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = g(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        // The outer `g` (smaller scope among candidates whose byte range
        // doesn't contain the call) wins.
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["outer_dim"]))
        );
    }

    /// Test 12: return dim referencing a param dim from an omitted arg.
    /// When a function has multiple params but the call only provides one
    /// positional arg, the unprovided param's dims are not in the binding.
    /// A return dim referencing that param passes through unchanged.
    #[test]
    fn test_user_function_return_dim_from_omitted_arg_passes_through() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "a"], w: Float[Array, "k"]) -> Float[Array, "a k"]:
    pass

def caller(y: Float[Array, "batch"]) -> None:
    z = f(y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // Only the first positional arg is provided, so `a` binds to
        // "batch" but `k` is unbound. The return shape ["a", "k"] becomes
        // ["batch", "k"] — `k` passes through as a fresh dim.
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "z"),
            Some(&shape(&["batch", "k"]))
        );
    }

    // ── Cross-file return-type tracing ──────────────────────────────────
    // Same-file helpers are covered above via `bind_user_function_args`; these
    // extend the coverage to helpers imported from another file on disk,
    // resolved through `resolve_imported_function_shape` +
    // `apply_imported_user_function`.

    /// `from mylib.helpers import project` then `y = project(x)` — the
    /// callee lives in a different file, found through the import map and
    /// `resolve_implementation`. Its jaxtyping annotations bind exactly like
    /// a same-file helper.
    #[test]
    fn test_cross_file_return_shape_bare_import_propagates() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let code = r#"
from mylib.helpers import project

def f(x: Float[Array, "batch 4"]) -> None:
    y = project(x)
"#;
        let tree = parse(code);
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["batch", "d_out"]))
        );
    }

    /// A relative import resolves next to the document, and only with its
    /// path. A decorated helper is found.
    #[test]
    fn test_cross_file_return_shape_relative_import_of_the_document() {
        let tmp = tempfile::tempdir().unwrap();
        let helper = "import jax\n@jax.jit\ndef project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass";
        fs::create_dir_all(tmp.path().join("pkg/sub")).unwrap();
        fs::write(tmp.path().join("pkg/helpers.py"), helper).unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let cases = [
            ("from .helpers import project", "pkg/main.py", true),
            ("from . import helpers as h", "pkg/main.py", true),
            ("from ..helpers import project", "pkg/sub/main.py", true),
            ("from .helpers import project", "pkg/sub/main.py", false),
            ("from .helpers import project", "main.py", false),
        ];
        for (import, document, found) in cases {
            let call = if import.contains(" as h") { "h.project(x)" } else { "project(x)" };
            let code = format!("{import}\n\ndef f(x: Float[Array, \"batch 4\"]) -> None:\n    y = {call}\n");
            let tree = parse(&code);
            let path = tmp.path().join(document);
            let analysis = with_document(Some(&path), || {
                analyze_layer_shapes(tree.root_node(), &code, &roots, read, 5, None).unwrap()
            });
            let expected = found.then(|| shape(&["batch", "d_out"]));
            assert_eq!(find_shape_in_scope(&analysis, "f", "y"), expected.as_ref(), "{code} in {document}");
            let without = analyze_layer_shapes(tree.root_node(), &code, &roots, read, 5, None).unwrap();
            assert_eq!(find_shape_in_scope(&without, "f", "y"), None, "{code} without a document");
        }
    }

    /// `import mylib.helpers as h` then `y = h.project(x)` — the qualified
    /// (module-attribute) call form, resolved the same way via
    /// `resolve_call_target` before the on-disk walk.
    #[test]
    fn test_cross_file_return_shape_qualified_import_propagates() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let code = r#"
import mylib.helpers as h

def f(x: Float[Array, "batch 4"]) -> None:
    y = h.project(x)
"#;
        let tree = parse(code);
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["batch", "d_out"]))
        );
    }

    /// Argument rank disagrees with the imported helper's declared param
    /// rank — the mismatch must still surface as a `ShapeError`, same as a
    /// same-file helper.
    #[test]
    fn test_cross_file_return_shape_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"a b\"]) -> Float[Array, \"a\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let code = r#"
from mylib.helpers import project

def f(x: Float[Array, "batch"]) -> None:
    y = project(x)
"#;
        let tree = parse(code);
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected one error: {:?}",
            analysis.errors
        );
    }

    /// `max_depth: 0` disables cross-file resolution entirely (mirrors
    /// `resolve_implementation`'s own `max_depth` contract) — the call
    /// silently goes unshaped rather than erroring.
    #[test]
    fn test_cross_file_return_shape_respects_max_depth() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let code = r#"
from mylib.helpers import project

def f(x: Float[Array, "batch 4"]) -> None:
    y = project(x)
"#;
        let tree = parse(code);
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 0, None).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape_in_scope(&analysis, "f", "y"), None);
    }

    /// The signature-extraction cache (`ResolutionCache::signatures`) avoids
    /// re-parsing the callee's file on repeat calls to the same imported
    /// helper, whether within one file or across two files sharing a
    /// session-lifetime cache.
    #[test]
    fn test_cross_file_signature_cache_avoids_reparsing() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("mylib")).unwrap();
        fs::write(
            tmp.path().join("mylib/helpers.py"),
            "def project(x: Float[Array, \"batch d_in\"]) -> Float[Array, \"batch d_out\"]:\n    pass",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let read_count = AtomicUsize::new(0);
        let counting_read = |path: &PathBuf| {
            read_count.fetch_add(1, Ordering::Relaxed);
            fs::read_to_string(path).ok()
        };

        let cache = new_resolution_cache();

        let a_py = r#"
from mylib.helpers import project

def f(x: Float[Array, "batch 4"]) -> None:
    y = project(x)
"#;
        let tree_a = parse(a_py);
        analyze_layer_shapes(tree_a.root_node(), a_py, &roots, counting_read, 5, Some(&cache))
            .unwrap();
        let reads_a = read_count.load(Ordering::Relaxed);
        assert!(reads_a > 0, "first file should hit disk");

        read_count.store(0, Ordering::Relaxed);

        let b_py = r#"
from mylib.helpers import project

def g(x: Float[Array, "batch 4"]) -> None:
    y = project(x)
"#;
        let tree_b = parse(b_py);
        analyze_layer_shapes(tree_b.root_node(), b_py, &roots, counting_read, 5, Some(&cache))
            .unwrap();
        let reads_b = read_count.load(Ordering::Relaxed);

        // The second file reuses both the `ResolutionCache::map` entry
        // (implementation resolution) and the new `signatures` entry
        // (jaxtyping extraction) for the shared `project` target, so it
        // reads strictly less than the first — mirroring
        // `test_resolution_cache_shared_across_files`'s assertion style. It
        // isn't zero because `extract_layer_assignments_scoped` separately
        // (and unconditionally) probes every assigned call as a possible
        // layer constructor via `resolve_call_signature`, which has no
        // signature cache of its own (a pre-existing, unrelated gap).
        assert!(
            reads_b < reads_a,
            "second file should read less than the first (signature cache reuse): \
             reads_a={reads_a} reads_b={reads_b}"
        );
        assert!(
            cache.hits.load(Ordering::Relaxed) >= 1,
            "expected at least one implementation-cache hit after analyzing the second file"
        );
    }
}

mod vmap_shape_inference_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// Helper: find a shape for `var` in the scope named `scope_name`.
    fn find_shape_in_scope<'a>(
        analysis: &'a LayerShapeAnalysis,
        scope_name: &str,
        var: &str,
    ) -> Option<&'a Vec<String>> {
        analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some(scope_name))
            .and_then(|s| s.shapes.get(var))
    }

    /// Test 1: basic jax.vmap shape inference.
    /// vf = jax.vmap(f) applied to a batched input peels axis 0,
    /// applies f's shape rule, then prepends the batch dim.
    #[test]
    fn test_vmap_basic_jax() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "y"),
            Some(&shape(&["B", "m"]))
        );
    }

    /// Test 2: basic equinox.filter_vmap shape inference.
    /// Same as test 1 but with `import equinox as eqx; eqx.filter_vmap(f)`.
    #[test]
    fn test_filter_vmap_basic_equinox() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = eqx.filter_vmap(f)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "y"),
            Some(&shape(&["B", "m"]))
        );
    }

    /// Test 3: vmap with in_axes=1.
    /// Batch dim is peeled from position 1; default out_axes=0 prepends
    /// to front.
    #[test]
    fn test_vmap_in_axes_one() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f, in_axes=1)

def caller(x: Float[Array, "n B"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "y"),
            Some(&shape(&["B", "m"]))
        );
    }

    /// Test 4: vmap with out_axes=1.
    /// Default in_axes=0 peels from front; out_axes=1 inserts batch dim
    /// at position 1.
    #[test]
    fn test_vmap_out_axes_one() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f, out_axes=1)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "y"),
            Some(&shape(&["m", "B"]))
        );
    }

    /// Test 5: multi-arg vmap where batch dims disagree emits an error.
    #[test]
    fn test_vmap_multi_arg_batch_dims_must_match() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"], y: Float[Array, "k"]) -> Float[Array, "n k"]:
    pass

vf = jax.vmap(f)

def caller(a: Float[Array, "B n"], b: Float[Array, "C k"]) -> None:
    z = vf(a, b)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(
            analysis.errors.len(),
            1,
            "expected 1 error, got {:?}",
            analysis.errors
        );
        assert_eq!(analysis.errors[0].variable, "z");
        assert!(
            analysis.errors[0].message.contains("batch dims disagree"),
            "error should mention batch dims disagree: {:?}",
            analysis.errors[0].message
        );
        // No shape recorded for z when there's a mismatch.
        assert_eq!(find_shape_in_scope(&analysis, "caller", "z"), None);
    }

    /// Test 6: wrapped function without annotations — silently skips.
    #[test]
    fn test_vmap_wrapped_function_without_annotations_skips_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x):
    pass

vf = jax.vmap(f)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(find_shape_in_scope(&analysis, "caller", "y"), None);
    }

    /// Test 7: tuple in_axes is skipped silently (not a scalar int).
    #[test]
    fn test_vmap_tuple_in_axes_skipped_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f, in_axes=(0, 1))

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // vf was not recorded in vmap_targets because in_axes=(0,1) isn't
        // a scalar int, so the call `y = vf(x)` falls through — no shape,
        // no error.
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(find_shape_in_scope(&analysis, "caller", "y"), None);
    }

    /// Test 8: non-literal function arg (dotted name) is skipped.
    #[test]
    fn test_vmap_non_literal_function_arg_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        // Pass a dotted name as the wrapped function — not a bare ident.
        // Our parse logic rejects dotted names (contains '.').
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

# module.f is not a bare identifier
vf = jax.vmap(module.f)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // vf was not recorded because "module.f" contains a dot.
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(find_shape_in_scope(&analysis, "caller", "y"), None);
    }

    /// Test 9: arg rank insufficient for in_axes emits an error.
    /// vmap(f) with in_axes=0 on a rank-1 input → peeled to rank 0.
    /// Then f expects rank 1 for param 'x' but gets rank 0.
    #[test]
    fn test_vmap_arg_rank_insufficient_emits_error() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f)

def caller(x: Float[Array, "scalar"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        // The arg has rank 1, in_axes=0 peels axis 0, leaving rank 0.
        // Then f expects rank 1 for param 'x' but gets rank 0.
        // This should produce an error from bind_and_substitute.
        assert_eq!(
            analysis.errors.len(),
            1,
            "expected 1 error, got {:?}",
            analysis.errors
        );
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0].message.contains("f"),
            "error should mention function 'f': {:?}",
            analysis.errors[0].message
        );
        // No shape for y when there's a binding error.
        assert_eq!(find_shape_in_scope(&analysis, "caller", "y"), None);
    }

    /// Test 10: arg variable name matches wrapped function's param name.
    /// def f(x: ...) -> ... and caller uses `vf(x)` where the caller
    /// also names its variable `x`. This previously tripped a
    /// mode-detection heuristic in bind_and_substitute. Now both
    /// callers normalize to (param_name, shape) before calling
    /// the helper, so this case should work correctly.
    #[test]
    fn test_vmap_arg_var_name_matches_wrapped_param_name() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax

def f(x: Float[Array, "n"]) -> Float[Array, "m"]:
    pass

vf = jax.vmap(f)

def caller(x: Float[Array, "B n"]) -> None:
    y = vf(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "caller", "y"),
            Some(&shape(&["B", "m"]))
        );
    }
}

mod recursive_evaluator_tests {
    use super::*;

    // Tests routed through analyze_layer_shapes (the public API)
    // to catch wiring regressions and the synthetic-name-leak class of bug.

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn find_shape_in_scope<'a>(
        analysis: &'a LayerShapeAnalysis,
        scope_name: &str,
        var: &str,
    ) -> Option<&'a Vec<String>> {
        analysis
            .scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some(scope_name))
            .and_then(|s| s.shapes.get(var))
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    #[test]
    fn test_nested_call_shape_error_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        // a @ b with mismatched inner dims should produce a matmul error
        let code = r#"
import jax.numpy as jnp
def f(a: Float[Array, "3 5"], b: Float[Array, "4 2"]):
    y = jnp.exp(a @ b)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            analysis.errors.len(),
            1,
            "expected exactly one error, got {:?}",
            analysis.errors
        );
        assert!(
            analysis.errors[0].message.contains("matmul dimension mismatch"),
            "unexpected error message: {}",
            analysis.errors[0].message
        );
    }

    #[test]
    fn test_bare_call_shape_error_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"], y: Float[Array, "4 2"]):
    jnp.matmul(x, y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            analysis.errors.len(),
            1,
            "expected exactly one error, got {:?}",
            analysis.errors
        );
        assert!(
            analysis.errors[0].message.contains("matmul dimension mismatch"),
            "unexpected error message: {}",
            analysis.errors[0].message
        );
        // No LHS — variable should be empty.
        assert_eq!(analysis.errors[0].variable, "");
    }

    #[test]
    fn test_bare_call_compatible_shapes_no_error_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"], y: Float[Array, "5 2"]):
    jnp.matmul(x, y)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
    }

    #[test]
    fn test_no_synth_keys_in_scopes_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jnp.exp(jnp.reshape(x, (5, 3)))
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        for scope in &analysis.scopes {
            for key in scope.shapes.keys() {
                assert!(
                    !key.starts_with("__synth_"),
                    "__synth_* key leaked into scope.shapes: {}",
                    key
                );
            }
        }
        // Sanity: y should have the correct shape
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["5", "3"]))
        );
    }

    #[test]
    fn test_chained_method_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "3 4"]):
    y = x.reshape(3, 4).sum(axis=1)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["3"]))
        );
    }

    #[test]
    fn test_self_attribute_direct_field_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
class M:
    A_log: Float[Array, "d_inner d_state"]

    def __call__(self, x: Float[Array, "seq d_inner"]):
        B = self.A_log
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "B"),
            Some(&shape(&["d_inner", "d_state"]))
        );
    }

    #[test]
    fn test_self_attribute_through_astype_and_unary_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        // Mirrors the Mamba SelectiveStateSpace repro in issue #31.
        let code = r#"
import jax.numpy as jnp
class M:
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, "d_inner"]

    def __call__(self, x: Float[Array, "seq d_inner"]):
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "A"),
            Some(&shape(&["d_inner", "d_state"]))
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "D"),
            Some(&shape(&["d_inner"]))
        );
    }

    #[test]
    fn test_qualified_free_function_still_resolves_after_self_attr() {
        // Regression: the self-attribute receiver path must fall through for
        // qualified free functions like jax.nn.softplus(x).
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
def f(x: Float[Array, "3 5"]):
    y = jax.nn.softplus(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_direct_self_attr_layer_call() {
        // Direct `self.fc(x)` application (no vmap) — the common forward style.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx
class M(eqx.Module):
    fc: eqx.nn.Linear

    def __init__(self, key):
        self.fc = eqx.nn.Linear(4, 7, key=key)

    def __call__(self, x: Float[Array, "batch 4"]):
        y = self.fc(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "errors: {:?}", analysis.errors);
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "y"),
            Some(&shape(&["batch", "7"]))
        );
    }

    #[test]
    fn test_direct_self_attr_conv_through_activation() {
        // `jax.nn.relu(self.conv(x))` — direct call nested inside an activation.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import equinox as eqx
class M(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, key):
        self.conv = eqx.nn.Conv2d(3, 16, 3, key=key)

    def __call__(self, x: Float[Array, "3 32 32"]):
        h = jax.nn.relu(self.conv(x))
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "h"),
            Some(&shape(&["16", "30", "30"]))
        );
    }

    #[test]
    fn test_fused_qkv_split_factor_cancellation() {
        // Linear(d, 3*d) then split(qkv, 3) must cancel to `d`, so the later
        // proj layer's in_features matches.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import jax.numpy as jnp
import equinox as eqx
class M(eqx.Module):
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, d, key):
        self.qkv = eqx.nn.Linear(d, d * 3, key=key)
        self.proj = eqx.nn.Linear(d, d, key=key)

    def __call__(self, x: Float[Array, "seq d"]):
        qkv = jax.vmap(self.qkv)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        out = jax.vmap(self.proj)(v)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "errors: {:?}", analysis.errors);
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "v"),
            Some(&shape(&["seq", "d"]))
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "out"),
            Some(&shape(&["seq", "d"]))
        );
    }

    #[test]
    fn test_inline_vmap_self_attr_layers_stay_class_scoped() {
        // The same attribute name in two classes must use each class's layer.
        // A file-global last-wins lookup makes class A use class B's dimensions.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import equinox as eqx
class A(eqx.Module):
    qkv: eqx.nn.Linear

    def __init__(self, key):
        self.qkv = eqx.nn.Linear(4, 12, key=key)

    def run_a(self, x: Float[Array, "seq_a 4"]):
        y = jax.vmap(self.qkv)(x)

class B(eqx.Module):
    qkv: eqx.nn.Linear

    def __init__(self, key):
        self.qkv = eqx.nn.Linear(8, 24, key=key)

    def run_b(self, x: Float[Array, "seq_b 8"]):
        y = jax.vmap(self.qkv)(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "errors: {:?}", analysis.errors);
        assert_eq!(
            find_shape_in_scope(&analysis, "run_a", "y"),
            Some(&shape(&["seq_a", "12"]))
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "run_b", "y"),
            Some(&shape(&["seq_b", "24"]))
        );
    }

    #[test]
    fn test_inline_vmap_layer_error_uses_source_names() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import equinox as eqx
class M(eqx.Module):
    qkv: eqx.nn.Linear

    def __init__(self, key):
        self.qkv = eqx.nn.Linear(8, 16, key=key)

    def run(self, x: Float[Array, "seq 7"]):
        y = jax.vmap(self.qkv)(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

        assert_eq!(analysis.errors.len(), 1, "errors: {:?}", analysis.errors);
        assert_eq!(
            analysis.errors[0].message,
            "Linear layer 'self.qkv' expected input last dim 8, got 7 for 'x'"
        );
    }

    fn inherit_h(code: &str, scope: &str) -> Option<Vec<String>> {
        let tmp = tempfile::tempdir().unwrap();
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "errors: {:?}", analysis.errors);
        find_shape_in_scope(&analysis, scope, "h").cloned()
    }

    #[test]
    fn test_inherited_self_attr_layer_with_unrelated_same_name_class() {
        let code = r#"
import torch.nn as nn
class Base(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(96)
class Child(Base):
    def forward(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
class Other(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(32)
"#;
        assert_eq!(inherit_h(code, "forward"), Some(shape(&["b", "96"])));
    }

    #[test]
    fn test_inherited_self_attr_layer_multi_level() {
        let code = r#"
import torch.nn as nn
class G(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(96)
class Mid(G):
    pass
class Child(Mid):
    def forward(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
class Other(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(32)
"#;
        assert_eq!(inherit_h(code, "forward"), Some(shape(&["b", "96"])));
    }

    #[test]
    fn test_inherited_self_attr_layer_follows_mro_order() {
        let code = r#"
import torch.nn as nn
class A(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(8)
class B(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(16)
class C(A, B):
    def forward(self, x: Float[Array, "b 8"]):
        h = self.norm(x)
class D(B, A):
    def forward2(self, x: Float[Array, "b 16"]):
        h = self.norm(x)
"#;
        assert_eq!(inherit_h(code, "forward"), Some(shape(&["b", "8"])));
        assert_eq!(inherit_h(code, "forward2"), Some(shape(&["b", "16"])));
    }

    #[test]
    fn test_inherited_self_attr_layer_diamond_uses_c3() {
        let code = r#"
import torch.nn as nn
class Top(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(1)
class L(Top):
    pass
class R(Top):
    def __init__(self):
        self.norm = nn.LayerNorm(2)
class D(L, R):
    def forward(self, x: Float[Array, "b 2"]):
        h = self.norm(x)
"#;
        assert_eq!(inherit_h(code, "forward"), Some(shape(&["b", "2"])));
    }

    #[test]
    fn test_child_override_of_base_self_attr_layer() {
        let code = r#"
import torch.nn as nn
class Base(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(96)
class Child(Base):
    def __init__(self):
        self.norm = nn.LayerNorm(32)
    def forward(self, x: Float[Array, "b 32"]):
        h = self.norm(x)
"#;
        assert_eq!(inherit_h(code, "forward"), Some(shape(&["b", "32"])));
    }

    #[test]
    fn test_unresolved_base_keeps_self_attr_layer_unknown() {
        let code = r#"
import torch.nn as nn
from lib import Foreign
class Base(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(96)
class C1(Foreign):
    def forward(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
class C2(Foreign, Base):
    def forward2(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
class C3(Base, Foreign):
    def forward3(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
"#;
        assert_eq!(inherit_h(code, "forward"), None);
        assert_eq!(inherit_h(code, "forward2"), None);
        assert_eq!(inherit_h(code, "forward3"), Some(shape(&["b", "96"])));
    }

    #[test]
    fn test_unrelated_class_does_not_borrow_lone_self_attr_layer() {
        let code = r#"
import torch.nn as nn
class Other(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(32)
class Child(nn.Module):
    def forward(self, x: Float[Array, "b 96"]):
        h = self.norm(x)
"#;
        assert_eq!(inherit_h(code, "forward"), None);
    }

    #[test]
    fn test_inherited_self_attr_alias_with_unrelated_same_name_class() {
        let code = r#"
import jax.numpy as jnp
import equinox as eqx
class Base(eqx.Module):
    def __init__(self, dt_rank):
        self.rank = dt_rank
class Child(Base):
    def run(self, x: Float[Array, "seq combined"]):
        h, rest = jnp.split(x, [self.rank], axis=-1)
class Other(eqx.Module):
    def __init__(self, other_rank):
        self.rank = other_rank
"#;
        assert_eq!(inherit_h(code, "run"), Some(shape(&["seq", "dt_rank"])));
    }

    #[test]
    fn test_symbolic_dim_normalization_self_attr_alias() {
        // `self.dt_rank` (from a split index) and `dt_rank` (the Linear
        // in_features) are the same value via `self.dt_rank = dt_rank`. Without
        // normalization this mismatches; with it, the vmap layer resolves.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import jax.numpy as jnp
import equinox as eqx
class M(eqx.Module):
    proj: eqx.nn.Linear
    dt_rank: int

    def __init__(self, dt_rank, key):
        self.proj = eqx.nn.Linear(dt_rank, 8, key=key)
        self.dt_rank = dt_rank

    def __call__(self, x: Float[Array, "seq combined"]):
        delta, rest = jnp.split(x, [self.dt_rank], axis=-1)
        y = jax.vmap(self.proj)(delta)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(
            analysis.errors.is_empty(),
            "symbolic dim mismatch should be normalized away: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "y"),
            Some(&shape(&["seq", "8"]))
        );
        // The split output dim is canonicalized to the bare identifier.
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "delta"),
            Some(&shape(&["seq", "dt_rank"]))
        );
    }

    #[test]
    fn test_symbolic_dim_normalization_class_scoped_no_cross_class_collision() {
        // Two classes both alias `self.rank` (to different identifiers).
        // With a file-global (last-wins) alias map, class A's `self.rank`
        // would incorrectly resolve using class B's alias (whichever class
        // is parsed last), normalizing `delta` in A to the wrong identifier.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
import jax.numpy as jnp
import equinox as eqx
class A(eqx.Module):
    proj: eqx.nn.Linear
    rank: int

    def __init__(self, dt_rank, key):
        self.proj = eqx.nn.Linear(dt_rank, 8, key=key)
        self.rank = dt_rank

    def run_a(self, x: Float[Array, "seq combined"]):
        delta, rest = jnp.split(x, [self.rank], axis=-1)
        y = jax.vmap(self.proj)(delta)

class B(eqx.Module):
    proj: eqx.nn.Linear
    rank: int

    def __init__(self, other_rank, key):
        self.proj = eqx.nn.Linear(other_rank, 8, key=key)
        self.rank = other_rank

    def run_b(self, x: Float[Array, "seq combined"]):
        delta, rest = jnp.split(x, [self.rank], axis=-1)
        y = jax.vmap(self.proj)(delta)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "errors: {:?}", analysis.errors);

        // Each class's `delta` normalizes to its OWN class's alias, not the
        // other class's.
        assert_eq!(
            find_shape_in_scope(&analysis, "run_a", "delta"),
            Some(&shape(&["seq", "dt_rank"]))
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "run_b", "delta"),
            Some(&shape(&["seq", "other_rank"]))
        );
    }

    #[test]
    fn test_subscript_integer_drops_axis() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"seq d\"]):\n    y = x[0]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(find_shape_in_scope(&analysis, "f", "y"), Some(&shape(&["d"])));
    }

    #[test]
    fn test_subscript_integer_on_rank1_is_scalar() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"n\"]):\n    y = x[0]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&Vec::<String>::new())
        );
    }

    #[test]
    fn test_subscript_slice_stop() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"seq hidden\"]):\n    y = x[:, :3]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(find_shape_in_scope(&analysis, "f", "y"), Some(&shape(&["seq", "min(3, hidden)"])));
    }

    #[test]
    fn test_subscript_numeric_slice_folds() {
        let tmp = tempfile::tempdir().unwrap();
        // x[1:3, 0]: axis0 -> 3-1=2, axis1 integer -> dropped.
        let code = "def f(x: Float[Array, \"10 5\"]):\n    y = x[1:3, 0]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(find_shape_in_scope(&analysis, "f", "y"), Some(&shape(&["2"])));
    }

    #[test]
    fn test_subscript_ellipsis_newaxis() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"a b\"]):\n    y = x[..., None]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["a", "b", "1"]))
        );
    }

    #[test]
    fn test_subscript_newaxis_middle() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"a b\"]):\n    y = x[:, None]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["a", "1", "b"]))
        );
    }

    #[test]
    fn test_subscript_full_slice_preserves() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"seq d\"]):\n    y = x[:]\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["seq", "d"]))
        );
    }

    #[test]
    fn test_subscript_feeds_downstream() {
        // A sliced result must carry a shape into the next op (the whole point
        // of subscript support: stop blackholing the rest of the function).
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"seq d\"]):\n    row = x[0]\n    z = jnp.exp(row)\n";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(find_shape_in_scope(&analysis, "f", "z"), Some(&shape(&["d"])));
    }

    #[test]
    fn test_assignment_shapes_record_each_reassignment() {
        // Reassigning x must yield one record per assignment line (issue #28),
        // unlike scope.shapes which keeps only the final shape.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(a: Float[Array, "3 5"], b: Float[Array, "5 2"]):
    x = a @ b
    x = jnp.reshape(x, (6,))
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        let xs: Vec<_> = analysis
            .assignment_shapes
            .iter()
            .filter(|r| r.name == "x")
            .collect();
        assert_eq!(
            xs.len(),
            2,
            "expected two records for x: {:?}",
            analysis.assignment_shapes
        );
        assert_eq!(xs[0].shape, shape(&["3", "2"]));
        assert_eq!(xs[1].shape, shape(&["6"]));
        assert_ne!(xs[0].line, xs[1].line);
    }

    #[test]
    fn test_assignment_shapes_skip_annotated() {
        // Annotated assignments shouldn't produce inlay records (the user
        // already wrote the type).
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(a: Float[Array, "3 5"], b: Float[Array, "5 2"]):
    c: Float[Array, "3 2"] = a @ b
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(
            analysis.assignment_shapes.iter().all(|r| r.name != "c"),
            "annotated assignment leaked into inlay records: {:?}",
            analysis.assignment_shapes
        );
    }

    #[test]
    fn test_tuple_unpack_split_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "10 6"]):
    a, b = jnp.split(x, 2, axis=1)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "a"),
            Some(&shape(&["10", "3"]))
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "b"),
            Some(&shape(&["10", "3"]))
        );
    }

    #[test]
    fn test_tuple_unpack_split_chained_source_order_via_analyze() {
        // The split input comes from an earlier assignment and a later
        // assignment consumes a split output — verifies interleaved ordering.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "10 6"]):
    y = jnp.exp(x)
    a, b = jnp.split(y, 2, axis=1)
    z = jnp.exp(a)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "z"),
            Some(&shape(&["10", "3"]))
        );
    }

    #[test]
    fn test_tuple_unpack_shape_attribute_via_analyze() {
        // L, _ = x.shape — L is an integer dim (zero-rank), _ is skipped.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
def f(x: Float[Array, "seq d"]):
    L, _ = x.shape
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "L"),
            Some(&Vec::<String>::new())
        );
        assert_eq!(find_shape_in_scope(&analysis, "f", "_"), None);
    }

    #[test]
    fn test_inline_vmap_of_self_attr_layer_via_analyze() {
        // jax.vmap(self.input_proj)(x): self.input_proj is a Linear built in
        // __init__; vmap maps it over x's leading axis. (issue #35)
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx
import jax
class M:
    input_proj: eqx.nn.Linear

    def __init__(self):
        self.input_proj = eqx.nn.Linear(4, 7)

    def __call__(self, x: Float[Array, "seq 4"]):
        y = jax.vmap(self.input_proj)(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "unexpected errors: {:?}", analysis.errors);
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "y"),
            Some(&shape(&["seq", "7"]))
        );
    }

    #[test]
    fn test_inline_vmap_of_self_attr_layer_mismatch_via_analyze() {
        // x's per-element last dim (5) disagrees with Linear in_features (4).
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx
import jax
class M:
    input_proj: eqx.nn.Linear

    def __init__(self):
        self.input_proj = eqx.nn.Linear(4, 7)

    def __call__(self, x: Float[Array, "seq 5"]):
        y = jax.vmap(self.input_proj)(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(analysis.errors.len(), 1, "expected one error: {:?}", analysis.errors);
    }

    #[test]
    fn test_inline_vmap_of_bare_function_via_analyze() {
        // jax.vmap(g)(x) with g a user function — inline form, no intermediate
        // binding. Reuses the existing vmap-call expansion.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
def g(v: Float[Array, "d"]) -> Float[Array, "d"]:
    return v

def f(x: Float[Array, "batch d"]):
    y = jax.vmap(g)(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["batch", "d"]))
        );
    }

    #[test]
    fn test_nested_vmap_of_self_attr_layer_via_analyze() {
        // jax.vmap(jax.vmap(self.input_proj))(x): two leading batch axes
        // (batch, seq) get peeled before applying the Linear layer to the
        // per-element "features" axis, then both are re-prepended.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx
import jax
class M:
    input_proj: eqx.nn.Linear

    def __init__(self):
        self.input_proj = eqx.nn.Linear(4, 7)

    def __call__(self, x: Float[Array, "batch seq 4"]):
        y = jax.vmap(jax.vmap(self.input_proj))(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(analysis.errors.is_empty(), "unexpected errors: {:?}", analysis.errors);
        assert_eq!(
            find_shape_in_scope(&analysis, "__call__", "y"),
            Some(&shape(&["batch", "seq", "7"]))
        );
    }

    #[test]
    fn test_nested_vmap_of_self_attr_layer_mismatch_via_analyze() {
        // Same as above but the per-element last dim (5) disagrees with the
        // Linear's in_features (4) — the mismatch must still surface after
        // peeling both batch axes.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import equinox as eqx
import jax
class M:
    input_proj: eqx.nn.Linear

    def __init__(self):
        self.input_proj = eqx.nn.Linear(4, 7)

    def __call__(self, x: Float[Array, "batch seq 5"]):
        y = jax.vmap(jax.vmap(self.input_proj))(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(analysis.errors.len(), 1, "expected one error: {:?}", analysis.errors);
    }

    #[test]
    fn test_nested_vmap_of_bare_function_via_analyze() {
        // jax.vmap(jax.vmap(g))(x) with g a user function — two batch axes
        // peeled around g's single-example "d"-shaped input/output.
        let tmp = tempfile::tempdir().unwrap();
        let code = r#"
import jax
def g(v: Float[Array, "d"]) -> Float[Array, "d"]:
    return v

def f(x: Float[Array, "batch seq d"]):
    y = jax.vmap(jax.vmap(g))(x)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["batch", "seq", "d"]))
        );
    }

    #[test]
    fn test_binary_op_in_nested_call_via_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        // z = jnp.exp(a @ b) — compatible matmul inside nested call
        let code = r#"
import jax.numpy as jnp
def f(a: Float[Array, "3 5"], b: Float[Array, "5 2"]):
    z = jnp.exp(a @ b)
"#;
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();
        assert!(
            analysis.errors.is_empty(),
            "unexpected errors: {:?}",
            analysis.errors
        );
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "z"),
            Some(&shape(&"3 2".split(' ').collect::<Vec<_>>()))
        );
    }
}

#[cfg(test)]
mod scan_tuple_carry_corpus_mirror_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// End-to-end mirror of corpus/lax_scan_ema.py's `running_ema`: a
    /// homogeneous `(mean, var)` tuple carry threaded through
    /// `jax.lax.scan` via an aliased variable (`init = (mean0, var0)`),
    /// then re-destructured from the flat scan-result binding
    /// (`final_mean, final_var = final_carry`). Lazy call-site parameter
    /// seeding now shapes `ema_step`'s internal body too: `carry` is seeded
    /// from `init`'s shape, so `mean, var = carry` resolves, and `means`
    /// (`ys`) is traced from the body's second return element
    /// (`new_mean`) with `xs`'s leading axis prepended.
    #[test]
    fn test_running_ema_tuple_carry_mirror() {
        let code = "\
import jax
import jax.numpy as jnp

def ema_step(carry, x: Float[Array, \"features\"], decay: float = 0.9):
    mean, var = carry
    delta = x - mean
    new_mean = mean + (1 - decay) * delta
    new_var = decay * var + (1 - decay) * delta * delta
    return (new_mean, new_var), new_mean

def running_ema(
    xs: Float[Array, \"seq features\"],
    mean0: Float[Array, \"features\"],
    var0: Float[Array, \"features\"],
):
    init = (mean0, var0)
    final_carry, means = jax.lax.scan(ema_step, init, xs)
    final_mean, final_var = final_carry
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "init"), None);
        assert_eq!(find_shape(&analysis, "final_carry"), None);
        assert_eq!(find_shape(&analysis, "final_mean"), Some(&shape(&["features"])));
        assert_eq!(find_shape(&analysis, "final_var"), Some(&shape(&["features"])));
        // `means` (ys): xs's leading "seq" axis prepended to the body's
        // per-step `new_mean` output.
        assert_eq!(
            find_shape(&analysis, "means"),
            Some(&shape(&["seq", "features"]))
        );
        // Seeded carry destructuring inside the body now resolves.
        assert_eq!(find_shape(&analysis, "mean"), Some(&shape(&["features"])));
    }

    /// End-to-end mirror of corpus/eqx_lstm_gru_scan.py's `LSTMEncoder`:
    /// an inline `(h0, c0)` tuple carry through `jax.lax.scan`, plus the
    /// direct single-step `self.lstm(x, (h0, c0))` call outside the scan
    /// (already worked before this change). Lazy call-site parameter
    /// seeding now shapes the nested `body` closure too: `carry` is seeded
    /// from `(h0, c0)`'s shape, so `h, c = carry` resolves, and `hs` is
    /// traced from the body's second return element (`h_new`) with
    /// `encoded`'s leading axis prepended.
    #[test]
    fn test_lstm_encoder_tuple_carry_mirror() {
        let code = "\
import jax
import equinox as eqx

class LSTMEncoder(eqx.Module):
    encoder: eqx.nn.MLP
    lstm: eqx.nn.LSTMCell

    def __init__(self, in_size, hidden, key):
        self.encoder = eqx.nn.MLP(in_size, hidden, width_size=hidden, depth=2, key=key)
        self.lstm = eqx.nn.LSTMCell(hidden, hidden, key=key)

    def __call__(
        self,
        xs: Float[Array, \"seq in_size\"],
        h0: Float[Array, \"hidden\"],
        c0: Float[Array, \"hidden\"],
    ):
        encoded = jax.vmap(self.encoder)(xs)
        step_h = self.lstm(encoded[0], (h0, c0))

        def body(carry, x):
            h, c = carry
            h_new = self.lstm(x, (h, c))
            return (h_new, c), h_new

        carry_final, hs = jax.lax.scan(body, (h0, c0), encoded)
        h_final, c_final = carry_final
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        // `eqx.nn.LSTMCell` gives the tuple `(h, c)`, so `step_h`, `h_new`,
        // and `hs`, which the scan builds from `h_new`, are no arrays. The
        // final carry keeps the shape of the init carry.
        assert_eq!(find_shape(&analysis, "step_h"), None);
        assert_eq!(find_shape(&analysis, "carry_final"), None);
        assert_eq!(find_shape(&analysis, "h_final"), Some(&shape(&["hidden"])));
        assert_eq!(find_shape(&analysis, "c_final"), Some(&shape(&["hidden"])));
        assert_eq!(find_shape(&analysis, "hs"), None);
        // Seeded carry destructuring inside the nested `body` now resolves.
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["hidden"])));
    }
}

/// Corpus-driven shape-coverage harness. Runs the analyzer over real model
/// files in `corpus/` and ranks the assignments it cannot shape ("dark
/// spots") by frequency, so the highest-impact gaps prioritize themselves.
/// Run with: `cargo test corpus_coverage_report -- --nocapture`.
#[cfg(test)]
mod coverage_harness {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(_path: &PathBuf) -> Option<String> {
        None
    }

    #[test]
    fn corpus_coverage_report() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus");
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "py").unwrap_or(false))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no corpus files in {:?}", dir);

        let roots: Vec<PathBuf> = Vec::new();
        let mut total = 0usize;
        let mut shaped = 0usize;
        let mut buckets: BTreeMap<String, usize> = BTreeMap::new();
        let mut per_file: Vec<(String, usize, usize)> = Vec::new();
        let mut details: Vec<(String, u32, String)> = Vec::new();

        for path in &files {
            let code = fs::read_to_string(path).unwrap();
            let tree = parse(&code);
            let report =
                analyze_coverage(tree.root_node(), &code, &roots, read, 5, None).unwrap();
            total += report.total;
            shaped += report.shaped;
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            for d in &report.dark {
                *buckets.entry(d.kind.clone()).or_default() += 1;
                details.push((name.clone(), d.line + 1, d.kind.clone()));
            }
            per_file.push((name, report.shaped, report.total));
        }

        let mut ranked: Vec<(String, usize)> = buckets.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let pct = |s: usize, t: usize| 100.0 * (s as f64) / (t.max(1) as f64);
        println!("\n=== corpus shape coverage ===");
        for (name, s, t) in &per_file {
            println!("  {:<22} {}/{} ({:.0}%)", name, s, t, pct(*s, *t));
        }
        println!(
            "  {:<22} {}/{} ({:.0}%)",
            "TOTAL",
            shaped,
            total,
            pct(shaped, total)
        );
        println!("\n=== dark spots by frequency ===");
        for (kind, n) in &ranked {
            println!("  {:>3}  {}", n, kind);
        }
        if !details.is_empty() {
            println!("\n=== dark spots (detail) ===");
            for (file, line, kind) in &details {
                println!("  {}:{}  {}", file, line, kind);
            }
        }
        println!();

        // Regression floor: coverage shouldn't silently drop. Tighten as gaps close.
        // Raised from 85 to 90 after lazy call-site parameter seeding closed
        // the last 9 dark spots (corpus is 176/176 = 100% as of this wave).
        assert!(
            pct(shaped, total) >= 90.0,
            "corpus coverage regressed to {:.0}%",
            pct(shaped, total)
        );
    }
}

#[cfg(test)]
mod augmented_assignment_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_plus_equals_broadcast_keeps_shape() {
        let code = "def f(x: Float[Array, \"batch f\"], bias: Float[Array, \"f\"]):\n    x += bias";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "f"])));
        assert!(
            analysis
                .assignment_shapes
                .iter()
                .any(|r| r.name == "x" && r.shape == shape(&["batch", "f"]))
        );
    }

    #[test]
    fn test_plus_equals_incompatible_errors() {
        let code =
            "def f(x: Float[Array, \"batch f\"], y: Float[Array, \"batch g\"]):\n    x += y";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "x");
    }

    #[test]
    fn test_matmul_equals_updates_shape() {
        let code = "def f(h: Float[Array, \"batch d1\"], w: Float[Array, \"d1 d2\"]):\n    h @= w";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["batch", "d2"])));
    }

    #[test]
    fn test_times_equals_scalar_literal_skips_silently() {
        let code = "def f(x: Float[Array, \"batch f\"]):\n    x *= 2";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "f"])));
    }

    #[test]
    fn test_unproven_right_side_evicts_the_shape() {
        for (op, rhs) in [("@=", "w"), ("+=", "w"), ("*=", "w"), ("@=", "2")] {
            let code = format!("def f(x: Float[Array, \"n 8\"], w):\n    x {op} {rhs}\n    y = x\n");
            let analysis = analyze(&code);
            assert!(analysis.errors.is_empty(), "{op} {rhs}");
            assert_eq!(find_shape(&analysis, "y"), None, "{op} {rhs}");
        }
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"a 8\"]):\n    n, m = x.shape\n    n += 1\n    z = jnp.zeros((n, m))\n";
        assert_ne!(find_shape(&analyze(code), "z"), Some(&shape(&["a", "8"])));
    }

    #[test]
    fn test_plus_equals_with_call_rhs() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"batch f\"], y: Float[Array, \"batch f\"]):\n    x += jnp.exp(y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "f"])));
    }
}

#[cfg(test)]
mod embedding_layer_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_torch_embedding_appends_dim() {
        let code = "import torch\ndef f(tokens: Int[Array, \"batch seq\"]):\n    emb = torch.nn.Embedding(10000, 512)\n    y = emb(tokens)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "seq", "512"]))
        );
    }

    #[test]
    fn test_equinox_embedding_appends_dim() {
        let code = "import equinox as eqx\ndef f(tokens: Int[Array, \"seq\"]):\n    emb = eqx.nn.Embedding(10000, 512)\n    y = emb(tokens)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["seq", "512"])));
    }
}

#[cfg(test)]
mod corpus_no_false_errors {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tree_sitter::Parser;

    /// Corpus files are valid model code — analysis must not publish
    /// diagnostics for any of them. Dark spots (no shape) are fine;
    /// errors are not.
    #[test]
    fn corpus_files_produce_no_errors() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus");
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("py") {
                continue;
            }
            let code = fs::read_to_string(&path).unwrap();
            let tree = parser.parse(&code, None).unwrap();
            let analysis =
                analyze_layer_shapes(tree.root_node(), &code, &[], |_| None, 5, None).unwrap();
            assert!(
                analysis.errors.is_empty(),
                "{:?} produced errors: {:?}",
                path.file_name().unwrap(),
                analysis.errors
            );
        }
    }
}

#[cfg(test)]
mod symbolic_ctor_dim_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    /// Issue #47 repro: concat-into-Linear gate where the Linear's in_features
    /// text ("input_size + hidden_size") can never string-match the
    /// concatenated annotation dims ("hidden+features"). Must produce no
    /// error and propagate the layer's output width.
    #[test]
    fn test_concat_into_symbolic_linear_no_false_error() {
        let code = "import jax\nimport jax.numpy as jnp\nimport equinox as eqx\n\nclass GRUCell(eqx.Module):\n    wz: eqx.nn.Linear\n\n    def __init__(self, input_size, hidden_size, key):\n        self.wz = eqx.nn.Linear(input_size + hidden_size, hidden_size, key=key)\n\n    def step(self, h: Float[Array, \"hidden\"], x: Float[Array, \"features\"]):\n        hx = jnp.concatenate([h, x])\n        z = jax.nn.sigmoid(self.wz(hx))\n        return z";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(
            find_shape(&analysis, "z"),
            Some(&vec!["hidden_size".to_string()])
        );
    }

    /// A user name with the text of a mark (`__free_`, `__is_`) cannot be
    /// told apart from the mark, so its dim is unknown, never an integer.
    #[test]
    fn a_name_with_the_text_of_a_mark_gives_no_dim() {
        let module = |init: &str, ctor: &str| {
            format!("import torch.nn as nn\n{init}class M(nn.Module):\n    def __init__(self, n__is_3, __free_k):\n        self.l = nn.Linear(8, {ctor})\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        h = self.l(x)\n        z = h + y\n")
        };
        let function = |params: &str, size: &str| {
            format!("import jax.numpy as jnp\ndef f({params}, y: Float[Array, \"4\"]):\n    h = jnp.zeros(({size},))\n    z = h + y\n")
        };
        for code in [
            module("", "n__is_3"),
            module("", "__free_k"),
            module("__free_OUT = 4\n", "__free_OUT"),
            module("__free_OUT__is_3 = 4\n", "__free_OUT__is_3"),
            function("n__is_3", "n__is_3"),
            function("__free_n__is_3", "__free_n__is_3"),
            "def f(h: Float[Array, \"__free_n__is_3\"], y: Float[Array, \"4\"]):\n    z = h + y\n".to_string(),
        ] {
            let tree = parse(&code);
            let analysis = analyze_layer_shapes(tree.root_node(), &code, &[], |_| None, 5, None).unwrap();
            assert!(analysis.errors.is_empty(), "{code}: {:?}", analysis.errors);
            let h = find_shape(&analysis, "h");
            assert!(h.is_none_or(|h| h == &["?"]), "{code}: {h:?}");
        }
    }
}

#[cfg(test)]
mod pooling_layer_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// End-to-end mirror of corpus/cnn_pool.py's forward pass.
    #[test]
    fn test_cnn_with_pooling_forward() {
        let code = "import torch\nimport torch.nn as nn\ndef f(x: Float[Array, \"3 32 32\"]):\n    conv1 = nn.Conv2d(3, 16, 3, padding=1)\n    pool1 = nn.MaxPool2d(2)\n    conv2 = nn.Conv2d(16, 32, 3, padding=1)\n    pool2 = nn.AdaptiveAvgPool2d(1)\n    fc = nn.Linear(32, 10)\n    h = conv1(x)\n    h = pool1(h)\n    h = conv2(h)\n    h = pool2(h)\n    pooled = h.mean(axis=(1, 2))\n    logits = fc(pooled)\n    return logits";
        let tree = parse(code);

        let analysis =
            analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["32", "1", "1"])));
        assert_eq!(find_shape(&analysis, "pooled"), Some(&shape(&["32"])));
        assert_eq!(find_shape(&analysis, "logits"), Some(&shape(&["10"])));
    }
}

#[cfg(test)]
mod literal_binop_scan_method_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_binop_scalar_literal_keeps_array_shape() {
        let code = "def f(h: Float[Array, \"batch d\"]):\n    normed = h / 2.0\n    scaled = 3 * h";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "normed"), Some(&shape(&["batch", "d"])));
        assert_eq!(find_shape(&analysis, "scaled"), Some(&shape(&["batch", "d"])));
    }

    #[test]
    fn test_binop_nested_gate_expression() {
        let code = "def f(z: Float[Array, \"d\"], h: Float[Array, \"d\"], t: Float[Array, \"d\"]):\n    h_new = (1 - z) * h + z * t";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "h_new"), Some(&shape(&["d"])));
    }

    #[test]
    fn test_binop_nested_mismatch_still_errors() {
        let code = "def f(z: Float[Array, \"d1\"], h: Float[Array, \"d2\"]):\n    bad = (1 - z) * h";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
    }

    #[test]
    fn test_scan_binds_final_carry_from_init() {
        // `body`'s params are un-annotated, but lazy call-site parameter
        // seeding shapes it anyway: `c` <- `h0`'s shape, so `return c, c`
        // traces `hs` to `c`'s (seeded) shape with `xs`'s leading axis
        // prepended.
        let code = "import jax\ndef f(h0: Float[Array, \"hidden\"], xs: Float[Array, \"seq features\"]):\n    def body(c, x):\n        return c, c\n    h_final, hs = jax.lax.scan(body, h0, xs)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "h_final"), Some(&shape(&["hidden"])));
        assert_eq!(find_shape(&analysis, "hs"), Some(&shape(&["seq", "hidden"])));
    }

    #[test]
    fn test_scan_init_keyword() {
        let code = "import jax\ndef f(h0: Float[Array, \"hidden\"], xs: Float[Array, \"seq features\"]):\n    def body(c, x):\n        return c, c\n    h_final, hs = jax.lax.scan(body, init=h0, xs=xs)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "h_final"), Some(&shape(&["hidden"])));
    }

    #[test]
    fn test_scan_inline_tuple_carry_homogeneous_shape() {
        // `(h0, c0)` inline at the call site — both same shape, so the
        // carry is representable as that one shape (`shape_of_tuple`).
        let code = "import jax\ndef f(h0: Float[Array, \"hidden\"], c0: Float[Array, \"hidden\"], xs: Float[Array, \"seq hidden\"]):\n    def body(carry, x):\n        return carry, x\n    final_carry, ys = jax.lax.scan(body, (h0, c0), xs)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "final_carry"), None);
    }

    #[test]
    fn test_scan_aliased_tuple_literal_carry() {
        // `init = (mean0, var0)` bound earlier, then passed by name —
        // resolves the same way as the inline form since `init`'s own
        // assignment already stored the homogeneous shape. `step`'s
        // un-annotated params are seeded from the call site: `x` <-
        // `xs`'s shape minus its leading axis, so `return carry, x`
        // traces `means` to `x`'s (seeded) shape with `xs`'s leading axis
        // prepended back on.
        let code = "import jax.numpy as jnp\nimport jax\ndef f(xs: Float[Array, \"seq features\"], mean0: Float[Array, \"features\"], var0: Float[Array, \"features\"]):\n    def step(carry, x):\n        return carry, x\n    init = (mean0, var0)\n    final_carry, means = jax.lax.scan(step, init, xs)\n    final_mean, final_var = final_carry";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "init"), None);
        assert_eq!(find_shape(&analysis, "final_carry"), None);
        assert_eq!(find_shape(&analysis, "final_mean"), Some(&shape(&["features"])));
        assert_eq!(find_shape(&analysis, "final_var"), Some(&shape(&["features"])));
        assert_eq!(
            find_shape(&analysis, "means"),
            Some(&shape(&["seq", "features"]))
        );
    }

    #[test]
    fn test_scan_heterogeneous_tuple_carry_stays_unshaped() {
        // `(a, b)` with DIFFERENT shapes isn't representable as one shape —
        // must stay `None`, not silently pick one element's shape.
        let code = "import jax\ndef f(a: Float[Array, \"m\"], b: Float[Array, \"n\"], xs: Float[Array, \"seq k\"]):\n    def body(carry, x):\n        return carry, x\n    final_carry, ys = jax.lax.scan(body, (a, b), xs)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "final_carry"), None);
    }

    #[test]
    fn test_rnn_cell_tuple_lhs_binds_both_hidden_sized() {
        let code = "import equinox as eqx\nclass M:\n    def __init__(self, key):\n        self.lstm = eqx.nn.LSTMCell(3, 5, key=key)\n\n    def __call__(self, x: Float[Array, \"in_size\"], h0: Float[Array, \"hidden\"], c0: Float[Array, \"hidden\"]):\n        h, c = self.lstm(x, (h0, c0))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["5"])));
    }

    #[test]
    fn test_self_method_call_propagates_return_shape() {
        let code = "class M:\n    def step(self, h: Float[Array, \"hidden\"], x: Float[Array, \"features\"]) -> Float[Array, \"hidden\"]:\n        return h\n\n    def run(self, a: Float[Array, \"hidden\"], b: Float[Array, \"features\"]):\n        out = self.step(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["hidden"])));
    }

    #[test]
    fn test_self_method_call_rank_mismatch_errors() {
        let code = "class M:\n    def step(self, h: Float[Array, \"hidden\"]) -> Float[Array, \"hidden\"]:\n        return h\n\n    def run(self, a: Float[Array, \"batch hidden\"]):\n        out = self.step(a)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("expected rank 1"));
    }

    #[test]
    fn test_self_method_call_traces_return_without_annotation() {
        // `forward` has no `-> ReturnType` annotation — its return shape is
        // traced from the bare `return logits` statement using `forward`'s
        // own already-computed body shapes (methods are analyzed in one
        // global source-order pass, so `forward`, defined first, has
        // already had `logits` shaped by the time `loss` calls it).
        let code = "import torch.nn as nn\nclass M:\n    def __init__(self, in_features, hidden, n_classes):\n        self.fc1 = nn.Linear(in_features, hidden)\n        self.fc2 = nn.Linear(hidden, n_classes)\n\n    def forward(self, x: Float[Array, \"batch in_features\"]):\n        h = self.fc1(x)\n        logits = self.fc2(h)\n        return logits\n\n    def loss(self, x: Float[Array, \"batch in_features\"]):\n        out = self.forward(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "logits"), Some(&shape(&["batch", "n_classes"])));
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["batch", "n_classes"])));
    }

    #[test]
    fn test_self_method_call_traced_return_substitutes_dims() {
        // Same as above, but the caller passes a DIFFERENT batch dim name —
        // confirms real substitution, not just a name coincidence.
        let code = "import torch.nn as nn\nclass M:\n    def __init__(self, in_features, hidden):\n        self.fc1 = nn.Linear(in_features, hidden)\n\n    def forward(self, x: Float[Array, \"batch in_features\"]):\n        h = self.fc1(x)\n        return h\n\n    def run(self, y: Float[Array, \"n in_features\"]):\n        out = self.forward(y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["n", "hidden"])));
    }

    #[test]
    fn test_self_method_call_non_identifier_return_not_traced() {
        // `return h + 1` isn't a bare identifier — v1 doesn't trace inline
        // expressions, so the call stays unshaped rather than guessing.
        let code = "class M:\n    def forward(self, h: Float[Array, \"d\"]):\n        return h + 1\n\n    def run(self, a: Float[Array, \"d\"]):\n        out = self.forward(a)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_body_return_tracing_preserves_scope_and_branch_rules() {
        for (body, traceable) in [
            ("return x", true),
            ("if flag:\n    return x\nreturn x", true),
            ("if flag:\n    return x\nreturn other", false),
            ("if flag:\n    return\nreturn x", false),
            ("return x, other", false),
            ("pass", false),
            ("def inner():\n    return other\nreturn x", true),
            ("class Inner:\n    def f(self):\n        return other\nreturn x", true),
        ] {
            // Exercise the original tree's nonzero offsets and nested scopes.
            let indented = body.replace('\n', "\n        ");
            let code = format!(
                "# π😀\nclass M:\n    @decorator\n    def helper(self, x: Float[Array, \"d\"]):\n        {indented}\n\n    def run(self, a: Float[Array, \"n\"], b: Float[Array, \"m\"]):\n        first = self.helper(a)\n        second = self.helper(b)"
            );
            let analysis = analyze(&code);
            assert!(analysis.errors.is_empty(), "{body}: {:?}", analysis.errors);
            for (name, dim) in [("first", "n"), ("second", "m")] {
                let expected = traceable.then(|| shape(&[dim]));
                assert_eq!(find_shape(&analysis, name), expected.as_ref(), "{body}");
            }
        }
    }

    #[test]
    fn test_chained_reshape_neg_one_then_astype_with_known_receiver() {
        // Mirrors corpus/scope_soup.py:31's `h.reshape(-1).astype(jnp.float32)`
        // chain, but with a receiver whose shape IS known — confirms the
        // chain itself (single `-1` reshape + `.astype`) resolves fine, so
        // the corpus dark spot is solely about the receiver's shape being
        // unknown (an un-annotated nested-closure param), not an
        // arg-parsing or chained-receiver gap in `reshape`/`astype`.
        let code = "import jax.numpy as jnp\ndef f(h: Float[Array, \"6 4\"]):\n    y = h.reshape(-1).astype(jnp.float32)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["24"])));
    }

    #[test]
    fn test_nested_closure_call_with_annotated_param_resolves() {
        // A locally-defined (nested) function called by bare name, with its
        // OWN param annotated — resolves via the same `bind_user_function_args`
        // path as a top-level helper or a `self.<method>` call. Confirms
        // nested-function scopes ARE reachable by call resolution; the
        // corpus/scope_soup.py `inner(h)` dark spot (item 5) is instead
        // because `inner`'s param has NO annotation and isn't seeded from
        // this call site (same structural gap as the un-annotated `scan`/
        // cell body params in items 2-3).
        let code = "def outer(x: Float[Array, \"6 4\"]):\n    def inner(h: Float[Array, \"6 4\"]):\n        y = h.reshape(-1)\n        return y\n    out = inner(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["24"])));
    }
}

#[cfg(test)]
mod inline_layer_application_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_equinox_linear_constructed_and_applied_inline() {
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    y = eqx.nn.Linear(3, 5)(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_inline_layer_concrete_mismatch_errors() {
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 4\"]):\n    y = eqx.nn.Linear(3, 5)(x)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("got 4"));
    }

    #[test]
    fn test_inline_layer_nested_in_activation() {
        let code = "import jax\nimport equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    y = jax.nn.relu(eqx.nn.Linear(3, 5)(x))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
    }
}

#[cfg(test)]
mod flax_layer_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// End-to-end mirror of corpus/flax_mlp.py: channels-last conv stem,
    /// avg_pool, inline Dense applications.
    #[test]
    fn test_flax_forward_pass() {
        let code = "import flax.linen as nn\ndef f(x: Float[Array, \"32 32 3\"]):\n    h = nn.Conv(features=16, kernel_size=(3, 3))(x)\n    h = nn.relu(h)\n    h = nn.avg_pool(h, window_shape=(2, 2), strides=(2, 2))\n    flat = h.reshape(-1)\n    h2 = nn.Dense(features=64)(flat)\n    logits = nn.Dense(features=10)(h2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["16", "16", "16"])));
        assert_eq!(find_shape(&analysis, "flat"), Some(&shape(&["4096"])));
        assert_eq!(find_shape(&analysis, "h2"), Some(&shape(&["64"])));
        assert_eq!(find_shape(&analysis, "logits"), Some(&shape(&["10"])));
    }

    #[test]
    fn test_flax_dense_symbolic_input() {
        let code = "import flax.linen as nn\ndef f(x: Float[Array, \"batch d\"]):\n    y = nn.Dense(features=128)(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "128"])));
    }

    #[test]
    fn test_flax_conv_nondefault_stride_refused() {
        // v1 models only stride-1/SAME; a strided flax Conv must not produce
        // a (wrong) shape.
        let code = "import flax.linen as nn\ndef f(x: Float[Array, \"32 32 3\"]):\n    h = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "h"), None);
    }
}

#[cfg(test)]
mod einops_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_rearrange_patchify() {
        let code = "from einops import rearrange\ndef f(img: Float[Array, \"3 224 224\"]):\n    patches = rearrange(img, \"c (h p1) (w p2) -> (h w) (p1 p2 c)\", p1=16, p2=16)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "patches"), Some(&shape(&["196", "768"])));
    }

    #[test]
    fn test_rearrange_symbolic_transpose() {
        let code = "from einops import rearrange\ndef f(x: Float[Array, \"b s d\"]):\n    y = rearrange(x, \"b s d -> s b d\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["s", "b", "d"])));
    }

    #[test]
    fn test_rearrange_symbolic_merge() {
        let code = "from einops import rearrange\ndef f(x: Float[Array, \"b s d\"]):\n    y = rearrange(x, \"b s d -> (b s) d\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b*s", "d"])));
    }

    #[test]
    fn test_reduce_mean() {
        let code = "from einops import reduce\ndef f(x: Float[Array, \"n d\"]):\n    pooled = reduce(x, \"n d -> d\", \"mean\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "pooled"), Some(&shape(&["d"])));
    }

    #[test]
    fn test_repeat_new_axis() {
        let code = "from einops import repeat\ndef f(cls: Float[Array, \"dim\"]):\n    stacked = repeat(cls, \"d -> n d\", n=4)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "stacked"), Some(&shape(&["4", "dim"])));
    }

    #[test]
    fn test_rearrange_rank_mismatch_errors() {
        let code = "from einops import rearrange\ndef f(x: Float[Array, \"b d\"]):\n    y = rearrange(x, \"b s d -> s b d\")";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("expects rank 3"));
    }

    #[test]
    fn test_rearrange_indivisible_errors() {
        let code = "from einops import rearrange\ndef f(img: Float[Array, \"3 224 224\"]):\n    patches = rearrange(img, \"c (h p1) (w p2) -> (h w) (p1 p2 c)\", p1=15, p2=15)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("not divisible"));
    }

    #[test]
    fn test_ellipsis_pattern_skips() {
        // Ellipsis on only one side isn't modelled.
        let code = "from einops import rearrange\ndef f(x: Float[Array, \"b s d\"]):\n    y = rearrange(x, \"... d -> d\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), None);
    }
}

#[cfg(test)]
mod linalg_tuple_unpacking_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_svd_unpacking() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"n d\"]):\n    u, s, vt = jnp.linalg.svd(a)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "u"), Some(&shape(&["n", "n"])));
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["min(n,d)"])));
        assert_eq!(find_shape(&analysis, "vt"), Some(&shape(&["d", "d"])));
    }

    #[test]
    fn test_qr_unpacking_concrete() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"6 4\"]):\n    q, r = jnp.linalg.qr(a)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "q"), Some(&shape(&["6", "4"])));
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&["4", "4"])));
    }

    #[test]
    fn test_eigh_unpacking() {
        let code = "import jax.numpy as jnp\ndef f(w: Float[Array, \"d d\"]):\n    evals, evecs = jnp.linalg.eigh(w)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "evals"), Some(&shape(&["d"])));
        assert_eq!(find_shape(&analysis, "evecs"), Some(&shape(&["d", "d"])));
    }

    #[test]
    fn test_meshgrid_xy_indexing() {
        let code = "import jax.numpy as jnp\ndef f(xs: Float[Array, \"nx\"], ys: Float[Array, \"ny\"]):\n    gx, gy = jnp.meshgrid(xs, ys)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "gx"), Some(&shape(&["ny", "nx"])));
        assert_eq!(find_shape(&analysis, "gy"), Some(&shape(&["ny", "nx"])));
    }

    #[test]
    fn test_svd_with_keyword_skips() {
        // Non-default modes aren't modelled; must skip, not guess.
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"n d\"]):\n    u, s, vt = jnp.linalg.svd(a, full_matrices=False)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "u"), None);
    }
}

#[cfg(test)]
mod multihead_attention_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// End-to-end mirror of corpus/torch_attention.py's forward pass.
    #[test]
    fn test_multihead_attention_tuple_unpacking() {
        let code = "import torch\nimport torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(512, 8)\n        self.norm = nn.LayerNorm(512)\n        self.ff = nn.Linear(512, 512)\n\n    def forward(self, x: Float[Array, \"seq 512\"]):\n        attn_out, attn_weights = self.attn(x, x, x)\n        h = x + attn_out\n        h = self.norm(h)\n        h += self.ff(h)\n        return h";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "attn_out"), Some(&shape(&["seq", "512"])));
        assert_eq!(
            find_shape(&analysis, "attn_weights"),
            Some(&shape(&["seq", "seq"]))
        );
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["seq", "512"])));
    }

    #[test]
    fn test_multihead_attention_cross_attention_weights() {
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(512, 8)\n\n    def forward(self, q: Float[Array, \"tgt 512\"], kv: Float[Array, \"src 512\"]):\n        out, w = self.attn(q, kv, kv)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["tgt", "512"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["tgt", "src"])));
    }

    /// Case, constructor, call arguments, query, key, expected weights.
    type AttentionRow<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str, Option<&'a [&'a str]>);

    #[test]
    fn test_multihead_attention_batched_weights() {
        let rows: &[AttentionRow] = &[
            ("sequence first", "nn.MultiheadAttention(16, 4)", "q, k, k", "7 2 16", "9 2 16", Some(&["2", "7", "9"])),
            ("batch first", "nn.MultiheadAttention(16, 4, batch_first=True)", "q, k, k", "2 7 16", "2 9 16", Some(&["2", "7", "9"])),
            ("per head", "nn.MultiheadAttention(16, 4)", "q, k, k, average_attn_weights=False", "7 2 16", "9 2 16", Some(&["2", "4", "7", "9"])),
            ("no weights", "nn.MultiheadAttention(16, 4)", "q, k, k, need_weights=False", "7 2 16", "9 2 16", None),
            ("zero attention key", "nn.MultiheadAttention(16, 4, add_zero_attn=True)", "q, k, k", "7 2 16", "9 2 16", Some(&["2", "7", "10"])),
        ];
        for (case, ctor, call, query, key, expected) in rows {
            let code = format!("import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = {ctor}\n\n    def forward(self, q: Float[Array, \"{query}\"], k: Float[Array, \"{key}\"]):\n        out, w = self.attn({call})");
            let analysis = analyze(&code);
            assert!(analysis.errors.is_empty(), "case: {case}: {:?}", analysis.errors);
            assert_eq!(find_shape(&analysis, "out"), Some(&shape(&query.split(' ').collect::<Vec<_>>())), "case: {case}");
            assert_eq!(find_shape(&analysis, "w"), expected.map(shape).as_ref(), "case: {case}");
        }
    }

    #[test]
    fn test_multihead_attention_single_lhs_binds_nothing() {
        // `out = self.attn(x, x, x)` (no tuple unpacking) binds the 2-tuple
        // `(output, weights)` of torch, so `out` has no shape.
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(512, 8)\n\n    def forward(self, x: Float[Array, \"seq 512\"]):\n        out = self.attn(x, x, x)";
        let analysis = analyze(code);

        // torch gives a tuple `(output, weights)`: `out` is no array.
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_multihead_attention_first_element_cross_attention_uses_query_shape() {
        // Query and key/value have different sequence lengths — the primary
        // output (`[0]`) still follows the query's shape, not the key/value's.
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(512, 8)\n\n    def forward(self, q: Float[Array, \"tgt 512\"], kv: Float[Array, \"src 512\"]):\n        out = self.attn(q, kv, kv)[0]";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["tgt", "512"])));
    }

    #[test]
    fn test_multihead_attention_vmap_single_lhs_binds_nothing() {
        // `out = jax.vmap(self.attn)(q, k, v)` — a torch MHA gives a tuple,
        // so `out` has no shape.
        let code = "import torch.nn as nn\nimport jax\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(512, 8)\n\n    def forward(self, q: Float[Array, \"batch seq 512\"], k: Float[Array, \"batch seq 512\"], v: Float[Array, \"batch seq 512\"]):\n        out = jax.vmap(self.attn)(q, k, v)";
        let analysis = analyze(code);

        // torch gives a tuple `(output, weights)`: `out` is no array.
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_lstm_single_lhs_binds_nothing() {
        // `out = self.lstm(x)` (no tuple unpacking) binds the tuple
        // `(output, final_state)`; `self.lstm(x)[0]` is the output.
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm = nn.LSTM(10, 20)\n\n    def forward(self, x: Float[Array, \"seq 10\"]):\n        out = self.lstm(x)\n        first = self.lstm(x)[0]";
        let analysis = analyze(code);

        // torch gives a tuple `(output, (h_n, c_n))`: `out` is no array.
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), None);
        assert_eq!(find_shape(&analysis, "first"), Some(&shape(&["seq", "20"])));
    }

    #[test]
    fn test_gru_single_lhs_binds_nothing() {
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.gru = nn.GRU(10, 20)\n\n    def forward(self, x: Float[Array, \"seq 10\"]):\n        out = self.gru(x)";
        let analysis = analyze(code);

        // torch gives a tuple `(output, h_n)`: `out` is no array.
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), None);
    }
}

/// Correctness and timing harnesses over `bench/benchmark_large.py` (regenerate
/// with `python3 bench/generate_benchmark.py`). The default suite runs one
/// correctness pass; the repeated timing harness stays ignored.
/// Run timing with: `cargo test --release bench_large_file -- --ignored --nocapture`.
#[cfg(test)]
mod bench_harness {
    use super::*;
    use std::fs;
    use std::time::Instant;
    use tree_sitter::Parser;

    #[test]
    fn benchmark_large_diagnostics_match_oracle() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/benchmark_large.py");
        let code = fs::read_to_string(&path).unwrap();
        let roots: Vec<PathBuf> = Vec::new();
        let cache = new_resolution_cache();
        let read = |_: &PathBuf| -> Option<String> { None };
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let analysis = analyze_layer_shapes(
            tree.root_node(),
            &code,
            &roots,
            read,
            5,
            Some(&cache),
        )
        .unwrap();

        assert_benchmark_diagnostics(&code, tree.root_node(), &analysis.errors);
    }

    #[test]
    #[ignore]
    fn bench_large_file() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/benchmark_large.py");
        let code = fs::read_to_string(&path).unwrap();
        let lines = code.lines().count();
        let roots: Vec<PathBuf> = Vec::new();
        let cache = new_resolution_cache();

        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();

        let (_, _, warm) = run_with(&mut parser, &code, &roots, &cache);
        let mut parse_times = Vec::new();
        let mut analyze_times = Vec::new();
        for _ in 0..10 {
            let (p, a, _) = run_with(&mut parser, &code, &roots, &cache);
            parse_times.push(p);
            analyze_times.push(a);
        }
        parse_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        analyze_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &[f64]| v[v.len() / 2];
        let shaped = warm.assignment_shapes.len();
        println!("\n=== bench: {} lines ===", lines);
        println!("  parse    median {:.1} ms  max {:.1} ms", med(&parse_times), parse_times[parse_times.len() - 1]);
        println!("  analyze  median {:.1} ms  max {:.1} ms", med(&analyze_times), analyze_times[analyze_times.len() - 1]);
        println!("  shapes inferred: {}   errors: {}", shaped, warm.errors.len());
        let mut msgs: std::collections::BTreeMap<String, usize> = Default::default();
        for e in &warm.errors {
            *msgs.entry(e.message.clone()).or_default() += 1;
        }
        for (m, n) in msgs.iter().take(30) {
            println!("    {:>3}x  {}", n, m);
        }
        assert!(shaped > 1000, "expected >1000 inferred shapes, got {shaped}");

        let tree = parser.parse(&code, None).unwrap();
        assert_benchmark_diagnostics(&code, tree.root_node(), &warm.errors);
        // Keystroke budget: parse + analyze should stay well under one frame-ish
        // interval even in debug. Release should be far below this.
        assert!(
            med(&analyze_times) < 1000.0,
            "analyze median {:.1} ms exceeds 1s regression tripwire",
            med(&analyze_times)
        );
    }

    /// Repeated calls expose reparsing costs that the model corpus does not.
    /// Run with `cargo test --release bench_repeated_helper_calls -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_repeated_helper_calls() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("helper.py"),
            "def identity(x: Float[Array, \"b d\"]) -> Float[Array, \"b d\"]:\n    return x\n",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let mut local = String::from("def identity(x: Float[Array, \"b d\"]):\n");
        for i in 0..100 {
            local.push_str(&format!("    h{i} = x + x\n"));
        }
        local.push_str("    return h99\n");

        for (label, prefix) in [
            ("imported", "from helper import identity\n".to_string()),
            ("same-file", local),
        ] {
            for calls in [100, 200, 400] {
                let mut code = format!("{prefix}\ndef caller(x: Float[Array, \"b d\"]):\n");
                for i in 0..calls {
                    code.push_str(&format!("    y{i} = identity(x)\n"));
                }
                let mut parser = Parser::new();
                parser
                    .set_language(&tree_sitter_python::LANGUAGE.into())
                    .unwrap();
                let cache = new_resolution_cache();
                let (_, _, warm) = run_with(&mut parser, &code, &roots, &cache);
                assert!(warm.errors.is_empty(), "{:?}", warm.errors);
                let outputs: Vec<_> = warm.assignment_shapes.iter()
                    .filter(|rec| rec.name.starts_with('y'))
                    .collect();
                assert_eq!(outputs.len(), calls);
                assert!(outputs.iter().all(|rec| rec.shape == ["b", "d"]));
                let mut times: Vec<_> = (0..9)
                    .map(|_| run_with(&mut parser, &code, &roots, &cache).1)
                    .collect();
                times.sort_by(f64::total_cmp);
                println!(
                    "{label}: {calls} calls, analyze median {:.1} ms",
                    times[times.len() / 2]
                );
            }
        }
    }

    fn assert_benchmark_diagnostics(
        code: &str,
        root: tree_sitter::Node,
        errors: &[ShapeError],
    ) {
        // A count-only check can hide one missed marker behind one false positive.
        // Require exactly one diagnostic inside each marked top-level definition.
        let marked_ranges: Vec<_> = code
            .match_indices("# expected-error:")
            .map(|(byte, _)| {
                let mut definition = root.descendant_for_byte_range(byte, byte + 1).unwrap();
                while let Some(parent) = definition.parent() {
                    if parent.kind() == "module" {
                        break;
                    }
                    definition = parent;
                }
                definition.byte_range()
            })
            .collect();
        let mut diagnostic_counts = vec![0; marked_ranges.len()];
        for error in errors {
            let Some(index) = marked_ranges
                .iter()
                .position(|range| range.contains(&error.range.start_byte))
            else {
                panic!(
                    "unexpected diagnostic outside an expected-error block at line {}: {}",
                    error.range.start_point.row + 1,
                    error.message
                );
            };
            diagnostic_counts[index] += 1;
        }
        assert_eq!(
            diagnostic_counts,
            vec![1; marked_ranges.len()],
            "each '# expected-error:' block must produce exactly one diagnostic"
        );

        let mut expected = std::collections::BTreeMap::new();
        expected.insert(
            "Linear layer 'self.fc2' expected input last dim 32, got 64 for 'h'".to_string(),
            5,
        );
        for i in [0, 6, 12, 18, 24] {
            expected.insert(
                format!(
                    "matmul dimension mismatch: a last dim k{i} != b second-to-last dim j{i}"
                ),
                1,
            );
            expected.insert(
                format!(
                    "concatenate dimension mismatch at axis 1: expected c{i}, got e{i}"
                ),
                1,
            );
        }
        let mut actual = std::collections::BTreeMap::new();
        for error in errors {
            *actual.entry(error.message.clone()).or_default() += 1;
        }
        assert_eq!(actual, expected, "benchmark diagnostics changed");
    }

    fn run_with(
        parser: &mut Parser,
        code: &str,
        roots: &[PathBuf],
        cache: &ResolutionCache,
    ) -> (f64, f64, LayerShapeAnalysis) {
        let read = |path: &PathBuf| fs::read_to_string(path).ok();
        let t0 = Instant::now();
        let tree = parser.parse(code, None).unwrap();
        let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let analysis =
            analyze_layer_shapes(tree.root_node(), code, roots, read, 5, Some(cache)).unwrap();
        let analyze_ms = t1.elapsed().as_secs_f64() * 1000.0;
        (parse_ms, analyze_ms, analysis)
    }
}

#[cfg(test)]
mod jax_lax_higher_order_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_lax_map_over_user_function() {
        let code = "import jax\ndef step(x: Float[Array, \"d\"]) -> Float[Array, \"d\"]:\n    return x\ndef f(xs: Float[Array, \"n d\"]):\n    ys = jax.lax.map(step, xs)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "ys"), Some(&shape(&["n", "d"])));
    }

    #[test]
    fn test_lax_while_loop_carry_invariant() {
        let code = "import jax\ndef f(state: Float[Array, \"b d\"]):\n    out = jax.lax.while_loop(cond_fn, body_fn, state)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_lax_fori_loop_carry_invariant() {
        let code = "import jax\ndef f(state: Float[Array, \"b d\"]):\n    out = jax.lax.fori_loop(0, 10, body_fn, state)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_lax_cond_conservatively_unknown() {
        let code = "import jax\ndef f(x: Float[Array, \"b d\"]):\n    out = jax.lax.cond(pred, true_fn, false_fn, x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_lax_top_k_tuple_unpacking() {
        let code = "import jax\ndef f(x: Float[Array, \"b n\"]):\n    values, indices = jax.lax.top_k(x, 5)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["b", "5"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["b", "5"])));
    }

    #[test]
    fn test_lax_sort_key_val_tuple_unpacking() {
        let code = "import jax\ndef f(keys: Float[Array, \"n\"], values: Float[Array, \"n\"]):\n    sk, sv = jax.lax.sort_key_val(keys, values)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "sk"), Some(&shape(&["n"])));
        assert_eq!(find_shape(&analysis, "sv"), Some(&shape(&["n"])));
    }

    #[test]
    fn test_lax_sort_single_operand_shape_preserving() {
        let code = "import jax\ndef f(x: Float[Array, \"n\"]):\n    y = jax.lax.sort(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n"])));
    }

    #[test]
    fn test_lax_concatenate_reuses_jnp_concatenate_semantics() {
        let code = "import jax\ndef f(a: Float[Array, \"m d\"], b: Float[Array, \"n d\"]):\n    y = jax.lax.concatenate([a, b], 0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["m+n", "d"])));
    }

    #[test]
    fn test_lax_transpose_permutes_axes() {
        let code = "import jax\ndef f(x: Float[Array, \"a b c\"]):\n    y = jax.lax.transpose(x, (2, 0, 1))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["c", "a", "b"])));
    }

    #[test]
    fn test_lax_broadcast_in_dim_target_shape() {
        let code = "import jax\ndef f(x: Float[Array, \"3\"]):\n    y = jax.lax.broadcast_in_dim(x, (2, 3, 4), (1,))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["2", "3", "4"])));
    }

    #[test]
    fn test_lax_dynamic_slice_uses_slice_sizes() {
        let code = "import jax\ndef f(x: Float[Array, \"8 8\"]):\n    y = jax.lax.dynamic_slice(x, (0, 0), (2, 3))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["2", "3"])));
    }

    #[test]
    fn test_lax_conv_general_dilated_default_layout() {
        let code = "import jax\ndef f(x: Float[Array, \"1 3 8 8\"], w: Float[Array, \"16 3 3 3\"]):\n    y = jax.lax.conv_general_dilated(x, w, (1, 1), 'VALID')";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["1", "16", "6", "6"])));
    }

    #[test]
    fn test_lax_gather_classified_but_conservatively_unknown() {
        let code = "import jax\ndef f(x: Float[Array, \"8 8\"], idx: Float[Array, \"4 2\"]):\n    y = jax.lax.gather(x, idx, dimension_numbers, (1, 1))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), None);
    }
}

#[cfg(test)]
mod jax_numpy_new_builtin_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_hsplit_tuple_unpacking() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"4 6\"]):\n    a, b = jnp.hsplit(x, 2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["4", "3"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["4", "3"])));
    }

    #[test]
    fn test_vsplit_tuple_unpacking() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"4 6\"]):\n    a, b = jnp.vsplit(x, 2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["2", "6"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["2", "6"])));
    }

    #[test]
    fn test_nonzero_tuple_unpacking_shares_length() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"4 6\"]):\n    rows, cols = jnp.nonzero(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "rows"), Some(&shape(&["nonzero(x)"])));
        assert_eq!(find_shape(&analysis, "cols"), Some(&shape(&["nonzero(x)"])));
    }

    #[test]
    fn test_diagflat_flattens_then_squares() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"2 3\"]):\n    y = jnp.diagflat(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["6", "6"])));
    }

    #[test]
    fn test_select_approximate_shape() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"n\"], b: Float[Array, \"n\"]):\n    y = jnp.select([c1, c2], [a, b], 0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n"])));
    }

    #[test]
    fn test_kron_elementwise_product() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"2 3\"], b: Float[Array, \"4 5\"]):\n    y = jnp.kron(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["8", "15"])));
    }

    #[test]
    fn test_block_nested_assembly() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"2 3\"], b: Float[Array, \"2 4\"], c: Float[Array, \"5 3\"], d: Float[Array, \"5 4\"]):\n    y = jnp.block([[a, b], [c, d]])";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["7", "7"])));
    }

    #[test]
    fn test_take_along_axis_matches_indices() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"4 8\"], idx: Float[Array, \"4 1\"]):\n    y = jnp.take_along_axis(a, idx, axis=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "1"])));
    }

    #[test]
    fn test_argwhere_known_rank_unknown_count() {
        let code = "import jax.numpy as jnp\ndef f(mask: Float[Array, \"4 8\"]):\n    y = jnp.argwhere(mask)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["nonzero(mask)", "2"])));
    }

    #[test]
    fn test_bincount_conservatively_unknown() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"100\"]):\n    y = jnp.bincount(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["?"])));
    }
}

#[cfg(test)]
mod linear_algebra_extra_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_linalg_solve_follows_rhs_shape() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"n n\"], b: Float[Array, \"n k\"]):\n    x = jnp.linalg.solve(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["n", "k"])));
    }

    #[test]
    fn test_linalg_cholesky_shape_preserving_square() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"n n\"]):\n    l = jnp.linalg.cholesky(a)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "l"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn test_torch_linalg_pinv_swaps_last_two_dims() {
        let code = "import torch\ndef f(a: Float[Array, \"m n\"]):\n    b = torch.linalg.pinv(a)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "m"])));
    }

    #[test]
    fn test_torch_linalg_matrix_rank_drops_last_two_dims() {
        let code = "import torch\ndef f(a: Float[Array, \"b m n\"]):\n    r = torch.linalg.matrix_rank(a)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_torch_linalg_lstsq_solution_matrix_rhs() {
        let code = "import torch\ndef f(a: Float[Array, \"m n\"], b: Float[Array, \"m k\"]):\n    solution, residuals, rank, sv = torch.linalg.lstsq(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "solution"), Some(&shape(&["n", "k"])));
        assert_eq!(find_shape(&analysis, "residuals"), None);
    }

    #[test]
    fn test_cross_broadcasts_batch_dims() {
        let code = "import torch\ndef f(a: Float[Array, \"n 3\"], b: Float[Array, \"1 3\"]):\n    y = torch.cross(a, b, dim=-1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "3"])));

        let bad_code = "import torch\ndef f(a: Float[Array, \"n 3\"], b: Float[Array, \"3\"]):\n    bad = torch.cross(a, b, dim=-1)";
        let bad_analysis = analyze(bad_code);

        assert_eq!(bad_analysis.errors.len(), 1);
        let error = &bad_analysis.errors[0];
        assert_eq!(error.variable, "bad");
        assert!(error.message.contains("same number of dimensions"));
        assert_eq!(
            &bad_code[error.range.start_byte..error.range.end_byte],
            "(a, b, dim=-1)"
        );
        assert_eq!(find_shape(&bad_analysis, "bad"), None);
    }

    #[test]
    fn test_linalg_norm_reduction_axis_semantics() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"b d\"]):\n    n = jnp.linalg.norm(x, axis=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "n"), Some(&shape(&["b"])));
    }
}

#[cfg(test)]
mod einops_extra_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_einops_einsum_matmul() {
        let code = "import einops\ndef f(a: Float[Array, \"m n\"], b: Float[Array, \"n p\"]):\n    y = einops.einsum(a, b, \"m n, n p -> m p\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["m", "p"])));
    }

    #[test]
    fn test_einops_pack_single_star_axis() {
        let code = "import einops\ndef f(a: Float[Array, \"2 d\"], b: Float[Array, \"3 d\"]):\n    packed, ps = einops.pack([a, b], \"* d\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "packed"), Some(&shape(&["5", "d"])));
        assert_eq!(find_shape(&analysis, "ps"), None);
    }

    #[test]
    fn test_einops_unpack_conservatively_unknown() {
        let code = "import einops\ndef f(packed: Float[Array, \"5 d\"]):\n    a, b = einops.unpack(packed, ps, \"* d\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
    }

    #[test]
    fn test_einops_parse_shape_never_infers_array_shape() {
        let code = "import einops\ndef f(x: Float[Array, \"h w\"]):\n    dims = einops.parse_shape(x, \"h w\")";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "dims"), None);
    }
}

#[cfg(test)]
mod jax_nn_extra_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_one_hot_appends_num_classes() {
        let code = "import jax.nn as jnn\ndef f(x: Float[Array, \"batch\"]):\n    y = jnn.one_hot(x, 10)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "10"])));
    }

    #[test]
    fn test_dot_product_attention_output_shape() {
        let code = "import jax.nn as jnn\ndef f(q: Float[Array, \"b sq h dk\"], k: Float[Array, \"b sk h dk\"], v: Float[Array, \"b sk h dv\"]):\n    o = jnn.dot_product_attention(q, k, v)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "o"), Some(&shape(&["b", "sq", "h", "dv"])));
    }

    #[test]
    fn test_logsumexp_reduction_axis_semantics() {
        let code = "import jax.nn as jnn\ndef f(x: Float[Array, \"b d\"]):\n    y = jnn.logsumexp(x, axis=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_relu_still_shape_preserving() {
        // Regression check: carving one_hot out of the generic shape-
        // preserving jax.nn match arm must not disturb the other activations.
        let code = "import jax.nn as jnn\ndef f(x: Float[Array, \"b d\"]):\n    y = jnn.relu(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }
}

/// End-to-end coverage for the torch indexing/selection/tuple-output method
/// and free-function additions, plus the `x.T` / `x.mT` properties and the
/// new `torch.nn.functional` / `torch.nn.utils.rnn` free functions.
#[cfg(test)]
mod torch_new_builtin_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_x_t_reverses_all_dims() {
        let code = "def f(x: Float[Array, \"a b c\"]):\n    y = x.T";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_x_mt_swaps_last_two_dims() {
        let code = "def f(x: Float[Array, \"a b c\"]):\n    y = x.mT";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["a", "c", "b"])));
    }

    #[test]
    fn test_x_mt_requires_rank_2() {
        let code = "def f(x: Float[Array, \"a\"]):\n    y = x.mT";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_method_topk_tuple_unpacking() {
        let code = "def f(x: Float[Array, \"b n\"]):\n    values, indices = x.topk(5)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["b", "5"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["b", "5"])));
    }

    #[test]
    fn test_free_topk_tuple_unpacking_explicit_dim() {
        let code = "import torch\ndef f(x: Float[Array, \"b n\"]):\n    values, indices = torch.topk(x, 3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["3", "n"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["3", "n"])));
    }

    #[test]
    fn test_method_chunk_tuple_unpacking() {
        let code = "def f(x: Float[Array, \"6 4\"]):\n    a, b, c = x.chunk(3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["2", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["2", "4"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["2", "4"])));
    }

    #[test]
    fn test_free_chunk_uneven_last_piece_smaller() {
        let code = "import torch\ndef f(x: Float[Array, \"7\"]):\n    a, b, c = torch.chunk(x, 3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["1"])));
    }

    #[test]
    fn test_method_unbind_tuple_unpacking() {
        let code = "def f(x: Float[Array, \"3 4 5\"]):\n    a, b, c = x.unbind(0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["4", "5"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["4", "5"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["4", "5"])));
    }

    #[test]
    fn test_unbind_mismatched_count_errors() {
        let code = "def f(x: Float[Array, \"3 4 5\"]):\n    a, b = x.unbind(0)";
        let analysis = analyze(code);

        assert!(!analysis.errors.is_empty());
    }

    #[test]
    fn test_method_split_tuple_unpacking() {
        // Real torch `Tensor.split` semantics (`KnownFunction::TorchSplit`):
        // the 2nd arg is a chunk *size*, not a section count — splitting
        // axis size 6 into chunks of size 3 yields 2 outputs of size 3 each.
        let code = "def f(x: Float[Array, \"6 4\"]):\n    a, b = x.split(3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_method_split_tuple_unpacking_uneven_remainder() {
        // Real torch semantics: split_size=3 on axis size 10 → ceil(10/3)=4
        // chunks, the last being the smaller remainder (1), not 3 equal
        // sections like jnp/np.split.
        let code = "def f(x: Float[Array, \"10 4\"]):\n    a, b, c, d = x.split(3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["1", "4"])));
    }

    #[test]
    fn test_free_function_torch_split_uneven_remainder() {
        let code =
            "import torch\ndef f(x: Float[Array, \"10 4\"]):\n    a, b, c, d = torch.split(x, 3, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["1", "4"])));
    }

    #[test]
    fn test_torch_tensor_split_keeps_section_count_semantics() {
        // `torch.tensor_split` keeps jnp/np section-*count* semantics (it
        // does NOT get real `torch.split`'s chunk-size treatment) — 6 split
        // into 2 sections → 3 each.
        let code =
            "import torch\ndef f(x: Float[Array, \"6 4\"]):\n    a, b = torch.tensor_split(x, 2, dim=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_matmul_operator_commutative_symbolic_dims_no_false_error() {
        // Regression: dims were compared as raw strings, so `d*2` and `2*d`
        // (same dim, different spelling) produced a false "matmul dimension
        // mismatch" through the `@` operator's `apply_matmul_shape`.
        let code = concat!(
            "def f(a: Float[Array, \"m d*2\"], b: Float[Array, \"2*d n\"]):\n",
            "    out = a @ b\n",
        );
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["m", "n"])));
    }

    #[test]
    fn test_method_kthvalue_tuple_unpacking() {
        let code = "def f(x: Float[Array, \"b n\"]):\n    values, indices = x.kthvalue(2, dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["b"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_method_median_dim_tuple_unpacking() {
        let code = "def f(x: Float[Array, \"b n\"]):\n    values, indices = x.median(dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["b"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_free_mode_dim_tuple_unpacking() {
        let code = "import torch\ndef f(x: Float[Array, \"b n\"]):\n    values, indices = torch.mode(x, dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "values"), Some(&shape(&["b"])));
        assert_eq!(find_shape(&analysis, "indices"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_method_gather_matches_index_shape() {
        let code = "def f(x: Float[Array, \"b n\"], idx: Float[Array, \"b 3\"]):\n    y = x.gather(1, idx)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "3"])));
    }

    #[test]
    fn test_method_index_select_dim_from_index_length() {
        let code = "def f(x: Float[Array, \"4 10\"], idx: Float[Array, \"3\"]):\n    y = x.index_select(1, idx)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "3"])));
    }

    #[test]
    fn test_method_narrow_replaces_dim_with_length() {
        let code = "def f(x: Float[Array, \"4 10\"]):\n    y = x.narrow(1, 2, 5)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "5"])));
    }

    #[test]
    fn test_method_select_removes_dim() {
        let code = "def f(x: Float[Array, \"4 10 3\"]):\n    y = x.select(1, 0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "3"])));
    }

    #[test]
    fn test_method_masked_select_conservatively_unknown() {
        let code = "def f(x: Float[Array, \"4 10\"]):\n    y = x.masked_select(mask)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["?"])));
    }

    #[test]
    fn test_method_masked_fill_shape_preserving() {
        let code = "def f(x: Float[Array, \"4 10\"]):\n    y = x.masked_fill(mask, 0.0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "10"])));
    }

    #[test]
    fn test_method_unfold_appends_window_dim() {
        let code = "def f(x: Float[Array, \"10\"]):\n    y = x.unfold(0, 2, 1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["9", "2"])));
    }

    #[test]
    fn test_method_view_as_reshape_as_expand_as() {
        let code = "def f(x: Float[Array, \"12\"], other: Float[Array, \"3 4\"]):\n    y = x.view_as(other)\n    z = x.reshape_as(other)\n    w = x.expand_as(other)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_method_item_is_scalar() {
        let code = "def f(x: Float[Array, \"1\"]):\n    y = x.item()";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&Vec::new()));
    }

    #[test]
    fn test_method_new_zeros_shape_from_arg() {
        let code = "def f(x: Float[Array, \"4\"]):\n    y = x.new_zeros((3, 4))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_method_clone_and_dtype_casts_shape_preserving() {
        let code = "def f(x: Float[Array, \"4 3\"]):\n    y = x.clone()\n    z = x.float()\n    w = x.cuda()";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "3"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["4", "3"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["4", "3"])));
    }

    #[test]
    fn test_method_norm_axis_reduction() {
        let code = "def f(x: Float[Array, \"4 3\"]):\n    y = x.norm(dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4"])));
    }

    #[test]
    fn test_method_diagonal_and_triu_tril() {
        let code = "def f(x: Float[Array, \"4 4\"]):\n    y = x.diagonal()\n    z = x.triu()\n    w = x.tril()";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["4", "4"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["4", "4"])));
    }

    #[test]
    fn test_free_take_along_dim_matches_indices() {
        let code = "import torch\ndef f(a: Float[Array, \"4 8\"], idx: Float[Array, \"4 1\"]):\n    y = torch.take_along_dim(a, idx, dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["4", "1"])));
    }

    #[test]
    fn test_free_torch_unique_conservatively_unknown() {
        let code = "import torch\ndef f(x: Float[Array, \"4 8\"]):\n    y = torch.unique(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["?"])));
    }

    #[test]
    fn test_free_combinations_n_choose_r() {
        let code = "import torch\ndef f(x: Float[Array, \"5\"]):\n    y = torch.combinations(x, 2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["10", "2"])));
    }

    #[test]
    fn test_free_cartesian_prod() {
        let code = "import torch\ndef f(a: Float[Array, \"2\"], b: Float[Array, \"3\"]):\n    y = torch.cartesian_prod(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["6", "2"])));
    }

    #[test]
    fn test_free_block_diag() {
        let code = "import torch\ndef f(a: Float[Array, \"2 3\"], b: Float[Array, \"4 5\"]):\n    y = torch.block_diag(a, b)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["6", "8"])));
    }

    #[test]
    fn test_functional_conv2d_output_shape() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"8 3 32 32\"], w: Float[Array, \"16 3 3 3\"]):\n    y = F.conv2d(x, w, padding=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "16", "32", "32"]))
        );
    }

    #[test]
    fn test_functional_max_pool2d_output_shape() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"8 3 32 32\"]):\n    y = F.max_pool2d(x, 2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "3", "16", "16"]))
        );
    }

    #[test]
    fn test_functional_avg_pool2d_output_shape() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"8 3 32 32\"]):\n    y = F.avg_pool2d(x, 2, 2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "3", "16", "16"]))
        );
    }

    #[test]
    fn test_functional_softmax_shape_preserving() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"b d\"]):\n    y = F.softmax(x, dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_functional_one_hot_appends_num_classes() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"batch\"]):\n    y = F.one_hot(x, 10)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "10"])));
    }

    #[test]
    fn test_functional_embedding_appends_embed_dim() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"batch seq\"], weight: Float[Array, \"1000 64\"]):\n    y = F.embedding(x, weight)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "seq", "64"]))
        );
    }

    #[test]
    fn test_functional_interpolate_scale_factor() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"8 3 16 16\"]):\n    y = F.interpolate(x, scale_factor=2)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["8", "3", "32", "32"]))
        );
    }

    #[test]
    fn test_pad_sequence_batch_first() {
        let code = "import torch\ndef f(a: Float[Array, \"3 8\"], b: Float[Array, \"5 8\"]):\n    y = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["2", "pad_len", "8"]))
        );
    }
}

#[cfg(test)]
mod shape_error_kind_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    // `jax.lax.dot_general` is approximated as matmul (see `llm.txt`); a
    // batch-dim mismatch it flags may be a false positive from that
    // approximation, so it's classified `Approximation`, not `Mismatch`.
    #[test]
    fn dot_general_mismatch_is_classified_approximation() {
        let code = "import jax\ndef f(a: Float[Array, \"2 3 4\"], b: Float[Array, \"5 4 6\"]):\n    y = jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))))";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(analysis.errors[0].kind, ShapeErrorKind::Approximation);
    }

    // A genuine `jnp.matmul` dimension mismatch (not `dot_general`) stays a
    // real `Mismatch` even though it also resolves to `KnownFunction::Matmul`.
    #[test]
    fn real_matmul_mismatch_is_classified_mismatch() {
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"2 3\"], b: Float[Array, \"4 5\"]):\n    y = jnp.matmul(a, b)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(analysis.errors[0].kind, ShapeErrorKind::Mismatch);
    }

    fn transpose_shape(imports: &str, params: &str, expr: &str) -> (Option<Vec<String>>, usize) {
        let code = format!("{imports}def f({params}):\n    y = {expr}");
        let analysis = analyze(&code);
        (find_shape(&analysis, "y").cloned(), analysis.errors.len())
    }

    fn dims(dims: &[&str]) -> Option<Vec<String>> {
        Some(dims.iter().map(|d| d.to_string()).collect())
    }

    const ABC: &str = "x: Float[Array, \"a b c\"]";
    const AB: &str = "x: Float[Array, \"a b\"]";

    #[test]
    fn torch_free_transpose_swaps_two_dims() {
        let imp = "import torch\n";
        assert_eq!(transpose_shape(imp, ABC, "torch.transpose(x, 0, 1)"), (dims(&["b", "a", "c"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "torch.transpose(x, -1, 0)"), (dims(&["c", "b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "torch.transpose(x, dim0=1, dim1=2)"), (dims(&["a", "c", "b"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "torch.swapdims(x, 0, 2)"), (dims(&["c", "b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "torch.swapaxes(x, 0, 2)"), (dims(&["c", "b", "a"]), 0));
    }

    #[test]
    fn torch_free_transpose_unknown_dim_is_unknown() {
        let (shape, errors) = transpose_shape("import torch\n", ABC, "torch.transpose(x, 0, d)");
        assert_eq!((shape, errors), (None, 0));
    }

    #[test]
    fn numpy_free_transpose_keeps_permutation() {
        let imp = "import jax.numpy as jnp\n";
        assert_eq!(transpose_shape(imp, ABC, "jnp.transpose(x, (2, 0, 1))"), (dims(&["c", "a", "b"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "jnp.transpose(x, [1, 0, 2])"), (dims(&["b", "a", "c"]), 0));
        assert_eq!(transpose_shape(imp, AB, "jnp.swapaxes(x, 0, 1)"), (dims(&["b", "a"]), 0));
    }

    #[test]
    fn torch_method_transpose_swaps_at_any_rank() {
        let imp = "import torch\n";
        assert_eq!(transpose_shape(imp, AB, "x.transpose(0, 1)"), (dims(&["b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.transpose(0, 1)"), (dims(&["b", "a", "c"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.transpose(-1, -2)"), (dims(&["a", "c", "b"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.swapdims(0, 2)"), (dims(&["c", "b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.swapaxes(0, 2)"), (dims(&["c", "b", "a"]), 0));
    }

    #[test]
    fn method_transpose_dim_keywords_are_torch_swap() {
        for imp in ["", "import jax.numpy as jnp\n"] {
            assert_eq!(transpose_shape(imp, ABC, "x.transpose(dim0=0, dim1=1)"), (dims(&["b", "a", "c"]), 0));
        }
    }

    #[test]
    fn numpy_method_transpose_is_permutation() {
        let imp = "import jax.numpy as jnp\n";
        assert_eq!(transpose_shape(imp, AB, "x.transpose(0, 1)"), (dims(&["a", "b"]), 0));
        assert_eq!(transpose_shape(imp, AB, "x.transpose(1, 0)"), (dims(&["b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.transpose(2, 0, 1)"), (dims(&["c", "a", "b"]), 0));
        assert_eq!(transpose_shape(imp, AB, "x.transpose()"), (dims(&["b", "a"]), 0));
        assert_eq!(transpose_shape(imp, ABC, "x.transpose((1, 0, 2))"), (dims(&["b", "a", "c"]), 0));
    }

    #[test]
    fn numpy_method_transpose_wrong_rank_is_error() {
        let (shape, errors) = transpose_shape("import numpy as np\n", ABC, "x.transpose(0, 1)");
        assert_eq!((shape, errors), (None, 1));
    }

    #[test]
    fn unknown_backend_method_transpose_never_errors() {
        for imp in ["", "import torch\nimport numpy as np\n"] {
            assert_eq!(transpose_shape(imp, ABC, "x.transpose(0, 1)"), (None, 0));
            assert_eq!(transpose_shape(imp, AB, "x.transpose(0, 1)"), (None, 0));
            assert_eq!(transpose_shape(imp, AB, "x.transpose(1, 0)"), (dims(&["b", "a"]), 0));
        }
    }

    #[test]
    fn unknown_backend_equal_dims_transpose_is_known() {
        let (shape, errors) = transpose_shape("", "x: Float[Array, \"n n\"]", "x.transpose(0, 1)");
        assert_eq!((shape, errors), (dims(&["n", "n"]), 0));
    }

    #[test]
    fn transposed_properties_do_not_regress() {
        let imp = "import torch\n";
        assert_eq!(transpose_shape(imp, ABC, "x.mT"), (dims(&["a", "c", "b"]), 0));
        assert_eq!(transpose_shape(imp, AB, "x.T"), (dims(&["b", "a"]), 0));
    }

    // `jnp.transpose`/`.permute`-style full-rank axis permutations with a
    // genuine duplicate-axis bug stay `Mismatch`.
    #[test]
    fn method_permute_duplicate_axis_is_classified_mismatch() {
        let code = "def f(x: Float[Array, \"a b c\"]):\n    y = x.permute(0, 0, 1)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(analysis.errors[0].kind, ShapeErrorKind::Mismatch);
    }
}

/// RNN `(output, state)` tuple-LHS binding (`out, h = self.gru(x)`, `out,
/// (h, c) = self.lstm(x)`) — extends `tuple_rhs_shapes` for `self.<attr>`
/// calls resolving to `LayerKind::Rnn`. The state is `[D*num_layers, N,
/// H_out]` (`layers::rnn_outputs`).
#[cfg(test)]
mod rnn_tuple_unpacking_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_gru_tuple_unpacking_unbatched() {
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.gru = nn.GRU(10, 20)\n\n    def forward(self, x: Float[Array, \"seq 10\"]):\n        out, h = self.gru(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["seq", "20"])));
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["1", "20"])));
    }

    #[test]
    fn test_gru_tuple_unpacking_batched_state_keeps_batch_axis() {
        // Batched input (seq, batch, features): the output keeps all dims
        // (last -> hidden); h_n is (num_layers*num_directions, batch, hidden).
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.gru = nn.GRU(10, 20)\n\n    def forward(self, x: Float[Array, \"seq batch 10\"]):\n        out, h = self.gru(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(
            find_shape(&analysis, "out"),
            Some(&shape(&["seq", "batch", "20"]))
        );
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["1", "batch", "20"])));
    }

    #[test]
    fn test_lstm_nested_h_c_state_pattern() {
        // `out, (h, c) = self.lstm(x)` — one level of nested tuple-pattern
        // support (`TupleTarget::Nested`): both `h` and `c` bind to the same
        // state shape.
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm = nn.LSTM(10, 20)\n\n    def forward(self, x: Float[Array, \"seq 10\"]):\n        out, (h, c) = self.lstm(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["seq", "20"])));
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["1", "20"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["1", "20"])));
    }

    fn lstm_block(ctor: &str, input: &str, lhs: &str) -> String {
        format!("import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm = {ctor}\n        self.fc = nn.Linear(32, 4)\n\n    def forward(self, x: Float[Array, \"{input}\"]):\n        {lhs} = self.lstm(x)\n        y = self.fc(o)")
    }

    /// Case, constructor, input, LHS, expected `o`, `h`, `c`.
    type LstmRow<'a> = (&'a str, &'a str, &'a str, &'a str, Option<&'a [&'a str]>, Option<&'a [&'a str]>, Option<&'a [&'a str]>);

    #[test]
    fn test_lstm_options() {
        let rows: &[LstmRow] = &[
            // A bidirectional output feeds `Linear(32, 4)` without an error.
            ("bidirectional", "nn.LSTM(8, 16, bidirectional=True)", "5 3 8", "o, (h, c)", Some(&["5", "3", "32"]), Some(&["2", "3", "16"]), Some(&["2", "3", "16"])),
            ("batch first layers", "nn.LSTM(8, 16, num_layers=3, batch_first=True, bidirectional=True)", "3 5 8", "o, (h, c)", Some(&["3", "5", "32"]), Some(&["6", "3", "16"]), Some(&["6", "3", "16"])),
            // h and c differ under a projection: one shape cannot bind both.
            ("projection", "nn.LSTM(8, 32, proj_size=16, bidirectional=True)", "5 3 8", "o, (h, c)", Some(&["5", "3", "32"]), None, None),
            // An LSTM state is a pair, so a plain name gets no tensor shape.
            ("state name", "nn.LSTM(8, 16, bidirectional=True)", "5 3 8", "o, h", Some(&["5", "3", "32"]), None, None),
        ];
        for (case, ctor, input, lhs, o, h, c) in rows {
            let analysis = analyze(&lstm_block(ctor, input, lhs));
            assert!(analysis.errors.is_empty(), "case: {case}: {:?}", analysis.errors);
            assert_eq!(find_shape(&analysis, "o"), o.map(shape).as_ref(), "case: {case}");
            assert_eq!(find_shape(&analysis, "h"), h.map(shape).as_ref(), "case: {case}");
            assert_eq!(find_shape(&analysis, "c"), c.map(shape).as_ref(), "case: {case}");
        }
    }

    #[test]
    fn test_lstm_unidirectional_output_mismatch_is_reported() {
        let analysis = analyze(&lstm_block("nn.LSTM(8, 16)", "5 3 8", "o, (h, c)"));
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("expected input last dim 32, got 16"));
    }

    #[test]
    fn test_rnn_tuple_wrong_arity_skips() {
        // A plain RNN/GRU/LSTM call really returns a 2-tuple; unpacking into
        // 3 targets isn't modelled.
        let code = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.gru = nn.GRU(10, 20)\n\n    def forward(self, x: Float[Array, \"seq 10\"]):\n        a, b, c = self.gru(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), None);
    }
}

/// `ShapeError.related` population for two-operand binary-op mismatches
/// (`shape_of_binary_operator` in `analysis.rs`): related range/message
/// point at the right-hand operand ("the other operand" relative to the
/// mismatch, which the primary message already names first as the left
/// operand).
#[cfg(test)]
mod binary_op_related_information_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    #[test]
    fn matmul_mismatch_points_at_exact_annotation_dimensions() {
        let code = r#"import jax.numpy as jnp
def f(a: Float[Array, "2 3"], b: Float[Array, "4 5"]):
    y = a @ b"#;
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let err = &analysis.errors[0];
        let (related_range, related_message) = err.related.as_ref().expect("related info");
        assert_eq!(related_message, "other operand `b`: shape [4, 5]");
        assert_eq!(&code[err.range.start_byte..err.range.end_byte], "3");
        assert_eq!(&code[related_range.start_byte..related_range.end_byte], "4");
    }

    #[test]
    fn elementwise_mismatch_points_at_exact_annotation_dimensions() {
        let code = r#"import jax.numpy as jnp
def f(a: Float[Array, "3"], b: Float[Array, "4"]):
    y = a + b"#;
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let err = &analysis.errors[0];
        let (related_range, related_message) = err.related.as_ref().expect("related info");
        assert_eq!(related_message, "other operand `b`: shape [4]");
        assert_eq!(&code[err.range.start_byte..err.range.end_byte], "3");
        assert_eq!(&code[related_range.start_byte..related_range.end_byte], "4");
    }

    #[test]
    fn matmul_batch_mismatch_points_at_exact_annotation_dimensions() {
        let code = r#"def f(a: Float[Array, "left_batch rows inner"], b: Float[Array, "right_batch inner cols"]):
    y = a @ b"#;
        let analysis = analyze(code);
        let err = &analysis.errors[0];
        let (related_range, _) = err.related.as_ref().expect("related info");

        assert_eq!(&code[err.range.start_byte..err.range.end_byte], "left_batch");
        assert_eq!(&code[related_range.start_byte..related_range.end_byte], "right_batch");
    }

    #[test]
    fn matmul_marks_only_a_proved_rank_two_transpose_fix() {
        let code = r#"def f(a: Float[Array, "m k"], b: Float[Array, "n k"]):
    y = a @ b"#;
        let analysis = analyze(code);
        let fix = analysis.errors[0].fix.as_ref().expect("transpose fix");
        let ShapeFix::AppendTranspose { operand_range, .. } = fix;
        assert_eq!(&code[operand_range.start_byte..operand_range.end_byte], "b");

        let unsafe_code = r#"def f(a: Float[Array, "m k"], b: Float[Array, "n p"]):
    y = a @ b"#;
        let unsafe_analysis = analyze(unsafe_code);
        assert!(unsafe_analysis.errors[0].fix.is_none());

        let batched_code = r#"def f(a: Float[Array, "batch m k"], b: Float[Array, "other n k"]):
    y = a @ b"#;
        let batched_analysis = analyze(batched_code);
        assert!(batched_analysis.errors[0].fix.is_none());
    }

    #[test]
    fn non_binary_op_errors_have_no_related_info() {
        // Sanity check: errors from a non-binary-op site (e.g. a
        // known-function shape rule) don't get a spurious `related` value —
        // only the binary-operator sites populate it.
        let code = "import jax.numpy as jnp\ndef f(a: Float[Array, \"2 3\"], b: Float[Array, \"5 6\"]):\n    y = jnp.matmul(a, b)";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert!(analysis.errors[0].related.is_none());
    }
}





#[cfg(test)]
mod lazy_call_site_parameter_seeding_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    /// Mirrors corpus/scope_soup.py's `inner(h)` dark spot: a nested
    /// closure whose OWN param has no annotation is now shaped by seeding
    /// it from the call site's argument shape.
    #[test]
    fn test_nested_closure_unannotated_param_seeded_from_call_site() {
        let code = "def outer(x: Float[Array, \"6 4\"]):\n    def inner(h):\n        y = h.reshape(-1)\n        return y\n    out = inner(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["24"])));
        // First-call-wins write-back also shapes the closure's own body
        // (hover/inlay UX inside `inner`).
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["24"])));
    }

    /// A same-file helper with an un-annotated param, called from two
    /// different sites with DIFFERENT argument shapes. Each call site must
    /// resolve to ITS OWN correct shape — the second (non-first)
    /// specialization must not reuse or be polluted by the first's seeded
    /// values (a `HashMap::entry(..).or_insert(..)` bug where the second
    /// call's seed would be silently dropped because the key was already
    /// present from the first call's first-call-wins write-back).
    #[test]
    fn test_multi_call_site_specialization_resolves_independently() {
        let code = "\
def double(x):
    y = x + x
    return y

def f(a: Float[Array, \"3 4\"], b: Float[Array, \"5\"]):
    out_a = double(a)
    out_b = double(b)
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out_a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "out_b"), Some(&shape(&["5"])));
    }

    /// An error in the body of the callee names the variable of its own
    /// statement, not the variable of the call.
    #[test]
    fn test_callee_body_error_keeps_its_own_variable() {
        for (helper, variable) in [
            ("def helper(a):\n    b = a + a.T\n    return b\n", "b"),
            ("def helper(a):\n    return a + a.T\n", ""),
        ] {
            let code = format!("{helper}\ndef f(x: Float[Array, \"n 8\"]):\n    y = helper(x)\n");
            let analysis = analyze(&code);
            assert!(!analysis.errors.is_empty());
            for error in &analysis.errors {
                assert_eq!(error.variable, variable, "{:?}", analysis.errors);
                assert_eq!(error.range.start_point.row, 1);
            }
        }
    }

    /// Same as above but with three call sites and a keyword-argument call
    /// mixed in, to exercise both positional and keyword seeding across
    /// multiple independent specializations of the same scope.
    #[test]
    fn test_multi_call_site_specialization_keyword_and_positional() {
        let code = "\
def project(inp):
    out = inp.sum(axis=0)
    return out

def f(a: Float[Array, \"3 4\"], b: Float[Array, \"5 6 7\"]):
    p_a = project(a)
    p_b = project(inp=b)
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "p_a"), Some(&shape(&["4"])));
        assert_eq!(find_shape(&analysis, "p_b"), Some(&shape(&["6", "7"])));
    }

    /// Direct recursion with an un-annotated carry param must not hang or
    /// stack-overflow: re-entering a scope already being specialized
    /// returns `None` for that inner call, so the recursive call itself
    /// stays unshaped, but that must not crash or poison the outer call's
    /// own (unrelated) shape.
    #[test]
    fn test_recursion_guard_direct_self_call() {
        let code = "\
def loop(carry, n):
    if n == 0:
        return carry
    return loop(carry, n - 1)

def f(x: Float[Array, \"batch d\"]):
    out = loop(x, 3)
";
        let analysis = analyze(code);

        // No panic/hang is the primary assertion; a correct-but-conservative
        // outcome (the recursive inner call can't be resolved, so the
        // multi-return-statement trace sees divergent targets and stays
        // dark) is acceptable — soundness over completeness for recursion.
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    }

    /// Mutual recursion (`ping` <-> `pong`) with un-annotated params must
    /// also terminate rather than infinitely re-entering each other's
    /// specialization.
    #[test]
    fn test_recursion_guard_mutual_recursion() {
        let code = "\
def ping(carry, n):
    if n <= 0:
        return carry
    return pong(carry, n - 1)

def pong(carry, n):
    if n <= 0:
        return carry
    return ping(carry, n - 1)

def f(x: Float[Array, \"batch d\"]):
    out = ping(x, 4)
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    }

    /// A same-file helper with an un-annotated param whose body computes a
    /// `return <expr>` (not a bare identifier) — extends the older
    /// bare-identifier-only return trace to arbitrary expressions.
    #[test]
    fn test_seeded_function_return_expr_tracing() {
        let code = "\
def scaled(x):
    return x * 2.0

def f(a: Float[Array, \"batch d\"]):
    out = scaled(a)
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&["batch", "d"])));
    }

    /// `return <expr>` tracing also covers a chained-call expression, not
    /// just a binary operator.
    #[test]
    fn test_seeded_function_return_chained_call_expr() {
        let code = "\
def flatten_sum(x):
    return x.reshape(-1).sum()

def f(a: Float[Array, \"3 4\"]):
    out = flatten_sum(a)
";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "out"), Some(&shape(&[])));
    }

    /// A fully-annotated same-file function is unaffected by this
    /// mechanism — it still goes through the older annotation-only path,
    /// and a non-identifier return still isn't traced (regression check
    /// mirroring `test_self_method_call_non_identifier_return_not_traced`).
    #[test]
    fn test_fully_annotated_function_unaffected_by_seeding() {
        let code = "class M:\n    def forward(self, h: Float[Array, \"d\"]):\n        return h + 1\n\n    def run(self, a: Float[Array, \"d\"]):\n        out = self.forward(a)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_torch_max_two_tensors_broadcasts() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"3 1\"]):\n    r = torch.max(x, y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_torch_min_two_tensors_broadcasts() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"4\"]):\n    r = torch.min(x, other=y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_torch_max_min_mismatch_stays_unknown() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"5 4\"]):\n    r = torch.max(x, y)\n    s = torch.min(x, y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "r"), None);
        assert_eq!(find_shape(&analysis, "s"), None);
    }

    #[test]
    fn test_torch_max_dim_returns_tuple_so_unknown() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"]):\n    a = torch.max(x, 1)\n    b = torch.min(x, dim=1)\n    c = torch.max(x, 1)[0]\n    d = torch.max(x, dim=1).values\n    v, i = torch.max(x, 1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["a", "b", "c", "d", "v", "i"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
    }

    #[test]
    fn test_torch_max_no_dim_is_scalar() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"]):\n    r = torch.max(x)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&[])));
    }

    #[test]
    fn test_tensor_max_min_methods_broadcast() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"3 1\"]):\n    a = x.max(y)\n    b = x.min(y)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_tensor_max_dim_method_returns_tuple_so_unknown() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"]):\n    a = x.max(1)\n    b = x.min(dim=1)\n    v, i = x.max(dim=1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["a", "b", "v", "i"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
    }

    #[test]
    fn test_numpy_like_max_axis_keeps_reduction() {
        let code = "import jax.numpy as jnp\nimport numpy as np\ndef f(x: Float[Array, \"3 4\"]):\n    a = jnp.max(x, 1)\n    b = np.max(x, axis=1)\n    c = x.max(1)\n    d = jnp.min(x, axis=0)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["4"])));
    }

    #[test]
    fn test_torch_max_scalar_dim_is_unknown() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"], n: int):\n    a = torch.max(x, n)\n    b = x.max(n)\n    c = torch.min(x, 1)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["a", "b", "c"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
    }

    #[test]
    fn test_concatenate_unknown_axis_and_none_axis_are_not_axis_zero() {
        let code = "import torch\nimport jax.numpy as jnp\ndef f(x: Float[Array, \"3 4\"], u: Float[Array, \"3 4\"], n: int):\n    a = torch.cat((x, u), n)\n    b = jnp.concatenate((x, u), axis=None)\n    c = jnp.concatenate([x, u], n)\n    d = torch.cat(dim=1, tensors=(x, u))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
        assert_eq!(find_shape(&analysis, "c"), None);
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["3", "8"])));
    }

    #[test]
    fn test_einsum_sequence_operand_is_torch_only() {
        let code = "import numpy as np\nimport torch\ndef f(x: Float[Array, \"3 4\"], u: Float[Array, \"3 4\"]):\n    a = np.einsum('i,i->', [1,2,3], [4,5,6])\n    b = np.einsum('ij,ij->ij', (x, u))\n    c = torch.einsum('ij,ij->ij', [x, u])";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(analysis.errors[0].variable, "b");
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_torch_cat_with_empty_vector_is_not_an_error() {
        let code = "import torch\ndef f(x: Float[Array, \"3 4\"]):\n    a = torch.cat([x, torch.empty(0)])\n    b = torch.cat((x, torch.empty(0)))";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    }

    #[test]
    fn test_concatenate_tuple_element_error_reported_once() {
        let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"3 4\"], u: Float[Array, \"3 4\"]):\n    a = jnp.concatenate((x @ x, u))\n    b = jnp.concatenate([x @ x, u])";
        let analysis = analyze(code);

        assert_eq!(analysis.errors.len(), 2, "{:?}", analysis.errors);
    }
}

#[cfg(test)]
mod tuple_literal_unpack_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn hint(analysis: &LayerShapeAnalysis, line: u32, name: &str) -> Option<Vec<String>> {
        analysis
            .assignment_shapes
            .iter()
            .find(|r| r.line == line && r.name == name)
            .map(|r| r.shape.clone())
    }

    const HEAD: &str = "import jax.numpy as jnp\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"5\"], z: Float[Array, \"6\"], w: Float[Array, \"7\"]):\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEAD}{body}"))
    }

    #[test]
    fn reshape_with_shape_unpack_product_names_no_internal_symbol() {
        let code = "import torch\ndef f(x: Float[Array, \"b s c\"]):\n    a, b, c = x.shape\n    n2 = x.reshape(a * b, c)\n    n3 = torch.reshape(x, (a * b, c))\n    n4 = x.view(a * b, -1)\n";
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        for r in &a.assignment_shapes {
            for d in &r.shape {
                assert!(!d.contains("__"), "{} -> {:?}", r.name, r.shape);
            }
        }
        assert_eq!(hint(&a, 3, "n2"), Some(shape(&["b*s", "c"])));
        assert_eq!(hint(&a, 4, "n3"), Some(shape(&["b*s", "c"])));
        assert_eq!(hint(&a, 5, "n4"), None);
    }

    #[test]
    fn unpacked_shape_names_stand_for_their_dims() {
        let a = analyze("def f(x: Float[Array, \"b s\"], w: Float[Array, \"b s\"]):\n    a, b = x.shape\n    y = x.reshape(a, b)\n    z = y + w\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(hint(&a, 2, "y"), Some(shape(&["b", "s"])));
    }

    #[test]
    fn rebound_shape_name_stops_standing_for_a_dim() {
        let a = analyze("import torch\ndef f(x: Float[Array, \"b s\"]):\n    a, b = x.shape\n    a = 7\n    y = x.reshape(a * b, 2)\n");
        assert_eq!(hint(&a, 4, "y"), None);
    }

    #[test]
    fn error_messages_show_source_text_for_inline_operands() {
        let code = "import jax.numpy as jnp\nfrom einops import rearrange\nimport torch\ndef f(x: Float[Array, \"b s c\"], z: Float[Array, \"3 4\"]):\n    e1 = rearrange(jnp.exp(x), \"a b -> b a\")\n    e2 = torch.exp(z).permute(0, 1, 2)\n";
        let a = analyze(code);
        assert_eq!(a.errors.len(), 2, "{:?}", a.errors);
        for e in &a.errors {
            assert!(!e.message.contains("__"), "{}", e.message);
        }
        assert!(a.errors.iter().any(|e| e.message.contains("for 'jnp.exp(x)'")), "{:?}", a.errors);
        assert!(a.errors.iter().any(|e| e.message.contains("transpose of 'torch.exp(z)'")), "{:?}", a.errors);
    }

    #[test]
    fn plain_names_bind_elementwise() {
        let a = run("    a, b = x, y\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["5"])));
        assert_eq!(hint(&a, 2, "a"), Some(shape(&["3", "4"])));
        assert_eq!(hint(&a, 2, "b"), Some(shape(&["5"])));
    }

    #[test]
    fn bracketed_forms_bind_elementwise() {
        let a = run("    a, b = (x, y)\n    c, d = [x, y]\n    (e, g) = (y, x)\n    [h, i] = [y, x]\n    j, = x,\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "d"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "e"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "g"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "h"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "i"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "j"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn swap_reads_right_side_before_any_write() {
        let a = run("    a, b = x, y\n    a, b = b, a\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(hint(&a, 3, "a"), Some(shape(&["5"])));
        assert_eq!(hint(&a, 3, "b"), Some(shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn rotation_of_three_names() {
        let a = run("    a, b, c = x, y, z\n    a, b, c = b, c, a\n");
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["6"])));
        assert_eq!(find_shape(&a, "c"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn swap_in_a_loop_is_unknown_after_the_loop() {
        let a = run("    for i in range(2):\n        x, y = y, x\n    q = x + jnp.zeros(3)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "x"), None);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn swap_through_a_temp_in_a_loop_is_unknown_after_the_loop() {
        let a = run("    for i in range(2):\n        t = x\n        x = y\n        y = t\n    q = x + jnp.zeros(3)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "x"), None);
    }

    #[test]
    fn swap_in_a_while_loop_is_unknown_after_the_loop() {
        let a = run("    while cond():\n        x, y = y, x\n");
        assert_eq!(find_shape(&a, "x"), None);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn loop_carried_name_with_a_stable_shape_stays_known() {
        let a = run("    for i in range(2):\n        x = jnp.exp(x)\n        t = x\n");
        assert_eq!(find_shape(&a, "x"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "t"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn name_written_from_a_changing_loop_name_is_unknown_after_the_loop() {
        let a = run("    for i in range(2):\n        t = x\n        x = y\n        y = t\n");
        assert_eq!(find_shape(&a, "t"), None);
    }

    #[test]
    fn unrelated_unpack_in_a_loop_keeps_its_shape() {
        let a = run("    for i in range(2):\n        a, b = x, y\n");
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn swap_in_a_loop_inside_a_called_helper_is_unknown() {
        let code = "def helper(a, b):\n    for i in range(2):\n        a, b = b, a\n    return a\ndef outer(x: Float[Array, \"3 4\"], y: Float[Array, \"5\"]):\n    r = helper(x, y)\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "r"), None);
    }

    #[test]
    fn duplicate_target_keeps_the_last_element() {
        let a = run("    a, a = x, y\n");
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["5"])));
    }

    #[test]
    fn starred_target_takes_the_middle_and_is_never_a_tensor() {
        let a = run("    rest = x\n    a, *rest, c = x, y, z, w\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "c"), Some(&shape(&["7"])));
        assert_eq!(find_shape(&a, "rest"), None);
    }

    #[test]
    fn starred_first_and_last_positions() {
        let a = run("    *init, last = x, y, z\n    first, *tail = x, y, z\n");
        assert_eq!(find_shape(&a, "init"), None);
        assert_eq!(find_shape(&a, "last"), Some(&shape(&["6"])));
        assert_eq!(find_shape(&a, "first"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "tail"), None);
    }

    #[test]
    fn starred_target_may_be_empty() {
        let a = run("    a, *mid, b = x, y\n");
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&a, "mid"), None);
    }

    #[test]
    fn too_few_elements_for_a_starred_pattern_is_unknown() {
        let a = run("    a = x\n    b = y\n    a, b, *m, c = x, y\n");
        assert_eq!(find_shape(&a, "a"), None);
        assert_eq!(find_shape(&a, "b"), None);
        assert_eq!(find_shape(&a, "c"), None);
    }

    #[test]
    fn differing_lengths_are_unknown_and_evict_stale_bindings() {
        let a = run("    p = x\n    q = y\n    p, q = x, y, z\n    r = x\n    s = y\n    t = z\n    r, s, t = x, y\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        for name in ["p", "q", "r", "s", "t"] {
            assert_eq!(find_shape(&a, name), None, "{name}");
        }
    }

    #[test]
    fn unknown_element_leaves_only_its_own_target_unknown() {
        let a = run("    b = y\n    a, b = x, mystery(y)\n    c, d = mystery(x), z\n");
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "b"), None);
        assert_eq!(find_shape(&a, "c"), None);
        assert_eq!(find_shape(&a, "d"), Some(&shape(&["6"])));
    }

    #[test]
    fn integer_literals_stay_unknown() {
        let a = run("    n, m = 3, 4\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "n"), None);
        assert_eq!(find_shape(&a, "m"), None);
    }

    #[test]
    fn element_expressions_are_evaluated() {
        let a = run("    a, b = x.T, y * 2.0\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["4", "3"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["5"])));
    }

    #[test]
    fn element_error_is_reported_once_and_named_after_its_target() {
        let a = run("    a, b = x @ x, z\n");
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(a.errors[0].variable, "a");
        assert_eq!(find_shape(&a, "a"), None);
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["6"])));
    }

    #[test]
    fn splat_on_the_right_is_unknown() {
        let a = run("    a = x\n    b = y\n    a, b = *x, y\n");
        assert_eq!(find_shape(&a, "a"), None);
        assert_eq!(find_shape(&a, "b"), None);
    }

    #[test]
    fn nested_target_with_a_tuple_element() {
        let a = run("    (a, b), c = (x, x), z\n    (d, e), g = (x, y), z\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "a"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "b"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "c"), Some(&shape(&["6"])));
        assert_eq!(find_shape(&a, "d"), None);
        assert_eq!(find_shape(&a, "e"), None);
        assert_eq!(find_shape(&a, "g"), Some(&shape(&["6"])));
    }

    #[test]
    fn nested_target_with_a_tensor_element_is_unknown() {
        let a = run("    (a, b), c = x, z\n");
        assert_eq!(find_shape(&a, "a"), None);
        assert_eq!(find_shape(&a, "b"), None);
        assert_eq!(find_shape(&a, "c"), Some(&shape(&["6"])));
    }

    #[test]
    fn coverage_counts_a_literal_unpack_as_shaped() {
        let code = format!("{HEAD}    a, b = x, y\n    c, d = b, a\n");
        let tree = parse(&code);
        let report = analyze_coverage(tree.root_node(), &code, &[], |_| None, 5, None).unwrap();
        assert!(report.dark.is_empty(), "{:?}", report.dark);
        assert_eq!(report.total, 2);
        assert_eq!(report.shaped, 2);
    }

    #[test]
    fn unpack_inside_a_nested_scope_uses_that_scope() {
        let code = "def outer(x: Float[Array, \"3 4\"], y: Float[Array, \"5\"]):\n    def swap(a, b):\n        a, b = b, a\n        return a\n    r = swap(x, y)\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "r"), Some(&shape(&["5"])));
    }
}

#[cfg(test)]
mod closure_free_variable_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEAD: &str = "import jax\nimport jax.numpy as jnp\ndef outer(x: Float[Array, \"n 4\"], w: Float[Array, \"4 6\"], v: Float[Array, \"4 5\"]):\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEAD}{body}"))
    }

    #[test]
    fn closure_reads_enclosing_parameter() {
        let a = run("    def level1(t):\n        u = t @ w\n        return u\n    y = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
        assert_eq!(find_shape(&a, "u"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn closure_reads_enclosing_local() {
        let a = run("    k = jnp.zeros((4, 8))\n    def level1(t):\n        return t @ k\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn variable_is_read_at_call_time_not_definition_time() {
        let a = run("    def level1(t):\n        return t @ w\n    w = jnp.zeros((4, 7))\n    y = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "7"])));
    }

    #[test]
    fn each_call_sees_the_variable_as_it_is_then() {
        let a = run("    def level1(t):\n        return t @ w\n    y1 = level1(x)\n    w = jnp.zeros((4, 7))\n    y2 = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y1"), Some(&shape(&["n", "6"])));
        assert_eq!(find_shape(&a, "y2"), Some(&shape(&["n", "7"])));
    }

    #[test]
    fn reassignment_to_unknown_does_not_leave_a_stale_shape() {
        let a = run("    def level1(t):\n        return t @ w\n    w = mystery(w)\n    y = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn call_before_the_variable_exists_is_unknown() {
        let a = run("    def level1(t):\n        return t @ late\n    y0 = level1(x)\n    late = jnp.zeros((4, 5))\n    y1 = level1(x)\n");
        assert_eq!(find_shape(&a, "y0"), None);
        assert_eq!(find_shape(&a, "y1"), Some(&shape(&["n", "5"])));
    }

    #[test]
    fn definition_time_value_never_reaches_a_deferred_call() {
        let a = run("    def f(t):\n        return t @ w\n    def g(t):\n        return f(t)\n    w = jnp.zeros((4, 9))\n    y = g(x)\n    w = jnp.zeros((4, 3))\n    y2 = g(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "9"])));
        assert_eq!(find_shape(&a, "y2"), Some(&shape(&["n", "3"])));
    }

    #[test]
    fn annotated_closure_that_calls_a_closure_is_evaluated_at_its_own_call() {
        let a = run("    def f(t):\n        return t @ w\n    def g(t: Float[Array, \"n 4\"]):\n        r = f(t)\n        return r\n    w = jnp.zeros((4, 9))\n    y = g(x)\n    w = jnp.zeros((4, 3))\n    y2 = g(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "9"])));
        assert_eq!(find_shape(&a, "y2"), Some(&shape(&["n", "3"])));
    }

    #[test]
    fn local_assigned_later_shadows_the_enclosing_name() {
        let a = run("    def level1(t):\n        u = t @ w\n        w = t.sum(axis=0)\n        return u\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn local_binding_wins_over_enclosing_name() {
        let a = run("    def level1(t):\n        w = t.sum(axis=0)\n        return w\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["4"])));
    }

    #[test]
    fn parameter_shadows_the_enclosing_name() {
        let a = run("    def level1(t, w):\n        return t @ w\n    y = level1(x, v)\n    y_missing = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "5"])));
        assert_eq!(find_shape(&a, "y_missing"), None);
    }

    #[test]
    fn nonlocal_read_resolves_like_a_free_variable() {
        let a = run("    def level1(t):\n        nonlocal w\n        return t @ w\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn nonlocal_rebinding_makes_the_enclosing_variable_unknown() {
        let a = run("    def bump(t):\n        nonlocal w\n        w = w * 2.0\n        return t\n    z = bump(x)\n    y = x @ w\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn nonlocal_rebinding_hides_the_variable_from_sibling_closures() {
        let a = run("    def bump(t):\n        nonlocal w\n        w = t\n        return t\n    def use(t):\n        return t @ w\n    bump(x)\n    y = use(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn nonlocal_assigned_name_is_usable_after_the_assignment() {
        let a = run("    def reset(t):\n        nonlocal w\n        w = t\n        return w @ t.T\n    z = reset(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn call_outside_the_enclosing_function_does_not_see_its_variables() {
        let code = "def a(x: Float[Array, \"n 4\"], w: Float[Array, \"4 6\"]):\n    def helper(t):\n        return t @ w\n    return helper(x)\n\ndef b(x: Float[Array, \"m 4\"]):\n    r = helper(x)\n";
        let an = analyze(code);
        assert_eq!(find_shape(&an, "r"), None);
    }

    #[test]
    fn comprehension_variable_does_not_shadow_the_enclosing_name() {
        let a = run("    def level1(t, ws):\n        sq = [w for w in ws]\n        return t @ w\n    y = level1(x, v)\n");
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn comprehension_body_stays_unknown() {
        let a = run("    def level1(ts):\n        outs = [t @ w for t in ts]\n        return outs\n    y = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
        assert_eq!(find_shape(&a, "outs"), None);
    }

    #[test]
    fn fully_annotated_closure_reads_enclosing_variable() {
        let a = run("    def level1(t: Float[Array, \"n 4\"]):\n        u = t @ w\n        return u\n    y1 = level1(x)\n    w = jnp.zeros((4, 7))\n    y2 = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y1"), Some(&shape(&["n", "6"])));
        assert_eq!(find_shape(&a, "y2"), Some(&shape(&["n", "7"])));
    }

    #[test]
    fn fully_annotated_closure_error_is_not_duplicated() {
        let a = run("    def level1(t: Float[Array, \"n 4\"], q: Float[Array, \"5 6\"]):\n        bad = t @ q\n        u = t @ w\n        return u\n    p = jnp.zeros((5, 6))\n    y = level1(x, p)\n");
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn mismatch_with_the_call_time_value_is_a_true_error() {
        let a = run("    def level1(t):\n        return t @ w\n    w = jnp.zeros((5, 6))\n    y = level1(x)\n");
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn two_level_closure_reaches_the_outermost_variable() {
        let a = run("    def level1(t):\n        def level2(s):\n            return s @ w\n        return level2(t)\n    y = level1(x)\n");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn scan_body_reads_enclosing_variable() {
        let code = "import jax\nimport jax.numpy as jnp\ndef outer(xs: Float[Array, \"t b d\"], W: Float[Array, \"d d\"], h0: Float[Array, \"b d\"]):\n    def step(c, x):\n        h = jnp.tanh(c @ W + x)\n        return h, h\n    hT, ys = jax.lax.scan(step, h0, xs)\n";
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "ys"), Some(&shape(&["t", "b", "d"])));
    }

    #[test]
    fn same_named_closures_in_different_functions_resolve_to_their_own() {
        let code = "def a(x: Float[Array, \"n 4\"], w: Float[Array, \"4 6\"]):\n    def level1(t):\n        return t @ w\n    ya = level1(x)\n\ndef b(x: Float[Array, \"m 4\"]):\n    def level1(t):\n        return t * 2.0\n    yb = level1(x)\n";
        let an = analyze(code);
        assert_eq!(find_shape(&an, "ya"), Some(&shape(&["n", "6"])));
        assert_eq!(find_shape(&an, "yb"), Some(&shape(&["m", "4"])));
    }

    #[test]
    fn every_binding_form_makes_the_name_local() {
        let forms = [
            "with ctx() as w:\n            pass",
            "for w in range(3):\n            pass",
            "try:\n            pass\n        except Exception as w:\n            pass",
            "import numpy as w",
            "from numpy import ones as w",
            "(w := mystery(t))",
            "def w():\n            pass",
            "class w:\n            pass",
            "w: int",
            "a, (b, *w) = mystery(t)",
            "match mystery(t):\n                case [w, _]:\n                    pass",
        ];
        for form in forms {
            let body = format!(
                "    def level1(t):\n        if mystery(t):\n            {form}\n        return t @ w\n    y = level1(x)\n"
            );
            let a = run(&body);
            assert_eq!(find_shape(&a, "y"), None, "{form}");
        }
        let a = run("    def level1(t):\n        if mystery(t):\n            pass\n        return t @ w\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "6"])));
    }

    #[test]
    fn global_name_is_not_the_enclosing_variable() {
        let a = run("    def level1(t):\n        global w\n        return t @ w\n    y = level1(x)\n");
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn method_closure_through_a_class_does_not_see_function_locals() {
        let code = "def outer(x: Float[Array, \"n 4\"], w: Float[Array, \"4 6\"]):\n    class C:\n        def m(self, t):\n            return t @ w\n        def n(self, t):\n            r = self.m(t)\n            return r\n    return C\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "r"), None);
    }
}

#[cfg(test)]
mod user_submodule_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const EQX: &str = "import jax\nimport equinox as eqx\n";
    const TORCH: &str = "import torch\nimport torch.nn as nn\n";
    const FLAX: &str = "import flax.linen as nn\n";

    fn eqx_block(call_def: &str, ret: &str) -> String {
        format!(
            "class Block(eqx.Module):\n    w: eqx.nn.Linear\n    def __init__(self, key):\n        self.w = eqx.nn.Linear(4, 8, key=key)\n    {call_def}\n        h = jax.vmap(self.w)(x)\n        {ret}\n"
        )
    }

    fn eqx_model(call: &str) -> String {
        format!(
            "class Model(eqx.Module):\n    blk: Block\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n 4\"]):\n        {call}\n"
        )
    }

    fn eqx_pair(call_def: &str, ret: &str, call: &str) -> LayerShapeAnalysis {
        let code = format!("{EQX}{}{}", eqx_block(call_def, ret), eqx_model(call));
        analyze(&code)
    }

    #[test]
    fn annotated_call_with_expression_return() {
        let a = eqx_pair(
            "def __call__(self, x: Float[Array, \"s 4\"]):",
            "return jax.nn.gelu(h)",
            "out = self.blk(x)",
        );
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "out"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn unannotated_call_is_seeded_from_the_argument() {
        let a = eqx_pair("def __call__(self, x):", "return jax.nn.gelu(h)", "out = self.blk(x)");
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "out"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn return_annotation_is_bound_to_the_argument_dims() {
        let a = eqx_pair(
            "def __call__(self, x: Float[Array, \"s 4\"]) -> Float[Array, \"s 8\"]:",
            "return h",
            "out = self.blk(x)",
        );
        assert_eq!(find_shape(&a, "out"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn expression_argument_is_a_valid_call_argument() {
        let a = eqx_pair(
            "def __call__(self, x):",
            "return h",
            "out = self.blk(x + x)",
        );
        assert_eq!(find_shape(&a, "out"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn annotated_callee_hover_keeps_its_own_dims() {
        let a = eqx_pair(
            "def __call__(self, x: Float[Array, \"s 4\"]):",
            "return h",
            "out = self.blk(x)",
        );
        assert_eq!(find_shape(&a, "h"), Some(&shape(&["s", "8"])));
        assert!(a.assignment_shapes.iter().all(|r| r.name != "h" || r.shape == shape(&["s", "8"])));
    }

    #[test]
    fn argument_rank_mismatch_is_a_true_error() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"a b 4\"]):\n        out = self.blk(x)\n",
            eqx_block("def __call__(self, x: Float[Array, \"s 4\"]):", "return h")
        );
        let a = analyze(&code);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(find_shape(&a, "out"), None);
    }

    #[test]
    fn named_argument_dim_against_an_integer_annotation_is_not_an_error() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n d\"]):\n        out = self.blk(x)\n",
            eqx_block("def __call__(self, x: Float[Array, \"s 4\"]):", "return h")
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
    }

    #[test]
    fn variadic_annotation_is_not_a_rank_error() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n m 4\"]):\n        out = self.blk(x)\n",
            eqx_block("def __call__(self, x: Float[Array, \"... 4\"]):", "return h")
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
    }

    #[test]
    fn integer_mismatch_against_the_annotation_is_a_true_error() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n 5\"]):\n        out = self.blk(x)\n",
            eqx_block("def __call__(self, x: Float[Array, \"s 4\"]):", "return h")
        );
        let a = analyze(&code);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    const DIM_BLOCK: &str = "class Block(eqx.Module):\n    def __init__(self, dim, key):\n        self.lin = eqx.nn.Linear(dim, dim, key=key)\n    def __call__(self, x: Float[Array, \"dim\"]) -> Float[Array, \"dim\"]:\n        return self.lin(x)\n";

    fn dim_block_call(arg: &str) -> LayerShapeAnalysis {
        analyze(&format!(
            "{EQX}import jax.numpy as jnp\n{DIM_BLOCK}class Net(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(8, key)\n    def __call__(self, x):\n        h = self.blk({arg})\n"
        ))
    }

    #[test]
    fn argument_conflicting_with_a_constructor_dim_is_unknown() {
        let a = dim_block_call("jnp.zeros(9)");
        assert_eq!(find_shape(&a, "h"), None);
    }

    #[test]
    fn argument_matching_the_constructor_dim_is_known() {
        let a = dim_block_call("jnp.zeros(8)");
        assert_eq!(find_shape(&a, "h"), Some(&shape(&["8"])));
    }

    #[test]
    fn two_call_sites_of_one_module_get_their_own_shapes() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n 4\"], z: Float[Array, \"m 4\"]):\n        first = self.blk(x)\n        second = self.blk(z)\n        third = self.blk(mystery)\n",
            eqx_block("def __call__(self, x):", "return h")
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "first"), Some(&shape(&["n", "8"])));
        assert_eq!(find_shape(&a, "second"), Some(&shape(&["m", "8"])));
        assert_eq!(find_shape(&a, "third"), None);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
    }

    #[test]
    fn module_defined_after_its_user_is_resolved() {
        let code = format!(
            "{EQX}{}{}",
            eqx_model("out = self.blk(x)"),
            eqx_block("def __call__(self, x):", "return h")
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "out"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn torch_forward_is_the_call_target() {
        let code = format!(
            "{TORCH}class Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 8)\n    def forward(self, x):\n        return self.fc(x)\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Block()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
    }

    const TORCH_BASE: &str = "class Base(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 8)\n    def forward(self, x):\n        return self.fc(x)\n";
    const TORCH_NET: &str = "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Wide()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.blk(x)\n";

    #[test]
    fn inherited_forward_is_found_through_the_mro() {
        let code = format!("{TORCH}{TORCH_BASE}class Wide(Base):\n    pass\n{TORCH_NET}");
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
    }

    #[test]
    fn subclass_that_reassigns_a_field_of_the_base_stays_unknown() {
        let code = format!(
            "{TORCH}{TORCH_BASE}class Wide(Base):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 16)\n{TORCH_NET}"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn unknown_base_before_the_call_method_stays_unknown() {
        let code = format!(
            "{TORCH}class Wide(ExternalBase):\n    pass\n{TORCH_NET}"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn same_field_name_in_two_classes_uses_each_own_module() {
        let code = format!(
            "{TORCH}class P(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 8)\n    def forward(self, x):\n        return self.fc(x)\nclass Q(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 16)\n    def forward(self, x):\n        return self.fc(x)\nclass A(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = P()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        ya = self.blk(x)\nclass B(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Q()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        yb = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "ya"), Some(&shape(&["b", "8"])));
        assert_eq!(find_shape(&a, "yb"), Some(&shape(&["b", "16"])));
    }

    #[test]
    fn field_assigned_two_different_classes_stays_unknown() {
        let code = format!(
            "{TORCH}class P(nn.Module):\n    def forward(self, x):\n        return x\nclass Q(nn.Module):\n    def forward(self, x):\n        return x.T\nclass A(nn.Module):\n    def __init__(self, flag):\n        super().__init__()\n        if flag:\n            self.blk = P()\n        else:\n            self.blk = Q()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn field_with_unresolvable_value_never_falls_back_to_a_same_named_function() {
        let code = format!(
            "{TORCH}def blk(x):\n    return x.T\nclass A(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = make_block()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn recursive_module_terminates() {
        let code = format!(
            "{TORCH}class Rec(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.inner = Rec()\n    def forward(self, x):\n        return self.inner(x)\nclass Ping(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.o = Pong()\n    def forward(self, x):\n        return self.o(x)\nclass Pong(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.o = Ping()\n    def forward(self, x):\n        return self.o(x)\nclass Top(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.r = Rec()\n        self.p = Ping()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.r(x)\n        z = self.p(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
        assert_eq!(find_shape(&a, "z"), None);
    }

    #[test]
    fn nested_modules_are_resolved_to_the_bounded_depth() {
        let code = format!(
            "{TORCH}class Leaf(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 8)\n    def forward(self, x):\n        return self.fc(x)\nclass Mid(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.leaf = Leaf()\n    def forward(self, x):\n        return self.leaf(x)\nclass Top(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.mid = Mid()\n    def forward(self, x: Float[Array, \"b 4\"]):\n        y = self.mid(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
    }

    fn ctor_net(ctor: &str, init_sig: &str) -> String {
        format!(
            "{TORCH}class Block(nn.Module):\n    def __init__({init_sig}):\n        super().__init__()\n        self.dim = dim\n        self.fc = nn.Linear(dim, dim)\n    def forward(self, x):\n        return self.fc(x)\nclass Net(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.blk = {ctor}\n    def forward(self, x: Float[Array, \"b 8\"]):\n        y = self.blk(x)\n"
        )
    }

    #[test]
    fn constructor_argument_replaces_the_callee_parameter_name() {
        let a = analyze(&ctor_net("Block(8)", "self, dim"));
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
        let a = analyze(&ctor_net("Block(dim=8)", "self, dim"));
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
        let a = analyze(&ctor_net("Block(d)", "self, dim"));
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn unbound_constructor_parameter_stays_unknown() {
        let a = analyze(&ctor_net("Block()", "self, dim=8"));
        assert_eq!(find_shape(&a, "y"), None);
        let a = analyze(&ctor_net("Block(make_dim())", "self, dim"));
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn callee_name_is_not_confused_with_a_caller_dim_of_the_same_name() {
        // The callee parameter `dim` is 8; the caller's own dim `dim` is a
        // different quantity, so the result cannot be named by either.
        let code = ctor_net("Block(8)", "self, dim").replace("\"b 8\"", "\"b dim\"");
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn vmapped_module_peels_and_restores_the_batch_dim() {
        let code = format!(
            "{EQX}class Block(eqx.Module):\n    w: eqx.nn.Linear\n    def __init__(self, key):\n        self.w = eqx.nn.Linear(4, 8, key=key)\n    def __call__(self, x):\n        return jax.nn.relu(self.w(x))\nclass Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n 4\"]):\n        y = jax.vmap(self.blk)(x)\n"
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn vmap_of_a_layer_accepts_an_expression_argument() {
        let code = format!(
            "{EQX}class M(eqx.Module):\n    norm: eqx.nn.LayerNorm\n    def __init__(self):\n        self.norm = eqx.nn.LayerNorm(4)\n    def __call__(self, h: Float[Array, \"s 4\"], x: Float[Array, \"s 4\"]):\n        y = jax.vmap(self.norm)(h + x)\n        z = jax.vmap(self.norm)(jax.nn.relu(h) * 2.0)\n"
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["s", "4"])));
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["s", "4"])));
    }

    #[test]
    fn vmap_of_a_layer_reports_an_error_inside_the_expression_argument_once() {
        let code = format!(
            "{EQX}class M(eqx.Module):\n    norm: eqx.nn.LayerNorm\n    def __init__(self):\n        self.norm = eqx.nn.LayerNorm(4)\n    def __call__(self, h: Float[Array, \"s 4\"], x: Float[Array, \"s 5\"]):\n        y = jax.vmap(self.norm)(h + x)\n"
        );
        let a = analyze(&code);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn vmap_of_a_function_accepts_an_expression_argument() {
        let code = "import jax\ndef f(v: Float[Array, \"d\"]) -> Float[Array, \"d\"]:\n    return v\ndef g(a: Float[Array, \"n 3\"], b: Float[Array, \"n 3\"]):\n    y = jax.vmap(f)(a - b)\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "3"])));
    }

    #[test]
    fn flax_inline_module_uses_its_keyword_field() {
        let code = format!(
            "{FLAX}class Block(nn.Module):\n    features: int\n    @nn.compact\n    def __call__(self, x):\n        return nn.Dense(self.features)(x)\nclass Net(nn.Module):\n    @nn.compact\n    def __call__(self, x: Float[Array, \"b 4\"]):\n        y = Block(features=16)(x)\n        z = Block(16)(x)\n"
        );
        let a = analyze(&code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "16"])));
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["b", "16"])));
    }

    #[test]
    fn flax_positional_field_of_a_module_with_an_unknown_base_stays_unknown() {
        let code = format!(
            "{FLAX}class Block(ExternalBase):\n    features: int\n    @nn.compact\n    def __call__(self, x):\n        return nn.Dense(self.features)(x)\nclass Net(nn.Module):\n    @nn.compact\n    def __call__(self, x: Float[Array, \"b 4\"]):\n        y = Block(16)(x)\n        z = Block(features=16)(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["b", "16"])));
    }

    #[test]
    fn attribute_computed_from_a_parameter_is_not_that_parameter() {
        let code = format!(
            "{TORCH}class Block(nn.Module):\n    def __init__(self, dim, hidden):\n        super().__init__()\n        self.hid = hidden * 2\n        self.up = nn.Linear(dim, self.hid)\n    def forward(self, x):\n        return self.up(x)\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Block(8, 32)\n    def forward(self, x: Float[Array, \"b 8\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn attribute_copied_from_a_parameter_takes_its_value() {
        let code = format!(
            "{TORCH}class Block(nn.Module):\n    def __init__(self, dim, hidden):\n        super().__init__()\n        self.hid = hidden\n        self.up = nn.Linear(dim, self.hid)\n    def forward(self, x):\n        return self.up(x)\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Block(8, 32)\n    def forward(self, x: Float[Array, \"b 8\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "32"])));
    }

    #[test]
    fn flax_setup_submodule_is_resolved() {
        let code = format!(
            "{FLAX}class Block(nn.Module):\n    @nn.compact\n    def __call__(self, x):\n        return nn.Dense(8)(x)\nclass Net(nn.Module):\n    def setup(self):\n        self.blk = Block()\n    def __call__(self, x: Float[Array, \"b 4\"]):\n        y = self.blk(x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["b", "8"])));
    }

    #[test]
    fn star_arguments_make_the_binding_unknown() {
        let a = analyze(&ctor_net("Block(*d, 8)", "self, dim"));
        assert_eq!(find_shape(&a, "y"), None);
        let a = analyze(&ctor_net("Block(**d)", "self, dim"));
        assert_eq!(find_shape(&a, "y"), None);
        let a = analyze(&ctor_net("Block(8)", "self, dim").replace("self.blk(x)", "self.blk(*x)"));
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn lax_map_over_a_module_peels_the_leading_axis() {
        let code = format!(
            "{EQX}class Block(eqx.Module):\n    w: eqx.nn.Linear\n    def __init__(self, key):\n        self.w = eqx.nn.Linear(4, 8, key=key)\n    def __call__(self, x):\n        return self.w(x)\nclass Model(eqx.Module):\n    def __init__(self, key):\n        self.blk = Block(key)\n    def __call__(self, x: Float[Array, \"n 4\"]):\n        y = jax.lax.map(self.blk, x)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "8"])));
    }

    #[test]
    fn vmapped_module_with_two_arguments_needs_one_batch_dim() {
        let code = format!(
            "{EQX}class Pair(eqx.Module):\n    def __call__(self, a: Float[Array, \"d\"], b: Float[Array, \"d\"]):\n        return a + b\nclass Model(eqx.Module):\n    def __init__(self):\n        self.p = Pair()\n    def __call__(self, x: Float[Array, \"n 4\"], y: Float[Array, \"n 4\"], z: Float[Array, \"m 4\"]):\n        ok = jax.vmap(self.p)(x, y)\n        bad = jax.vmap(self.p)(x, z)\n"
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "ok"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&a, "bad"), None);
    }

    #[test]
    fn annotation_only_field_stays_unknown() {
        let code = format!(
            "{EQX}{}class Model(eqx.Module):\n    blk: Block\n    def __call__(self, x: Float[Array, \"n 4\"]):\n        out = self.blk(x)\n",
            eqx_block("def __call__(self, x):", "return h")
        );
        let a = analyze(&code);
        assert_eq!(find_shape(&a, "out"), None);
    }
}

#[cfg(test)]
mod scalar_operand_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEADER: &str = "import math\nimport torch\nimport jax.numpy as jnp\nimport numpy as np\nfrom jaxtyping import Float, Array\n";

    /// `out = <expr>` inside a function with `ys: [t, d]` and `zs: [t]`.
    fn out_of(setup: &str, expr: &str) -> (Option<Vec<String>>, usize) {
        let code = format!(
            "{HEADER}def f(ys: Float[Array, \"t d\"], zs: Float[Array, \"t\"], k, n: int):\n{setup}    out = {expr}\n"
        );
        let analysis = analyze(&code);
        (find_shape(&analysis, "out").cloned(), analysis.errors.len())
    }

    #[test]
    fn test_call_returning_a_known_scalar_keeps_the_array_shape() {
        for expr in [
            "ys / jnp.sqrt(2.0)",
            "jnp.sqrt(2.0) * ys",
            "ys / np.sqrt(2)",
            "ys - jnp.exp(-1.0)",
            "ys / jnp.sqrt(2.0 * math.pi)",
            "ys / jnp.sqrt(jnp.sqrt(2.0))",
            "ys / (2 * jnp.sqrt(2.0))",
        ] {
            assert_eq!(out_of("", expr), (Some(shape(&["t", "d"])), 0), "{expr}");
        }
    }

    #[test]
    fn test_python_scalar_calls_and_constants_keep_the_array_shape() {
        for expr in [
            "ys / math.sqrt(2.0)",
            "ys / math.sqrt(k)",
            "ys * math.pi",
            "ys / math.sqrt(n * 2)",
            "ys / (2 * math.sqrt(k))",
            "ys * (1.0 / math.sqrt(k))",
            "ys / float(k)",
            "ys / int(k)",
            "ys / len(zs)",
            "ys / ys.shape[-1]",
            "ys / zs.shape[0]",
            "ys + (-math.e)",
            "ys / jnp.sqrt(ys.shape[-1])",
        ] {
            assert_eq!(out_of("", expr), (Some(shape(&["t", "d"])), 0), "{expr}");
        }
    }

    #[test]
    fn test_names_bound_only_to_scalars_keep_the_array_shape() {
        for setup in [
            "    e = 1e-5\n",
            "    e: float = 1e-5\n",
            "    e = math.sqrt(k)\n",
            "    e = 2.0 * math.pi\n",
            "    d = ys.shape[-1]\n    e = 1.0 / math.sqrt(d)\n",
            "    e = 1.0\n    e = e * 2\n",
            "    e = 1.0\n    e *= 2\n",
        ] {
            assert_eq!(
                out_of(setup, "ys / e"),
                (Some(shape(&["t", "d"])), 0),
                "{setup}"
            );
        }
    }

    #[test]
    fn test_scalar_operand_still_reports_real_mismatches() {
        let (out, errors) = out_of("", "(ys / math.sqrt(2.0)) * zs");
        assert_eq!((out, errors), (None, 1));
        let (out, errors) = out_of("", "(ys / jnp.sqrt(2.0)) @ ys");
        assert_eq!((out, errors), (None, 1));
    }

    #[test]
    fn test_unknown_operands_are_never_treated_as_scalars() {
        for expr in [
            "ys / k",
            "ys / mystery",
            "ys / helper(zs)",
            "ys / zs.sum_all()",
            "ys / ys.shape",
            "ys / ys.shape[1:]",
            "ys / torch.sqrt(2.0)",
            "ys / jnp.sqrt(k)",
            "ys / jnp.maximum(2.0, k)",
            "ys / math.prod(zs)",
            "ys / max(k, 1)",
            "ys / abs(k)",
            "ys / (2 * k)",
            "ys / jnp.sqrt(2.0, where=zs)",
            "ys / jnp.sqrt(x=2.0)",
        ] {
            assert_eq!(out_of("", expr), (None, 0), "{expr}");
        }
    }

    #[test]
    fn test_names_with_a_non_scalar_binding_stay_unknown() {
        for setup in [
            "    e = 1.0\n    e = helper(zs)\n",
            "    e = 1.0\n    for e in helper(zs):\n        pass\n",
            "    e = 1.0\n    e += helper(zs)\n",
            "    e, g = 1.0, helper(zs)\n",
            "    e = 1.0\n    with open('x') as e:\n        pass\n",
            "    e = 1.0\n    if (e := helper(zs)) is None:\n        pass\n",
            "    e = 1.0\n    def e():\n        pass\n",
            "    e = 1.0\n    import e\n",
            "    e = 1.0\n    try:\n        pass\n    except ValueError as e:\n        pass\n",
            "    e = 1.0\n    match zs:\n        case [e, *_]:\n            pass\n",
            "    e = 1.0\n    del e\n",
            "    e = mystery\n",
            "    e = k\n",
            "    e: float\n",
            "    global e\n    e = 1.0\n",
            "    e = 1.0\n    def bump():\n        nonlocal e\n        e = helper(zs)\n    bump()\n",
        ] {
            assert_eq!(out_of(setup, "ys / e"), (None, 0), "{setup}");
        }
    }

    #[test]
    fn test_module_constants_are_scalars_only_when_bound_once() {
        let prefix = format!("{HEADER}EPS = 1e-5\nSCALE = 2.0\n");
        let analyze_with = |tail: &str, body: &str| {
            let code = format!(
                "{prefix}{tail}def f(ys: Float[Array, \"t d\"]):\n{body}    out = ys / EPS\n"
            );
            find_shape(&analyze(&code), "out").cloned()
        };
        assert_eq!(analyze_with("", ""), Some(shape(&["t", "d"])));
        assert_eq!(analyze_with("SCALE = 3.0\nEPS = 2 * SCALE\n", ""), Some(shape(&["t", "d"])));
        assert_eq!(analyze_with("EPS = mystery()\n", ""), None);
        assert_eq!(analyze_with("", "    global EPS\n"), None);
        assert_eq!(analyze_with("", "    EPS = mystery(ys)\n"), None);
        assert_eq!(analyze_with("from other import *\n", ""), None);
        assert_eq!(analyze_with("def g():\n    global EPS\n    EPS = mystery()\n", ""), None);
    }

    #[test]
    fn test_scalar_calls_bind_a_rank_0_shape() {
        let code = format!("{HEADER}def f():\n    a = jnp.sqrt(2.0)\n    b = np.exp(1)\n    c = jnp.maximum(2.0, 3.0)\n    d = jnp.sqrt(torch)\n");
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "d"), None);
    }

    #[test]
    fn test_torch_elementwise_functions_reject_python_scalars() {
        let code = format!("{HEADER}def f():\n    a = torch.sqrt(2.0)\n    b = torch.add(1.0, 2.0)\n");
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
    }

    #[test]
    fn test_shadowed_builtins_are_not_scalar_calls() {
        let code = format!(
            "{HEADER}def len(x):\n    return x\n\ndef f(ys: Float[Array, \"t d\"], zs: Float[Array, \"t\"]):\n    out = ys / len(zs)\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "out"), None);
    }

    #[test]
    fn test_parameter_or_local_shadowing_a_builtin_is_not_a_scalar_call() {
        for body in [
            "def f(x: Float[Array, \"3 4\"], len):\n    y = x * len(x)\n",
            "def f(x: Float[Array, \"3 4\"], k):\n    len = k\n    y = x * len(2)\n",
        ] {
            let analysis = analyze(&format!("{HEADER}{body}"));
            assert_eq!(find_shape(&analysis, "y"), None, "{body}");
        }
        let code = format!("{HEADER}def f(x: Float[Array, \"3 4\"]):\n    y = x * len(x)\n");
        assert_eq!(find_shape(&analyze(&code), "y"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn test_axis_reducing_jax_nn_functions_reject_scalars() {
        let code = format!(
            "{HEADER}import jax\nimport flax.linen as fnn\ndef f(x: Float[Array, \"3 4\"]):\n    a = x * jax.nn.softmax(2.0)\n    b = x * jax.nn.log_softmax(2.0)\n    c = x * fnn.softmax(2.0)\n    d = x * jax.nn.standardize(2.0)\n    e = x * jax.nn.relu(2.0)\n"
        );
        let analysis = analyze(&code);
        for name in ["a", "b", "c", "d"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
        assert_eq!(find_shape(&analysis, "e"), Some(&shape(&["3", "4"])));
    }
}

#[cfg(test)]
mod bmm_and_cross_entropy_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEADER: &str = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom jaxtyping import Float, Array\n";

    #[test]
    fn test_bmm_function_method_and_keywords() {
        let code = format!(
            "{HEADER}def f(a: Float[Array, \"b n m\"], c: Float[Array, \"b m p\"]):\n    x = torch.bmm(a, c)\n    y = a.bmm(c)\n    z = torch.bmm(input=a, mat2=c)\n    w = torch.bmm(a, c.transpose(1, 2).transpose(1, 2))\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["x", "y", "z", "w"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["b", "n", "p"])), "{name}");
        }
    }

    #[test]
    fn test_bmm_errors_for_rank_batch_and_inner_dim() {
        for (decls, call, fragment) in [
            ("a: Float[Array, \"n m\"], c: Float[Array, \"m p\"]", "torch.bmm(a, c)", "3D"),
            ("a: Float[Array, \"b n m\"], c: Float[Array, \"1 m p\"]", "torch.bmm(a, c)", "batch"),
            ("a: Float[Array, \"b n m\"], c: Float[Array, \"b k p\"]", "torch.bmm(a, c)", "dimension mismatch"),
            ("a: Float[Array, \"b n m\"], c: Float[Array, \"b k p\"]", "a.bmm(c)", "dimension mismatch"),
        ] {
            let code = format!("{HEADER}def f({decls}):\n    x = {call}\n");
            let analysis = analyze(&code);
            assert_eq!(analysis.errors.len(), 1, "{call} {decls}");
            assert!(analysis.errors[0].message.contains(fragment), "{:?}", analysis.errors[0]);
            assert_eq!(find_shape(&analysis, "x"), None);
        }
    }

    #[test]
    fn test_bmm_with_unknown_operand_is_unknown() {
        let code = format!("{HEADER}def f(a: Float[Array, \"b n m\"], c):\n    x = torch.bmm(a, c)\n");
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), None);
    }

    #[test]
    fn test_cross_entropy_through_every_import_spelling() {
        let code = format!(
            "{HEADER}def f(logits: Float[Array, \"n c\"], y: Float[Array, \"n\"]):\n    a = F.cross_entropy(logits, y)\n    b = torch.nn.functional.cross_entropy(logits, y, reduction='none')\n    c = F.cross_entropy(logits, y, reduction=\"sum\")\n    d = F.nll_loss(logits, y, reduction='none')\n    e = F.cross_entropy(logits, y, reduction=mode)\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["n"])));
        assert_eq!(find_shape(&analysis, "e"), None);
    }

    #[test]
    fn test_cross_entropy_with_probability_and_spatial_targets() {
        let code = format!(
            "{HEADER}def f(x: Float[Array, \"n c h w\"], idx: Float[Array, \"n h w\"], p: Float[Array, \"n c h w\"], v: Float[Array, \"c\"], q: Float[Array, \"c\"]):\n    a = F.cross_entropy(x, idx, reduction='none')\n    b = F.cross_entropy(x, p, reduction='none')\n    c = F.cross_entropy(v, q, reduction='none')\n    d = F.cross_entropy(x.permute(0, 2, 3, 1).reshape(-1, 5), idx.reshape(-1))\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "h", "w"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "h", "w"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&[])));
    }

    #[test]
    fn test_cross_entropy_target_mismatch_is_an_error() {
        for target in ["Float[Array, \"m\"]", "Float[Array, \"n 1\"]", "Float[Array, \"\"]"] {
            let code = format!(
                "{HEADER}def f(logits: Float[Array, \"n c\"], y: {target}):\n    loss = F.cross_entropy(logits, y)\n"
            );
            let analysis = analyze(&code);
            let expected = usize::from(!target.ends_with("\"\"]"));
            assert_eq!(analysis.errors.len(), expected, "{target}: {:?}", analysis.errors);
            assert_eq!(find_shape(&analysis, "loss"), None);
        }
    }

    #[test]
    fn test_cross_entropy_chained_from_a_layer_output() {
        let code = format!(
            "{HEADER}def f(x: Float[Array, \"n 8\"], y: Float[Array, \"n\"]):\n    head = nn.Linear(8, 3)\n    logits = head(x)\n    loss = F.cross_entropy(logits, y)\n    per_item = F.cross_entropy(logits, y, reduction='none')\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "loss"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "per_item"), Some(&shape(&["n"])));
    }
}

#[cfg(test)]
mod unpack_and_broadcast_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEADER: &str = "import math\nimport torch\nimport jax\nimport jax.numpy as jnp\nimport numpy as np\nfrom jaxtyping import Float, Array\n";

    fn names(body: &str, wanted: &[&str]) -> Vec<Option<Vec<String>>> {
        let code = format!(
            "{HEADER}def f(ys: Float[Array, \"t d\"], zs: Float[Array, \"t\"], k, xs: Float[Array, \"seq 4\"]):\n{body}"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{body}: {:?}", analysis.errors);
        wanted.iter().map(|n| find_shape(&analysis, n).cloned()).collect()
    }

    #[test]
    fn test_unpacking_a_tensor_binds_its_rows() {
        let row = Some(shape(&["4"]));
        for body in [
            "    x = torch.zeros(2, 4)\n    a, b = x\n",
            "    x = jnp.ones((2, 4))\n    a, b = x\n",
            "    x = np.zeros((2, 4)) * 2\n    a, b = x\n",
            "    x = torch.zeros(2, 4)\n    y = x\n    a, b = y\n",
        ] {
            assert_eq!(names(body, &["a", "b"]), vec![row.clone(), row.clone()], "{body}");
        }
        let code = format!(
            "{HEADER}def f(x: Float[Array, \"2 4\"], w: Float[Array, \"n 4\"]):\n    a, b = x\n    c, d = w\n"
        );
        let analysis = analyze(&code);
        for n in ["a", "b", "c", "d"] {
            assert_eq!(find_shape(&analysis, n), Some(&shape(&["4"])), "{n}");
        }
    }

    #[test]
    fn test_unpacking_a_tensor_with_a_wrong_or_unknown_count_stays_unknown() {
        for body in [
            "    x = torch.zeros(3, 4)\n    a, b = x\n",
            "    x = torch.zeros(2, 4)\n    a, b, c = x\n",
            "    x = torch.zeros(2, 4)\n    (a, b), c = x\n",
            "    x = torch.zeros(())\n    a, b = x\n",
            "    a, b = k\n",
            "    x = helper(ys)\n    a, b = x\n",
            "    x = torch.zeros(2, 4)\n    x = helper(ys)\n    a, b = x\n",
        ] {
            assert_eq!(names(body, &["a", "b"]), vec![None, None], "{body}");
        }
    }

    #[test]
    fn test_unpacking_a_seeded_untyped_parameter_stays_unknown() {
        let code = format!(
            "{HEADER}def g(p):\n    a, b = p\n    return a\n\ndef f(x: Float[Array, \"2 4\"]):\n    r = g(x)\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
    }

    #[test]
    fn test_unpacking_a_tuple_carry_keeps_the_shared_shape() {
        let code = format!(
            "{HEADER}def f(h0: Float[Array, \"hid\"], c0: Float[Array, \"hid\"], xs: Float[Array, \"seq hid\"]):\n    def body(carry, x):\n        h, c = carry\n        return (h, c), x\n    init = (h0, c0)\n    fc, ys = jax.lax.scan(body, init, xs)\n    a, b = fc\n    s = (h0, c0)\n    p, q = s\n"
        );
        let analysis = analyze(&code);
        for n in ["a", "b", "p", "q", "h", "c"] {
            assert_eq!(find_shape(&analysis, n), Some(&shape(&["hid"])), "{n}");
        }
        for n in ["fc", "init", "s"] {
            assert_eq!(find_shape(&analysis, n), None, "{n}");
        }
    }

    #[test]
    fn test_a_tuple_carry_is_no_array_operand() {
        let code = format!(
            "{HEADER}def f(x: Float[Array, \"n 8\"], w: Float[Array, \"8 4\"]):\n    t = (x, x)\n    a = jnp.asarray(t)\n    b = t @ w\n    c, d = t\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for n in ["t", "a", "b"] {
            assert_eq!(find_shape(&analysis, n), None, "{n}");
        }
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "8"])));
        assert!(!analysis.assignment_shapes.iter().any(|r| r.name == "t"));
    }

    #[test]
    fn test_two_operand_functions_broadcast_both_operands() {
        for func in [
            "jnp.maximum", "jnp.minimum", "np.maximum", "np.fmin", "jnp.add", "np.multiply",
            "jnp.power", "np.arctan2", "jnp.copysign", "np.less", "jnp.logical_and",
            "torch.add", "torch.mul", "torch.pow", "torch.maximum", "torch.minimum",
            "torch.atan2", "torch.fmod", "torch.eq",
        ] {
            let lib = if func.starts_with("torch") { "torch" } else if func.starts_with("np") { "np" } else { "jnp" };
            let zeros = if lib == "torch" { "torch.zeros(1)" } else { &format!("{lib}.zeros(1)") };
            let body = format!("    out = {func}({zeros}, ys)\n    rev = {func}(ys, {zeros})\n");
            let got = names(&body, &["out", "rev"]);
            assert_eq!(got, vec![Some(shape(&["t", "d"])); 2], "{func}");
        }
        let got = names("    out = jnp.maximum(ys, zs)\n", &["out"]);
        assert_eq!(got, vec![None], "t vs d does not broadcast");
        let got = names("    out = jnp.maximum(jnp.zeros(1), zs)\n    s = torch.add(ys, 1.0)\n", &["out", "s"]);
        assert_eq!(got, vec![Some(shape(&["t"])), Some(shape(&["t", "d"]))]);
    }

    #[test]
    fn test_two_operand_functions_stay_unknown_when_unsure() {
        for expr in [
            "jnp.maximum(jnp.zeros(3), k)",
            "jnp.maximum(ys, x2=zs)",
            "jnp.maximum(ys)",
            "jnp.maximum(ys, zs, ys)",
            "torch.add(ys, other=jnp.zeros(3))",
        ] {
            assert_eq!(names(&format!("    out = {expr}\n"), &["out"]), vec![None], "{expr}");
        }
    }

    #[test]
    fn test_clip_and_clamp_broadcast_array_bounds() {
        let got = names(
            "    a = jnp.clip(jnp.zeros(1), ys, 1.0)\n    b = ys.clamp(min=zs.unsqueeze(1))\n    c = jnp.clip(ys, 0.0, 1.0)\n    d = ys.clamp(0.0, 1.0)\n    e = torch.clip(ys, max=jnp.zeros(3))\n",
            &["a", "b", "c", "d", "e"],
        );
        assert_eq!(
            got,
            vec![
                Some(shape(&["t", "d"])),
                Some(shape(&["t", "d"])),
                Some(shape(&["t", "d"])),
                Some(shape(&["t", "d"])),
                None,
            ]
        );
    }
}

#[cfg(test)]
mod lone_quote_tests {
    use super::*;
    use tree_sitter::Parser;

    #[test]
    fn broken_strings_do_not_panic() {
        for code in [
            "def f(x: Float[Array']) -> Float[Array, \"b\"]:\n    return x\n",
            "def f(x) -> Float[Array d\"]:\n    return x\n",
            "def f(x: Float[Array, \"",
            "def f(x: Float[Array, \"\n    pass\n",
            "def f(x: Float[Array, '):\n    pass\n",
            "def f(x: Float[Array, \"\"\"):\n    pass\n",
            "def f(x: Float[Array, \"\"\"\n",
            "x: Float[Array, \"\n",
            "import jax.numpy as jnp\ny = jnp.einsum(\"",
            "import jax.numpy as jnp\ny = jnp.einsum(\",\n",
            "import jax.numpy as jnp\ny = jnp.einsum(\"\n",
            "import jax.numpy as jnp\nx: Float[Array, \"2 3\"] = g()\ny = x.reshape(\"\n",
            "from einops import rearrange\ny = rearrange(x, \"\n",
            "import torch.nn.functional as F\ny = F.cross_entropy(a, b, reduction=\"\n",
            "import torch.nn.functional as F\ny = F.pad(x, \"\n",
        ] {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
            let tree = parser.parse(code, None).unwrap();
            let read = |_: &PathBuf| -> Option<String> { None };
            let roots: Vec<PathBuf> = Vec::new();
            let _ = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None);
        }
    }
}

#[cfg(test)]
mod branch_merge_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn hint(analysis: &LayerShapeAnalysis, line: u32, name: &str) -> Option<Vec<String>> {
        analysis
            .assignment_shapes
            .iter()
            .find(|r| r.line == line && r.name == name)
            .map(|r| r.shape.clone())
    }

    const HEAD: &str = "import jax.numpy as jnp\ndef f(x: Float[Array, \"3 4\"], y: Float[Array, \"5\"], c, d):\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEAD}{body}"))
    }

    fn known(body: &str, name: &str, dims: &[&str]) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, name), Some(&shape(dims)), "{body}");
    }

    fn unknown(body: &str, name: &str) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{body}");
    }

    #[test]
    fn a_shape_assigned_in_a_branch_without_else_is_unknown_after_it() {
        let a = run("    if c:\n        x = jnp.zeros(5)\n    q = x\n");
        assert_eq!(find_shape(&a, "x"), None);
        assert_eq!(find_shape(&a, "q"), None);
        assert_eq!(hint(&a, 3, "x"), Some(shape(&["5"])));
    }

    #[test]
    fn a_branch_that_keeps_the_shape_keeps_it_known() {
        known("    if c:\n        x = jnp.ones((3, 4))\n", "x", &["3", "4"]);
    }

    #[test]
    fn exhaustive_if_else_with_equal_shapes_is_known() {
        known("    if c:\n        a = jnp.zeros(5)\n    else:\n        a = jnp.ones(5)\n", "a", &["5"]);
        unknown("    if c:\n        a = jnp.zeros(5)\n    else:\n        a = jnp.ones(6)\n", "a");
    }

    #[test]
    fn elif_chain_needs_every_arm_to_agree() {
        let same = "    if c:\n        a = jnp.zeros(5)\n    elif d:\n        a = jnp.zeros(5)\n    else:\n        a = jnp.zeros(5)\n";
        known(same, "a", &["5"]);
        let last_differs = "    if c:\n        a = jnp.zeros(5)\n    elif d:\n        a = jnp.zeros(5)\n    else:\n        a = jnp.zeros(6)\n";
        unknown(last_differs, "a");
        let middle_differs = "    if c:\n        a = jnp.zeros(5)\n    elif d:\n        a = jnp.zeros(6)\n    else:\n        a = jnp.zeros(5)\n";
        unknown(middle_differs, "a");
    }

    #[test]
    fn a_name_new_in_some_branches_keeps_the_shape_the_branches_give_it() {
        known("    if c:\n        a = jnp.zeros(5)\n", "a", &["5"]);
        known("    if c:\n        a = jnp.zeros(5)\n    elif d:\n        a = jnp.zeros(5)\n", "a", &["5"]);
        unknown("    if c:\n        a = jnp.zeros(5)\n    elif d:\n        a = jnp.zeros(6)\n", "a");
    }

    #[test]
    fn a_name_bound_by_unknown_means_before_the_branch_is_not_new() {
        unknown("    a = mystery()\n    if c:\n        a = jnp.zeros(5)\n", "a");
        let param = "import jax.numpy as jnp\ndef g(p, c):\n    if c:\n        p = jnp.zeros(5)\n";
        assert_eq!(find_shape(&analyze(param), "p"), None);
    }

    #[test]
    fn a_path_that_returns_or_raises_does_not_reach_the_join() {
        for end in ["return x", "raise ValueError()"] {
            let body = format!("    if c:\n        x = jnp.zeros(5)\n        {end}\n    q = x\n");
            let a = run(&body);
            assert_eq!(find_shape(&a, "q"), Some(&shape(&["3", "4"])), "{end}");
            assert_eq!(hint(&a, 5, "q"), Some(shape(&["3", "4"])), "{end}");
            assert_eq!(hint(&a, 3, "x"), Some(shape(&["5"])), "{end}");
        }
    }

    #[test]
    fn only_the_arms_that_fall_through_are_joined() {
        known("    if c:\n        r = jnp.zeros(5)\n    else:\n        return x\n", "r", &["5"]);
        let both = "    if c:\n        a = jnp.zeros(5)\n        return a\n    else:\n        a = jnp.zeros(6)\n        return a\n";
        unknown(both, "a");
        let agree = "    if c:\n        a = jnp.zeros(5)\n        return a\n    else:\n        a = jnp.zeros(5)\n        return a\n";
        known(agree, "a", &["5"]);
        known("    if c:\n        t = jnp.zeros(5)\n        return t\n", "t", &["5"]);
    }

    #[test]
    fn a_loop_body_may_run_zero_times() {
        unknown("    for i in range(3):\n        x = jnp.zeros(5)\n", "x");
        unknown("    while c:\n        x = jnp.zeros(5)\n", "x");
        known("    for i in range(3):\n        x = jnp.ones((3, 4))\n", "x", &["3", "4"]);
        known("    for i in range(3):\n        a = jnp.zeros(5)\n", "a", &["5"]);
    }

    #[test]
    fn loop_exits_by_break_join_with_the_end_of_the_body() {
        let body = "    while True:\n        a = jnp.zeros(5)\n        if c:\n            break\n";
        known(body, "a", &["5"]);
        let differ = "    while True:\n        if c:\n            a = jnp.zeros(6)\n            break\n        a = jnp.zeros(5)\n";
        unknown(differ, "a");
        for rebind in ["for a in y:\n                pass", "with c as a:\n                pass", "import a"] {
            let body = format!("    while True:\n        try:\n            a = jnp.zeros(5)\n            break\n        finally:\n            {rebind}\n");
            unknown(&body, "a");
        }
    }

    #[test]
    fn a_name_that_changes_in_an_iteration_is_unknown_after_the_loop() {
        let nested = "    h = x\n    for i in range(3):\n        for j in range(2):\n            h = jnp.exp(h)\n";
        known(nested, "h", &["3", "4"]);
        unknown("    for i in range(2):\n        x = x.T\n", "x");
        let through_continue = "    for i in range(3):\n        if i:\n            continue\n        x = x.T\n";
        unknown(through_continue, "x");
        let inner = "    for i in range(3):\n        for j in range(2):\n            x = x.T\n        x = x.T\n";
        unknown(inner, "x");
    }

    #[test]
    fn a_loop_target_hides_an_earlier_shape() {
        unknown("    for x in y:\n        pass\n", "x");
    }

    #[test]
    fn break_and_continue_end_the_path_where_they_stand() {
        let brk = "    for i in range(3):\n        a = jnp.zeros(5)\n        if i:\n            a = jnp.zeros(6)\n            break\n";
        unknown(brk, "a");
        let cont = "    for i in range(3):\n        if i:\n            continue\n        t = jnp.zeros(5)\n";
        known(cont, "t", &["5"]);
        let cont_keeps = "    for i in range(3):\n        if i:\n            x = jnp.zeros(5)\n            continue\n        x = jnp.ones((3, 4))\n";
        unknown(cont_keeps, "x");
        let brk_guard = "    for i in range(3):\n        if i:\n            break\n        t = jnp.zeros(5)\n";
        known(brk_guard, "t", &["5"]);
    }

    #[test]
    fn a_loop_else_runs_when_no_break_leaves_the_loop() {
        let body = "    for i in range(3):\n        pass\n    else:\n        a = jnp.zeros(5)\n";
        known(body, "a", &["5"]);
        let with_break = "    for i in range(3):\n        a = jnp.zeros(6)\n        break\n    else:\n        a = jnp.zeros(5)\n";
        unknown(with_break, "a");
    }

    #[test]
    fn try_except_joins_the_body_and_the_handlers() {
        known("    try:\n        a = jnp.zeros(5)\n    except E:\n        a = jnp.zeros(5)\n", "a", &["5"]);
        unknown("    try:\n        a = jnp.zeros(5)\n    except E:\n        a = jnp.zeros(6)\n", "a");
        unknown("    try:\n        a = jnp.zeros(5)\n    except E:\n        pass\n", "a");
        known("    try:\n        a = jnp.zeros(5)\n    except E:\n        raise\n", "a", &["5"]);
        unknown("    try:\n        x = jnp.zeros(5)\n    except E:\n        pass\n", "x");
    }

    #[test]
    fn a_handler_cannot_trust_what_the_body_wrote() {
        let body = "    try:\n        x = jnp.zeros(5)\n        z = f()\n    except E:\n        q = x\n";
        let a = run(body);
        assert_eq!(hint(&a, 6, "q"), None);
    }

    #[test]
    fn try_else_and_finally() {
        let with_else = "    try:\n        a = jnp.zeros(5)\n    except E:\n        return x\n    else:\n        a = jnp.zeros(6)\n";
        known(with_else, "a", &["6"]);
        let fin = "    try:\n        a = jnp.zeros(5)\n    finally:\n        b = jnp.zeros(6)\n";
        known(fin, "a", &["5"]);
        known(fin, "b", &["6"]);
        known("    try:\n        x = jnp.zeros(5)\n    finally:\n        x = jnp.ones((3, 4))\n", "x", &["3", "4"]);
        known("    try:\n        x = jnp.zeros(5)\n    finally:\n        pass\n", "x", &["5"]);
    }

    #[test]
    fn match_joins_its_cases() {
        let both = "    match c:\n        case 1:\n            a = jnp.zeros(5)\n        case _:\n            a = jnp.zeros(5)\n";
        known(both, "a", &["5"]);
        let differ = "    match c:\n        case 1:\n            a = jnp.zeros(5)\n        case _:\n            a = jnp.zeros(6)\n";
        unknown(differ, "a");
        let open = "    match c:\n        case 1:\n            x = jnp.zeros(5)\n";
        unknown(open, "x");
        let guarded = "    match c:\n        case 1:\n            x = jnp.zeros(5)\n        case n if n:\n            x = jnp.zeros(5)\n";
        unknown(guarded, "x");
        let capture = "    match c:\n        case 1:\n            x = jnp.zeros(5)\n        case n:\n            x = jnp.zeros(5)\n";
        known(capture, "x", &["5"]);
    }

    #[test]
    fn with_blocks_run_once_unless_they_swallow_errors() {
        known("    with ctx():\n        a = jnp.zeros(5)\n", "a", &["5"]);
                unknown("    with suppress(E):\n        a = jnp.zeros(5)\n", "a");
        unknown("    with pytest.raises(E):\n        x = jnp.zeros(5)\n", "x");
    }

    #[test]
    fn nested_branches_join_from_the_inside_out() {
        unknown("    if c:\n        if d:\n            x = jnp.zeros(5)\n", "x");
        known("    if c:\n        if d:\n            x = jnp.ones((3, 4))\n", "x", &["3", "4"]);
        let inner_returns = "    if c:\n        if d:\n            x = jnp.zeros(5)\n            return x\n        q = x\n";
        let a = run(inner_returns);
        assert_eq!(hint(&a, 6, "q"), Some(shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "x"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn each_arm_starts_from_the_state_before_the_branch() {
        let body = "    if c:\n        x = jnp.zeros(5)\n    else:\n        q = x\n";
        let a = run(body);
        assert_eq!(hint(&a, 5, "q"), Some(shape(&["3", "4"])));
    }

    #[test]
    fn code_after_a_branch_reads_the_joined_state() {
        let body = "    if c:\n        a = jnp.zeros(5)\n    else:\n        a = jnp.zeros(5)\n    b = a + jnp.ones(5)\n";
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(hint(&a, 6, "b"), Some(shape(&["5"])));
    }

    #[test]
    fn a_mismatch_inside_a_branch_still_reports() {
        let body = "    b = jnp.zeros((6, 2))\n    if c:\n        x = jnp.zeros((3, 5))\n        z = x @ b\n";
        let a = run(body);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn no_error_comes_from_a_state_the_join_made_unknown() {
        let body = "    b = jnp.zeros((4, 2))\n    if c:\n        x = jnp.zeros((3, 7))\n    z = x @ b\n";
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        let straight = "    b = jnp.zeros((4, 2))\n    x = jnp.zeros((3, 7))\n    z = x @ b\n";
        assert_eq!(run(straight).errors.len(), 1);
    }

    #[test]
    fn a_called_helper_with_a_branch_is_joined_too() {
        let same = "import jax.numpy as jnp\ndef helper(a: Float[Array, \"3 4\"], c):\n    if c:\n        b = a\n    else:\n        b = jnp.ones((3, 4))\n    return b\ndef outer(x: Float[Array, \"3 4\"]):\n    r = helper(x, True)\n";
        assert_eq!(find_shape(&analyze(same), "r"), Some(&shape(&["3", "4"])));
        let differ = "import jax.numpy as jnp\ndef helper(a: Float[Array, \"3 4\"], c):\n    if c:\n        b = a\n    else:\n        b = jnp.ones(5)\n    return b\ndef outer(x: Float[Array, \"3 4\"]):\n    r = helper(x, True)\n";
        assert_eq!(find_shape(&analyze(differ), "r"), None);
        let seeded = "import jax.numpy as jnp\ndef helper(a, c):\n    if c:\n        b = a\n    else:\n        b = jnp.ones(5)\n    return b\ndef outer(x: Float[Array, \"3 4\"]):\n    r = helper(x, True)\n";
        assert_eq!(find_shape(&analyze(seeded), "r"), None);
    }

    #[test]
    fn a_branch_in_one_function_does_not_touch_another() {
        let code = "def f(x: Float[Array, \"3 4\"], c):\n    if c:\n        x = jnp.zeros(5)\ndef g(x: Float[Array, \"3 4\"]):\n    y = x\n";
        let a = analyze(code);
        assert_eq!(hint(&a, 4, "y"), Some(shape(&["3", "4"])));
    }

    #[test]
    fn loops_nested_in_branches_still_close_by_themselves() {
        let body = "    if c:\n        for i in range(2):\n            x, y2 = y, x\n";
        let a = run(body);
        assert_eq!(find_shape(&a, "x"), None);
    }

    #[test]
    fn module_level_branches_join_too() {
        let code = "import jax.numpy as jnp\nx = jnp.zeros(5)\nif c:\n    x = jnp.zeros(6)\nz = x\nw = jnp.zeros(3)\nif c:\n    w = jnp.zeros(3)\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "x"), None);
        assert_eq!(find_shape(&a, "z"), None);
        assert_eq!(find_shape(&a, "w"), Some(&shape(&["3"])));
        let fresh = "import jax.numpy as jnp\nif c:\n    n = jnp.zeros(6)\n";
        assert_eq!(find_shape(&analyze(fresh), "n"), Some(&shape(&["6"])));
        let read_before = "import jax.numpy as jnp\nprint(n)\nif c:\n    n = jnp.zeros(6)\n";
        assert_eq!(find_shape(&analyze(read_before), "n"), None);
        let in_function_only = "import jax.numpy as jnp\ndef g(n):\n    return n\nif c:\n    n = jnp.zeros(6)\n";
        assert_eq!(find_shape(&analyze(in_function_only), "n"), Some(&shape(&["6"])));
        let global_elsewhere = "import jax.numpy as jnp\ndef g():\n    global n\n    n = 1\nif c:\n    n = jnp.zeros(6)\n";
        assert_eq!(find_shape(&analyze(global_elsewhere), "n"), None);
        let wildcard = "import jax.numpy as jnp\nfrom lib import *\nif c:\n    n = jnp.zeros(6)\n";
        assert_eq!(find_shape(&analyze(wildcard), "n"), None);
        let main_block = "import jax.numpy as jnp\nif __name__ == \"__main__\":\n    a = jnp.zeros(4)\n    b = a + a\n";
        assert_eq!(find_shape(&analyze(main_block), "b"), Some(&shape(&["4"])));
    }
}

#[cfg(test)]
mod module_global_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    const HEAD: &str = "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEAD}{body}"))
    }

    fn known(body: &str, name: &str, dims: &[&str]) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}\n{body}", a.errors);
        let want: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
        assert_eq!(find_shape(&a, name), Some(&want), "{body}");
    }

    fn unknown(body: &str, name: &str) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}\n{body}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{body}");
    }

    #[test]
    fn a_function_reads_a_module_global() {
        known("W = jnp.zeros((3, 4))\ndef f():\n    out = W\n    return out\n", "out", &["3", "4"]);
        known(
            "W = jnp.zeros((3, 4))\ndef f(x: Float[Array, \"3 4\"]):\n    out = x + W\n",
            "out",
            &["3", "4"],
        );
        known(
            "W = jnp.zeros((3, 4))\nclass C:\n    def m(self):\n        out = W\n",
            "out",
            &["3", "4"],
        );
        known(
            "W = jnp.zeros((3, 4))\ndef f():\n    def g():\n        return W\n    out = W\n",
            "out",
            &["3", "4"],
        );
    }

    #[test]
    fn a_true_mismatch_with_a_module_global_is_an_error() {
        let a = run("W = jnp.zeros((3, 4))\ndef f(x: Float[Array, \"5 6\"]):\n    out = x @ W\n");
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn a_name_the_function_binds_shadows_the_global() {
        let g = "W = jnp.zeros((3, 4))\n";
        unknown(&format!("{g}def f(W):\n    out = W\n"), "out");
        unknown(&format!("{g}def f():\n    W = mystery()\n    out = W\n"), "out");
        unknown(&format!("{g}def f():\n    for W in ws:\n        pass\n    out = W\n"), "out");
        known(&format!("{g}def f():\n    W = jnp.zeros(7)\n    out = W\n"), "out", &["7"]);
        unknown(
            &format!("{g}def f():\n    W = mystery()\n    def h():\n        return W\n    out = W\n"),
            "out",
        );
        unknown(
            &format!("{g}def f():\n    def h():\n        W = mystery()\n    W = mystery()\n    out = W\n"),
            "out",
        );
        unknown(&format!("{g}def f():\n    out = W\n    W = mystery()\n"), "out");
        known(&format!("{g}def f(W: Float[Array, \"9\"]):\n    out = W\n"), "out", &["9"]);
    }

    #[test]
    fn a_global_statement_makes_the_name_unknown() {
        let code = "W = jnp.zeros((3, 4))\ndef set_it():\n    global W\n    W = jnp.zeros(5)\ndef f():\n    out = W\n";
        unknown(code, "out");
        let reads = "W = jnp.zeros((3, 4))\ndef f():\n    global W\n    out = W\n";
        unknown(reads, "out");
    }

    #[test]
    fn the_global_needs_one_shape_over_all_assignments() {
        known("W = jnp.zeros(3)\nW = jnp.ones(3)\ndef f():\n    out = W\n", "out", &["3"]);
        unknown("W = jnp.zeros(3)\nW = jnp.ones(4)\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\nW = mystery()\ndef f():\n    out = W\n", "out");
        unknown(
            "if c:\n    W = jnp.zeros(3)\nelse:\n    W = jnp.zeros(4)\ndef f():\n    out = W\n",
            "out",
        );
        known(
            "if c:\n    W = jnp.zeros(3)\nelse:\n    W = jnp.ones(3)\ndef f():\n    out = W\n",
            "out",
            &["3"],
        );
        unknown("W = jnp.zeros(3)\nfor i in r:\n    W = W[None]\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\nW += jnp.ones((2, 3))\ndef f():\n    out = W\n", "out");
        known("W = jnp.zeros(3)\nW += 1\ndef f():\n    out = W\n", "out", &["3"]);
    }

    #[test]
    fn an_assignment_after_the_function_makes_the_global_unknown() {
        unknown("def f():\n    out = W\nW = jnp.zeros(3)\n", "out");
        unknown("W = jnp.zeros(3)\ndef f():\n    out = W\nW = jnp.zeros(3)\n", "out");
        unknown(
            "def g(x):\n    return f(x)\nW = jnp.zeros(3)\nW = jnp.zeros(4)\ndef f(x):\n    out = x + W\n    return out\ndef h():\n    return g(jnp.zeros(4))\n",
            "out",
        );
    }

    #[test]
    fn other_bindings_of_the_global_name_make_it_unknown() {
        unknown("W, V = mystery()\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\nfor W in ws:\n    pass\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\nwith ctx() as W:\n    pass\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\nimport W\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\ndef W():\n    pass\ndef f():\n    out = W\n", "out");
        unknown("W = jnp.zeros(3)\ndel W\ndef f():\n    out = W\n", "out");
        unknown("from lib import *\nW = jnp.zeros(3)\ndef f():\n    out = W\n", "out");
        unknown("W = V = jnp.zeros(3)\ndef f():\n    out = W\n", "out");
        unknown("import W\ndef f():\n    out = W\n", "out");
    }

    #[test]
    fn a_class_body_name_is_not_a_module_global() {
        unknown("class C:\n    W = jnp.zeros(9)\n    def m(self):\n        out = W\n", "out");
        unknown(
            "W = jnp.zeros((3, 4))\nclass C:\n    W = jnp.zeros(9)\ndef f():\n    out = W\n",
            "out",
        );
        unknown(
            "W = jnp.zeros((3, 4))\nclass C:\n    W: Float[Array, \"9\"]\ndef f():\n    out = W\n",
            "out",
        );
        unknown(
            "W = jnp.zeros((3, 4))\nclass C:\n    def W(self):\n        pass\ndef f():\n    out = W\n",
            "out",
        );
        known(
            "W = jnp.zeros((3, 4))\nclass C:\n    V = jnp.zeros(9)\ndef f():\n    out = W\n",
            "out",
            &["3", "4"],
        );
    }

    #[test]
    fn a_class_body_read_does_not_see_a_global_through_a_function() {
        known(
            "W = jnp.zeros((3, 4))\ndef f():\n    class C:\n        W = jnp.zeros(9)\n        def m(self):\n            return W\n    out = W\n",
            "out",
            &["3", "4"],
        );
        unknown(
            "W = jnp.zeros((3, 4))\ndef f():\n    class C:\n        W = jnp.zeros(9)\n        out = W\n",
            "out",
        );
    }

    #[test]
    fn self_attribute_does_not_read_a_module_global() {
        unknown(
            "W = jnp.zeros((3, 4))\nclass C:\n    def m(self):\n        out = self.W\n",
            "out",
        );
        known(
            "class C:\n    W: Float[Array, \"3 4\"]\n    def m(self):\n        out = self.W\n",
            "out",
            &["3", "4"],
        );
        unknown(
            "W = jnp.zeros((3, 4))\nclass C:\n    W: Float[Array, \"3 4\"]\n    def m(self):\n        out = self.W\n",
            "out",
        );
    }

    #[test]
    fn a_lambda_or_comprehension_target_shadows_the_global() {
        let g = "W = jnp.zeros((3, 4))\n";
        unknown(&format!("{g}def f(xs):\n    out = jnp.stack([W for W in xs])\n"), "out");
        unknown(&format!("{g}def f():\n    out = (lambda W: W)(1)\n"), "out");
    }

    #[test]
    fn module_constants_work_as_dims() {
        known("D = 64\ndef f():\n    out = jnp.zeros((D, D))\n", "out", &["64", "64"]);
        known(
            "D = 64\ndef f(x: Float[Array, \"64 64\"]):\n    out = x + jnp.zeros((D, D))\n",
            "out",
            &["64", "64"],
        );
        known("D = 64\nD = 32\ndef f():\n    out = jnp.zeros((D, D))\n", "out", &["D", "D"]);
        known("D = 64\ndef f(D):\n    out = jnp.zeros((D, D))\n", "out", &["D", "D"]);
    }

    #[test]
    fn two_names_for_one_constant_are_not_a_mismatch() {
        let two = "D = 64\nE = 64\ndef f():\n    x = jnp.zeros((D,))\n    y = jnp.zeros((E,))\n    z = x + y\n";
        known(two, "z", &["64"]);
        let right = "D = 64\ndef f(a: Float[Array, \"64 3\"]):\n    z = a + jnp.zeros((D, 3))\n";
        known(right, "z", &["64", "3"]);
        let wrong = "D = 64\ndef f(a: Float[Array, \"65 3\"]):\n    z = a + jnp.zeros((D, 3))\n";
        assert_eq!(run(wrong).errors.len(), 1);
        known("D = 64\ndef f(x):\n    out = jnp.zeros((D, x.shape[0]))\n", "out", &["64", "x.shape[0]"]);
    }

    #[test]
    fn a_constant_needs_one_literal_and_no_other_binding() {
        let sym = &["D", "3"];
        known("D = 64\ndef f():\n    D += 1\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 1\nD += 1\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 64\ndef f():\n    global D\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 64\nD = len(xs)\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 0x40\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 64\nclass C:\n    D = 32\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D = 64\ndef f(D: int):\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D: Float[Array, \"2\"] = 64\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", sym);
        known("D: int = 64\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", &["64", "3"]);
        known("D = 64\ndef f():\n    D = 65\n    out = jnp.zeros((D, 3))\n", "out", &["65", "3"]);
        known("D = 64\nclass C:\n    D = 32\n    def m(self):\n        out = jnp.zeros((D, 3))\n", "out", &["64", "3"]);
    }

    #[test]
    fn a_class_inside_a_function_does_not_bind_in_the_function() {
        unknown("def f():\n    class C:\n        W = jnp.zeros(9)\n    out = W\n", "out");
        unknown(
            "def f():\n    class C:\n        W: Float[Array, \"9\"]\n    out = W\n",
            "out",
        );
    }

    #[test]
    fn a_module_level_read_is_unchanged() {
        known("W = jnp.zeros((3, 4))\nout = W\n", "out", &["3", "4"]);
    }

    #[test]
    fn a_constant_named_like_an_annotation_symbol_equals_the_symbol() {
        let symbol = "D = 64\ndef f(x: Float[Array, \"D 3\"]):\n    y = x + jnp.zeros((D, 3))\n";
        known(symbol, "y", &["D", "3"]);
        let local = "def f(x: Float[Array, \"D 3\"]):\n    D = 64\n    y = x + jnp.zeros((D, 3))\n";
        known(local, "y", &["D", "3"]);
        // Another function's annotation uses the symbol: the name is shared.
        let apart = "D = 64\ndef g(x: Float[Array, \"D 3\"]):\n    return x\ndef f():\n    out = jnp.zeros((D, 3))\n";
        known(apart, "out", &["64", "3"]);
        let inside = "D = 64\ndef g(x: Float[Array, \"2*D 3\"]):\n    return x\ndef f():\n    out = jnp.zeros((D, 3))\n";
        known(inside, "out", &["64", "3"]);
        known("D = 64\ndef g(x: Float[Array, \"E 3\"]):\n    return x\ndef f():\n    out = jnp.zeros((D, 3))\n", "out", &["64", "3"]);
    }

    #[test]
    fn many_distinct_globals_cost_a_linear_scan() {
        let time = |n: usize| {
            let mut code = String::from(HEAD);
            for i in 0..n {
                code.push_str(&format!("W{i} = jnp.zeros((3, 4))\nD{i} = {i}\n"));
            }
            for i in 0..n {
                code.push_str(&format!(
                    "def g{i}(a):\n    out = W{i}\n    z = jnp.zeros((D{i}, 2))\n    return out\n"
                ));
            }
            let started = std::time::Instant::now();
            let a = analyze(&code);
            let elapsed = started.elapsed();
            assert_eq!(find_shape(&a, "out"), Some(&vec!["3".to_string(), "4".to_string()]));
            elapsed.as_secs_f64()
        };
        let best = |n| (0..3).map(|_| time(n)).fold(f64::MAX, f64::min);
        let (small, large) = (best(300), best(1200));
        // Four times the globals: a linear scan takes about four times as
        // long, a scan per global about sixteen times.
        assert!(large < small * 10.0, "{small:.3}s for 300, {large:.3}s for 1200");
    }
}

#[cfg(test)]
mod comprehension_lambda_tests {
    use super::*;
    use tree_sitter::Parser;

    const HEAD: &str = "import jax\nimport jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\n\ndef f(xs: Float[Array, \"5 3 4\"], w: Float[Array, \"4 2\"], v: Float[Array, \"5 4\"], sym: Float[Array, \"n 3 4\"], y: Float[Array, \"3 4\"], s: Float[Array, \"\"], lst):\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        let code = format!("{HEAD}{body}");
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(&code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), &code, &[], |_| None, 5, None).unwrap()
    }

    fn body(lines: &[&str]) -> String {
        lines.iter().map(|l| format!("    {l}\n")).collect()
    }

    #[track_caller]
    fn known(lines: &[&str], name: &str, dims: &[&str]) {
        let a = run(&body(lines));
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        let want: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
        assert_eq!(find_shape(&a, name), Some(&want), "{lines:?}");
    }

    #[track_caller]
    fn unknown(lines: &[&str], name: &str) {
        let a = run(&body(lines));
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{lines:?}");
    }

    #[test]
    fn stack_of_a_comprehension_over_an_array() {
        known(&["out = jnp.stack([x @ w for x in xs])"], "out", &["5", "3", "2"]);
        unknown(&["out = jnp.stack([r * 2.0 for x in xs for r in x])"], "out");
    }

    #[test]
    fn concatenate_of_a_comprehension() {
        known(&["out = jnp.concatenate([x @ w for x in xs])"], "out", &["15", "2"]);
        known(&["out = jnp.concatenate([x @ w for x in xs], axis=1)"], "out", &["3", "10"]);
    }

    #[test]
    fn comprehension_over_enumerate_zip_and_range() {
        known(&["out = jnp.stack([x * i for i, x in enumerate(xs)])"], "out", &["5", "3", "4"]);
        known(&["out = jnp.stack([a + b for a, b in zip(xs, xs)])"], "out", &["5", "3", "4"]);
        known(&["out = jnp.stack([a[0] + b for a, b in zip(xs, v)])"], "out", &["5", "4"]);
        known(&["out = jnp.stack([w * k for k in range(3)])"], "out", &["3", "4", "2"]);
        known(&["out = jnp.stack([xs[i] for i in range(5)])"], "out", &["5", "3", "4"]);
        known(&["out = jnp.stack([xs[i] for i in range(len(xs))])"], "out", &["5", "3", "4"]);
        known(&["out = jnp.stack([xs[i] for i in range(xs.shape[0])])"], "out", &["5", "3", "4"]);
        known(&["out = jnp.stack([w for _ in range(2, 5)])"], "out", &["3", "4", "2"]);
        known(&["out = jnp.stack([w for _ in range(5, 0, -2)])"], "out", &["3", "4", "2"]);
        let max = i64::MAX;
        unknown(&[&format!("out = jnp.stack([w for _ in range(-{max}, {max})])")], "out");
        unknown(&[&format!("out = jnp.stack([w for _ in range({max}, -{max}, -1)])")], "out");
        unknown(&["out = jnp.stack([w for _ in range(5, 0, -9223372036854775808)])"], "out");
        known(&[&format!("out = jnp.stack([w for _ in range(0, {max}, {max})])")], "out", &["1", "4", "2"]);
    }

    #[test]
    fn zip_stops_at_the_shortest_array() {
        known(&["out = jnp.stack([a[0] for a, b in zip(xs, xs[:3])])"], "out", &["3", "4"]);
        let short = "Float[Array, \"2 3 4\"]";
        let code = format!(
            "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\n\ndef g(xs: Float[Array, \"5 3 4\"], zs: {short}):\n    out = jnp.stack([a + b for a, b in zip(xs, zs)])\n"
        );
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let a = analyze_layer_shapes(tree.root_node(), &code, &[], |_| None, 5, None).unwrap();
        assert_eq!(find_shape(&a, "out"), Some(&vec!["2".to_string(), "3".into(), "4".into()]));
    }

    #[test]
    fn sum_of_a_comprehension() {
        known(&["out = sum(x @ w for x in xs)"], "out", &["3", "2"]);
        known(&["out = sum([x @ w for x in xs])"], "out", &["3", "2"]);
        unknown(&["out = sum(x @ w for x in sym)"], "out");
        unknown(&["out = sum(x @ w for x in xs if lst)"], "out");
        unknown(&["sum = lst", "out = sum(x @ w for x in xs)"], "out");
    }

    #[test]
    fn the_loop_variable_does_not_leak() {
        known(
            &["x = jnp.ones((7,))", "out = jnp.stack([x @ w for x in xs])", "after = x"],
            "after",
            &["7"],
        );
        known(
            &["x = jnp.ones((7,))", "out = jnp.stack([x @ w for x in xs])"],
            "out",
            &["5", "3", "2"],
        );
    }

    #[test]
    fn a_loop_variable_of_unknown_shape_hides_the_outer_name() {
        unknown(&["x = jnp.ones((7, 7))", "out = jnp.stack([x * 1.0 for x in lst])"], "out");
        unknown(&["x = jnp.ones((7, 7))", "out = jnp.stack([x * 1.0 for x, _ in lst])"], "out");
        unknown(&["x = jnp.ones((7, 7))", "out = jnp.stack([x * 1.0 for x in (y, y)])"], "out");
    }

    #[test]
    fn an_integer_loop_variable_hides_an_outer_integer() {
        known(&["i = 3", "out = jnp.stack([y * i for i in range(2)])"], "out", &["2", "3", "4"]);
        unknown(&["i = jnp.ones(3)", "out = jnp.stack([y * i for i in lst])"], "out");
    }

    #[test]
    fn only_arrays_are_iterated_by_their_first_axis() {
        unknown(&["out = jnp.stack([x @ w for x in sym])"], "out");
        unknown(&["out = jnp.stack([x for x in s])"], "out");
        unknown(&["out = jnp.stack([x for x in lst])"], "out");
        unknown(&["pair = (y, y)", "out = jnp.stack([x * 2.0 for x in pair])"], "out");
        unknown(&["out = jnp.stack([x * 2.0 for x in (y, y)])"], "out");
        unknown(&["out = jnp.stack([x for x in xs if lst])"], "out");
        unknown(&["out = jnp.stack(x @ w for x in xs)"], "out");
    }

    #[test]
    fn a_dim_that_names_the_loop_variable_is_unknown() {
        unknown(&["out = jnp.stack([jnp.zeros((i, 2)) for i in range(3)])"], "out");
        unknown(&["out = jnp.stack([jnp.zeros((2, k)) for k in range(3)])"], "out");
    }

    #[test]
    fn nested_comprehensions() {
        known(
            &["out = jnp.stack([jnp.stack([r * 2.0 for r in x]) for x in xs])"],
            "out",
            &["5", "3", "4"],
        );
    }

    #[test]
    fn a_comprehension_body_error_is_reported_once() {
        let a = run(&body(&["bad = jnp.ones((9, 2))", "out = jnp.stack([x @ bad for x in xs])"]));
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn an_error_that_depends_on_the_loop_integer_is_not_reported() {
        let a = run(&body(&["out = jnp.stack([y @ jnp.ones((i, 2)) for i in range(5, 7)])"]));
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        let a = run(&body(&["out = y @ jnp.ones((7, 2))"]));
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        // A loop variable named like a word of the message reads no dim.
        let a = run(&body(&["out = jnp.stack([jnp.concatenate([y, v], axis=1) for axis in range(3)])"]));
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn a_direct_lambda_call() {
        known(&["out = (lambda x: x * 2)(y)"], "out", &["3", "4"]);
        known(&["out = (lambda x: x @ w)(xs)"], "out", &["5", "3", "2"]);
        known(&["out = (lambda a, b: a @ b)(y, w)"], "out", &["3", "2"]);
        known(&["out = (lambda a, b: a @ b)(b=w, a=y)"], "out", &["3", "2"]);
        known(&["out = (lambda x, k=2: x * k)(y)"], "out", &["3", "4"]);
        known(&["out = (lambda x: x @ w)(y)"], "out", &["3", "2"]);
    }

    #[test]
    fn a_lambda_parameter_hides_the_outer_name() {
        known(&["x = jnp.ones((9, 9))", "out = (lambda x: x * 2)(y)"], "out", &["3", "4"]);
        unknown(&["x = jnp.ones((9, 9))", "out = (lambda x: x * 2)(lst)"], "out");
        unknown(&["x = jnp.ones((9, 9))", "out = (lambda x, z: x * 2)(z=y)"], "out");
        known(&["n = 3", "out = (lambda n: n * 2)(y)"], "out", &["3", "4"]);
        unknown(&["n = 3", "out = (lambda n: y * n)(lst)"], "out");
        known(&["out = (lambda: y * 2)()"], "out", &["3", "4"]);
    }

    #[test]
    fn a_lambda_that_is_not_modelled_is_unknown() {
        unknown(&["out = (lambda x: x * 2)(y, y)"], "out");
        unknown(&["out = (lambda x: x * 2)()"], "out");
        unknown(&["out = (lambda x: (x, x))(y)"], "out");
        unknown(&["out = (lambda *a: a[0])(y)"], "out");
        unknown(&["out = (lambda x, **k: x)(y)"], "out");
        unknown(&["out = (lambda x: x)(*lst)"], "out");
        unknown(&["out = (lambda x: x)(y, z=1)"], "out");
    }

    #[test]
    fn vmap_of_a_lambda() {
        known(&["out = jax.vmap(lambda x: x @ w)(xs)"], "out", &["5", "3", "2"]);
        known(&["out = jax.vmap(lambda a, b: a + b)(xs, xs)"], "out", &["5", "3", "4"]);
        known(&["out = jax.vmap(lambda x: x.sum(), in_axes=1)(xs)"], "out", &["3"]);
        known(&["out = jax.vmap(lambda x: x * 2.0, out_axes=1)(xs)"], "out", &["3", "5", "4"]);
        known(&["out = jax.vmap(jax.vmap(lambda x: x * 2.0))(xs)"], "out", &["5", "3", "4"]);
        known(&["out = jax.vmap(lambda x: jax.vmap(lambda r: r * 2.0)(x))(xs)"], "out", &["5", "3", "4"]);
    }

    #[test]
    fn vmap_of_a_lambda_that_is_not_modelled_is_unknown() {
        unknown(&["out = jax.vmap(lambda x: x * 2.0, in_axes=(0,))(xs)"], "out");
        unknown(&["out = jax.vmap(lambda x: (x, x))(xs)"], "out");
        unknown(&["out = jax.vmap(lambda x: x * 2.0)(lst)"], "out");
        unknown(&["out = jax.vmap(lambda x: x * 2.0)((xs, xs))"], "out");
        unknown(&["out = jax.vmap(lambda x: x * 2.0)(x=xs)"], "out");
        unknown(&["x = jnp.ones((9, 9))", "out = jax.vmap(lambda x: x * 2.0)(lst)"], "out");
    }

    #[test]
    fn vmap_of_a_lambda_reports_a_body_error_once() {
        let a = run(&body(&["bad = jnp.ones((9, 2))", "out = jax.vmap(lambda x: x @ bad)(xs)"]));
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        let a = run(&body(&["out = jax.vmap(lambda a, b: a + b)(xs, w)"]));
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
    }

    #[test]
    fn lax_map_with_a_lambda() {
        known(&["out = jax.lax.map(lambda x: x @ w, xs)"], "out", &["5", "3", "2"]);
        unknown(&["out = jax.lax.map(lambda x: x @ w, (xs, xs))"], "out");
        unknown(&["out = jax.lax.map(lambda x: x @ w, lst)"], "out");
    }

    #[test]
    fn scan_with_a_lambda() {
        known(
            &["c, ys = jax.lax.scan(lambda c, x: (c + x, c * x), v[0], v)"],
            "ys",
            &["5", "4"],
        );
        known(
            &["c, ys = jax.lax.scan(lambda c, x: (c + x, c * x), v[0], v)"],
            "c",
            &["4"],
        );
        known(
            &["c, ys = jax.lax.scan(lambda c, x: (c, x.sum()), v[0], v)"],
            "ys",
            &["5"],
        );
        unknown(&["c, ys = jax.lax.scan(lambda c, x: (c, None), v[0], v)"], "ys");
        unknown(&["c, ys = jax.lax.scan(lambda c, x: (c, (x, x)), v[0], v)"], "ys");
        unknown(&["c, ys = jax.lax.scan(lambda c, x: c, v[0], v)"], "ys");
        unknown(&["c, ys = jax.lax.scan(lambda c, x: (c, x), v[0], lst)"], "ys");
        unknown(&["c, ys = jax.lax.scan(lambda c, x, z: (c, x), v[0], v)"], "ys");
    }

    #[test]
    fn a_lambda_reads_the_variables_around_it() {
        known(&["k = jnp.ones((4,))", "out = jax.vmap(lambda x: x * k)(xs)"], "out", &["5", "3", "4"]);
        known(&["out = jax.vmap(lambda x: x @ w)(xs)", "w = jnp.ones((9, 9))"], "out", &["5", "3", "2"]);
    }

    #[test]
    fn module_level_comprehensions_and_lambdas() {
        let code = "import jax\nimport jax.numpy as jnp\n\nxs = jnp.ones((5, 3, 4))\nw = jnp.ones((4, 2))\nstacked = jnp.stack([x @ w for x in xs])\nmapped = jax.vmap(lambda x: x @ w)(xs)\ntotal = sum(x @ w for x in xs)\n";
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        let a = analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        let want = |dims: &[&str]| Some(dims.iter().map(|d| d.to_string()).collect::<Vec<_>>());
        assert_eq!(find_shape(&a, "stacked").cloned(), want(&["5", "3", "2"]));
        assert_eq!(find_shape(&a, "mapped").cloned(), want(&["5", "3", "2"]));
        assert_eq!(find_shape(&a, "total").cloned(), want(&["3", "2"]));
        assert_eq!(find_shape(&a, "x"), None);
    }
}

#[cfg(test)]
mod loss_layer_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEADER: &str = "import torch\nimport torch.nn as nn\nfrom torch.nn import MSELoss as Mse\nfrom jaxtyping import Float, Array\n";

    fn run(params: &str, body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEADER}def f({params}):\n{body}"))
    }

    const CLASS_PARAMS: &str = "logits: Float[Array, \"n c\"], y: Float[Array, \"n\"]";

    #[test]
    fn classification_losses_follow_the_functional_rule_and_the_constructor_reduction() {
        let body = "    ce = nn.CrossEntropyLoss()\n    a = ce(logits, y)\n    ce_none = nn.CrossEntropyLoss(reduction='none')\n    b = ce_none(logits, y)\n    nll = nn.NLLLoss(reduction=\"sum\")\n    c = nll(logits, y)\n    nll_none = nn.NLLLoss(None, None, -100, None, 'none')\n    d = nll_none(logits, y)\n    e = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1, reduction='none')(logits, y)\n    g = nn.CrossEntropyLoss(ignore_index=3)(logits, y)\n";
        let analysis = run(CLASS_PARAMS, body);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for (name, dims) in [("a", &[][..]), ("b", &["n"]), ("c", &[]), ("d", &["n"]), ("e", &["n"]), ("g", &[])] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(dims)), "{name}");
        }
    }

    #[test]
    fn cross_entropy_takes_probability_targets_and_spatial_inputs_but_nll_does_not() {
        let params = "x: Float[Array, \"n c h w\"], idx: Float[Array, \"n h w\"], p: Float[Array, \"n c h w\"]";
        let body = "    ce = nn.CrossEntropyLoss(reduction='none')\n    a = ce(x, idx)\n    b = ce(x, p)\n    nll = nn.NLLLoss(reduction='none')\n    c = nll(x, idx)\n    d = nll(x, p)\n";
        let analysis = run(params, body);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "h", "w"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "h", "w"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "h", "w"])));
        assert_eq!(find_shape(&analysis, "d"), None);
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
    }

    #[test]
    fn classification_loss_target_mismatch_is_an_error() {
        let params = "logits: Float[Array, \"n c\"], y: Float[Array, \"m\"]";
        let analysis = run(params, "    loss = nn.CrossEntropyLoss()(logits, y)\n");
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "loss"), None);
    }

    #[test]
    fn regression_losses_reduce_to_a_scalar_or_keep_the_broadcast_shape() {
        let params = "p: Float[Array, \"n c\"], t: Float[Array, \"n c\"], row: Float[Array, \"c\"], col: Float[Array, \"n 1\"]";
        let mut body = String::new();
        for (i, layer) in ["MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss"].iter().enumerate() {
            body += &format!(
                "    m{i} = nn.{layer}()\n    mean{i} = m{i}(p, t)\n    s{i} = nn.{layer}(reduction='sum')(p, t)\n    none{i} = nn.{layer}(reduction='none')(p, t)\n    row{i} = nn.{layer}(reduction='none')(p, row)\n    col{i} = nn.{layer}(reduction='none')(p, col)\n"
            );
        }
        let analysis = run(params, &body);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for i in 0..4 {
            assert_eq!(find_shape(&analysis, &format!("mean{i}")), Some(&shape(&[])));
            assert_eq!(find_shape(&analysis, &format!("s{i}")), Some(&shape(&[])));
            for name in ["none", "row", "col"] {
                assert_eq!(
                    find_shape(&analysis, &format!("{name}{i}")),
                    Some(&shape(&["n", "c"])),
                    "{name}{i}"
                );
            }
        }
    }

    #[test]
    fn regression_loss_broadcasts_a_column_against_a_row_as_torch_does() {
        let params = "p: Float[Array, \"4 1\"], t: Float[Array, \"4\"]";
        let analysis = run(params, "    a = nn.MSELoss(reduction='none')(p, t)\n    b = nn.MSELoss()(p, t)\n");
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["4", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&[])));
    }

    #[test]
    fn huber_loss_reads_its_reduction_from_the_first_position() {
        let params = "p: Float[Array, \"n c\"], t: Float[Array, \"n c\"]";
        let body = "    a = nn.HuberLoss('none')(p, t)\n    b = nn.HuberLoss(delta=2.0, reduction='none')(p, t)\n    c = nn.HuberLoss(2.0)(p, t)\n    d = nn.SmoothL1Loss(None, None, 'none', beta=0.5)(p, t)\n";
        let analysis = run(params, body);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "c"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "c"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["n", "c"])));
        assert_eq!(find_shape(&analysis, "c"), None);
    }

    #[test]
    fn regression_loss_with_provably_different_dims_is_an_error() {
        for (target, error) in [("5 3", true), ("3 4", true), ("n 3", false), ("2 4 3", false), ("3", false), ("4", true)] {
            let params = format!("p: Float[Array, \"4 3\"], t: Float[Array, \"{target}\"]");
            let analysis = run(&params, "    loss = nn.MSELoss(reduction='none')(p, t)\n");
            assert_eq!(analysis.errors.len(), usize::from(error), "{target}: {:?}", analysis.errors);
        }
        let analysis = run(
            "p: Float[Array, \"n c\"], t: Float[Array, \"m c\"]",
            "    mean = nn.MSELoss()(p, t)\n    none = nn.MSELoss(reduction='none')(p, t)\n",
        );
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "mean"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "none"), None);
    }

    #[test]
    fn binary_cross_entropy_needs_the_same_shape_without_broadcasting() {
        let ok = run(
            "p: Float[Array, \"n 1\"], t: Float[Array, \"n 1\"]",
            "    a = nn.BCELoss()(p, t)\n    b = nn.BCEWithLogitsLoss(reduction='none')(p, t)\n    c = nn.BCEWithLogitsLoss(pos_weight=w, reduction='sum')(p, t)\n    d = nn.BCELoss(weight=w, reduction='none')(p, t)\n",
        );
        assert!(ok.errors.is_empty(), "{:?}", ok.errors);
        assert_eq!(find_shape(&ok, "a"), Some(&shape(&[])));
        assert_eq!(find_shape(&ok, "b"), Some(&shape(&["n", "1"])));
        assert_eq!(find_shape(&ok, "c"), Some(&shape(&[])));
        assert_eq!(find_shape(&ok, "d"), Some(&shape(&["n", "1"])));
        for (p, t) in [("n 1", "n"), ("4 3", "5 3"), ("1 3", "4 3"), ("n", "n 1")] {
            for layer in ["BCELoss", "BCEWithLogitsLoss"] {
                let params = format!("p: Float[Array, \"{p}\"], t: Float[Array, \"{t}\"]");
                let analysis = run(&params, &format!("    loss = nn.{layer}()(p, t)\n"));
                assert_eq!(analysis.errors.len(), 1, "{layer} {p} / {t}: {:?}", analysis.errors);
                assert_eq!(find_shape(&analysis, "loss"), None);
            }
        }
        let symbolic = run(
            "p: Float[Array, \"n c\"], t: Float[Array, \"m c\"]",
            "    loss = nn.BCELoss()(p, t)\n",
        );
        assert!(symbolic.errors.is_empty(), "{:?}", symbolic.errors);
        assert_eq!(find_shape(&symbolic, "loss"), Some(&shape(&[])));
    }

    #[test]
    fn a_loss_with_a_value_dependent_reduction_keeps_its_checks_but_not_its_shape() {
        let body = "    a = nn.MSELoss(reduction=mode)(p, t)\n    b = nn.MSELoss(size_average=False)(p, t)\n    c = nn.CrossEntropyLoss(reduce=False)(p, t)\n    d = nn.BCELoss(reduction=self_mode)(p, t)\n";
        let analysis = run("p: Float[Array, \"n c\"], t: Float[Array, \"n c\"]", body);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["a", "b", "c", "d"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
        let bad = run(
            "p: Float[Array, \"4 3\"], t: Float[Array, \"5 3\"]",
            "    a = nn.MSELoss(reduction=mode)(p, t)\n",
        );
        assert_eq!(bad.errors.len(), 1, "{:?}", bad.errors);
    }

    #[test]
    fn an_unknown_target_leaves_the_loss_unknown_without_an_error() {
        let analysis = run(
            "p: Float[Array, \"n c\"]",
            "    a = nn.MSELoss()(p, mystery)\n    b = nn.CrossEntropyLoss()(p, mystery())\n    c = nn.BCELoss()(p)\n",
        );
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), None);
        assert_eq!(find_shape(&analysis, "b"), None);
        assert_eq!(find_shape(&analysis, "c"), None);
    }

    #[test]
    fn keyword_arguments_and_aliased_imports_reach_the_loss_rule() {
        let params = "p: Float[Array, \"n c\"], t: Float[Array, \"n c\"], y: Float[Array, \"n\"]";
        let body = "    a = Mse(reduction='none')(p, t)\n    b = nn.MSELoss(reduction='none')(p, target=t)\n    c = nn.CrossEntropyLoss(reduction='none')(p, target=y)\n    d = nn.MSELoss(reduction='none')(input=p, target=t)\n";
        let analysis = run(params, body);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "c"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "c"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n"])));
        assert_eq!(find_shape(&analysis, "d"), None);
    }

    #[test]
    fn a_loss_held_in_a_field_is_applied_in_the_method_and_in_an_expression() {
        let code = format!(
            "{HEADER}class Net(nn.Module):\n    def __init__(self, reduction='mean'):\n        super().__init__()\n        self.head = nn.Linear(8, 3)\n        self.crit = nn.CrossEntropyLoss(reduction='none')\n        self.mse = nn.MSELoss()\n\n    def forward(self, x: Float[Array, \"n 8\"], y: Float[Array, \"n\"], t: Float[Array, \"n 3\"]):\n        logits = self.head(x)\n        per = self.crit(logits, y)\n        total = self.mse(logits, t)\n        both = self.crit(logits, y).sum() + self.mse(logits, t)\n        return per, total, both\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "per"), Some(&shape(&["n"])));
        assert_eq!(find_shape(&analysis, "total"), Some(&shape(&[])));
        assert_eq!(find_shape(&analysis, "both"), Some(&shape(&[])));
    }

    #[test]
    fn an_application_names_an_inline_input_by_its_source() {
        let params = "logits: Float[Array, \"n c\"], y: Float[Array, \"n\"]";
        let body = "    nll = nn.NLLLoss()\n    loss = nll(torch.log_softmax(logits, dim=1), torch.abs(y).long())\n";
        let analysis = run(params, body);
        let application = &analysis.applications[0];
        assert_eq!(application.input, "torch.log_softmax(logits, dim=1)");
        assert_eq!(application.extra_inputs, ["torch.abs(y).long()"]);
    }

    #[test]
    fn a_loss_fed_by_a_layer_output_checks_the_layer_output() {
        let code = format!(
            "{HEADER}def f(x: Float[Array, \"n 8\"], y: Float[Array, \"n\"]):\n    head = nn.Linear(8, 3)\n    crit = nn.CrossEntropyLoss(reduction='none')\n    logits = head(x)\n    per = crit(logits, y)\n    bad = crit(logits, x)\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "per"), Some(&shape(&["n"])));
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "bad"), None);
    }
}

#[cfg(test)]
mod module_container_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEADER: &str = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport equinox as eqx\nfrom jaxtyping import Float, Array\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, d_out)\n\n    def forward(self, x):\n        return self.fc(x)\n\nclass Ann(nn.Module):\n    def __init__(self, d_out):\n        super().__init__()\n        self.fc = nn.Linear(8, d_out)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        return self.fc(x)\n\nclass Pair(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.up = nn.Linear(d, 2 * d)\n        self.down = nn.Linear(2 * d, d)\n\n    def forward(self, x):\n        return self.down(self.up(x))\n\n";

    /// A module `Net` whose `__init__` runs `init` and whose `forward` takes
    /// `x: Float[Array, "n 8"]` and runs `body`.
    fn net(init: &str, body: &str) -> LayerShapeAnalysis {
        let indent = |text: &str| {
            text.lines()
                .map(|line| format!("        {line}\n"))
                .collect::<String>()
        };
        analyze(&format!(
            "{HEADER}class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n{}\n    def forward(self, x: Float[Array, \"n 8\"], i: int):\n{}",
            indent(init),
            indent(body)
        ))
    }

    fn known(analysis: &LayerShapeAnalysis, name: &str, dims: &[&str]) {
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(analysis, name), Some(&shape(dims)), "{name}");
    }

    fn unknown(analysis: &LayerShapeAnalysis, name: &str) {
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(analysis, name), None, "{name}");
    }

    const LOOP: &str = "for layer in self.layers:\n    x = layer(x)\ny = x";

    #[test]
    fn a_loop_chains_the_modules_of_a_list_in_order() {
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])", LOOP);
        known(&a, "y", &["n", "4"]);
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Pair(16), Block(16, 2)])", LOOP);
        known(&a, "y", &["n", "2"]);
        let a = net("self.layers = [Block(8, 16), Block(16, 4)]", LOOP);
        known(&a, "y", &["n", "4"]);
        let a = net("self.layers = (Block(8, 16), Block(16, 4))", LOOP);
        known(&a, "y", &["n", "4"]);
    }

    #[test]
    fn a_loop_chains_framework_layers_and_modules_together() {
        let a = net(
            "self.layers = nn.ModuleList([nn.Linear(8, 16), nn.ReLU(), Block(16, 4)])",
            LOOP,
        );
        known(&a, "y", &["n", "4"]);
        let a = net("self.layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(16, 3)])", LOOP);
        known(&a, "y", &["n", "3"]);
    }

    #[test]
    fn a_loop_with_enumerate_chains_the_same_way() {
        let body = "for k, layer in enumerate(self.layers):\n    x = layer(x)\ny = x";
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])", body);
        known(&a, "y", &["n", "4"]);
    }

    #[test]
    fn a_chain_with_a_mismatch_reports_the_failing_element() {
        let a = net("self.layers = nn.ModuleList([Ann(16), Ann(4)])", LOOP);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert!(a.errors[0].message.contains("expected 8, got 16"), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
    }

    #[test]
    fn comments_in_the_loop_do_not_change_the_chain() {
        let body = "for layer in self.layers:  # each block\n    # apply it\n    x = layer(x)  # in order\ny = x";
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])", body);
        known(&a, "y", &["n", "4"]);
    }

    #[test]
    fn a_loop_over_an_empty_list_leaves_the_input_alone() {
        let a = net("self.layers = nn.ModuleList([])", LOOP);
        known(&a, "y", &["n", "8"]);
    }

    #[test]
    fn a_comprehension_repeats_one_module() {
        let a = net("self.layers = nn.ModuleList([Block(8, 8) for _ in range(4)])", LOOP);
        known(&a, "y", &["n", "8"]);
        let a = net("self.layers = nn.ModuleList(Block(8, 8) for _ in range(n_layers))", LOOP);
        known(&a, "y", &["n", "8"]);
        let a = net("self.layers = nn.ModuleList([Pair(8) for _ in range(n_layers)])", LOOP);
        known(&a, "y", &["n", "8"]);
    }

    #[test]
    fn a_repeated_module_that_changes_the_shape_needs_a_known_count() {
        // Applied once: [n, 8] -> [n, 16]. A second application is a mismatch.
        let a = net("self.layers = nn.ModuleList([Block(8, 16) for _ in range(n_layers)])", LOOP);
        unknown(&a, "y");
        let a = net("self.layers = nn.ModuleList([Block(8, 16) for _ in range(1)])", LOOP);
        known(&a, "y", &["n", "16"]);
        let a = net("self.layers = nn.ModuleList([Ann(16) for _ in range(3)])", LOOP);
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
        let a = net("self.layers = nn.ModuleList([Block(8, 16) for _ in range(0)])", LOOP);
        known(&a, "y", &["n", "8"]);
    }

    #[test]
    fn elements_that_depend_on_the_loop_variable_are_not_one_module() {
        let a = net("self.layers = nn.ModuleList([Block(8, 8 + k) for k in range(3)])", LOOP);
        unknown(&a, "y");
        let a = net("self.layers = nn.ModuleList([Block(8, 8) for _ in range(3) if flag])", LOOP);
        known(&a, "y", &["n", "8"]);
    }

    #[test]
    fn sequential_of_modules_applies_in_order() {
        let a = net(
            "self.net = nn.Sequential(Block(8, 16), nn.ReLU(), Block(16, 4))",
            "y = self.net(x)",
        );
        known(&a, "y", &["n", "4"]);
        let a = net(
            "self.net = nn.Sequential(*[Block(8, 8) for _ in range(3)])",
            "y = self.net(x)",
        );
        known(&a, "y", &["n", "8"]);
        let a = net("self.net = nn.Sequential(Ann(16), Ann(4))", "y = self.net(x)");
        assert_eq!(a.errors.len(), 1, "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), None);
        // A list is not callable.
        let a = net("self.net = nn.ModuleList([Block(8, 8)])", "y = self.net(x)");
        unknown(&a, "y");
    }

    #[test]
    fn indexing_picks_one_element() {
        let init = "self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])";
        let a = net(init, "a = self.layers[0](x)\nb = self.layers[1](a)\nc = self.layers[-1](a)\nd = self.layers[i](x)\ne = self.layers[5](x)");
        known(&a, "a", &["n", "16"]);
        known(&a, "b", &["n", "4"]);
        known(&a, "c", &["n", "4"]);
        unknown(&a, "d");
        unknown(&a, "e");
        let wrong = net(
            "self.layers = nn.ModuleList([Ann(16), Ann(4)])",
            "a = self.layers[0](x)\nb = self.layers[0](a)",
        );
        assert_eq!(wrong.errors.len(), 1, "{:?}", wrong.errors);
    }

    #[test]
    fn an_unknown_index_is_known_when_every_element_agrees() {
        let a = net(
            "self.layers = nn.ModuleList([Block(8, 4), Block(8, 4)])\nself.other = nn.ModuleList([Block(8, 4), Block(8, 5)])",
            "a = self.layers[i](x)\nb = self.other[i](x)\nc = self.layers[1:](x)",
        );
        known(&a, "a", &["n", "4"]);
        unknown(&a, "b");
        unknown(&a, "c");
    }

    #[test]
    fn a_comprehension_indexed_by_an_unknown_index_is_the_module() {
        let a = net(
            "self.layers = nn.ModuleList([Block(8, 4) for _ in range(n_layers)])",
            "a = self.layers[i](x)\nb = self.layers[0](x)",
        );
        known(&a, "a", &["n", "4"]);
        known(&a, "b", &["n", "4"]);
    }

    #[test]
    fn a_loop_body_with_more_than_the_chain_statement_sees_one_element() {
        let body = "for layer in self.layers:\n    h = layer(x)\n    x = F.relu(h)\ny = x\nz = h";
        let a = net("self.layers = nn.ModuleList([Block(8, 8) for _ in range(n_layers)])", body);
        known(&a, "y", &["n", "8"]);
        known(&a, "z", &["n", "8"]);
        // Different elements: the call stands for a different module per
        // iteration, so no shape holds for every iteration.
        let a = net("self.layers = nn.ModuleList([Block(8, 8), Block(8, 4)])", body);
        unknown(&a, "z");
        unknown(&a, "y");
    }

    #[test]
    fn a_loop_that_changes_a_shape_across_iterations_is_unknown_after_it() {
        let body = "for layer in self.layers:\n    h = layer(x)\n    x = F.relu(h)\ny = x";
        let a = net("self.layers = nn.ModuleList([Block(8, 16) for _ in range(n_layers)])", body);
        unknown(&a, "y");
    }

    #[test]
    fn a_loop_with_an_early_exit_or_a_rebound_variable_is_not_a_chain() {
        let a = net(
            "self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])",
            "for layer in self.layers:\n    x = layer(x)\n    break\ny = x",
        );
        unknown(&a, "y");
        let a = net(
            "self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])",
            "for layer in self.layers:\n    layer = Block(8, 8)\n    x = layer(x)\ny = x",
        );
        unknown(&a, "y");
        let a = net(
            "self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])",
            "for layer in self.layers:\n    x = layer(x)\nelse:\n    x = x\ny = x",
        );
        unknown(&a, "y");
    }

    #[test]
    fn a_list_changed_after_the_constructor_is_unknown() {
        for change in [
            "self.layers.append(Block(4, 4))",
            "self.layers.extend([Block(4, 4)])",
            "self.layers.insert(0, Block(4, 4))",
            "self.layers[0] = Block(8, 8)",
            "del self.layers[0]",
            "self.layers += [Block(4, 4)]",
            "self.layers = nn.ModuleList([Block(8, 8)])",
        ] {
            let init = format!("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])\n{change}");
            let a = net(&init, LOOP);
            unknown(&a, "y");
        }
        let code = format!(
            "{HEADER}class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n\nmodel = Net()\nmodel.layers.append(Block(4, 4))\n"
        );
        let a = analyze(&code);
        unknown(&a, "y");
    }

    #[test]
    fn a_list_changed_through_another_name_is_unknown() {
        for change in [
            "ls = self.layers\nls.append(Block(8, 5))",
            "ls = self.layers\nls[0] = Block(8, 5)",
            "ls = self.layers\ndel ls[0]",
            "getattr(self, \"layers\").append(Block(8, 5))",
            "getattr(self, 'layers')[0] = Block(8, 5)",
            "setattr(self, \"layers\", nn.ModuleList([Block(8, 5)]))",
            "register(self.layers)",
            "self.alias = self.layers",
            "self.net = nn.Sequential(*self.layers)",
        ] {
            let init = format!("self.layers = nn.ModuleList([Block(8, 8)])\n{change}");
            unknown(&net(&init, LOOP), "y");
        }
        let method = |body: &str| {
            format!(
                "{HEADER}class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 8)])\n\n    def grow(self):\n{body}\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n"
            )
        };
        unknown(&analyze(&method("        ls = self.layers\n        ls.append(Block(8, 5))\n")), "y");
        unknown(&analyze(&method("        return self.layers\n")), "y");
        let outside = format!(
            "{HEADER}class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 8)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n\nmodel = Net()\nm = model.layers\nm.append(Block(8, 5))\n"
        );
        unknown(&analyze(&outside), "y");
    }

    #[test]
    fn a_list_kept_by_one_class_reaches_only_the_classes_of_its_instances() {
        let grab = "    def grab(self):\n        ls = self.layers\n        ls.append(None)\n\n";
        let net = "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n";
        let other = format!(
            "class Other(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 2)])\n\n{grab}"
        );
        known(&analyze(&format!("{HEADER}{other}{net}")), "y", &["n", "4"]);
        // An instance of the subclass runs the forward of its base.
        let child = format!("\n\nclass Child(Net):\n{grab}");
        unknown(&analyze(&format!("{HEADER}{net}{child}")), "y");
        // A base that the subclass does not inherit from is not reached.
        let sibling = format!("\n\nclass Sibling(Net):\n    pass\n\n\nclass Child(Sibling):\n{grab}");
        unknown(&analyze(&format!("{HEADER}{net}{sibling}")), "y");
        let apart = format!("{net}\n\nclass Left(Net):\n    pass\n\n\nclass Right(nn.Module):\n{grab}");
        known(&analyze(&format!("{HEADER}{apart}")), "y", &["n", "4"]);
    }

    #[test]
    fn reading_a_list_without_keeping_it_leaves_it_known() {
        let read = "self.layers = nn.ModuleList([Block(8, 8)])\nk = len(self.layers)\nls = list(self.layers)\nls.append(Block(8, 5))\nparams = self.layers.parameters()\nfirst = self.layers[0]\nrest = self.layers[1:]\nif self.layers:\n    k = 1\nmore = [l for l in self.layers]\nfor i, l in enumerate(self.layers):\n    pass\nok = (self.layers) is not None\nflag = not self.layers";
        known(&net(read, LOOP), "y", &["n", "8"]);
        let indexed = "self.layers = nn.ModuleList([Block(8, 8), Block(8, 8)])\nz = self.layers[0](x)\nk = len(self.layers)";
        known(&net(indexed, LOOP), "y", &["n", "8"]);
    }

    #[test]
    fn the_list_of_another_class_does_not_leak() {
        let code = format!(
            "{HEADER}class Other(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 2)])\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n"
        );
        known(&analyze(&code), "y", &["n", "4"]);
    }

    #[test]
    fn a_list_of_equinox_layers_chains_in_a_call() {
        let code = "import equinox as eqx\nimport jax\nfrom jaxtyping import Float, Array\n\nclass Mlp(eqx.Module):\n    layers: list\n\n    def __init__(self, key):\n        k1, k2 = jax.random.split(key)\n        self.layers = [eqx.nn.Linear(8, 16, key=k1), eqx.nn.Linear(16, 4, key=k2)]\n\n    def __call__(self, x: Float[Array, \"8\"]):\n        for layer in self.layers:\n            x = layer(x)\n        y = x\n        return y\n";
        known(&analyze(code), "y", &["4"]);
    }

    #[test]
    fn an_equinox_sequential_of_user_modules_chains() {
        let code = "import equinox as eqx\nfrom jaxtyping import Float, Array\n\nclass Up(eqx.Module):\n    w: eqx.nn.Linear\n\n    def __init__(self, key):\n        self.w = eqx.nn.Linear(8, 16, key=key)\n\n    def __call__(self, x):\n        return self.w(x)\n\nclass Net(eqx.Module):\n    net: eqx.nn.Sequential\n\n    def __init__(self, key):\n        self.net = eqx.nn.Sequential([Up(key), eqx.nn.Linear(16, 3, key=key)])\n\n    def __call__(self, x: Float[Array, \"8\"]):\n        y = self.net(x, key=key)\n        return y\n";
        known(&analyze(code), "y", &["3"]);
    }

    #[test]
    fn nested_containers_use_the_sequence_of_their_own_class() {
        let code = format!(
            "{HEADER}class Stage(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.layers = nn.ModuleList([Block(d, d), Block(d, d)])\n\n    def forward(self, x):\n        for layer in self.layers:\n            x = layer(x)\n        return x\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.stages = nn.ModuleList([Stage(8), Stage(8)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        for stage in self.stages:\n            x = stage(x)\n        y = x\n        return y\n"
        );
        known(&analyze(&code), "y", &["n", "8"]);
    }

    #[test]
    fn a_loop_variable_hides_a_layer_of_the_same_name() {
        let body = "layer = nn.Linear(8, 3)\nfor layer in self.layers:\n    x = layer(x)\ny = x";
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])", body);
        known(&a, "y", &["n", "4"]);
    }

    #[test]
    fn the_inlay_hint_of_a_chain_statement_shows_the_result() {
        let a = net("self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])", LOOP);
        let hint = a
            .assignment_shapes
            .iter()
            .find(|r| r.name == "x")
            .map(|r| r.shape.clone());
        assert_eq!(hint, Some(shape(&["n", "4"])));
    }
}

#[cfg(test)]
mod imported_module_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn analyze_in(
        dir: &std::path::Path,
        code: &str,
        cache: Option<&ResolutionCache>,
    ) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        let roots = vec![dir.to_path_buf()];
        analyze_layer_shapes(
            tree.root_node(),
            code,
            &roots,
            |p: &PathBuf| fs::read_to_string(p).ok(),
            8,
            cache,
        )
        .unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const BLOCKS: &str = "import torch.nn as nn\nfrom jaxtyping import Float, Array\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, d_out)\n\n    def forward(self, x):\n        return self.fc(x)\n\nclass Strict(nn.Module):\n    def __init__(self, d_out):\n        super().__init__()\n        self.fc = nn.Linear(8, d_out)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        return self.fc(x)\n\ndef helper(d):\n    return d\n\nclass Pair(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.up = Block(d, d)\n        self.down = Block(d, d)\n\n    def forward(self, x):\n        return self.down(self.up(x))\n";

    const HEAD: &str = "import torch.nn as nn\nimport blocks\nfrom blocks import Block, Strict, Pair, helper\nfrom jaxtyping import Float, Array\n\n";

    fn write(dir: &tempfile::TempDir, name: &str, text: &str) -> PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, text).unwrap();
        path
    }

    fn net(init: &str, body: &str) -> String {
        let indent = |text: &str| text.lines().map(|l| format!("        {l}\n")).collect::<String>();
        format!(
            "{HEAD}class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n{}\n    def forward(self, x: Float[Array, \"n 8\"], d):\n{}",
            indent(init),
            indent(body)
        )
    }

    #[test]
    fn a_free_dim_passes_through_a_module_of_another_file() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let code = net(
            "pass",
            "y = jnp.zeros((1, 4))\nf = Block(8, d)(x)\ng = f + y\nh = Block(8, 4)(x)\ni = h + y\nj = Block(d, 2)(f)\n",
        )
        .replace("import blocks\n", "import blocks\nimport jax.numpy as jnp\n");
        let analysis = analyze_in(dir.path(), &code, None);
        assert_eq!(analysis.errors, Vec::new());
        assert_eq!(find_shape(&analysis, "f"), Some(&shape(&["n", "d"])));
        assert_eq!(find_shape(&analysis, "g"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "i"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "j"), Some(&shape(&["n", "2"])));
    }

    #[test]
    fn a_module_of_another_file_is_applied_like_one_of_this_file() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let code = net(
            "self.blk = Block(8, 16)\nself.via = blocks.Block(8, 3)",
            "a = self.blk(x)\nb = Block(8, 5)(x)\nc = self.via(x)\nd2 = blocks.Block(8, 7)(x)\ne = Pair(8)(x)\nf = Block(8, d)(x)",
        );
        let analysis = analyze_in(dir.path(), &code, None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for (name, dims) in [
            ("a", ["n", "16"]),
            ("b", ["n", "5"]),
            ("c", ["n", "3"]),
            ("d2", ["n", "7"]),
            ("e", ["n", "8"]),
            ("f", ["n", "d"]),
        ] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&dims)), "{name}");
        }
    }

    #[test]
    fn constructor_values_come_through_keywords_aliases_and_literals() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir,
            "blocks.py",
            &format!("{BLOCKS}\nclass Mix(nn.Module):\n    def __init__(self, scale=1):\n        super().__init__()\n\n    def forward(self, a, b):\n        return a + b\n"),
        );
        let code = "import torch.nn as nn\nfrom blocks import Block, Mix\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self, dim):\n        super().__init__()\n        self.dim = dim\n        self.a = Block(self.dim, 16)\n        self.b = Block(d_in=dim, d_out=3)\n        self.c = Mix(scale=0.5)\n\n    def forward(self, x: Float[Array, \"n dim\"], y: Float[Array, \"n dim\"], z: Float[Array, \"n 9\"]):\n        a = self.a(x)\n        b = self.b(x)\n        c = self.c(x, y)\n        d = self.c(b=y, a=x)\n        e = self.c(x, z)\n";
        let analysis = analyze_in(dir.path(), code, None);
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("incompatible"), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "16"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "3"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "dim"])));
        assert_eq!(find_shape(&analysis, "d"), Some(&shape(&["n", "dim"])));
        assert_eq!(find_shape(&analysis, "e"), None);
    }

    #[test]
    fn a_constant_of_the_class_file_gives_its_value_to_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir,
            "fixed.py",
            "import torch.nn as nn\n\nOUT = 4\n\nclass Fixed(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, OUT)\n\n    def forward(self, x):\n        return self.fc(x)\n\nclass Literal(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n\n    def forward(self, x):\n        return self.fc(x)\n",
        );
        let code = "import torch.nn as nn\nfrom fixed import Fixed, Literal\nfrom jaxtyping import Float, Array\n\n\ndef g(x: Float[Array, \"n 8\"], y: Float[Array, \"n 4\"]):\n    f1 = Fixed()(x)\n    z = f1 + y\n    f2 = Literal()(x)\n    w = f2 + y\n";
        let analysis = analyze_in(dir.path(), code, None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "f1"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "f2"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["n", "4"])));
    }

    #[test]
    fn a_parameter_named_like_an_imported_module_is_not_the_module() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let head = "import blocks\nfrom blocks import Block\nfrom jaxtyping import Float, Array\n\n";
        let analyze = |body: &str| analyze_in(dir.path(), &format!("{head}{body}"), None);
        let param = analyze("def f(x: Float[Array, \"n 8\"], blocks):\n    y = blocks.Block(8, 4)(x)\n");
        assert_eq!(find_shape(&param, "y"), None);
        let local = analyze("def f(x: Float[Array, \"n 8\"]):\n    blocks = other()\n    y = blocks.Block(8, 4)(x)\n");
        assert_eq!(find_shape(&local, "y"), None);
        let class = analyze("def f(x: Float[Array, \"n 8\"], Block):\n    y = Block(8, 4)(x)\n");
        assert_eq!(find_shape(&class, "y"), None);
        let outer = analyze("def f(x: Float[Array, \"n 8\"], blocks):\n    def g():\n        return blocks.Block(8, 4)(x)\n    y = g()\n");
        assert_eq!(find_shape(&outer, "y"), None);
        let plain = analyze("def f(x: Float[Array, \"n 8\"], other):\n    y = blocks.Block(8, 4)(x)\n    z = Block(8, 3)(x)\n");
        assert_eq!(find_shape(&plain, "y"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&plain, "z"), Some(&shape(&["n", "3"])));
    }

    #[test]
    fn a_module_of_another_file_checks_its_annotations_at_the_call() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let code = net(
            "self.blk = Strict(16)",
            "ok = self.blk(x)\nbad = self.blk(ok)",
        );
        let analysis = analyze_in(dir.path(), &code, None);
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        assert_eq!(
            analysis.errors[0].message,
            "call to self.blk: argument 'x' dim 1 expected 8, got 16"
        );
        assert_eq!(find_shape(&analysis, "ok"), Some(&shape(&["n", "16"])));
        assert_eq!(find_shape(&analysis, "bad"), None);
    }

    #[test]
    fn a_constant_named_like_a_word_of_the_message_keeps_the_error() {
        for constant in ["width", "dim"] {
            let dir = tempfile::tempdir().unwrap();
            write(
                &dir,
                "scaled.py",
                &format!("import torch.nn as nn\n\n{constant} = 2\n\nclass Scaled(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 16)\n        self.k = {constant}\n\n    def forward(self, x):\n        return self.fc(x)\n"),
            );
            let code = format!("from scaled import Scaled\n{}", net("self.blk = Scaled()", "a = self.blk(x)"));
            let analysis = analyze_in(dir.path(), &code, None);
            assert_eq!(analysis.errors.len(), 1, "{constant}: {:?}", analysis.errors);
        }
    }

    #[test]
    fn a_problem_inside_the_other_file_is_not_an_error_of_the_call() {
        let dir = tempfile::tempdir().unwrap();
        let broken = format!(
            "{BLOCKS}\ndef broken(a: Float[Array, \"n 3\"], b: Float[Array, \"n 4\"]):\n    return a + b\n"
        );
        write(&dir, "blocks.py", &broken);
        let analysis = analyze_in(dir.path(), &net("self.blk = Block(8, 16)", "a = self.blk(x)"), None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "16"])));
    }

    #[test]
    fn the_probe_keeps_what_the_class_names_and_leaves_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let unrelated: String = (0..300)
            .map(|i| format!("\n\nclass Unrelated{i}(nn.Module):\n    def forward(self, x: Float[Array, \"n 3\"], y: Float[Array, \"n 4\"]):\n        return x + y\n"))
            .collect();
        write(
            &dir,
            "inherit.py",
            &format!("import torch.nn as nn\nfrom jaxtyping import Float, Array\n\nclass Base(nn.Module):\n    def __init__(self, d_in):\n        super().__init__()\n        self.fc = nn.Linear(d_in, 5)\n\n    def forward(self, x):\n        return self.fc(x)\n\nclass Child(Base):\n    pass\n{unrelated}"),
        );
        let code = "import torch.nn as nn\nfrom inherit import Child\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c = Child(8)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.c(x)\n";
        let analysis = analyze_in(dir.path(), code, None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "5"])));
    }

    #[test]
    fn modules_of_another_file_take_part_in_sequences() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let code = net(
            "self.layers = nn.ModuleList([Block(8, 16), blocks.Block(16, 4)])\nself.seq = nn.Sequential(Block(8, 8), Pair(8))",
            "for layer in self.layers:\n    x = layer(x)\ny = x\nz = self.seq(y)",
        );
        let analysis = analyze_in(dir.path(), &code, None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "4"])));
    }

    #[test]
    fn what_is_not_a_user_class_stays_unknown() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", BLOCKS);
        let code = net(
            "self.f = helper(8)\nself.g = Missing(8)\nself.h = Block(cfg.dim, 8)\nself.k = Block(*sizes)\nself.m = Block(8, d * 2 if flag else 3)",
            "a = self.f(x)\nb = self.g(x)\nc = self.h(x)\ne = self.k(x)\nf = self.m(x)",
        );
        let analysis = analyze_in(dir.path(), &code, None);
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        for name in ["a", "b", "c", "e", "f"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
    }

    #[test]
    fn a_library_class_is_not_analyzed() {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("site-packages");
        fs::create_dir_all(&library).unwrap();
        fs::write(library.join("lib.py"), BLOCKS).unwrap();
        let code = "import torch.nn as nn\nfrom lib import Block\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Block(8, 16)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        a = self.blk(x)\n";
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        let roots = vec![library.clone()];
        let analysis = analyze_layer_shapes(
            tree.root_node(),
            code,
            &roots,
            |p: &PathBuf| fs::read_to_string(p).ok(),
            8,
            None,
        )
        .unwrap();
        assert_eq!(find_shape(&analysis, "a"), None);
    }

    const PROJ: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 4\"]:\n    w = x\n    v = w\n    return v\n";
    const OTHER_PROJ: &str = "def proj(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]: ...\n";

    fn imported_function_shape(files: &[(&str, &str)], body: &str) -> Option<Vec<String>> {
        let dir = tempfile::tempdir().unwrap();
        for (name, text) in files {
            write(&dir, name, text);
        }
        let code = format!(
            "import helper\nfrom helper import proj\nfrom jaxtyping import Float, Array\n\ndef g(x: Float[Array, \"n 3\"]):\n    {body}\n"
        );
        find_shape(&analyze_in(dir.path(), &code, None), "y").cloned()
    }

    #[test]
    fn an_imported_function_is_the_definition_at_module_level() {
        let method = OTHER_PROJ.replace("def proj(x", "def proj(self, x");
        let nested = format!("class C:\n    {method}");
        let inner = format!("def outer():\n    {OTHER_PROJ}");
        let in_itself = PROJ.replace("    w = x\n", &format!("    {OTHER_PROJ}    w = x\n"));
        for tail in [nested.as_str(), inner.as_str()] {
            for helper in [format!("{PROJ}{tail}"), format!("{tail}{PROJ}")] {
                let found = imported_function_shape(&[("helper.py", &helper)], "y = proj(x)");
                assert_eq!(found, Some(shape(&["n", "4"])), "{helper}");
            }
        }
        let found = imported_function_shape(&[("helper.py", &in_itself)], "y = proj(x)");
        assert_eq!(found, Some(shape(&["n", "4"])));
        let reexport = format!("{PROJ}{nested}");
        let found = imported_function_shape(
            &[("helper.py", "from impl import proj\n"), ("impl.py", &reexport)],
            "y = proj(x)",
        );
        assert_eq!(found, Some(shape(&["n", "4"])));
        let found = imported_function_shape(&[("helper.py", &format!("{PROJ}{nested}"))], "y = helper.proj(x)");
        assert_eq!(found, Some(shape(&["n", "4"])));
    }

    #[test]
    fn an_imported_name_bound_only_in_another_scope_is_not_found() {
        let method = OTHER_PROJ.replace("def proj(x", "def proj(self, x");
        for helper in [format!("class C:\n    {method}"), format!("def outer():\n    {OTHER_PROJ}")] {
            let found = imported_function_shape(&[("helper.py", &helper)], "y = proj(x)");
            assert_eq!(found, None, "{helper}");
        }
    }

    #[test]
    fn a_name_bound_again_in_its_module_stays_unknown() {
        let rebinds = [
            "def proj(x): ...\n".to_string(),
            OTHER_PROJ.to_string(),
            format!("@dec\n{OTHER_PROJ}"),
            format!("if c:\n    {OTHER_PROJ}"),
            "class proj: ...\n".to_string(),
            "proj = wrap(proj)\n".to_string(),
            "proj: Any = wrap(proj)\n".to_string(),
            "proj += 1\n".to_string(),
            "a, proj = pair\n".to_string(),
            "(a, [proj, b]) = pair\n".to_string(),
            "*proj, a = pair\n".to_string(),
            "if c:\n    proj = other\n".to_string(),
            "try:\n    pass\nexcept E:\n    proj = other\n".to_string(),
            "for proj in items:\n    pass\n".to_string(),
            "with lock as proj:\n    pass\n".to_string(),
            "try:\n    pass\nexcept E as proj:\n    pass\n".to_string(),
            "print(proj := other)\n".to_string(),
            "z = [(proj := v) for v in vs]\n".to_string(),
            "del proj\n".to_string(),
            "import other as proj\n".to_string(),
            "from other import thing as proj\n".to_string(),
            "from other import proj\n".to_string(),
            "from other import *\n".to_string(),
            "if c:\n    from other import *\n".to_string(),
            "def set_it():\n    global proj\n    proj = other\n".to_string(),
            "class C:\n    def set_it(self):\n        global proj\n        proj = other\n".to_string(),
            "match v:\n    case [proj]:\n        pass\n".to_string(),
            "match v:\n    case {\"k\": proj}:\n        pass\n".to_string(),
            "match v:\n    case [*proj]:\n        pass\n".to_string(),
            "match v:\n    case 1 as proj:\n        pass\n".to_string(),
            "type proj = int\n".to_string(),
        ];
        for tail in &rebinds {
            let helper = format!("{PROJ}{tail}");
            let found = imported_function_shape(&[("helper.py", &helper)], "y = proj(x)");
            assert_eq!(found, None, "{helper}");
            let found = imported_function_shape(
                &[("helper.py", &format!("from impl import proj\n{tail}")), ("impl.py", PROJ)],
                "y = proj(x)",
            );
            assert_eq!(found, None, "re-export: {tail}");
            let found = imported_function_shape(
                &[("helper.py", "from impl import proj\n"), ("impl.py", &helper)],
                "y = proj(x)",
            );
            assert_eq!(found, None, "behind a re-export: {helper}");
        }
    }

    #[test]
    fn a_name_used_but_not_bound_keeps_its_module_definition() {
        let others = [
            "proj.attr = 1\n",
            "proj.attr += 1\n",
            "table[proj] = 1\n",
            "print(proj)\n",
            "z = [proj for proj in vs]\n",
            "z = {proj: 1 for proj in vs}\n",
            "z = lambda proj: proj\n",
            "def other(proj):\n    proj = 1\n    return proj\n",
            "def other():\n    proj = 1\n",
            "class C:\n    proj = 1\n",
            "class C:\n    def proj(self): ...\n",
            "import proj.sub as other\n",
            "from proj import other\n",
            "from other import proj as thing\n",
            "with lock as (a, b):\n    pass\n",
            "def other():\n    global q\n    q = 1\n",
        ];
        for tail in others {
            let helper = format!("{PROJ}{tail}");
            let found = imported_function_shape(&[("helper.py", &helper)], "y = proj(x)");
            assert_eq!(found, Some(shape(&["n", "4"])), "{helper}");
        }
        let helper = format!("from other import *\n{PROJ}");
        let found = imported_function_shape(&[("helper.py", &helper)], "y = proj(x)");
        assert_eq!(found, Some(shape(&["n", "4"])));
        let found = imported_function_shape(&[("helper.py", "from proj import proj\n"), ("proj.py", PROJ)], "y = proj(x)");
        assert_eq!(found, Some(shape(&["n", "4"])));
    }

    #[test]
    fn an_attribute_of_an_imported_function_is_not_the_function() {
        let found = imported_function_shape(&[("helper.py", PROJ)], "y = helper.proj.inner(x)");
        assert_eq!(found, None);
    }

    fn block_shape(tail: &str) -> Option<Vec<String>> {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "blocks.py", &format!("{BLOCKS}{tail}"));
        let analysis = analyze_in(dir.path(), &net("self.blk = Block(8, 16)", "y = self.blk(x)"), None);
        find_shape(&analysis, "y").cloned()
    }

    #[test]
    fn an_imported_class_bound_again_in_its_module_stays_unknown() {
        for tail in [
            "Block = wrap(Block)\n",
            "if c:\n    Block = other\n",
            "class Block(nn.Module):\n    def forward(self, x):\n        return x\n",
            "@dec\nclass Block(nn.Module):\n    def forward(self, x):\n        return x\n",
            "if c:\n    class Block(nn.Module):\n        def forward(self, x):\n            return x\n",
            "def Block(a, b):\n    return a\n",
            "from other import Block\n",
            "from other import *\n",
            "def swap():\n    global Block\n    Block = other\n",
        ] {
            assert_eq!(block_shape(tail), None, "{tail}");
        }
    }

    #[test]
    fn an_imported_class_is_the_definition_at_module_level() {
        for tail in [
            "class Holder:\n    class Block(nn.Module):\n        def forward(self, x):\n            return x\n",
            "class Holder:\n    def Block(self, a, b):\n        return a\n",
            "def outer():\n    class Block(nn.Module):\n        def forward(self, x):\n            return x\n",
            "def outer():\n    def Block(a, b):\n        return a\n",
        ] {
            assert_eq!(block_shape(tail), Some(shape(&["n", "16"])), "{tail}");
        }
    }

    #[test]
    fn a_file_over_the_probe_limit_gives_an_unknown_shape() {
        let dir = tempfile::tempdir().unwrap();
        let filler = "v = 1\n".repeat(90_000);
        write(&dir, "blocks.py", &format!("{BLOCKS}{filler}"));
        let analysis = analyze_in(dir.path(), &net("self.blk = Block(8, 16)", "a = self.blk(x)"), None);
        assert_eq!(find_shape(&analysis, "a"), None);
    }

    #[test]
    fn imports_in_a_cycle_end() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir,
            "cyc_a.py",
            "import torch.nn as nn\nfrom cyc_b import B\n\nclass A(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.b = B()\n\n    def forward(self, x):\n        return self.b(x)\n",
        );
        write(
            &dir,
            "cyc_b.py",
            "import torch.nn as nn\nfrom cyc_a import A\n\nclass B(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = A()\n\n    def forward(self, x):\n        return self.a(x)\n",
        );
        let code = "import torch.nn as nn\nfrom cyc_a import A\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = A()\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.a(x)\n";
        let analysis = analyze_in(dir.path(), code, None);
        assert_eq!(find_shape(&analysis, "y"), None);
    }

    #[test]
    fn nested_imports_stop_at_the_probe_limit() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir,
            "lvl_c.py",
            "import torch.nn as nn\n\nclass C(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n\n    def forward(self, x):\n        return self.fc(x)\n",
        );
        for (name, inner, class, import) in [
            ("lvl_b.py", "C", "B", "from lvl_c import C"),
            ("lvl_a.py", "B", "A", "from lvl_b import B"),
        ] {
            write(
                &dir,
                name,
                &format!(
                    "import torch.nn as nn\n{import}\n\nclass {class}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.inner = {inner}()\n\n    def forward(self, x):\n        return self.inner(x)\n"
                ),
            );
        }
        let head = "import torch.nn as nn\nfrom lvl_a import A\nfrom lvl_b import B\nfrom lvl_c import C\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = A()\n        self.b = B()\n        self.c = C()\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n";
        let analysis = analyze_in(
            dir.path(),
            &format!("{head}        a = self.a(x)\n        b = self.b(x)\n        c = self.c(x)\n"),
            None,
        );
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "4"])));
        // Three files deep is past the limit: unknown, not a wrong shape.
        assert_eq!(find_shape(&analysis, "a"), None);
    }

    #[test]
    fn a_probe_result_follows_the_file_it_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "blocks.py", BLOCKS);
        let cache = new_resolution_cache();
        let code = net("self.blk = Block(8, 16)", "a = self.blk(x)");
        let first = analyze_in(dir.path(), &code, Some(&cache));
        assert_eq!(find_shape(&first, "a"), Some(&shape(&["n", "16"])));
        let again = analyze_in(dir.path(), &code, Some(&cache));
        assert_eq!(find_shape(&again, "a"), Some(&shape(&["n", "16"])));

        // The class now outputs `d_out + 1` features. Keep the old time and
        // size, so that only the editor event shows the change.
        let before = fs::metadata(&path).unwrap().modified().unwrap();
        fs::write(&path, BLOCKS.replace("nn.Linear(d_in, d_out)", "nn.Linear(d_in, 9)")).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(before)
            .unwrap();
        cache.revalidate(std::slice::from_ref(&path));
        let changed = analyze_in(dir.path(), &code, Some(&cache));
        assert_eq!(find_shape(&changed, "a"), Some(&shape(&["n", "9"])));

        fs::remove_file(&path).unwrap();
        cache.revalidate(std::slice::from_ref(&path));
        let gone = analyze_in(dir.path(), &code, Some(&cache));
        assert_eq!(find_shape(&gone, "a"), None);
    }

    #[test]
    fn a_change_in_a_file_the_class_imports_reaches_the_call() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir,
            "outer.py",
            "import torch.nn as nn\nfrom inner import Inner\n\nclass Outer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.inner = Inner()\n\n    def forward(self, x):\n        return self.inner(x)\n",
        );
        let inner = write(
            &dir,
            "inner.py",
            "import torch.nn as nn\n\nclass Inner(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n\n    def forward(self, x):\n        return self.fc(x)\n",
        );
        let code = "import torch.nn as nn\nfrom outer import Outer\nfrom jaxtyping import Float, Array\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.o = Outer()\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.o(x)\n";
        let cache = new_resolution_cache();
        assert_eq!(
            find_shape(&analyze_in(dir.path(), code, Some(&cache)), "y"),
            Some(&shape(&["n", "4"]))
        );
        fs::write(&inner, fs::read_to_string(&inner).unwrap().replace("(8, 4)", "(8, 5)")).unwrap();
        cache.revalidate(std::slice::from_ref(&inner));
        assert_eq!(
            find_shape(&analyze_in(dir.path(), code, Some(&cache)), "y"),
            Some(&shape(&["n", "5"]))
        );
    }
}

#[cfg(test)]
mod rebinding_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn hint(analysis: &LayerShapeAnalysis, line: u32, name: &str) -> Option<Vec<String>> {
        analysis
            .assignment_shapes
            .iter()
            .find(|r| r.line == line && r.name == name)
            .map(|r| r.shape.clone())
    }

    const HEAD: &str = "import jax.numpy as jnp\ndef f(x: Float[Array, \"3 4\"], c, d):\n";

    fn run(body: &str) -> LayerShapeAnalysis {
        analyze(&format!("{HEAD}{body}"))
    }

    #[track_caller]
    fn known(body: &str, name: &str, dims: &[&str]) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}\n{body}", a.errors);
        assert_eq!(find_shape(&a, name), Some(&shape(dims)), "{body}");
    }

    #[track_caller]
    fn unknown(body: &str, name: &str) {
        let a = run(body);
        assert!(a.errors.is_empty(), "{:?}\n{body}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{body}");
    }

    #[track_caller]
    fn unknown_in(code: &str, name: &str) {
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{code}");
    }

    #[track_caller]
    fn known_in(code: &str, name: &str, dims: &[&str]) {
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), Some(&shape(dims)), "{code}");
    }

    #[test]
    fn a_name_a_case_pattern_captures_is_rebound() {
        for pattern in [
            "[w]",
            "_ as w",
            "(1 | 2) as w",
            "Point(x=w)",
            "Point(w)",
            "{\"k\": w}",
            "{\"k\": 1, **w}",
            "[1, *w]",
            "w",
            "[w] | (w,)",
        ] {
            let body = format!("    w = x\n    match c:\n        case {pattern}:\n            pass\n    z = w\n");
            unknown(&body, "z");
        }
    }

    #[test]
    fn a_name_a_case_does_not_capture_keeps_its_shape() {
        for pattern in ["Point()", "Color.RED", "[v]", "Point(x=v)", "1 | 2", "{\"w\": v}"] {
            let body = format!("    w = x\n    match c:\n        case {pattern}:\n            pass\n    z = w\n");
            known(&body, "z", &["3", "4"]);
        }
    }

    #[test]
    fn a_captured_name_is_unknown_inside_the_case_and_its_guard() {
        let a = run("    w = x\n    match c:\n        case [w] if w:\n            z = w\n");
        assert_eq!(hint(&a, 5, "z"), None);
        // A failed pattern may have bound the name before it failed.
        unknown("    w = x\n    match c:\n        case [w, 1]:\n            pass\n        case _:\n            z = w\n", "z");
    }

    #[test]
    fn a_read_before_the_loop_rewrites_the_name_is_unknown() {
        let a = run("    y = x\n    while c:\n        t = y\n        y = y.T\n");
        assert_eq!(hint(&a, 4, "t"), None);
        assert_eq!(hint(&a, 5, "y"), None);
        assert_eq!(find_shape(&a, "t"), None);
        let a = run("    y = x\n    for i in range(3):\n        t = y\n        y = y.T\n");
        assert_eq!(hint(&a, 4, "t"), None);
        let a = run("    y = x\n    for i in range(3):\n        if c:\n            y = y.T\n            continue\n        t = y\n        y = y.T\n");
        assert_eq!(hint(&a, 7, "t"), None);
    }

    #[test]
    fn a_read_of_a_name_that_keeps_its_shape_stays_known_in_the_loop() {
        let a = run("    y = x\n    for i in range(3):\n        t = y\n        y = y + 0\n");
        assert_eq!(hint(&a, 4, "t"), Some(shape(&["3", "4"])));
        assert_eq!(hint(&a, 5, "y"), Some(shape(&["3", "4"])));
        assert_eq!(find_shape(&a, "t"), Some(&shape(&["3", "4"])));
    }

    #[test]
    fn only_the_hints_that_read_the_old_shape_are_taken_back() {
        // The body writes `y` before it reads it: every iteration agrees.
        let a = run("    y = x\n    for i in range(3):\n        y = jnp.zeros(5)\n        t = y\n");
        assert_eq!(hint(&a, 4, "y"), Some(shape(&["5"])));
        assert_eq!(hint(&a, 5, "t"), Some(shape(&["5"])));
        // A write in a branch does not write it on every path.
        let a = run("    y = x\n    for i in range(3):\n        if c:\n            y = jnp.zeros(5)\n        t = y\n");
        assert_eq!(hint(&a, 5, "y"), Some(shape(&["5"])));
        assert_eq!(hint(&a, 6, "t"), None);
        // What is written from a name that is not right is not right.
        let a = run("    y = x\n    for i in range(3):\n        t = y\n        u = t\n        z = jnp.zeros(2)\n        y = y.T\n");
        assert_eq!(hint(&a, 4, "t"), None);
        assert_eq!(hint(&a, 5, "u"), None);
        assert_eq!(hint(&a, 6, "z"), Some(shape(&["2"])));
        // The same in an inner loop.
        let a = run("    y = x\n    for i in range(3):\n        for j in range(2):\n            t = y\n            y = y.T\n");
        assert_eq!(hint(&a, 5, "t"), None);
    }

    #[test]
    fn a_name_that_changes_in_the_body_is_unknown_in_the_else_clause() {
        let a = run("    y = x\n    for i in range(3):\n        y = y.T\n    else:\n        t = y\n");
        assert_eq!(hint(&a, 6, "t"), None);
    }

    #[test]
    fn a_name_set_in_the_first_iteration_only_is_not_trusted() {
        let for_loop = "    for i in d:\n        if i == 0:\n            y = x\n        y2 = y\n        y = x.T\n";
        unknown(for_loop, "y2");
        let a = run(for_loop);
        assert_eq!(hint(&a, 5, "y2"), None);
    }

    #[test]
    fn a_name_the_loop_feeds_back_is_not_new_at_the_branch() {
        let body = "    for i in d:\n        if i == 0:\n            h = jnp.zeros(3)\n        r = h + 0\n        h = jnp.concatenate([h, h])\n";
        unknown(body, "r");
        let counted = "    i = 0\n    while i < 3:\n        if i == 0:\n            h = jnp.zeros(3)\n        r = h + 0\n        h = jnp.concatenate([h, h])\n        i += 1\n";
        unknown(counted, "r");
        let shape_changes = "    for i in d:\n        if i == 0:\n            h = jnp.zeros(3)\n        for j in d:\n            h = h[None]\n        r = h\n";
        unknown(shape_changes, "r");
    }

    #[test]
    fn a_name_only_the_branch_writes_stays_new_in_a_loop() {
        known("    for i in d:\n        if i == 0:\n            h = jnp.zeros(3)\n        r = h + 0\n", "r", &["3"]);
        known("    for i in d:\n        if i == 0:\n            h = jnp.zeros(3)\n        r = h + 0\n        q = r\n", "q", &["3"]);
    }

    #[test]
    fn a_list_target_of_with_hides_an_earlier_shape() {
        for target in ["[x]", "(x, y)", "[y, *x]", "x", "(y, (x, z))"] {
            unknown(&format!("    with c as {target}:\n        pass\n    r = x + 0\n"), "r");
        }
        known("    with c as x.y:\n        pass\n    r = x + 0\n", "r", &["3", "4"]);
        known("    with c as [y.x]:\n        pass\n    r = x + 0\n", "r", &["3", "4"]);
    }

    #[test]
    fn an_imported_alias_of_a_swallowing_manager_is_found() {
        let body = "    with sp(ValueError):\n        x = jnp.zeros(4)\n    r = x + 0\n";
        for import in ["from contextlib import suppress as sp", "from pytest import raises as sp"] {
            unknown_in(&format!("{import}\n{HEAD}{body}"), "r");
        }
        known_in(&format!("from m import other as sp\n{HEAD}{body}"), "r", &["4"]);
        let plain = "    with suppress(E):\n        x = jnp.zeros(4)\n    r = x + 0\n";
        unknown_in(&format!("from contextlib import suppress\n{HEAD}{plain}"), "r");
    }

    #[test]
    fn a_walrus_rebinds_its_name_wherever_it_stands() {
        for body in [
            "    if (x := jnp.zeros(4)).sum() > 0:\n        pass\n    r = x + 0\n",
            "    y = 1 if c else (x := jnp.zeros(4))\n    r = x + 0\n",
            "    ok = c and (x := jnp.zeros(4))\n    r = x + 0\n",
            "    z = [(x := jnp.zeros(4)) for _ in d]\n    r = x + 0\n",
            "    [(x := jnp.zeros(4)) for _ in d]\n    r = x + 0\n",
            "    while (x := d):\n        pass\n    r = x + 0\n",
            "    g(a=(x := 1))\n    r = x + 0\n",
            "    q = (x := jnp.zeros(4)) + x\n    r = x + 0\n",
        ] {
            unknown(body, "r");
        }
        let a = run("    if (w := x.T).shape[0] > 3:\n        pass\n    v = w\n");
        assert_eq!(find_shape(&a, "v"), None);
    }

    #[test]
    fn a_walrus_of_another_name_leaves_the_shape() {
        known("    if (q := d).shape[0] > 3:\n        pass\n    r = x + 0\n", "r", &["3", "4"]);
        known("    z = [(q := v) for v in d]\n    r = x + 0\n", "r", &["3", "4"]);
    }

    #[test]
    fn a_definition_an_import_or_a_delete_rebinds_the_name() {
        unknown("    k = x\n    import os as k\n    s = k\n", "s");
        unknown("    k = x\n    from os import path as k\n    s = k\n", "s");
        unknown("    k = x\n    import k.sub\n    s = k\n", "s");
        unknown("    q = x\n    class q:\n        pass\n    r = q\n", "r");
        unknown("    d2 = x\n    def d2():\n        pass\n    t = d2\n", "t");
        unknown("    d2 = x\n    @dec\n    def d2():\n        pass\n    t = d2\n", "t");
        unknown("    d2 = x\n    async def d2():\n        pass\n    t = d2\n", "t");
        known("    d2 = x\n    def other():\n        pass\n    t = d2\n", "t", &["3", "4"]);
        known("    d2 = x\n    import os\n    t = d2\n", "t", &["3", "4"]);
    }

    #[test]
    fn a_definition_an_import_or_a_delete_in_a_branch_makes_the_name_unknown() {
        unknown("    if c:\n        def x():\n            pass\n    r = x + 0\n", "r");
        unknown("    if c:\n        class x:\n            pass\n    r = x + 0\n", "r");
        unknown("    if c:\n        import os as x\n    r = x + 0\n", "r");
        unknown("    if c:\n        from os import path as x\n    r = x + 0\n", "r");
        unknown("    if c:\n        del x\n    r = x + 0\n", "r");
        known("    if c:\n        def other():\n            pass\n    r = x + 0\n", "r", &["3", "4"]);
        known("    if c:\n        del x.attr\n    r = x + 0\n", "r", &["3", "4"]);
    }

    #[test]
    fn a_method_does_not_rebind_the_name_of_the_module() {
        let code = "import jax.numpy as jnp\nstep = jnp.zeros(3)\nclass C:\n    def step(self):\n        pass\nz = step\n";
        known_in(code, "z", &["3"]);
    }

    #[test]
    fn a_module_level_definition_counts_as_an_earlier_binding() {
        let code = "import jax.numpy as jnp\ndef n2():\n    pass\nif cond:\n    n2 = jnp.zeros((3,))\nz2 = n2\nclass n3:\n    pass\nif cond:\n    n3 = jnp.zeros((3,))\nz3 = n3\n";
        let a = analyze(code);
        assert_eq!(find_shape(&a, "z2"), None);
        assert_eq!(find_shape(&a, "z3"), None);
        known_in("import jax.numpy as jnp\nif cond:\n    n4 = jnp.zeros((3,))\nz4 = n4\n", "z4", &["3"]);
        known_in("import jax.numpy as jnp\nif cond:\n    n5 = jnp.zeros((3,))\nz5 = n5\ndef other():\n    pass\n", "z5", &["3"]);
    }

    #[test]
    fn a_name_defined_twice_in_one_scope_resolves_to_neither() {
        let nested = "from jaxtyping import Float, Array\ndef k(x: Float[Array, \"3 4\"], c):\n    def inner(v):\n        return v.T\n    if c:\n        def inner(v):\n            return v\n    r = inner(x)\n";
        unknown_in(nested, "r");
        let top = "from jaxtyping import Float, Array\ndef inner(v):\n    return v.T\nif c:\n    def inner(v):\n        return v\ndef k(x: Float[Array, \"3 4\"]):\n    r = inner(x)\n";
        unknown_in(top, "r");
        let once = "from jaxtyping import Float, Array\ndef k(x: Float[Array, \"3 4\"], c):\n    def inner(v):\n        return v.T\n    r = inner(x)\n";
        known_in(once, "r", &["4", "3"]);
        let apart = "from jaxtyping import Float, Array\ndef k(x: Float[Array, \"3 4\"]):\n    def inner(v):\n        return v.T\n    r = inner(x)\ndef j(x: Float[Array, \"3 4\"]):\n    def inner(v):\n        return v\n    s = inner(x)\n";
        let a = analyze(apart);
        assert_eq!(find_shape(&a, "r"), Some(&shape(&["4", "3"])));
        assert_eq!(find_shape(&a, "s"), Some(&shape(&["3", "4"])));
    }

    const METHOD_HEAD: &str = "import jax\nimport jax.numpy as jnp\nfrom jaxtyping import Float, Array\n";
    const TRANSPOSE: &str = "        return v.T\n";
    const RUN: &str = "    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\n";

    fn methods(body: &str) -> String {
        format!("{METHOD_HEAD}{body}")
    }

    #[test]
    fn a_self_call_stays_in_the_class_of_the_call() {
        let code = format!("class A:\n    def helper(self, v):\n{TRANSPOSE}{RUN}class B:\n    def helper(self, v):\n        return v\n");
        known_in(&methods(&code), "r", &["4", "3"]);
        let annotated = "class A:\n    def helper(self, v: Float[Array, \"a b\"]) -> Float[Array, \"b a\"]:\n        return v.T\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\nclass B:\n    def helper(self, v: Float[Array, \"a\"]) -> Float[Array, \"a\"]:\n        return v\n";
        known_in(&methods(annotated), "r", &["4", "3"]);
    }

    #[test]
    fn a_self_call_never_reaches_an_unrelated_class_or_a_function() {
        unknown_in(&methods(&format!("class A:\n{RUN}class B:\n    def helper(self, v):\n        return v\n")), "r");
        unknown_in(&methods(&format!("def helper(v):\n    return v\nclass A:\n{RUN}")), "r");
        unknown_in(&methods(&format!("class A:\n    def other(self, x):\n        def helper(v):\n            return v\n{RUN}")), "r");
    }

    #[test]
    fn a_self_call_follows_the_method_resolution_order() {
        let base = format!("class Base:\n    def helper(self, v):\n{TRANSPOSE}class Other:\n    def helper(self, v):\n        return v\n");
        known_in(&methods(&format!("{base}class Sub(Base):\n{RUN}")), "r", &["4", "3"]);
        known_in(&methods(&format!("{base}class Sub(Other, Base):\n{RUN}")), "r", &["3", "4"]);
        known_in(&methods(&format!("{base}class Sub(Base):\n    def helper(self, v):\n        return v\n{RUN}")), "r", &["3", "4"]);
    }

    #[test]
    fn a_self_call_stops_at_a_base_the_file_does_not_define() {
        let base = format!("from lib import Opaque\nclass Base:\n    def helper(self, v):\n{TRANSPOSE}");
        unknown_in(&methods(&format!("{base}class Sub(Opaque):\n{RUN}")), "r");
        unknown_in(&methods(&format!("{base}class Sub(Opaque, Base):\n{RUN}")), "r");
        known_in(&methods(&format!("{base}class Sub(Base, Opaque):\n{RUN}")), "r", &["4", "3"]);
    }

    #[test]
    fn a_self_call_with_an_override_below_it_is_unknown() {
        let base = format!("class Base:\n    def helper(self, v):\n{TRANSPOSE}{RUN}");
        unknown_in(&methods(&format!("{base}class Sub(Base):\n    def helper(self, v):\n        return v\n")), "r");
        unknown_in(&methods(&format!("{base}class Sub(Base):\n    helper = None\n")), "r");
        unknown_in(&methods(&format!("{base}class Mixin:\n    def helper(self, v):\n        return v\nclass Both(Mixin, Base):\n    pass\n")), "r");
        unknown_in(&methods(&format!("{base}from lib import Opaque\nclass Sub(Opaque, Base):\n    def helper(self, v):\n        return v\n")), "r");
        known_in(&methods(&format!("{base}class Sub(Base):\n    pass\n")), "r", &["4", "3"]);
        known_in(&methods(&format!("{base}class Mixin:\n    def helper(self, v):\n        return v\nclass Both(Base, Mixin):\n    pass\n")), "r", &["4", "3"]);
    }

    #[test]
    fn a_class_attribute_or_an_instance_field_hides_a_method() {
        let base = format!("class Base:\n    def helper(self, v):\n{TRANSPOSE}");
        unknown_in(&methods(&format!("{base}class Sub(Base):\n    helper = None\n{RUN}")), "r");
        unknown_in(&methods(&format!("class A:\n    def helper(self, v):\n{TRANSPOSE}    helper = make(helper)\n{RUN}")), "r");
        unknown_in(&methods(&format!("class A:\n    def __init__(self):\n        self.helper = make()\n    def helper(self, v):\n{TRANSPOSE}{RUN}")), "r");
        let field_below = format!("{base}    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\nclass Sub(Base):\n    def __init__(self):\n        self.helper = make()\n");
        unknown_in(&methods(&field_below), "r");
    }

    #[test]
    fn a_class_call_follows_the_bases() {
        let base = "class Base:\n    @classmethod\n    def helper(cls, v):\n        return v.T\n";
        let sub = "class Sub(Base):\n    @classmethod\n    def build(cls, x: Float[Array, \"3 4\"]):\n        r = cls.helper(x)\nclass Other:\n    @classmethod\n    def helper(cls, v):\n        return v\n";
        known_in(&methods(&format!("{base}{sub}")), "r", &["4", "3"]);
    }

    #[test]
    fn a_super_call_starts_after_the_class_of_the_call() {
        let base = format!("class Base:\n    def helper(self, v):\n{TRANSPOSE}");
        let sub = "    def helper(self, v):\n        r = super().helper(v)\n        return r\n    def run(self, x: Float[Array, \"3 4\"]):\n        q = super().helper(x)\n";
        known_in(&methods(&format!("{base}class Sub(Base):\n{sub}")), "q", &["4", "3"]);
        let two_args = "class Sub(Base):\n    def run(self, x: Float[Array, \"3 4\"]):\n        q = super(Sub, self).helper(x)\n";
        known_in(&methods(&format!("{base}{two_args}")), "q", &["4", "3"]);
        let other_class = "class Sub(Base):\n    def run(self, x: Float[Array, \"3 4\"]):\n        q = super(Base, self).helper(x)\n";
        unknown_in(&methods(&format!("{base}{other_class}")), "q");
        unknown_in(&methods(&format!("{base}class Sub:\n{sub}")), "q");
        unknown_in(&methods(&format!("from lib import Opaque\n{base}class Sub(Opaque, Base):\n{sub}")), "q");
        unknown_in(&methods(&format!("from lib import Opaque\n{base}class Sub(Opaque):\n{sub}")), "q");
        let unrelated = "class Other:\n    def helper(self, v):\n        return v\n";
        unknown_in(&methods(&format!("{unrelated}class Sub:\n{sub}")), "q");
    }

    #[test]
    fn a_super_call_follows_the_order_of_a_subclass() {
        let top = format!("class Top:\n    def helper(self, v):\n{TRANSPOSE}class Left(Top):\n    def run(self, x: Float[Array, \"3 4\"]):\n        q = super().helper(x)\nclass Right(Top):\n");
        let right = "    def helper(self, v):\n        return v\n";
        unknown_in(&methods(&format!("{top}{right}class Both(Left, Right):\n    pass\n")), "q");
        known_in(&methods(&format!("{top}{right}")), "q", &["4", "3"]);
    }

    #[test]
    fn a_static_or_class_method_binds_the_right_parameters() {
        let stat = "class A:\n    @staticmethod\n    def helper(v):\n        return v.T\n";
        known_in(&methods(&format!("{stat}{RUN}")), "r", &["4", "3"]);
        let by_cls = "    @classmethod\n    def build(cls, x: Float[Array, \"3 4\"]):\n        r = cls.helper(x)\n";
        known_in(&methods(&format!("{stat}{by_cls}")), "r", &["4", "3"]);
        let clsm = "class A:\n    @classmethod\n    def helper(cls, v):\n        return v.T\n";
        known_in(&methods(&format!("{clsm}{RUN}")), "r", &["4", "3"]);
        known_in(&methods(&format!("{clsm}{by_cls}")), "r", &["4", "3"]);
        let inst = format!("class A:\n    def helper(self, v):\n{TRANSPOSE}");
        unknown_in(&methods(&format!("{inst}{by_cls}")), "r");
    }

    #[test]
    fn a_receiver_in_a_static_method_or_a_function_is_not_the_class() {
        let inst = format!("class A:\n    def helper(self, v):\n{TRANSPOSE}");
        let in_static = "    @staticmethod\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\n";
        unknown_in(&methods(&format!("{inst}{in_static}")), "r");
        let free = "def run(self, x: Float[Array, \"3 4\"]):\n    r = self.helper(x)\n";
        unknown_in(&methods(&format!("{inst}{free}")), "r");
        let closure = "    def run(self, x: Float[Array, \"3 4\"]):\n        def inner(y):\n            return self.helper(y)\n        r = inner(x)\n";
        known_in(&methods(&format!("{inst}{closure}")), "r", &["4", "3"]);
    }

    #[test]
    fn a_method_that_is_not_a_plain_call_stays_unknown() {
        let prop = "class A:\n    @property\n    def helper(self):\n        return jnp.zeros((1, 2))\n";
        unknown_in(&methods(&format!("{prop}{RUN}")), "r");
        let other_receiver = "class A:\n    def helper(this, v):\n        return v.T\n";
        unknown_in(&methods(&format!("{other_receiver}{RUN}")), "r");
        let twice = "class A:\n    def helper(self, v):\n        return v.T\n    if c:\n        def helper(self, v):\n            return v\n";
        unknown_in(&methods(&format!("{twice}{RUN}")), "r");
        let recursive = "class Base:\n    def helper(self, v):\n        return v\nclass A(Base):\n    def helper(self, v):\n        r = self.helper(v)\n        return v.T\n    def run(self, x: Float[Array, \"3 4\"]):\n        q = self.helper(x)\n";
        known_in(&methods(recursive), "q", &["4", "3"]);
    }

    #[test]
    fn a_user_method_wins_over_the_array_method_of_the_same_name() {
        let code = "class A:\n    def sum(self, v):\n        return v.T\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.sum(x)\n";
        known_in(&methods(code), "r", &["4", "3"]);
    }

    #[test]
    fn a_bare_call_never_reaches_a_method() {
        let code = "class A:\n    def helper(self, v):\n        return v\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = helper(x)\n";
        unknown_in(&methods(code), "r");
        let code = format!("def helper(v):\n{TRANSPOSE}class A:\n    def helper(self, v):\n        return v\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = helper(x)\n");
        known_in(&methods(&code).replace("return v.T", "    return v.T"), "r", &["4", "3"]);
        let in_body = "def helper(v):\n    return v.T\nclass A:\n    def helper(v):\n        return v\n    x: Float[Array, \"3 4\"] = make()\n    r = helper(x)\n";
        unknown_in(&methods(in_body), "r");
    }

    #[test]
    fn a_vmap_of_a_name_finds_the_function_the_call_sees() {
        let closures = "def a(xs: Float[Array, \"b 3 4\"]):\n    def f(v: Float[Array, \"p q\"]) -> Float[Array, \"q p\"]:\n        return v.T\n    r = jax.vmap(f)(xs)\ndef c(xs: Float[Array, \"b 3 4\"]):\n    def f(v: Float[Array, \"p q\"]) -> Float[Array, \"p q\"]:\n        return v\n    s = jax.vmap(f)(xs)\n";
        let a = analyze(&methods(closures));
        assert_eq!(find_shape(&a, "r"), Some(&shape(&["b", "4", "3"])));
        assert_eq!(find_shape(&a, "s"), Some(&shape(&["b", "3", "4"])));

        let named = closures.replace("r = jax.vmap(f)(xs)", "vf = jax.vmap(f)\n    r = vf(xs)").replace("s = jax.vmap(f)(xs)", "vf = jax.vmap(f)\n    s = vf(xs)");
        let a = analyze(&methods(&named));
        assert_eq!(find_shape(&a, "r"), Some(&shape(&["b", "4", "3"])));
        assert_eq!(find_shape(&a, "s"), Some(&shape(&["b", "3", "4"])));

        let map = closures.replace("jax.vmap(f)(xs)", "jax.lax.map(f, xs)");
        let a = analyze(&methods(&map));
        assert_eq!(find_shape(&a, "r"), Some(&shape(&["b", "4", "3"])));
        assert_eq!(find_shape(&a, "s"), Some(&shape(&["b", "3", "4"])));

        let hidden = "def a():\n    def f(v: Float[Array, \"p q\"]) -> Float[Array, \"q p\"]:\n        return v.T\ndef c(xs: Float[Array, \"b 3 4\"]):\n    r = jax.vmap(f)(xs)\n";
        unknown_in(&methods(hidden), "r");
    }

    #[test]
    fn a_vmap_of_a_name_never_reaches_a_method() {
        let method = "class A:\n    def f(self, v: Float[Array, \"p q\"]) -> Float[Array, \"q p\"]:\n        return v.T\n    def run(self, xs: Float[Array, \"b 3 4\"]):\n        r = jax.vmap(f)(xs)\n";
        unknown_in(&methods(method), "r");
        unknown_in(&methods(&method.replace("jax.vmap(f)(xs)", "jax.lax.map(f, xs)")), "r");
        let shadow = format!("def f(v: Float[Array, \"p q\"]) -> Float[Array, \"p q\"]:\n    return v\n{method}");
        known_in(&methods(&shadow), "r", &["b", "3", "4"]);
    }

    #[test]
    fn a_closure_hides_a_smaller_function_of_the_same_name() {
        let code = "def f(v):\n    return v\ndef a(x: Float[Array, \"3 4\"]):\n    def f(v):\n        y = v\n        z = y\n        return z.T\n    r = f(x)\n";
        known_in(&methods(code), "r", &["4", "3"]);
        let vmapped = "def f(v: Float[Array, \"p q\"]) -> Float[Array, \"p q\"]:\n    return v\ndef a(xs: Float[Array, \"b 3 4\"]):\n    def f(v: Float[Array, \"p q\"]) -> Float[Array, \"q p\"]:\n        y = v\n        z = y\n        return z.T\n    r = jax.vmap(f)(xs)\n";
        known_in(&methods(vmapped), "r", &["b", "4", "3"]);
    }

    #[test]
    fn a_scan_body_never_reaches_a_method_or_a_foreign_closure() {
        let body = "        def step(c, y):\n            return c, y.T\n";
        let code = format!("def a():\n{}def run(c0: Float[Array, \"2\"], xs: Float[Array, \"b 3 4\"]):\n    c, ys = jax.lax.scan(step, c0, xs)\n", body.replace("        ", "    "));
        unknown_in(&methods(&code), "ys");
        let method = format!("class A:\n{}    def run(self, c0: Float[Array, \"2\"], xs: Float[Array, \"b 3 4\"]):\n        c, ys = jax.lax.scan(step, c0, xs)\n", body.replace("        ", "    "));
        unknown_in(&methods(&method), "ys");
    }
}

#[cfg(test)]
mod layer_constant_dim_tests {
    use super::*;
    use tree_sitter::Parser;

    const HEAD: &str = "import torch.nn as nn\nfrom jaxtyping import Float, Array\n";
    const USE_FC: &str = "    def forward(self, x: Float[Array, \"n 8\"], t: Float[Array, \"n 4\"]):\n        y = self.fc(x)\n        z = y + t\n        return self.fc(x) + t\n";

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|d| d.to_string()).collect()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let code = format!("{HEAD}{code}");
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(&code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), &code, &[], |_: &PathBuf| None, 5, None).unwrap()
    }

    #[track_caller]
    fn holds(code: &str, name: &str, dims: &[&str]) {
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), Some(&shape(dims)), "{code}");
    }

    #[track_caller]
    fn unknown(code: &str, name: &str) {
        let a = analyze(code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{code}");
    }

    fn module(before: &str, init: &str, ctor: &str, after: &str) -> String {
        format!("{before}class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n{init}        self.fc = {ctor}\n{USE_FC}{after}")
    }

    #[test]
    fn a_module_constant_is_the_dim_of_a_layer() {
        holds(&module("OUT = 4\n", "", "nn.Linear(8, OUT)", ""), "y", &["n", "4"]);
        holds(&module("OUT = 4\nOUT = 4\n", "", "nn.Linear(8, OUT)", ""), "y", &["n", "4"]);
        holds(&module("OUT = 2\n", "", "nn.Linear(8, OUT * 2)", ""), "y", &["n", "4"]);
        holds(&module("OUT = 2\n", "", "nn.Linear(8, (OUT + 2) * 3 // 3)", ""), "y", &["n", "4"]);
        holds(&module("IN = 8\nOUT = 4\n", "", "nn.Linear(in_features=IN, out_features=OUT)", ""), "y", &["n", "4"]);
        holds(&module("OUT = 4\n", "", "nn.Sequential(nn.Linear(8, OUT), nn.ReLU())", ""), "y", &["n", "4"]);
    }

    #[test]
    fn a_module_level_layer_and_a_local_layer_read_the_constant() {
        let body = "def f(x: Float[Array, \"n 8\"], t: Float[Array, \"n 4\"]):\n    y = l(x)\n    z = y + t\n";
        holds(&format!("OUT = 4\nl = nn.Linear(8, OUT)\n{body}"), "y", &["n", "4"]);
        let local = "def f(x: Float[Array, \"n 8\"], t: Float[Array, \"n 4\"]):\n    d = 4\n    l = nn.Linear(8, d)\n    y = l(x)\n    z = y + t\n";
        holds(local, "y", &["n", "4"]);
        let inline = "OUT = 4\ndef f(x: Float[Array, \"n 8\"], t: Float[Array, \"n 4\"]):\n    y = nn.Linear(8, OUT)(x)\n    z = y + t\n";
        holds(inline, "y", &["n", "4"]);
    }

    #[test]
    fn a_constant_reaches_every_layer_kind_that_takes_dims() {
        let cases: &[(&str, &str, &[&str])] = &[
            ("C = 16\nK = 3\nP = 1\n", "nn.Conv2d(3, C, K, padding=P)", &["1", "16", "32", "32"]),
            ("C = 16\n", "nn.ConvTranspose2d(3, C, 2, stride=2)", &["1", "16", "64", "64"]),
            ("K = 2\n", "nn.MaxPool2d(K)", &["1", "3", "16", "16"]),
            ("S = 7\n", "nn.AdaptiveAvgPool2d(S)", &["1", "3", "7", "7"]),
            ("D = 1\n", "nn.Flatten(D)", &["1", "3072"]),
            ("S = 2\n", "nn.Upsample(scale_factor=S)", &["1", "3", "64", "64"]),
        ];
        for (before, ctor, want) in cases {
            let code = format!("{before}class M(nn.Module):\n    def __init__(self):\n        self.layer = {ctor}\n    def forward(self, x: Float[Array, \"1 3 32 32\"]):\n        y = self.layer(x)\n");
            holds(&code, "y", want);
        }
        let embed = "V = 100\nD = 4\nclass M(nn.Module):\n    def __init__(self):\n        self.layer = nn.Embedding(V, D)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.layer(x)\n";
        holds(embed, "y", &["n", "8", "4"]);
        let lstm = "H = 4\nclass M(nn.Module):\n    def __init__(self):\n        self.layer = nn.LSTM(8, H)\n    def forward(self, x: Float[Array, \"s n 8\"], t: Float[Array, \"s n 4\"]):\n        y, (h, c) = self.layer(x)\n        z = y + t\n";
        holds(lstm, "y", &["s", "n", "4"]);
        let list = "OUT = 4\nclass M(nn.Module):\n    def __init__(self):\n        self.layers = nn.ModuleList([nn.Linear(8, OUT)])\n    def forward(self, x: Float[Array, \"n 8\"], t: Float[Array, \"n 4\"]):\n        y = self.layers[0](x)\n        z = y + t\n";
        holds(list, "y", &["n", "4"]);
        let rearrange = "from einops.layers.torch import Rearrange\nP = 2\nclass M(nn.Module):\n    def __init__(self):\n        self.layer = Rearrange(\"b (h p) -> b h p\", p=P)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.layer(x)\n";
        holds(rearrange, "y", &["n", "4", "2"]);
    }

    #[test]
    fn a_constant_finds_a_real_mismatch() {
        let a = analyze(&module("IN = 16\n", "", "nn.Linear(IN, 4)", ""));
        assert_eq!(a.errors.len(), 2, "{:?}", a.errors);
        assert!(a.errors[0].message.contains("expected input last dim 16"), "{:?}", a.errors);
    }

    #[test]
    fn a_self_attribute_holds_a_constant() {
        for (before, init, ctor) in [
            ("OUT = 4\n", "        self.out = OUT\n", "nn.Linear(8, self.out)"),
            ("", "        self.out = 4\n", "nn.Linear(8, self.out)"),
            ("OUT = 2\n", "        self.out = OUT * 2\n", "nn.Linear(8, self.out)"),
            ("OUT = 2\n", "        self.out = OUT\n", "nn.Linear(8, self.out * 2)"),
            ("", "        out = 4\n        self.out = out\n", "nn.Linear(8, self.out)"),
        ] {
            holds(&module(before, init, ctor, ""), "y", &["n", "4"]);
        }
        let class_level = "class M(nn.Module):\n    OUT = 4\n    def __init__(self):\n        self.fc = nn.Linear(8, self.OUT)\n";
        holds(&format!("{class_level}{USE_FC}"), "y", &["n", "4"]);
        let base = "OUT = 4\nclass B(nn.Module):\n    def __init__(self):\n        self.out = OUT\nclass M(B):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, self.out)\n";
        holds(&format!("{base}{USE_FC}"), "y", &["n", "4"]);
    }

    #[test]
    fn an_attribute_does_not_read_a_global_of_its_own_name() {
        holds(&module("out = 4\n", "        self.out = 9\n", "nn.Linear(8, self.out)", "").replace("Array, \"n 4\"", "Array, \"n 9\""), "y", &["n", "9"]);
        let param = "out = 4\nclass M(nn.Module):\n    def __init__(self, out):\n        self.fc = nn.Linear(8, out)\n";
        let a = analyze(&format!("{param}    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.fc(x)\n"));
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "out"])));
    }

    #[test]
    fn an_attribute_with_two_values_is_unknown() {
        for (init, after) in [
            ("        self.out = 4\n", "    def reset(self):\n        self.out = 5\n"),
            ("        self.out = 4\n        self.out += 1\n", ""),
            ("        self.out = 4\n", "class S(M):\n    def __init__(self):\n        self.out = 5\n"),
            ("        self.out, self.b = 4, 5\n", ""),
            ("        self.out = cfg.out\n", ""),
        ] {
            unknown(&module("", init, "nn.Linear(8, self.out)", after), "y");
        }
        let class_level = "class M(nn.Module):\n    OUT = 4\n    def __init__(self):\n        self.fc = nn.Linear(8, self.OUT)\n";
        unknown(&format!("{class_level}{USE_FC}class S(M):\n    OUT = 5\n"), "y");
        unknown(&format!("{}{USE_FC}class S(M):\n    def __init__(self):\n        self.OUT = 5\n", class_level), "y");
    }

    #[test]
    fn a_value_the_file_cannot_prove_gives_no_false_error() {
        for before in [
            "OUT = 4\nOUT = 5\n",
            "OUT = compute()\n",
            "from config import OUT\n",
            "from config import *\nOUT = 4\n",
            "OUT = 4\nOUT += 1\n",
            "OUT = 4\ndef bump():\n    global OUT\n    OUT = 5\n",
            "class C:\n    OUT = 5\nOUT = 4\n",
        ] {
            unknown(&module(before, "", "nn.Linear(8, OUT)", ""), "y");
        }
        unknown(&module("", "", "nn.Linear(8, cfg.out)", ""), "y");
        let lstm = "H = load()\nclass M(nn.Module):\n    def __init__(self):\n        self.layer = nn.LSTM(8, H)\n    def forward(self, x: Float[Array, \"s n 8\"], t: Float[Array, \"s n 4\"]):\n        y, (h, c) = self.layer(x)\n        z = y + t\n";
        unknown(lstm, "y");
        unknown(&module("", "", "nn.Linear(8, int(4.0))", ""), "y");
        unknown(&module("", "        d = load()\n", "nn.Linear(8, d)", ""), "y");
    }

    #[test]
    fn a_write_below_the_reader_is_unknown() {
        unknown(&module("", "", "nn.Linear(8, OUT)", "OUT = 4\n"), "y");
        unknown(&module("OUT = 4\n", "", "nn.Linear(8, OUT)", "OUT = 4\n"), "y");
        unknown(&module("OUT = 4\n", "", "nn.Linear(8, OUT)", "OUT = 5\n"), "y");
    }

    #[test]
    fn an_unproven_input_dim_keeps_the_output() {
        holds(&module("IN = load()\n", "", "nn.Linear(IN, 4)", ""), "y", &["n", "4"]);
        let chain = "H = load()\nclass M(nn.Module):\n    def __init__(self):\n        self.fc = nn.Sequential(nn.Linear(8, H), nn.Linear(H, 4))\n";
        holds(&format!("{chain}{USE_FC}"), "y", &["n", "4"]);
    }

    #[test]
    fn a_local_or_a_parameter_hides_the_constant() {
        let param = "OUT = 4\nclass M(nn.Module):\n    def __init__(self, OUT):\n        self.fc = nn.Linear(8, OUT)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.fc(x)\n";
        assert_eq!(find_shape(&analyze(param), "y"), Some(&shape(&["n", "OUT"])));
        unknown(&module("OUT = 4\n", "        OUT = load()\n", "nn.Linear(8, OUT)", ""), "y");
    }

    #[test]
    fn a_constant_named_like_an_annotation_symbol_equals_the_symbol() {
        let code = "D = 64\nclass M(nn.Module):\n    def __init__(self):\n        self.fc = nn.Linear(8, D)\n    def forward(self, x: Float[Array, \"n 8\"], t: Float[Array, \"n D\"]):\n        y = self.fc(x)\n        z = y + t\n";
        holds(code, "y", &["n", "64"]);
        holds(code, "z", &["n", "D"]);
    }
}

#[cfg(test)]
mod import_scope_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    const HEAD: &str = "from jaxtyping import Float, Array\n";

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|d| d.to_string()).collect()
    }

    fn analyze_in(dir: Option<&std::path::Path>, code: &str) -> LayerShapeAnalysis {
        let code = format!("{HEAD}{code}");
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let roots: Vec<PathBuf> = dir.iter().map(|d| d.to_path_buf()).collect();
        analyze_layer_shapes(tree.root_node(), &code, &roots, |p: &PathBuf| fs::read_to_string(p).ok(), 8, None)
            .unwrap()
    }

    #[track_caller]
    fn holds(code: &str, name: &str, dims: &[&str]) {
        let a = analyze_in(None, code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), Some(&shape(dims)), "{code}");
    }

    #[track_caller]
    fn unknown(code: &str, name: &str) {
        let a = analyze_in(None, code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        assert_eq!(find_shape(&a, name), None, "{code}");
    }

    #[test]
    fn a_function_import_binds_only_in_that_function() {
        let reader = "def read(k: Float[Array, \"3 4\"]):\n    y = k.reshape(4, 3)\n";
        for writer in [
            "def write():\n    import os as k\n    return k.getcwd()\n",
            "def write():\n    from os import path as k\n    return k\n",
            "class C:\n    def write(self):\n        import os as k\n",
            "class C:\n    import os as k\n",
        ] {
            holds(&format!("{writer}{reader}"), "y", &["4", "3"]);
            holds(&format!("{reader}{writer}"), "y", &["4", "3"]);
        }
    }

    #[test]
    fn a_function_import_reaches_its_nested_scopes() {
        holds("def f():\n    import numpy as q\n    y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        holds("def f():\n    y = q.zeros((2, 3))\n    import numpy as q\n", "y", &["2", "3"]);
        holds("def f():\n    import numpy as q\n    def g():\n        y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        holds("def f():\n    import numpy as q\n    class C:\n        def m(self):\n            y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        holds("def f():\n    if a:\n        import numpy as q\n    y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        unknown("def f():\n    import numpy as q\ndef g():\n    y = q.zeros((2, 3))\n", "y");
    }

    #[test]
    fn a_class_import_binds_only_in_the_class_body() {
        holds("class C:\n    import numpy as q\n    y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        unknown("class C:\n    import numpy as q\n    def m(self):\n        y = q.zeros((2, 3))\n", "y");
        unknown("class C:\n    import numpy as q\n    f = lambda self: q.zeros((2, 3)) + (y := q.zeros((2, 3)))\n", "y");
        unknown("class C:\n    import numpy as q\n    class D:\n        y = q.zeros((2, 3))\n", "y");
        holds("import numpy as q\nclass C:\n    import os as q\n    def m(self):\n        y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        holds("import os as q\ndef f():\n    import numpy as q\n    class C:\n        import os as q\n        def m(self):\n            y = q.zeros((2, 3))\n", "y", &["2", "3"]);
    }

    #[test]
    fn a_nested_import_wins_over_the_module_import() {
        holds("import os as q\ndef f():\n    import numpy as q\n    y = q.zeros((2, 3))\n", "y", &["2", "3"]);
        unknown("import numpy as q\ndef f():\n    import os as q\n    y = q.zeros((2, 3))\n", "y");
        holds("import numpy as q\ndef f():\n    import os as q\ndef g():\n    y = q.zeros((2, 3))\n", "y", &["2", "3"]);
    }

    #[test]
    fn a_global_import_in_a_function_binds_for_the_file() {
        let code = "def setup():\n    global q\n    import numpy as q\ndef f():\n    y = q.zeros((2, 3))\n";
        holds(code, "y", &["2", "3"]);
        unknown("import numpy as q\ndef f():\n    global q\n    q = load()\n    y = q.zeros((2, 3))\n", "y");
    }

    #[test]
    fn a_module_rebinding_hides_the_import_in_python_order() {
        let head = "import numpy as k\n";
        let z = "k.zeros((2, 3))";
        for bind in [
            "k = load()",
            "k += 1",
            "k, j = load()",
            "for k in xs:\n    pass",
            "with a as k:\n    pass",
            "del k",
            "(k := load())",
            "def k():\n    pass",
            "class k:\n    pass",
            "if c:\n    k = load()",
            "def s():\n    global k\n    k = load()",
            "def s():\n    global k\n    def k():\n        pass",
            "class C:\n    global k\n    k = load()",
        ] {
            unknown(&format!("{head}{bind}\ny = {z}\n"), "y");
            unknown(&format!("{head}{bind}\ndef f():\n    y = {z}\n"), "y");
            unknown(&format!("{head}def f():\n    y = {z}\n{bind}\n"), "y");
            unknown(&format!("{head}def f():\n    g = lambda: (y := {z})\n{bind}\n"), "y");
            unknown(&format!("{head}class C:\n    def m(self):\n        y = {z}\n{bind}\n"), "y");
            unknown(&format!("{head}{bind}\nclass C:\n    y = {z}\n"), "y");
            holds(&format!("{head}y = {z}\n{bind}\n"), "y", &["2", "3"]);
            holds(&format!("{head}class C:\n    y = {z}\n{bind}\n"), "y", &["2", "3"]);
        }
        unknown(&format!("{head}for i in r:\n    y = {z}\n    k = load()\n"), "y");
        unknown(&format!("{head}while c:\n    y = {z}\n    k = load()\n"), "y");
        unknown(&format!("{head}k = 1\nimport numpy as k\ny = {z}\n"), "y");
        holds(&format!("{head}x = k.ones((1, 2))\nk = k.zeros((2, 3))\ny = k\n"), "x", &["1", "2"]);
        holds(&format!("{head}k = k.zeros((2, 3))\ny = k\n"), "y", &["2", "3"]);
    }

    #[test]
    fn a_name_that_keeps_the_import_stays_an_import() {
        let z = "k.zeros((2, 3))";
        for head in [
            "import numpy as k\nk.seed = 1\n",
            "import numpy as k\nk.flags[0] = 1\n",
            "import numpy as k\nwith k.errstate() as e:\n    pass\n",
            "import numpy as k\ndef s():\n    global k\n    return k\n",
            "import numpy as k\nclass C:\n    k = load()\n",
            "import numpy as k\ndef s():\n    k = load()\n",
            "try:\n    import numpy as k\nexcept ImportError:\n    k = None\n",
        ] {
            holds(&format!("{head}y = {z}\n"), "y", &["2", "3"]);
            holds(&format!("{head}def f():\n    y = {z}\n"), "y", &["2", "3"]);
        }
    }

    #[test]
    fn a_scope_inside_a_rebinding_module_keeps_its_own_import() {
        let head = "import os as k\nk = load()\n";
        holds(&format!("{head}def f():\n    import numpy as k\n    y = k.zeros((2, 3))\n"), "y", &["2", "3"]);
        holds(
            &format!("{head}def f():\n    import numpy as k\n    def g():\n        y = k.zeros((2, 3))\n"),
            "y",
            &["2", "3"],
        );
        unknown(
            &format!("{head}def f():\n    import numpy as k\n    def g():\n        global k\n        y = k.zeros((2, 3))\n"),
            "y",
        );
        unknown("np = load()\ny = np.zeros((2, 3))\n", "y");
        unknown("def f():\n    y = torch.zeros(2, 3)\ntorch = load()\n", "y");
    }

    #[test]
    fn a_rebound_name_reads_its_new_value() {
        holds("import numpy as np\nimport os as w\nw = np.zeros((2, 3))\ny = w.reshape(3, 2)\n", "y", &["3", "2"]);
    }

    #[test]
    fn a_global_rebinding_makes_a_class_name_opaque() {
        let code = |bind: &str| {
            format!("class P:\n    def helper(self, v):\n        return v.T\n{bind}\nclass B(P):\n    def go(self, x: Float[Array, \"3 4\"]):\n        y = self.helper(x)\n")
        };
        let head = "from jaxtyping import Float, Array\n";
        unknown(&format!("{head}{}", code("def s():\n    global P\n    P = other")), "y");
        unknown(&format!("{head}{}", code("def s():\n    global P\n    from e import P")), "y");
        holds(&format!("{head}{}", code("def s():\n    global P\n    return P")), "y", &["4", "3"]);
        holds(&format!("{head}{}", code("def s():\n    P = other")), "y", &["4", "3"]);
    }

    #[test]
    fn a_name_that_a_scope_binds_hides_the_import() {
        for (head, rhs) in [("import numpy as np", "np.zeros((2, 3))"), ("import torch", "torch.zeros(2, 3)")] {
            let name = rhs.split('.').next().unwrap();
            for body in [
                format!("def f({name}):\n    y = {rhs}\n"),
                format!("def f(*{name}):\n    y = {rhs}\n"),
                format!("def f(**{name}):\n    y = {rhs}\n"),
                format!("def f(a, {name}=None):\n    y = {rhs}\n"),
                format!("def f({name}: int):\n    y = {rhs}\n"),
                format!("def f({name}: int = 3):\n    y = {rhs}\n"),
                format!("def f():\n    {name} = load()\n    y = {rhs}\n"),
                format!("def f():\n    y = {rhs}\n    {name} = load()\n"),
                format!("def f():\n    for {name} in xs:\n        y = {rhs}\n"),
                format!("def f():\n    with a as {name}:\n        y = {rhs}\n"),
                format!("def f():\n    def {name}():\n        pass\n    y = {rhs}\n"),
                format!("def f():\n    class {name}:\n        pass\n    y = {rhs}\n"),
                format!("def f({name}):\n    def g():\n        y = {rhs}\n"),
                format!("def f({name}):\n    import os\n    y = {rhs}\n"),
                format!("def f():\n    import numpy as {name}\n    {name} = load()\n    y = {rhs}\n"),
                format!("class C:\n    {name} = load()\n    y = {rhs}\n"),
            ] {
                unknown(&format!("{head}\n{body}"), "y");
                unknown(&body, "y");
            }
            holds(
                &format!("{head}\ndef f({name}):\n    pass\ndef g():\n    y = {rhs}\n"),
                "y",
                &["2", "3"],
            );
            holds(
                &format!("{head}\nclass C:\n    {name} = load()\n    def m(self):\n        y = {rhs}\n"),
                "y",
                &["2", "3"],
            );
        }
    }

    #[test]
    fn a_lambda_or_a_comprehension_hides_the_import_inside_itself() {
        let head = "import numpy as np\n";
        unknown(&format!("{head}def f(xs):\n    y = [np.zeros((2, 3)) for np in xs]\n    z = sum(y)\n"), "z");
        let code = format!("{head}def f(xs):\n    g = lambda np: (y := np.zeros((2, 3)))\n");
        unknown(&code, "y");
        let code = format!("{head}def f(xs):\n    a = [np for np in xs]\n    y = np.zeros((2, 3))\n");
        holds(&code, "y", &["2", "3"]);
        let code = format!("{head}def f(xs, g=lambda np: np):\n    y = np.zeros((2, 3))\n");
        holds(&code, "y", &["2", "3"]);
    }

    #[test]
    fn a_default_value_reads_the_scope_around_the_function() {
        let code = "import numpy as np\ndef f(np=np.pi, a=[np for np in xs]):\n    return np\ng = lambda np=np.e: np\n";
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(code, None).unwrap();
        let imports = ImportScopes::new(tree.root_node(), code).unwrap();
        let hidden = |at: &str| !is_imported(imports.at(code.find(at).unwrap()), "np");
        assert!(!hidden("np.pi"));
        assert!(hidden("np for np"));
        assert!(hidden("np\ng"));
        assert!(!hidden("np.e"));
    }

    #[test]
    fn a_parameter_is_read_as_the_array_it_holds() {
        let code = "import numpy as x\ndef f(x: Float[Array, \"3 4\"]):\n    y = x.reshape(4, 3)\n    z = x.T\n    w = x.sum(0).reshape(2, 2)\n";
        holds(code, "y", &["4", "3"]);
        let code = "from torch import nn\ndef f(x: Float[Array, \"n 8\"]):\n    fc = nn.Linear(8, 4)\n    y = fc(x)\ndef g(nn, x: Float[Array, \"n 8\"]):\n    fc = nn.Linear(8, 4)\n    z = fc(x)\n";
        let a = analyze_in(None, code);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&a, "z"), None);
    }

    #[test]
    fn a_layer_from_a_function_import_keeps_its_shape() {
        let code = "def f(x: Float[Array, \"n 8\"]):\n    from torch import nn\n    fc = nn.Linear(8, 4)\n    y = fc(x)\n    z = nn.Linear(8, 5)(x)\n";
        let a = analyze_in(None, code);
        assert!(a.errors.is_empty(), "{:?}", a.errors);
        assert_eq!(find_shape(&a, "y"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&a, "z"), Some(&shape(&["n", "5"])));
        let code = "def f():\n    from torch import nn\ndef g(x: Float[Array, \"n 8\"]):\n    fc = nn.Linear(8, 4)\n    y = fc(x)\n";
        assert_eq!(find_shape(&analyze_in(None, code), "y"), None);
    }

    #[test]
    fn a_parameter_named_like_a_module_is_not_the_module() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("blocks.py"),
            "import torch.nn as nn\nclass Block(nn.Module):\n    def __init__(self, a, b):\n        super().__init__()\n        self.fc = nn.Linear(a, b)\n    def forward(self, x):\n        return self.fc(x)\n",
        )
        .unwrap();
        let root = Some(dir.path());
        let imported = "import torch.nn as nn\nimport blocks\n";
        let code = format!("{imported}def f(x: Float[Array, \"n 8\"]):\n    y = blocks.Block(8, 4)(x)\n");
        assert_eq!(find_shape(&analyze_in(root, &code), "y"), Some(&shape(&["n", "4"])));
        for body in [
            "def f(blocks, x: Float[Array, \"n 8\"]):\n    y = blocks.Block(8, 4)(x)\n",
            "def f(x: Float[Array, \"n 8\"]):\n    blocks = load()\n    y = blocks.Block(8, 4)(x)\n",
            "def f(blocks, x: Float[Array, \"n 8\"]):\n    m = blocks.Block(8, 4)\n    y = m(x)\n",
            "class M(nn.Module):\n    def __init__(self, blocks):\n        super().__init__()\n        self.m = blocks.Block(8, 4)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.m(x)\n",
        ] {
            let a = analyze_in(root, &format!("{imported}{body}"));
            assert_eq!(find_shape(&a, "y"), None, "{body}");
            assert!(a.errors.is_empty(), "{:?}\n{body}", a.errors);
        }
        let code = "import torch.nn as nn\ndef f(x: Float[Array, \"n 8\"]):\n    import blocks\n    y = blocks.Block(8, 4)(x)\n";
        assert_eq!(find_shape(&analyze_in(root, code), "y"), Some(&shape(&["n", "4"])));
    }

    #[test]
    fn a_function_import_still_names_the_backend_of_the_file() {
        let body = "def f(x: Float[Array, \"3 4\"]):\n    y = x.max(0)\n";
        let numpy = format!("def g():\n    import numpy\n{body}");
        assert_eq!(find_shape(&analyze_in(None, &numpy), "y"), Some(&shape(&["4"])));
        assert_eq!(find_shape(&analyze_in(None, &format!("import numpy\n{body}")), "y"), Some(&shape(&["4"])));
        assert_eq!(find_shape(&analyze_in(None, &format!("def g():\n    import torch\n{body}")), "y"), None);
    }
}

#[cfg(test)]
mod binding_review_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    const HEAD: &str = "import torch.nn as nn\nfrom jaxtyping import Float, Array\n";

    fn analyze_in(dir: Option<&std::path::Path>, code: &str) -> LayerShapeAnalysis {
        let code = format!("{HEAD}{code}");
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let roots: Vec<PathBuf> = dir.iter().map(|d| d.to_path_buf()).collect();
        analyze_layer_shapes(tree.root_node(), &code, &roots, |p: &PathBuf| fs::read_to_string(p).ok(), 8, None)
            .unwrap()
    }

    #[track_caller]
    fn check(dir: Option<&std::path::Path>, code: &str, name: &str, dims: Option<&[&str]>) {
        let a = analyze_in(dir, code);
        assert!(a.errors.is_empty(), "{:?}\n{code}", a.errors);
        let want: Option<Vec<String>> = dims.map(|d| d.iter().map(|d| d.to_string()).collect());
        assert_eq!(find_shape(&a, name), want.as_ref(), "{code}");
    }

    #[track_caller]
    fn holds(code: &str, name: &str, dims: &[&str]) {
        check(None, code, name, Some(dims));
    }

    #[track_caller]
    fn unknown(code: &str, name: &str) {
        check(None, code, name, None);
    }

    #[test]
    fn the_last_function_import_of_a_name_wins() {
        unknown("def f():\n    import numpy as k\n    import os as k\n    a = k.zeros((2, 3))\n", "a");
        holds("def f():\n    import os as k\n    import numpy as k\n    a = k.zeros((2, 3))\n", "a", &["2", "3"]);
        unknown("def f():\n    from numpy import zeros as z\n    from os import path as z\n    a = z((2, 3))\n", "a");
        holds("def f():\n    from os import path as z\n    from numpy import zeros as z\n    a = z((2, 3))\n", "a", &["2", "3"]);
        unknown("def f(c):\n    import numpy as k\n    if c:\n        import os as k\n    a = k.zeros((2, 3))\n", "a");
    }

    #[test]
    fn a_nested_global_reads_the_module_import() {
        let nested = "def f():\n    import numpy as k\n    def g():\n        global k\n        a = k.zeros((2, 3))\n";
        unknown(&format!("import os as k\n{nested}"), "a");
        unknown(nested, "a");
        holds("import numpy as k\ndef f():\n    import os as k\n    def g():\n        global k\n        a = k.zeros((2, 3))\n", "a", &["2", "3"]);
        let inner = "import numpy as k\ndef f():\n    import os as k\n    def g():\n        global k\n        def h():\n            a = k.zeros((2, 3))\n";
        holds(inner, "a", &["2", "3"]);
    }

    fn class_attr(body: &str) -> String {
        format!("class M(nn.Module):\n{body}    def __init__(self):\n        self.l = nn.Linear(8, self.h)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.l(x)\n")
    }

    #[test]
    fn every_write_in_a_class_body_counts() {
        holds(&class_attr("    h = 4\n"), "y", &["n", "4"]);
        holds(&class_attr("    h = 4\n    if FLAG:\n        h = 4\n"), "y", &["n", "4"]);
        for body in [
            "    h = 4\n    h += 4\n",
            "    h = 4\n    if FLAG:\n        h = 8\n",
            "    h = 4\n    h, w = 8, 9\n",
            "    h = 4\n    for h in (8, 16):\n        pass\n",
            "    h = 4\n    with a() as h:\n        pass\n",
            "    h = 4\n    del h\n",
        ] {
            unknown(&class_attr(body), "y");
        }
    }

    #[test]
    fn a_for_or_with_target_writes_the_attribute() {
        let attr = |write: &str| format!("class M(nn.Module):\n    def __init__(self):\n        self.h = 4\n{write}        self.l = nn.Linear(8, self.h)\n    def forward(self, x: Float[Array, \"n 8\"]):\n        y = self.l(x)\n");
        holds(&attr(""), "y", &["n", "4"]);
        for write in [
            "        for self.h in (8, 9):\n            pass\n",
            "        for self.h, w in pairs:\n            pass\n",
            "        with a() as self.h:\n            pass\n",
            "        with a() as (self.h, w):\n            pass\n",
        ] {
            unknown(&attr(write), "y");
        }
        let method = "class B:\n    def helper(self, x):\n        return x.T\n    def s(self):\n        WRITE\n    def go(self, x: Float[Array, \"3 4\"]):\n        y = self.helper(x)\n";
        holds(&method.replace("WRITE", "pass"), "y", &["4", "3"]);
        unknown(&method.replace("WRITE", "for self.helper in others:\n            pass"), "y");
        unknown(&method.replace("WRITE", "with a() as self.helper:\n            pass"), "y");
    }

    #[test]
    fn a_conditional_dim_stays_unknown() {
        let code = |before: &str, dim: &str| format!("{before}class M(nn.Module):\n    def __init__(self, big, d):\n        self.l = nn.Linear(8, {dim})\n    def forward(self, x: Float[Array, \"n 8\"], z: Float[Array, \"n 4\"]):\n        y = self.l(x)\n        w = y + z\n");
        unknown(&code("OUT = 4\n", "OUT if big else 8"), "y");
        unknown(&code("", "d if big else 8"), "y");
        unknown(&code("", "big or 4"), "y");
        unknown(&code("", "not big"), "y");
    }

    const P: &str = "class P(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n    def forward(self, x):\n        return self.l(x)\n";
    const INPUTS: &str = "x: Float[Array, \"n 8\"], z: Float[Array, \"n 4\"]";

    fn constructor_cases(head: &str) -> Vec<(String, Option<Vec<&'static str>>)> {
        let call = |constant: &str, body: &str| format!("{head}{constant}def f({INPUTS}):\n{body}");
        let held = |constant: &str, arg: &str| format!("{head}{constant}class H(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.p = P({arg})\n    def forward(self, {INPUTS}):\n        y = self.p(x)\n        w = y + z\n");
        let four = || Some(vec!["n", "4"]);
        let use_y = "    y = P(OUT)(x)\n    w = y + z\n";
        vec![
            (call("OUT = 4\n", use_y), four()),
            (call("OUT = 2\n", "    y = P(OUT * 2)(x)\n    w = y + z\n"), four()),
            (call("", "    d = 4\n    y = P(d)(x)\n    w = y + z\n"), four()),
            (call("OUT = 4\n", "    y = P(d=OUT)(x)\n    w = y + z\n"), four()),
            (held("OUT = 4\n", "OUT"), four()),
            (call("OUT = load()\n", use_y), None),
            (call("OUT = 4\nOUT = 8\n", use_y), None),
            (call("", "    d = load()\n    y = P(d)(x)\n    w = y + z\n"), None),
            (held("OUT = load()\n", "OUT"), None),
        ]
    }

    #[test]
    fn a_constructor_argument_gives_its_value() {
        for (code, want) in constructor_cases(P) {
            check(None, &code, "y", want.as_deref());
        }
        let param = format!("{P}def f(d, x: Float[Array, \"n 8\"]):\n    y = P(d)(x)\n");
        holds(&param, "y", &["n", "d"]);
    }

    #[test]
    fn a_constructor_argument_gives_its_value_across_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("blocks.py"), format!("{HEAD}{P}")).unwrap();
        for (code, want) in constructor_cases("from blocks import P\n") {
            check(Some(dir.path()), &code, "y", want.as_deref());
        }
        let param = "from blocks import P\ndef f(d, x: Float[Array, \"n 8\"]):\n    y = P(d)(x)\n";
        check(Some(dir.path()), param, "y", Some(&["n", "d"]));
    }

    #[test]
    fn an_unproven_argument_keeps_an_output_that_does_not_read_it() {
        let dir = tempfile::tempdir().unwrap();
        let fixed = "class Q(nn.Module):\n    def __init__(self, act):\n        super().__init__()\n        self.act = act\n        self.l = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.l(x)\n";
        fs::write(dir.path().join("blocks.py"), format!("{HEAD}{fixed}")).unwrap();
        let code = |head: &str| format!("{head}ACT = load()\ndef f({INPUTS}):\n    y = Q(ACT)(x)\n");
        check(None, &code(fixed), "y", Some(&["n", "4"]));
        check(Some(dir.path()), &code("from blocks import Q\n"), "y", Some(&["n", "4"]));
    }

    const CLASSES: &str = "class A:\n    def helper(self, x):\n        return x.T\nclass O:\n    def helper(self, x):\n        return x\n";

    fn receiver(body: &str, params: &str) -> String {
        format!("{CLASSES}class B(A):\n    def go(self, x: Float[Array, \"3 4\"]{params}):\n{body}")
    }

    #[test]
    fn a_rebound_receiver_is_not_the_receiver() {
        holds(&receiver("        self.other = O()\n        y = self.helper(x)\n", ""), "y", &["4", "3"]);
        for body in [
            "        self = O()\n        y = self.helper(x)\n",
            "        if o:\n            self = o\n        y = self.helper(x)\n",
            "        for self in os:\n            pass\n        y = self.helper(x)\n",
            "        with os as self:\n            pass\n        y = self.helper(x)\n",
            "        (self := O())\n        y = self.helper(x)\n",
            "        self = O()\n        def inner():\n            return self.helper(x)\n        y = inner()\n",
            "        def inner():\n            nonlocal self\n            self = O()\n        inner()\n        y = self.helper(x)\n",
            "        y = [self.helper(x) for self in os][0]\n",
        ] {
            unknown(&receiver(body, ""), "y");
        }
    }

    #[test]
    fn a_rebound_super_is_not_the_builtin() {
        let own = "class B(A):\n    def helper(self, x):\n        return x\n    def go(self, x: Float[Array, \"3 4\"]SUPER):\n        BODY\n";
        let code = |params: &str, body: &str| format!("{CLASSES}{}", own.replace("SUPER", params).replace("BODY", body));
        holds(&code("", "y = super().helper(x)"), "y", &["4", "3"]);
        unknown(&code(", super=None", "y = super().helper(x)"), "y");
        unknown(&code("", "super = make\n        y = super().helper(x)"), "y");
        unknown(&format!("super = make\n{}", code("", "y = super().helper(x)")), "y");
        unknown(&code("", "self = O()\n        y = super().helper(x)"), "y");
    }

    #[test]
    fn a_rebound_base_name_is_not_the_class() {
        let parent = "class P:\n    def helper(self, x):\n        return x.T\n";
        let child = "class B(P):\n    def go(self, x: Float[Array, \"3 4\"]):\n        y = self.helper(x)\n";
        holds(&format!("{parent}{child}"), "y", &["4", "3"]);
        for rebinding in [
            "P = other\n",
            "from e import P\n",
            "if c:\n    P = other\n",
            "for P in others:\n    pass\n",
            "def P():\n    pass\n",
        ] {
            unknown(&format!("{parent}{rebinding}{child}"), "y");
        }
    }
}

#[cfg(test)]
mod free_dim_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn messages(analysis: &LayerShapeAnalysis) -> Vec<&str> {
        analysis.errors.iter().map(|e| e.message.as_str()).collect()
    }

    const HEAD: &str = "import torch\nimport torch.nn as nn\nimport jax.numpy as jnp\nimport flax.linen as fnn\nimport equinox as eqx\nfrom jaxtyping import Float, Array\n\n";

    /// A module with a layer `self.l = nn.Linear(8, d)` and `BODY` in `forward`.
    fn module(body: &str) -> String {
        let body: String = body.lines().map(|l| format!("        {l}\n")).collect();
        format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d, e):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n        self.k = nn.Linear(8, e)\n        self.d = d\n\n    def forward(self, x: Float[Array, \"n 8\"], y: Float[Array, \"n 4\"], z: Float[Array, \"3 4\"], w: Float[Array, \"n m\"]):\n        h = self.l(x)\n{body}"
        )
    }

    #[test]
    fn a_parameter_dim_does_not_conflict_with_an_integer() {
        let analysis = analyze(&module("s = h + y\nt = y - h"));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["n", "d"])));
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "t"), Some(&shape(&["n", "4"])));
    }

    #[test]
    fn a_parameter_dim_does_not_conflict_with_another_symbol() {
        let analysis = analyze(&module("s = h + w\nu = self.k(x)\nv = h + u\nw2 = u + h"));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["n", "m"])));
        // Two free names: the result is one of them, and the file cannot say which.
        assert!(!has_shape(&analysis, "v"));
        assert!(!has_shape(&analysis, "w2"));
    }

    #[test]
    fn the_same_parameter_dim_stays_on_both_sides() {
        let analysis = analyze(&module("s = h + self.l(x)"));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["n", "d"])));
    }

    #[test]
    fn a_broadcast_of_one_keeps_the_parameter_dim() {
        let analysis = analyze(&module("o = jnp.ones((1, 1))\ns = h + o"));
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["n", "d"])));
    }

    #[test]
    fn two_different_integers_still_report() {
        let analysis = analyze(&module("q = jnp.zeros((5, 7))\ns = jnp.zeros((5, 1)) + q\nr = z + q"));
        assert_eq!(messages(&analysis), vec!["elementwise + incompatible shapes, got [3, 4] and [5, 7]"]);
    }

    #[test]
    fn an_integer_on_another_axis_still_reports_and_names_the_plain_symbol() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"3 8\"], y: Float[Array, \"5 4\"]):\n        h = self.l(x)\n        s = h + y\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), vec!["elementwise + incompatible shapes, got [3, d] and [5, 4]"]);
    }

    #[test]
    fn an_annotation_symbol_is_still_a_claim() {
        let code = format!(
            "{HEAD}def f(a: Float[Array, \"n d\"], b: Float[Array, \"n 4\"], c: Float[Array, \"n e\"]):\n    p = a + b\n    q = a + c\n"
        );
        assert_eq!(analyze(&code).errors.len(), 2);
    }

    #[test]
    fn an_annotation_with_the_name_of_a_parameter_makes_no_conflict() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"n 8\"], y: Float[Array, \"n 4\"], v: Float[Array, \"n d\"]):\n        h = self.l(x)\n        s = h + y\n        u = h + v\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "u"), Some(&shape(&["n", "d"])));
    }

    #[test]
    fn branches_and_repeats_see_one_name_in_a_parameter_and_an_annotation() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d, c):\n        super().__init__()\n        self.l = nn.Linear(d, d)\n        self.r = nn.Sequential(*[nn.Linear(d, d) for _ in range(c)])\n\n    def forward(self, x: Float[Array, \"n d\"], c):\n        if c:\n            y = x\n        else:\n            y = self.l(x)\n        z = x\n        for _ in range(3):\n            z = self.l(z)\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "d"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "d"])));
    }

    #[test]
    fn a_dataclass_field_does_not_conflict() {
        let code = format!(
            "{HEAD}class F(fnn.Module):\n    features: int\n\n    @fnn.compact\n    def __call__(self, x: Float[Array, \"n 8\"], y: Float[Array, \"n 4\"]):\n        h = fnn.Dense(self.features)(x)\n        s = h + y\n\nclass E(eqx.Module):\n    features: int\n    l: eqx.nn.Linear\n\n    def __init__(self, features, key):\n        self.features = features\n        self.l = eqx.nn.Linear(8, features, key=key)\n\n    def __call__(self, x: Float[Array, \"8\"], y: Float[Array, \"4\"]):\n        g = self.l(x)\n        t = g + y\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["n", "features"])));
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "t"), Some(&shape(&["4"])));
    }

    #[test]
    fn a_known_value_at_the_call_site_still_specializes() {
        let code = format!(
            "{HEAD}class P(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x):\n        return self.l(x)\n\nclass Q(nn.Module):\n    def __init__(self, c):\n        super().__init__()\n        self.p = P(c)\n\n    def forward(self, x):\n        return self.p(x)\n\nclass F(fnn.Module):\n    features: int\n\n    @fnn.compact\n    def __call__(self, x):\n        return fnn.Dense(self.features)(x)\n\ndef f(x: Float[Array, \"n 8\"], y: Float[Array, \"n 5\"], o):\n    a = P(4)(x)\n    b = Q(4)(x)\n    c = F(4)(x)\n    d = a + y\n    e = Q(o)(x)\n    g = e + y\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "e"), Some(&shape(&["n", "o"])));
        assert_eq!(find_shape(&analysis, "g"), Some(&shape(&["n", "5"])));
        assert_eq!(analysis.errors.len(), 1, "{:?}", messages(&analysis));
    }

    #[test]
    fn module_composition_keeps_the_symbol() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.a = nn.Linear(8, d)\n        self.b = nn.Linear(d, 4)\n        self.net = nn.Sequential(nn.Linear(8, d), nn.ReLU(), nn.Linear(d, 4))\n        self.twice = nn.Linear(8, d * 2)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        h = self.b(self.a(x))\n        i = self.net(x)\n        j = self.twice(x)\n        k = j + jnp.zeros((1, 4))\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "i"), Some(&shape(&["n", "4"])));
        assert_eq!(find_shape(&analysis, "j"), Some(&shape(&["n", "d * 2"])));
    }

    #[test]
    fn a_size_argument_of_a_parameter_or_attribute_does_not_conflict() {
        let analysis = analyze(&module(
            "a = torch.zeros(3, self.d)\nb = a + z\nc = x.view(3, self.d)\ne = c + z\nf = jnp.zeros((3, d))\ng = f + z\n",
        ));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "d"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "e"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "g"), Some(&shape(&["3", "4"])));
        let code = format!(
            "{HEAD}def f(d, b: Float[Array, \"3 4\"]):\n    z = jnp.zeros((3, d))\n    s = z + b\n    n = load()\n    u = jnp.zeros((3, n))\n    v = u + b\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "u"), Some(&shape(&["3", "n"])));
    }

    #[test]
    fn every_comparison_of_two_dims_allows_a_parameter_dim() {
        let analysis = analyze(&module(
            "a = jnp.concatenate([h, y], axis=0)\nb = torch.stack([h, y])\nc = h @ jnp.zeros((4, 2))\ne = jnp.zeros((4, 2))\nf = g(h)\nl = jnp.linalg.inv(h)\nm = jnp.concatenate([y, h], axis=0)\n",
        ));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["n+n", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["2", "n", "4"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["n", "2"])));
        assert_eq!(find_shape(&analysis, "m"), Some(&shape(&["n+n", "4"])));
        let call = format!(
            "{HEAD}def g(a: Float[Array, \"n 4\"]) -> Float[Array, \"n 2\"]:\n    pass\n\nclass M(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        h = g(self.l(x))\n"
        );
        let analysis = analyze(&call);
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["n", "2"])));
    }

    #[test]
    fn no_marked_name_leaves_the_analysis() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        h = self.l(x)\n        s = h.reshape(-1)\n"
        );
        let analysis = analyze(&code);
        let text = format!("{:?}", analysis);
        assert!(!text.contains("__free"), "{text}");
    }

    const ANNOTATED: &str = "def h(a: Float[Array, \"n d\"]):\n    return a\n\n\n";

    fn no_errors(code: &str) -> LayerShapeAnalysis {
        let analysis = analyze(code);
        assert_eq!(messages(&analysis), Vec::<&str>::new(), "{code}");
        analysis
    }

    #[test]
    fn a_name_around_a_class_hides_a_function_from_a_method() {
        let program = |outer: &str, call: &str| {
            format!(
                "{HEAD}import jax\n\ndef f(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n    return x\n\n\ndef h({outer}):\n    class C:\n        def m(self, x: Float[Array, \"n 3\"]):\n            y = {call}\n"
            )
        };
        for call in ["f(x)", "jax.vmap(f)(x)"] {
            assert_eq!(find_shape(&analyze(&program("f", call)), "y"), None, "{call}");
        }
        let local = program("c", "f(x)").replace("    class C:", "    f = load()\n    class C:");
        assert_eq!(find_shape(&analyze(&local), "y"), None);
        let closure = program("c", "f(x)").replace(
            "    class C:",
            "    def f(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 1\"]:\n        return x\n    class C:",
        );
        assert_eq!(find_shape(&analyze(&closure), "y"), Some(&shape(&["n", "1"])));
        let two_classes = program("c", "f(x)").replace(
            "    class C:\n        def m(self, x: Float[Array, \"n 3\"]):\n            y = f(x)\n",
            "    f = load()\n    class C:\n        class D:\n            def m(self, x: Float[Array, \"n 3\"]):\n                y = f(x)\n",
        );
        assert_eq!(find_shape(&analyze(&two_classes), "y"), None);
        assert_eq!(find_shape(&analyze(&program("c", "f(x)")), "y"), Some(&shape(&["n", "9"])));
    }

    #[test]
    fn a_local_of_unknown_value_is_no_annotation_claim() {
        for bind in ["d = cfg.d", "d = load()", "for d in cfg:\n        pass"] {
            let code = format!("{HEAD}{ANNOTATED}def g(cfg):\n    {bind}\n    x = jnp.zeros((3, d))\n    y = x + jnp.ones((3, 4))\n");
            assert_eq!(find_shape(&no_errors(&code), "y"), Some(&shape(&["3", "4"])), "{bind}");
        }
        let global = format!("{HEAD}{ANNOTATED}d = cfg.d\n\n\ndef g():\n    x = jnp.zeros((3, d))\n    y = x + jnp.ones((3, 4))\n");
        assert_eq!(find_shape(&no_errors(&global), "y"), Some(&shape(&["3", "4"])));
        let attribute = format!(
            "{HEAD}{ANNOTATED}class M(nn.Module):\n    def __init__(self, cfg):\n        self.d = cfg.d\n        self.l = nn.Linear(8, self.d)\n\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        r = self.l(x) + y\n        s = torch.zeros(3, self.d) + torch.zeros(3, 4)\n"
        );
        no_errors(&attribute);
        let layer = format!(
            "{HEAD}{ANNOTATED}class M(nn.Module):\n    def __init__(self, cfg):\n        d = cfg.d\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        r = self.l(x) + y\n"
        );
        assert_eq!(find_shape(&no_errors(&layer), "r"), None);
    }

    #[test]
    fn a_literal_or_an_unbound_name_stays_an_annotation_claim() {
        let code = format!(
            "{HEAD}D = 64\n\n\ndef f(x: Float[Array, \"n D\"], y: Float[Array, \"n 8\"]):\n    a = jnp.zeros((3, D))\n    b = jnp.zeros((3, n))\n    bad = a + jnp.zeros((3, 7))\n"
        );
        let analysis = analyze(&code);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "64"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "n"])));
        assert_eq!(analysis.errors.len(), 1);
    }

    #[test]
    fn two_free_dims_give_an_unknown_shape() {
        for line in [
            "w = jnp.add(x, y)",
            "w = jnp.maximum(x, y)",
            "w = torch.add(x, y)",
            "w = x.clone()\n    w += y",
            "w = x.clone()\n    w -= y",
            "w = jnp.zeros((a, 3, 3))\n    w @= jnp.zeros((b, 3, 3))",
        ] {
            let code = format!(
                "{HEAD}def f(a, b):\n    x = jnp.zeros((3, a))\n    y = jnp.zeros((3, b))\n    {line}\n    bad = w + jnp.zeros((2, 7))\n    return w\n\n\ndef g(a, b):\n    p = f(a, b)\n"
            );
            let analysis = analyze(&code);
            assert_eq!(find_shape(&analysis, "w"), None, "{line}");
            assert_eq!(find_shape(&analysis, "p"), None, "{line}");
            assert_eq!(messages(&analysis), Vec::<&str>::new(), "{line}");
        }
    }

    #[test]
    fn no_unproven_dim_leaves_the_analysis() {
        let head = "def f(a, b, c, cfg):\n    x = jnp.zeros((3, a))\n    y = jnp.zeros((3, b))\n    t = torch.zeros(3, a)\n    u = torch.zeros(3, b)\n";
        for line in [
            "return jnp.add(x, y)",
            "return x + y",
            "w = t.add(u)\n    return w",
            "t += u\n    return t",
            "x /= y\n    return x",
            "w = jnp.where(c, x, y)\n    return w",
            "w = torch.where(c, t, u)\n    return w",
            "w = jnp.stack([x, y])\n    return w",
            "w = torch.cat([t, u])\n    return w",
            "if c:\n        w = x\n    else:\n        w = y\n    return w",
            "if c:\n        return x\n    return y",
            "w = x if c else y\n    return w",
            "w = jnp.add(x, y).sum(0)\n    return w",
            "return jnp.add(x, y), x",
            "w = [jnp.add(x, y)]\n    return w[0]",
            "w = jnp.zeros((a, 3, 3)) @ jnp.zeros((b, 3, 3))\n    return w",
            "w = jnp.matmul(jnp.zeros((a, 3, 3)), jnp.zeros((b, 3, 3)))\n    return w",
            "w = nn.Embedding(cfg.d, cfg.e)\n    r = w(t.long())\n    return r",
            "w = nn.LayerNorm(cfg.d)(t)\n    return w",
            "w = t.view(cfg['d'], -1)\n    return w",
        ] {
            let code = format!("{HEAD}{head}    {line}\n\n\ndef g(a, b, c, cfg):\n    p = f(a, b, c, cfg)\n    q = p + jnp.zeros((2, 7))\n");
            let analysis = analyze(&code);
            let text = format!("{analysis:?}");
            assert!(!text.contains("__unproven") && !text.contains("__free"), "{line}: {text}");
        }
        let layers = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, cfg):\n        self.a = nn.LayerNorm(cfg.d)\n        self.b = nn.Sequential(nn.Linear(8, cfg[\"d\"]), nn.LayerNorm(cfg.d), nn.Linear(cfg.e, 3))\n        self.r = nn.LSTM(8, cfg.h)\n\n    def forward(self, x: Float[Array, \"b 8\"]):\n        r = self.a(x)\n        s = self.b(x)\n        o, h = self.r(x)\n"
        );
        let text = format!("{:?}", analyze(&layers));
        assert!(!text.contains("__unproven"), "{text}");
    }

    #[test]
    fn a_slice_bound_of_unknown_value_gives_an_unknown_size() {
        for (expression, want) in [
            ("v[:d] + v", ["3", "4"]),
            ("v[:, :d] + v", ["3", "4"]),
            ("v[:d]", ["?", "4"]),
            ("v[:, :d + 1]", ["3", "?"]),
            ("v[:len(d)]", ["?", "4"]),
            ("v[:cfg[\"d\"]]", ["?", "4"]),
        ] {
            let code = format!("{HEAD}def f(d, cfg):\n    v = torch.zeros(3, 4)\n    a = {expression}\n");
            assert_eq!(find_shape(&no_errors(&code), "a"), Some(&shape(&want)), "{expression}");
        }
    }

    #[test]
    fn a_slice_with_a_known_bound_keeps_its_size() {
        for (expression, want) in [
            ("v[:2]", ["2", "4"]),
            ("v[:9]", ["3", "4"]),
            ("v[:k]", ["2", "4"]),
            ("v[:-1]", ["2", "4"]),
            ("v[:-9]", ["0", "4"]),
            ("v[:, :]", ["3", "4"]),
            ("x[:n]", ["n", "4"]),
        ] {
            let code = format!(
                "{HEAD}k = 2\n\n\ndef f(x: Float[Array, \"n 4\"]):\n    v = torch.zeros(3, 4)\n    a = {expression}\n"
            );
            assert_eq!(find_shape(&no_errors(&code), "a"), Some(&shape(&want)), "{expression}");
        }
    }

    #[test]
    fn a_size_argument_of_a_method_on_any_receiver_does_not_conflict() {
        let chained = format!(
            "{HEAD}def f(d):\n    v = torch.zeros(3, 4)\n    a = v.clone().reshape(3, d) + v\n"
        );
        assert_eq!(find_shape(&no_errors(&chained), "a"), Some(&shape(&["3", "4"])));
        for expression in ["(v + 1).reshape(3, d) + v", "v[0:3].reshape(3, d) + v", "self.v.reshape(3, d) + v"] {
            no_errors(&format!("{HEAD}def f(self, d):\n    v = torch.zeros(3, 4)\n    a = {expression}\n"));
        }
        let expand = format!(
            "{HEAD}def f(d):\n    v = torch.zeros(3, 4)\n    b = v.unsqueeze(0).expand(d, 3, 4) + torch.zeros(2, 3, 4)\n"
        );
        assert_eq!(find_shape(&no_errors(&expand), "b"), Some(&shape(&["2", "3", "4"])));
    }

    #[test]
    fn a_subscript_size_does_not_conflict() {
        let code = format!("{HEAD}def f(cfg):\n    v = torch.zeros(3, 4)\n    a = torch.zeros(3, cfg[\"d\"]) + v\n");
        assert_eq!(find_shape(&no_errors(&code), "a"), Some(&shape(&["3", "4"])));
        let layer = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self, cfg):\n        self.l = nn.Linear(8, cfg[\"d\"])\n\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        r = self.l(x) + y\n"
        );
        assert_eq!(find_shape(&no_errors(&layer), "r"), None);
    }

    #[test]
    fn a_string_argument_of_a_layer_stays_a_string() {
        let code = format!(
            "{HEAD}class M(nn.Module):\n    def __init__(self):\n        self.c = nn.Conv2d(3, 8, 3, padding=\"same\")\n\n    def forward(self, x: Float[Array, \"b 3 32 32\"]):\n        r = self.c(x)\n"
        );
        assert_eq!(find_shape(&no_errors(&code), "r"), Some(&shape(&["b", "8", "32", "32"])));
    }

    #[test]
    fn a_nonlocal_rebind_from_a_method_poisons_the_name() {
        let around = "    def f(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n        return x\n";
        let same = format!(
            "{HEAD}import jax\n\ndef h():\n{around}    class C:\n        def m(self, x: Float[Array, \"n 3\"]):\n            nonlocal f\n            f = load()\n            y = f(x)\n"
        );
        assert_eq!(find_shape(&analyze(&same), "y"), None);
        let other = format!(
            "{HEAD}import jax\n\ndef h():\n{around}    class C:\n        def s(self):\n            nonlocal f\n            f = load()\n\n        def m(self, x: Float[Array, \"n 3\"]):\n            y = f(x)\n            z = jax.vmap(f)(x)\n"
        );
        let analysis = analyze(&other);
        assert_eq!(find_shape(&analysis, "y"), None);
        assert_eq!(find_shape(&analysis, "z"), None);
        let in_h = format!(
            "{HEAD}def h(x: Float[Array, \"n 3\"]):\n{around}    class C:\n        def s(self):\n            nonlocal f\n            f = load()\n\n    y = f(x)\n"
        );
        assert_eq!(find_shape(&analyze(&in_h), "y"), None);
        let no_closure = format!(
            "{HEAD}def h():\n    v = torch.zeros(3, 4)\n    class C:\n        def s(self):\n            nonlocal v\n            v = load()\n    y = v\n"
        );
        assert_eq!(find_shape(&analyze(&no_closure), "y"), None);
        let untouched = format!(
            "{HEAD}def h():\n{around}    class C:\n        def m(self, x: Float[Array, \"n 3\"]):\n            y = f(x)\n"
        );
        assert_eq!(find_shape(&analyze(&untouched), "y"), Some(&shape(&["n", "9"])));
    }

    #[test]
    fn a_literal_name_is_its_integer_next_to_an_annotation_symbol() {
        let annotated = "def h(a: Float[Array, \"n d\"]):\n    return a\n\n\n";
        let local = format!(
            "{HEAD}{annotated}def g():\n    d = 4\n    x = jnp.zeros((3, d))\n    y = x + jnp.ones((3, 4))\n"
        );
        let analysis = no_errors(&local);
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["3", "4"])));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "4"])));

        for bind in ["    d = 4\n", ""] {
            let top = if bind.is_empty() { "d = 4\n\n\n" } else { "" };
            let layer = format!(
                "{HEAD}{top}{annotated}class M(nn.Module):\n    def __init__(self):\n{bind}        self.l = nn.Linear(8, d)\n\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        r = self.l(x) + y\n"
            )
            .replace("    d = 4\n        self.l", "        d = 4\n        self.l");
            assert_eq!(messages(&analyze(&layer)), Vec::<&str>::new(), "{bind}");
        }

        let module = format!(
            "{HEAD}d = 4\n\n\n{annotated}def g():\n    x = jnp.zeros((3, d))\n    y = x + jnp.ones((3, 4))\n"
        );
        assert_eq!(find_shape(&no_errors(&module), "y"), Some(&shape(&["3", "4"])));

        let attribute = format!(
            "{HEAD}{annotated}class M(nn.Module):\n    def __init__(self):\n        self.d = 4\n        self.l = nn.Linear(8, self.d)\n\n    def forward(self, x: Float[Array, \"b 8\"], y: Float[Array, \"b 4\"]):\n        r = self.l(x) + y\n"
        );
        no_errors(&attribute);

        let one = format!(
            "{HEAD}{annotated}def g():\n    d = 1\n    y = jnp.zeros((3, d)) + jnp.ones((3, 5))\n"
        );
        assert_eq!(find_shape(&no_errors(&one), "y"), Some(&shape(&["3", "5"])));
    }

    #[test]
    fn a_literal_name_equals_the_symbol_of_the_same_name() {
        let code = format!(
            "{HEAD}D = 64\n\n\ndef f(x: Float[Array, \"3 D\"]) -> Float[Array, \"3 D\"]:\n    a = jnp.zeros((3, D))\n    ok = a + jnp.zeros((3, 64))\n    same = a + x\n    both = x + a\n    twice = jnp.zeros((3, D * 2))\n    fine = twice + jnp.zeros((3, 128))\n    head = a[:, :D]\n    return a\n"
        );
        let analysis = no_errors(&code);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "64"])));
        assert_eq!(find_shape(&analysis, "ok"), Some(&shape(&["3", "64"])));
        assert_eq!(find_shape(&analysis, "same"), Some(&shape(&["3", "D"])));
        assert_eq!(find_shape(&analysis, "head"), Some(&shape(&["3", "64"])));
    }

    #[test]
    fn a_literal_name_keeps_the_true_error_and_a_symbol_stays_a_claim() {
        let literal = format!(
            "{HEAD}def h(a: Float[Array, \"n d\"]):\n    return a\n\n\ndef g():\n    d = 4\n    bad = jnp.zeros((3, d)) + jnp.ones((3, 5))\n"
        );
        assert_eq!(analyze(&literal).errors.len(), 1);
        assert!(!messages(&analyze(&literal))[0].contains("__"), "{:?}", messages(&analyze(&literal)));
        let unbound = format!(
            "{HEAD}def h(a: Float[Array, \"n d\"]):\n    x = jnp.zeros((3, d))\n    bad = x + jnp.ones((3, 4))\n"
        );
        assert_eq!(analyze(&unbound).errors.len(), 1);
    }

    #[test]
    fn a_slice_of_an_integer_axis_follows_python() {
        let run = |index: &str, source: &str| {
            let code = format!("{HEAD}v = {source}\nr = v[{index}]\n");
            find_shape(&no_errors(&code), "r").cloned()
        };
        for (index, dims) in [
            ("1:", vec!["2", "4"]),
            ("-2:", vec!["2", "4"]),
            ("::2", vec!["2", "4"]),
            ("1:3:2", vec!["1", "4"]),
            (":2:2", vec!["1", "4"]),
            ("::-1", vec!["3", "4"]),
            ("::-2", vec!["2", "4"]),
            ("2:0:-1", vec!["2", "4"]),
            ("-1:-4:-1", vec!["3", "4"]),
            (":1:-1", vec!["1", "4"]),
            ("1::-1", vec!["2", "4"]),
            ("0:1:-1", vec!["0", "4"]),
            (":-5", vec!["0", "4"]),
            ("-10:", vec!["3", "4"]),
            ("10:", vec!["0", "4"]),
            ("2:1", vec!["0", "4"]),
            ("-2:-1", vec!["1", "4"]),
            (":100", vec!["3", "4"]),
            ("::100", vec!["1", "4"]),
            ("None:2", vec!["2", "4"]),
            (":, 1:", vec!["3", "3"]),
            (":, ::2", vec!["3", "2"]),
            ("1:, 1:3", vec!["2", "2"]),
            ("..., 1:", vec!["3", "3"]),
            ("None, 1:", vec!["1", "2", "4"]),
            ("::2, ::3, None", vec!["2", "2", "1"]),
        ] {
            assert_eq!(run(index, "jnp.zeros((3, 4))"), Some(shape(&dims)), "{index}");
        }
        assert_eq!(run("::0", "torch.zeros(3, 4)"), None);
    }

    #[test]
    fn a_literal_name_bound_gives_the_integer_of_a_slice() {
        let code = format!(
            "{HEAD}def f():\n    k = 1\n    v = torch.zeros(3, 4)\n    a = v[k:]\n    b = v[::k + 1]\n    c = v[1 + 1:]\n"
        );
        let analysis = no_errors(&code);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["2", "4"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["2", "4"])));
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["1", "4"])));
    }

    #[test]
    fn a_symbolic_slice_bound_is_an_unknown_size() {
        for (index, other, want) in [
            ("0:d", "torch.zeros(2, 4)", ["?", "4"]),
            ("d:", "torch.zeros(2, 4)", ["?", "4"]),
            ("1:d", "torch.zeros(2, 4)", ["?", "4"]),
            ("::d", "torch.zeros(2, 4)", ["?", "4"]),
            (":d", "torch.zeros(2, 4)", ["?", "4"]),
            ("f(a[\"x:y\"]):", "torch.zeros(2, 4)", ["?", "4"]),
            (":, 0:d", "torch.zeros(3, 2)", ["3", "?"]),
            (":, 1:d", "torch.zeros(3, 2)", ["3", "?"]),
        ] {
            let code = format!(
                "{HEAD}def f(d, a):\n    v = torch.zeros(3, 4)\n    r = v[{index}]\n    s = r + {other}\n"
            );
            assert_eq!(find_shape(&no_errors(&code), "r"), Some(&shape(&want)), "{index}");
        }
    }

    #[test]
    fn a_slice_of_a_symbolic_axis_has_a_form_or_an_unknown_size() {
        let run = |index: &str| {
            let code = format!(
                "{HEAD}def f(w: Float[Array, \"n m\"], k):\n    r = w[{index}]\n"
            );
            find_shape(&no_errors(&code), "r").cloned()
        };
        for (index, want) in [
            (":2", ["min(2, n)", "m"]),
            ("1:", ["n-1", "m"]),
            ("-1:", ["min(1, n)", "m"]),
            (":-1", ["n-1", "m"]),
            ("::2", ["ceil(n/2)", "m"]),
            ("1:3", ["min(2, n-1)", "m"]),
            (":k", ["?", "m"]),
            (":m", ["?", "m"]),
            (":, :2", ["n", "min(2, m)"]),
            (":, ::2", ["n", "ceil(m/2)"]),
        ] {
            assert_eq!(run(index), Some(shape(&want)), "{index}");
        }
        for index in [":", "::", "::1", "0:", "0::1", "::-1", ":n", ":, :m", "...", "None:None"] {
            assert_eq!(run(index), Some(shape(&["n", "m"])), "{index}");
        }
    }

    #[test]
    fn a_string_padding_of_a_conv_gives_the_size() {
        let run = |layer: &str, input: &str| {
            let code = format!(
                "{HEAD}class M(nn.Module):\n    def __init__(self, p=\"same\"):\n        self.c = {layer}\n\n    def forward(self, x: Float[Array, \"{input}\"]):\n        r = self.c(x)\n"
            );
            find_shape(&no_errors(&code), "r").cloned()
        };
        for (layer, input, dims) in [
            ("nn.Conv2d(3, 8, 3, padding=\"same\")", "b 3 8 8", vec!["b", "8", "8", "8"]),
            ("nn.Conv2d(3, 8, 5, 1, 'same')", "b 3 8 8", vec!["b", "8", "8", "8"]),
            ("nn.Conv2d(3, 8, 3, padding=\"valid\")", "b 3 8 8", vec!["b", "8", "6", "6"]),
            ("nn.Conv2d(3, 8, 3, padding=\"same\")", "b 3 h w", vec!["b", "8", "h", "w"]),
            ("nn.Conv2d(3, 8, 3, padding=\"valid\")", "b 3 h w", vec!["b", "8", "h-2", "w-2"]),
            ("nn.Conv1d(3, 8, 3, padding=\"same\")", "b 3 8", vec!["b", "8", "8"]),
            ("nn.Conv1d(3, 8, 3, padding=\"valid\")", "b 3 8", vec!["b", "8", "6"]),
            ("nn.Conv3d(3, 8, 3, padding=\"same\")", "b 3 8 8 8", vec!["b", "8", "8", "8", "8"]),
            ("nn.Conv3d(3, 8, 3, padding=\"valid\")", "b 3 8 8 8", vec!["b", "8", "6", "6", "6"]),
            ("eqx.nn.Conv2d(3, 8, 3, padding=\"SAME\", key=k)", "3 8 8", vec!["8", "8", "8"]),
            ("eqx.nn.Conv2d(3, 8, 3, padding=\"SAME_LOWER\", key=k)", "3 8 8", vec!["8", "8", "8"]),
            ("eqx.nn.Conv2d(3, 8, 3, padding=\"VALID\", key=k)", "3 8 8", vec!["8", "6", "6"]),
            ("eqx.nn.ConvTranspose2d(3, 8, 3, padding=\"SAME\", key=k)", "3 8 8", vec!["8", "8", "8"]),
            ("eqx.nn.ConvTranspose2d(3, 8, 3, padding=\"VALID\", key=k)", "3 8 8", vec!["8", "10", "10"]),
            ("fnn.Conv(8, (3, 3))", "b 8 8 3", vec!["b", "8", "8", "8"]),
            ("fnn.Conv(8, (3, 3), padding=\"SAME\")", "b 8 8 3", vec!["b", "8", "8", "8"]),
            ("fnn.Conv(8, (3, 3), padding=\"CIRCULAR\")", "b 8 8 3", vec!["b", "8", "8", "8"]),
            ("fnn.Conv(8, (3, 5), padding=\"VALID\")", "b 8 8 3", vec!["b", "6", "4", "8"]),
            ("fnn.Conv(8, (3,), padding=\"valid\")", "b 8 3", vec!["b", "6", "8"]),
            ("fnn.Conv(8, (3, 3), padding=1)", "b 8 8 3", vec!["b", "8", "8", "8"]),
            ("fnn.Conv(8, (3, 3), padding=0)", "b h w 3", vec!["b", "h-2", "w-2", "8"]),
        ] {
            assert_eq!(run(layer, input), Some(shape(&dims)), "{layer}");
        }
        for (layer, input) in [
            ("nn.Conv2d(3, 8, 3, stride=2, padding=\"same\")", "b 3 8 8"),
            ("nn.Conv2d(3, 8, 3, padding=\"zeros\")", "b 3 8 8"),
            ("nn.Conv2d(3, 8, 3, padding=p)", "b 3 8 8"),
            ("nn.ConvTranspose2d(3, 8, 3, padding=\"same\")", "b 3 8 8"),
            ("eqx.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=\"SAME\", key=k)", "3 8 8"),
            ("nn.MaxPool2d(3, padding=\"same\")", "b 3 8 8"),
            ("fnn.Conv(8, (3, 3), padding=\"BOGUS\")", "b 8 8 3"),
            ("fnn.Conv(8, (3, 3), padding=((1, 1), (2, 2)))", "b 8 8 3"),
            ("fnn.Conv(8, (3, 3), padding=q)", "b 8 8 3"),
            ("fnn.Conv(8, (3, 3), padding=\"VALID\", kernel_dilation=2)", "b 8 8 3"),
            ("fnn.Conv(8, (3, 3), input_dilation=2)", "b 8 8 3"),
        ] {
            assert_eq!(run(layer, input), None, "{layer}");
        }
    }

    #[test]
    fn an_attribute_of_an_instance_is_no_integer_for_a_size() {
        let indirect = format!(
            "{HEAD}class C:\n    def __init__(self):\n        self.f = 4\n        setattr(self, \"f\", 5)\n        self.i = 4\n        o = self\n        o.i = 5\n\n    def m(self):\n        a = torch.zeros(3, self.f)\n        b = torch.zeros(3, self.i)\n"
        );
        let analysis = no_errors(&indirect);
        assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "self.f"])));
        assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "self.i"])));
        for outside in ["c.d = 7", "C.d = 7", "pass"] {
            let code = format!(
                "{HEAD}class C:\n    def __init__(self):\n        self.d = 4\n\n    def m(self):\n        a = torch.zeros(3, self.d)\n        b = a + torch.zeros(3, 7)\n        s = torch.zeros(5, 4)[:self.d]\n\n\nc = C()\n{outside}\nc.m()\n"
            );
            let analysis = no_errors(&code);
            assert_eq!(find_shape(&analysis, "a"), Some(&shape(&["3", "self.d"])), "{outside}");
            assert_eq!(find_shape(&analysis, "b"), Some(&shape(&["3", "7"])), "{outside}");
            assert_eq!(find_shape(&analysis, "s"), Some(&shape(&["?", "4"])), "{outside}");
        }
    }
}

#[cfg(test)]
mod bare_callee_scope_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Option<Vec<String>> {
        Some(dims.iter().map(|dim| dim.to_string()).collect())
    }

    /// `f` and `g` map a vector to a vector of another size, `step` is a scan
    /// body. `xs` in a test body has the shape `B n`.
    const HEAD: &str = "import jax\nimport jax.numpy as jnp\nfrom jax import vmap\nfrom jaxtyping import Float, Array\n\ndef f(x: Float[Array, \"n\"]) -> Float[Array, \"m\"]:\n    pass\n\ndef g(x: Float[Array, \"n\"]) -> Float[Array, \"k\"]:\n    pass\n\ndef step(c, x):\n    return c, x\n\n";

    fn program(rest: &str) -> String {
        format!("{HEAD}{rest}")
    }

    /// A shape of `y` in the function of the body, with no error.
    fn y_in(function: &str, code: &str) -> Option<Vec<String>> {
        let analysis = analyze(code);
        assert_eq!(analysis.errors.len(), 0, "{:?}\n{code}", analysis.errors);
        let scope = analysis.scopes.iter().find(|s| s.function_name.as_deref() == Some(function));
        scope.and_then(|s| s.shapes.get("y").cloned())
    }

    #[test]
    fn a_parameter_hides_a_function_from_every_callable_path() {
        for call in [
            "y = vmap(f)(xs)",
            "y = jax.vmap(jax.vmap(f))(xs)",
            "vf = vmap(f)\n    y = vf(xs)",
            "y = jax.lax.map(f, xs)",
            "y = f(xs)",
        ] {
            let code = program(&format!("def apply(f, xs: Float[Array, \"B n\"]):\n    {call}\n"));
            assert_eq!(y_in("apply", &code), None, "{call}");
        }
        for parameters in ["*f", "**f", "*, f", "f=g", "f: int", "f, /", "*, f=g"] {
            let code = program(&format!("def apply(xs: Float[Array, \"B n\"], {parameters}):\n    y = vmap(f)(xs)\n"));
            assert_eq!(y_in("apply", &code), None, "{parameters}");
        }
        let scan = "def apply(step, c: Float[Array, \"h\"], xs: Float[Array, \"B n\"]):\n    c2, y = jax.lax.scan(step, c, xs)\n";
        assert_eq!(y_in("apply", &program(scan)), None);
    }

    #[test]
    fn every_local_binder_hides_a_function() {
        for (bind, body) in [
            ("f = load()", "    "),
            ("for f in fs:", "        "),
            ("with open_it() as f:", "        "),
            ("from other import f", "    "),
            ("(f := load())", "    "),
            ("f, h = load()", "    "),
            ("try:\n        pass\n    except E as f:", "        "),
            ("match v:\n        case [f]:", "            "),
        ] {
            for call in ["y = vmap(f)(xs)", "y = jax.lax.map(f, xs)", "y = f(xs)"] {
                let code = program(&format!(
                    "def apply(fs, v, xs: Float[Array, \"B n\"]):\n    {bind}\n{body}{call}\n"
                ));
                assert_eq!(y_in("apply", &code), None, "{bind} / {call}");
            }
        }
        let scan = "def apply(c: Float[Array, \"h\"], xs: Float[Array, \"B n\"]):\n    step = load()\n    c2, y = jax.lax.scan(step, c, xs)\n";
        assert_eq!(y_in("apply", &program(scan)), None);
    }

    #[test]
    fn a_comprehension_or_a_lambda_hides_a_function() {
        for line in [
            "y = jnp.exp(jnp.stack([vmap(f)(xs) for f in fs])[0])",
            "y = jnp.stack([f(xs) for f in fs])",
            "y = jnp.stack([jax.lax.map(f, xs) for f in fs])",
            "y = jnp.stack(list(vmap(f)(xs) for f in fs))",
            "y = jax.lax.map(lambda f: vmap(f)(xs), fs)",
            "y = jax.lax.map(lambda f: f(xs), fs)",
        ] {
            let code = program(&format!(
                "def apply(fs: Float[Array, \"T n\"], xs: Float[Array, \"B n\"]):\n    {line}\n"
            ));
            assert_eq!(y_in("apply", &code), None, "{line}");
        }
        let scan = "def apply(fs: Float[Array, \"T n\"], c: Float[Array, \"h\"], xs: Float[Array, \"B n\"]):\n    y = jnp.stack([jax.lax.scan(step, c, xs)[1] for step in fs])\n";
        assert_eq!(y_in("apply", &program(scan)), None);
    }

    #[test]
    fn an_enclosing_function_hides_a_function_from_its_closures() {
        let code = program(
            "def apply(f, xs: Float[Array, \"B n\"]):\n    def inner():\n        y = vmap(f)(xs)\n    inner()\n",
        );
        assert_eq!(y_in("inner", &code), None);
        let local = program(
            "def apply(xs: Float[Array, \"B n\"]):\n    f = load()\n    def inner():\n        y = vmap(f)(xs)\n    inner()\n",
        );
        assert_eq!(y_in("inner", &local), None);
        let nonlocal = program(
            "def apply(xs: Float[Array, \"B n\"]):\n    def f(x: Float[Array, \"n\"]) -> Float[Array, \"q\"]:\n        pass\n    def swap():\n        nonlocal f\n        f = load()\n    y = vmap(f)(xs)\n",
        );
        assert_eq!(y_in("apply", &nonlocal), None);
    }

    #[test]
    fn a_second_binding_of_a_function_name_leaves_it_unknown() {
        let reader = "def read(xs: Float[Array, \"B n\"]):\n    y = vmap(f)(xs)\n";
        for other in [
            "f = load()\n",
            "for f in fs:\n    pass\n",
            "with open_it() as f:\n    pass\n",
            "class f:\n    pass\n",
            "def swap():\n    global f\n    f = load()\n",
        ] {
            let code = program(&format!("{other}\n{reader}"));
            assert_eq!(y_in("read", &code), None, "{other}");
        }
    }

    #[test]
    fn a_name_that_is_not_hidden_still_finds_its_function() {
        let read = "def read(xs: Float[Array, \"B n\"], a: Float[Array, \"n\"], c: Float[Array, \"h\"]):\n";
        for (body, want) in [
            ("y = vmap(f)(xs)", ["B", "m"].as_slice()),
            ("y = jax.lax.map(g, xs)", &["B", "k"]),
            ("y = f(a)", &["m"]),
            ("h = [g for g in xs]\n    y = vmap(g)(xs)", &["B", "k"]),
            ("c2, y = jax.lax.scan(step, c, xs)", &["B", "n"]),
        ] {
            assert_eq!(y_in("read", &program(&format!("{read}    {body}\n"))), shape(want), "{body}");
        }
        let nested = program(
            "def apply(xs: Float[Array, \"B n\"]):\n    def f(x: Float[Array, \"n\"]) -> Float[Array, \"q\"]:\n        pass\n    def inner():\n        y = vmap(f)(xs)\n    inner()\n",
        );
        assert_eq!(y_in("inner", &nested), shape(&["B", "q"]));
        let method = program(
            "class M:\n    def read(self, g, xs: Float[Array, \"B n\"]):\n        y = vmap(f)(xs)\n",
        );
        assert_eq!(y_in("read", &method), shape(&["B", "m"]));
    }

    #[test]
    fn a_callable_passed_to_a_parameter_stays_unknown_without_a_false_error() {
        let callee = "def apply(f, xs):\n    return vmap(f)(xs)\n\n";
        let caller = "def run(a: Float[Array, \"B n\"]):\n    y = apply(g, a)\n";
        assert_eq!(y_in("run", &program(&format!("{callee}{caller}"))), None);
        let wrong_size = "def small(x: Float[Array, \"3\"]) -> Float[Array, \"m\"]:\n    pass\n\n";
        let other_size = "def run(a: Float[Array, \"B 4\"]):\n    y = apply(g, a)\n";
        let renamed = callee.replace("(f,", "(small,").replace("vmap(f)", "vmap(small)");
        assert_eq!(y_in("run", &program(&format!("{wrong_size}{renamed}{other_size}"))), None);
    }

    #[test]
    fn a_vmap_name_belongs_to_the_scope_that_binds_it() {
        let elsewhere = "def make():\n    vf = vmap(f)\n\n";
        let read = "def read(xs: Float[Array, \"B n\"]):\n    y = vf(xs)\n";
        assert_eq!(y_in("read", &program(&format!("{elsewhere}{read}"))), None);
        let parameter = "def read(vf, xs: Float[Array, \"B n\"]):\n    y = vf(xs)\n";
        let code = program(&format!("vf = vmap(f)\n\n{parameter}"));
        assert_eq!(y_in("read", &code), None);
        let comprehension = "def read(fs, xs: Float[Array, \"B n\"]):\n    y = jnp.stack([vf(xs) for vf in fs])\n";
        let code = program(&format!("vf = vmap(f)\n\n{comprehension}"));
        assert_eq!(y_in("read", &code), None);
        let other = "def other():\n    vf = vmap(g)\n";
        let own = "def read(xs: Float[Array, \"B n\"]):\n    vf = vmap(f)\n    y = vf(xs)\n";
        assert_eq!(y_in("read", &program(&format!("{own}\n{other}"))), shape(&["B", "m"]));
        let module = format!("vf = vmap(g)\n\n{read}\n{other}");
        assert_eq!(y_in("read", &program(&module)), shape(&["B", "k"]));
        let closure = "def outer(xs: Float[Array, \"B n\"]):\n    vf = vmap(f)\n    def inner():\n        y = vf(xs)\n    inner()\n";
        assert_eq!(y_in("inner", &program(closure)), shape(&["B", "m"]));
    }

    #[test]
    fn a_class_body_hides_a_function_from_its_own_statements_only() {
        let mut body = String::from("xs: Float[Array, \"B n\"] = load()\n\nclass A:\n    f = load()\n");
        body.push_str("    y = vmap(f)(xs)\n");
        let analysis = analyze(&program(&body));
        assert_eq!(analysis.scopes[0].shapes.get("y"), None);
        let method = "class A:\n    f = load()\n    def read(self, xs: Float[Array, \"B n\"]):\n        y = vmap(f)(xs)\n";
        assert_eq!(y_in("read", &program(method)), shape(&["B", "m"]));
    }

    #[test]
    fn a_callable_or_vmap_shadowed_at_the_call_stays_unknown() {
        let vmap = "def read(vmap, xs: Float[Array, \"B n\"]):\n    y = vmap(f)(xs)\n";
        assert_eq!(y_in("read", &program(vmap)), None);
    }

    #[test]
    fn a_vmap_name_with_a_second_binding_stays_unknown() {
        for lines in [
            "vf = vmap(f)\n    vf = load()",
            "vf = vmap(f)\n    if c:\n        vf = load()",
            "if c:\n        vf = load()\n    else:\n        vf = vmap(f)",
            "if c:\n        vf = vmap(g)\n    else:\n        vf = vmap(f)",
            "vf = vmap(f)\n    vf = jax.vmap(f, in_axes=(0, 1))",
            "vf = vmap(f)\n    for vf in fs:\n        pass",
        ] {
            let code = program(&format!(
                "def read(c, fs, xs: Float[Array, \"B n\"]):\n    {lines}\n    y = vf(xs)\n"
            ));
            assert_eq!(y_in("read", &code), None, "{lines}");
        }
    }
}

#[cfg(test)]
mod self_wrap_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    const HEAD: &str = "import jax\nimport functools\nimport equinox as eqx\nfrom jaxtyping import Float, Array\n\n";
    const F: &str = "def f(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n    return x\n\n";
    const READ: &str = "def read(x: Float[Array, \"n 3\"]):\n    y = f(x)\n";

    fn y_in(dir: &std::path::Path, code: &str) -> Option<Vec<String>> {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(code, None).unwrap();
        let roots = vec![dir.to_path_buf()];
        let read = |p: &PathBuf| fs::read_to_string(p).ok();
        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 8, None).unwrap();
        find_shape(&analysis, "y").cloned()
    }

    fn y_of(code: &str) -> Option<Vec<String>> {
        y_in(tempfile::tempdir().unwrap().path(), code)
    }

    fn found() -> Option<Vec<String>> {
        Some(vec!["n".to_string(), "9".to_string()])
    }

    const KEEPS: [&str; 10] = [
        "f = jax.jit(f)\n",
        "f = jax.jit(f, static_argnums=(1,))\n",
        "f = jax.checkpoint(f)\n",
        "f = jax.remat(f)\n",
        "f = eqx.filter_jit(f)\n",
        "f = functools.partial(f)\n",
        "f: Any = jax.jit(f)\n",
        "f = jax.jit(f)\nf = jax.checkpoint(f)\n",
        "if c:\n    f = jax.jit(f)\n",
        "from jax import jit\nf = jit(f)\n",
    ];

    const OTHER: [&str; 13] = [
        "f = wrap(f)\n",
        "f = jax.grad(f)\n",
        "f = jax.vmap(f)\n",
        "f = jax.jit(g)\n",
        "f = jax.jit(jax.vmap(f))\n",
        "f = jax.jit(f, 1)\n",
        "f = jax.jit(f, **options)\n",
        "f = jax.jit(*f)\n",
        "f = functools.partial(f, 2)\n",
        "f = functools.partial(f, x=2)\n",
        "f = jit(f)\n",
        "f = jax.jit(f)\nf = load()\n",
        "from torch import jit\nf = jit(f)\n",
    ];

    #[test]
    fn a_wrap_that_keeps_the_signature_keeps_the_function() {
        for wrap in KEEPS {
            assert_eq!(y_of(&format!("{HEAD}{F}{wrap}\n{READ}")), found(), "{wrap}");
            let vmapped = READ.replace("n 3", "b n 3").replace("f(x)", "jax.vmap(f)(x)");
            let want = Some(vec!["b".to_string(), "n".to_string(), "9".to_string()]);
            assert_eq!(y_of(&format!("{HEAD}{F}{wrap}\n{vmapped}")), want, "{wrap}");
        }
        let call = "x: Float[Array, \"n 3\"] = load()\ny = f(x)\n";
        assert_eq!(y_of(&format!("{HEAD}{F}f = jax.jit(f)\n{call}")), found());
    }

    #[test]
    fn a_nested_wrap_keeps_the_nested_function() {
        let nested = "def read(x: Float[Array, \"n 3\"]):\n    def f(v: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n        return v\n    f = jax.jit(f)\n    y = f(x)\n";
        assert_eq!(y_of(&format!("{HEAD}{nested}")), found());
    }

    #[test]
    fn any_other_rebinding_stays_unknown() {
        for wrap in OTHER {
            assert_eq!(y_of(&format!("{HEAD}{F}{wrap}\n{READ}")), None, "{wrap}");
        }
    }

    #[test]
    fn a_wrap_needs_the_proven_transform() {
        for head in [
            "import flax as jax\nimport functools\n",
            "import jax\nimport functools\njax = load()\n",
            "class jax:\n    pass\n",
        ] {
            assert_eq!(y_of(&format!("{head}{F}f = jax.jit(f)\n\n{READ}")), None, "{head}");
        }
        let shadowed = format!("{HEAD}def read(x: Float[Array, \"n 3\"], jit):\n    def f(v: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n        return v\n    f = jit(f)\n    y = f(x)\n");
        assert_eq!(y_of(&format!("from jax import jit\n{shadowed}")), None);
    }

    #[test]
    fn a_wrap_of_a_parameter_or_a_global_stays_unknown() {
        let parameter = format!("{HEAD}{F}def read(f, x: Float[Array, \"n 3\"]):\n    f = jax.jit(f)\n    y = f(x)\n");
        assert_eq!(y_of(&parameter), None);
        let local = format!("{HEAD}{F}def read(x: Float[Array, \"n 3\"]):\n    f = jax.jit(f)\n    y = f(x)\n");
        assert_eq!(y_of(&local), None);
        let global = format!("{HEAD}{F}def set_it():\n    global f\n    f = jax.jit(f)\n\n{READ}");
        assert_eq!(y_of(&global), None);
    }

    #[test]
    fn a_vmap_sequence_on_one_name_stays_unknown() {
        let code = format!(
            "{HEAD}{F}def g(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 1\"]:\n    return x\n\ndef read(x: Float[Array, \"b n 3\"]):\n    vf = jax.vmap(f)\n    a = vf(x)\n    vf = jax.vmap(g)\n    y = vf(x)\n"
        );
        assert_eq!(y_of(&code), None);
    }

    #[test]
    fn a_wrap_in_the_module_of_an_import_keeps_the_function() {
        let reader = format!("{HEAD}from helper import f\n\n{READ}");
        for wrap in KEEPS.iter().filter(|wrap| !wrap.starts_with("f: Any")) {
            let dir = tempfile::tempdir().unwrap();
            let helper = format!("{HEAD}{F}{wrap}");
            fs::write(dir.path().join("helper.py"), helper).unwrap();
            assert_eq!(y_in(dir.path(), &reader), found(), "{wrap}");
        }
        for wrap in OTHER {
            let dir = tempfile::tempdir().unwrap();
            fs::write(dir.path().join("helper.py"), format!("{HEAD}{F}{wrap}")).unwrap();
            assert_eq!(y_in(dir.path(), &reader), None, "{wrap}");
        }
    }

    #[test]
    fn a_wrap_of_an_imported_name_keeps_the_function() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("helper.py"), format!("from jaxtyping import Float, Array\n\n{F}")).unwrap();
        let code = |wrap: &str| format!("{HEAD}from helper import f\n{wrap}\n{READ}");
        assert_eq!(y_in(dir.path(), &code("f = jax.jit(f)\n")), found());
        for wrap in ["f = wrap(f)\n", "f = jax.grad(f)\n", "f = jax.jit(f)\nf = load()\n"] {
            assert_eq!(y_in(dir.path(), &code(wrap)), None, "{wrap}");
        }
    }

    #[test]
    fn a_wrap_does_not_keep_a_class() {
        let dir = tempfile::tempdir().unwrap();
        let helper = "import jax\nclass f:\n    pass\nf = jax.jit(f)\n";
        fs::write(dir.path().join("helper.py"), helper).unwrap();
        let symbol = crate::python_ast::find_top_level_symbol;
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(helper, None).unwrap();
        assert_eq!(symbol(tree.root_node(), helper, "f").unwrap(), None);
    }
}

#[cfg(test)]
mod size_soundness_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    const HEAD: &str = "import torch\nimport torch.nn as nn\nimport jax.numpy as jnp\nimport flax.linen as fnn\nimport equinox as eqx\nfrom jaxtyping import Float, Array\n\n";
    const SYMBOLS: &str = "def h(a: Float[Array, \"n m d k\"]):\n    return a\n\n\n";

    fn y_of(body: &str) -> Option<Vec<String>> {
        let code = format!("{HEAD}{body}");
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{code}\n{:?}", analysis.errors);
        find_shape(&analysis, "y").cloned()
    }

    fn y_in_function(lines: &str) -> Option<Vec<String>> {
        y_of(&format!("{SYMBOLS}def g():\n{lines}"))
    }

    fn conv_y(ctor: &str, input: &str) -> Option<Vec<String>> {
        y_of(&format!("key = 0\nl = {ctor}\ny = l({input})\n"))
    }

    fn errors(body: &str) -> usize {
        analyze(&format!("{HEAD}{body}")).errors.len()
    }

    #[test]
    fn a_sum_of_literal_marks_is_the_integer() {
        let concat = "    d = 4\n    x = jnp.zeros((d, 3))\n    y = jnp.concatenate([x, x], axis=0)\n";
        assert_eq!(y_in_function(concat), Some(shape(&["8", "3"])));
        let fine = format!("{concat}    z = y + jnp.zeros((8, 3))\n");
        assert_eq!(y_in_function(&fine), Some(shape(&["8", "3"])));
        let bad = format!("{SYMBOLS}def g():\n{concat}    z = y + jnp.zeros((9, 3))\n");
        assert_eq!(errors(&bad), 1);
        let inline = "    d = 4\n    x = jnp.zeros((d, 3))\n    y = jnp.concatenate([x, x], axis=0) + jnp.zeros((8, 3))\n";
        assert_eq!(y_in_function(inline), Some(shape(&["8", "3"])));
        let three = "    d = 4\n    x = jnp.zeros((d, 3))\n    y = jnp.concatenate([x, x, x], axis=0)\n";
        assert_eq!(y_in_function(three), Some(shape(&["12", "3"])));
    }

    #[test]
    fn a_folded_literal_sum_is_not_a_claim_against_an_annotation() {
        let code = "def g(p: Float[Array, \"2*d 3\"]):\n    d = 4\n    x = jnp.zeros((d, 3))\n    z = jnp.concatenate([x, x], axis=0)\n    y = z + p\n";
        assert!(y_of(code).is_some());
    }

    #[test]
    fn a_literal_mark_next_to_a_free_name_conflicts_with_nothing() {
        let code = format!(
            "{SYMBOLS}def g(q):\n    d = 4\n    x = jnp.zeros((d, 3))\n    w = jnp.zeros((q, 3))\n    z = jnp.concatenate([x, w], axis=0)\n    y = z + jnp.zeros((9, 3))\n"
        );
        y_of(&code);
    }

    #[test]
    fn a_layer_formula_of_a_literal_mark_folds() {
        let conv = "k = 3\nl = nn.Conv2d(3, 8, k)\n";
        let call = "torch.zeros(2, 3, 8, 8)";
        let good = format!("{SYMBOLS}{conv}y = l({call})\n");
        assert_eq!(y_of(&good), Some(shape(&["2", "8", "6", "6"])));
        let bad = format!("{SYMBOLS}{conv}y = l({call}) + torch.zeros(2, 8, 7, 7)\n");
        assert_eq!(errors(&bad), 1);
        let stride = format!("{SYMBOLS}k = 3\nl = nn.Conv2d(3, 8, 3, stride=k)\ny = l({call})\n");
        assert_eq!(y_of(&stride), Some(shape(&["2", "8", "2", "2"])));
        let pad = format!("{SYMBOLS}k = 3\nl = nn.Conv2d(3, 8, 3, padding=k)\ny = l({call})\n");
        assert_eq!(y_of(&pad), Some(shape(&["2", "8", "12", "12"])));
        let pool = format!("{SYMBOLS}k = 3\nl = nn.MaxPool2d(k)\ny = l({call})\n");
        assert_eq!(y_of(&pool), Some(shape(&["2", "3", "2", "2"])));
        let chained = format!(
            "{SYMBOLS}k = 3\nl = nn.Conv2d(3, 8, k)\nm = nn.Conv2d(8, 4, k)\ny = m(l({call}))\n"
        );
        assert_eq!(y_of(&chained), Some(shape(&["2", "4", "4", "4"])));
        let dilated = format!("{SYMBOLS}k = 2\nl = nn.Conv2d(3, 8, 3, dilation=k)\ny = l({call})\n");
        assert_eq!(y_of(&dilated), Some(shape(&["2", "8", "4", "4"])));
        let sequence = format!(
            "{SYMBOLS}k = 3\nl = nn.Sequential(nn.Conv2d(3, 8, k), nn.ReLU())\ny = l({call})\n"
        );
        assert_eq!(y_of(&sequence), Some(shape(&["2", "8", "6", "6"])));
        let attribute = format!(
            "{SYMBOLS}k = 3\nclass M(nn.Module):\n    def __init__(self):\n        self.l = nn.Conv2d(3, 8, k)\n\n    def forward(self, x: Float[Array, \"2 3 8 8\"]):\n        y = self.l(x)\n"
        );
        assert_eq!(y_of(&attribute), Some(shape(&["2", "8", "6", "6"])));
        let wrong_channels = format!(
            "{SYMBOLS}k = 8\nl = nn.Conv2d(3, k, 1)\nm = nn.Conv2d(4, 8, 1)\ny = m(l({call}))\n"
        );
        assert_eq!(errors(&wrong_channels), 1);
    }

    #[test]
    fn a_rebound_slice_stop_is_not_the_symbol() {
        const W: &str = "    w = jnp.zeros((3, 4))\n";
        let annotated = |lines: &str| {
            format!("def g(w: Float[Array, \"n m\"], c):\n{lines}    y = w[:n]\n")
        };
        let unknown = Some(shape(&["?", "m"]));
        assert_eq!(y_of(&annotated("    n = 2\n")), Some(shape(&["min(2, n)", "m"])));
        for lines in [
            "    for n in range(3):\n        pass\n",
            "    n = c()\n",
            "    n -= 1\n",
            "    n: int\n",
            "    with c as n:\n        pass\n",
            "    import n\n",
        ] {
            assert_eq!(y_of(&annotated(lines)), unknown, "{lines}");
        }
        assert_eq!(y_of("def g(w: Float[Array, \"n m\"], n):\n    y = w[:n]\n"), unknown);
        assert_eq!(y_of("n = 2\ndef g(w: Float[Array, \"n m\"]):\n    y = w[:n]\n"), Some(shape(&["min(2, n)", "m"])));
        assert_eq!(y_of("n = load()\ndef g(w: Float[Array, \"n m\"]):\n    y = w[:n]\n"), unknown);
        let global = "def s():\n    global n\n    n = 2\n\ndef g(w: Float[Array, \"n m\"]):\n    y = w[:n]\n";
        assert_eq!(y_of(global), unknown);
        let inner = "def o(n):\n    def g(w: Float[Array, \"n m\"]):\n        y = w[:n]\n";
        assert_eq!(y_of(inner), unknown);
        let attribute = "class C:\n    def __init__(self, c):\n        self.n = c()\n\n    def g(self, w: Float[Array, \"n m\"]):\n        y = w[:self.n]\n";
        assert_eq!(y_of(attribute), unknown);
        assert_eq!(y_of(&format!("def g(n):\n{W}    y = w[:n]\n")), Some(shape(&["?", "4"])));
    }

    #[test]
    fn an_unbound_symbol_is_the_axis_of_its_own_slice() {
        let code = "def g(w: Float[Array, \"n m\"]):\n    y = w[:n]\n";
        assert_eq!(y_of(code), Some(shape(&["n", "m"])));
        let tail = "def g(w: Float[Array, \"n m\"]):\n    y = w[0:n, :m]\n";
        assert_eq!(y_of(tail), Some(shape(&["n", "m"])));
        let alias = "class C:\n    def __init__(self, n):\n        self.n = n\n\n    def g(self, w: Float[Array, \"n m\"]):\n        y = w[:self.n]\n";
        assert_eq!(y_of(alias), Some(shape(&["n", "m"])));
    }

    #[test]
    fn a_dilation_scales_the_kernel_of_a_conv() {
        let input = "torch.zeros(2, 3, 8, 8)";
        for (ctor, want) in [
            ("nn.Conv2d(3, 8, 3, padding=\"valid\", dilation=2)", Some(["2", "8", "4", "4"])),
            ("nn.Conv2d(3, 8, 3, padding=\"same\", dilation=2)", Some(["2", "8", "8", "8"])),
            ("nn.Conv2d(3, 8, 3, dilation=(2, 3))", Some(["2", "8", "4", "2"])),
            ("nn.Conv2d(3, 8, 3, dilation=[2, 2])", Some(["2", "8", "4", "4"])),
            ("nn.Conv2d(3, 8, 3, 1, 1, 2)", Some(["2", "8", "6", "6"])),
            ("nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)", Some(["2", "8", "3", "3"])),
            ("eqx.nn.Conv2d(3, 8, 3, padding=\"VALID\", dilation=2, key=key)", Some(["2", "8", "4", "4"])),
            ("eqx.nn.Conv2d(3, 8, 3, padding=0, dilation=(2, 3), key=key)", Some(["2", "8", "4", "2"])),
            ("nn.Conv2d(3, 8, 3, dilation=q)", None),
            ("nn.Conv2d(3, 8, 3, dilation=(q, 2))", None),
            ("nn.Conv2d(3, 8, 3, dilation=(2, 2, 2))", None),
            ("nn.Conv2d(3, 8, 3, dilation=0)", None),
        ] {
            assert_eq!(conv_y(ctor, input), want.map(|dims| shape(&dims)), "{ctor}");
        }
        let code = "def g(x: Float[Array, \"b 3 n\"]):\n    l = nn.Conv1d(3, 8, 3, dilation=2)\n    y = l(x)\n";
        assert_eq!(y_of(code), Some(shape(&["b", "8", "n-4"])));
        let kernel = "def g(x: Float[Array, \"3 8\"], q):\n    l = nn.Conv1d(3, 8, q, dilation=2)\n    y = l(x)\n";
        assert_eq!(y_of(kernel), None);
    }

    #[test]
    fn an_array_index_is_not_a_scalar() {
        let run = |lines: &str| y_of(&format!("def g(c, b: bool, f: float):\n    x = jnp.zeros((3, 4))\n{lines}"));
        for index in [
            "i",
            "x > 0",
            "c()",
            "i, 1:",
            ":, i",
            "(0, 1)",
            "True",
            "b",
            "f",
            "[[0, 1]]",
            "[True, False, True]",
            "[0, 1], 0",
            "0, [0, 1]",
            "[0, 1], [1, 2]",
            "[]",
            "0.5",
        ] {
            let lines = format!("    i = jnp.array([0, 1])\n    y = x[{index}]\n");
            assert_eq!(run(&lines), None, "{index}");
        }
        for (index, want) in [
            ("[0, 1]", vec!["2", "4"]),
            (":, [0, 1]", vec!["3", "2"]),
            ("[0, 1], 1:", vec!["2", "3"]),
            ("..., [0, 1, 2]", vec!["3", "3"]),
            ("None, [0, 1]", vec!["1", "2", "4"]),
            ("[-1]", vec!["1", "4"]),
        ] {
            assert_eq!(run(&format!("    y = x[{index}]\n")), Some(shape(&want)), "{index}");
        }
    }

    #[test]
    fn an_index_with_an_unknown_kind_is_unknown() {
        assert_eq!(y_of("def g(x: Float[Array, \"3 4\"], i):\n    y = x[i]\n"), None);
        assert_eq!(y_of("def g(x: Float[Array, \"3 4\"], c):\n    y = x[c.i]\n"), None);
        let loaded = "def g(x: Float[Array, \"3 4\"]):\n    i = load()\n    y = x[i]\n";
        assert_eq!(y_of(loaded), None);
        let mixed = "def g(x: Float[Array, \"3 4\"]):\n    i = 0\n    i = load()\n    y = x[i]\n";
        assert_eq!(y_of(mixed), None);
        let over_items = "def g(x: Float[Array, \"3 4\"], c):\n    for i in c:\n        y = x[i]\n";
        assert_eq!(y_of(over_items), None);
        let shadowed = "range = load\ndef g(x: Float[Array, \"3 4\"]):\n    for i in range(3):\n        y = x[i]\n";
        assert_eq!(y_of(shadowed), None);
        let global = "def s():\n    global i\n    i = load()\n\nfor i in range(3):\n    v = torch.zeros(3, 4)\n    y = v[i]\n";
        assert_eq!(y_of(global), None);
        let rebound = "def g(x: Float[Array, \"3 4\"]):\n    for i in range(3):\n        i = load()\n        y = x[i]\n";
        assert_eq!(y_of(rebound), None);
    }

    #[test]
    fn a_proven_integer_index_drops_the_axis() {
        let dropped = Some(shape(&["4"]));
        for body in [
            "def g(x: Float[Array, \"3 4\"]):\n    y = x[0]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    y = x[-1]\n",
            "def g(x: Float[Array, \"3 4\"], i: int):\n    y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"], i: int = 0):\n    y = x[i + 1]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    for i in range(3):\n        y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"], rows):\n    for i, r in enumerate(rows):\n        y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    i = 1\n    y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    i = x.shape[0] - 1\n    y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    i = len(x) // 2\n    y = x[i]\n",
            "def g(x: Float[Array, \"3 4\"]):\n    i = 0\n    i += 1\n    y = x[i]\n",
            "n = 1\ndef g(x: Float[Array, \"3 4\"]):\n    y = x[n]\n",
        ] {
            assert_eq!(y_of(body), dropped, "{body}");
        }
        let comprehension = "def g(x: Float[Array, \"3 4\"]):\n    y = jnp.stack([x[i] for i in range(2)])\n";
        assert_eq!(y_of(comprehension), Some(shape(&["2", "4"])));
    }

    #[test]
    fn a_negative_step_on_a_torch_tensor_is_unknown() {
        for body in [
            "v = torch.zeros(3, 4)\ny = v[::-1]\n",
            "v = torch.zeros(3, 4).float()\ny = v[::-2]\n",
            "def g(v: torch.Tensor):\n    w = v.view(3, 4)\n    y = w[::-1]\n",
            "from torch import Tensor\ndef g(v: Float[Tensor, \"3 4\"]):\n    y = v[::-1]\n",
            "def g(v: Float[torch.Tensor, \"3 4\"]):\n    y = (v + 1)[:, ::-1]\n",
        ] {
            assert_eq!(y_of(body), None, "{body}");
        }
        assert_eq!(y_of("s = -1\nv = torch.zeros(3, 4)\ny = v[::s]\n"), Some(shape(&["?", "4"])));
        assert_eq!(y_of("v = jnp.zeros((3, 4))\ny = v[::-1]\n"), Some(shape(&["3", "4"])));
        let array = "def g(v: Float[Array, \"3 4\"]):\n    y = v[::-1]\n";
        assert_eq!(y_of(array), Some(shape(&["3", "4"])));
        let converted = "v = torch.zeros(3, 4)\nw = v.numpy()\ny = w[0]\n";
        assert_eq!(y_of(converted), None);
        let positive = "v = torch.zeros(3, 4)\ny = v[::2]\n";
        assert_eq!(y_of(positive), Some(shape(&["2", "4"])));
    }

    #[test]
    fn a_torch_padding_string_is_lower_case() {
        let input = "torch.zeros(2, 3, 8, 8)";
        for padding in ["SAME", "Same", "VALID", "Valid", "same_lower"] {
            let ctor = format!("nn.Conv2d(3, 8, 3, padding=\"{padding}\")");
            assert_eq!(conv_y(&ctor, input), None, "{padding}");
        }
        let same = conv_y("nn.Conv2d(3, 8, 3, padding=\"same\")", input);
        assert_eq!(same, Some(shape(&["2", "8", "8", "8"])));
        let valid = conv_y("nn.Conv2d(3, 8, 3, padding=\"valid\")", input);
        assert_eq!(valid, Some(shape(&["2", "8", "6", "6"])));
        let upper = conv_y("eqx.nn.Conv2d(3, 8, 3, padding=\"Valid\", key=key)", input);
        assert_eq!(upper, Some(shape(&["2", "8", "6", "6"])));
    }

    #[test]
    fn a_kernel_larger_than_the_input_has_no_negative_size() {
        let input = "torch.zeros(2, 3, 8, 8)";
        for ctor in [
            "nn.Conv2d(3, 8, 20)",
            "nn.Conv2d(3, 8, 20, stride=30)",
            "nn.Conv2d(3, 8, 3, padding=\"valid\", dilation=9)",
            "nn.MaxPool2d(20)",
            "nn.MaxPool2d(20, stride=30)",
            "nn.ConvTranspose2d(3, 8, 1, padding=5)",
            "eqx.nn.Conv2d(3, 8, 20, key=key)",
            "nn.Unfold(20)",
        ] {
            assert_eq!(conv_y(ctor, input), None, "{ctor}");
        }
        let literal = format!("{SYMBOLS}k = 20\nl = nn.Conv2d(3, 8, k)\ny = l({input})\n");
        assert_eq!(y_of(&literal), None);
        let flax = |ctor: &str| y_of(&format!("l = {ctor}\ny = l(jnp.zeros((1, 7, 5, 2)))\n"));
        let valid = flax("fnn.Conv(4, (9, 9), padding=\"VALID\")");
        assert_eq!(valid, Some(shape(&["1", "0", "0", "4"])));
        let one = flax("fnn.Conv(4, (9, 9), padding=1)");
        assert_eq!(one, Some(shape(&["1", "1", "0", "4"])));
        let functional = |call: &str| {
            y_of(&format!(
                "import torch.nn.functional as F\nimport jax\nx = torch.zeros(2, 3, 8, 8)\nw = torch.zeros(8, 3, 20, 20)\nv = jnp.zeros((2, 3, 8, 8))\ny = {call}\n"
            ))
        };
        assert_eq!(functional("F.conv2d(x, w)"), None);
        assert_eq!(functional("F.max_pool2d(x, 20)"), None);
        assert_eq!(functional("F.avg_pool2d(x, 20, 30)"), None);
        let lax = functional("jax.lax.conv_general_dilated(v, jnp.zeros((8, 3, 20, 20)), (1, 1), \"VALID\")");
        assert_eq!(lax, None);
    }

    #[test]
    fn nonlocal_in_a_class_body_poisons_the_name() {
        let call = "    def f(x: Float[Array, \"n 3\"]) -> Float[Array, \"n 9\"]:\n        return x\n";
        let code = format!(
            "def h(x: Float[Array, \"n 3\"]):\n{call}    class C:\n        nonlocal f\n        f = load()\n    y = f(x)\n"
        );
        assert_eq!(y_of(&code), None);
        let value = "def h():\n    f = torch.zeros(3, 4)\n    class C:\n        nonlocal f\n        f = load()\n    y = f\n";
        assert_eq!(y_of(value), None);
        let nested = "def h():\n    f = torch.zeros(3, 4)\n    class C:\n        class D:\n            nonlocal f\n            f = load()\n    y = f\n";
        assert_eq!(y_of(nested), None);
        let outer = "def h():\n    f = torch.zeros(3, 4)\n    def i():\n        class C:\n            nonlocal f\n            f = load()\n    y = f\n";
        assert_eq!(y_of(outer), None);
        let read_only = "def h():\n    f = torch.zeros(3, 4)\n    class C:\n        nonlocal f\n        g = f\n    y = f\n";
        assert_eq!(y_of(read_only), Some(shape(&["3", "4"])));
        let other_scope = "def h():\n    f = torch.zeros(3, 4)\n    y = f\n\ndef i():\n    f = 1\n    class C:\n        nonlocal f\n        f = load()\n";
        assert_eq!(y_of(other_scope), Some(shape(&["3", "4"])));
    }

    #[test]
    fn a_parameter_padding_is_unknown_without_an_integer_proof() {
        let module = |signature: &str, layer: &str, call: &str| {
            format!(
                "class M(nn.Module):\n    def __init__(self, {signature}):\n        super().__init__()\n        self.c = {layer}\n\n    def forward(self, x):\n        return self.c(x)\n\nm = M({call})\ny = m.c(torch.zeros(2, 3, 8, 8))\nz = m(torch.zeros(2, 3, 8, 8))\n"
            )
        };
        let results = |code: &str| {
            let code = format!("{HEAD}{code}");
            let analysis = analyze(&code);
            assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
            (find_shape(&analysis, "y").cloned(), find_shape(&analysis, "z").cloned())
        };
        for layer in [
            "nn.Conv2d(3, 8, 3, padding=p)",
            "nn.MaxPool2d(2, padding=p)",
            "nn.Conv2d(3, 8, 3, 1, p)",
            "eqx.nn.Conv2d(3, 8, 3, padding=p, key=0)",
            "nn.ConvTranspose2d(3, 8, 3, padding=p)",
        ] {
            for call in ["\"same\"", "1"] {
                assert_eq!(results(&module("p", layer, call)), (None, None), "{layer} {call}");
            }
        }
        let direct = "def g(p, x: Float[Array, \"2 3 8 8\"]):\n    l = nn.Conv2d(3, 8, 3, padding=p)\n    y = l(x)\n";
        assert_eq!(y_of(direct), None);
        let inline = "def g(p, x: Float[Array, \"2 3 8 8\"]):\n    y = nn.Conv2d(3, 8, 3, padding=p)(x)\n";
        assert_eq!(y_of(inline), None);
        let sequence = "def g(p, x: Float[Array, \"2 3 8 8\"]):\n    l = nn.Sequential(nn.Conv2d(3, 8, 3, padding=p), nn.ReLU())\n    y = l(x)\n";
        assert_eq!(y_of(sequence), None);
        let flax = "def g(p, x: Float[Array, \"1 8 8 3\"]):\n    l = fnn.Conv(4, (3, 3), padding=p)\n    y = l(x)\n";
        assert_eq!(y_of(flax), None);

        let proven = "def g(p: int, x: Float[Array, \"2 3 8 8\"]):\n    l = nn.Conv2d(3, 8, 3, padding=p)\n    y = l(x)\n";
        assert_eq!(y_of(proven), Some(shape(&["2", "8", "8+2*p-3+1", "8+2*p-3+1"])));
        let half = "def g(q, x: Float[Array, \"2 3 8 8\"]):\n    l = nn.Conv2d(3, 8, q, padding=q // 2)\n    y = l(x)\n";
        assert!(y_of(half).is_some());
        let bound = "p = 1\nl = nn.Conv2d(3, 8, 3, padding=p)\ny = l(torch.zeros(2, 3, 8, 8))\n";
        assert_eq!(y_of(bound), Some(shape(&["2", "8", "8", "8"])));
    }

    #[test]
    fn a_flax_pool_follows_its_padding() {
        let pool = |call: &str| y_of(&format!("x = jnp.zeros((1, 7, 5, 2))\ny = {call}\n"));
        for function in ["max_pool", "avg_pool"] {
            for (arguments, want) in [
                ("(2, 2), (2, 2), \"SAME\"", Some(["1", "4", "3", "2"])),
                ("(3, 3), (2, 2), padding=\"SAME\"", Some(["1", "4", "3", "2"])),
                ("(3, 3), strides=(2, 2), padding=\"same\"", Some(["1", "4", "3", "2"])),
                ("(3, 3), padding=\"SAME\"", Some(["1", "7", "5", "2"])),
                ("(9, 9), (2, 2), \"SAME\"", Some(["1", "4", "3", "2"])),
                ("(2, 2), (2, 2)", Some(["1", "3", "2", "2"])),
                ("(2, 2), (2, 2), \"VALID\"", Some(["1", "3", "2", "2"])),
                ("(9, 9), (2, 2)", Some(["1", "0", "0", "2"])),
                ("(9, 9)", Some(["1", "0", "0", "2"])),
                ("(2, 2), (2, 2), ((1, 1), (0, 0))", None),
                ("(2, 2), (2, 2), padding=p", None),
                ("(2, 2), (2, 2), \"CIRCULAR\"", None),
            ] {
                let call = format!("fnn.{function}(x, {arguments})");
                assert_eq!(pool(&call), want.map(|dims| shape(&dims)), "{call}");
            }
        }
    }
}

#[cfg(test)]
mod symbolic_slice_tests {
    use super::*;
    use tree_sitter::Parser;

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    const HEAD: &str = "import torch\nimport torch.nn as nn\nimport jax.numpy as jnp\nimport einops\nfrom jaxtyping import Float, Array\n\n";

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn no_errors(body: &str) -> LayerShapeAnalysis {
        let code = format!("{HEAD}{body}");
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{code}\n{:?}", analysis.errors);
        let text = format!("{analysis:?}");
        assert!(!text.contains("__free") && !text.contains("__unproven"), "{text}");
        analysis
    }

    /// The shape of `y = <expression>` in a function with `x` of `[n, m]`, `b`
    /// of `[n, 4]`, and the integer `k` as a parameter.
    fn y(expression: &str) -> Option<Vec<String>> {
        let body = format!(
            "def f(x: Float[Array, \"n m\"], b: Float[Array, \"n 4\"], k):\n    y = {expression}\n"
        );
        find_shape(&no_errors(&body), "y").cloned()
    }

    fn y_is(expression: &str, want: &[&str]) {
        assert_eq!(y(expression), Some(shape(want)), "{expression}");
    }

    fn has_error(expression: &str) -> bool {
        let code = format!(
            "{HEAD}def f(x: Float[Array, \"n m\"]):\n    y = {expression}\n"
        );
        !analyze(&code).errors.is_empty()
    }

    /// The shape of `y` after `s = <slice>` and `y = <use of s>`.
    fn y_of_s(slice: &str, expression: &str) -> Option<Vec<String>> {
        y(&format!("0\n    s = {slice}\n    y = {expression}"))
    }

    #[test]
    fn a_slice_of_a_symbolic_axis_has_an_exact_size() {
        for (index, size) in [
            ("1:", "n-1"),
            (":-1", "n-1"),
            ("3:", "n-3"),
            (":-3", "n-3"),
            ("1:-2", "n-3"),
            (":", "n"),
            ("0:", "n"),
            (":n", "n"),
            ("2:n", "n-2"),
            (":4", "min(4, n)"),
            ("-4:", "min(4, n)"),
            ("0:4", "min(4, n)"),
            ("2:4", "min(2, n-2)"),
            ("-4:-1", "min(3, n-1)"),
            ("2:2", "0"),
            ("3:2", "0"),
            (":0", "0"),
            ("-1:-4", "0"),
            ("::2", "ceil(n/2)"),
            ("1::2", "ceil((n-1)/2)"),
            (":4:2", "ceil(min(4, n)/2)"),
            ("::-2", "ceil(n/2)"),
            ("::-1", "n"),
        ] {
            y_is(&format!("x[{index}]"), &[size, "m"]);
        }
    }

    #[test]
    fn a_slice_of_a_free_axis_has_the_same_form() {
        let code = "def f(a, b):\n    x = jnp.zeros((a, b))\n    y = x[:, 1:]\n    z = x[:, :4]\n    w = x[:2]\n";
        let analysis = no_errors(code);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["a", "b-1"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["a", "min(4, b)"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["min(2, a)", "b"])));
    }

    #[test]
    fn slices_add_up() {
        y_is("x[1:][1:]", &["n-2", "m"]);
        y_is("x[1:][:-1]", &["n-2", "m"]);
        y_is("x[:, 1:][:, :-2]", &["n", "m-3"]);
        y_is("x[1:][:4]", &["min(4, n-1)", "m"]);
    }

    #[test]
    fn an_unproven_bound_keeps_the_rank_and_the_other_axes() {
        for expression in ["x[:, k:]", "x[:, :k]", "x[:, ::k]", "x[:, k:k + 1]", "x[:, -4:4]", "x[:, 1::-1]", "x[:, :len(k)]"] {
            y_is(expression, &["n", "?"]);
        }
        y_is("x[k:]", &["?", "m"]);
        y_is("x[k:, k:]", &["?", "?"]);
        y_is("b[:, k:]", &["n", "?"]);
        y_is("x[:, k:][:, 1:]", &["n", "?"]);
        y_is("x[:, k:] + x[:, k:]", &["n", "?"]);
    }

    #[test]
    fn a_step_of_zero_stays_unknown() {
        assert_eq!(y("x[::0]"), None);
    }

    #[test]
    fn a_min_form_conflicts_with_nothing() {
        y_is("x[:, :4] + b", &["n", "4"]);
        y_is("b + x[:, :4]", &["n", "4"]);
        y_is("x[:, :4] + x", &["n", "m"]);
        y_is("x[:, :4] + x[:, :4]", &["n", "min(4, m)"]);
        y_is("x[:, :4] * jnp.zeros((1, 3))", &["n", "3"]);
        y_is("x[:, :4] @ jnp.zeros((3, 9))", &["n", "9"]);
        y_is("x[:, :4] @ x.T", &["n", "n"]);
        y_is("jnp.concatenate([x[:, :4], b], axis=0)", &["n+n", "4"]);
        y_is("jnp.stack([x[:, :4], b])", &["2", "n", "4"]);
        assert_eq!(y("x[:, :4] + x[:, :3]"), None);
        y_is("x[:, :4] + x[:, k:]", &["n", "min(4, m)"]);
        y_is("x[:, k:] + x[:, :4]", &["n", "min(4, m)"]);
    }

    #[test]
    fn a_difference_compares_by_its_name_and_its_integer() {
        y_is("x[1:] + x[:-1]", &["n-1", "m"]);
        y_is("x[1:-1] + x[2:]", &["n-2", "m"]);
        y_is("x[1:] @ x.T[:, 1:]", &["n-1", "n-1"]);
        y_is("x[:, 1:-1] + x[:, 2:]", &["n", "m-2"]);
        y_is("x[:, 1:] + jnp.zeros((1, 5))", &["n", "5"]);
        y_is("jnp.zeros((1, 5)) + x[:, 1:]", &["n", "5"]);
        y_is("x[:, 1:] @ jnp.zeros((5, 2))", &["n", "2"]);
        y_is("x[:, 1:] + b[:, :1]", &["n", "m-1"]);
        assert!(has_error("x[:, 1:] + x"));
        assert!(has_error("x + x[:, :-1]"));
        assert!(has_error("x[:, 1:] @ x.T"));
        assert!(has_error("x[:, 2:] + x[:, 1:]"));
        assert!(has_error("jnp.stack([x[1:], x])"));
    }

    #[test]
    fn a_difference_may_equal_another_name() {
        let free = "def f(a, x: Float[Array, \"n m\"]):\n    y = x[:, 1:] + jnp.zeros((1, a))\n    z = x[:, 1:] + x[:, :4]\n";
        let analysis = no_errors(free);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "m-1"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "m-1"])));
        let body = "def f(x: Float[Array, \"n m\"], w: Float[Array, \"s m\"]):\n    y = x[1:] + w\n    z = x[1:] @ w.T\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["s", "m"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n-1", "s"])));
    }

    #[test]
    fn a_difference_cancels_in_a_sum() {
        let body = "def f(x: Float[Array, \"n m\"], c: Float[Array, \"1 m\"], d: Float[Array, \"n 1\"]):\n    y = jnp.concatenate([x[1:], c]) + x\n    z = jnp.concatenate([x[:, 1:], d], axis=1) + x\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "m"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "m"])));
        y_is("jnp.concatenate([x[1:], x[-1:]]) + x", &["n", "m"]);
        y_is("jnp.concatenate([x[1:], x[:1]], axis=0)", &["n-1+min(1, n)", "m"]);
        y_is("jnp.stack([x[1:], x[:-1]])", &["2", "n-1", "m"]);
    }

    #[test]
    fn a_product_keeps_the_difference_together() {
        for (slice, expression, want) in [
            ("x[:, 1:]", "s.reshape(-1)", vec!["n*(m-1)"]),
            ("x[1:]", "s.reshape(-1)", vec!["(n-1)*m"]),
            ("x[:, 1:]", "s.reshape(s.shape[0], -1)", vec!["n", "m-1"]),
            ("x[:, :4]", "s.reshape(-1)", vec!["n*min(4, m)"]),
            ("x[:, :4]", "s.reshape(s.shape[0], -1)", vec!["n", "min(4, m)"]),
            ("x[:, k:]", "s.reshape(s.shape[0], -1)", vec!["n", "?"]),
            ("x[:, 1:]", "einops.rearrange(s, \"a b -> (a b)\")", vec!["n*(m-1)"]),
            ("x[:, 1:]", "s.reshape(-1) + jnp.zeros((6,))", vec!["6"]),
            ("x[:, ::2]", "s.reshape(-1)", vec!["n*ceil(m/2)"]),
        ] {
            assert_eq!(y_of_s(slice, expression), Some(shape(&want)), "{expression}");
        }
    }

    #[test]
    fn the_downstream_of_a_slice_has_shapes() {
        let body = "def f(x: Float[Array, \"b n d\"], w: Float[Array, \"d e\"]):\n    c = x[:, :4]\n    h = c.sum(axis=1)\n    w = w[:, :4]\n    v = x @ w\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "c"), Some(&shape(&["b", "min(4, n)", "d"])));
        assert_eq!(find_shape(&analysis, "h"), Some(&shape(&["b", "d"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["d", "min(4, e)"])));
        assert_eq!(find_shape(&analysis, "v"), Some(&shape(&["b", "n", "min(4, e)"])));
    }

    #[test]
    fn a_layer_takes_a_slice_dim_as_a_size() {
        let body = "def f(x: Float[Array, \"n m\"], k):\n    c = x[:, :4]\n    l = nn.Linear(c.shape[1], 8)\n    y = l(c)\n    u = x[:, k:]\n    p = nn.Linear(u.shape[1], 8)\n    z = p(u)\n    d = x[:, 1:]\n    q = nn.Linear(d.shape[1], 8)\n    r = q(d)\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "8"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "8"])));
        assert_eq!(find_shape(&analysis, "r"), Some(&shape(&["n", "8"])));
        let wide = "def f(x: Float[Array, \"n m\"]):\n    y = nn.Linear(4, 8)(x[:, :4])\n    z = nn.Linear(9, 8)(x[:, 1:])\n";
        no_errors(wide);
    }

    #[test]
    fn a_call_returns_a_slice_dim() {
        let body = "def g(a: Float[Array, \"p q\"]):\n    c = a[:, 1:]\n    return c\n\n\ndef h(a: Float[Array, \"p q\"]):\n    c = a[:, :4]\n    return c\n\n\ndef t(a: Float[Array, \"p q\"], k: int):\n    c = a[:, k:]\n    return c\n\n\ndef f(x: Float[Array, \"n m\"]):\n    y = g(x)\n    z = h(x)\n    w = t(x, 2)\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "m-1"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n", "min(4, m)"])));
        assert_eq!(find_shape(&analysis, "w"), Some(&shape(&["n", "?"])));
    }

    #[test]
    fn a_comma_in_a_size_stays_in_one_dim() {
        for (call, want) in [
            ("jnp.zeros((min(k, 2), 2))", vec!["min(k, 2)", "2"]),
            ("jnp.zeros((min(k,2), 2))", vec!["min(k,2)", "2"]),
            ("jnp.zeros((2, max(k, 2), 3))", vec!["2", "max(k, 2)", "3"]),
            ("jnp.zeros([min(k, 2), 2])", vec!["min(k, 2)", "2"]),
            ("jnp.ones((min(k, 2),))", vec!["min(k, 2)"]),
            ("torch.zeros(min(k, 2), 2)", vec!["min(k, 2)", "2"]),
            ("torch.zeros((min(k, 2), 2))", vec!["min(k, 2)", "2"]),
            ("torch.zeros(size=(min(k, 2), 2))", vec!["min(k, 2)", "2"]),
            ("x.reshape((min(k, 2), 6))", vec!["min(k, 2)", "6"]),
            ("x.reshape(min(k, 2), 6)", vec!["min(k, 2)", "6"]),
            ("jnp.broadcast_to(x, (min(k, 2), 3, 4))", vec!["min(k, 2)", "3", "4"]),
            ("jnp.full((min(k, 2), 2), 0.0)", vec!["min(k, 2)", "2"]),
            ("jnp.zeros((min(k, 2), 2)) + jnp.zeros((3, 2))", vec!["3", "2"]),
        ] {
            let body = format!("def f(k, x: Float[Array, \"3 4\"]):\n    y = {call}\n");
            assert_eq!(find_shape(&no_errors(&body), "y"), Some(&shape(&want)), "{call}");
        }
    }

    #[test]
    fn a_comma_in_a_literal_is_no_list_of_integers() {
        for call in ["jnp.pad(x, (min(1, 2), 3))", "jnp.pad(x, min(1, 2))"] {
            let body = format!("def f(x: Float[Array, \"3 4\"]):\n    y = {call}\n");
            assert_eq!(find_shape(&no_errors(&body), "y"), None, "{call}");
        }
    }

    #[test]
    fn a_comma_in_a_layer_size_stays_in_one_dim() {
        let body = "def f(k, x: Float[Array, \"2 3 8 8\"]):\n    l = nn.AdaptiveAvgPool2d((min(k, 2), 4))\n    y = l(x)\n    m = nn.Unflatten(1, (min(k, 2), 3))\n    z = m(jnp.zeros((2, 6)))\n    p = nn.Conv(4, (min(k, 2), 3))\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), None);
        assert_eq!(find_shape(&analysis, "z"), None);
    }

    #[test]
    fn min_of_a_shape_function_conflicts_with_nothing() {
        let body = "def f(x: Float[Array, \"n m\"]):\n    y = jnp.diag(x) + jnp.zeros((5,))\n    z = jnp.diag(x) + x[:, 0]\n";
        let analysis = no_errors(body);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["5"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["n"])));
    }

    #[test]
    fn a_second_write_stops_the_attribute_as_the_axis() {
        let program = |init: &str, stop: &str| {
            format!(
                "class C:\n    def __init__(self, n):\n{init}\n\n    def f(self, w: Float[Array, \"n m\"]):\n        y = w[:{stop}]\n"
            )
        };
        for init in [
            "        self.n = n\n        self.n = 1",
            "        self.n = 1\n        self.n = n",
            "        self.n = n\n        self.n += 1",
            "        self.n = n\n        self.n = load()",
        ] {
            let analysis = no_errors(&program(init, "self.n"));
            assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["?", "m"])), "{init}");
        }
        let one = no_errors(&program("        self.n = n", "self.n"));
        assert_eq!(find_shape(&one, "y"), Some(&shape(&["n", "m"])));
        let later = "class C:\n    def __init__(self, n):\n        self.n = n\n\n    def g(self):\n        self.n = 1\n\n    def f(self, w: Float[Array, \"n m\"]):\n        y = w[:self.n]\n";
        assert_eq!(find_shape(&no_errors(later), "y"), Some(&shape(&["?", "m"])));
        let sub = "class C:\n    def __init__(self, n):\n        self.n = n\n\n    def f(self, w: Float[Array, \"n m\"]):\n        y = w[:self.n]\n\n\nclass D(C):\n    def __init__(self, n):\n        super().__init__(n)\n        self.n = 1\n";
        assert_eq!(find_shape(&no_errors(sub), "y"), Some(&shape(&["?", "m"])));
    }

    /// `y = <expression>` with `x` of `[n, m]`, and with `a` and `b` of the
    /// annotation `dims`.
    fn annotated(dims: &str, expression: &str) -> Option<Vec<String>> {
        let body = format!(
            "def f(a: Float[Array, \"{dims}\"], b: Float[Array, \"{dims}\"]):\n    y = {expression}\n"
        );
        find_shape(&no_errors(&body), "y").cloned()
    }

    #[test]
    fn a_shape_argument_that_is_an_expression_has_no_rank() {
        for expression in [
            "jnp.zeros((2,) * 3)",
            "jnp.zeros([2, 3] * 2)",
            "jnp.zeros((2, 3) if k else (3, 2))",
            "jnp.zeros(tuple(max(1, d) for d in x.shape))",
            "jnp.zeros(tuple(d for d in (2, 3)))",
            "jnp.zeros((d for d in (2, 3)))",
            "jnp.zeros([d for d in (2, 3)])",
            "jnp.zeros(tuple((2, 3)))",
            "jnp.zeros(get(1))",
            "jnp.zeros(get(k, 2))",
            "jnp.zeros((get(1)))",
            "jnp.zeros(x.shape[1:])",
            "jnp.zeros((*x.shape, 3))",
            "jnp.zeros((2, 3)[0:1])",
            "jnp.zeros(shape=(2, 3) + (k,) * 2)",
            "jnp.zeros(())",
            "x.reshape(x.shape[::-1])",
            "x.reshape(get(1))",
            "jnp.broadcast_to(x, (2,) * 2)",
            "jnp.broadcast_to(x, get(k, 2))",
            "jnp.zeros((2, 3) + [4])",
            "jnp.zeros((k) + (4,))",
            "jnp.zeros(unknown.shape + (4,))",
        ] {
            assert_eq!(y(expression), None, "{expression}");
        }
    }

    #[test]
    fn a_sum_of_tuples_is_a_shape() {
        for (expression, want) in [
            ("jnp.zeros((2, 3) + (4,))", vec!["2", "3", "4"]),
            ("jnp.zeros([2, 3] + [4])", vec!["2", "3", "4"]),
            ("jnp.zeros(x.shape + (3,))", vec!["n", "m", "3"]),
            ("jnp.zeros((1,) + x.shape + (k,))", vec!["1", "n", "m", "k"]),
            ("x.reshape(x.shape + (1,))", vec!["n", "m", "1"]),
            ("x.reshape((1,) + x.shape)", vec!["1", "n", "m"]),
            ("jnp.broadcast_to(x, (2,) + x.shape)", vec!["2", "n", "m"]),
        ] {
            y_is(expression, &want);
        }
        let t = "def f(x: Float[Array, \"n m\"]):\n    t = jnp.zeros((2, 3) + (4,))\n    y = t.sum(axis=2)\n";
        assert_eq!(find_shape(&no_errors(t), "y"), Some(&shape(&["2", "3"])));
    }

    #[test]
    fn a_plain_size_argument_keeps_its_rank() {
        // `k` is a parameter without an annotation: it may hold a tuple.
        assert_eq!(y("jnp.zeros(k)"), None);
        y_is("jnp.zeros(5)", &["5"]);
        y_is("jnp.zeros((min(k, 2), 2))", &["min(k, 2)", "2"]);
        y_is("jnp.zeros((k, 2))", &["k", "2"]);
        y_is("jnp.zeros(x.shape[0])", &["x.shape[0]"]);
        y_is("x.reshape(-1)", &["n*m"]);
    }

    #[test]
    fn a_slice_of_an_axis_without_a_name_has_no_form() {
        for (dims, index, want) in [
            ("_ m", "1:", vec!["?", "m"]),
            ("_ m", "::2", vec!["?", "m"]),
            ("_ m", ":4", vec!["?", "m"]),
            ("_ m", ":", vec!["_", "m"]),
            ("_ m", ":, 1:", vec!["_", "m-1"]),
            ("_n m", "1:", vec!["?", "m"]),
            ("#n m", "1:", vec!["?", "m"]),
            ("#n m", ":2", vec!["?", "m"]),
            ("batch=4 m", "1:", vec!["?", "m"]),
            ("batch=4 m", ":4", vec!["?", "m"]),
            ("m _", ":, 1:", vec!["m", "?"]),
            ("m *b", "1:", vec!["m-1", "*b"]),
            ("m ...", "1:", vec!["m-1", "..."]),
            ("*b m", "..., 1:", vec!["*b", "m-1"]),
            ("... m", "..., :2", vec!["...", "min(2, m)"]),
            ("*b m", ":", vec!["*b", "m"]),
            ("*b m", "::-1", vec!["*b", "m"]),
            ("*b m", ":, :", vec!["*b", "m"]),
            ("*b m", "..., None", vec!["*b", "m", "1"]),
            ("*b m", "None", vec!["1", "*b", "m"]),
        ] {
            let got = annotated(dims, &format!("a[{index}]"));
            assert_eq!(got, Some(shape(&want)), "{dims} {index}");
        }
    }

    #[test]
    fn an_index_after_a_variadic_axis_has_no_position() {
        for dims in ["*b m", "... m", "*#b m", "#*b m", "*_ m", "k *b m"] {
            let skip = usize::from(dims.starts_with('k'));
            let prefix = ":, ".repeat(skip);
            for index in ["1:", ":, 1:", "0", ":, 0", "::2", ":, None", "1:, ..., 1:", "..., 1:, 1:", "[0, 1]"] {
                let got = annotated(dims, &format!("a[{prefix}{index}]"));
                assert_eq!(got, None, "{dims} {prefix}{index}");
            }
        }
    }

    #[test]
    fn a_sum_with_an_axis_without_a_name_is_no_error() {
        for (dims, expression) in [
            ("_ m", "a[1:] + b"),
            ("_ m", "a[2:] + b[1:]"),
            ("#n m", "a[1:] + b"),
            ("_n m", "a[1:] + b"),
            ("batch=4 m", "a[1:] + b"),
        ] {
            annotated(dims, expression);
        }
    }

    #[test]
    fn a_call_form_on_an_axis_without_a_name_is_unknown() {
        for dims in ["_ m", "#n m", "batch=4 m"] {
            let body = format!(
                "def g(p: Float[Array, \"q m\"]) -> Float[Array, \"q-1 m\"]:\n    return p[1:]\n\ndef f(a: Float[Array, \"{dims}\"]):\n    y = g(a)\n"
            );
            assert_eq!(find_shape(&no_errors(&body), "y"), Some(&shape(&["?", "m"])), "{dims}");
        }
    }

    #[test]
    fn a_slice_of_a_min_axis_folds_its_limit() {
        for (expression, want) in [
            ("x[:4][5:]", "0"),
            ("x[:4][4:]", "0"),
            ("x[:4][:-5]", "0"),
            ("x[:4][:-4]", "0"),
            ("x[:2][3:]", "0"),
            ("x[:4][2:-3]", "0"),
            ("x[:4][:7]", "min(4, n)"),
            ("x[:4][:2]", "min(2, n)"),
            ("x[:4][-7:]", "min(4, n)"),
            ("x[:4][-2:]", "min(2, n)"),
            ("x[:4][1:]", "min(3, n-1)"),
            ("x[:4][:-1]", "min(3, n-1)"),
            ("x[:4][1:2]", "min(1, n-1)"),
            ("x[:4][1:9]", "min(3, n-1)"),
            ("x[:4][-3:-1]", "min(2, n-1)"),
            ("x[:4][::2]", "ceil(min(4, n)/2)"),
            ("x[-4:][5:]", "0"),
            ("x[1:3][1:]", "min(1, n-2)"),
            ("x[:4][-1:1]", "?"),
            ("x[:4][::-2]", "ceil(min(4, n)/2)"),
            ("x[:4][1::-1]", "?"),
        ] {
            y_is(expression, &[want, "m"]);
        }
        y_is("x[:4][5:] + jnp.zeros((1, 1))", &["0", "m"]);
    }

    #[test]
    fn a_slice_of_a_ceil_axis_folds_or_is_unknown() {
        for (expression, want) in [
            ("x[::2][:3]", "min(3, ceil(n/2))"),
            ("x[::2][-3:]", "min(3, ceil(n/2))"),
            ("x[::2][::3]", "ceil(ceil(n/2)/3)"),
            ("x[::2][2:2]", "0"),
            ("x[::2][1:]", "?"),
            ("x[::2][:-1]", "?"),
            ("x[::2][1:3]", "?"),
            ("x[::2][-3:-1]", "?"),
        ] {
            y_is(expression, &[want, "m"]);
        }
    }

    #[test]
    fn a_form_on_a_min_axis_is_not_below_zero() {
        let body = "def g(p: Float[Array, \"q m\"]) -> Float[Array, \"q-5 m\"]:\n    return p\n\ndef f(x: Float[Array, \"n m\"]):\n    y = g(x[:4])\n";
        assert_eq!(find_shape(&no_errors(body), "y"), Some(&shape(&["?", "m"])));
    }

    #[test]
    fn a_substitution_next_to_a_product_keeps_its_value_together() {
        let run = |result: &str, sizes: &str| {
            let body = format!(
                "def g(a: Float[Array, \"q r\"]) -> Float[Array, \"{result}\"]:\n    return a\n\ndef f(b: Float[Array, \"{sizes}\"]):\n    y = g(b)\n"
            );
            find_shape(&no_errors(&body), "y").cloned()
        };
        for (result, sizes, want) in [
            ("q/r", "s t*u", "s/(t*u)"),
            ("q/r", "s t/u", "s/(t/u)"),
            ("q/r", "s t-u", "s/(t-u)"),
            ("q/r", "s*t u", "(s*t)/u"),
            ("q*r", "s t/u", "s*(t/u)"),
            ("q*r", "s t*u", "s*(t*u)"),
            ("q*r", "s t+u", "s*(t+u)"),
            ("q/r", "s 2", "s/2"),
            ("q/r", "s t", "s/t"),
            ("q+r", "s t*u", "s+t*u"),
            ("q-r", "s t+u", "s-(t+u)"),
        ] {
            assert_eq!(run(result, sizes), Some(shape(&[want])), "{result} {sizes}");
        }
    }
}

