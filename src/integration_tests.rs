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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap();

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
mod end_to_end_layer_shape_tests {
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

    fn scopes_from(shapes: HashMap<String, Vec<String>>) -> Vec<FunctionShapeScope> {
        vec![FunctionShapeScope {
            function_name: None,
            start_byte: 0,
            end_byte: usize::MAX,
            shapes,
            return_shape: None,
            param_order: Vec::new(),
        }]
    }

    #[test]
    fn test_extracts_layers_and_propagates_single_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_extracts_layers_and_propagates_chain() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nl1 = eqx.nn.Linear(3, 5)\nl2 = eqx.nn.Linear(5, 7)\ny = l1(x)\nz = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(scopes[0].shapes.get("z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_reports_mismatch_in_end_to_end_pipeline() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "4"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "y");
        assert!(errors[0].message.contains("expected input last dim 3"));
        assert!(!scopes[0].shapes.contains_key("y"));
    }

    #[test]
    fn test_unknown_input_in_end_to_end_pipeline_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::new());

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert!(!scopes[0].shapes.contains_key("y"));
    }

    #[test]
    fn test_in_place_layer_application_uses_old_input_shape_then_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\nx = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("x"), Some(&shape(&["batch", "5"])));
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

    #[test]
    fn test_resolves_call_signature_through_reexport() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(3, out_features=5)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None)
            .unwrap()
            .unwrap();

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
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None)
            .unwrap()
            .unwrap();

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
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None)
            .unwrap()
            .unwrap();

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
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None)
            .unwrap()
            .unwrap();

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

        let source = "import jax.numpy as jnp\nout = jnp.concatenate(xs, axis=1)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None)
            .unwrap()
            .unwrap();

        assert_eq!(found.signature.owner, None);
        assert_eq!(found.signature.name, "concatenate");
        assert_eq!(found.bindings.get("arrays"), Some(&"xs".to_string()));
        assert_eq!(found.bindings.get("axis"), Some(&"1".to_string()));
    }

    #[test]
    fn test_returns_none_when_implementation_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let source = "import missing\nx = missing.Foo()";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found =
            resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5, None).unwrap();

        assert_eq!(found, None);
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
    fn test_module_file_wins_over_package_init_for_same_prefix() {
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
                tmp.path().join("foo/bar.py"),
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
                assert_eq!(bindings, expected);
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
        ann_numpy_ndarray_rejected_for_now,
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
        ann_shape_with_comma_kept_as_token,
        "def f(x: Float[Array, \"b, d\"]): pass",
        [("x", ["b,", "d"])]
    );
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

    fn scopes_from(shapes: HashMap<String, Vec<String>>) -> Vec<FunctionShapeScope> {
        vec![FunctionShapeScope {
            function_name: None,
            start_byte: 0,
            end_byte: usize::MAX,
            shapes,
            return_shape: None,
            param_order: Vec::new(),
        }]
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
    fn test_torch_nn_linear_mismatch_reports_shape_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\nlayer = torch.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "4"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "y");
        assert!(errors[0].message.contains("expected input last dim 3"));
        assert!(!scopes[0].shapes.contains_key("y"));
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
        // range should cover the `x @ y` expression
        assert_eq!(&code[analysis.errors[0].range.start_byte..analysis.errors[0].range.end_byte], "x @ y");
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
        LayerKind::Conv1d {
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
        }
    }

    fn conv2d(
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
    ) -> LayerKind {
        LayerKind::Conv2d {
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
        }
    }

    fn conv3d(
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
    ) -> LayerKind {
        LayerKind::Conv3d {
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
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
        assert_eq!(
            find_shape_in_scope(&analysis, "f", "y"),
            Some(&shape(&["seq", "3"]))
        );
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
        assert!(
            pct(shaped, total) >= 75.0,
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
        let code = "import jax\ndef f(h0: Float[Array, \"hidden\"], xs: Float[Array, \"seq features\"]):\n    def body(c, x):\n        return c, c\n    h_final, hs = jax.lax.scan(body, h0, xs)";
        let analysis = analyze(code);

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "h_final"), Some(&shape(&["hidden"])));
        assert_eq!(find_shape(&analysis, "hs"), None);
    }

    #[test]
    fn test_scan_init_keyword() {
        let code = "import jax\ndef f(h0: Float[Array, \"hidden\"], xs: Float[Array, \"seq features\"]):\n    def body(c, x):\n        return c, c\n    h_final, hs = jax.lax.scan(body, init=h0, xs=xs)";
        let analysis = analyze(code);

        assert_eq!(find_shape(&analysis, "h_final"), Some(&shape(&["hidden"])));
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
        let code = "from einops import rearrange\ndef f(x: Float[Array, \"b s d\"]):\n    y = rearrange(x, \"... d -> d ...\")";
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
}

/// Timing harness over `bench/benchmark_large.py` (regenerate with
/// `python3 bench/generate_benchmark.py`). Not part of the default suite.
/// Run with: `cargo test --release bench_large_file -- --ignored --nocapture`.
#[cfg(test)]
mod bench_harness {
    use super::*;
    use std::fs;
    use std::time::Instant;
    use tree_sitter::Parser;

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
        let deliberate = code.matches("# expected-error:").count();
        assert_eq!(
            warm.errors.len(),
            deliberate,
            "diagnostics should match the deliberate '# expected-error:' markers exactly \
             (fewer = missed detection, more = false positives)"
        );
        // Keystroke budget: parse + analyze should stay well under one frame-ish
        // interval even in debug. Release should be far below this.
        assert!(
            med(&analyze_times) < 1000.0,
            "analyze median {:.1} ms exceeds 1s regression tripwire",
            med(&analyze_times)
        );
    }

    fn run_with(
        parser: &mut Parser,
        code: &str,
        roots: &[PathBuf],
        cache: &ResolutionCache,
    ) -> (f64, f64, LayerShapeAnalysis) {
        let read = |_: &PathBuf| -> Option<String> { None };
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
