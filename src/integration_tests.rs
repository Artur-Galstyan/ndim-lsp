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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "3"])));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_analyzes_chained_layer_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    y = l1(x)\n    z = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.layers.len(), 1);
        assert_eq!(analysis.applications.len(), 1);
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "5"])));
        assert!(!has_shape(&analysis, "a"));
    }

    #[test]
    fn test_missing_layer_implementation_keeps_only_annotation_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(analysis.errors[0].variable, "a");
        assert_eq!(analysis.errors[1].variable, "b");
        assert_eq!(find_shape(&analysis, "good"), Some(&shape(&["batch", "9"])));
    }

    #[test]
    fn test_analysis_failed_assignment_does_not_overwrite_existing_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(y: Float[Array, \"old shape\"], x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["old", "shape"])));
    }

    #[test]
    fn test_analysis_error_range_covers_failing_call_arguments() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
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

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
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

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
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

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
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

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
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
            resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5).unwrap();

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
    apply_err_case!(
        apply_case_sensitive_symbols,
        &["Batch", "Features"],
        linear("features", "out"),
        "got Features"
    );
    apply_err_case!(
        apply_whitespace_not_normalized_in_dims,
        &["features "],
        linear("features", "out"),
        "got features "
    );
    apply_err_case!(
        apply_expression_dim_exact_mismatch,
        &["hidden * 2"],
        linear("hidden*2", "out"),
        "got hidden * 2"
    );
    apply_err_case!(
        apply_empty_last_dim_mismatch,
        &[""],
        linear("features", "out"),
        "got "
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
    apply_err_case!(
        apply_comma_dimension_mismatch,
        &["batch", "features"],
        linear("features,", "out"),
        "got features"
    );
    apply_err_case!(
        apply_question_mark_dimension_mismatch,
        &["batch", "features"],
        linear("features?", "out"),
        "got features"
    );
    apply_err_case!(
        apply_ellipsis_dimension_mismatch,
        &["batch", "features"],
        linear("...", "out"),
        "got features"
    );
    apply_err_case!(
        apply_colon_dimension_mismatch,
        &["batch", "time"],
        linear("time:2", "out"),
        "got time"
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["24"])));
    }

    #[test]
    fn test_propagates_method_sum_axis_kwarg() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = x.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_propagates_method_reshape_multi_positional() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"6 4\"]):\n    y = x.reshape(3, 8)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["3", "8"])));
    }

    #[test]
    fn test_chained_method_calls_in_source_order() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"2 3 4\"]):\n    y = x.flatten()\n    z = y.sum(axis=0)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_free_call_unknown_module_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"batch features\"]):\n    y = helpers.transform(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_method_call_error_range_covers_args() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "def f(x: Float[Array, \"6 4\"]):\n    y = x.reshape(3, 9)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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
        }]
    }

    #[test]
    fn test_single_torch_nn_linear_flow() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = torch.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_np_ones_like_preserves_symbolic_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.ones_like(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_torch_empty_like_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.empty_like(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_torch_chained_zeros_like_ones_like() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.zeros_like(x)\n    z = torch.ones_like(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch", "features"])));
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
        let code =
            "def f(a: Float[Array, \"batch k\"], b: Float[Array, \"k n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "n"])));
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"batch 3\"], b: Float[Array, \"5 n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("matmul dimension mismatch"));
    }

    #[test]
    fn test_matmul_batch_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"2 3 4\"], b: Float[Array, \"5 4 6\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("matmul batch dimension mismatch"));
    }

    #[test]
    fn test_matmul_matching_batch_dims() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"b m k\"], b: Float[Array, \"b k n\"]):\n    y = a @ b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "m", "n"])));
    }

    #[test]
    fn test_elementwise_add_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"batch features\"], b: Float[Array, \"batch features\"]):\n    y = a + b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_elementwise_mul_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"a b\"], b: Float[Array, \"a c\"]):\n    y = a * b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("elementwise *"));
        assert!(analysis.errors[0].message.contains("a, b"));
        assert!(analysis.errors[0].message.contains("a, c"));
    }

    #[test]
    fn test_elementwise_sub_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"b d\"], b: Float[Array, \"b d\"]):\n    y = a - b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_elementwise_sub_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"x y\"], b: Float[Array, \"x z\"]):\n    y = a - b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("elementwise -"));
    }

    #[test]
    fn test_elementwise_div_success() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"b d\"], b: Float[Array, \"b d\"]):\n    y = a / b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_elementwise_div_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(a: Float[Array, \"p q\"], b: Float[Array, \"p r\"]):\n    y = a / b";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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
        fs::write(
            tmp.path().join("equinox/__init__.py"),
            "from . import nn",
        )
        .unwrap();
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30"])));
    }

    // ── Torch Conv2d same ──

    #[test]
    fn test_torch_conv2d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30"])));
    }

    // ── in_channels mismatch -> ShapeError ──

    #[test]
    fn test_equinox_conv2d_channels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 4 32 32\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("expected 3 input channels"));
        assert!(analysis.errors[0].message.contains("got 4"));
    }

    #[test]
    fn test_torch_conv2d_channels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 5 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert!(analysis.errors[0].message.contains("expected 3 input channels"));
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        // L_out = floor((64 + 2*0 - 5)/1) + 1 = 60
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "60"])));
    }

    #[test]
    fn test_torch_conv1d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 64\"]):\n    layer = torch.nn.Conv1d(3, 16, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "60"])));
    }

    // ── Conv3d ──

    #[test]
    fn test_equinox_conv3d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32 32\"]):\n    layer = eqx.nn.Conv3d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30", "30"])));
    }

    #[test]
    fn test_torch_conv3d_concrete_dims() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32 32\"]):\n    layer = torch.nn.Conv3d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30", "30"])));
    }

    // ── Conv2d with stride=2, padding=1 ──

    #[test]
    fn test_equinox_conv2d_stride2_padding1() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = eqx.nn.Conv2d(3, 16, 3, stride=2, padding=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        // H_out = floor((32 + 2*1 - 3)/2) + 1 = floor(31/2) + 1 = 15 + 1 = 16
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "16", "16"])));
    }

    #[test]
    fn test_torch_conv2d_stride2_padding1() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "16", "16"])));
    }

    // ── Symbolic input: in_channels matches as symbol ──

    #[test]
    fn test_equinox_conv2d_symbolic_in_channels() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch in_c H W\"]):\n    layer = eqx.nn.Conv2d(in_c, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30"])));
    }

    #[test]
    fn test_equinox_conv2d_keyword_constructor() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = eqx.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30"])));
    }

    // ── from-import alias form ──

    #[test]
    fn test_from_torch_nn_import_conv2d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "from torch.nn import Conv2d\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    layer = Conv2d(3, 16, 3)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "30", "30"])));
    }

    // ── Direct apply_layer_application unit tests ──

    fn conv1d(in_channels: &str, out_channels: &str, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
        LayerKind::Conv1d {
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
        }
    }

    fn conv2d(in_channels: &str, out_channels: &str, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
        LayerKind::Conv2d {
            in_channels: in_channels.to_string(),
            out_channels: out_channels.to_string(),
            kernel_size: kernel_size.to_string(),
            stride: stride.to_string(),
            padding: padding.to_string(),
        }
    }

    fn conv3d(in_channels: &str, out_channels: &str, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
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
    fn test_conv2d_symbolic_channels_mismatch_error() {
        let app = layer_app("x", conv2d("in_c", "16", "3", "1", "0"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "other", "H", "W"]))]);
        let err = apply_layer_application(&app, &shapes).unwrap_err();
        assert!(err.contains("expected in_c input channels"));
        assert!(err.contains("got other"));
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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
        fs::write(
            tmp.path().join("equinox/__init__.py"),
            "from . import nn",
        )
        .unwrap();
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "x"), Some(&shape(&["batch", "features"])));
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.BatchNorm2d on (batch, 16, H, W) ──

    #[test]
    fn test_torch_batchnorm2d_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "H", "W"])));
    }

    // ── torch.nn.LayerNorm on (batch, 16) ──

    #[test]
    fn test_torch_layernorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16\"]):\n    ln = torch.nn.LayerNorm([16])\n    y = ln(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.ReLU ──

    #[test]
    fn test_torch_relu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3 32 32\"]):\n    relu = torch.nn.ReLU()\n    y = relu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "3", "32", "32"])));
    }

    // ── Chained: batchnorm then relu ──

    #[test]
    fn test_chained_batchnorm_relu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    bn = torch.nn.BatchNorm2d(16)\n    relu = torch.nn.ReLU()\n    y = bn(x)\n    z = relu(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "H", "W"])));
        assert_eq!(find_shape(&analysis, "z"), Some(&shape(&["batch", "16", "H", "W"])));
    }

    // ── Symbolic input shape ──

    #[test]
    fn test_symbolic_input_shape_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"a b c\"]):\n    drop = torch.nn.Dropout(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── equinox.nn.GroupNorm ──

    #[test]
    fn test_equinox_groupnorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch channels H W\"]):\n    gn = eqx.nn.GroupNorm(4, channels)\n    y = gn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "channels", "H", "W"])));
    }

    // ── equinox.nn.PReLU ──

    #[test]
    fn test_equinox_prelu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_equinox_nn(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    prelu = eqx.nn.PReLU()\n    y = prelu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.GELU ──

    #[test]
    fn test_torch_gelu_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    gelu = torch.nn.GELU()\n    y = gelu(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.Sigmoid ──

    #[test]
    fn test_torch_sigmoid_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    sig = torch.nn.Sigmoid()\n    y = sig(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.Tanh ──

    #[test]
    fn test_torch_tanh_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    tanh = torch.nn.Tanh()\n    y = tanh(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.Softmax ──

    #[test]
    fn test_torch_softmax_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    sm = torch.nn.Softmax(dim=1)\n    y = sm(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    // ── torch.nn.GroupNorm ──

    #[test]
    fn test_torch_groupnorm_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 16 H W\"]):\n    gn = torch.nn.GroupNorm(4, 16)\n    y = gn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "16", "H", "W"])));
    }

    // ── Chain with Linear: shape-preserving layer after Linear ──

    #[test]
    fn test_shape_preserving_after_linear() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        let code = "import torch\ndef f(x: Float[Array, \"batch 3\"]):\n    linear = torch.nn.Linear(3, 5)\n    relu = torch.nn.ReLU()\n    y = linear(x)\n    z = relu(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["8", "D", "H", "W"])));
    }

    #[test]
    fn test_dropout2d_unbatched_accepts_3d() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // Dropout2d on (C, H, W) — exactly at min rank 3
        let code = "import torch\ndef f(x: Float[Array, \"16 H W\"]):\n    drop = torch.nn.Dropout2d(0.5)\n    y = drop(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["8", "D", "H", "W"])));
    }

    #[test]
    fn test_batchnorm1d_accepts_2d_input() {
        let tmp = tempfile::tempdir().unwrap();
        write_fake_torch_nn(&tmp);
        // BatchNorm1d on (C, L) — exactly at min rank 2
        let code = "import torch\ndef f(x: Float[Array, \"16 L\"]):\n    bn = torch.nn.BatchNorm1d(16)\n    y = bn(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["h+3", "w+7"])));
    }

    #[test]
    fn test_f_pad_preserves_symbolic_with_addition() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"height width\"]):\n    y = F.pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["height+3", "width+3"])));
    }

    #[test]
    fn test_f_pad_dynamic_pad_variable_returns_none() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"h w\"]):\n    y = F.pad(x, pad_width)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        // Dynamic pad width variable cannot be statically parsed; should return no shape, not error
        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_f_pad_invalid_pad_does_not_crash() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"h w\"]):\n    y = F.pad(x, \"invalid\")";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        // Invalid/unparseable pad does not crash; returns no shape
        assert!(analysis.errors.is_empty());
        assert!(!has_shape(&analysis, "y"));
    }

    #[test]
    fn test_from_import_pad() {
        let code = "from torch.nn.functional import pad\ndef f(x: Float[Array, \"n\"]):\n    y = pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n+3"])));
    }

    #[test]
    fn test_from_import_pad_alias() {
        let code = "from torch.nn.functional import pad as F_pad\ndef f(x: Float[Array, \"n\"]):\n    y = F_pad(x, (1, 2))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n+3"])));
    }

    #[test]
    fn test_f_pad_per_axis_numeric() {
        let code = "import torch.nn.functional as F\ndef f(x: Float[Array, \"10 20\"]):\n    y = F.pad(x, ((1, 2), (3, 4)))";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_np_argmax_axis_0_gives_features() {
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.argmax(x, axis=0)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_torch_argsort_preserves_batch_features() {
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.argsort(x)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_torch_cumsum_dim_1_preserves_batch_features() {
        let code = "import torch\ndef f(x: Float[Array, \"batch features\"]):\n    y = torch.cumsum(x, dim=1)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_chained_np_sort_then_any_propagates() {
        let code = "import numpy as np\ndef f(x: Float[Array, \"batch features\"]):\n    y = np.sort(x)\n    z = np.any(y, axis=-1)";
        let tree = parse(code);
        let roots = vec![];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn test_torch_linalg_inv_batched_square_preserves_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f(x: Float[Array, \"b n n\"]):\n    y = torch.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["b", "n", "n"]))
        );
    }

    #[test]
    fn test_linalg_inv_non_square_reports_error_no_output_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f(x: Float[Array, \"m n\"]):\n    y = np.linalg.inv(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(analysis.errors[0].message.contains("last two dimensions to match"));
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

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_np_identity_square() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f():\n    y = np.identity(n)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n", "n"])));
    }

    #[test]
    fn test_jnp_linspace_keyword_num() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import jax.numpy as jnp\ndef f():\n    y = jnp.linspace(0, 1, num=steps)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["steps"])));
    }

    #[test]
    fn test_torch_linspace_keyword_steps() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import torch\ndef f():\n    y = torch.linspace(0, 1, steps=steps)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["steps"])));
    }

    #[test]
    fn test_np_logspace_keyword_num() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import numpy as np\ndef f():\n    y = np.logspace(0, 3, num=n)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["n"])));
    }
}
