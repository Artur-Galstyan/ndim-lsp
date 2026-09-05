use std::collections::BTreeMap;

use serde::Deserialize;
use tree_sitter::Parser;

use crate::*;

#[derive(Deserialize)]
struct Fixtures {
    imports: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    inputs: BTreeMap<String, Vec<usize>>,
    bind: String,
    call: String,
    #[serde(default)]
    shapes: BTreeMap<String, Option<Vec<usize>>>,
    #[serde(default)]
    error: bool,
}

fn analyze(code: &str) -> LayerShapeAnalysis {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(code, None).unwrap();
    analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
}

fn find_shape<'a>(analysis: &'a LayerShapeAnalysis, name: &str) -> Option<&'a Vec<String>> {
    analysis
        .scopes
        .iter()
        .find_map(|scope| scope.shapes.get(name))
}

#[test]
fn jax_0_11_1_runtime_shape_fixtures() {
    // The Python oracle executes these same calls with these same input shapes.
    // Rust CI needs no Python/JAX installation or network access.
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("../bench/jax_compat/cases.json")).unwrap();
    for case in fixtures.cases {
        let parameters = case
            .inputs
            .iter()
            .map(|(name, dims)| {
                if dims.is_empty() {
                    return format!("{name}: int");
                }
                let dims = dims
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{name}: Float[Array, \"{dims}\"]")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let code = format!(
            "{}\ndef f({parameters}):\n    {} = {}\n",
            fixtures.imports, case.bind, case.call,
        );
        let analysis = analyze(&code);
        assert_eq!(
            !analysis.errors.is_empty(),
            case.error,
            "{}\n{code}\n{:?}",
            case.name,
            analysis.errors
        );
        for (name, dims) in case.shapes {
            let expected = dims.map(|dims| dims.iter().map(usize::to_string).collect::<Vec<_>>());
            assert_eq!(
                find_shape(&analysis, &name),
                expected.as_ref(),
                "{}: {name}\n{code}",
                case.name
            );
            let hint = analysis
                .assignment_shapes
                .iter()
                .find(|record| record.name == name);
            assert_eq!(
                hint.map(|record| &record.shape),
                expected.as_ref(),
                "{}: inlay hint for {name}",
                case.name
            );
        }
        if case.error {
            for name in case.bind.split(',').map(str::trim) {
                assert!(
                    find_shape(&analysis, name).is_none(),
                    "{}: stale shape for {name}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn jax_new_rules_preserve_symbolic_dimensions() {
    let code = r#"
import jax.numpy as jnp
import jax.scipy.linalg as la
import jax.scipy.special as sp
def f(x: Float[Array, "batch n"], y: Float[Array, "1 n"], n: int, k: int):
    values, indices = jnp.top_k(x, k)
    matrix = la.circulant(x)
    polynomial = la.companion(x)
    contrast = la.helmert(n)
    sized = la.dft(x.shape[-1])
    transformed = sp.boxcox(x, y)
"#;
    let analysis = analyze(code);
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    for (name, dims) in [
        ("values", vec!["batch", "k"]),
        ("indices", vec!["batch", "k"]),
        ("matrix", vec!["batch", "n", "n"]),
        ("polynomial", vec!["batch", "n-1", "n-1"]),
        ("contrast", vec!["n-1", "n"]),
        ("sized", vec!["n", "n"]),
        ("transformed", vec!["batch", "n"]),
    ] {
        let expected = dims.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(find_shape(&analysis, name), Some(&expected), "{name}");
    }
}

#[test]
fn jax_dynamic_options_and_unknown_operands_stay_unknown() {
    let imports = "import jax.numpy as jnp\nfrom jax import lax\nimport jax.scipy.linalg as la\nimport jax.scipy.special as sp";
    for (bind, call) in [
        ("out", "la.helmert(4, full=flag)"),
        ("out", "la.convolution_matrix(x, 4, mode=mode)"),
        ("out, other", "la.qr_multiply(a, c, pivoting=flag)"),
        ("out, other", "jnp.top_k(a, 2, axis=axis)"),
        ("out", "sp.boxcox(x, unknown)"),
        ("out", "lax.broadcast_like(x, unknown)"),
        ("out", "lax.stack([x, x], axis=axis)"),
        ("out, other", "lax.unstack(a, axis=axis)"),
        ("out", "la.dft(unknown_call())"),
    ] {
        // Pre-existing shapes must be evicted, not reused when inference fails.
        let code = format!(
            "{imports}\ndef f(x: Float[Array, \"4\"], a: Float[Array, \"4 2\"], c: Float[Array, \"4\"], axis, flag, mode, unknown):\n    out = x\n    other = x\n    {bind} = {call}\n"
        );
        let analysis = analyze(&code);
        assert!(analysis.errors.is_empty(), "{call}: {:?}", analysis.errors);
        for name in bind.split(',').map(str::trim) {
            assert!(find_shape(&analysis, name).is_none(), "{call}: {name}");
        }
    }
}

#[test]
fn stack_sequences_stay_distinct_across_backends() {
    for module in ["jax.lax", "jax.numpy", "numpy", "torch"] {
        let axis = if module == "torch" { "dim" } else { "axis" };
        for sequence in ["(x, x)", "[x + x, x]", "(x.reshape(3, 4), x)"] {
            let code = format!(
                "import {module} as lib\ndef f(x: Float[Array, \"3 4\"]):\n    out = lib.stack({sequence}, {axis}=-1)\n"
            );
            let analysis = analyze(&code);
            let expected = vec!["3".to_string(), "4".to_string(), "2".to_string()];
            assert!(analysis.errors.is_empty(), "{code}: {:?}", analysis.errors);
            assert_eq!(find_shape(&analysis, "out"), Some(&expected), "{code}");
        }
    }
}

#[test]
fn jax_new_names_do_not_match_unrelated_modules() {
    for module in [
        "numpy",
        "scipy.special",
        "scipy.linalg",
        "custom",
        "jax.custom",
    ] {
        for name in [
            "top_k",
            "boxcox",
            "circulant",
            "qr_multiply",
            "broadcast_like",
        ] {
            let target = ResolvedTarget {
                dots: 0,
                parts: module
                    .split('.')
                    .chain([name])
                    .map(str::to_string)
                    .collect(),
            };
            assert_eq!(classify_known_function(&target), None, "{module}.{name}");
        }
    }
}
