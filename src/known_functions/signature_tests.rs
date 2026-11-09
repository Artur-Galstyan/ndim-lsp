use crate::*;
use tree_sitter::Parser;

const TORCH: &str = "import torch\n";
const NUMPY: &str = "import numpy as np\n";
const JAX: &str = "import jax\nimport jax.numpy as jnp\n";
const NONE: &str = "";

/// The shape of `y = expr` and the count of errors, in a file with `imports`
/// and a function with `params`.
fn run(imports: &str, params: &str, expr: &str) -> (Option<Vec<String>>, usize) {
    let code = format!("{imports}def f({params}):\n    y = {expr}\n");
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
    let tree = parser.parse(&code, None).unwrap();
    let analysis = analyze_layer_shapes(tree.root_node(), &code, &[], |_| None, 5, None).unwrap();
    let shape = analysis.scopes.iter().find_map(|scope| scope.shapes.get("y")).cloned();
    (shape, analysis.errors.len())
}

type Case = (&'static str, &'static str, &'static str, Option<&'static [&'static str]>, usize);

/// Each case is `(imports, params, expr, shape, errors)`. Reports every failing case.
fn check(cases: &[Case]) {
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(imports, params, expr, shape, errors)| {
            let expected = (shape.map(|dims| dims.iter().map(|d| d.to_string()).collect()), *errors);
            let got = run(imports, params, expr);
            (got != expected).then(|| format!("{imports:?} {expr}: expected {expected:?}, got {got:?}"))
        })
        .collect();
    assert!(failures.is_empty(), "failed:\n{}", failures.join("\n"));
}

const AB: &str = "x: Float[Array, \"a b\"]";
const A3: &str = "x: Float[Array, \"a 3\"]";

#[test]
#[rustfmt::skip]
fn quantile_and_norm_read_their_own_signature() {
    check(&[
        (NUMPY, AB, "np.percentile(x, 50)", Some(&[]), 0),
        (NUMPY, AB, "np.percentile(x, 50, 1)", Some(&["a"]), 0),
        (NUMPY, AB, "np.quantile(x, 0.5, axis=0, keepdims=True)", Some(&["1", "b"]), 0),
        (NUMPY, "x: Float[Array, \"a b\"], q: Float[Array, \"k\"]", "np.quantile(x, q, 1)", Some(&["k", "a"]), 0),
        (NUMPY, "x: Float[Array, \"a b\"], q: float", "np.quantile(x, q, 1)", Some(&["a"]), 0),
        (NUMPY, "x: Float[Array, \"a b\"], q", "np.quantile(x, q, 1)", None, 0),
        (JAX, AB, "jnp.median(x, 1)", Some(&["a"]), 0),
        (JAX, AB, "jnp.quantile(x, 0.5, 1, None, False, 'linear', True)", Some(&["a", "1"]), 0),
        (TORCH, AB, "torch.quantile(x, 0.5, 1, True)", Some(&["a", "1"]), 0),
        (NUMPY, AB, "np.linalg.norm(x, 1)", Some(&[]), 0),
        (NUMPY, AB, "np.linalg.norm(x, 2, 1)", Some(&["a"]), 0),
        (JAX, AB, "jnp.linalg.norm(x, 2, 0, True)", Some(&["1", "b"]), 0),
        (TORCH, AB, "torch.linalg.norm(x, 2, 1)", Some(&["a"]), 0),
        (TORCH, AB, "torch.norm(x, 2, 1, True)", Some(&["a", "1"]), 0),
        (TORCH, AB, "x.norm(2)", Some(&[]), 0),
        (TORCH, AB, "x.norm(2, 1)", Some(&["a"]), 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn positional_keepdims_follows_the_library() {
    check(&[
        (TORCH, AB, "x.sum(1, True)", Some(&["a", "1"]), 0),
        (TORCH, AB, "torch.mean(x, -1, True)", Some(&["a", "1"]), 0),
        (TORCH, AB, "torch.std(x, 1, True, True)", Some(&["a", "1"]), 0),
        (TORCH, AB, "x.var(1, True)", Some(&["a"]), 0),
        (NUMPY, AB, "np.sum(x, 1, None)", Some(&["a"]), 0),
        (NUMPY, AB, "np.sum(x, 1, None, None, True)", Some(&["a", "1"]), 0),
        (NUMPY, AB, "np.max(x, 1, None, True)", Some(&["a", "1"]), 0),
        (NUMPY, AB, "x.sum(1, None, None, True)", Some(&["a", "1"]), 0),
        (NUMPY, AB, "x.std(1, None, None, 0, True)", Some(&["a", "1"]), 0),
        (NONE, AB, "x.sum(1, True)", None, 0),
        (NONE, AB, "x.sum(1)", Some(&["a"]), 0),
        (NONE, AB, "x.sum(2)", None, 1),
        (NUMPY, AB, "np.sum(x, 1, keepdims=k)", None, 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn repeat_method_follows_the_library() {
    check(&[
        (NUMPY, AB, "x.repeat(3, 0)", Some(&["a*3", "b"]), 0),
        (TORCH, AB, "x.repeat(2, 3)", Some(&["a*2", "b*3"]), 0),
        (TORCH, AB, "x.repeat(2)", None, 0),
        (TORCH, AB, "x.repeat_interleave(2, 1)", Some(&["a", "b*2"]), 0),
        (TORCH, AB, "torch.repeat_interleave(x, 2, 1)", Some(&["a", "b*2"]), 0),
        (NONE, AB, "x.repeat_interleave(2, 1)", Some(&["a", "b*2"]), 0),
        (NONE, AB, "x.repeat(3, 0)", None, 0),
        (NUMPY, "x: Float[Array, \"a b\"], r: Int[Array, \"a\"]", "x.repeat(r, 0)", None, 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn squeeze_of_a_dim_that_is_not_one() {
    check(&[
        (TORCH, AB, "x.squeeze(1)", None, 0),
        (TORCH, A3, "x.squeeze(1)", Some(&["a", "3"]), 0),
        (TORCH, A3, "torch.squeeze(x, 1)", Some(&["a", "3"]), 0),
        (TORCH, "x: Float[Array, \"a 1\"]", "x.squeeze(1)", Some(&["a"]), 0),
        (NUMPY, AB, "x.squeeze(1)", None, 0),
        (NUMPY, AB, "np.squeeze(x, 1)", None, 0),
        (NUMPY, A3, "np.squeeze(x, 1)", None, 1),
        (NUMPY, A3, "x.squeeze(1)", None, 1),
        (NONE, A3, "x.squeeze(1)", None, 0),
        (JAX, A3, "jax.lax.squeeze(x, (1,))", None, 1),
    ]);
}

#[test]
#[rustfmt::skip]
fn dot_general_reads_dimension_numbers() {
    check(&[
        (JAX, "a: Float[Array, \"m k\"], b: Float[Array, \"k n\"]", "jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))", Some(&["m", "n"]), 0),
        (JAX, "a: Float[Array, \"k m\"], b: Float[Array, \"k n\"]", "jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())))", Some(&["m", "n"]), 0),
        (JAX, "a: Float[Array, \"b m k\"], b: Float[Array, \"b k n\"]", "jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))))", Some(&["b", "m", "n"]), 0),
        (JAX, "a: Float[Array, \"m k\"], b: Float[Array, \"n k\"]", "jax.lax.dot_general(a, b, dimension_numbers=(((1,), (1,)), ((), ())))", Some(&["m", "n"]), 0),
        (JAX, "a: Float[Array, \"m k\"], b: Float[Array, \"k n\"]", "jax.lax.dot_general(a, b, dims)", None, 0),
        (JAX, "a: Float[Array, \"2 3\"], b: Float[Array, \"4 5\"]", "jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))", None, 1),
    ]);
}

#[test]
#[rustfmt::skip]
fn literal_overflow_gives_unknown() {
    let x = "x: Float[Array, \"1 1 8\"], w: Float[Array, \"4 1 3\"]";
    check(&[
        (TORCH, x, "torch.nn.functional.max_pool1d(x, kernel_size=9223372036854775807)", None, 0),
        (TORCH, x, "torch.nn.functional.max_pool1d(x, 2, padding=9223372036854775807)", None, 0),
        (TORCH, x, "torch.nn.functional.conv1d(x, w, dilation=9223372036854775807)", None, 0),
        (TORCH, x, "torch.nn.functional.conv1d(x, w, padding=9223372036854775807)", None, 0),
        (TORCH, x, "torch.nn.functional.pad(x, (9223372036854775807, 9223372036854775807))", None, 0),
        (JAX, x, "jax.lax.pad(x, 0.0, ((0, 0, 0), (0, 0, 0), (0, 9223372036854775807, 9223372036854775807)))", None, 0),
        (JAX, x, "jax.lax.reduce_window(x, 0.0, f, (1, 1, 2), (1, 1, 9223372036854775807), ((0, 0), (0, 0), (9223372036854775807, 9223372036854775807)))", None, 0),
        (JAX, x, "jax.lax.slice(x, (0, 0, -9223372036854775807), (1, 1, 9223372036854775807))", None, 0),
        (NUMPY, x, "np.arange(-9223372036854775807, 9223372036854775807)", None, 0),
        (TORCH, "x: Float[Array, \"100000\"]", "torch.combinations(x, 5)", None, 0),
    ]);
}

#[test]
fn torch_split_overflow_gives_unknown() {
    let shapes = std::collections::HashMap::from([("x".to_string(), vec!["n".to_string()])]);
    let args = [
        CallArgument::Positional { value: "x".to_string() },
        CallArgument::Positional { value: "18446744073709551615".to_string() },
    ];
    assert_eq!(super::compute_torch_split_shapes(&args, &shapes, Some(3)), Ok(None));
}
