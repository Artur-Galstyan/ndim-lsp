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

const F: &str = "import torch\nimport torch.nn.functional as F\n";

#[test]
#[rustfmt::skip]
fn tensor_methods() {
    let xy = "x: Float[Array, \"a b\"], y: Float[Array, \"b c\"], v: Float[Array, \"b\"]";
    check(&[
        (TORCH, AB, "x.exp()", Some(&["a", "b"]), 0),
        (TORCH, AB, "x.rsqrt()", Some(&["a", "b"]), 0),
        (TORCH, AB, "x.log_softmax(-1)", Some(&["a", "b"]), 0),
        (TORCH, AB, "x.relu_()", Some(&["a", "b"]), 0),
        (TORCH, "x: Float[Array, \"a b\"], y: Float[Array, \"c\"]", "x.type_as(y)", Some(&["a", "b"]), 0),
        (TORCH, "x: Float[Array, \"a b\"], m: Bool[Array, \"a b\"]", "x.masked_fill_(m, 0.0)", Some(&["a", "b"]), 0),
        (TORCH, AB, "x.t()", Some(&["b", "a"]), 0),
        (TORCH, "x: Float[Array, \"a\"]", "x.t()", Some(&["a"]), 0),
        (TORCH, "x: Float[Array, \"a b c\"]", "x.t()", None, 1),
        (TORCH, xy, "x.mm(y)", Some(&["a", "c"]), 0),
        (TORCH, xy, "x.matmul(y)", Some(&["a", "c"]), 0),
        (TORCH, xy, "x.mv(v)", Some(&["a"]), 0),
        (TORCH, xy, "y.mm(x)", None, 1),
    ]);
}

#[test]
#[rustfmt::skip]
fn torch_matrix_products() {
    let xy = "x: Float[Array, \"a b\"], y: Float[Array, \"b c\"], v: Float[Array, \"b\"], bias: Float[Array, \"c\"]";
    let batched = "i: Float[Array, \"n a c\"], x: Float[Array, \"n a b\"], y: Float[Array, \"n b c\"], z: Float[Array, \"*s a b\"]";
    check(&[
        (TORCH, xy, "torch.mm(x, y)", Some(&["a", "c"]), 0),
        (TORCH, xy, "torch.mm(x, v)", None, 1),
        (TORCH, xy, "torch.mv(x, v)", Some(&["a"]), 0),
        (TORCH, xy, "torch.addmm(bias, x, y)", Some(&["a", "c"]), 0),
        (TORCH, xy, "torch.addmm(x, x, y)", None, 1),
        (TORCH, batched, "torch.baddbmm(i, x, y)", Some(&["n", "a", "c"]), 0),
        (TORCH, batched, "torch.mm(z, y)", None, 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn functional_layers() {
    let lin = "x: Float[Array, \"n i\"], w: Float[Array, \"o i\"], u: Float[Array, \"i\"], b: Float[Array, \"o\"]";
    let attention = "q: Float[Array, \"n h l e\"], k: Float[Array, \"n g s e\"], v: Float[Array, \"n g s f\"], w: Float[Array, \"n g s d\"]";
    check(&[
        (F, lin, "F.linear(x, w, b)", Some(&["n", "o"]), 0),
        (F, lin, "F.linear(x, u)", Some(&["n"]), 0),
        (F, "x: Float[Array, \"n 3\"], w: Float[Array, \"o 4\"]", "F.linear(x, w)", None, 1),
        (F, "x: Float[Array, \"n c h\"]", "F.layer_norm(x, (h,))", Some(&["n", "c", "h"]), 0),
        (F, "x: Float[Array, \"n c h\"], m: Float[Array, \"c\"]", "F.batch_norm(x, m, m)", Some(&["n", "c", "h"]), 0),
        (F, "x: Float[Array, \"n c h\"]", "F.group_norm(x, 2)", Some(&["n", "c", "h"]), 0),
        (F, "x: Float[Array, \"n c h\"]", "F.instance_norm(x)", Some(&["n", "c", "h"]), 0),
        (F, "x: Float[Array, \"n c h\"]", "F.alpha_dropout(x, 0.1)", Some(&["n", "c", "h"]), 0),
        (F, attention, "F.scaled_dot_product_attention(q, k, v)", Some(&["n", "h", "l", "f"]), 0),
        (F, "q: Float[Array, \"l 4\"], k: Float[Array, \"s 5\"], v: Float[Array, \"s f\"]", "F.scaled_dot_product_attention(q, k, v)", None, 1),
        (F, "q: Float[Array, \"*b l e\"], k: Float[Array, \"*b s e\"], v: Float[Array, \"*b s f\"]", "F.scaled_dot_product_attention(q, k, v)", None, 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn functional_losses() {
    let xy = "x: Float[Array, \"n 3\"], y: Float[Array, \"n 3\"], z: Float[Array, \"n 4\"]";
    check(&[
        (F, xy, "F.mse_loss(x, y)", Some(&[]), 0),
        (F, xy, "F.mse_loss(x, y, reduction='none')", Some(&["n", "3"]), 0),
        (F, xy, "F.mse_loss(x, y, None, None, 'none')", Some(&["n", "3"]), 0),
        (F, xy, "F.mse_loss(x, y, reduce=False)", None, 0),
        (F, xy, "F.mse_loss(x, y, reduction='avg')", None, 1),
        (F, xy, "F.l1_loss(x, z)", None, 1),
        (F, xy, "F.smooth_l1_loss(x, y, reduction='sum')", Some(&[]), 0),
        (F, xy, "F.huber_loss(x, y, 'none')", Some(&["n", "3"]), 0),
        (F, xy, "F.binary_cross_entropy(x, y, None, None, None, 'none')", Some(&["n", "3"]), 0),
        (F, xy, "F.binary_cross_entropy_with_logits(x, z)", None, 1),
    ]);
}

#[test]
#[rustfmt::skip]
fn data_dependent_sizes() {
    let xm = "x: Float[Array, \"a b\"], m: Bool[Array, \"a b\"]";
    check(&[
        (NUMPY, AB, "np.unique(x)", Some(&["?"]), 0),
        (JAX, AB, "jnp.unique(x, equal_nan=False)", Some(&["?"]), 0),
        (TORCH, AB, "torch.unique(x, sorted=True)", Some(&["?"]), 0),
        (NUMPY, AB, "np.unique(x, return_counts=True)", None, 0),
        (NUMPY, AB, "np.unique(x, True)", None, 0),
        (NUMPY, AB, "np.unique(x, axis=0)", None, 0),
        (JAX, AB, "jnp.unique(x, size=4)", None, 0),
        (TORCH, xm, "torch.masked_select(x, m)", Some(&["?"]), 0),
        (TORCH, xm, "x.masked_select(m)", Some(&["?"]), 0),
        (NUMPY, xm, "np.extract(m, x)", Some(&["?"]), 0),
        (NUMPY, "x: Float[Array, \"a b\"], c: Bool[Array, \"b\"]", "np.compress(c, x)", Some(&["?"]), 0),
        (NUMPY, "x: Float[Array, \"a b\"], c: Bool[Array, \"b\"]", "np.compress(c, x, axis=1)", None, 0),
        (NUMPY, "x: Int[Array, \"a\"]", "np.bincount(x)", Some(&["?"]), 0),
        (TORCH, "x: Int[Array, \"a\"]", "torch.bincount(x)", Some(&["?"]), 0),
    ]);
}

#[test]
#[rustfmt::skip]
fn linalg_additions() {
    let abc = "a: Float[Array, \"m k\"], b: Float[Array, \"k p\"], c: Float[Array, \"p n\"], v: Float[Array, \"n\"], s: Float[Array, \"k k\"]";
    check(&[
        (JAX, "x: Float[Array, \"b n n\"]", "jnp.linalg.eigvals(x)", Some(&["b", "n"]), 0),
        (NUMPY, "x: Float[Array, \"n n\"]", "np.linalg.eigvalsh(x)", Some(&["n"]), 0),
        (TORCH, "x: Float[Array, \"3 4\"]", "torch.linalg.eigvals(x)", None, 1),
        (NUMPY, "x: Float[Array, \"n n\"]", "np.linalg.matrix_power(x, 3)", Some(&["n", "n"]), 0),
        (NUMPY, abc, "np.linalg.multi_dot([a, b, c])", Some(&["m", "n"]), 0),
        (JAX, abc, "jnp.linalg.multi_dot([a, b, c, v])", Some(&["m"]), 0),
        (TORCH, abc, "torch.linalg.multi_dot([a, c])", None, 1),
        (NUMPY, abc, "np.linalg.multi_dot([a, s, b, c])", Some(&["m", "n"]), 0),
    ]);
}

#[test]
fn slogdet_unpacks_two_scalars() {
    let code = "import jax.numpy as jnp\ndef f(x: Float[Array, \"n n\"]):\n    sign, logdet = jnp.linalg.slogdet(x)\n";
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
    let tree = parser.parse(code, None).unwrap();
    let analysis = analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap();
    let shape = |name: &str| analysis.scopes.iter().find_map(|scope| scope.shapes.get(name)).cloned();
    assert_eq!((shape("sign"), shape("logdet")), (Some(Vec::new()), Some(Vec::new())));
    assert_eq!(run(NUMPY, "x: Float[Array, \"3 4\"]", "np.linalg.slogdet(x)"), (None, 1));
}
