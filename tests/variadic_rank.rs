//! A variadic dim stands for any number of axes, so the count of the dims of a
//! shape with one is only a lower bound. No rule may report a rank error for it.

use ndim_lsp::{LayerShapeAnalysis, analyze_layer_shapes};
use tree_sitter::Parser;

fn analyze(code: &str) -> LayerShapeAnalysis {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
    let tree = parser.parse(code, None).unwrap();
    analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
}

const HEAD: &str = "import jax\nimport jax.numpy as jnp\nimport numpy as np\nimport torch\nimport einops\nfrom jaxtyping import Array, Float\n";

fn body(params: &str, lines: &str) -> String {
    format!("{HEAD}def f({params}):\n{}\n", lines.lines().map(|line| format!("    {line}\n")).collect::<String>())
}

fn messages(analysis: &LayerShapeAnalysis) -> Vec<String> {
    analysis.errors.iter().map(|error| error.message.clone()).collect()
}

const VARIADIC: [&str; 6] = ["...", "*b", "*b n m", "... n m", "n *b m", "#*b n"];

/// Each rule on an input whose dims count less than, the same as, or more than
/// the rule asks for.
#[test]
fn no_rule_reports_a_rank_error_for_a_variadic_dim() {
    let calls = [
        "jnp.vstack([x, w])",
        "jnp.hstack([x, w])",
        "jnp.dstack([x, w])",
        "jnp.column_stack([x, w])",
        "jnp.append(x, w, axis=2)",
        "jnp.block([x, w])",
        "jnp.block([[x, w], [w, x]])",
        "x.sum(axis=5)",
        "x.mean(axis=-5)",
        "jnp.max(x, axis=(0, 7))",
        "x.transpose(0, 3, 1, 2)",
        "x.transpose(1, 0)",
        "jnp.transpose(x, (2, 0, 1))",
        "x.permute(3, 2, 1, 0)",
        "jnp.swapaxes(x, 0, 6)",
        "jnp.moveaxis(x, 0, 6)",
        "x.squeeze(5)",
        "jnp.expand_dims(x, 6)",
        "x.unsqueeze(6)",
        "x.reshape(4, 5)",
        "x.reshape(-1)",
        "x.flatten(1, 4)",
        "torch.flatten(x, 3)",
        "jnp.einsum(\"ij,jk->ik\", x, w)",
        "jnp.einsum(\"ijk,k->ij\", x, w)",
        "jnp.einsum(\"i->i\", x)",
        "einops.rearrange(x, \"a b c -> a c b\")",
        "einops.reduce(x, \"a b c d -> a\", \"mean\")",
        "jnp.tile(x, (1, 2, 3, 4))",
        "jnp.repeat(x, 2, axis=5)",
        "x.repeat(1, 2, 3, 4)",
        "jnp.pad(x, ((0, 1), (0, 1), (0, 1)))",
        "jnp.flip(x, axis=5)",
        "jnp.roll(x, 1, axis=5)",
        "jnp.take(x, 0, axis=5)",
        "jnp.take_along_axis(x, w, axis=5)",
        "jnp.squeeze(x, axis=5)",
        "jnp.diagonal(x)",
        "jnp.trace(x)",
        "jnp.linalg.inv(x)",
        "jnp.linalg.det(x)",
        "jnp.linalg.norm(x, axis=5)",
        "jnp.matmul(x, w)",
        "jnp.dot(x, w)",
        "jnp.tensordot(x, w, axes=1)",
        "jnp.inner(x, w)",
        "jnp.outer(x, w)",
        "jnp.cross(x, w)",
        "jnp.add(x, w)",
        "jnp.where(x > 0, x, w)",
        "jnp.broadcast_to(x, (2, 3))",
        "jax.nn.softmax(x, axis=5)",
        "jnp.argmax(x, axis=5)",
        "jnp.sort(x, axis=5)",
        "jnp.cumsum(x, axis=5)",
        "jnp.diff(x, axis=5)",
        "x[:, :, :, 0]",
        "x[0, 1, 2, 3]",
        "jax.vmap(jnp.sum)(x)",
        "jax.vmap(jnp.sum, in_axes=3)(x)",
        "jax.vmap(jnp.add, in_axes=(0, 3))(x, w)",
        "jnp.split(x, 2, axis=5)",
        "jnp.array_split(x, 2, axis=5)",
        "torch.chunk(x, 2, dim=5)",
        "torch.unbind(x, dim=5)",
        "torch.topk(x, 2, dim=5)",
        "torch.gather(x, 5, w)",
        "torch.index_select(x, 5, w)",
        "torch.narrow(x, 5, 0, 1)",
        "torch.nn.functional.cross_entropy(x, w)",
        "torch.nn.functional.mse_loss(x, w)",
        "torch.nn.functional.conv2d(x, w)",
        "torch.nn.functional.max_pool2d(x, 2)",
        "torch.nn.functional.pixel_shuffle(x, 2)",
        "torch.nn.functional.interpolate(x, scale_factor=2)",
        "torch.nn.functional.one_hot(x, 3)",
    ];
    let mut wrong = Vec::new();
    for variadic in VARIADIC {
        for fixed in ["k", "n m", "a n m d e"] {
            for call in calls {
                let params = format!("x: Float[Array, \"{variadic}\"], w: Float[Array, \"{fixed}\"]");
                let code = body(&params, &format!("y = {call}"));
                let found = messages(&analyze(&code));
                if !found.is_empty() {
                    wrong.push(format!("x {variadic:?} w {fixed:?} {call}: {found:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} false errors:\n{}", wrong.len(), wrong.join("\n"));
}

/// The two inputs of each pair agree on each proven axis, so only the rank is
/// uncertain.
#[test]
fn a_join_reports_no_error_when_only_the_rank_is_uncertain() {
    let pairs = [
        ("...", "k"),
        ("...", "a n m d e"),
        ("*b", "k"),
        ("*b", "a n m d e"),
        ("*b n m", "n m"),
        ("*b n m", "a c n m"),
        ("... n m", "n m"),
        ("n *b m", "n m"),
        ("n *b m", "n a c m"),
        ("#*b n", "k n"),
    ];
    let calls = [
        "jnp.concatenate([x, w], axis=0)",
        "jnp.concatenate([w, x], axis=0)",
        "jnp.concatenate([x, w], axis=-1)",
        "jnp.concatenate([x, w], axis=3)",
        "torch.cat([x, w], dim=1)",
        "jnp.stack([x, w])",
        "jnp.stack([w, x], axis=-1)",
        "jnp.stack([x, w], axis=4)",
    ];
    for (left, right) in pairs {
        for call in calls {
            let params = format!("x: Float[Array, \"{left}\"], w: Float[Array, \"{right}\"]");
            let found = messages(&analyze(&body(&params, &format!("y = {call}"))));
            assert_eq!(found, Vec::<String>::new(), "x {left:?} w {right:?} {call}");
        }
    }
}

/// An axis that alignment proves conflicts, or a rank under the lower bound, is
/// an error with no shape.
#[test]
fn a_join_reports_a_proven_conflict_and_gives_no_shape() {
    let result = |left: &str, right: &str, call: &str| {
        let params = format!("x: Float[Array, \"{left}\"], o: Float[Array, \"{right}\"]");
        let analysis = analyze(&body(&params, &format!("y = {call}")));
        assert_eq!(analysis.scopes.iter().find_map(|scope| scope.shapes.get("y")), None, "{call}");
        messages(&analysis)
    };
    let stack = "jnp.stack([x, o], axis=-1)";
    let concat = "jnp.concatenate([x, o], axis=-1)";
    assert_eq!(result("*b 3", "*b 4", stack), ["stack dimension mismatch at axis 1: expected 3, got 4"]);
    assert_eq!(result("*b 3", "*b 4", "torch.stack([x, o])"), ["stack dimension mismatch at axis 1: expected 3, got 4"]);
    assert_eq!(result("1 *b 3", "2 *b 3", concat), ["concatenate dimension mismatch at axis 0: expected 1, got 2"]);
    assert_eq!(result("*b 3 5", "*b 4 5", concat), ["concatenate dimension mismatch at axis 1: expected 3, got 4"]);
    assert_eq!(result("*b 5 3", "*b 5 4", "jnp.concatenate([x, o], axis=0)"), ["concatenate dimension mismatch at axis 2: expected 3, got 4"]);
    assert_eq!(result("*b n m", "k", stack), ["stack expected all inputs to have rank 2 or more, got rank 1"]);
    assert_eq!(result("*b n m", "k", "jnp.stack([x, o])"), ["stack expected all inputs to have rank 2 or more, got rank 1"]);
    assert_eq!(result("*b n m", "k", concat), ["concatenate expected all inputs to have rank 2 or more, got rank 1"]);
    assert_eq!(result("k", "*b n m", "jnp.concatenate([x, o], axis=0)"), ["concatenate expected all inputs to have rank 2 or more, got rank 1"]);
}

/// The axis of a concatenate may differ. A dim can be that axis when the
/// variadic dim is empty.
#[test]
fn a_concatenate_does_not_compare_a_dim_that_can_be_its_axis() {
    let result = |left: &str, right: &str, call: &str| {
        let params = format!("x: Float[Array, \"{left}\"], o: Float[Array, \"{right}\"]");
        messages(&analyze(&body(&params, &format!("y = {call}"))))
    };
    let none = Vec::<String>::new();
    assert_eq!(result("*b 3", "*b 4", "jnp.concatenate([x, o], axis=-1)"), none);
    assert_eq!(result("*b 3", "*b 4", "jnp.concatenate([x, o], axis=0)"), none);
    assert_eq!(result("*b 3 5", "*b 4 5", "jnp.concatenate([x, o], axis=1)"), none);
    assert_eq!(result("1 *b 3", "2 *b 3", "jnp.concatenate([x, o], axis=0)"), none);
    assert_eq!(result("1 *b 3", "2 *b 3", "jnp.concatenate([x, o], axis=-2)"), none);
}

#[test]
fn a_call_reads_a_variadic_rank_as_a_lower_bound() {
    for param in ["...", "*c"] {
        let code = format!(
            "{HEAD}def g(a: Float[Array, \"{param}\"]):\n    return a\ndef f(x: Float[Array, \"n m\"]):\n    y = g(x[1:])\n"
        );
        assert_eq!(messages(&analyze(&code)), Vec::<String>::new(), "{param}");
    }
}

#[test]
fn a_call_still_reports_what_a_variadic_dim_cannot_explain() {
    let code = format!(
        "{HEAD}def g(a: Float[Array, \"*c n m\"]):\n    return a\ndef f(x: Float[Array, \"k\"]):\n    y = g(x)\n"
    );
    assert_eq!(messages(&analyze(&code)), ["call to g: argument 'a' expected rank 3, got rank 1"]);
}

#[test]
fn a_call_gives_a_return_shape_only_for_a_proven_axis() {
    let result = |param: &str, ret: &str, arg: &str| {
        let code = format!(
            "{HEAD}def g(a: Float[Array, \"{param}\"]) -> Float[Array, \"{ret}\"]:\n    return a\ndef f(x: Float[Array, \"{arg}\"]):\n    y = g(x)\n"
        );
        let analysis = analyze(&code);
        assert_eq!(messages(&analysis), Vec::<String>::new());
        analysis.scopes.iter().find_map(|scope| scope.shapes.get("y").cloned())
    };
    let dims = |dims: &[&str]| Some(dims.iter().map(|dim| dim.to_string()).collect::<Vec<_>>());
    assert_eq!(result("*c n", "n", "a b d"), dims(&["d"]));
    assert_eq!(result("n *c", "n", "a b d"), dims(&["a"]));
    assert_eq!(result("*c n", "*c n", "a b d"), None);
    assert_eq!(result("*c n", "*c n", "a d"), dims(&["a", "d"]));
    assert_eq!(result("*c n", "*c n", "*b d"), dims(&["*b", "d"]));
    assert_eq!(result("p n", "p n", "*b d"), dims(&["?", "d"]));
}

#[test]
fn concatenate_and_stack_keep_only_an_axis_that_no_variadic_dim_moves() {
    let result = |params: &str, call: &str| {
        let analysis = analyze(&body(params, &format!("y = {call}")));
        assert_eq!(messages(&analysis), Vec::<String>::new(), "{call}");
        analysis.scopes.iter().find_map(|scope| scope.shapes.get("y").cloned())
    };
    let dims = |dims: &[&str]| Some(dims.iter().map(|dim| dim.to_string()).collect::<Vec<_>>());
    let params = "x: Float[Array, \"*b n m\"], w: Float[Array, \"n m\"]";
    assert_eq!(result(params, "jnp.concatenate([x[..., 1:], w[..., 1:]], axis=0)"), None);
    assert_eq!(result(params, "jnp.concatenate([x, w], axis=-1)"), dims(&["*b", "n", "m+m"]));
    assert_eq!(result(params, "jnp.concatenate([x, x], axis=-2)"), dims(&["*b", "n+n", "m"]));
    assert_eq!(result(params, "jnp.concatenate([x, x], axis=-3)"), None);
    assert_eq!(result(params, "jnp.concatenate([x, x], axis=1)"), None);
    assert_eq!(result(params, "jnp.stack([x, w], axis=-1)"), dims(&["*b", "n", "m", "2"]));
    assert_eq!(result(params, "jnp.stack([x, w], axis=0)"), dims(&["2", "*b", "n", "m"]));
    assert_eq!(result(params, "jnp.stack([x, w], axis=1)"), None);
}

#[test]
fn operators_align_from_the_right_up_to_a_variadic_dim() {
    let found = |left: &str, right: &str, call: &str| {
        let params = format!("x: Float[Array, \"{left}\"], w: Float[Array, \"{right}\"]");
        messages(&analyze(&body(&params, &format!("y = {call}")))).len()
    };
    assert_eq!(found("a *b m", "3 4 m", "x + w"), 0);
    assert_eq!(found("*b n", "k n", "x * w"), 0);
    assert_eq!(found("n *b m d", "a p m d e", "x @ w"), 0);
    assert_eq!(found("*b n m", "k", "x + w"), 1);
    assert_eq!(found("*b n m", "n m", "x @ w"), 1);
}

#[test]
fn a_layer_keeps_an_error_on_the_last_axis_and_drops_a_rank_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("torch/nn")).unwrap();
    std::fs::write(dir.path().join("torch/__init__.py"), "from . import nn").unwrap();
    let layers = "class Linear:\n    def __init__(self, in_features, out_features, bias=True): pass\nclass Conv2d:\n    def __init__(self, in_channels, out_channels, kernel_size): pass\n";
    std::fs::write(dir.path().join("torch/nn/__init__.py"), layers).unwrap();
    let found = |layer: &str, dims: &str| {
        let code = format!(
            "{HEAD}import torch.nn as nn\ndef f(x: Float[Array, \"{dims}\"]):\n    l = nn.{layer}\n    y = l(x)\n"
        );
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let roots = [dir.path().to_path_buf()];
        let read = |path: &std::path::PathBuf| std::fs::read_to_string(path).ok();
        let analysis = analyze_layer_shapes(tree.root_node(), &code, &roots, read, 5, None).unwrap();
        analysis.errors.len()
    };
    assert_eq!(found("Linear(3, 4)", "... 5"), 1);
    assert_eq!(found("Linear(3, 4)", "... 3"), 0);
    assert_eq!(found("Conv2d(3, 4, 1)", "..."), 0);
    assert_eq!(found("Conv2d(3, 4, 1)", "*b"), 0);
}
