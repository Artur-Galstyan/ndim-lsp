//! One analysis of a hostile file stays bounded in text, memory, and time.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

use ndim_lsp::{
    LayerShapeAnalysis, ResolutionCache, UnitMemo, analyze_layer_shapes,
    analyze_layer_shapes_with_memo, new_resolution_cache,
};
use tempfile::TempDir;
use tree_sitter::Parser;

struct Tracked;

static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let now = NOW.fetch_add(layout.size(), Relaxed) + layout.size();
        PEAK.fetch_max(now, Relaxed);
        LARGEST.fetch_max(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NOW.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let now = NOW.fetch_add(new_size, Relaxed) + new_size;
        PEAK.fetch_max(now, Relaxed);
        LARGEST.fetch_max(new_size, Relaxed);
        NOW.fetch_sub(layout.size(), Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static TRACKED: Tracked = Tracked;

const LINES: &str = "import jax\nimport jax.numpy as jnp\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport equinox as eqx\nfrom jaxtyping import Array, Float\n";

/// The text of the file for `lines` repeats of `step` after `head`.
fn repeated(head: &str, step: &str, lines: usize) -> String {
    let mut code = format!("{LINES}{head}\n");
    for at in 0..lines {
        code.push_str(&step.replace("{at}", &at.to_string()).replace("{next}", &(at + 1).to_string()));
        code.push('\n');
    }
    code
}

const X: &str = "def f(x: Float[Array, \"n m\"], k: int):";

fn chains(lines: usize) -> Vec<(&'static str, String)> {
    let local = |step: &str| repeated(X, &format!("    {step}"), lines);
    let mut paths: Vec<(&str, String)> = vec![
        ("concatenate", local("x = jnp.concatenate([x, x], axis=0)")),
        ("concatenate_wide", local("x = jnp.concatenate([x, x, x, x, x, x, x, x], axis=0)")),
        ("torch_cat", local("x = torch.cat((x, x), dim=0)")),
        ("hstack", local("x = jnp.hstack([x, x])")),
        ("add", local("x = x + x")),
        ("stack_flatten", local("x = jnp.stack([x, x]).reshape(-1, x.shape[-1])")),
        ("tile", local("x = jnp.tile(x, (x.shape[0], 1))")),
        ("repeat", local("x = jnp.repeat(x, x.shape[0], axis=0)")),
        ("torch_repeat", local("x = x.repeat(x.shape[0], 1)")),
        ("reshape_product", local("x = x.reshape(x.shape[0] * x.shape[0], -1)")),
        ("reshape_size", local("x = x.reshape(x.size, 1)")),
        ("flatten", local("x = jnp.stack([x, x], axis=2).reshape(x.shape[0], -1)")),
        ("outer", local("x = jnp.outer(x, x)")),
        ("kron", local("x = jnp.kron(x, x)")),
        ("einsum", local("x = jnp.einsum(\"ab,cd->acbd\", x, x).reshape(x.shape[0] * x.shape[0], -1)")),
        ("broadcast_to", local("x = jnp.broadcast_to(x, (x.shape[0], x.shape[1]))")),
        ("pad", local("x = jnp.pad(x, x.shape[0])")),
        ("slice_min", local("x = x[:x.shape[0]]")),
        ("slice_min_k", local("x = x[:k, :k]")),
        ("slice_step", local("x = x[::2]")),
        ("slice_ceil", local("x = jnp.concatenate([x[::k], x[::k]], axis=0)")),
        ("slice_min_doubled", local("x = jnp.concatenate([x[:k], x[1:]], axis=0)")),
        ("qr", local("q, r = jnp.linalg.qr(x)\n    x = jnp.concatenate([q, r], axis=1)")),
        ("svd", local("u, s, v = jnp.linalg.svd(x)\n    x = jnp.outer(s, s)")),
        ("split", local("a, b = jnp.split(x, 2, axis=0)\n    x = jnp.concatenate([a, b, a, b], axis=0)")),
        ("chunk", local("a, b = torch.chunk(x, 2, dim=0)\n    x = torch.cat([a, b, a, b], dim=0)")),
        ("arange", local("x = jnp.arange(x.shape[0], x.shape[0] + x.shape[0])[:, None]")),
        ("linspace", local("x = jnp.zeros((x.shape[0] * x.shape[0], 1))")),
        ("scalar_chain", repeated(format!("{X}\n    a0 = x.shape[0]").as_str(), "    a{next} = a{at} + a{at}\n    z{at} = jnp.zeros((a{next}, 1))", lines)),
        ("scalar_chain_mul", repeated(format!("{X}\n    a0 = x.shape[0]").as_str(), "    a{next} = a{at} * a{at}\n    z{at} = jnp.zeros((a{next}, 1))", lines)),
        ("scalar_augmented", local("k += k\n    z = jnp.zeros((k, 1))")),
        ("shape_add", local("s = x.shape + x.shape\n    x = jnp.zeros(s)")),
        ("shape_add_inline", local("x = jnp.zeros(x.shape + x.shape)")),
        ("tensordot", local("x = jnp.tensordot(x, x, axes=0)")),
        ("loop_nested", format!("{LINES}{X}\n{}", (0..lines.min(40)).map(|depth| format!("{}for i{depth} in range(2):\n{}x = jnp.concatenate([x, x], axis=0)\n", " ".repeat(4 + depth), " ".repeat(5 + depth))).collect::<String>())),
        ("long_concat", format!("{LINES}{X}\n    y = jnp.concatenate([{}x], axis=0)\n", "x, ".repeat(lines * 100))),
        ("wide_fan", repeated("def g0(y):\n    return y", &format!("def g{{next}}(y):\n    return jnp.stack([{}g{{at}}(y)])", "g{at}(y), ".repeat(9)), 9) + "def f(x: Float[Array, \"n m\"]):\n    z = g9(x)\n"),
        ("shape_star", local("x = jnp.zeros((*x.shape, *x.shape))")),
        ("reshape_star", local("x = x.reshape(*x.shape, *x.shape)")),
        ("call_annotated", repeated("def g(y: Float[Array, \"a b\"]) -> Float[Array, \"a*a+a b\"]:\n    return y\ndef f(x: Float[Array, \"n m\"]):", "    x = g(x)", lines)),
        ("call_nested", repeated("def g(y: Float[Array, \"a b\"]) -> Float[Array, \"a*a+a b\"]:\n    return y\ndef f(x: Float[Array, \"n m\"]):\n    y = x", "    y = g(g(y))", lines)),
        ("call_seeded", repeated("def g(y):\n    return jnp.concatenate([y, y], axis=0)\ndef f(x: Float[Array, \"n m\"]):", "    x = g(x)", lines)),
        ("call_fan", repeated("def g0(y):\n    return y", "def g{next}(y):\n    return g{at}(y) + g{at}(y)", lines) + &format!("def f(x: Float[Array, \"n m\"]):\n    z = g{lines}(x)\n")),
        ("call_tower", repeated("def g0(y):\n    return jnp.concatenate([y, y], axis=0)", "def g{next}(y):\n    return jnp.concatenate([g{at}(y), g{at}(y)], axis=0)", lines) + &format!("def f(x: Float[Array, \"n m\"]):\n    z = g{lines}(x)\n")),
        ("call_recursive", repeated("def g(y, n):\n    if n == 0:\n        return y\n    return g(jnp.concatenate([y, y], axis=0), n - 1)\ndef f(x: Float[Array, \"n m\"]):", "    x = g(x, 3)", lines)),
        ("layer_chain", repeated("class M(nn.Module):\n    def __init__(self, c, k):\n        self.c = nn.Conv1d(c, c, k, stride=k)\n    def forward(self, x: Float[Array, \"b c n\"]):", "        x = self.c(x)", lines)),
        ("layer_sequential", format!("{LINES}class M(nn.Module):\n    def __init__(self, c, k):\n        self.s = nn.Sequential({})\n    def forward(self, x: Float[Array, \"b c n\"]):\n        y = self.s(x)\n", vec!["nn.Conv1d(c, c, k, stride=k)"; lines].join(", "))),
        ("layer_upsample", repeated("class M(nn.Module):\n    def __init__(self, c, k):\n        self.u = nn.Upsample(scale_factor=k)\n    def forward(self, x: Float[Array, \"b c n\"]):", "        x = torch.cat([self.u(x), self.u(x)], dim=2)", lines)),
        ("layer_constant", repeated("a0 = 3", "a{next} = a{at} * a{at}", lines) + "layer = nn.Linear(a10, a20)\n"),
        ("layer_dim_chain", repeated("def f(d):\n    a0 = d", "    a{next} = a{at} * a{at} + a{at}\n    l{at} = nn.Linear(a{at}, a{next})\n    y{at} = l{at}(jnp.zeros((1, a{at})))", lines)),
        ("module_chain", repeated("class M0(eqx.Module):\n    l: eqx.nn.Linear\n    def __init__(self, d):\n        self.l = eqx.nn.Linear(d, d)\n    def __call__(self, x: Float[Array, \"d\"]):\n        return self.l(x)", "class M{next}(eqx.Module):\n    m: M{at}\n    def __init__(self, d):\n        self.m = M{at}(d * d + d)\n    def __call__(self, x):\n        return self.m(x)", lines) + &format!("class Top(eqx.Module):\n    m: M{lines}\n    def __init__(self, d):\n        self.m = M{lines}(d)\n    def __call__(self, x):\n        return self.m(x)\n")),
        ("literal_overflow", repeated("def f():\n    x = jnp.zeros((3, 2))", "    x = jnp.concatenate([x, x], axis=0)", lines)),
        ("literal_product", repeated("def f():\n    x = jnp.zeros((3, 2))", "    x = x.reshape(x.shape[0] * x.shape[0], -1)", lines)),
        ("literal_tile", repeated("def f():\n    x = jnp.zeros((3, 2))", "    x = jnp.tile(x, (x.shape[0], 1))", lines)),
        ("loop", local("for i in range(3):\n        x = jnp.concatenate([x, x], axis=0)")),
        ("branch", local("if k:\n        x = jnp.concatenate([x, x], axis=0)\n    else:\n        x = jnp.concatenate([x, x, x], axis=0)")),
        ("attribute", repeated("class M:\n    def __init__(self, x: Float[Array, \"n m\"]):\n        self.x = x\n    def f(self):\n        y = self.x", "        y = jnp.concatenate([y, y], axis=0)", lines)),
        ("vmap", local("x = jax.vmap(lambda y: jnp.concatenate([y, y]))(x)")),
    ];
    paths.push(("long_annotation", format!("{LINES}def f(x: Float[Array, \"{} m\"]):\n    y = x\n", "n+".repeat(lines * 1000) + "n")));
    paths
}

struct Measure {
    peak: usize,
    largest: usize,
    dim: usize,
    rank: usize,
    time: Duration,
    last: Option<Vec<String>>,
    errors: usize,
}

/// A small `torch.nn`, so that the layers of a chain have rules.
fn torch_stub() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("torch/nn")).unwrap();
    fs::write(dir.path().join("torch/__init__.py"), "from . import nn").unwrap();
    let layers = "class Module:\n    pass\nclass Linear:\n    def __init__(self, in_features, out_features, bias=True): pass\nclass Conv1d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1): pass\nclass ConvTranspose1d:\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): pass\nclass Upsample:\n    def __init__(self, size=None, scale_factor=None): pass\nclass Sequential:\n    def __init__(self, *args): pass\n";
    fs::write(dir.path().join("torch/nn/__init__.py"), layers).unwrap();
    dir
}

fn analyze(code: &str, root: &Path) -> LayerShapeAnalysis {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
    let tree = parser.parse(code, None).unwrap();
    let roots = [root.to_path_buf()];
    let read = |path: &PathBuf| fs::read_to_string(path).ok();
    analyze_layer_shapes(tree.root_node(), code, &roots, read, 5, None).unwrap()
}

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn measure(code: &str, root: &Path) -> Measure {
    let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let start = Instant::now();
    let base = NOW.load(Relaxed);
    PEAK.store(base, Relaxed);
    LARGEST.store(0, Relaxed);
    let analysis = analyze(code, root);
    let time = start.elapsed();
    let peak = PEAK.load(Relaxed) - base;
    let largest = LARGEST.load(Relaxed);
    let scopes = analysis.scopes.iter().flat_map(|scope| scope.shapes.values());
    let shapes: Vec<_> = scopes.chain(analysis.assignment_shapes.iter().map(|a| &a.shape)).collect();
    let dim = shapes.iter().flat_map(|shape| shape.iter()).map(String::len).max().unwrap_or(0);
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let last = analysis.assignment_shapes.last().map(|a| a.shape.clone());
    Measure { peak, largest, dim, rank, time, last, errors: analysis.errors.len() }
}

const MAX_DIM_TEXT: usize = 256;
const MAX_RANK: usize = 64;

/// The limits of one analysis of a chain. A debug build runs the test too.
const MAX_PEAK: usize = 24 << 20;
const MAX_ONE_ALLOCATION: usize = 4 << 20;
const MAX_TIME: Duration = Duration::from_secs(20);

fn assert_bounded(name: &str, measured: &Measure) {
    assert!(measured.dim <= MAX_DIM_TEXT, "{name}: a dim of {} bytes", measured.dim);
    assert!(measured.rank <= MAX_RANK, "{name}: rank {}", measured.rank);
    assert!(measured.peak <= MAX_PEAK, "{name}: peak of {} bytes", measured.peak);
    assert!(measured.largest <= MAX_ONE_ALLOCATION, "{name}: {} bytes in one step", measured.largest);
    assert!(measured.time <= MAX_TIME, "{name}: {:?}", measured.time);
}

#[test]
fn each_chain_of_sixty_lines_stays_bounded() {
    let stub = torch_stub();
    for (name, code) in chains(60) {
        assert_bounded(name, &measure(&code, stub.path()));
    }
}

/// The rank stays and the axis that grows becomes `?`. The axes that do not
/// grow keep their text.
#[test]
fn a_dim_over_the_limit_is_unknown_and_the_rank_stays() {
    let stub = torch_stub();
    let code = |name: &str| chains(60).into_iter().find(|(path, _)| *path == name).unwrap().1;
    for (name, last) in [
        ("concatenate", ["?", "m"]),
        ("concatenate_wide", ["?", "m"]),
        ("torch_cat", ["?", "m"]),
        ("hstack", ["n", "?"]),
        ("tile", ["?", "m"]),
        ("repeat", ["?", "m"]),
        ("outer", ["?", "?"]),
        ("kron", ["?", "?"]),
        ("split", ["?", "m"]),
        ("call_annotated", ["?", "m"]),
        ("call_nested", ["?", "m"]),
        ("call_seeded", ["?", "m"]),
        ("vmap", ["n", "?"]),
        ("long_annotation", ["?", "m"]),
    ] {
        let measured = measure(&code(name), stub.path());
        assert_eq!(measured.last.as_deref(), Some(last.map(String::from).as_slice()), "{name}");
        assert_eq!(measured.errors, 0, "{name}");
    }
}

/// A short chain keeps its exact text: the limit changes nothing below it.
#[test]
fn a_dim_under_the_limit_keeps_its_text() {
    let stub = torch_stub();
    let code = repeated(X, "    x = jnp.concatenate([x, x], axis=0)", 6);
    let last = measure(&code, stub.path()).last.unwrap();
    assert_eq!(last, [vec!["n"; 64].join("+"), "m".to_string()]);
}

/// A literal that overflows the integers of the rules stops being a literal. A
/// debug build panics on the overflow.
#[test]
fn a_literal_dim_does_not_overflow() {
    let stub = torch_stub();
    for lines in [62, 63, 64, 65, 66, 130] {
        let code = repeated("def f():\n    x = jnp.zeros((3, 2))", "    x = jnp.concatenate([x, x], axis=0)", lines);
        assert_bounded("literal", &measure(&code, stub.path()));
    }
    let code = repeated("def f():\n    x = jnp.zeros((3000000000, 2))", "    x = jnp.outer(x, x)", 12);
    assert_bounded("literal product", &measure(&code, stub.path()));
}

/// A literal that overflows the integers of a layer rule is `?`. The rank stays.
/// A debug build panics on the overflow.
#[test]
fn a_layer_size_does_not_overflow() {
    let stub = torch_stub();
    let max = i64::MAX;
    let last = |layer: String, calls: usize| {
        let head = format!("class M(nn.Module):\n    def __init__(self):\n        self.l = nn.{layer}\n    def forward(self, x: Float[Array, \"1 1 3\"]):");
        measure(&repeated(&head, "        x = self.l(x)", calls), stub.path()).last.unwrap()
    };
    assert_eq!(last("Upsample(scale_factor=2)".to_string(), 70), ["1", "1", "?"]);
    assert_eq!(last("Upsample(scale_factor=2)".to_string(), 3), ["1", "1", "24"]);
    assert_eq!(last(format!("ConvTranspose1d(1, 1, 3, stride={max})"), 1), ["1", "1", "?"]);
    assert_eq!(last(format!("ConvTranspose1d(1, 1, 3, padding={max})"), 1), ["1", "1", "?"]);
    assert_eq!(last(format!("Conv1d(1, 1, 3, padding={max})"), 1), ["1", "1", "?"]);
    assert_eq!(last(format!("Conv1d(1, 1, 3, dilation={max})"), 1), ["1", "1", "?"]);
    assert_eq!(last("Conv1d(1, 1, 3, padding=2)".to_string(), 1), ["1", "1", "5"]);
}

/// A short nesting gives its shape, and a very deep one gives no analysis and
/// no overflow of the stack.
#[test]
fn nesting_has_a_limit() {
    let stub = torch_stub();
    let nested = |open: &str, close: &str, count: usize| {
        format!("{LINES}{X}\n    y = {}x{}\n", open.repeat(count), close.repeat(count))
    };
    for (open, close, count) in [("(", ")", 100), ("-", "", 100), ("jnp.abs(", ")", 50)] {
        let measured = measure(&nested(open, close, count), stub.path());
        assert_eq!(measured.last, Some(vec!["n".to_string(), "m".to_string()]), "{open}");
    }
    for (open, close, count) in [("(", ")", 200_000), ("-", "", 200_000), ("jnp.abs(", ")", 50_000)] {
        let measured = measure(&nested(open, close, count), stub.path());
        assert_eq!(measured.last, None, "{open}");
        assert!(measured.time <= MAX_TIME, "{open}: {:?}", measured.time);
    }
}

/// The memo path of the server refuses a deep file too, also when an edit adds
/// the deep statement to a file that it already holds.
#[test]
fn an_edit_that_adds_a_deep_statement_gets_no_analysis() {
    let analyze_with = |code: &str, memo: &mut UnitMemo, cache: &ResolutionCache| {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes_with_memo(tree.root_node(), code, &[], |_| None, 5, Some(cache), memo).unwrap()
    };
    let (mut memo, cache) = (UnitMemo::default(), new_resolution_cache());
    let plain = format!("{LINES}{X}\n    y = x\n");
    assert_eq!(analyze_with(&plain, &mut memo, &cache).assignment_shapes.len(), 1);
    let deep = format!("{plain}{}z{}\n", "(".repeat(100_000), ")".repeat(100_000));
    assert_eq!(analyze_with(&deep, &mut memo, &cache), LayerShapeAnalysis::default());
    assert_eq!(analyze_with(&plain, &mut memo, &cache).assignment_shapes.len(), 1);
}

/// The deepest tree that the analysis accepts fits a stack of half the size of
/// the 2 MB that an LSP worker thread has. Debug frames are many times larger.
#[test]
fn the_deepest_accepted_tree_fits_the_stack() {
    let stack = if cfg!(debug_assertions) { 64 << 20 } else { 1 << 20 };
    let run = std::thread::Builder::new().stack_size(stack).spawn(|| {
        let stub = torch_stub();
        for count in [300, 480, 900, 990] {
            let code = format!("{LINES}{X}\n    y = x{}\n", " + x".repeat(count));
            assert_bounded("long sum", &measure(&code, stub.path()));
        }
        let code = format!("{LINES}{X}\n    y = {}x{}\n", "(".repeat(490), ")".repeat(490));
        assert_bounded("parentheses", &measure(&code, stub.path()));
        let code = format!("{LINES}{X}\n    y = {}x{}\n", "jnp.abs(".repeat(320), ")".repeat(320));
        assert_bounded("calls", &measure(&code, stub.path()));
    });
    run.unwrap().join().unwrap();
}

/// A chain of names is linear: each name runs its check one time.
#[test]
fn a_chain_of_names_is_linear_in_time() {
    let stub = torch_stub();
    let step = "    a{next} = a{at} + a{at}\n    z{at} = jnp.zeros((a{next}, 1))";
    let code = repeated(&format!("{X}\n    a0 = x.shape[0]"), step, 400);
    assert_bounded("names", &measure(&code, stub.path()));
}

/// A chain of names that runs from the last name to the first fits the stack
/// of a thread with the default size: a check stops at the depth limit.
#[test]
fn a_deep_chain_of_names_fits_the_stack() {
    let run = std::thread::spawn(|| {
        let names = 5_000;
        let mut code = String::from("import jax.numpy as jnp\n");
        for i in (1..=names).rev() {
            code.push_str(&format!("x{i} = x{} + 1\n", i - 1));
        }
        code.push_str(&format!("x0 = 1\ny = jnp.zeros((3, 4)) * x{names}\nz = jnp.zeros((3, 4)) * x1\n"));
        let stub = torch_stub();
        let measured = measure(&code, stub.path());
        assert!(measured.time <= MAX_TIME, "{:?}", measured.time);
        assert_eq!(measured.last, Some(vec!["3".to_string(), "4".to_string()]));
    });
    run.join().unwrap();
}

/// One statement with a wide tree of calls stops at the limit of its steps.
#[test]
fn a_wide_tree_of_calls_stops_at_the_step_limit() {
    let stub = torch_stub();
    let calls = "g{at}(y), ".repeat(19);
    let step = format!("def g{{next}}(y):\n    return jnp.stack([{calls}g{{at}}(y)])");
    let code = repeated("def g0(y):\n    return y", &step, 9) + "def f(x: Float[Array, \"n m\"]):\n    z = g9(x)\n";
    let measured = measure(&code, stub.path());
    assert!(measured.peak <= 64 << 20, "peak of {} bytes", measured.peak);
    assert!(measured.time <= MAX_TIME, "{:?}", measured.time);
}

/// A tree of calls over a callee with one wide statement. Each call of the
/// callee costs its body, not one step.
fn wide_callee_tree(names: usize) -> String {
    let head = format!("def g0(y):\n    return jnp.stack([{}y])", "y, ".repeat(names));
    let calls = "g{at}(y) + ".repeat(3);
    let step = format!("def g{{next}}(y):\n    return {calls}g{{at}}(y)");
    repeated(&head, &step, 7) + "def f(x: Float[Array, \"n m\"]):\n    z = g7(x)\n"
}

#[test]
fn a_call_pays_for_the_body_of_the_callee() {
    let stub = torch_stub();
    let measured = measure(&wide_callee_tree(500), stub.path());
    assert!(measured.time <= Duration::from_secs(5), "{:?}", measured.time);
    assert_eq!(measured.errors, 0);
}

/// The steps of a statement do not depend on the statements before it, so a
/// statement at the limit gives the same result in an incremental run.
#[test]
fn a_statement_at_the_step_limit_is_the_same_in_an_incremental_run() {
    let analyze_with = |code: &str, memo: &mut UnitMemo, cache: &ResolutionCache| {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        let tree = parser.parse(code, None).unwrap();
        analyze_layer_shapes_with_memo(tree.root_node(), code, &[], |_| None, 5, Some(cache), memo).unwrap()
    };
    let stub = torch_stub();
    let (mut memo, cache) = (UnitMemo::default(), new_resolution_cache());
    let code = wide_callee_tree(50);
    assert_eq!(analyze_with(&code, &mut memo, &cache), analyze(&code, stub.path()));
    for edit in ["    w = z\n", "    w = jnp.stack([x, x])\n", "    w = g7(x)\n    v = g0(x)\n"] {
        let edited = format!("{code}{edit}");
        assert_eq!(analyze_with(&edited, &mut memo, &cache), analyze(&edited, stub.path()), "{edit}");
    }
    let edited = code.replace("def g0(y):\n", "def g0(y):\n    print(y)\n");
    assert_eq!(analyze_with(&edited, &mut memo, &cache), analyze(&edited, stub.path()));
}

/// A body that fits gets its shape.
#[test]
fn a_call_under_the_step_limit_keeps_its_shape() {
    let stub = torch_stub();
    let code = format!("{LINES}def g(y):\n    return jnp.stack([y, y])\n{X}\n    z = g(g(g(x)))\n");
    let measured = measure(&code, stub.path());
    assert_eq!(measured.last, Some(["2", "2", "2", "n", "m"].map(String::from).to_vec()));
}

#[test]
#[ignore = "prints the growth of each path, run it under a time limit"]
fn growth_probe() {
    let lines: usize = std::env::var("NDIM_GROWTH_LINES").map_or(8, |text| text.parse().unwrap());
    let only = std::env::var("NDIM_GROWTH_ONLY").ok();
    let stub = torch_stub();
    for (name, code) in chains(lines) {
        if only.as_ref().is_some_and(|only| !only.split(',').any(|o| o == name)) {
            continue;
        }
        let m = measure(&code, stub.path());
        println!("{name:20} peak {:>10} one {:>10} dim {:>6} rank {:>3} {:>9.1?} {:?}", m.peak, m.largest, m.dim, m.rank, m.time, m.last);
    }
}

