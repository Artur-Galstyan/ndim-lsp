//! Broken text never makes the analysis panic: every corpus file, cut at
//! pseudo-random byte offsets and with single lines removed.

use std::{fs, panic, path::PathBuf};

use ndim_lsp::analyze_layer_shapes;
use tree_sitter::Parser;

const CUTS: usize = 200;
const DELETIONS: usize = 20;

const HOSTILE_IMPORTS: &str = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport jax.numpy as jnp\nimport einops\n";

/// Lines with an integer `{b}` at the limits of the integer types.
const HOSTILE_LINES: &[&str] = &[
    "y = x.reshape(x.shape[{b}], -1)\n    z = x.reshape(x.shape[{b}] * 2, -1)", "y = x.reshape({b}, -1)", "y = x[{b}:]", "y = x[{b}::{b}]",
    "y = x[::{b}]", "y = jnp.split(x, {b})", "y = x.chunk({b})", "a, b = x.split({b})",
    "y = jnp.arange({b}, {b})", "y = jnp.zeros(({b}, {b})) * x", "y = jnp.repeat(x, {b}, axis=0)",
    "y = jnp.tile(x, ({b}, {b}))", "y = jnp.pad(x, (({b}, {b}), (0, 0)))", "y = F.pad(x, ({b}, {b}))",
    "y = jnp.expand_dims(x, {b})", "y = jnp.sum(x, axis={b})", "y = jnp.moveaxis(x, {b}, 0)", "y = x.topk({b})",
    "y = jnp.roll(x, {b}, axis={b})", "y = jnp.linspace(0, 1, {b})", "y = jnp.eye({b}, {b})", "y = x.view({b}, {b})",
    "y = jnp.broadcast_to(x, ({b}, {b}))", "y = jnp.stack([x, x], axis={b})", "y = x.repeat({b}, {b})",
    "y = x.flatten({b}, {b})", "y = x.narrow(0, {b}, {b})", "y = einops.repeat(x, 'a b -> a b c', c={b})",
    "y = einops.rearrange(x, '(a c) b -> a c b', c={b})", "y = jnp.array_split(x, {b})", "y = jnp.split(x, [{b}, {b}])",
    "y = torch.split(x, [{b}, -1])", "y = nn.Linear({b}, {b})(x)", "y = nn.Conv2d(3, 8, {b}, stride={b}, padding={b})(x)",
    "y = F.max_pool2d(x, {b})", "y = jnp.zeros({b} * {b})", "y = jnp.ones(({b} + {b},))", "y = x.new_zeros({b} - {b})",
];

/// xorshift64 with a fixed seed, so that a failure repeats.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n.max(1) as u64) as usize
    }
}

fn analyze(text: &str) -> Result<(), String> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
    let tree = parser.parse(text, None).unwrap();
    let _ = analyze_layer_shapes(tree.root_node(), text, &[], |_| None, 5, None);
    Ok(())
}

fn variants(text: &str, rng: &mut Rng) -> Vec<String> {
    let mut out = Vec::new();
    for _ in 0..CUTS {
        let mut at = rng.below(text.len() + 1);
        while !text.is_char_boundary(at) {
            at -= 1;
        }
        out.push(text[..at].to_string());
    }
    let lines: Vec<&str> = text.lines().collect();
    for _ in 0..DELETIONS {
        let skip = rng.below(lines.len());
        let kept: Vec<&str> = lines.iter().enumerate().filter(|(i, _)| *i != skip).map(|(_, l)| *l).collect();
        out.push(kept.join("\n"));
    }
    out
}

#[test]
fn broken_corpus_files_do_not_panic() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let mut files: Vec<PathBuf> = fs::read_dir(&corpus)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "py"))
        .collect();
    files.sort();
    assert!(!files.is_empty());
    let mut inputs: Vec<(String, String)> = Vec::new();
    for big in ["9223372036854775807", "-9223372036854775808", "18446744073709551615"] {
        for line in HOSTILE_LINES {
            let line = line.replace("{b}", big);
            for imports in ["", HOSTILE_IMPORTS] {
                for dims in ["4 6", "a b"] {
                    let text = format!("{imports}def f(x: Float[Array, \"{dims}\"]):\n    {line}\n");
                    inputs.push(("snippet".to_string(), text));
                }
            }
        }
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for file in &files {
        let text = fs::read_to_string(file).unwrap();
        let name = file.file_name().unwrap().to_string_lossy().to_string();
        inputs.extend(variants(&text, &mut rng).into_iter().map(|variant| (name.clone(), variant)));
    }
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let mut failures = Vec::new();
    for (name, text) in &inputs {
        if panic::catch_unwind(|| analyze(text)).is_err() {
            failures.push(format!("{name}: {:?}", text.lines().last()));
        }
    }
    panic::set_hook(hook);
    assert!(failures.is_empty(), "{} panics:\n{}", failures.len(), failures.join("\n"));
}
