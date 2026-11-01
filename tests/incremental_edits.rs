//! Seeded edit sequences over `corpus/` and `bench/stress/`. After every edit
//! the incrementally reparsed tree, and the analysis built on it, must equal a
//! fresh parse and a fresh analysis.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndim_lsp::{
    LayerShapeAnalysis, ResolutionCache, UnitMemo, analyze_layer_shapes,
    analyze_layer_shapes_with_memo, first_analysis_difference, first_tree_difference,
    new_resolution_cache, reparse,
};
use tree_sitter::{Parser, Tree};

const SEED: u64 = 0x5eed_1abe_1ed1_7e57;

const TOKENS: [&str; 25] = [
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ":",
    ",",
    "=",
    "\n",
    "    ",
    "\t",
    "\"",
    "'",
    "\"\"\"",
    "#",
    "lambda x: ",
    "def f(x):\n    return x\n",
    "class C:\n",
    "import os\n",
    "return ",
    "@",
    "*",
    "if x:\n",
    "x = jnp.zeros((3, 4))\n",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn line_range(text: &str, at: usize) -> (usize, usize) {
    let start = text[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = text[at..].find('\n').map_or(text.len(), |i| at + i + 1);
    (start, end)
}

/// Offsets of the lines that start in column 0 with `prefix`.
fn line_starts(text: &str, prefix: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        if line.starts_with(prefix) {
            starts.push(at);
        }
        at += line.len();
    }
    starts
}

/// The lines from `start` up to the next line that starts in column 0 with
/// something other than white space.
fn block_range(text: &str, start: usize) -> (usize, usize) {
    let mut end = start;
    for (i, line) in text[start..].split_inclusive('\n').enumerate() {
        if i > 0 && !line.starts_with([' ', '\t', '\n']) {
            break;
        }
        end += line.len();
    }
    (start, end)
}

/// Offsets of the lines in column 0 that can start a top-level statement.
fn statement_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        if line.starts_with(|c: char| c.is_alphabetic() || c == '@') {
            starts.push(at);
        }
        at += line.len();
    }
    starts
}

fn pick(rng: &mut Rng, items: &[usize]) -> Option<usize> {
    (!items.is_empty()).then(|| items[rng.below(items.len())])
}

/// Append `suffix` to the name after `keyword` on the line at `start`.
fn rename_definition(text: &str, start: usize, keyword: &str, suffix: &str) -> Option<String> {
    let name_at = start + keyword.len();
    let name_len = text[name_at..].find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    let mut out = text.to_string();
    out.insert_str(name_at + name_len, suffix);
    Some(out)
}

/// A structural edit, or `None` when the text has nothing to apply it to.
fn structural_edit(rng: &mut Rng, text: &str, kind: usize) -> Option<String> {
    match kind {
        0 => {
            let at = pick(rng, &line_starts(text, "class "))?;
            rename_definition(text, at, "class ", "Renamed")
        }
        1 => {
            let at = pick(rng, &line_starts(text, "def "))?;
            rename_definition(text, at, "def ", "_renamed")
        }
        2 => {
            let (s, e) = block_range(text, pick(rng, &line_starts(text, "def "))?);
            let block = text[s..e].to_string();
            let mut out = text.to_string();
            out.replace_range(s..e, "");
            let target = pick(rng, &statement_starts(&out))?;
            out.insert_str(target, &block);
            Some(out)
        }
        3 => {
            let target = pick(rng, &statement_starts(text))?;
            let mut out = text.to_string();
            out.insert_str(
                target,
                &format!("def added_{}(x):\n    return x\n\n\n", rng.below(1000)),
            );
            Some(out)
        }
        4 => {
            let (s, e) = block_range(text, pick(rng, &line_starts(text, "def "))?);
            let mut out = text.to_string();
            out.replace_range(s..e, "");
            Some(out)
        }
        _ => {
            let mut imports = line_starts(text, "import ");
            imports.extend(line_starts(text, "from "));
            let at = pick(rng, &imports)?;
            let end = text[at..].find('\n').map_or(text.len(), |i| at + i);
            let line = &text[at..end];
            let edited = match line.rfind(" as ") {
                Some(i) => format!("{}_2", &line[..i + 4 + line[i + 4..].trim_end().len()]),
                None if line.contains('(') || line.ends_with(['\\', ',']) => return None,
                None => format!("{line} as aliased"),
            };
            let mut out = text.to_string();
            out.replace_range(at..end, &edited);
            Some(out)
        }
    }
}

/// One random edit of `text`. `prev` is the text before the last edit, for
/// the revert case.
fn mutate(rng: &mut Rng, text: &str, prev: &str, paste: &str) -> String {
    let at = floor_boundary(text, rng.below(text.len() + 1));
    let mut out = text.to_string();
    match rng.below(14) {
        0..=3 => out.insert_str(at, TOKENS[rng.below(TOKENS.len())]),
        4 => {
            if let Some(c) = text[at..].chars().next() {
                out.replace_range(at..at + c.len_utf8(), "");
            }
        }
        5 => {
            let (s, e) = line_range(text, at);
            out.replace_range(s..e, "");
        }
        6 => {
            let (s, e) = line_range(text, at);
            let mut line = text[s..e].to_string();
            if !line.ends_with('\n') {
                line.insert(0, '\n');
            }
            out.insert_str(e, &line);
        }
        7 => {
            if rng.below(2) == 0 {
                let from = floor_boundary(paste, rng.below(paste.len().saturating_sub(300) + 1));
                let to = floor_boundary(paste, from + 300);
                out.insert_str(at, &paste[from..to]);
            } else {
                return prev.to_string();
            }
        }
        kind => {
            if let Some(edited) = structural_edit(rng, text, kind - 8) {
                return edited;
            }
            out.insert_str(at, TOKENS[rng.below(TOKENS.len())]);
        }
    }
    out
}

fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
    entries.sort_by_key(|e| e.path());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            python_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "py") {
            out.push(p);
        }
    }
}

fn test_files() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    python_files(&root.join("corpus"), &mut paths);
    python_files(&root.join("bench/stress"), &mut paths);
    paths
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p, text)
        })
        .collect()
}

fn new_parser() -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .unwrap();
    parser
}

fn assert_same_tree(label: &str, step: usize, incremental: &Tree, fresh: &Tree) {
    if let Some(diff) = first_tree_difference(incremental, fresh) {
        panic!("{label} step {step}: tree differs at {diff}");
    }
}

fn assert_same_analysis(label: &str, step: usize, a: &LayerShapeAnalysis, b: &LayerShapeAnalysis) {
    if let Some(part) = first_analysis_difference(a, b) {
        panic!("{label} step {step}: {part} differ");
    }
}

fn fresh_analysis(path: &Path, text: &str, tree: &Tree) -> LayerShapeAnalysis {
    let roots = vec![path.parent().unwrap().to_path_buf()];
    analyze_layer_shapes(
        tree.root_node(),
        text,
        &roots,
        |p| std::fs::read_to_string(p).ok(),
        8,
        None,
    )
    .unwrap()
}

/// Run `steps_for(len)` chained edits on each file. `path` is the incremental
/// analysis path under test: it gets the new text and the incrementally
/// reparsed tree, and must return what a fresh analysis returns.
fn run_sequences(
    steps_for: impl Fn(usize) -> usize,
    mut path: impl FnMut(&Path, &str, &Tree) -> LayerShapeAnalysis,
    check_analysis: bool,
) -> usize {
    let paste = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/stress/paste_block.txt"),
    )
    .unwrap();
    let mut parser = new_parser();
    let mut total = 0;
    for (n, (file, original)) in test_files().into_iter().enumerate() {
        let label = file.file_name().unwrap().to_string_lossy().to_string();
        let mut rng = Rng(SEED ^ (n as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut old_text = original.clone();
        let mut prev = original;
        let mut tree = parser.parse(&old_text, None).unwrap();
        for step in 0..steps_for(old_text.len()) {
            let new_text = mutate(&mut rng, &old_text, &prev, &paste);
            let incremental = reparse(&mut parser, &old_text, &tree, &new_text).unwrap();
            let fresh = parser.parse(&new_text, None).unwrap();
            assert_same_tree(&label, step, &incremental, &fresh);
            if check_analysis {
                // The analysis of the tree in hand, with and without the memo.
                let got = path(&file, &new_text, &incremental);
                let want = fresh_analysis(&file, &new_text, &incremental);
                assert_same_analysis(&label, step, &got, &want);
                let fresh_want = fresh_analysis(&file, &new_text, &fresh);
                assert_same_analysis(&label, step, &got, &fresh_want);
            }
            prev = std::mem::replace(&mut old_text, new_text);
            tree = incremental;
            total += 1;
        }
    }
    total
}

#[test]
fn incremental_parse_equals_fresh_parse() {
    let steps = run_sequences(
        |len| {
            if len > 200_000 {
                3
            } else if len > 20_000 {
                20
            } else {
                300
            }
        },
        |_, _, _| unreachable!(),
        false,
    );
    assert!(steps > 3000, "only {steps} steps");
}

/// The analysis path under test: a memo and a resolution cache per file,
/// kept from step to step like the server keeps them.
fn memo_path() -> impl FnMut(&Path, &str, &Tree) -> LayerShapeAnalysis {
    let cache = new_resolution_cache();
    let mut memos: HashMap<PathBuf, UnitMemo> = HashMap::new();
    move |file, text, tree| {
        let roots = vec![file.parent().unwrap().to_path_buf()];
        analyze_layer_shapes_with_memo(
            tree.root_node(),
            text,
            &roots,
            |p| std::fs::read_to_string(p).ok(),
            8,
            Some(&cache),
            memos.entry(file.to_path_buf()).or_default(),
        )
        .unwrap()
    }
}

#[test]
fn incremental_analysis_equals_fresh_analysis() {
    run_sequences(
        |len| {
            if len > 200_000 {
                2
            } else if len > 20_000 {
                4
            } else {
                40
            }
        },
        memo_path(),
        true,
    );
}

const HELPER_A: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const HELPER_B: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"k d\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const HELPER_SAME_SIZE: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d j\"]) -> Float[Array, \"b j\"]:\n    return x @ w\n";
const HELPER_MAIN: &str = "import torch.nn as nn\nfrom jaxtyping import Float, Array\nfrom helper import proj\n\nclass M:\n    def __init__(self):\n        self.f = nn.Linear(3, 4)\n\ndef f(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]):\n    y = proj(x, w)\n    return y\n\ndef g(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]):\n    z = proj(x, w)\n    return z\n";

/// A helper saved through the server's path (`revalidate` with the saved
/// file), between edits of the importing file.
#[test]
fn incremental_analysis_survives_helper_saves() {
    let tmp = tempfile::tempdir().unwrap();
    let helper = tmp.path().join("helper.py");
    let main = tmp.path().join("main.py");
    std::fs::write(&helper, HELPER_A).unwrap();
    let roots = vec![tmp.path().to_path_buf()];
    let cache: Arc<ResolutionCache> = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let mut parser = new_parser();
    let paste = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/stress/paste_block.txt"),
    )
    .unwrap();
    let mut rng = Rng(SEED ^ 0xfeed);
    let (mut text, mut prev) = (HELPER_MAIN.to_string(), HELPER_MAIN.to_string());
    let mut tree = parser.parse(&text, None).unwrap();
    let mut hits = 0;
    let mut kept = std::collections::HashSet::new();
    for step in 0..120 {
        match rng.below(4) {
            0 => {
                let content = [HELPER_A, HELPER_B, HELPER_SAME_SIZE][rng.below(3)];
                let before = std::fs::metadata(&helper).unwrap().modified().unwrap();
                std::fs::write(&helper, content).unwrap();
                // Keep the old time, so only the editor event shows the save.
                // The cache tells two events apart by the file's time and
                // size, so each pair is kept once.
                if rng.below(2) == 0 && !kept.contains(&(before, content.len())) {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&helper)
                        .unwrap()
                        .set_modified(before)
                        .unwrap();
                }
                let after = std::fs::metadata(&helper).unwrap();
                kept.insert((after.modified().unwrap(), content.len()));
                cache.revalidate(std::slice::from_ref(&helper));
            }
            _ => {
                let edited = mutate(&mut rng, &text, &prev, &paste);
                tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
                prev = std::mem::replace(&mut text, edited);
            }
        }
        std::fs::write(&main, &text).unwrap();
        let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        hits += memo.stats().layer_hits;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        assert_same_analysis("helper", step, &got, &want);
    }
    assert!(hits > 0);
}

/// A digit run and the line it is on, picked by position.
fn number_at(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let start = (from..bytes.len()).find(|&i| {
        bytes[i].is_ascii_digit() && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
    })?;
    let end = (start..bytes.len())
        .find(|&i| !bytes[i].is_ascii_digit())
        .unwrap_or(bytes.len());
    Some((start, end))
}

/// An edit that keeps the text well formed, so that the analysis has units
/// to replay. `None` when the text has nothing to apply it to.
fn semantic_edit(rng: &mut Rng, text: &str, prev: &str) -> Option<String> {
    let at = floor_boundary(text, rng.below(text.len() + 1));
    let mut out = text.to_string();
    match rng.below(11) {
        0 | 1 => {
            let (s, e) = number_at(text, at)?;
            let n: u32 = text[s..e].parse().unwrap_or(1);
            out.replace_range(s..e, &((n + 1 + rng.below(3) as u32) % 997).to_string());
        }
        2 => {
            let starts = statement_starts(text);
            let k = rng.below(starts.len().saturating_sub(2));
            let (a, b, c) = (*starts.get(k)?, *starts.get(k + 1)?, *starts.get(k + 2)?);
            out.replace_range(a..c, &format!("{}{}", &text[b..c], &text[a..b]));
        }
        3 => {
            let (s, e) = line_range(text, at);
            if !text[s..e].starts_with("    ") || text[s..e].trim_start().starts_with("def ") || text[s..e].trim_start().starts_with("class ") {
                return None;
            }
            out.replace_range(s..e, "");
        }
        4 => {
            let (s, e) = block_range(text, pick(rng, &line_starts(text, "def "))?);
            let block = &text[s..e];
            let name_end = block[4..].find('(')? + 4;
            let copy = format!("{}_copy{}", &block[..name_end], &block[name_end..]);
            out.insert_str(e, &format!("\n\n{copy}"));
        }
        5 => {
            let (s, e) = block_range(text, pick(rng, &line_starts(text, "def "))?);
            out.replace_range(s..e, "");
        }
        6 => {
            let start = pick(rng, &line_starts(text, "def "))?;
            let name_at = start + 4;
            let len = text[name_at..].find('(')?;
            let name = text[name_at..name_at + len].to_string();
            if name.starts_with("__") {
                return None;
            }
            out = text.replace(&name, &format!("{name}_r"));
        }
        7 => {
            let (s, e) = line_range(text, at);
            if text[s..e].trim().is_empty() || text[s..e].contains(['"', '\'', '\\']) {
                return None;
            }
            let end = e - usize::from(text[..e].ends_with('\n'));
            out.insert_str(end, "  # note");
        }
        8 => {
            let (s, e) = line_range(text, at);
            let line = &text[s..e];
            let swapped = if line.contains(" @ ") {
                line.replacen(" @ ", " * ", 1)
            } else if line.contains(" + ") {
                line.replacen(" + ", " @ ", 1)
            } else {
                return None;
            };
            out.replace_range(s..e, &swapped);
        }
        9 => {
            let target = pick(rng, &statement_starts(text))?;
            out.insert_str(
                target,
                &format!("K_{} = jnp.zeros(({}, 4))\n\n\n", rng.below(50), rng.below(9) + 1),
            );
        }
        _ => return Some(prev.to_string()),
    }
    Some(out)
}

#[derive(Default)]
struct ReplayTotals {
    steps: usize,
    full: usize,
    units: usize,
    replayed: usize,
}

/// `steps` chained semantic edits of `file`. After each, the analysis through
/// the memo must equal the fresh analysis of the same tree.
fn run_semantic(file: &Path, original: String, steps: usize, seed: u64) -> ReplayTotals {
    let paste = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/stress/paste_block.txt"),
    )
    .unwrap();
    let mut parser = new_parser();
    let mut rng = Rng(seed);
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![file.parent().unwrap().to_path_buf()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let reset = original.clone();
    let (mut text, mut prev) = (original.clone(), original);
    let mut tree = parser.parse(&text, None).unwrap();
    let mut totals = ReplayTotals::default();
    let label = file.file_name().unwrap().to_string_lossy().to_string();
    for step in 0..steps {
        // Random edits pile up damage that slows every later step: start over
        // from the original now and then, through the same memo.
        let edited = if step % 50 == 49 {
            reset.clone()
        } else if rng.below(5) == 0 {
            mutate(&mut rng, &text, &prev, &paste)
        } else {
            semantic_edit(&mut rng, &text, &prev)
                .unwrap_or_else(|| mutate(&mut rng, &text, &prev, &paste))
        };
        let t0 = std::time::Instant::now();
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        prev = std::mem::replace(&mut text, edited);
        let t1 = t0.elapsed();
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        let t2 = t0.elapsed();
        let want_cache = new_resolution_cache();
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, Some(&want_cache)).unwrap();
        let t3 = t0.elapsed();
        if std::env::var_os("NDIM_CHAIN_TIMES").is_some() {
            let broken = tree.root_node().children(&mut tree.walk()).filter(|n| n.has_error()).map(|n| n.byte_range().len()).sum::<usize>();
            let s = memo.stats();
            eprintln!("step {step}: reparse {t1:?} memo {:?} fresh {:?} units {} replayed {} full {} error bytes {broken}/{}", t2 - t1, t3 - t2, s.units, s.replayed_units, s.propagate_full, text.len());
        }
        if let Some(part) = first_analysis_difference(&got, &want) {
            std::fs::write(std::env::temp_dir().join("ndim-failing-step.py"), &text).ok();
            panic!("{label} step {step}: {part} differ");
        }
        let stats = memo.stats();
        totals.steps += 1;
        totals.full += usize::from(stats.propagate_full);
        totals.units += stats.units;
        totals.replayed += stats.replayed_units;
    }
    totals
}

#[test]
fn replayed_analysis_equals_fresh_analysis_after_semantic_edits() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut totals = ReplayTotals::default();
    for (n, name) in ["corpus/mamba_ssm.py", "corpus/lax_scan_ema.py", "test_python.py"]
        .iter()
        .enumerate()
    {
        let file = root.join(name);
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let t = run_semantic(&file, text, 80, SEED ^ (n as u64 + 7));
        totals.steps += t.steps;
        totals.full += t.full;
        totals.units += t.units;
        totals.replayed += t.replayed;
    }
    println!("{} steps, {} full, {}/{} units replayed", totals.steps, totals.full, totals.replayed, totals.units);
    assert!(totals.replayed > 0, "no step replayed anything");
}

/// 2000 chained edits of the biggest file. Run in release:
/// `cargo test --release --test incremental_edits stress_main_two_thousand -- --ignored --nocapture`
#[test]
#[ignore]
fn stress_main_two_thousand_chained_edits() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/stress/stress_main.py");
    let text = std::fs::read_to_string(&file).unwrap();
    let steps = std::env::var("NDIM_CHAIN_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let seed = std::env::var("NDIM_CHAIN_SEED").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0x57e55);
    let t = run_semantic(&file, text, steps, SEED ^ seed);
    let line = format!(
        "{} steps, {} full, {}/{} units replayed",
        t.steps, t.full, t.replayed, t.units
    );
    println!("{line}");
    std::fs::write(std::env::temp_dir().join("ndim-chain-stats.txt"), line).ok();
    assert!(t.replayed > 0);
}

#[test]
fn reparse_after_a_syntax_error_equals_a_fresh_parse() {
    let base = "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\ndef f(x: Float[Array, \"3 4\"]):\n    y = x.T\n    z = y @ x\n    return z\n\ndef g(a: Float[Array, \"2 2\"]):\n    b = a + a\n    return b\n";
    let mut parser = new_parser();
    let mut text = base.to_string();
    let mut tree = parser.parse(&text, None).unwrap();
    for at in [148, 176] {
        let mut new = text.clone();
        new.insert_str(at, "'''");
        tree = reparse(&mut parser, &text, &tree, &new).unwrap();
        text = new;
        let fresh = parser.parse(&text, None).unwrap();
        if let Some(diff) = first_tree_difference(&tree, &fresh) {
            panic!("edit at {at}: tree differs at {diff}");
        }
    }
}

const BRANCHES: &str = r#"import jax.numpy as jnp
from jaxtyping import Float, Array


def pick(x: Float[Array, "3 4"], c):
    if c:
        r = x
    else:
        r = jnp.ones((3, 4))
    return r


def early(x: Float[Array, "3 4"], c):
    if c:
        x = jnp.zeros(5)
        return x
    q = x
    return q


def loops(x: Float[Array, "3 4"], n):
    for i in range(n):
        t = jnp.zeros(5)
        if i:
            continue
        x = jnp.ones((3, 4))
    return x


def guarded(x: Float[Array, "3 4"], c):
    try:
        a = jnp.zeros(5)
    except ValueError:
        a = jnp.zeros(5)
    finally:
        b = a
    return a


def caller(x: Float[Array, "3 4"]):
    p = pick(x, True)
    e = early(x, False)
    l = loops(x, 3)
    g = guarded(x, 1)
    return p, e, l, g


n = jnp.zeros(3)
if flag:
    m = jnp.zeros(6)
else:
    n = jnp.zeros(3)
k = m
"#;

/// Each pair swaps one fragment of `BRANCHES` for another that changes what a
/// branch, a loop, or a join depends on: the shape of an arm, whether an arm
/// leaves, whether the `else` clause exists, a mention of a name before the
/// branch (in the function, or in an earlier statement of the module), a
/// `global` statement, a wildcard import. A step applies an edit or undoes one.
const BRANCH_EDITS: [(&str, &str); 22] = [
    ("        r = jnp.ones((3, 4))", "        r = jnp.ones((3, 5))"),
    ("        x = jnp.zeros(5)\n        return x", "        x = jnp.zeros(5)\n        x = x[None]"),
    ("    else:\n        r = jnp.ones((3, 4))", "    elif c > 1:\n        r = jnp.ones((3, 4))"),
    ("    for i in range(n):\n        t = jnp.zeros(5)", "    while n:\n        t = jnp.zeros(5)"),
    ("            continue\n", "            break\n"),
    ("    except ValueError:\n        a = jnp.zeros(5)", "    except ValueError:\n        a = jnp.zeros(6)"),
    ("    finally:\n        b = a", "    finally:\n        a = jnp.ones(7)"),
    (
        "def pick(x: Float[Array, \"3 4\"], c):\n    if c:",
        "def pick(x: Float[Array, \"3 4\"], c):\n    r = mystery()\n    if c:",
    ),
    (
        "def early(x: Float[Array, \"3 4\"], c):\n    if c:",
        "def early(x: Float[Array, \"3 4\"], c):\n    q = mystery()\n    if c:",
    ),
    ("n = jnp.zeros(3)\nif flag:", "print(m)\nn = jnp.zeros(3)\nif flag:"),
    ("n = jnp.zeros(3)\nif flag:", "dict(m=1)\nn = jnp.zeros(3)\nif flag:"),
    ("n = jnp.zeros(3)\nif flag:", "obj.m\nn = jnp.zeros(3)\nif flag:"),
    ("import jax.numpy as jnp\n", "import jax.numpy as jnp\nfrom lib import *\n"),
    (
        "def caller(x: Float[Array, \"3 4\"]):",
        "def setter():\n    global m\n    m = 1\n\n\ndef caller(x: Float[Array, \"3 4\"]):",
    ),
    ("    p = pick(x, True)", "    p = pick(x, False)"),
    ("else:\n    n = jnp.zeros(3)\nk = m", "else:\n    n = jnp.zeros(4)\nk = m"),
    (
        "    try:\n        a = jnp.zeros(5)",
        "    with suppress(E):\n        a = jnp.zeros(5)\n    try:\n        a = jnp.zeros(5)",
    ),
    ("        t = jnp.zeros(5)\n        if i:", "        t = x\n        if i:"),
    ("    q = x\n    return q", "    match c:\n        case [x]:\n            pass\n    q = x\n    return q"),
    (
        "def pick(x: Float[Array, \"3 4\"], c):\n    if c:",
        "def pick(x: Float[Array, \"3 4\"], c):\n    if (x := mystery()):\n        pass\n    if c:",
    ),
    (
        "def caller(x: Float[Array, \"3 4\"]):",
        "def helper(v):\n    return v\n\n\ndef helper(v):\n    return v.T\n\n\ndef caller(x: Float[Array, \"3 4\"]):",
    ),
    ("    e = early(x, False)", "    e = early(x, False)\n    h = helper(x)"),
];

#[test]
fn replayed_analysis_equals_fresh_analysis_after_branch_edits() {
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut rng = Rng(SEED ^ 0xb4a9c8);
    // Independent statements, so that an edit dirties a small share of the file
    // and the plan replays the rest instead of running everything.
    let filler: String = (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect();
    let mut text = format!("{BRANCHES}{filler}");
    let mut tree = parser.parse(&text, None).unwrap();
    let mut replayed = 0;
    for step in 0..400 {
        let (from, to) = BRANCH_EDITS[rng.below(BRANCH_EDITS.len())];
        let edited = if text.contains(to) && rng.below(2) == 0 {
            text.replacen(to, from, 1)
        } else if text.contains(from) {
            text.replacen(from, to, 1)
        } else {
            text.clone()
        };
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("branch edits step {step}: {part} differ\n{text}");
        }
        replayed += memo.stats().replayed_units;
    }
    assert!(replayed > 0, "no step replayed a unit");
}

/// A module-level branch assigns `m` in one arm. Whether `m` keeps its shape
/// after the branch depends on the statements before it: a keyword or an
/// attribute named `m` does not bind a module name (the top-level name model
/// does not link them), a read, a `global m` in a function, or a wildcard
/// import does. Each edit must give the same analysis from the memo as from a
/// fresh run, and `k = m` must show the answer.
#[test]
fn module_level_join_follows_the_mentions_before_it() {
    let filler: String = (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect();
    let base = format!(
        "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\n\nn = jnp.zeros(3)\nif flag:\n    m = jnp.zeros(6)\nelse:\n    n = jnp.zeros(3)\nk = m\n{filler}"
    );
    let at = "n = jnp.zeros(3)\nif flag:";
    let steps: [(&str, &str, bool); 20] = [
        (at, "dict(m=1)\nn = jnp.zeros(3)\nif flag:", true),
        (at, "obj.m\nn = jnp.zeros(3)\nif flag:", true),
        (at, "print(m)\nn = jnp.zeros(3)\nif flag:", false),
        ("print(m)\n", "", true),
        ("dict(m=1)\nobj.m\n", "", true),
        ("\n\n\nn = jnp.zeros(3)", "\n\n\ndef setter():\n    global m\n    m = 1\n\n\nn = jnp.zeros(3)", false),
        ("\n\n\ndef setter():\n    global m\n    m = 1\n", "\n", true),
        ("import jax.numpy as jnp\n", "import jax.numpy as jnp\nfrom lib import *\n", false),
        ("from lib import *\n", "", true),
        ("    m = jnp.zeros(6)\nelse:", "    m = jnp.zeros(6)\n    n = jnp.zeros(3)\nelse:", true),
        ("else:\n    n = jnp.zeros(3)\nk = m", "else:\n    m = jnp.zeros(6)\nk = m", true),
        ("else:\n    m = jnp.zeros(6)\nk = m", "else:\n    m = jnp.zeros(7)\nk = m", false),
        ("    m = jnp.zeros(7)\nk = m", "    m = jnp.zeros(6)\nk = m", true),
        // A `def`, a `class`, or an import of the name is an earlier binding.
        ("else:\n    m = jnp.zeros(6)\nk = m", "else:\n    n = jnp.zeros(3)\nk = m", true),
        (at, "def m():\n    pass\nn = jnp.zeros(3)\nif flag:", false),
        ("def m():\n    pass\n", "class m:\n    pass\n", false),
        ("class m:\n    pass\n", "import os as m\n", false),
        ("import os as m\n", "", true),
        (at, "def other():\n    pass\nn = jnp.zeros(3)\nif flag:", true),
        ("def other():\n    pass\n", "def m():\n    pass\n", false),
    ];
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = base.clone();
    let mut tree = parser.parse(&text, None).unwrap();
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    let known = |a: &LayerShapeAnalysis| a.scopes[0].shapes.contains_key("k");
    assert!(known(&run(&tree, &text, &mut memo)));
    let mut replayed = 0;
    for (step, (from, to, keeps)) in steps.iter().enumerate() {
        assert!(text.contains(from), "step {step}");
        let edited = text.replacen(from, to, 1);
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("step {step}: {part} differ");
        }
        assert_eq!(known(&want), *keeps, "step {step}");
    }
    assert!(replayed > 0);
}

/// A function reads a module global `W` and a module constant `D`. Whether
/// the reader sees a shape depends on statements elsewhere in the file: a
/// second write of `W`, a write after the reader, a `global W`, a class body
/// that binds `W`, a branch that gives `W` two shapes; and for `D`, another
/// literal, a parameter named `D`. Each edit must give the same analysis from
/// the memo as from a fresh run, and `out` and `z` must show the answer.
#[test]
fn module_global_follows_the_statements_around_it() {
    let filler: String = (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect();
    let base = format!(
        "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\nW = jnp.zeros((3, 4))\nD = 64\n\n\ndef use(x: Float[Array, \"3 4\"]):\n    out = x + W\n    return out\n\n\ndef dims():\n    z = jnp.zeros((D, 2))\n    return z\n{filler}"
    );
    let dims = |d: &[&str]| Some(d.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let w = dims(&["3", "4"]);
    let none: Option<Vec<String>> = None;
    let at_end = "\n\nW = jnp.zeros((3, 4))\n";
    type Dims = Option<Vec<String>>;
    let steps: Vec<(&str, String, Dims, Dims)> = vec![
        ("D = 64\n", "D = 32\n".into(), w.clone(), dims(&["32", "2"])),
        ("D = 32\n", "D = 64\n".into(), w.clone(), dims(&["64", "2"])),
        ("\n\ndef dims():", "\n\nW = jnp.zeros((3, 4))\n\n\ndef dims():".into(), none.clone(), dims(&["64", "2"])),
        ("\n\nW = jnp.zeros((3, 4))\n\n\ndef dims():", "\n\ndef dims():".into(), w.clone(), dims(&["64", "2"])),
        (
            "\n\ndef dims():",
            "\n\ndef setter():\n    global W\n    W = 1\n\n\ndef dims():".into(),
            none.clone(),
            dims(&["64", "2"]),
        ),
        ("\n\ndef setter():\n    global W\n    W = 1\n\n\ndef dims():", "\n\ndef dims():".into(), w.clone(), dims(&["64", "2"])),
        ("\n\ndef use(", "\n\nclass K:\n    W = jnp.zeros(9)\n\n\ndef use(".into(), none.clone(), dims(&["64", "2"])),
        ("\n\nclass K:\n    W = jnp.zeros(9)\n\n\ndef use(", "\n\ndef use(".into(), w.clone(), dims(&["64", "2"])),
        ("D = 64\n", "D = 64\nD = 32\n".into(), w.clone(), dims(&["D", "2"])),
        ("D = 64\nD = 32\n", "D = 64\n".into(), w.clone(), dims(&["64", "2"])),
        ("def dims():", "def dims(D):".into(), w.clone(), dims(&["D", "2"])),
        ("def dims(D):", "def dims():".into(), w.clone(), dims(&["64", "2"])),
        (
            "W = jnp.zeros((3, 4))\nD",
            "if flag:\n    W = jnp.zeros((3, 4))\nelse:\n    W = jnp.zeros((3, 6))\nD".into(),
            none.clone(),
            dims(&["64", "2"]),
        ),
        (
            "if flag:\n    W = jnp.zeros((3, 4))\nelse:\n    W = jnp.zeros((3, 6))\nD",
            "W = jnp.zeros((3, 4))\nD".into(),
            w.clone(),
            dims(&["64", "2"]),
        ),
        ("W = jnp.zeros((3, 4))\nD", "D".into(), none.clone(), dims(&["64", "2"])),
        ("\n\ndef dims():", format!("{at_end}\n\ndef dims():"), none.clone(), dims(&["64", "2"])),
        (at_end, "\n\n".into(), none.clone(), dims(&["64", "2"])),
        ("D = 64\n", "W = jnp.zeros((3, 4))\nD = 64\n".into(), w.clone(), dims(&["64", "2"])),
    ];
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = base;
    let mut tree = parser.parse(&text, None).unwrap();
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    let find = |a: &LayerShapeAnalysis, name: &str| {
        a.scopes.iter().find_map(|s| s.shapes.get(name).cloned())
    };
    let first = run(&tree, &text, &mut memo);
    assert_eq!(find(&first, "out"), w);
    assert_eq!(find(&first, "z"), dims(&["64", "2"]));
    let mut replayed = 0;
    for (step, (from, to, out, z)) in steps.iter().enumerate() {
        assert!(text.contains(from), "step {step}");
        let edited = text.replacen(from, to, 1);
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("step {step}: {part} differ\n{text}");
        }
        assert_eq!(&find(&want, "out"), out, "step {step} out\n{text}");
        assert_eq!(&find(&want, "z"), z, "step {step} z\n{text}");
    }
    assert!(replayed > 0);
}

/// Lambdas and comprehensions read module globals (`XS`, `V`), a module
/// constant (`K`), and the builtins `sum` and `range`, all defined in other
/// statements. Their bodies bind names of their own that hide those.
const COMPREHENSIONS: &str = r#"import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

XS = jnp.zeros((5, 3, 4))
V = jnp.zeros((4, 2))
K = 3


def stacked(w: Float[Array, "4 2"]):
    out = jnp.stack([x @ w for x in XS])
    return out


def mapped(w: Float[Array, "4 2"]):
    m = jax.vmap(lambda x: x @ w)(XS)
    return m


def counted():
    c = jnp.stack([jnp.zeros((2,)) for _ in range(K)])
    return c


def total(xs: Float[Array, "5 3"]):
    s = sum(x * 2.0 for x in xs)
    return s


def named(y: Float[Array, "3 4"]):
    r = (lambda x: V)(y)
    return r


def hidden():
    q = jnp.stack([V for V in range(2)])
    return q
"#;

/// Each pair swaps one fragment of `COMPREHENSIONS` for another that changes
/// what a body reads: the shape of a global, a second write of it, the count of
/// a `range`, a name that hides a global (a parameter, a loop target), a
/// definition of `sum` or `range`, a `global` statement, a wildcard import.
const COMPREHENSION_EDITS: [(&str, &str); 17] = [
    ("XS = jnp.zeros((5, 3, 4))", "XS = jnp.zeros((6, 3, 4))"),
    ("XS = jnp.zeros((5, 3, 4))\n", "XS = jnp.zeros((5, 3, 4))\nXS = jnp.zeros((7, 3, 4))\n"),
    ("V = jnp.zeros((4, 2))", "V = jnp.zeros((4, 3))"),
    ("K = 3\n", "K = 4\n"),
    ("K = 3\n", "K = 3\nK = 5\n"),
    ("lambda x: V", "lambda V: V"),
    ("(lambda x: V)(y)", "(lambda x, V=1: V)(y)"),
    ("for V in range(2)", "for _ in range(2)"),
    ("for _ in range(K)", "for _ in range(2, K)"),
    ("lambda x: x @ w", "lambda w: x @ w"),
    ("\n\ndef total(", "\n\nsum = max\n\n\ndef total("),
    ("\n\ndef counted(", "\n\nrange = list\n\n\ndef counted("),
    ("\n\ndef counted(", "\n\ndef setter():\n    global XS\n    XS = 1\n\n\ndef counted("),
    ("import jax\n", "import jax\nfrom lib import *\n"),
    ("import jax\n", "import jax\nfrom numpy import sum\n"),
    ("def total(xs: Float[Array, \"5 3\"]):", "def total(xs: Float[Array, \"5 3\"], sum=0):"),
    ("for x in xs)", "for x in xs if x)"),
];

fn comprehension_filler() -> String {
    (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect()
}

#[test]
fn replayed_analysis_equals_fresh_analysis_after_comprehension_edits() {
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut rng = Rng(SEED ^ 0xc0a7e1);
    let mut text = format!("{COMPREHENSIONS}{}", comprehension_filler());
    let mut tree = parser.parse(&text, None).unwrap();
    let mut replayed = 0;
    for step in 0..400 {
        let (from, to) = COMPREHENSION_EDITS[rng.below(COMPREHENSION_EDITS.len())];
        let edited = if text.contains(to) && rng.below(2) == 0 {
            text.replacen(to, from, 1)
        } else if text.contains(from) {
            text.replacen(from, to, 1)
        } else {
            text.clone()
        };
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("comprehension edits step {step}: {part} differ\n{text}");
        }
        replayed += memo.stats().replayed_units;
    }
    assert!(replayed > 0, "no step replayed a unit");
}

/// Walk `COMPREHENSIONS` through edits that change what each body reads. Each
/// step must give the same analysis from the memo as from a fresh run, and the
/// names `out`, `m`, `c`, `s`, `r`, `q` must show the answer that Python gives
/// (unknown where the file does not settle it).
#[test]
fn comprehension_and_lambda_follow_the_statements_around_them() {
    let d = |dims: &[&str]| Some(dims.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    type Dims = Option<Vec<String>>;
    // out, m, c, s, r, q
    type Answer = [Dims; 6];
    let start: Answer = [
        d(&["5", "3", "2"]),
        d(&["5", "3", "2"]),
        d(&["3", "2"]),
        d(&["3"]),
        d(&["4", "2"]),
        None,
    ];
    let none: Dims = None;
    let steps: Vec<(&str, &str, Answer)> = vec![
        (
            "XS = jnp.zeros((5, 3, 4))",
            "XS = jnp.zeros((6, 3, 4))",
            [d(&["6", "3", "2"]), d(&["6", "3", "2"]), d(&["3", "2"]), d(&["3"]), d(&["4", "2"]), None],
        ),
        (
            "XS = jnp.zeros((6, 3, 4))\n",
            "XS = jnp.zeros((6, 3, 4))\nXS = jnp.zeros((7, 3, 4))\n",
            [none.clone(), none.clone(), d(&["3", "2"]), d(&["3"]), d(&["4", "2"]), None],
        ),
        (
            "XS = jnp.zeros((6, 3, 4))\nXS = jnp.zeros((7, 3, 4))\n",
            "XS = jnp.zeros((5, 3, 4))\n",
            start.clone(),
        ),
        (
            "lambda x: V",
            "lambda V: V",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["3", "2"]), d(&["3"]), d(&["3", "4"]), None],
        ),
        (
            "(lambda V: V)(y)",
            "(lambda V: V)(mystery())",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["3", "2"]), d(&["3"]), None, None],
        ),
        ("(lambda V: V)(mystery())", "(lambda x: V)(y)", start.clone()),
        (
            "for V in range(2)",
            "for _ in range(2)",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["3", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "V = jnp.zeros((4, 2))",
            "V = jnp.zeros((4, 3))",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["3", "2"]), d(&["3"]), d(&["4", "3"]), d(&["2", "4", "3"])],
        ),
        (
            "V = jnp.zeros((4, 3))",
            "V = jnp.zeros((4, 2))",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["3", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "for _ in range(K)",
            "for _ in range(2, K)",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["1", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "K = 3\n",
            "K = 3\nK = 5\n",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), none.clone(), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "K = 3\nK = 5\n",
            "K = 4\n",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "\n\ndef total(",
            "\n\nsum = max\n\n\ndef total(",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), none.clone(), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "\n\nsum = max\n\n\ndef total(",
            "\n\ndef total(",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "\n\ndef counted(",
            "\n\nrange = list\n\n\ndef counted(",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), none.clone(), d(&["3"]), d(&["4", "2"]), none.clone()],
        ),
        (
            "\n\nrange = list\n\n\ndef counted(",
            "\n\ndef setter():\n    global XS\n    XS = 1\n\n\ndef counted(",
            [none.clone(), none.clone(), d(&["2", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "\n\ndef setter():\n    global XS\n    XS = 1\n\n\ndef counted(",
            "\n\ndef counted(",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "import jax\n",
            "import jax\nfrom lib import *\n",
            [none.clone(), none.clone(), none.clone(), none.clone(), none.clone(), none.clone()],
        ),
        (
            "import jax\nfrom lib import *\n",
            "import jax\n",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), d(&["3"]), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
        (
            "for x in xs)",
            "for x in xs if x)",
            [d(&["5", "3", "2"]), d(&["5", "3", "2"]), d(&["2", "2"]), none.clone(), d(&["4", "2"]), d(&["2", "4", "2"])],
        ),
    ];
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = format!("{COMPREHENSIONS}{}", comprehension_filler());
    let mut tree = parser.parse(&text, None).unwrap();
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    let find = |a: &LayerShapeAnalysis, name: &str| {
        a.scopes.iter().find_map(|s| s.shapes.get(name).cloned())
    };
    let names = ["out", "m", "c", "s", "r", "q"];
    let first = run(&tree, &text, &mut memo);
    for (name, want) in names.iter().zip(&start) {
        assert_eq!(&find(&first, name), want, "start {name}");
    }
    let mut replayed = 0;
    for (step, (from, to, answer)) in steps.iter().enumerate() {
        assert!(text.contains(from), "step {step}");
        let edited = text.replacen(from, to, 1);
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("step {step}: {part} differ\n{text}");
        }
        for (name, expected) in names.iter().zip(answer) {
            assert_eq!(&find(&want, name), expected, "step {step} {name}\n{text}");
        }
    }
    assert!(replayed > 0);
}

/// Modules held in lists, a `Sequential`, and a comprehension. `forward` reads
/// them with a call of the container, an index, and a loop. Each result
/// depends on what the constructor of the other unit (`Block`) and the other
/// lines of `Stack` give the fields.
const MODULES: &str = r#"import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Array


class Block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])
        self.net = nn.Sequential(Block(8, 8), nn.ReLU(), Block(8, 8))
        self.same = nn.ModuleList([Block(8, 8) for _ in range(3)])

    def forward(self, x: Float[Array, "n 8"], i: int):
        a = self.net(x)
        b = self.layers[0](x)
        h = x
        for layer in self.layers:
            h = layer(h)
        s = x
        for layer in self.same:
            s = layer(s)
        return a, b, h, s


def standalone(x: Float[Array, "n 8"]):
    y = x
    return y
"#;

/// Each pair changes one fragment of `MODULES` for another that changes what
/// a field holds: a ctor argument, the count of a comprehension, a change of
/// the list in place (in the class, and from another unit), a different
/// container, an index, a break in the loop, the module a loop runs over.
const MODULE_EDITS: [(&str, &str); 24] = [
    ("Block(16, 4)])", "Block(16, 5)])"),
    ("Block(8, 8), nn.ReLU(), Block(8, 8))", "Block(8, 8), nn.ReLU(), Block(8, 9))"),
    ("nn.Sequential(Block(8, 8), nn.ReLU(), Block(8, 8))", "nn.Sequential(Block(8, 8), Block(8, 16))"),
    ("range(3)", "range(n_layers)"),
    ("range(3)", "range(0)"),
    ("Block(8, 8) for _ in range(3)", "Block(8, 9) for _ in range(1)"),
    ("Block(8, 8) for _ in range(3)", "Block(8, 9) for _ in range(2)"),
    ("self.same = nn.ModuleList([Block(8, 8) for _ in range(3)])", "self.same = [Block(8, 8), Block(8, 8), Block(8, 8)]"),
    ("for layer in self.same:", "for layer in self.layers:"),
    ("h = layer(h)", "h = layer(h)\n            break"),
    ("self.layers[0](x)", "self.layers[i](x)"),
    ("self.layers[0](x)", "self.layers[-2](x)"),
    ("nn.ModuleList([Block(8, 16), Block(16, 4)])\n", "nn.ModuleList([Block(8, 16), Block(16, 4)])\n        self.layers.append(Block(4, 4))\n"),
    ("\n\ndef standalone", "\n\nstack = Stack()\nstack.layers.append(Block(1, 1))\n\n\ndef standalone"),
    ("self.layers = nn.ModuleList(", "self.layers = list("),
    ("return self.fc(x)", "return F.relu(self.fc(x))"),
    ("def forward(self, x):\n        return self.fc(x)", "def __call__(self, x):\n        return self.fc(x)"),
    ("nn.Linear(d_in, d_out)", "nn.Linear(d_in, d_out + 1)"),
    ("def forward(self, x):\n        return self.fc(x)", "def run(self, x):\n        return self.fc(x)"),
    ("x: Float[Array, \"n 8\"], i: int", "x: Float[Array, \"n 8\"], i"),
    (
        "nn.ModuleList([Block(8, 16), Block(16, 4)])\n",
        "nn.ModuleList([Block(8, 16), Block(16, 4)])\n        ls = self.layers\n        ls.append(Block(4, 4))\n",
    ),
    (
        "\n\ndef standalone",
        "\n\nstack = Stack()\nm = stack.layers\nm.append(Block(1, 1))\n\n\ndef standalone",
    ),
    (
        "\n\ndef standalone",
        "\n\nstack = Stack()\ngetattr(stack, \"layers\").append(Block(1, 1))\n\n\ndef standalone",
    ),
    ("for layer in self.layers:\n            h = layer(h)", "k = len(self.layers)\n        for layer in self.layers:\n            h = layer(h)"),
];

fn module_filler() -> String {
    (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect()
}

#[test]
fn replayed_analysis_equals_fresh_analysis_after_module_container_edits() {
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut rng = Rng(SEED ^ 0x30d1);
    let mut text = format!("{MODULES}{}", module_filler());
    let mut tree = parser.parse(&text, None).unwrap();
    let mut replayed = 0;
    for step in 0..400 {
        let (from, to) = MODULE_EDITS[rng.below(MODULE_EDITS.len())];
        let edited = if text.contains(to) && rng.below(2) == 0 {
            text.replacen(to, from, 1)
        } else if text.contains(from) {
            text.replacen(from, to, 1)
        } else {
            text.clone()
        };
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("module edits step {step}: {part} differ\n{text}");
        }
        replayed += memo.stats().replayed_units;
    }
    assert!(replayed > 0, "no step replayed a unit");
}

/// Walk `MODULES` through edits that change what a field holds. Each step
/// must give the same analysis from the memo as from a fresh run, and the
/// names `a`, `b`, `h`, `s` must show the answer that Python gives (unknown
/// where the analyzer does not follow the code, for example a list that other
/// code changes in place).
#[test]
fn module_containers_follow_the_statements_around_them() {
    let d = |dims: &[&str]| Some(dims.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    type Dims = Option<Vec<String>>;
    // a, b, h, s
    type Answer = [Dims; 4];
    let start: Answer = [d(&["n", "8"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "8"])];
    let none: Dims = None;
    let with = |a: Dims, b: Dims, h: Dims, s: Dims| -> Answer { [a, b, h, s] };
    let steps: Vec<(&str, &str, Answer)> = vec![
        ("Block(16, 4)])", "Block(16, 5)])", with(d(&["n", "8"]), d(&["n", "16"]), d(&["n", "5"]), d(&["n", "8"]))),
        ("Block(16, 5)])", "Block(16, 4)])", start.clone()),
        (
            "Block(8, 8), nn.ReLU(), Block(8, 8))",
            "Block(8, 8), nn.ReLU(), Block(8, 9))",
            with(d(&["n", "9"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "8"])),
        ),
        ("Block(8, 8), nn.ReLU(), Block(8, 9))", "Block(8, 8), nn.ReLU(), Block(8, 8))", start.clone()),
        ("range(3)", "range(n_layers)", start.clone()),
        ("range(n_layers)", "range(3)", start.clone()),
        (
            "Block(8, 8) for _ in range(3)",
            "Block(8, 9) for _ in range(1)",
            with(d(&["n", "8"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "9"])),
        ),
        (
            "Block(8, 9) for _ in range(1)",
            "Block(8, 9) for _ in range(0)",
            with(d(&["n", "8"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "8"])),
        ),
        ("Block(8, 9) for _ in range(0)", "Block(8, 8) for _ in range(3)", start.clone()),
        (
            "for layer in self.same:",
            "for layer in self.layers:",
            with(d(&["n", "8"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "4"])),
        ),
        ("for layer in self.layers:\n            s", "for layer in self.same:\n            s", start.clone()),
        (
            "h = layer(h)",
            "h = layer(h)\n            break",
            with(d(&["n", "8"]), d(&["n", "16"]), none.clone(), d(&["n", "8"])),
        ),
        ("h = layer(h)\n            break", "h = layer(h)", start.clone()),
        (
            "self.layers[0](x)",
            "self.layers[i](x)",
            with(d(&["n", "8"]), none.clone(), d(&["n", "4"]), d(&["n", "8"])),
        ),
        ("self.layers[i](x)", "self.layers[0](x)", start.clone()),
        (
            "nn.ModuleList([Block(8, 16), Block(16, 4)])\n",
            "nn.ModuleList([Block(8, 16), Block(16, 4)])\n        self.layers.append(Block(4, 4))\n",
            with(d(&["n", "8"]), none.clone(), none.clone(), d(&["n", "8"])),
        ),
        (
            "nn.ModuleList([Block(8, 16), Block(16, 4)])\n        self.layers.append(Block(4, 4))\n",
            "nn.ModuleList([Block(8, 16), Block(16, 4)])\n",
            start.clone(),
        ),
        (
            "\n\ndef standalone",
            "\n\nstack = Stack()\nstack.layers.append(Block(1, 1))\n\n\ndef standalone",
            with(d(&["n", "8"]), none.clone(), none.clone(), d(&["n", "8"])),
        ),
        (
            "\n\nstack = Stack()\nstack.layers.append(Block(1, 1))\n\n\ndef standalone",
            "\n\ndef standalone",
            start.clone(),
        ),
        (
            "self.layers = nn.ModuleList(",
            "self.layers = list(",
            with(d(&["n", "8"]), none.clone(), none.clone(), d(&["n", "8"])),
        ),
        ("self.layers = list(", "self.layers = nn.ModuleList(", start.clone()),
        ("return self.fc(x)", "return F.relu(self.fc(x))", start.clone()),
        (
            "def forward(self, x):\n        return F.relu(self.fc(x))",
            "def __call__(self, x):\n        return F.relu(self.fc(x))",
            start.clone(),
        ),
        (
            "self.same = nn.ModuleList([Block(8, 8) for _ in range(3)])",
            "self.same = [Block(8, 8), Block(8, 8), Block(8, 8)]",
            start.clone(),
        ),
        (
            "nn.Sequential(Block(8, 8), nn.ReLU(), Block(8, 8))",
            "nn.Sequential(Block(8, 8), Block(8, 16))",
            with(d(&["n", "16"]), d(&["n", "16"]), d(&["n", "4"]), d(&["n", "8"])),
        ),
        (
            "def __call__(self, x):\n        return F.relu(self.fc(x))",
            "def run(self, x):\n        return F.relu(self.fc(x))",
            with(none.clone(), none.clone(), none.clone(), none.clone()),
        ),
    ];
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = format!("{MODULES}{}", module_filler());
    let mut tree = parser.parse(&text, None).unwrap();
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    let find = |a: &LayerShapeAnalysis, name: &str| {
        a.scopes.iter().find_map(|s| s.shapes.get(name).cloned())
    };
    let names = ["a", "b", "h", "s"];
    let first = run(&tree, &text, &mut memo);
    for (name, want) in names.iter().zip(&start) {
        assert_eq!(&find(&first, name), want, "start {name}");
    }
    let mut replayed = 0;
    for (step, (from, to, answer)) in steps.iter().enumerate() {
        assert!(text.contains(from), "step {step}: {from}");
        let edited = text.replacen(from, to, 1);
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("step {step}: {part} differ\n{text}");
        }
        for (name, expected) in names.iter().zip(answer) {
            assert_eq!(&find(&want, name), expected, "step {step} {name}\n{text}");
        }
    }
    assert!(replayed > 0);
}

const MODULE_HELPER_A: &str = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, d_out)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_HELPER_B: &str = "import torch.nn as nn\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, 9)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_HELPER_GONE: &str = "import torch.nn as nn\n\ndef Block(d_in, d_out):\n    return nn.Linear(d_in, d_out)\n";
const MODULE_HELPER_DEEP: &str = "import torch.nn as nn\nfrom module_leaf import Leaf\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.leaf = Leaf(d_out)\n\n    def forward(self, x):\n        return self.leaf(x)\n";
const MODULE_HELPER_CONST_A: &str = "import torch.nn as nn\n\nOUT = 4\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, OUT)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_HELPER_CONST_B: &str = "import torch.nn as nn\n\nOUT = 5\n\nclass Block(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.fc = nn.Linear(d_in, OUT)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_LEAF_A: &str = "import torch.nn as nn\n\nclass Leaf(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.fc = nn.Linear(8, d)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_LEAF_B: &str = "import torch.nn as nn\n\nclass Leaf(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.fc = nn.Linear(8, 6)\n\n    def forward(self, x):\n        return self.fc(x)\n";
const MODULE_MAIN: &str = "import torch.nn as nn\nfrom jaxtyping import Float, Array\nfrom module_helper import Block\n\n\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blk = Block(8, 16)\n        self.layers = nn.ModuleList([Block(8, 16), Block(16, 4)])\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        a = self.blk(x)\n        for layer in self.layers:\n            x = layer(x)\n        return a, x\n\n\ndef g(x: Float[Array, \"n 8\"]):\n    b = Block(8, 3)(x)\n    return b\n";

/// A module class saved in the file that defines it, between edits of the file
/// that calls it. The class imports a second file, so a save of either reaches
/// the call.
#[test]
fn incremental_analysis_survives_imported_module_saves() {
    let tmp = tempfile::tempdir().unwrap();
    let helper = tmp.path().join("module_helper.py");
    let leaf = tmp.path().join("module_leaf.py");
    let main = tmp.path().join("main.py");
    std::fs::write(&helper, MODULE_HELPER_A).unwrap();
    std::fs::write(&leaf, MODULE_LEAF_A).unwrap();
    let roots = vec![tmp.path().to_path_buf()];
    let cache: Arc<ResolutionCache> = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let mut parser = new_parser();
    let paste = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/stress/paste_block.txt"),
    )
    .unwrap();
    let mut rng = Rng(SEED ^ 0x1a0d);
    let (mut text, mut prev) = (MODULE_MAIN.to_string(), MODULE_MAIN.to_string());
    let mut tree = parser.parse(&text, None).unwrap();
    let mut kept = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let toggles = [
        ("Block(8, 3)", "Block(8, 4)"),
        ("Block(16, 4)", "Block(16, 5)"),
        ("self.blk = Block(8, 16)", "self.blk = Block(8, d)"),
        ("for layer in self.layers:", "for layer in reversed(self.layers):"),
        ("from module_helper import Block", "from module_helper import Block as Block, helper"),
    ];
    for step in 0..200 {
        match rng.below(8) {
            0..=3 => {
                let (file, content) = match rng.below(7) {
                    0 => (&helper, MODULE_HELPER_A),
                    1 => (&helper, MODULE_HELPER_B),
                    2 => (&helper, MODULE_HELPER_GONE),
                    3 => (&helper, MODULE_HELPER_DEEP),
                    4 => (&helper, MODULE_HELPER_CONST_A),
                    5 => (&helper, MODULE_HELPER_CONST_B),
                    _ => (&leaf, [MODULE_LEAF_A, MODULE_LEAF_B][rng.below(2)]),
                };
                let before = std::fs::metadata(file).unwrap().modified().unwrap();
                std::fs::write(file, content).unwrap();
                // Keep the old time, so only the editor event shows the save.
                if rng.below(2) == 0 && !kept.contains(&(file.clone(), before, content.len())) {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(file)
                        .unwrap()
                        .set_modified(before)
                        .unwrap();
                }
                let after = std::fs::metadata(file).unwrap();
                kept.insert((file.clone(), after.modified().unwrap(), content.len()));
                cache.revalidate(std::slice::from_ref(file));
            }
            4..=6 => {
                let (from, to) = toggles[rng.below(toggles.len())];
                let edited = if text.contains(to) {
                    text.replacen(to, from, 1)
                } else {
                    text.replacen(from, to, 1)
                };
                tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
                prev = std::mem::replace(&mut text, edited);
            }
            _ => {
                let edited = mutate(&mut rng, &text, &prev, &paste);
                tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
                prev = std::mem::replace(&mut text, edited);
            }
        }
        std::fs::write(&main, &text).unwrap();
        let got = analyze_layer_shapes_with_memo(
            tree.root_node(),
            &text,
            &roots,
            read,
            8,
            Some(&cache),
            &mut memo,
        )
        .unwrap();
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        assert_same_analysis("module helper", step, &got, &want);
        if let Some(shape) = got.scopes.iter().find_map(|s| s.shapes.get("a")) {
            seen.insert(shape.clone());
        }
    }
    // The saves reached the call: the output has taken more than one shape.
    assert!(seen.len() > 1, "{seen:?}");
}

/// A save of the class changes the answer of a call that the memo replays.
#[test]
fn a_saved_module_class_reaches_a_replayed_call() {
    let tmp = tempfile::tempdir().unwrap();
    let helper = tmp.path().join("module_helper.py");
    let leaf = tmp.path().join("module_leaf.py");
    std::fs::write(&helper, MODULE_HELPER_DEEP).unwrap();
    std::fs::write(&leaf, MODULE_LEAF_A).unwrap();
    let roots = vec![tmp.path().to_path_buf()];
    let cache: Arc<ResolutionCache> = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let mut parser = new_parser();
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = format!("{MODULE_MAIN}{}", module_filler());
    let mut tree = parser.parse(&text, None).unwrap();
    let find = |a: &LayerShapeAnalysis, name: &str| {
        a.scopes.iter().find_map(|s| s.shapes.get(name).cloned())
    };
    let strings = |dims: &[&str]| Some(dims.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    // Block(8, 16) goes to Leaf(16): a is [n, 16]. Leaf(3) gives [n, 3].
    let first = run(&tree, &text, &mut memo);
    assert_eq!(find(&first, "a"), strings(&["n", "16"]));
    assert_eq!(find(&first, "b"), strings(&["n", "3"]));
    let mut replayed = 0;
    for (file, content, a, b) in [
        (&leaf, MODULE_LEAF_B, strings(&["n", "6"]), strings(&["n", "6"])),
        (&leaf, MODULE_LEAF_A, strings(&["n", "16"]), strings(&["n", "3"])),
        (&helper, MODULE_HELPER_B, strings(&["n", "9"]), strings(&["n", "9"])),
        (&helper, MODULE_HELPER_GONE, None, None),
        (&helper, MODULE_HELPER_DEEP, strings(&["n", "16"]), strings(&["n", "3"])),
    ] {
        // An edit of the calling class, which runs it again on a stored probe.
        let toggled = if text.contains("return x, a") {
            text.replacen("return x, a", "return a, x", 1)
        } else {
            text.replacen("return a, x", "return x, a", 1)
        };
        tree = reparse(&mut parser, &text, &tree, &toggled).unwrap();
        text = toggled;
        run(&tree, &text, &mut memo);
        std::fs::write(file, content).unwrap();
        cache.revalidate(std::slice::from_ref(file));
        // An edit of a statement that shares no name with the class, so that
        // the class is replayed.
        let edited = format!("{text}\nunrelated_{} = 0\n", text.len());
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        assert_same_analysis("saved class", 0, &got, &want);
        assert_eq!(find(&got, "a"), a, "{content}");
        assert_eq!(find(&got, "b"), b, "{content}");
    }
    assert!(replayed > 0);
}

type Dims = Option<Vec<String>>;

fn dims(d: &[&str]) -> Dims {
    Some(d.iter().map(|s| s.to_string()).collect())
}

fn shape_named(a: &LayerShapeAnalysis, name: &str) -> Dims {
    a.scopes.iter().find_map(|s| s.shapes.get(name).cloned())
}

/// Apply the edits of `steps` to `base` in turn. After each, the analysis from
/// the memo must equal the fresh one, and `check` sees the fresh one with the
/// number of the step. Returns how many units the memo replayed.
fn walk_edits(
    base: &str,
    steps: &[(&str, &str)],
    mut check: impl FnMut(usize, &LayerShapeAnalysis, &str),
) -> usize {
    let mut parser = new_parser();
    let cache = new_resolution_cache();
    let mut memo = UnitMemo::default();
    let roots = vec![std::env::temp_dir()];
    let read = |p: &PathBuf| std::fs::read_to_string(p).ok();
    let mut text = base.to_string();
    let mut tree = parser.parse(&text, None).unwrap();
    let run = |tree: &Tree, text: &str, memo: &mut UnitMemo| {
        analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, Some(&cache), memo)
            .unwrap()
    };
    let first = run(&tree, &text, &mut memo);
    let fresh = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
    assert_same_analysis("walk start", 0, &first, &fresh);
    check(0, &fresh, &text);
    let mut replayed = 0;
    for (step, (from, to)) in steps.iter().enumerate() {
        assert!(text.contains(from), "step {}: {from:?} is not in\n{text}", step + 1);
        let edited = text.replacen(from, to, 1);
        tree = reparse(&mut parser, &text, &tree, &edited).unwrap();
        text = edited;
        let got = run(&tree, &text, &mut memo);
        replayed += memo.stats().replayed_units;
        let want = analyze_layer_shapes(tree.root_node(), &text, &roots, read, 8, None).unwrap();
        if let Some(part) = first_analysis_difference(&got, &want) {
            panic!("step {}: {part} differ\n{text}", step + 1);
        }
        check(step + 1, &want, &text);
    }
    replayed
}

fn independent_functions() -> String {
    (0..60)
        .map(|i| format!("\n\ndef fill_{i}(x: Float[Array, \"2 2\"]):\n    y = x @ x\n    return y\n"))
        .collect()
}

/// `Base.run` calls `self.helper`. Each edit is in another statement and
/// changes which method that call runs: an override, a class attribute, the
/// order of the bases, the kind of the method, a subclass that a mixin puts
/// before the base.
#[test]
fn a_method_call_follows_the_classes_around_it() {
    let base = format!(
        "from jaxtyping import Float, Array\n\n\nclass Base:\n    def helper(self, v):\n        return v.T\n\n    def run(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\n\n\nclass Sub(Base):\n    pass\n\n\nclass Other:\n    def helper(self, v):\n        return v\n{}",
        independent_functions()
    );
    let steps = [
        ("class Sub(Base):\n    pass", "class Sub(Base):\n    def helper(self, v):\n        return v"),
        ("class Sub(Base):\n    def helper(self, v):\n        return v", "class Sub(Base):\n    pass"),
        ("class Sub(Base):\n    pass", "class Sub(Base):\n    helper = None"),
        ("    helper = None", "    pass"),
        ("class Sub(Base):", "class Sub(Base, Other):"),
        ("class Sub(Base, Other):", "class Sub(Other, Base):"),
        ("class Sub(Other, Base):", "class Sub(Other):"),
        ("class Sub(Other):\n    pass", "class Sub(Base):\n    def __init__(self):\n        self.helper = make()"),
        ("    def __init__(self):\n        self.helper = make()", "    pass"),
        ("class Base:\n    def helper(self, v):", "class Base:\n    @property\n    def helper(self):"),
        ("    @property\n    def helper(self):", "    def helper(self, v):"),
        ("class Base:\n    def helper(self, v):", "class Base:\n    def renamed(self, v):"),
        ("class Base:\n    def renamed(self, v):", "class Base(Other):\n    def renamed(self, v):"),
    ];
    let t = dims(&["4", "3"]);
    let same = dims(&["3", "4"]);
    let none: Dims = None;
    let want = [
        t.clone(), none.clone(), t.clone(), none.clone(), t.clone(), t.clone(), none.clone(), t.clone(),
        none.clone(), t.clone(), none.clone(), t.clone(), none, same,
    ];
    let replayed = walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "r"), want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0, "no step replayed a unit");
}

/// A function reads `k.zeros`. Each edit is in another statement and changes
/// what `k` is there, or leaves it the same: a module import, an import in
/// another function, a `global` import, and the alias of `vmap` and the
/// backend that a function import gives to the whole file.
#[test]
fn an_import_binds_only_in_its_scope_across_edits() {
    let base = format!(
        "from jaxtyping import Float, Array\n\n\ndef writer():\n    return 1\n\n\ndef reader():\n    read_k = k.zeros((2, 3))\n\n\ndef param(q: Float[Array, \"3 4\"]):\n    read_q = q.reshape(4, 3)\n\n\nclass C:\n    pass\n{}",
        independent_functions()
    );
    let steps = [
        ("def writer():\n    return 1", "def writer():\n    import numpy as k\n    return 1"),
        ("    import numpy as k\n", "    global k\n    import numpy as k\n"),
        ("    global k\n    import numpy as k\n", ""),
        ("def writer():", "import numpy as k\n\n\ndef writer():"),
        ("def reader():\n", "def reader(k):\n"),
        ("def reader(k):\n", "def reader():\n"),
        ("def reader():\n", "def reader():\n    import os as k\n"),
        ("def reader():\n    import os as k\n", "def reader():\n"),
        ("import numpy as k\n\n\ndef writer():", "def writer():"),
        ("class C:\n    pass", "class C:\n    import numpy as k"),
        ("class C:\n    import numpy as k", "class C:\n    pass"),
        ("def writer():", "import os as q\n\n\ndef writer():"),
        ("import os as q\n\n\ndef writer():", "def writer():"),
    ];
    let two_three = dims(&["2", "3"]);
    let none: Dims = None;
    let want = [
        none.clone(), none.clone(), two_three.clone(), none.clone(), two_three.clone(), none.clone(),
        two_three.clone(), none.clone(), two_three, none.clone(), none.clone(), none.clone(), none.clone(),
        none,
    ];
    let replayed = walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "read_k"), want[step], "step {step}\n{text}");
        assert_eq!(shape_named(analysis, "read_q"), dims(&["4", "3"]), "step {step}\n{text}");
    });
    assert!(replayed > 0, "no step replayed a unit");
}

/// A function import of `vmap` under another name binds the vmapped name for
/// the whole file, and a function import of `torch` sets the backend of the
/// file: both reach readers in other statements.
#[test]
fn a_function_import_of_vmap_or_a_backend_reaches_the_file() {
    let base = format!(
        "import jax\nfrom jaxtyping import Float, Array\n\n\ndef double(x: Float[Array, \"3\"]) -> Float[Array, \"3\"]:\n    return x\n\n\ndef writer():\n    from jax import jit as v\n    batched = v(double)\n\n\ndef reader(b: Float[Array, \"7 3\"]):\n    r = batched(b)\n\n\ndef maxes(x: Float[Array, \"3 4\"]):\n    m = x.max(0)\n{}",
        independent_functions()
    );
    let steps = [
        ("from jax import jit as v", "from jax import vmap as v"),
        ("from jax import vmap as v", "from jax import jit as v"),
        ("def writer():\n", "def writer():\n    import torch\n"),
        ("def writer():\n    import torch\n", "def writer():\n"),
    ];
    let want = [None, dims(&["7", "3"]), None, None, None];
    let maxes = [dims(&["4"]), dims(&["4"]), dims(&["4"]), None, dims(&["4"])];
    walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "r"), want[step], "step {step}\n{text}");
        assert_eq!(shape_named(analysis, "m"), maxes[step], "step {step}\n{text}");
    });
}

/// A layer takes its dim from a constant in another statement. Each edit is
/// outside the class and changes what the dim is: the value, a second write, a
/// write below the class, an annotation of the name, a write of the attribute
/// in a subclass.
#[test]
fn a_layer_dim_follows_the_constant_and_the_statements_around_it() {
    let base = format!(
        "import torch.nn as nn\nfrom jaxtyping import Float, Array\n\nOUT = 4\nATTR = 6\n\n\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.w = ATTR\n        self.fc = nn.Linear(8, OUT)\n        self.gc = nn.Linear(8, self.w)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        fc_out = self.fc(x)\n        gc_out = self.gc(x)\n\n\ndef other(t: Float[Array, \"a b\"]):\n    return t\n{}",
        independent_functions()
    );
    let steps = [
        ("OUT = 4\n", "OUT = 5\n"),
        ("OUT = 5\n", "OUT = 4\nOUT = 7\n"),
        ("OUT = 4\nOUT = 7\n", "OUT = 4\n"),
        ("def other(", "OUT = 4\n\n\ndef other("),
        ("OUT = 4\n\n\ndef other(", "def other("),
        ("def other(t: Float[Array, \"a b\"])", "def other(t: Float[Array, \"a OUT\"])"),
        ("def other(t: Float[Array, \"a OUT\"])", "def other(t: Float[Array, \"a b\"])"),
        ("def other(", "class S(M):\n    def __init__(self):\n        self.w = 9\n\n\ndef other("),
        ("        self.w = 9\n", "        self.w = ATTR\n"),
        ("class S(M):\n    def __init__(self):\n        self.w = ATTR\n\n\n", ""),
        ("ATTR = 6\n", "ATTR = load()\n"),
        ("ATTR = load()\n", "ATTR = 6\n"),
        ("OUT = 4\n", "OUT = load()\n"),
        ("def other(t: Float[Array, \"a b\"])", "def other(t: Float[Array, \"a OUT\"])"),
        ("def other(t: Float[Array, \"a OUT\"])", "def other(t: Float[Array, \"a b\"])"),
    ];
    let four = dims(&["n", "4"]);
    let six = dims(&["n", "6"]);
    let none: Dims = None;
    let want: [(Dims, Dims); 16] = [
        (four.clone(), six.clone()),
        (dims(&["n", "5"]), six.clone()),
        (none.clone(), six.clone()),
        (four.clone(), six.clone()),
        (none.clone(), six.clone()),
        (four.clone(), six.clone()),
        (dims(&["n", "OUT"]), six.clone()),
        (four.clone(), six.clone()),
        (four.clone(), none.clone()),
        (four.clone(), six.clone()),
        (four.clone(), six.clone()),
        (four.clone(), none.clone()),
        (four, six.clone()),
        (none.clone(), six.clone()),
        (dims(&["n", "OUT"]), six.clone()),
        (none, six),
    ];
    let replayed = walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "fc_out"), want[step].0, "step {step}\n{text}");
        assert_eq!(shape_named(analysis, "gc_out"), want[step].1, "step {step}\n{text}");
    });
    assert!(replayed > 0, "no step replayed a unit");
}

/// A function builds a shape from the module constant `D`. An annotation
/// anywhere in the file that uses `D` as a dim makes `D` a name, not 64: the
/// memo must follow an edit of such an annotation in another statement.
#[test]
fn a_constant_follows_the_annotations_of_the_file() {
    let base = format!(
        "import jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\nD = 64\n\n\ndef dims():\n    z = jnp.zeros((D, 2))\n    return z\n\n\ndef other(x: Float[Array, \"n 3\"]):\n    return x\n{}",
        independent_functions()
    );
    let steps = [
        ("\"n 3\"", "\"D 3\""),
        ("\"D 3\"", "\"2*D 3\""),
        ("\"2*D 3\"", "\"n 3\""),
        ("    return x\n", "    return x\n\n\ndef third(x) -> Float[Array, \"D\"]:\n    return x\n"),
        ("def third(x) -> Float[Array, \"D\"]:", "def third(x) -> Float[Array, \"E\"]:"),
        ("D = 64\n", "D = 64\nE = 7\n"),
        ("\"E\"", "\"n\""),
    ];
    let want = [
        dims(&["64", "2"]),
        dims(&["D", "2"]),
        dims(&["D", "2"]),
        dims(&["64", "2"]),
        dims(&["D", "2"]),
        dims(&["64", "2"]),
        dims(&["64", "2"]),
        dims(&["64", "2"]),
    ];
    let replayed = walk_edits(&base, &steps, |step, a, text| {
        assert_eq!(shape_named(a, "z"), want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0);
}

const STACK_MAIN: &str = r#"import torch.nn as nn
from jaxtyping import Float, Array


class Block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(8, 8), Block(8, 4)])

    def forward(self, x: Float[Array, "n 8"]):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
"#;

/// Another statement keeps the list of a field under another name, or reaches
/// it by a string: the list is then not the one the constructor built. The
/// memo must follow the edit, in either direction.
#[test]
fn a_list_follows_the_statements_that_keep_it() {
    let base = format!("{STACK_MAIN}{}", independent_functions());
    let steps = [
        ("\n\ndef fill_0(", "\n\ndef grow(stack):\n    ls = stack.layers\n    ls.append(None)\n\n\ndef fill_0("),
        ("ls = stack.layers\n    ls.append(None)", "ls = list(stack.layers)\n    ls.append(None)"),
        ("ls = list(stack.layers)", "ls = getattr(stack, \"layers\")"),
        ("getattr(stack, \"layers\")", "getattr(stack, \"other\")"),
        ("ls = getattr(stack, \"other\")", "k = len(stack.layers)"),
        ("k = len(stack.layers)\n    ls.append(None)", "stack.layers[0] = None"),
        ("stack.layers[0] = None", "m = stack.layers"),
        ("\n\ndef grow(stack):\n    m = stack.layers\n", "\n"),
    ];
    let want = [
        dims(&["n", "4"]),
        None,
        dims(&["n", "4"]),
        None,
        dims(&["n", "4"]),
        dims(&["n", "4"]),
        None,
        None,
        dims(&["n", "4"]),
    ];
    let replayed = walk_edits(&base, &steps, |step, a, text| {
        assert_eq!(shape_named(a, "h"), want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0);
}

/// A name defined twice in one namespace has no one definition to call: the
/// second definition can stand in another statement, or inside the function.
#[test]
fn a_call_follows_the_definitions_of_its_name() {
    let base = format!(
        "from jaxtyping import Float, Array\n\n\ndef inner(v):\n    return v.T\n\n\ndef use(x: Float[Array, \"3 4\"], c):\n    r = inner(x)\n    def local(v):\n        return v.T\n    s = local(x)\n    return r, s\n{}",
        independent_functions()
    );
    let steps = [
        ("\n\ndef use(", "\n\nif flag:\n    def inner(v):\n        return v\n\n\ndef use("),
        ("if flag:\n    def inner(v):\n        return v\n", "if flag:\n    def other(v):\n        return v\n"),
        ("    s = local(x)", "    if c:\n        def local(v):\n            return v\n    s = local(x)"),
        ("        def local(v):\n            return v\n    s = local(x)", "        def alike(v):\n            return v\n    s = local(x)"),
        ("if flag:\n    def other(v):\n        return v\n", "if flag:\n    def inner(v):\n        return v\n"),
    ];
    let t = dims(&["4", "3"]);
    let want = [
        (t.clone(), t.clone()),
        (None, t.clone()),
        (t.clone(), t.clone()),
        (t.clone(), None),
        (t.clone(), t.clone()),
        (None, t.clone()),
    ];
    let replayed = walk_edits(&base, &steps, |step, a, text| {
        let got = (shape_named(a, "r"), shape_named(a, "s"));
        assert_eq!(got, want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0);
}

/// Whether a `with` body can stop part way depends on what the callee is
/// imported as, which is a statement of its own. An import edit changes what
/// every replayed record depends on, so each step runs everything.
#[test]
fn a_context_manager_follows_the_import_it_is_called_by() {
    let base = format!(
        "from contextlib import suppress as sp\nimport jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\n\ndef use(x: Float[Array, \"3\"]):\n    with sp(ValueError):\n        x = jnp.zeros(4)\n    r = x + 0\n    return r\n{}",
        independent_functions()
    );
    let steps = [
        ("from contextlib import suppress as sp", "from numpy import ones as sp"),
        ("from numpy import ones as sp", "from pytest import raises as sp"),
        ("from pytest import raises as sp", "sp = other"),
        ("sp = other", "from contextlib import suppress as sp"),
    ];
    let want = [None, dims(&["4"]), None, dims(&["4"]), None];
    walk_edits(&base, &steps, |step, a, text| {
        assert_eq!(shape_named(a, "r"), want[step], "step {step}\n{text}");
    });
}

/// A function has its own import of `k`, so the file map holds it whatever the
/// module does. An edit of the module import changes the module map alone, and
/// the readers in other statements must follow it.
#[test]
fn a_module_import_edit_reaches_the_readers_when_the_file_map_stays() {
    let base = format!(
        "import numpy as k\nimport torch.nn as nn\nfrom jaxtyping import Float, Array\n\n\ndef f(x):\n    import numpy as k\n    return x\n\n\ndef g():\n    a = k.zeros((2, 3))\n\n\nclass C(nn.Module):\n    def __init__(self):\n        self.l = k.Linear(8, 4)\n{}",
        independent_functions()
    );
    let steps = [
        ("import numpy as k\nimport torch", "import torch"),
        ("import torch.nn as nn\n", "import numpy as k\nimport torch.nn as nn\n"),
        ("import numpy as k\nimport torch", "import os as k\nimport torch"),
        ("    import numpy as k\n    return x", "    import os as k\n    return x"),
        ("import os as k\nimport torch", "import numpy as k\nimport torch"),
    ];
    let two_three = dims(&["2", "3"]);
    let want = [two_three.clone(), None, two_three.clone(), None, None, two_three];
    walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "a"), want[step], "step {step}\n{text}");
    });
}

/// `P(OUT)` takes its dim from a constant in another statement, and each edit
/// is outside the statement of the call.
#[test]
fn a_constructor_argument_follows_the_constant_across_edits() {
    let base = format!(
        "import torch.nn as nn\nfrom jaxtyping import Float, Array\n\nOUT = 4\n\n\nclass P(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.l = nn.Linear(8, d)\n\n    def forward(self, x):\n        return self.l(x)\n\n\ndef use(x: Float[Array, \"n 8\"]):\n    r = P(OUT)(x)\n\n\nclass H(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.p = P(OUT)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        s = self.p(x)\n{}",
        independent_functions()
    );
    let steps = [
        ("OUT = 4\n", "OUT = 5\n"),
        ("OUT = 5\n", "OUT = load()\n"),
        ("OUT = load()\n", "OUT = 4\nOUT = 7\n"),
        ("OUT = 4\nOUT = 7\n", "OUT = 4\n"),
        ("class P(nn.Module):", "OUT = 4\n\n\nclass P(nn.Module):"),
    ];
    let four = dims(&["n", "4"]);
    let want = [four.clone(), dims(&["n", "5"]), None, None, four.clone(), four];
    let replayed = walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "r"), want[step], "step {step}\n{text}");
        assert_eq!(shape_named(analysis, "s"), want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0, "no step replayed a unit");
}

/// `class B(P)` reads `P`, and another statement rebinds it or gives it back.
#[test]
fn a_base_follows_the_statements_that_rebind_its_name() {
    let base = format!(
        "from jaxtyping import Float, Array\n\n\nclass P:\n    def helper(self, v):\n        return v.T\n\n\ndef mid():\n    return 1\n\n\nclass B(P):\n    def go(self, x: Float[Array, \"3 4\"]):\n        r = self.helper(x)\n{}",
        independent_functions()
    );
    let steps = [
        ("def mid():", "P = other\n\n\ndef mid():"),
        ("P = other\n\n\n", ""),
        ("def mid():", "from e import P\n\n\ndef mid():"),
        ("from e import P\n\n\n", ""),
        ("def mid():", "def P():"),
        ("def P():", "def mid():"),
    ];
    let t = dims(&["4", "3"]);
    let want = [t.clone(), None, t.clone(), None, t.clone(), None, t];
    let replayed = walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "r"), want[step], "step {step}\n{text}");
    });
    assert!(replayed > 0, "no step replayed a unit");
}

/// A `self.<attr>` write through a `for` or `with` target is in another
/// statement than the reader when a subclass makes it.
#[test]
fn a_loop_write_of_an_attribute_follows_edits() {
    let base = format!(
        "import torch.nn as nn\nfrom jaxtyping import Float, Array\n\n\nclass M(nn.Module):\n    def __init__(self):\n        self.h = 4\n        self.l = nn.Linear(8, self.h)\n\n    def forward(self, x: Float[Array, \"n 8\"]):\n        r = self.l(x)\n\n\nclass S(M):\n    def t(self):\n        pass\n{}",
        independent_functions()
    );
    let steps = [
        ("    def t(self):\n        pass", "    def t(self):\n        for self.h in (8, 9):\n            pass"),
        ("        for self.h in (8, 9):\n            pass", "        pass"),
        ("    def t(self):\n        pass", "    def t(self):\n        with a() as self.h:\n            pass"),
        ("        with a() as self.h:\n            pass", "        pass"),
    ];
    let four = dims(&["n", "4"]);
    let want = [four.clone(), None, four.clone(), None, four];
    walk_edits(&base, &steps, |step, analysis, text| {
        assert_eq!(shape_named(analysis, "r"), want[step], "step {step}\n{text}");
    });
}
