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
