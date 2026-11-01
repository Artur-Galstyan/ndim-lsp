use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tree_sitter::{Parser, Point, Range, Tree};

use crate::incremental::*;
use crate::python_ast::{ClassSummary, SelfAttrWrite};
use crate::resolution::{ResolutionCache, clear_resolution_cache, new_resolution_cache};
use crate::*;

fn parse(text: &str) -> Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .unwrap();
    parser.parse(text, None).unwrap()
}

fn read(path: &PathBuf) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().flatten().collect();
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

fn all_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    python_files(&root.join("corpus"), &mut files);
    python_files(&root.join("bench"), &mut files);
    assert!(files.len() > 20);
    files
}

type WriteView = (String, (usize, usize), Option<(&'static str, usize, usize)>);

fn write_view(w: &SelfAttrWrite) -> WriteView {
    (
        w.attr.clone(),
        w.class,
        w.value.map(|n| (n.kind(), n.start_byte(), n.end_byte())),
    )
}

fn assert_same_prepass(label: &str, got: &PrePass, want: &PrePass) {
    assert_eq!(got.scopes, want.scopes, "{label}: scopes");
    assert_eq!(got.imports, want.imports, "{label}: imports");
    assert_eq!(
        got.layer_records, want.layer_records,
        "{label}: layer_records"
    );
    assert_eq!(
        got.self_attr_layers, want.self_attr_layers,
        "{label}: self_attr_layers"
    );
    let views = |p: &PrePass| p.writes.iter().map(write_view).collect::<Vec<_>>();
    assert_eq!(views(got), views(want), "{label}: writes");
    assert_eq!(got.hierarchy, want.hierarchy, "{label}: hierarchy");
}

struct Run {
    memo: UnitMemo,
    stats: MemoStats,
}

/// The pre-passes of `text` from `memo`, checked against the whole-tree
/// pre-passes.
fn checked_run(
    label: &str,
    memo: &UnitMemo,
    text: &str,
    roots: &[PathBuf],
    cache: &ResolutionCache,
) -> Run {
    let tree = parse(text);
    let root = tree.root_node();
    cache.revalidate(&[]);
    let want = PrePass::whole_tree(root, text, roots, &read, 8, Some(cache)).unwrap();
    let (got, next) = memo.prepass(root, text, roots, &read, 8, cache).unwrap();
    assert_same_prepass(label, &got, &want);
    Run {
        stats: next.stats(),
        memo: next,
    }
}

fn run_pair(old: &str, new: &str) -> (MemoStats, MemoStats) {
    let cache = new_resolution_cache();
    let first = checked_run("old", &UnitMemo::default(), old, &[], &cache);
    let second = checked_run("new", &first.memo, new, &[], &cache);
    (first.stats, second.stats)
}

#[test]
fn unit_merge_equals_whole_tree_on_all_files() {
    for file in all_files() {
        let text = fs::read_to_string(&file).unwrap();
        let roots = vec![file.parent().unwrap().to_path_buf()];
        let cache = new_resolution_cache();
        let label = file.display().to_string();
        let cold = checked_run(&label, &UnitMemo::default(), &text, &roots, &cache);
        assert_eq!(cold.stats.syntax_hits, 0, "{label}");
        let warm = checked_run(&label, &cold.memo, &text, &roots, &cache);
        let tree = parse(&text);
        let mut cursor = tree.walk();
        let broken = tree
            .root_node()
            .children(&mut cursor)
            .filter(|n| n.has_error())
            .count();
        assert_eq!(warm.stats.syntax_hits, warm.stats.units - broken, "{label}");
        assert!(
            warm.stats.layer_hits > 0 || warm.stats.units == 0,
            "{label}"
        );
    }
}

#[test]
fn unit_merge_equals_whole_tree_after_every_line_edit() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/mamba_ssm.py");
    let text = fs::read_to_string(&file).unwrap();
    let roots = vec![file.parent().unwrap().to_path_buf()];
    let cache = new_resolution_cache();
    let mut memo = checked_run("base", &UnitMemo::default(), &text, &roots, &cache).memo;
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    for i in 0..lines.len() {
        let mut edited = lines.clone();
        let tweaked = format!("{}  # edit\n", lines[i].trim_end());
        edited[i] = &tweaked;
        let run = checked_run(
            &format!("line {i}"),
            &memo,
            &edited.concat(),
            &roots,
            &cache,
        );
        assert!(run.stats.syntax_hits > 0, "line {i}: {:?}", run.stats);
        memo = run.memo;
    }
}

const CLASSES: &str = "import torch.nn as nn\n\nclass A:\n    def __init__(self):\n        self.f = nn.Linear(2, 3)\n        self.n = 4\n\nclass B(A):\n    def __init__(self):\n        self.g = nn.Linear(3, 4)\n";

#[test]
fn two_statements_on_one_line_with_a_semicolon() {
    let old = "import torch.nn as nn\na = 1; b = nn.Linear(3, 4)\nc = 2\n";
    let new = "import torch.nn as nn\na = 10; b = nn.Linear(3, 4)\nc = 2\n";
    let (_, stats) = run_pair(old, new);
    // The semicolon is a unit too. `a = 10` changed and the semicolon and
    // `b = ...` moved one column right, so only the import and `c = 2` are
    // reused.
    assert_eq!(stats.units, 5);
    assert_eq!(stats.syntax_hits, 2);
}

#[test]
fn a_char_before_the_first_line_shifts_every_unit() {
    let (_, stats) = run_pair(CLASSES, &format!("\n{CLASSES}"));
    assert_eq!(stats.syntax_hits, stats.units);
    assert_eq!(stats.layer_hits, stats.units);

    let (_, stats) = run_pair(CLASSES, &format!("x = 1; {CLASSES}"));
    // The import now starts in column 7, so it is not the same unit.
    assert_eq!(stats.units, 5);
    assert_eq!(stats.syntax_hits, 2);
}

#[test]
fn a_duplicated_unit_is_found_twice() {
    let old = "import torch.nn as nn\nl = nn.Linear(3, 4)\nz = 1\n";
    let new = "import torch.nn as nn\nl = nn.Linear(3, 4)\nz = 1\nl = nn.Linear(3, 4)\n";
    let (_, stats) = run_pair(old, new);
    assert_eq!(stats.syntax_hits, 4);
}

#[test]
fn a_moved_unit_is_found_by_its_text() {
    let a = "class A:\n    def __init__(self):\n        self.f = nn.Linear(2, 3)\n\n";
    let b = "class B:\n    def __init__(self):\n        self.f = nn.Linear(5, 6)\n\n";
    let head = "import torch.nn as nn\n\n";
    let (_, stats) = run_pair(&format!("{head}{a}{b}"), &format!("{head}{b}{a}"));
    assert_eq!(stats.syntax_hits, stats.units);
    assert_eq!(stats.layer_hits, stats.units);
}

#[test]
fn the_last_import_and_the_last_layer_win() {
    let old = "from a import X as t\nfrom b import Y as t\nimport torch.nn as nn\nl = nn.Linear(2, 3)\nl = nn.Linear(4, 5)\n";
    let new = "from b import Y as t\nfrom a import X as t\nimport torch.nn as nn\nl = nn.Linear(4, 5)\nl = nn.Linear(2, 3)\n";
    let cache = new_resolution_cache();
    let first = checked_run("old", &UnitMemo::default(), old, &[], &cache);
    let second = checked_run("new", &first.memo, new, &[], &cache);
    assert_eq!(second.stats.syntax_hits, 5);

    let tree = parse(new);
    let (pre, _) = first
        .memo
        .prepass(tree.root_node(), new, &[], &read, 8, &cache)
        .unwrap();
    assert_eq!(pre.imports.file()["t"].name, "X");
    assert_eq!(
        pre.layer_records.last().unwrap().kind,
        LayerKind::Linear {
            in_features: "2".into(),
            out_features: "3".into()
        }
    );
}

#[test]
fn an_edited_import_line_recomputes_every_layer_record() {
    let old = "import torch.nn as nn\nl = nn.Linear(2, 3)\nm = nn.Linear(3, 4)\n";
    let new = "import torch.nn as tn\nl = nn.Linear(2, 3)\nm = nn.Linear(3, 4)\n";
    let (_, stats) = run_pair(old, new);
    assert_eq!(stats.syntax_hits, 2);
    assert_eq!(stats.layer_hits, 0);
}

#[test]
fn a_unit_with_a_syntax_error_is_never_reused() {
    let old = "import torch.nn as nn\nx = (1,\ny = 2\n";
    let (_, stats) = run_pair(old, old);
    assert!(stats.syntax_hits < stats.units, "{stats:?}");
}

#[test]
fn memo_is_dropped_when_roots_or_depth_change() {
    let text = "import torch.nn as nn\nl = nn.Linear(2, 3)\n";
    let tree = parse(text);
    let cache = new_resolution_cache();
    let roots = vec![PathBuf::from("/a")];
    let (_, memo) = UnitMemo::default()
        .prepass(tree.root_node(), text, &roots, &read, 8, &cache)
        .unwrap();
    let (_, same) = memo
        .prepass(tree.root_node(), text, &roots, &read, 8, &cache)
        .unwrap();
    assert_eq!(same.stats().syntax_hits, 2);
    let other_roots = vec![PathBuf::from("/b")];
    let (_, moved) = memo
        .prepass(tree.root_node(), text, &other_roots, &read, 8, &cache)
        .unwrap();
    assert_eq!(moved.stats().syntax_hits, 0);
    let (_, deeper) = memo
        .prepass(tree.root_node(), text, &roots, &read, 9, &cache)
        .unwrap();
    assert_eq!(deeper.stats().syntax_hits, 0);
    let (_, other_cache) = memo
        .prepass(
            tree.root_node(),
            text,
            &roots,
            &read,
            8,
            &new_resolution_cache(),
        )
        .unwrap();
    assert_eq!(other_cache.stats().syntax_hits, 0);
}

#[test]
fn no_cache_means_no_memo() {
    let text = "import torch.nn as nn\nl = nn.Linear(2, 3)\n";
    let tree = parse(text);
    let mut memo = UnitMemo::default();
    let cache = new_resolution_cache();
    let roots: Vec<PathBuf> = Vec::new();
    analyze_layer_shapes_with_memo(
        tree.root_node(),
        text,
        &roots,
        read,
        8,
        Some(&cache),
        &mut memo,
    )
    .unwrap();
    assert!(!memo.is_empty());
    analyze_layer_shapes_with_memo(tree.root_node(), text, &roots, read, 8, None, &mut memo)
        .unwrap();
    assert!(memo.is_empty());
}

mod resolution {
    use super::*;

    const MAIN: &str = "from mymod import Block\nfrom other import Thing\nlayer = Block(3, 5)\nthing = Thing(1)\nz = 1\n";

    struct Setup {
        tmp: tempfile::TempDir,
        roots: Vec<PathBuf>,
        cache: Arc<ResolutionCache>,
    }

    fn setup() -> Setup {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("mymod.py"),
            "class Block:\n    def __init__(self, a, b): pass\n",
        )
        .unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        Setup {
            tmp,
            roots,
            cache: new_resolution_cache(),
        }
    }

    /// Reused layer records of a second run that follows `change`.
    fn layer_hits_after(s: &Setup, change: impl FnOnce(&Setup)) -> usize {
        let first = checked_run("first", &UnitMemo::default(), MAIN, &s.roots, &s.cache);
        let again = checked_run("again", &first.memo, MAIN, &s.roots, &s.cache);
        assert_eq!(again.stats.layer_hits, again.stats.units);
        change(s);
        checked_run("after", &again.memo, MAIN, &s.roots, &s.cache)
            .stats
            .layer_hits
    }

    #[test]
    fn a_helper_saved_with_other_content_drops_the_unit() {
        let s = setup();
        let hits = layer_hits_after(&s, |s| {
            fs::write(
                s.tmp.path().join("mymod.py"),
                "class Block:\n    def __init__(self, a, b, c): pass\n",
            )
            .unwrap();
        });
        assert_eq!(hits, 5 - 1);
    }

    #[test]
    fn a_helper_saved_with_the_same_mtime_drops_the_unit() {
        let s = setup();
        let path = s.tmp.path().join("mymod.py");
        let hits = layer_hits_after(&s, |s| {
            let before = fs::metadata(&path).unwrap().modified().unwrap();
            fs::write(&path, "class Block:\n    def __init__(self, x, y): pass\n").unwrap();
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(before)
                .unwrap();
            // Only an editor event sees a save that kept size and time.
            s.cache.revalidate(std::slice::from_ref(&path));
        });
        assert_eq!(hits, 5 - 1);
    }

    #[test]
    fn a_missing_helper_that_appears_drops_the_unit() {
        let s = setup();
        let hits = layer_hits_after(&s, |s| {
            fs::write(
                s.tmp.path().join("other.py"),
                "class Thing:\n    def __init__(self, a): pass\n",
            )
            .unwrap();
        });
        assert_eq!(hits, 5 - 1);
    }

    #[test]
    fn an_unrelated_file_change_keeps_every_unit() {
        let s = setup();
        let hits = layer_hits_after(&s, |s| {
            fs::write(s.tmp.path().join("unrelated.py"), "x = 1\n").unwrap();
        });
        assert_eq!(hits, 5);
    }

    #[test]
    fn a_cleared_cache_drops_every_unit_that_looked_at_disk() {
        let s = setup();
        let hits = layer_hits_after(&s, |s| clear_resolution_cache(&s.cache));
        assert_eq!(hits, 0);
    }

    #[test]
    fn a_unit_that_ran_during_a_drop_is_not_kept() {
        let s = setup();
        let path = s.tmp.path().join("mymod.py");
        // The cache holds `Block`, so the unit takes it from there and then
        // reads the helper, which an editor event changes at that moment.
        checked_run("warm", &UnitMemo::default(), MAIN, &s.roots, &s.cache);
        let dropper = |p: &PathBuf| {
            fs::write(
                &path,
                "class Block:\n    def __init__(self, a, b, c): pass\n",
            )
            .unwrap();
            s.cache.revalidate(std::slice::from_ref(&path));
            fs::read_to_string(p).ok()
        };
        let tree = parse(MAIN);
        let (_, memo) = UnitMemo::default()
            .prepass(tree.root_node(), MAIN, &s.roots, &dropper, 8, &s.cache)
            .unwrap();
        let again = checked_run("again", &memo, MAIN, &s.roots, &s.cache);
        assert_eq!(again.stats.layer_hits, again.stats.units - 1);
    }
}

mod shift {
    use super::*;

    const BY: Delta = Delta::new(100, 7);

    fn point(row: usize, column: usize) -> Point {
        Point { row, column }
    }

    fn range(start: usize, end: usize, row: usize) -> Range {
        Range {
            start_byte: start,
            end_byte: end,
            start_point: point(row, 4),
            end_point: point(row + 1, 9),
        }
    }

    fn shifted<T: Shift + Clone>(value: &T, by: Delta) -> T {
        let mut copy = value.clone();
        copy.shift(by);
        copy
    }

    fn round_trip<T: Shift + Clone + PartialEq + std::fmt::Debug>(value: &T) {
        let back = Delta::new(-BY.bytes, -BY.rows);
        assert_eq!(&shifted(&shifted(value, BY), back), value);
    }

    #[test]
    fn a_point_moves_rows_only() {
        let mut p = point(3, 8);
        p.shift(BY);
        assert_eq!((p.row, p.column), (10, 8));
    }

    #[test]
    fn a_range_moves_bytes_and_rows() {
        let r = shifted(&range(10, 20, 2), BY);
        assert_eq!(r, range(110, 120, 9));
        round_trip(&range(10, 20, 2));
    }

    #[test]
    fn a_dimension_site_moves_its_range() {
        let site = ShapeDimensionSite {
            binding: Some("x".into()),
            axis: 1,
            value: "n".into(),
            range: range(10, 11, 2),
        };
        let got = shifted(&site, BY);
        assert_eq!(got.range, range(110, 111, 9));
        assert_eq!((got.axis, got.value.as_str()), (1, "n"));
        round_trip(&site);
    }

    #[test]
    fn a_scope_moves_its_bounds_and_sites() {
        let scope = FunctionShapeScope {
            function_name: Some("f".into()),
            start_byte: 5,
            end_byte: 50,
            shapes: HashMap::from([("x".to_string(), vec!["n".to_string()])]),
            return_shape: Some(vec!["n".into()]),
            param_order: vec!["x".into()],
            all_params: vec!["x".into()],
            dimension_sites: vec![ShapeDimensionSite {
                binding: None,
                axis: 0,
                value: "n".into(),
                range: range(12, 13, 1),
            }],
        };
        let got = shifted(&scope, BY);
        assert_eq!((got.start_byte, got.end_byte), (105, 150));
        assert_eq!(got.dimension_sites[0].range, range(112, 113, 8));
        assert_eq!(got.shapes, scope.shapes);
        round_trip(&scope);
    }

    #[test]
    fn a_layer_assignment_moves_its_byte() {
        let a = LayerAssignment {
            name: "l".into(),
            kind: LayerKind::Linear {
                in_features: "2".into(),
                out_features: "3".into(),
            },
            byte_position: 40,
        };
        assert_eq!(shifted(&a, BY).byte_position, 140);
        round_trip(&a);
    }

    #[test]
    fn class_ranges_move_but_the_no_class_range_does_not() {
        let kind = LayerKind::Linear {
            in_features: "2".into(),
            out_features: "3".into(),
        };
        let layer = |class_start, class_end, byte| ScopedSelfAttrLayer {
            byte,
            class_start,
            class_end,
            kind: kind.clone(),
        };
        let alias = |class_start, class_end| ScopedSelfAttrAlias {
            class_start,
            class_end,
            value: "v".into(),
        };
        assert_eq!(shifted(&layer(10, 90, 20), BY), layer(110, 190, 120));
        assert_eq!(shifted(&layer(0, usize::MAX, 20), BY), layer(0, usize::MAX, 120));
        assert_eq!(shifted(&alias(10, 90), BY), alias(110, 190));
        assert_eq!(shifted(&alias(0, usize::MAX), BY), alias(0, usize::MAX));
        round_trip(&layer(10, 90, 20));
        round_trip(&alias(10, 90));
    }

    #[test]
    fn an_assignment_shape_moves_its_row() {
        let a = AssignmentShape {
            line: 4,
            name: "y".into(),
            shape: vec!["n".into()],
        };
        assert_eq!(shifted(&a, BY).line, 11);
        round_trip(&a);
    }

    #[test]
    fn a_fix_moves_both_ranges() {
        let fix = ShapeFix::AppendTranspose {
            expression_range: range(10, 30, 1),
            operand_range: range(12, 14, 1),
        };
        let ShapeFix::AppendTranspose {
            expression_range,
            operand_range,
        } = shifted(&fix, BY);
        assert_eq!(expression_range, range(110, 130, 8));
        assert_eq!(operand_range, range(112, 114, 8));
        round_trip(&fix);
    }

    #[test]
    fn an_error_moves_range_related_and_fix() {
        let error = ShapeError::mismatch("y", "bad", range(10, 20, 2))
            .with_related(range(30, 40, 3), "other")
            .with_fix(ShapeFix::AppendTranspose {
                expression_range: range(10, 20, 2),
                operand_range: range(30, 40, 3),
            });
        let got = shifted(&error, BY);
        assert_eq!(got.range, range(110, 120, 9));
        assert_eq!(
            got.related,
            Some((range(130, 140, 10), "other".to_string()))
        );
        assert_eq!(
            got.fix,
            Some(ShapeFix::AppendTranspose {
                expression_range: range(110, 120, 9),
                operand_range: range(130, 140, 10),
            })
        );
        let bare = ShapeError::approximation("y", "bad", range(10, 20, 2));
        assert_eq!(shifted(&bare, BY).range, range(110, 120, 9));
        assert_eq!(shifted(&bare, BY).related, None);
        round_trip(&error);
    }

    #[test]
    fn an_application_moves_its_range() {
        let app = LayerApplication {
            variable: "y".into(),
            layer: "l".into(),
            input: "x".into(),
            kind: LayerKind::Linear {
                in_features: "2".into(),
                out_features: "3".into(),
            },
            range: range(10, 20, 2),
            extra_inputs: Vec::new(),
        };
        assert_eq!(shifted(&app, BY).range, range(110, 120, 9));
    }

    #[test]
    fn a_write_moves_its_class_and_value() {
        let write = |class, value: Option<(usize, usize)>| WriteRecord::for_test("a", class, value);
        assert_eq!(
            shifted(&write((10, 90), Some((20, 30))), BY),
            write((110, 190), Some((120, 130)))
        );
        assert_eq!(
            shifted(&write((0, usize::MAX), None), BY),
            write((0, usize::MAX), None)
        );
    }

    #[test]
    fn a_class_summary_moves_its_range() {
        let class = ClassSummary {
            name: Some("A".into()),
            range: (10, 90),
            bases: Vec::new(),
            fields: vec!["f".into()],
            attrs: vec!["a".to_string()],
        };
        assert_eq!(shifted(&class, BY).range, (110, 190));
        round_trip(&class);
    }

    #[test]
    fn a_unit_record_reaches_the_same_positions_it_came_from() {
        let text = "import torch.nn as nn\n\n\nclass A:\n    def f(self, x: Float[Array, \"n d\"]):\n        self.k = nn.Linear(2, 3)\n        return x\n";
        let tree = parse(text);
        let cache = new_resolution_cache();
        let (pre, memo) = UnitMemo::default()
            .prepass(tree.root_node(), text, &[], &read, 8, &cache)
            .unwrap();
        let moved = format!("\n\n{text}");
        let tree2 = parse(&moved);
        let (pre2, _) = memo
            .prepass(tree2.root_node(), &moved, &[], &read, 8, &cache)
            .unwrap();
        for (a, b) in pre.scopes[1..].iter().zip(&pre2.scopes[1..]) {
            assert_eq!(a.start_byte + 2, b.start_byte);
            let (sa, sb) = (&a.dimension_sites, &b.dimension_sites);
            assert_eq!(sa.len(), sb.len());
            for (x, y) in sa.iter().zip(sb) {
                assert_eq!(x.range.start_point.row + 2, y.range.start_point.row);
                assert_eq!(x.range.start_point.column, y.range.start_point.column);
            }
        }
        assert_eq!(pre2.scopes.len(), pre.scopes.len());
    }
}

mod propagate_memo {
    use super::*;

    const HEADER: &str = "import jax\nimport jax.numpy as jnp\nimport torch.nn as nn\n\n\n";

    fn filler(count: usize) -> String {
        (0..count)
            .map(|i| {
                format!(
                    "def filler_{i}(x: Float[Array, \"a{i} b{i}\"]):\n    y = jnp.sum(x, axis=0)\n    return y\n\n\n"
                )
            })
            .collect()
    }

    /// Each text analyzed through one memo, in order. Every result must equal
    /// the fresh analysis of the same text.
    fn chain(texts: &[String]) -> Vec<(LayerShapeAnalysis, MemoStats)> {
        let cache = new_resolution_cache();
        let mut memo = UnitMemo::default();
        let mut out = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let tree = parse(text);
            let got = analyze_layer_shapes_with_memo(
                tree.root_node(),
                text,
                &[],
                |_| None,
                8,
                Some(&cache),
                &mut memo,
            )
            .unwrap();
            let want =
                analyze_layer_shapes(tree.root_node(), text, &[], |_| None, 8, None).unwrap();
            if let Some(part) = first_analysis_difference(&got, &want) {
                panic!("text {i}: {part} differ\n{text}");
            }
            out.push((got, memo.stats()));
        }
        out
    }

    fn fresh(text: &str) -> LayerShapeAnalysis {
        let tree = parse(text);
        analyze_layer_shapes(tree.root_node(), text, &[], |_| None, 8, None).unwrap()
    }

    fn dirty(stats: &MemoStats) -> usize {
        stats.units - stats.replayed_units
    }

    /// `old` then `new` through one memo. The edit must change the fresh
    /// result, so a stale replay would show. Returns the stats of `new`.
    fn edit(old: &str, new: &str) -> MemoStats {
        let (old, new) = (format!("{HEADER}{old}"), format!("{HEADER}{new}"));
        assert!(
            first_analysis_difference(&fresh(&old), &fresh(&new)).is_some(),
            "the edit changes nothing:\n{new}"
        );
        // A pass over the old text first, so the memo holds its records.
        let run = chain(&[old.clone(), old, new]);
        run.last().unwrap().1
    }

    const FILLERS: usize = 6;

    #[test]
    fn an_edit_of_an_isolated_unit_replays_the_others() {
        let body = |op: &str| {
            format!(
                "{}def target(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]):\n    y = x {op} w\n    return y\n",
                filler(FILLERS)
            )
        };
        let stats = edit(&body("@"), &body("*"));
        assert!(!stats.propagate_full);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) <= 3, "{stats:?}");
    }

    const HELPER: &str = "def helper(t):\n    u = jnp.exp(t)\n    return u.sum(axis=-1)\n\n\n";
    const CALL_A: &str =
        "def call_a(x: Float[Array, \"p q\"]):\n    r = helper(x)\n    return r\n\n\n";
    const CALL_B: &str =
        "def call_b(x: Float[Array, \"m n\"]):\n    s = helper(x)\n    return s\n\n\n";

    #[test]
    fn first_call_wins_moves_to_the_next_caller_when_the_first_is_deleted() {
        let both = format!("{}{HELPER}{CALL_A}{CALL_B}", filler(FILLERS));
        let only_b = format!("{}{HELPER}{CALL_B}", filler(FILLERS));
        let stats = edit(&both, &only_b);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 2, "{stats:?}");
    }

    #[test]
    fn first_call_wins_moves_to_the_next_caller_when_the_first_is_edited() {
        let before = format!("{}{HELPER}{CALL_A}{CALL_B}", filler(FILLERS));
        let after = before.replace("r = helper(x)", "r = helper(x.T)");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 3, "{stats:?}");
    }

    #[test]
    fn phase_two_reads_state_written_by_a_later_caller() {
        let early = "def early(x: Float[Array, \"p q\"]):\n    return helper(x)\n\n\n";
        let before = format!("{}{HELPER}{early}{CALL_B}", filler(FILLERS));
        let after = before.replace("s = helper(x)", "s = helper(x.T)");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 3, "{stats:?}");
    }

    #[test]
    fn a_callee_edit_invalidates_its_callers() {
        let before = format!("{}{HELPER}{CALL_A}{CALL_B}", filler(FILLERS));
        let after = before.replace("axis=-1", "axis=0");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 3, "{stats:?}");
    }

    #[test]
    fn a_new_def_of_an_already_called_name_invalidates_the_callers() {
        let before = format!("{}{HELPER}{CALL_A}", filler(FILLERS));
        let twin = "def helper(t):\n    return t\n\n\n";
        let after = format!("{before}{twin}");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 2, "{stats:?}");
    }

    const BLOCK: &str = "class Block:\n    def __init__(self):\n        self.proj = nn.Linear(8, 16)\n\n    def __call__(self, x: Float[Array, \"b 8\"]):\n        return self.proj(x)\n\n\n";
    const MODEL: &str = "class Model:\n    def __init__(self):\n        self.blk = Block()\n\n    def __call__(self, x: Float[Array, \"b 8\"]):\n        h = self.blk(x)\n        return h\n\n\n";

    #[test]
    fn a_class_rename_invalidates_the_classes_that_use_it() {
        let before = format!("{}{BLOCK}{MODEL}", filler(FILLERS));
        let after = before.replace("class Block", "class Blocks");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 2, "{stats:?}");
    }

    #[test]
    fn a_duplicated_class_name_invalidates_the_classes_that_use_it() {
        let before = format!("{}{BLOCK}{MODEL}", filler(FILLERS));
        let twin = BLOCK.replace("nn.Linear(8, 16)", "nn.Linear(8, 32)");
        let after = format!("{before}{twin}");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 3, "{stats:?}");
    }

    #[test]
    fn self_method_calls_match_every_method_of_that_name() {
        let a = "class A:\n    def step(self, x: Float[Array, \"b d\"]) -> Float[Array, \"b d\"]:\n        return x\n\n\n";
        let b = "class B:\n    def step(self, x: Float[Array, \"c e\"]):\n        return x\n\n    def run(self, x: Float[Array, \"b d\"]):\n        y = self.step(x)\n        return y\n\n\n";
        let before = format!("{}{a}{b}", filler(FILLERS));
        let after = before.replace("-> Float[Array, \"b d\"]", "-> Float[Array, \"d b\"]");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 2, "{stats:?}");
    }

    #[test]
    fn a_second_module_vmap_of_a_name_invalidates_its_readers() {
        let funcs = "def f1(x: Float[Array, \"d\"]) -> Float[Array, \"d\"]:\n    return x\n\n\ndef f2(x: Float[Array, \"d\"]) -> Float[Array, \"k\"]:\n    return x[:3]\n\n\n";
        let make1 = "vf = jax.vmap(f1)\n\n\n";
        let make2 = "vf = jax.vmap(f2)\n\n\n";
        let user = "def user(xs: Float[Array, \"n d\"]):\n    ys = vf(xs)\n    return ys\n\n\n";
        let before = format!("{}{funcs}{make1}{make2}{user}", filler(FILLERS));
        let after = format!("{}{funcs}{make1}{user}", filler(FILLERS));
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 3, "{stats:?}");
    }

    #[test]
    fn a_module_constant_read_by_name_invalidates_its_readers() {
        let user = "def g(x: Float[Array, \"b d\"]):\n    y = x * K\n    return y\n\n\n";
        let before = format!("{}K = 4\n\n\n{user}", filler(FILLERS));
        let after = before.replace("K = 4", "K = jnp.ones((5,))");
        let stats = edit(&before, &after);
        assert!(stats.replayed_units >= FILLERS, "{stats:?}");
        assert!(dirty(&stats) >= 2, "{stats:?}");
    }

    #[test]
    fn a_global_statement_runs_everything_again() {
        let user = "def g(x: Float[Array, \"b d\"]):\n    y = x * K\n    return y\n\n\n";
        let before = format!("{}K = 4\n\n\n{user}", filler(FILLERS));
        let after = before.replace("    y = x * K", "    global K\n    K = jnp.ones((5,))\n    y = x * K");
        let stats = edit(&before, &after);
        assert!(stats.propagate_full, "{stats:?}");
    }

    #[test]
    fn a_wildcard_import_runs_everything_again() {
        let user = "def g(x: Float[Array, \"b d\"]):\n    y = x * K\n    return y\n\n\n";
        let before = format!("{}K = 4\n\n\n{user}", filler(FILLERS));
        let after = format!("from jax.numpy import *\n{before}").replace("K = 4", "K = ones((5,))");
        let stats = edit(&before, &after);
        assert!(stats.propagate_full, "{stats:?}");
    }

    #[test]
    fn a_changed_import_runs_everything_again() {
        let user = "def g(x: Float[Array, \"b d\"]):\n    y = jnp.exp(x)\n    return y\n\n\n";
        let before = format!("{}{user}", filler(FILLERS));
        let after = format!("{before}\nimport numpy as np\n");
        let cache = new_resolution_cache();
        let mut memo = UnitMemo::default();
        for text in [format!("{HEADER}{before}"), format!("{HEADER}{after}")] {
            let tree = parse(&text);
            analyze_layer_shapes_with_memo(
                tree.root_node(),
                &text,
                &[],
                |_| None,
                8,
                Some(&cache),
                &mut memo,
            )
            .unwrap();
        }
        assert!(memo.stats().propagate_full, "{:?}", memo.stats());
    }

    #[test]
    fn a_unit_with_a_syntax_error_is_never_replayed() {
        let user = "def g(x: Float[Array, \"b d\"]):\n    y = jnp.exp(x)\n    return y\n\n\n";
        let good = format!("{HEADER}{}{user}", filler(FILLERS));
        let broken = format!("{good}def broken(:\n    pass\n");
        let run = chain(&[good.clone(), broken.clone(), broken, good]);
        for (_, stats) in &run[1..3] {
            assert!(stats.replayed_units >= FILLERS, "{stats:?}");
            assert!(dirty(stats) >= 1, "{stats:?}");
        }
        assert!(dirty(&run[2].1) >= 1, "{:?}", run[2].1);
    }

    #[test]
    fn edits_with_nested_inline_calls_keep_internal_names_out() {
        let callee = "def callee(t, w: Float[Array, \"a b\"]):\n    y = jnp.exp(w) @ jnp.exp(w)\n    z = jnp.concatenate([jnp.exp(w), jnp.exp(t)], axis=0)\n    return z\n\n\n";
        let caller = "def caller(x: Float[Array, \"p q\"], w: Float[Array, \"a b\"]):\n    r = callee(x, w)\n    q = jnp.exp(r) @ jnp.exp(x)\n    return r\n\n\n";
        let base = format!("{HEADER}{}{callee}{caller}", filler(FILLERS));
        let edited = base.replace("jnp.exp(x)", "jnp.exp(x.T)");
        let again = base.replace("jnp.exp(t)", "jnp.tanh(t)");
        let run = chain(&[base.clone(), base, edited, again]);
        assert!(run.iter().all(|(a, _)| a.errors.iter().all(|e| {
            !e.message.contains("__synth_") && !e.variable.contains("__synth_")
        })));
    }

    #[test]
    fn replay_equals_fresh_on_all_files_after_one_line_edits() {
        for file in all_files() {
            let text = fs::read_to_string(&file).unwrap();
            let lines: Vec<&str> = text.split_inclusive('\n').collect();
            let roots = vec![file.parent().unwrap().to_path_buf()];
            let cache = new_resolution_cache();
            let mut memo = UnitMemo::default();
            let mut run = |t: &str| {
                let tree = parse(t);
                let got = analyze_layer_shapes_with_memo(
                    tree.root_node(),
                    t,
                    &roots,
                    read,
                    8,
                    Some(&cache),
                    &mut memo,
                )
                .unwrap();
                let want = analyze_layer_shapes(tree.root_node(), t, &roots, read, 8, None).unwrap();
                if let Some(part) = first_analysis_difference(&got, &want) {
                    panic!("{}: {part} differ", file.display());
                }
            };
            run(&text);
            run(&text);
            for i in (0..lines.len()).step_by(lines.len() / 12 + 1) {
                let mut edited = lines.clone();
                let mutated = format!("{}  # edit\n", edited[i].trim_end_matches('\n'));
                edited[i] = &mutated;
                run(&edited.concat());
            }
        }
    }
}

/// The shape of `y` in `new`, after the memo of `old`, and with no memo.
fn memo_y_shape(old: &str, new: &str) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let y_of = |analysis: &LayerShapeAnalysis| {
        analysis.scopes.iter().find_map(|scope| scope.shapes.get("y")).cloned()
    };
    let (cache, mut memo) = (new_resolution_cache(), UnitMemo::default());
    let mut run = |text: &str| {
        let tree = parse(text);
        analyze_layer_shapes_with_memo(tree.root_node(), text, &[], read, 5, Some(&cache), &mut memo).unwrap()
    };
    run(old);
    let memoized = run(new);
    let tree = parse(new);
    let fresh = analyze_layer_shapes(tree.root_node(), new, &[], read, 5, None).unwrap();
    (y_of(&memoized), y_of(&fresh))
}

#[test]
fn an_edit_of_another_unit_reaches_a_subscript() {
    let head = "import torch\nimport jax.numpy as jnp\nfrom jaxtyping import Float, Array\n\n";
    let reader = "def g(x: Float[Array, \"n m\"], w: Float[Array, \"n m\"]):\n    y = w[:n]\n    return y\n";
    let indexed = "def g(x: Float[Array, \"3 4\"]):\n    y = x[i]\n    return y\n";
    let stepped = "def g():\n    y = v[::-1]\n    return y\n";
    let looped = "def g(x: Float[Array, \"3 4\"]):\n    for k in range(3):\n        y = x[k]\n    return y\n";
    for (before, after, body) in [
        ("q = 1\n", "n = 2\n", reader),
        ("q = load()\n", "def s():\n    global n\n    n = 2\n", reader),
        ("i = 1\n", "i = load()\n", indexed),
        ("i = load()\n", "i = 1\n", indexed),
        ("v = torch.zeros(3, 4)\n", "v = jnp.zeros((3, 4))\n", stepped),
        ("v = jnp.zeros((3, 4))\n", "v = torch.zeros(3, 4)\n", stepped),
        ("q = 1\n", "range = load\n", looped),
    ] {
        for (old, new) in [(before, after), (after, before)] {
            let [old, new] = [old, new].map(|bind| format!("{head}{bind}\n{body}"));
            let (memoized, fresh) = memo_y_shape(&old, &new);
            assert_eq!(memoized, fresh, "{old}\n->\n{new}");
        }
    }
}

#[test]
fn an_edit_of_a_class_body_reaches_the_function_around_it() {
    let body = |line: &str| {
        format!("import torch\n\ndef h():\n    f = torch.zeros(3, 4)\n    class C:\n        nonlocal f\n{line}    y = f\n")
    };
    let (poisoned, plain) = (body("        f = load()\n"), body("        g = f\n"));
    assert_eq!(memo_y_shape(&plain, &poisoned), (None, None));
    let (memoized, fresh) = memo_y_shape(&poisoned, &plain);
    assert_eq!(memoized, fresh);
    assert!(memoized.is_some());
}
