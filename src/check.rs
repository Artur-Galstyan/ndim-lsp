//! `ndim-lsp check <path>...`: analyze files without an editor and print the
//! diagnostics like a compiler.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndim_lsp::{ShapeErrorKind, new_resolution_cache};

use crate::sync::{Encoding, LineIndex};
use crate::{
    Buffers, CachedAnalysis, analyze_text, assignment_line, python_site_packages_roots, with_src_layout,
};

const USAGE: &str = "usage: ndim-lsp check [--shapes] <path>...";

/// Run the check of `args` (the arguments after `check`). Returns the exit
/// code: 1 if a shape mismatch was found, 2 for a usage or read error, else 0.
/// Files run in sorted path order. A summary line goes to stderr.
pub fn run(args: &[String]) -> i32 {
    let shapes = args.iter().any(|a| a == "--shapes");
    let paths: Vec<&String> = args.iter().filter(|a| *a != "--shapes").collect();
    if paths.is_empty() || paths.iter().any(|p| p.starts_with('-')) {
        eprintln!("{USAGE}");
        return 2;
    }
    let mut files = Vec::new();
    for path in paths {
        let path = PathBuf::from(path);
        if path.is_dir() {
            python_files(&path, &mut files);
        } else if path.is_file() {
            files.push(path);
        } else {
            eprintln!("{}: no such file or directory", path.display());
            return 2;
        }
    }
    files.sort();
    let mut seen = std::collections::HashSet::new();
    files.retain(|f| seen.insert(std::fs::canonicalize(f).unwrap_or_else(|_| f.clone())));
    let root = std::env::current_dir().unwrap_or_default();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let mut search_roots = with_src_layout(std::slice::from_ref(&root));
    search_roots.extend(python_site_packages_roots(&[root]));
    let cache = new_resolution_cache();
    let mut code = 0;
    let (mut analyzed, mut errors, mut warnings) = (0, 0, 0);
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            eprintln!("{}: cannot read the file", file.display());
            code = 2;
            continue;
        };
        let document = std::fs::canonicalize(&file).ok();
        let run = match ndim_lsp::with_document(document.as_deref(), || {
            analyze_text(&text, None, &search_roots, &cache, &Buffers::new(), false)
        }) {
            Ok(run) => run,
            Err(e) => {
                eprintln!("{}: analysis failed: {e}", file.display());
                code = 2;
                continue;
            }
        };
        analyzed += 1;
        let path = file.display();
        let lines = LineIndex::new(&text);
        for error in &run.analysis.errors {
            let at = lines.position(&text, error.range.start_byte, Encoding::Utf8);
            let severity = match error.kind {
                ShapeErrorKind::Mismatch => {
                    code = code.max(1);
                    errors += 1;
                    "error"
                }
                ShapeErrorKind::Approximation => {
                    warnings += 1;
                    "warning"
                }
            };
            let message = if error.variable.is_empty() {
                error.message.clone()
            } else {
                format!("{}: {}", error.variable, error.message)
            };
            println!("{path}:{}:{}: {severity}: {message}", at.line + 1, at.character + 1);
        }
        if shapes {
            let cached = CachedAnalysis {
                analysis: Arc::new(run.analysis),
                text: Arc::from(text),
                tree: Arc::new(run.tree),
                lines: Arc::new(lines),
                memo: Arc::default(),
                reads: Arc::default(),
                version: 0,
                rev: 0,
            };
            let mut seen = std::collections::HashSet::new();
            for rec in &cached.analysis.assignment_shapes {
                let line = assignment_line(&cached, rec.line, &rec.name);
                if seen.insert((line, rec.name.as_str())) {
                    println!("{path}:{}: {}: [{}]", line + 1, rec.name, rec.shape.join(", "));
                }
            }
        }
    }
    eprintln!("{analyzed} files, {errors} errors, {warnings} warnings");
    code
}

/// Directories that a walk skips: environments, build output, and caches.
const SKIPPED: [&str; 7] = [".venv", "venv", "node_modules", "build", "dist", "__pycache__", "site-packages"];

/// The `*.py` files under `dir`, without hidden and `SKIPPED` directories.
fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
        if path.is_dir() {
            if !name.starts_with('.') && !SKIPPED.contains(&&*name) {
                python_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "py") {
            out.push(path);
        }
    }
}
