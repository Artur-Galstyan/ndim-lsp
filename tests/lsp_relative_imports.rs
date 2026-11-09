//! A file that the open document imports with a relative import is tracked
//! like an absolute import: a save, a watched-file event, or an edit of its
//! open buffer gives the result of a fresh server.

mod common;

use std::fs;
use std::path::Path;
use std::time::Duration;

use common::{Client, file_uri};
use serde_json::{Value, json};

const HELPER_OK: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const HELPER_BAD: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b d\"]:\n    return x @ w\n";

/// `main` calls the helper through `import`/`call`. With the bad helper the
/// result is `b 4`, so the product with `v` gives one error.
fn main_text(import: &str, call: &str) -> String {
    format!(
        "from jaxtyping import Float, Array\n{import}\n\ndef f(x: Float[Array, \"b 4\"], w: Float[Array, \"4 8\"], v: Float[Array, \"8 3\"]):\n    y = {call}(x, w) @ v\n    return y\n"
    )
}

/// `<root>/pkg/__init__.py`, `<root>/pkg/sub/__init__.py`, and the helper at
/// `helper` (relative to the root).
fn package(helper: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    fs::create_dir_all(root.join("pkg/sub")).unwrap();
    fs::write(root.join("pkg/__init__.py"), "").unwrap();
    fs::write(root.join("pkg/sub/__init__.py"), "").unwrap();
    fs::write(root.join(helper), HELPER_OK).unwrap();
    (dir, root)
}

fn messages(diagnostics: &[Value]) -> Vec<String> {
    diagnostics.iter().map(|d| d["message"].as_str().unwrap_or("").to_string()).collect()
}

fn fresh(root: &Path, uri: &str, text: &str) -> Vec<String> {
    messages(&Client::start(root, json!({})).open(uri, text))
}

const CASES: [(&str, &str, &str, &str); 3] = [
    ("pkg/main.py", "pkg/helpers.py", "from .helpers import proj", "proj"),
    ("pkg/main.py", "pkg/helpers.py", "from . import helpers", "helpers.proj"),
    ("pkg/sub/main.py", "pkg/helpers.py", "from .. import helpers", "helpers.proj"),
];

#[test]
fn save_and_watch_event_of_a_relative_import_give_the_fresh_result() {
    for (document, helper, import, call) in CASES {
        let (_dir, root) = package(helper);
        let uri = file_uri(&root.join(document));
        let helper = root.join(helper);
        let text = main_text(import, call);
        let mut c = Client::start(&root, json!({}));
        assert!(c.open(&uri, &text).is_empty(), "{import}");
        // An edit builds the memo, so the next run replays it.
        let from = c.published();
        c.change(&uri, 2, &format!("{text}\n"));
        assert!(c.wait_diagnostics(&uri, from).is_empty(), "{import}");

        fs::write(&helper, HELPER_BAD).unwrap();
        let from = c.published();
        c.notify("textDocument/didSave", json!({"textDocument": {"uri": file_uri(&helper)}}));
        let got = messages(&c.wait_diagnostics(&uri, from));
        assert_eq!(got.len(), 1, "stale result after save, {import}: {got:?}");
        assert_eq!(got, fresh(&root, &uri, &format!("{text}\n")), "{import}");

        fs::write(&helper, HELPER_OK).unwrap();
        let from = c.published();
        c.notify(
            "workspace/didChangeWatchedFiles",
            json!({"changes": [{"uri": file_uri(&helper), "type": 2}]}),
        );
        let got = messages(&c.wait_diagnostics(&uri, from));
        assert!(got.is_empty(), "stale error after the watch event, {import}: {got:?}");

        let from = c.published();
        c.change(&uri, 3, &text);
        assert!(c.wait_diagnostics(&uri, from).is_empty(), "{import}");
    }
}

#[test]
fn unsaved_buffer_of_a_relative_import_is_read() {
    for (document, helper, import, call) in CASES {
        let (_dir, root) = package(helper);
        let uri = file_uri(&root.join(document));
        let helper_uri = file_uri(&root.join(helper));
        let text = main_text(import, call);
        let mut c = Client::start(&root, json!({}));
        assert!(c.open(&uri, &text).is_empty(), "{import}");
        let from = c.published();
        c.change(&uri, 2, &format!("{text}\n"));
        assert!(c.wait_diagnostics(&uri, from).is_empty(), "{import}");

        c.open(&helper_uri, HELPER_OK);
        let from = c.published();
        c.change(&helper_uri, 2, HELPER_BAD);
        let got = c
            .wait_diagnostics_for(&uri, from, Duration::from_secs(5))
            .map(|d| messages(&d))
            .unwrap_or_else(|| panic!("no republish after the helper edit, {import}"));
        assert_eq!(got.len(), 1, "unsaved helper not read, {import}: {got:?}");

        let from = c.published();
        c.change(&helper_uri, 3, HELPER_OK);
        let got = c
            .wait_diagnostics_for(&uri, from, Duration::from_secs(5))
            .map(|d| messages(&d))
            .unwrap_or_else(|| panic!("no republish after the helper undo, {import}"));
        assert!(got.is_empty(), "stale error after the undo, {import}: {got:?}");
    }
}
