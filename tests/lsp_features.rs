//! Positions, hover, dimension references and rename, and check mode,
//! through the stdio server.

mod common;

use common::{Client, file_uri, project};
use serde_json::{Value, json};

/// The LSP position of the first `needle`: lines end at `\n`, `\r\n`, or `\r`.
fn pos(text: &str, needle: &str) -> (u32, u32) {
    let byte = text.find(needle).unwrap_or_else(|| panic!("{needle}"));
    let before = &text[..byte];
    let mut line = 0;
    let mut start = 0;
    let bytes = before.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' || (b == b'\r' && text.as_bytes().get(i + 1) != Some(&b'\n')) {
            line += 1;
            start = i + 1;
        }
    }
    (line, before[start..].encode_utf16().count() as u32)
}

fn line_text(text: &str, line: u32) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n").split('\n').nth(line as usize).unwrap().to_string()
}

fn start_of(range: &Value) -> (u32, u32) {
    let p = &range["start"];
    (p["line"].as_u64().unwrap() as u32, p["character"].as_u64().unwrap() as u32)
}

#[test]
fn positions_follow_the_editor_line_model() {
    let lone_cr = "import torch\nfrom jaxtyping import Float, Array\n# one\r# two\ndef f(a: Float[Array, \"m k\"], b: Float[Array, \"n k\"]):\n    y = a @ b\n    z = torch.zeros(3, 4)\n    w = torch.zeros(7)\rv = torch.zeros(8)\n    return z\n";
    let crlf = lone_cr.replace("\r", "").replace('\n', "\r\n");
    for text in [lone_cr.to_string(), crlf] {
        let (_dir, root) = project(&[]);
        let uri = file_uri(&root.join("main.py"));
        let mut c = Client::start(&root, json!({}));
        let diags = c.open(&uri, &text);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert_eq!(start_of(&diags[0]["range"]), pos(&text, "a @ b"), "{text:?}");

        let hints = c.request(
            "textDocument/inlayHint",
            json!({"textDocument": {"uri": uri},
                   "range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}}}),
        );
        let at = |label: &str| -> (u32, u32) {
            let h = hints.as_array().unwrap().iter().find(|h| h["label"] == label).unwrap();
            (h["position"]["line"].as_u64().unwrap() as u32, h["position"]["character"].as_u64().unwrap() as u32)
        };
        for (label, needle) in [(": [3, 4]", "z = torch"), (": [8]", "v = torch")] {
            let line = pos(&text, needle).0;
            let end = line_text(&text, line).encode_utf16().count() as u32;
            assert_eq!(at(label), (line, end), "{label} in {text:?}");
        }

        let (line, character) = pos(&text, "z = torch");
        let hover = c.at("hover", &uri, line, character, json!({}));
        assert_eq!(start_of(&hover["range"]), (line, character), "{text:?}");
        assert!(hover["contents"]["value"].as_str().unwrap().contains("\"3 4\""));
    }
}

const DIMS: &str = "from jaxtyping import Float, Array\n\ndef f(x: Float[Array, \"n d\"], y: Float[Array, \"n d\"]) -> Float[Array, \"n\"]:\n    z: Float[Array, \"n d\"] = x + y\n    return z.sum(1)\n\ndef g(x: Float[Array, \"n k\"], w: Float[Array, \"k*2\"]):\n    pass\n";

fn ranges(v: &Value) -> Vec<(u32, u32)> {
    v.as_array().unwrap().iter().map(|x| start_of(x.get("range").unwrap_or(x))).collect()
}

/// The positions of every `"n` dim of `f` in DIMS.
fn n_sites_of_f() -> Vec<(u32, u32)> {
    let f = &DIMS[..DIMS.find("def g").unwrap()];
    let mut out = Vec::new();
    for (i, line) in f.split('\n').enumerate() {
        for (c, _) in line.match_indices("\"n") {
            out.push((i as u32, c as u32 + 1));
        }
    }
    out
}

#[test]
fn references_and_highlights_of_a_dimension_stay_in_its_function() {
    let (_dir, root) = project(&[]);
    let uri = file_uri(&root.join("main.py"));
    let mut c = Client::start(&root, json!({}));
    c.open(&uri, DIMS);
    let want = n_sites_of_f();
    assert_eq!(want.len(), 4);
    let (line, character) = want[2];

    let all = c.at("references", &uri, line, character, json!({"context": {"includeDeclaration": true}}));
    assert_eq!(ranges(&all), want);
    assert!(all.as_array().unwrap().iter().all(|l| l["uri"] == uri.as_str()));
    let uses = c.at("references", &uri, line, character, json!({"context": {"includeDeclaration": false}}));
    assert_eq!(ranges(&uses), want[1..]);
    let highlights = c.at("documentHighlight", &uri, line, character, json!({}));
    assert_eq!(ranges(&highlights), want);

    let (gl, gc) = pos(DIMS, "n k\"");
    assert_eq!(ranges(&c.at("references", &uri, gl, gc, json!({"context": {"includeDeclaration": true}}))), [(gl, gc)]);
    let (cl, cc) = pos(DIMS, "k*2");
    assert_eq!(c.at("references", &uri, cl, cc, json!({"context": {"includeDeclaration": true}})), Value::Null);
}

fn rename(c: &mut Client, uri: &str, at: (u32, u32), name: &str) -> Value {
    let id = c.send_request(
        "textDocument/rename",
        json!({"textDocument": {"uri": uri}, "position": {"line": at.0, "character": at.1}, "newName": name}),
    );
    c.wait_response(id)
}

#[test]
fn rename_of_a_dimension_edits_its_function_and_rejects_bad_names() {
    let (_dir, root) = project(&[]);
    let uri = file_uri(&root.join("main.py"));
    let versioned = json!({"workspace": {"workspaceEdit": {"documentChanges": true}}});
    let mut c = Client::start(&root, versioned);
    c.open(&uri, DIMS);
    let sites = n_sites_of_f();

    let prepared = c.at("prepareRename", &uri, sites[1].0, sites[1].1, json!({}));
    assert_eq!(start_of(&prepared), sites[1]);
    assert_eq!(prepared["end"]["character"].as_u64(), Some(sites[1].1 as u64 + 1));
    let (dl, dc) = pos(DIMS, "def f");
    assert_eq!(c.at("prepareRename", &uri, dl, dc, json!({})), Value::Null);

    let edit = rename(&mut c, &uri, sites[0], "batch")["result"].clone();
    let doc = &edit["documentChanges"][0];
    assert_eq!(doc["textDocument"]["version"], 1);
    assert_eq!(ranges(&doc["edits"]), sites);
    assert!(doc["edits"].as_array().unwrap().iter().all(|e| e["newText"] == "batch"));

    for bad in ["2n", "n d", "", "_", "_n", "d"] {
        let reply = rename(&mut c, &uri, sites[0], bad);
        assert_eq!(reply["error"]["code"], -32602, "{bad}: {reply}");
    }
    let (gl, gc) = pos(DIMS, "k\"], w");
    let reply = rename(&mut c, &uri, (gl, gc), "m");
    assert_eq!(reply["error"]["code"], -32602, "k is read by `k*2`: {reply}");

    let (_dir2, root2) = project(&[]);
    let uri2 = file_uri(&root2.join("main.py"));
    let mut plain = Client::start(&root2, json!({}));
    plain.open(&uri2, DIMS);
    let edit = rename(&mut plain, &uri2, sites[0], "batch")["result"].clone();
    assert!(edit.get("documentChanges").is_none(), "{edit}");
    assert_eq!(ranges(&edit["changes"][uri2.as_str()]), sites);
}

#[test]
fn diagnostics_have_a_code_per_kind() {
    let (_dir, root) = project(&[]);
    let uri = file_uri(&root.join("main.py"));
    let mut c = Client::start(&root, json!({}));
    let text = "from jaxtyping import Float, Array\n\ndef f(a: Float[Array, \"m 3\"], b: Float[Array, \"4 k\"]):\n    return a @ b\n";
    let diags = c.open(&uri, text);
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0]["code"], "shape-mismatch");
    assert_eq!(diags[0]["source"], "ndim-lsp");
    // Two different names can be equal at run time.
    let names = "from jaxtyping import Float, Array\n\ndef f(a: Float[Array, \"m k\"], b: Float[Array, \"n k\"]):\n    return a @ b\n";
    let diags = c.open(&file_uri(&root.join("names.py")), names);
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0]["code"], "shape-approximation");
}

#[test]
fn utf8_positions_when_the_client_offers_them() {
    let text = "from jaxtyping import Float, Array\n\ndef f(é: Float[Array, \"m k\"], b: Float[Array, \"n k\"]):\n    y = é @ b\n    return y\n";
    let line = 2;
    let row = text.split('\n').nth(2).unwrap();
    let byte_col = row.find("k\"], b").unwrap() as u32;
    let use_col = row.find("k\"]):").unwrap() as u32;
    for (caps, enc, col) in [
        (json!({"general": {"positionEncodings": ["utf-8", "utf-16"]}}), "utf-8", byte_col),
        (json!({"general": {"positionEncodings": ["utf-16"]}}), "utf-16", byte_col - 1),
        (json!({}), "utf-16", byte_col - 1),
    ] {
        let (_dir, root) = project(&[]);
        let uri = file_uri(&root.join("main.py"));
        let mut c = Client::start(&root, caps);
        assert_eq!(c.init["capabilities"]["positionEncoding"], enc);
        let diags = c.open(&uri, text);
        assert_eq!(start_of(&diags[0]["range"]), (line + 1, 8), "{enc}");
        assert_eq!(diags[0]["range"]["end"]["character"], 14 - (byte_col - col), "{enc}");
        let shift = byte_col - col;
        let k = c.at("definition", &uri, line, use_col - shift, json!({}));
        assert_eq!(start_of(&k["range"]), (line, col), "{enc}");

        let from = c.published();
        c.notify(
            "textDocument/didChange",
            json!({"textDocument": {"uri": uri, "version": 2},
                   "contentChanges": [{"range": {"start": {"line": line, "character": col},
                                                 "end": {"line": line, "character": col + 1}},
                                       "text": "n"}]}),
        );
        assert_eq!(c.wait_diagnostics(&uri, from), Vec::<Value>::new(), "{enc}");
    }
}

const HELPER_OK: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const HELPER_BAD: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"k d\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const MAIN: &str = "from jaxtyping import Float, Array\nfrom helper import proj\n\ndef f(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]):\n    y = proj(x, w)\n    return y\n";

fn messages(diags: &[Value]) -> Vec<String> {
    diags.iter().map(|d| d["message"].as_str().unwrap().to_string()).collect()
}

/// The diagnostics of MAIN on a fresh server with `helper` on disk.
fn fresh_main(helper: &str) -> Vec<String> {
    let (_dir, root) = project(&[("helper.py", helper)]);
    let mut c = Client::start(&root, json!({}));
    messages(&c.open(&file_uri(&root.join("main.py")), MAIN))
}

#[test]
fn an_unsaved_helper_buffer_is_what_imports_read() {
    let (_dir, root) = project(&[("helper.py", HELPER_OK), ("main.py", MAIN)]);
    let (main, helper) = (file_uri(&root.join("main.py")), file_uri(&root.join("helper.py")));
    let (ok, bad) = (fresh_main(HELPER_OK), fresh_main(HELPER_BAD));
    assert!(ok.is_empty() && !bad.is_empty());
    let mut c = Client::start(&root, json!({}));
    assert_eq!(messages(&c.open(&main, MAIN)), ok);
    c.open(&helper, HELPER_OK);

    // Each edit counts, not only the first one after the disk state.
    for (version, text, want) in [(2, HELPER_BAD, &bad), (3, HELPER_OK, &ok), (4, HELPER_BAD, &bad)] {
        let from = c.published();
        c.change(&helper, version, text);
        assert_eq!(&messages(&c.wait_diagnostics(&main, from)), want, "version {version}");
    }

    // The closed buffer leaves the disk text.
    let from = c.published();
    c.notify("textDocument/didClose", json!({"textDocument": {"uri": helper}}));
    assert_eq!(messages(&c.wait_diagnostics(&main, from)), ok);
}

#[test]
fn a_helper_opened_with_unsaved_text_is_read_at_once() {
    let (_dir, root) = project(&[("helper.py", HELPER_OK), ("main.py", MAIN)]);
    let (main, helper) = (file_uri(&root.join("main.py")), file_uri(&root.join("helper.py")));
    let mut c = Client::start(&root, json!({}));
    assert!(c.open(&main, MAIN).is_empty());
    let from = c.published();
    c.open(&helper, HELPER_BAD);
    assert_eq!(messages(&c.wait_diagnostics(&main, from)), fresh_main(HELPER_BAD));
}

#[test]
fn a_helper_buffer_equal_to_the_disk_does_not_republish() {
    let (_dir, root) = project(&[("helper.py", HELPER_OK), ("main.py", MAIN)]);
    let (main, helper) = (file_uri(&root.join("main.py")), file_uri(&root.join("helper.py")));
    let mut c = Client::start(&root, json!({}));
    c.open(&main, MAIN);
    c.open(&helper, HELPER_OK);
    c.notify("textDocument/didClose", json!({"textDocument": {"uri": helper}}));
    let from = c.published();
    c.settle(500);
    let republished = c.notes.iter().filter(|n| n["method"] == "textDocument/publishDiagnostics").skip(from).any(|n| n["params"]["uri"] == main.as_str());
    assert!(!republished);
}

fn check(dir: &std::path::Path, args: &[&str]) -> (i32, String) {
    let (code, out, _) = check_with_summary(dir, args);
    (code, out)
}

fn check_with_summary(dir: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let out = std::process::Command::new(common::BIN)
        .arg("check")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

/// A directory walk skips environments and build output, an explicit file
/// always runs, files run once in sorted order, a `src` layout resolves from
/// the project root, and stderr gets a summary.
#[test]
fn check_mode_walks_a_project_like_a_compiler() {
    let bad = "from jaxtyping import Float, Array\n\ndef f(a: Float[Array, \"m 3\"], b: Float[Array, \"4 k\"]):\n    y = a @ b\n";
    let helper = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"]) -> Float[Array, \"b 5\"]:\n    return x\n";
    let user = "from jaxtyping import Float, Array\nimport pkg.other\n\ndef f(x: Float[Array, \"b 4\"], v: Float[Array, \"4 3\"]):\n    y = pkg.other.proj(x) @ v\n";
    let (_dir, root) = project(&[]);
    for dir in [".venv", "venv", "node_modules", "build", "dist", "__pycache__", "lib/site-packages", ".git", "src/pkg", "z"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        if dir != "src/pkg" {
            std::fs::write(root.join(dir).join("bad.py"), bad).unwrap();
        }
    }
    std::fs::write(root.join("src/pkg/__init__.py"), "").unwrap();
    std::fs::write(root.join("src/pkg/other.py"), helper).unwrap();
    std::fs::write(root.join("src/pkg/mod.py"), user).unwrap();
    std::fs::write(root.join("a.py"), bad).unwrap();
    let (code, out, summary) = check_with_summary(&root, &["z", ".", ".venv/bad.py"]);
    assert_eq!(code, 1, "{out}");
    let files: Vec<&str> = out.lines().map(|l| l.split(':').next().unwrap()).collect();
    assert_eq!(files, ["./a.py", "./src/pkg/mod.py", "./z/bad.py", ".venv/bad.py"], "{out}");
    assert!(out.contains("./src/pkg/mod.py:5:9: error: y: matmul dimension mismatch"), "{out}");
    assert_eq!(summary, "6 files, 4 errors, 0 warnings\n");
}

#[test]
fn check_mode_prints_diagnostics_like_a_compiler() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (code, out) = check(repo, &["corpus"]);
    assert_eq!(code, 0, "{out}");
    assert!(!out.contains(": error: "), "{out}");

    let bad = "from jaxtyping import Float, Array\n\ndef f(a: Float[Array, \"m 3\"], b: Float[Array, \"4 k\"]):\n    y = a @ b\n    z = a\n";
    let (_dir, root) = project(&[("bad.py", bad)]);
    let (code, out) = check(&root, &["--shapes", "bad.py"]);
    assert_eq!(code, 1, "{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines,
        [
            "bad.py:4:9: error: y: matmul dimension mismatch: a last dim 3 != b second-to-last dim 4",
            "bad.py:5: z: [m, 3]",
        ]
    );
    // A conflict of two different names is a warning, which does not fail the check.
    let names = "from jaxtyping import Float, Array\n\ndef f(a: Float[Array, \"m k\"], b: Float[Array, \"n k\"]):\n    y = a @ b\n";
    let (_dir, root) = project(&[("names.py", names)]);
    let (code, out) = check(&root, &["names.py"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(
        out.trim(),
        "names.py:4:9: warning: y: matmul dimension mismatch: a last dim k != b second-to-last dim n"
    );
    let (code, _) = check(&root, &[]);
    assert_eq!(code, 2);
}

/// The server resolves a `src` layout of a workspace folder, but not a `src`
/// that is a package itself.
#[test]
fn server_resolves_a_src_layout() {
    let helper = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"]) -> Float[Array, \"b 5\"]:\n    return x\n";
    let user = "from jaxtyping import Float, Array\nimport pkg.other\n\ndef f(x: Float[Array, \"b 4\"], v: Float[Array, \"4 3\"]):\n    y = pkg.other.proj(x) @ v\n";
    for (package, want) in [(false, 1), (true, 0)] {
        let (_dir, root) = project(&[]);
        std::fs::create_dir_all(root.join("src/pkg")).unwrap();
        std::fs::write(root.join("src/pkg/__init__.py"), "").unwrap();
        std::fs::write(root.join("src/pkg/other.py"), helper).unwrap();
        if package {
            std::fs::write(root.join("src/__init__.py"), "").unwrap();
        }
        let mut c = Client::start(&root, json!({}));
        let got = c.open(&file_uri(&root.join("src/pkg/mod.py")), user);
        assert_eq!(got.len(), want, "src is a package: {package}: {got:?}");
    }
}
