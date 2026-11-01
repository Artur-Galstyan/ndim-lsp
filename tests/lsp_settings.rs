//! Editor options through the real server: their defaults, and live changes
//! with `workspace/didChangeConfiguration`.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ndim-lsp");
const URI: &str = "file:///tmp/ndim-settings-test/main.py";

const ERROR: i64 = 1;
const WARNING: i64 = 2;
const HINT: i64 = 4;

const TEXT: &str = "import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

def f(a: Float[Array, \"2 3\"], b: Float[Array, \"4 5\"], c: Float[Array, \"2 3 4\"], d: Float[Array, \"5 4 6\"], e: Float[Array, \"αααα ββββ γγγγ\"]):
    y = jnp.matmul(a, b)
    z = jax.lax.dot_general(c, d, (((2,), (1,)), ((0,), (0,))))
    p = a
    q = e
    return p, q
";

const P_LABEL: &str = ": [2, 3]";
const Q_LABEL: &str = ": [αααα, ββββ, γγγγ]";
const Q_TOOLTIP: &str = "q: Float[Array, \"αααα ββββ γγγγ\"]";

fn full_capabilities() -> Value {
    json!({"workspace": {"configuration": true, "inlayHint": {"refreshSupport": true}}})
}

struct Client {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    next_id: i64,
    seen: Vec<Value>,
    responses: Vec<Value>,
    pulls: Vec<Value>,
    version: i64,
}

fn frame(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap();
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend(body);
    out
}

impl Client {
    fn start(options: Value, capabilities: Value) -> Client {
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let mut r = BufReader::new(out);
            loop {
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let l = line.trim();
                    if l.is_empty() {
                        break;
                    }
                    if let Some(n) = l.strip_prefix("Content-Length:") {
                        len = n.trim().parse().unwrap();
                    }
                }
                let mut buf = vec![0u8; len];
                if r.read_exact(&mut buf).is_err() {
                    return;
                }
                if tx.send(serde_json::from_slice(&buf).unwrap()).is_err() {
                    return;
                }
            }
        });
        let mut c = Client {
            child,
            stdin,
            rx,
            next_id: 0,
            seen: Vec::new(),
            responses: Vec::new(),
            pulls: Vec::new(),
            version: 1,
        };
        let id = c.request(
            "initialize",
            json!({"processId": null, "rootUri": null, "capabilities": capabilities,
                   "initializationOptions": options}),
        );
        c.response(id);
        c.notify("initialized", json!({}));
        c
    }

    fn write(&mut self, message: Value) {
        self.stdin.write_all(&frame(&message)).unwrap();
        self.stdin.flush().unwrap();
    }

    fn notify(&mut self, method: &str, params: Value) {
        self.write(json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    fn request(&mut self, method: &str, params: Value) -> i64 {
        self.next_id += 1;
        let id = self.next_id;
        self.write(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        id
    }

    /// `workspace/configuration` requests wait in `pulls` for the test to
    /// answer. Other server requests get an empty reply.
    fn pump(&mut self, timeout: Duration) {
        let Ok(v) = self.rx.recv_timeout(timeout) else {
            return;
        };
        match (v.get("method").and_then(Value::as_str), v.get("id")) {
            (Some("workspace/configuration"), Some(_)) => self.pulls.push(v.clone()),
            (Some(_), Some(id)) => {
                let id = id.clone();
                self.write(json!({"jsonrpc": "2.0", "id": id, "result": null}));
            }
            (Some(_), None) => {}
            (None, _) => {
                self.responses.push(v);
                return;
            }
        }
        self.seen.push(v);
    }

    fn response(&mut self, id: i64) -> Value {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(i) = self.responses.iter().position(|r| r["id"] == json!(id)) {
                return self.responses.remove(i);
            }
            assert!(Instant::now() < end, "response {id} hung");
            self.pump(Duration::from_millis(20));
        }
    }

    fn settle(&mut self, ms: u64) {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            self.pump(Duration::from_millis(10));
        }
    }

    fn count(&self, method: &str) -> usize {
        self.seen.iter().filter(|m| m["method"] == method).count()
    }

    fn analyses(&self) -> usize {
        self.seen
            .iter()
            .filter(|m| {
                m["method"] == "window/logMessage"
                    && m["params"]["message"]
                        .as_str()
                        .is_some_and(|s| s.starts_with("analysis:"))
            })
            .count()
    }

    fn wait_for(&mut self, method: &str, count: usize) {
        let end = Instant::now() + Duration::from_secs(5);
        while self.count(method) < count {
            assert!(Instant::now() < end, "no {method} number {count}");
            self.pump(Duration::from_millis(10));
        }
    }

    fn wait_for_pull(&mut self, count: usize) {
        let end = Instant::now() + Duration::from_secs(5);
        while self.pulls.len() < count {
            assert!(Instant::now() < end, "no workspace/configuration number {count}");
            self.pump(Duration::from_millis(10));
        }
    }

    fn open(&mut self) {
        let before = self.count("textDocument/publishDiagnostics");
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": URI, "languageId": "python", "version": 1, "text": TEXT}}),
        );
        self.wait_for("textDocument/publishDiagnostics", before + 1);
    }

    fn change_settings(&mut self, settings: Value) {
        self.notify("workspace/didChangeConfiguration", json!({"settings": settings}));
    }

    /// Change settings and wait for the diagnostics of the change.
    fn change_and_wait(&mut self, settings: Value) {
        let before = self.count("textDocument/publishDiagnostics");
        self.change_settings(settings);
        self.wait_for("textDocument/publishDiagnostics", before + 1);
    }

    fn answer_pull(&mut self, index: usize, value: Value) {
        let id = self.pulls[index]["id"].clone();
        self.write(json!({"jsonrpc": "2.0", "id": id, "result": [value]}));
    }

    fn hints(&mut self) -> Vec<Value> {
        let id = self.request(
            "textDocument/inlayHint",
            json!({"textDocument": {"uri": URI},
                   "range": {"start": {"line": 0, "character": 0},
                             "end": {"line": 20, "character": 0}}}),
        );
        let result = self.response(id)["result"].clone();
        result.as_array().unwrap().clone()
    }

    fn labels(&mut self) -> Vec<String> {
        self.hints()
            .iter()
            .map(|h| h["label"].as_str().unwrap().to_string())
            .collect()
    }

    /// The severity of the newest publish, sorted.
    fn severities(&self) -> Vec<i64> {
        let last = self
            .seen
            .iter()
            .rfind(|m| m["method"] == "textDocument/publishDiagnostics")
            .expect("no publish");
        let mut all: Vec<i64> = last["params"]["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["severity"].as_i64().unwrap())
            .collect();
        all.sort();
        all
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn defaults_keep_every_hint_uncut() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    let labels = c.labels();
    assert!(labels.contains(&P_LABEL.to_string()), "{labels:?}");
    assert!(labels.contains(&Q_LABEL.to_string()), "{labels:?}");
    assert_eq!(c.severities(), vec![ERROR, WARNING]);
}

#[test]
fn inlay_hints_false_returns_no_hints() {
    let mut c = Client::start(json!({"inlay_hints": false}), json!({}));
    c.open();
    assert_eq!(c.hints(), Vec::<Value>::new());
}

#[test]
fn max_length_cuts_on_a_char_boundary_and_keeps_the_tooltip() {
    // ": [αααα" is seven chars, so a byte cut at this length would split a char.
    let mut c = Client::start(json!({"inlay_hints_max_length": 8}), json!({}));
    c.open();
    let hints = c.hints();
    let q = hints
        .iter()
        .find(|h| h["tooltip"] == Q_TOOLTIP)
        .expect("no hint for q");
    assert_eq!(q["label"], ": [αααα…");
    assert!(c.labels().contains(&P_LABEL.to_string()), "a short label stays");
}

#[test]
fn max_length_equal_to_the_label_cuts_nothing() {
    let max = Q_LABEL.chars().count();
    let mut c = Client::start(json!({"inlay_hints_max_length": max}), json!({}));
    c.open();
    assert!(c.labels().contains(&Q_LABEL.to_string()));
}

#[test]
fn max_length_of_one_leaves_only_the_ellipsis() {
    let mut c = Client::start(json!({"inlay_hints_max_length": 1}), json!({}));
    c.open();
    assert!(c.labels().iter().all(|l| l == "…"));
}

#[test]
fn invalid_values_use_the_defaults() {
    for options in [
        json!({"inlay_hints": "no", "inlay_hints_max_length": -1}),
        json!({"inlay_hints": 0, "inlay_hints_max_length": "4"}),
        json!({"inlay_hints": null, "inlay_hints_max_length": 1.5}),
    ] {
        let mut c = Client::start(options, json!({}));
        c.open();
        assert!(c.labels().contains(&Q_LABEL.to_string()));
    }
}

#[test]
fn live_change_under_the_section_key_applies_all_four_options() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.change_and_wait(json!({"ndim-lsp": {
        "diagnostic_severity": "hint",
        "approximation_severity": "error",
        "inlay_hints_max_length": 8,
    }}));
    assert_eq!(c.severities(), vec![ERROR, HINT]);
    assert!(c.labels().contains(&": [αααα…".to_string()));

    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints": false}}));
    assert_eq!(c.hints(), Vec::<Value>::new());
}

#[test]
fn live_change_at_the_top_level_applies() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.change_and_wait(json!({"diagnostic_severity": "warning"}));
    assert_eq!(c.severities(), vec![WARNING, WARNING]);
}

#[test]
fn an_invalid_live_value_uses_the_default() {
    let mut c = Client::start(json!({"diagnostic_severity": "hint"}), json!({}));
    c.open();
    c.change_and_wait(json!({"diagnostic_severity": "loud"}));
    assert_eq!(c.severities(), vec![ERROR, WARNING]);
}

#[test]
fn a_key_that_is_absent_keeps_its_value() {
    let mut c = Client::start(json!({"diagnostic_severity": "hint"}), json!({}));
    c.open();
    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints_max_length": 8}}));
    assert_eq!(c.severities(), vec![WARNING, HINT]);
}

#[test]
fn a_notification_with_no_value_pulls_the_section() {
    for settings in [json!(null), json!({}), json!({"ndim-lsp": {}}), json!({"python": {"a": 1}})] {
        let mut c = Client::start(json!({}), full_capabilities());
        c.open();
        c.change_settings(settings);
        c.wait_for_pull(1);
        assert_eq!(c.pulls[0]["params"]["items"][0]["section"], "ndim-lsp");
        c.answer_pull(0, json!({"diagnostic_severity": "warning"}));
        c.wait_for("textDocument/publishDiagnostics", 2);
        assert_eq!(c.severities(), vec![WARNING, WARNING]);
    }
}

#[test]
fn a_pull_with_no_value_changes_nothing() {
    let mut c = Client::start(json!({"diagnostic_severity": "hint"}), full_capabilities());
    c.open();
    c.change_settings(json!(null));
    c.wait_for_pull(1);
    c.answer_pull(0, json!(null));
    c.settle(300);
    assert_eq!(c.count("textDocument/publishDiagnostics"), 1);
    assert_eq!(c.count("workspace/inlayHint/refresh"), 0);
    assert_eq!(c.severities(), vec![WARNING, HINT]);
}

#[test]
fn no_pull_without_the_client_capability() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.change_settings(json!(null));
    c.settle(300);
    assert!(c.pulls.is_empty());
}

#[test]
fn a_notification_with_a_value_does_not_pull() {
    let mut c = Client::start(json!({}), full_capabilities());
    c.open();
    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints": false}}));
    c.settle(100);
    assert!(c.pulls.is_empty());
}

#[test]
fn an_old_pull_reply_does_not_overwrite_a_newer_value() {
    let mut c = Client::start(json!({}), full_capabilities());
    c.open();
    c.change_settings(json!(null));
    c.wait_for_pull(1);
    c.change_and_wait(json!({"ndim-lsp": {"diagnostic_severity": "hint"}}));
    c.answer_pull(0, json!({"diagnostic_severity": "warning"}));
    c.settle(300);
    assert_eq!(c.count("textDocument/publishDiagnostics"), 2);
    assert_eq!(c.severities(), vec![WARNING, HINT]);
}

#[test]
fn a_change_refreshes_hints_when_the_client_supports_it() {
    let mut c = Client::start(json!({}), full_capabilities());
    c.open();
    c.change_settings(json!({"ndim-lsp": {"inlay_hints_max_length": 8}}));
    c.wait_for("workspace/inlayHint/refresh", 1);
}

#[test]
fn no_refresh_without_the_client_capability() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints_max_length": 8}}));
    c.settle(100);
    assert_eq!(c.count("workspace/inlayHint/refresh"), 0);
}

#[test]
fn equal_settings_send_nothing() {
    let mut c = Client::start(json!({"inlay_hints_max_length": 8}), full_capabilities());
    c.open();
    c.change_settings(json!({"ndim-lsp": {"inlay_hints_max_length": 8}}));
    c.settle(300);
    assert_eq!(c.count("textDocument/publishDiagnostics"), 1);
    assert_eq!(c.count("workspace/inlayHint/refresh"), 0);
}

#[test]
fn a_hint_change_republishes_from_the_cached_analysis() {
    let mut c = Client::start(json!({}), full_capabilities());
    c.open();
    c.settle(100);
    let analyses = c.analyses();
    assert_eq!(analyses, 1);
    let before = c.severities();

    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints": false}}));
    c.change_and_wait(json!({"ndim-lsp": {"inlay_hints": true, "inlay_hints_max_length": 8}}));
    assert_eq!(c.severities(), before);
    assert!(c.labels().contains(&": [αααα…".to_string()));
    c.settle(300);
    assert_eq!(c.analyses(), analyses, "a hint change started an analysis");
}

#[test]
fn a_severity_change_starts_no_analysis() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.settle(100);
    let analyses = c.analyses();
    c.change_and_wait(json!({"diagnostic_severity": "hint"}));
    c.settle(300);
    assert_eq!(c.analyses(), analyses);
}

#[test]
fn a_change_during_the_debounce_applies_to_the_pending_publish() {
    let mut c = Client::start(json!({}), json!({}));
    c.open();
    c.settle(100);
    let analyses = c.analyses();
    let published = c.count("textDocument/publishDiagnostics");

    c.version += 1;
    let version = c.version;
    c.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": URI, "version": version},
               "contentChanges": [{"text": format!("{TEXT}\n")}]}),
    );
    c.change_settings(json!({"diagnostic_severity": "hint"}));
    c.wait_for("textDocument/publishDiagnostics", published + 1);
    c.settle(300);
    assert_eq!(c.severities(), vec![WARNING, HINT]);
    assert_eq!(c.analyses(), analyses + 1);
    assert_eq!(c.count("textDocument/publishDiagnostics"), published + 1);
}
