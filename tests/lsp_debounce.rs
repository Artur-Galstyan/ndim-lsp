//! Server scheduling: debounce, single-flight analysis, stale results, and
//! shutdown. Spawns the built `ndim-lsp` binary and counts the server's own
//! `analysis: total=` log lines.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ndim-lsp");
const URI: &str = "file:///tmp/ndim_debounce_test/main.py";
const BASE: &str = "from jaxtyping import Float, Array\nimport jax.numpy as jnp\n\ndef f(x: Float[Array, \"batch features\"]):\n    y = jnp.zeros_like(x)\n    return y\n";

struct Pub {
    version: Option<i64>,
    diags: usize,
}

struct Client {
    child: Child,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    rx: Receiver<Value>,
    next_id: i64,
    pubs: Vec<Pub>,
    analyses: usize,
    resp: Vec<Value>,
}

fn frame(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap();
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend(body);
    out
}

impl Client {
    fn start() -> Client {
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        let reply = stdin.clone();
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
                let v: Value = serde_json::from_slice(&buf).unwrap();
                if v.get("method").is_some() && v.get("id").is_some() {
                    let ack = json!({"jsonrpc": "2.0", "id": v["id"], "result": null});
                    if let Some(w) = reply.lock().unwrap().as_mut() {
                        w.write_all(&frame(&ack)).unwrap();
                    }
                    continue;
                }
                if tx.send(v).is_err() {
                    return;
                }
            }
        });
        let mut c = Client {
            child,
            stdin,
            rx,
            next_id: 0,
            pubs: Vec::new(),
            analyses: 0,
            resp: Vec::new(),
        };
        let id = c.send_request(
            "initialize",
            json!({"processId": null, "rootUri": null, "capabilities": {}}),
        );
        c.wait_response(id);
        c.notify("initialized", json!({}));
        c
    }

    fn write(&self, bytes: &[u8]) {
        let mut g = self.stdin.lock().unwrap();
        let w = g.as_mut().unwrap();
        w.write_all(bytes).unwrap();
        w.flush().unwrap();
    }

    fn notify(&self, method: &str, params: Value) {
        self.write(&frame(&json!({"jsonrpc": "2.0", "method": method, "params": params})));
    }

    fn send_request(&mut self, method: &str, params: Value) -> i64 {
        self.next_id += 1;
        let id = self.next_id;
        self.write(&frame(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
        ));
        id
    }

    fn route(&mut self, v: Value) {
        match v.get("method").and_then(|m| m.as_str()) {
            Some("textDocument/publishDiagnostics") => self.pubs.push(Pub {
                version: v["params"]["version"].as_i64(),
                diags: v["params"]["diagnostics"].as_array().map_or(0, |d| d.len()),
            }),
            Some("window/logMessage") => {
                if v["params"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with("analysis: total="))
                {
                    self.analyses += 1;
                }
            }
            Some(_) => {}
            None => self.resp.push(v),
        }
    }

    fn pump(&mut self, timeout: Duration) -> bool {
        match self.rx.recv_timeout(timeout) {
            Ok(v) => {
                self.route(v);
                true
            }
            Err(_) => false,
        }
    }

    fn settle(&mut self, ms: u64) {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            self.pump(Duration::from_millis(10));
        }
    }

    fn wait_response(&mut self, id: i64) -> Value {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(i) = self.resp.iter().position(|r| r["id"] == json!(id)) {
                return self.resp.remove(i);
            }
            assert!(Instant::now() < end, "response {id} hung");
            self.pump(Duration::from_millis(20));
        }
    }

    fn wait_version(&mut self, version: i64) {
        let end = Instant::now() + Duration::from_secs(10);
        while !self.pubs.iter().any(|p| p.version == Some(version)) {
            assert!(Instant::now() < end, "no diagnostics for v{version}");
            self.pump(Duration::from_millis(20));
        }
    }

    fn open(&mut self) {
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": URI, "languageId": "python", "version": 1, "text": BASE}}),
        );
        self.wait_version(1);
        self.settle(100);
    }

    fn change_frame(version: i64) -> Vec<u8> {
        Self::change_frame_of(BASE, version)
    }

    fn change_frame_of(base: &str, version: i64) -> Vec<u8> {
        frame(&json!({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
            "textDocument": {"uri": URI, "version": version},
            "contentChanges": [{"text": format!("{base}# {version}\n")}]}}))
    }

    fn change(&self, version: i64) {
        self.write(&Self::change_frame(version));
    }

    fn inlay(&mut self) -> i64 {
        self.send_request(
            "textDocument/inlayHint",
            json!({"textDocument": {"uri": URI},
                   "range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}}}),
        )
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn burst_of_changes_runs_one_analysis_and_publishes_last_version() {
    let mut c = Client::start();
    c.open();
    let before = c.analyses;
    let from = c.pubs.len();
    let burst: Vec<u8> = (2..=51).flat_map(Client::change_frame).collect();
    c.write(&burst);
    c.wait_version(51);
    c.settle(400);
    let versions: Vec<i64> = c.pubs[from..].iter().filter_map(|p| p.version).collect();
    assert!(
        versions.windows(2).all(|w| w[0] < w[1]) && versions.last() == Some(&51),
        "published {versions:?}"
    );
    assert!(
        c.analyses - before <= 2,
        "{} analyses for one burst",
        c.analyses - before
    );
}

#[test]
fn inlay_requests_while_typing_share_one_analysis() {
    let mut c = Client::start();
    c.open();
    let before = c.analyses;
    let mut ids = Vec::new();
    for v in 2..=13 {
        c.change(v);
        ids.push(c.inlay());
        c.settle(100);
    }
    c.wait_version(13);
    let last = ids.pop().unwrap();
    let hints = c.wait_response(last);
    assert!(
        hints["result"].as_array().is_some_and(|h| !h.is_empty()),
        "the request after the last key must get fresh hints: {hints}"
    );
    for id in ids {
        c.wait_response(id);
    }
    c.settle(400);
    assert!(
        c.analyses - before <= 1,
        "{} analyses for 12 keystrokes",
        c.analyses - before
    );
}

#[test]
fn request_during_burst_is_answered_and_never_stale() {
    let mut c = Client::start();
    c.open();
    for v in 2..=20 {
        c.change(v);
    }
    let id = c.inlay();
    c.wait_response(id);
    c.wait_version(20);
    let before = c.analyses;
    let id = c.inlay();
    let hints = c.wait_response(id);
    assert!(hints["result"].as_array().is_some_and(|h| !h.is_empty()));
    assert_eq!(c.analyses, before, "a cached version must not re-analyze");
}

#[test]
fn close_during_analysis_publishes_only_the_clear() {
    let mut c = Client::start();
    c.open();
    c.change(2);
    c.settle(160);
    c.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": URI}}),
    );
    c.settle(800);
    let last = c.pubs.last().unwrap();
    assert!(
        last.version.is_none() && last.diags == 0,
        "the last publication must be the clear"
    );
}

#[test]
fn shutdown_during_analysis_answers_and_exits() {
    let mut c = Client::start();
    c.open();
    c.change(2);
    c.settle(160);
    let id = c.send_request("shutdown", json!(null));
    c.wait_response(id);
    c.notify("exit", json!(null));
    c.stdin.lock().unwrap().take();
    let end = Instant::now() + Duration::from_secs(5);
    while c.child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < end, "server did not exit");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A text whose analysis takes longer than the pause between keystrokes.
fn slow_text() -> String {
    let mut text = BASE.to_string();
    for i in 0..SLOW_COPIES {
        text.push_str(&format!(
            "\ndef g{i}(x: Float[Array, \"batch features\"]):\n    y = jnp.zeros_like(x)\n    z = y + x\n    return z\n"
        ));
    }
    text
}

const SLOW_COPIES: usize = 4000;

#[test]
fn slow_analyses_do_not_pile_up_while_typing() {
    let mut c = Client::start();
    let base = slow_text();
    c.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": URI, "languageId": "python", "version": 1, "text": base}}),
    );
    c.wait_version(1);
    c.settle(100);
    let before = c.analyses;
    for v in 2..=9 {
        c.write(&Client::change_frame_of(&base, v));
        c.settle(250);
    }
    c.wait_version(9);
    c.settle(400);
    assert!(
        c.analyses - before <= 3,
        "{} analyses for 8 slow keystrokes",
        c.analyses - before
    );
}
