//! A saved, created, or deleted imported file must give the same result as a
//! fresh server, and open documents must get new diagnostics.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ndim-lsp");

const HELPER_OK: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const HELPER_BAD: &str = "from jaxtyping import Float, Array\n\ndef proj(x: Float[Array, \"b d\"], w: Float[Array, \"k d\"]) -> Float[Array, \"b k\"]:\n    return x @ w\n";
const MAIN: &str = "from jaxtyping import Float, Array\nfrom helper import proj\n\ndef f(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]):\n    y = proj(x, w)\n    return y\n";

struct Client {
    child: Child,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    rx: Receiver<Value>,
    next_id: i64,
    pubs: Vec<(String, Vec<String>)>,
    resp: Vec<Value>,
    version: i64,
    uri: String,
}

fn frame(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap();
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend(body);
    out
}

fn file_uri(p: &Path) -> String {
    format!("file://{}", p.display())
}

impl Client {
    fn start(root: &Path) -> Client {
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
                        let _ = w.write_all(&frame(&ack));
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
            resp: Vec::new(),
            version: 0,
            uri: file_uri(&root.join("main.py")),
        };
        let root_uri = file_uri(root);
        let id = c.send_request(
            "initialize",
            json!({"processId": null, "rootUri": root_uri,
                   "workspaceFolders": [{"uri": root_uri, "name": "t"}],
                   "capabilities": {}}),
        );
        c.wait_response(id);
        c.notify("initialized", json!({}));
        c
    }

    fn notify(&self, method: &str, params: Value) {
        let bytes = frame(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
        let mut g = self.stdin.lock().unwrap();
        let w = g.as_mut().unwrap();
        w.write_all(&bytes).unwrap();
        w.flush().unwrap();
    }

    fn send_request(&mut self, method: &str, params: Value) -> i64 {
        self.next_id += 1;
        let id = self.next_id;
        let bytes = frame(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut g = self.stdin.lock().unwrap();
        let w = g.as_mut().unwrap();
        w.write_all(&bytes).unwrap();
        w.flush().unwrap();
        id
    }

    fn pump(&mut self, timeout: Duration) -> bool {
        let Ok(v) = self.rx.recv_timeout(timeout) else {
            return false;
        };
        match v.get("method").and_then(|m| m.as_str()) {
            Some("textDocument/publishDiagnostics") => {
                let msgs = v["params"]["diagnostics"]
                    .as_array()
                    .map(|d| {
                        d.iter()
                            .map(|x| x["message"].as_str().unwrap_or("").to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                let uri = v["params"]["uri"].as_str().unwrap_or("").to_string();
                self.pubs.push((uri, msgs));
            }
            Some(_) => {}
            None => self.resp.push(v),
        }
        true
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

    fn settle(&mut self, ms: u64) {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            self.pump(Duration::from_millis(10));
        }
    }

    /// Messages of the first publish for the main document after index `from`.
    fn wait_publish(&mut self, from: usize) -> Option<Vec<String>> {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some((_, m)) = self.pubs[from..].iter().find(|(u, _)| *u == self.uri) {
                return Some(m.clone());
            }
            if Instant::now() >= end {
                return None;
            }
            self.pump(Duration::from_millis(20));
        }
    }

    fn open(&mut self, text: &str) -> Vec<String> {
        self.version = 1;
        let from = self.pubs.len();
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": self.uri, "languageId": "python",
                                    "version": 1, "text": text}}),
        );
        let msgs = self.wait_publish(from).expect("no diagnostics after open");
        self.settle(300);
        msgs
    }

    fn change(&mut self, text: &str) -> Vec<String> {
        self.version += 1;
        let from = self.pubs.len();
        self.notify(
            "textDocument/didChange",
            json!({"textDocument": {"uri": self.uri, "version": self.version},
                   "contentChanges": [{"text": text}]}),
        );
        self.wait_publish(from)
            .expect("no diagnostics after change")
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn project(helper: Option<&str>) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    if let Some(h) = helper {
        fs::write(root.join("helper.py"), h).unwrap();
    }
    (dir, root)
}

fn fresh(root: &Path, text: &str) -> Vec<String> {
    Client::start(root).open(text)
}

#[test]
fn save_of_imported_file_republishes_like_a_fresh_server() {
    let (_dir, root) = project(Some(HELPER_OK));
    let helper = root.join("helper.py");
    let mut c = Client::start(&root);
    assert!(c.open(MAIN).is_empty());

    fs::write(&helper, HELPER_BAD).unwrap();
    let from = c.pubs.len();
    c.notify(
        "textDocument/didSave",
        json!({"textDocument": {"uri": file_uri(&helper)}}),
    );
    let got = c.wait_publish(from).expect("no republish after save");
    assert!(!got.is_empty(), "stale result after save");
    assert_eq!(got, fresh(&root, MAIN));
}

#[test]
fn watched_file_change_republishes_like_a_fresh_server() {
    let (_dir, root) = project(Some(HELPER_OK));
    let helper = root.join("helper.py");
    let mut c = Client::start(&root);
    c.open(MAIN);

    fs::write(&helper, HELPER_BAD).unwrap();
    let from = c.pubs.len();
    c.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": file_uri(&helper), "type": 2}]}),
    );
    let got = c
        .wait_publish(from)
        .expect("no republish after watch event");
    assert_eq!(got, fresh(&root, MAIN));
}

#[test]
fn change_without_notification_is_seen_at_the_next_analysis() {
    let (_dir, root) = project(Some(HELPER_OK));
    let mut c = Client::start(&root);
    c.open(MAIN);

    fs::write(root.join("helper.py"), HELPER_BAD).unwrap();
    let text = format!("{MAIN}\n");
    let got = c.change(&text);
    assert!(!got.is_empty(), "stale result without any notification");
    assert_eq!(got, fresh(&root, &text));
}

#[test]
fn deleted_imported_file_gives_the_fresh_result() {
    let (_dir, root) = project(Some(HELPER_BAD));
    let helper = root.join("helper.py");
    let mut c = Client::start(&root);
    assert!(!c.open(MAIN).is_empty());

    fs::remove_file(&helper).unwrap();
    let from = c.pubs.len();
    c.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": file_uri(&helper), "type": 3}]}),
    );
    let got = c.wait_publish(from).expect("no republish after delete");
    assert!(got.is_empty(), "stale result after delete: {got:?}");
    assert_eq!(got, fresh(&root, MAIN));
}

#[test]
fn created_imported_file_gives_the_fresh_result() {
    let (_dir, root) = project(None);
    let mut c = Client::start(&root);
    assert!(c.open(MAIN).is_empty());

    let helper = root.join("helper.py");
    fs::write(&helper, HELPER_BAD).unwrap();
    let from = c.pubs.len();
    c.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": file_uri(&helper), "type": 1}]}),
    );
    let got = c.wait_publish(from).expect("no republish after create");
    assert_eq!(got, fresh(&root, MAIN));
    assert!(!got.is_empty());
}

#[test]
fn save_of_unrelated_file_does_not_republish() {
    let (_dir, root) = project(Some(HELPER_OK));
    let other = root.join("other.py");
    fs::write(&other, "x = 1\n").unwrap();
    let mut c = Client::start(&root);
    c.open(MAIN);

    fs::write(&other, "x = 2\n").unwrap();
    let from = c.pubs.len();
    c.notify(
        "textDocument/didSave",
        json!({"textDocument": {"uri": file_uri(&other)}}),
    );
    c.settle(600);
    assert_eq!(c.pubs.len(), from, "republished for an unrelated file");
}

#[test]
fn added_workspace_folder_republishes_open_documents() {
    let (_a, root_a) = project(None);
    let (_b, root_b) = project(Some(HELPER_BAD));
    let mut c = Client::start(&root_a);
    assert!(c.open(MAIN).is_empty());

    let from = c.pubs.len();
    c.notify(
        "workspace/didChangeWorkspaceFolders",
        json!({"event": {"added": [{"uri": file_uri(&root_b), "name": "b"}], "removed": []}}),
    );
    let got = c
        .wait_publish(from)
        .expect("no republish after the folder change");
    assert!(!got.is_empty(), "stale result after the folder change");
}
