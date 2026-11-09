//! A stdio LSP client for the integration tests.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

pub const BIN: &str = env!("CARGO_BIN_EXE_ndim-lsp");

pub fn frame(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap();
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend(body);
    out
}

pub fn file_uri(p: &Path) -> String {
    format!("file://{}", p.display())
}

pub struct Client {
    child: Child,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    rx: Receiver<Value>,
    next_id: i64,
    /// Notifications from the server, in arrival order.
    pub notes: Vec<Value>,
    resp: Vec<Value>,
    /// The result of `initialize`.
    pub init: Value,
}

impl Client {
    /// Start the server with `root` as the workspace folder. Requests from the
    /// server get a null result.
    pub fn start(root: &Path, capabilities: Value) -> Client {
        let mut child = Command::new(BIN)
            .current_dir(root)
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
            notes: Vec::new(),
            resp: Vec::new(),
            init: Value::Null,
        };
        let root_uri = file_uri(root);
        c.init = c.request(
            "initialize",
            json!({"processId": null, "rootUri": root_uri,
                   "workspaceFolders": [{"uri": root_uri, "name": "t"}],
                   "capabilities": capabilities}),
        );
        c.notify("initialized", json!({}));
        c
    }

    fn write(&self, v: &Value) {
        let mut g = self.stdin.lock().unwrap();
        let w = g.as_mut().unwrap();
        w.write_all(&frame(v)).unwrap();
        w.flush().unwrap();
    }

    pub fn notify(&self, method: &str, params: Value) {
        self.write(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    pub fn send_request(&mut self, method: &str, params: Value) -> i64 {
        self.next_id += 1;
        let id = self.next_id;
        self.write(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        id
    }

    /// The result of a request.
    pub fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.send_request(method, params);
        self.wait_response(id)["result"].clone()
    }

    pub fn pump(&mut self, timeout: Duration) -> bool {
        let Ok(v) = self.rx.recv_timeout(timeout) else {
            return false;
        };
        if v.get("method").is_some() {
            self.notes.push(v);
        } else {
            self.resp.push(v);
        }
        true
    }

    pub fn wait_response(&mut self, id: i64) -> Value {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(i) = self.resp.iter().position(|r| r["id"] == json!(id)) {
                return self.resp.remove(i);
            }
            assert!(Instant::now() < end, "response {id} hung");
            self.pump(Duration::from_millis(20));
        }
    }

    pub fn settle(&mut self, ms: u64) {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            self.pump(Duration::from_millis(10));
        }
    }

    /// The number of diagnostics publications so far.
    pub fn published(&self) -> usize {
        self.notes
            .iter()
            .filter(|n| n["method"] == "textDocument/publishDiagnostics")
            .count()
    }

    /// The diagnostics of the first publication for `uri` after `from`
    /// publications, or `None` after `timeout`.
    pub fn wait_diagnostics_for(&mut self, uri: &str, from: usize, timeout: Duration) -> Option<Vec<Value>> {
        let end = Instant::now() + timeout;
        loop {
            let found = self
                .notes
                .iter()
                .filter(|n| n["method"] == "textDocument/publishDiagnostics")
                .skip(from)
                .find(|n| n["params"]["uri"] == uri);
            if let Some(n) = found {
                return Some(n["params"]["diagnostics"].as_array().cloned().unwrap_or_default());
            }
            if Instant::now() >= end {
                return None;
            }
            self.pump(Duration::from_millis(20));
        }
    }

    pub fn wait_diagnostics(&mut self, uri: &str, from: usize) -> Vec<Value> {
        self.wait_diagnostics_for(uri, from, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no diagnostics for {uri}"))
    }

    /// Open `uri` and return its first diagnostics.
    pub fn open(&mut self, uri: &str, text: &str) -> Vec<Value> {
        let from = self.published();
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": text}}),
        );
        self.wait_diagnostics(uri, from)
    }

    /// Replace the text of `uri`, without a wait.
    pub fn change(&self, uri: &str, version: i64, text: &str) {
        self.notify(
            "textDocument/didChange",
            json!({"textDocument": {"uri": uri, "version": version},
                   "contentChanges": [{"text": text}]}),
        );
    }

    /// `textDocument/<method>` at `(line, character)` of `uri`.
    pub fn at(&mut self, method: &str, uri: &str, line: u32, character: u32, extra: Value) -> Value {
        let mut params = json!({"textDocument": {"uri": uri},
                                "position": {"line": line, "character": character}});
        if let Value::Object(m) = extra {
            params.as_object_mut().unwrap().extend(m);
        }
        self.request(&format!("textDocument/{method}"), params)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A temporary workspace, canonicalized, with `files` written in it.
pub fn project(files: &[(&str, &str)]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    for (name, text) in files {
        std::fs::write(root.join(name), text).unwrap();
    }
    (dir, root)
}
