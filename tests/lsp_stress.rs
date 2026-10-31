//! End-to-end LSP stress benchmark and golden dump.
//!
//! Spawns the built `ndim-lsp` binary and talks JSON-RPC over stdio against the
//! generated project in `bench/stress/` (regenerate with
//! `python3 bench/generate_stress.py`).
//!
//! Both tests are `#[ignore]` so the normal suite stays fast:
//!
//! ```text
//! cargo test --release --test lsp_stress lsp_stress_benchmark -- --ignored --nocapture
//! NDIM_GOLDEN_OUT=/tmp/g.json cargo test --release --test lsp_stress golden_dump -- --ignored --nocapture
//! ```
//!
//! Env: `NDIM_STRESS_RUNS` (default 5), `NDIM_BENCH_OUT` (JSON results path),
//! `NDIM_GOLDEN_OUT` (golden dump path).
//!
//! Latency is reported as total = debounce + server compute + other. The
//! debounce is the fixed 150 ms in `Backend::did_change` (0 for didOpen).
//! Server compute is parsed from the server's own `analysis: total=Nms` log
//! (whole-millisecond resolution). Other is transport, JSON, diagnostic
//! building and scheduling. Peak RSS is sampled with `ps` every 25 ms (and
//! `VmHWM` on Linux), so it is a lower bound on the true peak.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ndim-lsp");
const DEBOUNCE_MS: f64 = 150.0;
const WAIT: Duration = Duration::from_secs(240);

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------- transport

struct Msg {
    at: Instant,
    v: Value,
}

struct Pub {
    at: Instant,
    uri: String,
    version: Option<i64>,
    diags: Vec<Value>,
}

fn write_body(w: &Mutex<ChildStdin>, body: &[u8]) {
    let mut g = w.lock().unwrap();
    write!(g, "Content-Length: {}\r\n\r\n", body.len()).unwrap();
    g.write_all(body).unwrap();
    g.flush().unwrap();
}

fn reader_thread(out: ChildStdout, tx: Sender<Msg>, stdin: Arc<Mutex<ChildStdin>>) {
    let mut r = BufReader::new(out);
    loop {
        let mut len = None;
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
                len = n.trim().parse::<usize>().ok();
            }
        }
        let Some(n) = len else { return };
        let mut buf = vec![0u8; n];
        if r.read_exact(&mut buf).is_err() {
            return;
        }
        let at = Instant::now();
        let v: Value = serde_json::from_slice(&buf).unwrap();
        if v.get("method").is_some() && v.get("id").is_some() {
            let reply = json!({"jsonrpc": "2.0", "id": v["id"], "result": null});
            write_body(&stdin, &serde_json::to_vec(&reply).unwrap());
            continue;
        }
        if tx.send(Msg { at, v }).is_err() {
            return;
        }
    }
}

fn ps_rss_kb(pid: u32) -> u64 {
    let mut best = 0;
    if let Ok(o) = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        best = String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse()
            .unwrap_or(0);
    }
    if let Ok(s) = fs::read_to_string(format!("/proc/{pid}/status")) {
        for l in s.lines() {
            if let Some(r) = l.strip_prefix("VmHWM:") {
                let kb: u64 = r.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                best = best.max(kb);
            }
        }
    }
    best
}

struct Server {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    rx: Receiver<Msg>,
    next_id: i64,
    pubs: Vec<Pub>,
    logs: Vec<(Instant, String)>,
    resp: HashMap<i64, Msg>,
    rss_kb: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    sampler: Option<JoinHandle<()>>,
    init_ms: f64,
}

impl Server {
    fn spawn(root: &Path) -> Server {
        let t_spawn = Instant::now();
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn ndim-lsp");
        let pid = child.id();
        let stdin = Arc::new(Mutex::new(child.stdin.take().unwrap()));
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        let s2 = stdin.clone();
        std::thread::spawn(move || reader_thread(out, tx, s2));
        let rss_kb = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (r2, st2) = (rss_kb.clone(), stop.clone());
        let sampler = std::thread::spawn(move || {
            while !st2.load(Ordering::Relaxed) {
                r2.fetch_max(ps_rss_kb(pid), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(25));
            }
        });
        let mut s = Server {
            child,
            stdin,
            rx,
            next_id: 0,
            pubs: Vec::new(),
            logs: Vec::new(),
            resp: HashMap::new(),
            rss_kb,
            stop,
            sampler: Some(sampler),
            init_ms: 0.0,
        };
        let root_uri = format!("file://{}", root.display());
        let _ = s.request(
            "initialize",
            json!({
                "processId": null,
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": "stress"}],
                "capabilities": {"textDocument": {"publishDiagnostics": {"versionSupport": true}}},
            }),
        );
        s.init_ms = t_spawn.elapsed().as_secs_f64() * 1000.0;
        s.notify("initialized", json!({}));
        s
    }

    fn route(&mut self, m: Msg) {
        if let Some(method) = m.v.get("method").and_then(|x| x.as_str()) {
            let p = &m.v["params"];
            match method {
                "textDocument/publishDiagnostics" => self.pubs.push(Pub {
                    at: m.at,
                    uri: p["uri"].as_str().unwrap_or("").to_string(),
                    version: p["version"].as_i64(),
                    diags: p["diagnostics"].as_array().cloned().unwrap_or_default(),
                }),
                "window/logMessage" => self
                    .logs
                    .push((m.at, p["message"].as_str().unwrap_or("").to_string())),
                _ => {}
            }
        } else if let Some(id) = m.v.get("id").and_then(|x| x.as_i64()) {
            self.resp.insert(id, m);
        }
    }

    fn pump(&mut self, timeout: Duration) -> bool {
        match self.rx.recv_timeout(timeout) {
            Ok(m) => {
                self.route(m);
                true
            }
            Err(_) => false,
        }
    }

    fn settle(&mut self, ms: u64) {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            self.pump(Duration::from_millis(20));
        }
    }

    /// Serialize first, stamp the clock, then write: JSON encoding is not billed.
    fn send_timed(&self, v: &Value) -> Instant {
        let body = serde_json::to_vec(v).unwrap();
        let t0 = Instant::now();
        write_body(&self.stdin, &body);
        t0
    }

    fn notify(&self, method: &str, params: Value) {
        self.send_timed(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    fn request_send(&mut self, method: &str, params: Value) -> (i64, Instant) {
        self.next_id += 1;
        let id = self.next_id;
        let t0 = self
            .send_timed(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        (id, t0)
    }

    fn request_wait(&mut self, method: &str, id: i64, t0: Instant) -> (Value, f64) {
        loop {
            if let Some(m) = self.resp.remove(&id) {
                let ms = m.at.saturating_duration_since(t0).as_secs_f64() * 1000.0;
                if let Some(e) = m.v.get("error") {
                    panic!("{method} failed: {e}");
                }
                return (m.v["result"].clone(), ms);
            }
            if !self.pump(WAIT) {
                panic!("timeout waiting for response to {method}");
            }
        }
    }

    fn request(&mut self, method: &str, params: Value) -> (Value, f64) {
        let (id, t0) = self.request_send(method, params);
        self.request_wait(method, id, t0)
    }

    fn wait_pub(&mut self, uri: &str, version: i64, from: usize) -> usize {
        loop {
            if let Some(i) = self.pubs[from..]
                .iter()
                .position(|p| p.uri == uri && p.version == Some(version))
            {
                return from + i;
            }
            if !self.pump(WAIT) {
                panic!("timeout waiting for diagnostics of {uri} v{version}");
            }
        }
    }

    /// Server-reported compute (ms) of the last `analysis:` log in (t0, t1].
    fn compute_ms(&self, t0: Instant, t1: Instant) -> f64 {
        self.logs
            .iter()
            .rev()
            .find(|(at, m)| *at >= t0 && *at <= t1 && m.starts_with("analysis: total="))
            .and_then(|(_, m)| {
                m["analysis: total=".len()..]
                    .split("ms")
                    .next()?
                    .parse()
                    .ok()
            })
            .unwrap_or(0.0)
    }

    fn analyses_between(&self, t0: Instant, t1: Instant) -> usize {
        self.logs
            .iter()
            .filter(|(at, m)| *at >= t0 && *at <= t1 && m.starts_with("analysis: total="))
            .count()
    }

    fn rss_mb(&self) -> f64 {
        self.rss_kb
            .fetch_max(ps_rss_kb(self.child.id()), Ordering::Relaxed);
        self.rss_kb.load(Ordering::Relaxed) as f64 / 1024.0
    }

    fn kill(mut self) -> f64 {
        let peak = self.rss_mb();
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(h) = self.sampler.take() {
            let _ = h.join();
        }
        peak
    }
}

// ---------------------------------------------------------------- documents

struct Doc {
    uri: String,
    version: i64,
    text: String,
    diags: Vec<Value>,
}

#[derive(Clone, Copy, Default)]
struct Sample {
    total: f64,
    compute: f64,
    debounce: f64,
}

fn open_doc(s: &mut Server, uri: &str, text: &str) -> (Doc, Sample) {
    let from = s.pubs.len();
    let t0 = s.send_timed(
        &json!({"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": {
        "textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": text}}}),
    );
    let i = s.wait_pub(uri, 1, from);
    let at = s.pubs[i].at;
    let sample = Sample {
        total: at.saturating_duration_since(t0).as_secs_f64() * 1000.0,
        compute: s.compute_ms(t0, at),
        debounce: 0.0,
    };
    let doc = Doc {
        uri: uri.to_string(),
        version: 1,
        text: text.to_string(),
        diags: s.pubs[i].diags.clone(),
    };
    (doc, sample)
}

fn change_doc(s: &mut Server, d: &mut Doc, text: String) -> Sample {
    d.version += 1;
    let from = s.pubs.len();
    let t0 = s.send_timed(
        &json!({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
        "textDocument": {"uri": d.uri, "version": d.version},
        "contentChanges": [{"text": text}]}}),
    );
    let i = s.wait_pub(&d.uri.clone(), d.version, from);
    let at = s.pubs[i].at;
    d.text = text;
    d.diags = s.pubs[i].diags.clone();
    Sample {
        total: at.saturating_duration_since(t0).as_secs_f64() * 1000.0,
        compute: s.compute_ms(t0, at),
        debounce: DEBOUNCE_MS,
    }
}

fn line_count(text: &str) -> usize {
    text.lines().count()
}

fn full_hints(s: &mut Server, d: &Doc) -> (Vec<Value>, f64) {
    let (r, ms) = s.request(
        "textDocument/inlayHint",
        json!({"textDocument": {"uri": d.uri},
               "range": {"start": {"line": 0, "character": 0},
                         "end": {"line": line_count(&d.text) + 1, "character": 0}}}),
    );
    (r.as_array().cloned().unwrap_or_default(), ms)
}

fn sorted_hints(mut h: Vec<Value>) -> Vec<Value> {
    h.sort_by_key(|x| {
        (
            x["position"]["line"].as_u64().unwrap_or(0),
            x["position"]["character"].as_u64().unwrap_or(0),
            x["label"].to_string(),
            x["tooltip"].to_string(),
        )
    });
    h
}

fn sorted_diags(mut d: Vec<Value>) -> Vec<Value> {
    d.sort_by_key(|x| {
        (
            x["range"]["start"]["line"].as_u64().unwrap_or(0),
            x["range"]["start"]["character"].as_u64().unwrap_or(0),
            x["message"].to_string(),
        )
    });
    d
}

// ---------------------------------------------------------------- text edits

fn site_row(text: &str, name: &str) -> usize {
    let marker = format!("# stress-site: {name}");
    text.lines()
        .position(|l| l.trim_end().ends_with(&marker))
        .unwrap_or_else(|| panic!("missing site {name}"))
}

fn rows(text: &str) -> Vec<&str> {
    text.split('\n').collect()
}

fn join(lines: Vec<String>) -> String {
    lines.join("\n")
}

fn append_to_line(text: &str, row: usize, s: &str) -> String {
    let mut v: Vec<String> = rows(text).into_iter().map(String::from).collect();
    v[row].push_str(s);
    join(v)
}

fn replace_in_line(text: &str, row: usize, from: &str, to: &str) -> String {
    let mut v: Vec<String> = rows(text).into_iter().map(String::from).collect();
    assert!(
        v[row].contains(from),
        "line {row} lacks {from:?}: {}",
        v[row]
    );
    v[row] = v[row].replacen(from, to, 1);
    join(v)
}

fn insert_after_row(text: &str, row: usize, block: &str) -> String {
    let mut v: Vec<String> = rows(text).into_iter().map(String::from).collect();
    let tail = v.split_off(row + 1);
    v.push(block.trim_end_matches('\n').to_string());
    v.extend(tail);
    join(v)
}

fn hint_label(hints: &[Value], row: usize, name: &str) -> Option<String> {
    hints
        .iter()
        .find(|h| {
            h["position"]["line"].as_u64() == Some(row as u64)
                && h["tooltip"]
                    .as_str()
                    .is_some_and(|t| t.starts_with(&format!("{name}:")))
        })
        .and_then(|h| h["label"].as_str().map(|s| s.to_string()))
}

fn diag_rows(d: &[Value]) -> Vec<usize> {
    d.iter()
        .map(|x| x["range"]["start"]["line"].as_u64().unwrap_or(0) as usize)
        .collect()
}

// ---------------------------------------------------------------- recording

struct Failure {
    scenario: String,
    kind: String,
    detail: String,
}

#[derive(Default)]
struct Recorder {
    samples: BTreeMap<String, Vec<Sample>>,
    scalars: BTreeMap<String, Vec<f64>>,
    failures: Vec<Failure>,
    notes: BTreeMap<String, String>,
    marker_failures: Vec<Value>,
    rss_open_mb: Vec<f64>,
    rss_peak_mb: Vec<f64>,
}

impl Recorder {
    fn sample(&mut self, name: &str, s: Sample) {
        self.samples.entry(name.to_string()).or_default().push(s);
    }
    fn scalar(&mut self, name: &str, v: f64) {
        self.scalars.entry(name.to_string()).or_default().push(v);
    }
    fn fail(&mut self, scenario: &str, kind: &str, detail: String) {
        let dup = self
            .failures
            .iter()
            .any(|f| f.scenario == scenario && f.kind == kind);
        if !dup {
            self.failures.push(Failure {
                scenario: scenario.to_string(),
                kind: kind.to_string(),
                detail,
            });
        }
    }
    fn note(&mut self, k: &str, v: String) {
        self.notes.insert(k.to_string(), v);
    }
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let idx = ((p * s.len() as f64).ceil() as usize).max(1) - 1;
    s[idx]
}

fn stats(v: &[f64]) -> Value {
    json!({"p50": r1(pct(v, 0.5)), "p95": r1(pct(v, 0.95)), "max": r1(pct(v, 1.0))})
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ---------------------------------------------------------------- markers

/// Checks `# expected-shape:` and `# expected-error:` markers in `text` against
/// the server's hints and diagnostics. Returns one JSON record per failure.
fn check_markers(text: &str, diags: &[Value], hints: &[Value]) -> (usize, Vec<Value>) {
    let mut by_line: HashMap<usize, Vec<(String, String)>> = HashMap::new();
    for h in hints {
        let row = h["position"]["line"].as_u64().unwrap_or(0) as usize;
        let name = h["tooltip"]
            .as_str()
            .and_then(|t| t.split(':').next())
            .unwrap_or("")
            .to_string();
        let label = h["label"]
            .as_str()
            .unwrap_or("")
            .trim_start_matches(": ")
            .to_string();
        by_line.entry(row).or_default().push((name, label));
    }
    let mut diag_by_line: HashMap<usize, Vec<String>> = HashMap::new();
    for d in diags {
        let row = d["range"]["start"]["line"].as_u64().unwrap_or(0) as usize;
        diag_by_line
            .entry(row)
            .or_default()
            .push(d["message"].as_str().unwrap_or("").to_string());
    }
    let mut checked = 0;
    let mut fails = Vec::new();
    for (row, line) in text.lines().enumerate() {
        if let Some((_, rest)) = line.split_once("# expected-shape: ") {
            for item in rest.split("; ") {
                let Some((name, shape)) = item.split_once('=') else {
                    continue;
                };
                checked += 1;
                let want = if shape == "[]" { "Scalar" } else { shape };
                let got = by_line
                    .get(&row)
                    .and_then(|v| v.iter().find(|(n, _)| n == name))
                    .map(|(_, l)| l.as_str());
                if got != Some(want) {
                    fails.push(json!({
                        "kind": if got.is_none() { "missing-shape" } else { "wrong-shape" },
                        "line": row + 1, "name": name, "expected": want, "got": got,
                        "source": line.trim(),
                    }));
                }
            }
        }
        let has_err = line.contains("# expected-error");
        let n = diag_by_line.get(&row).map_or(0, |v| v.len());
        if has_err {
            checked += 1;
            if n != 1 {
                fails.push(
                    json!({"kind": if n == 0 { "missing-error" } else { "duplicate-error" },
                    "line": row + 1, "got": diag_by_line.get(&row), "source": line.trim()}),
                );
            }
        } else if n > 0 {
            fails.push(json!({"kind": "unexpected-diagnostic", "line": row + 1,
                "got": diag_by_line.get(&row), "source": line.trim()}));
        }
    }
    (checked, fails)
}

fn group_key(f: &Value) -> String {
    let src = f["source"].as_str().unwrap_or("");
    let rhs = src.split_once(" = ").map_or(src, |x| x.1);
    let callee: String = rhs.chars().take_while(|c| *c != '(' && *c != ' ').collect();
    let callee: String = callee
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect();
    format!("{} {}", f["kind"].as_str().unwrap_or(""), callee)
}

// ---------------------------------------------------------------- oracle

/// Differential oracle: a fresh server that opens `text` must publish the same
/// diagnostics and inlay hints as the long-running server.
fn oracle(rec: &mut Recorder, scenario: &str, root: &Path, d: &Doc, run_hints: &[Value]) {
    let mut f = Server::spawn(root);
    let (fd, _) = open_doc(&mut f, &d.uri, &d.text);
    let (fh, _) = full_hints(&mut f, &fd);
    f.kill();
    let (a, b) = (sorted_diags(d.diags.clone()), sorted_diags(fd.diags));
    let (ha, hb) = (sorted_hints(run_hints.to_vec()), sorted_hints(fh));
    let mut diffs = Vec::new();
    if a != b {
        let first = a.iter().zip(b.iter()).find(|(x, y)| x != y);
        diffs.push(format!(
            "diagnostics differ: running={} fresh={} first_diff={:?}",
            a.len(),
            b.len(),
            first.map(|(x, y)| (x["message"].clone(), y["message"].clone()))
        ));
    }
    if ha != hb {
        let first = ha.iter().zip(hb.iter()).find(|(x, y)| x != y);
        diffs.push(format!(
            "hints differ: running={} fresh={} first_diff={}",
            ha.len(),
            hb.len(),
            first.map_or("(length only)".to_string(), |(x, y)| format!(
                "line {} running {} vs fresh {}",
                x["position"]["line"], x["label"], y["label"]
            ))
        ));
    }
    if !diffs.is_empty() {
        rec.fail(scenario, "oracle-mismatch", diffs.join("; "));
    }
}

// ---------------------------------------------------------------- one run

fn copy_project(dst: &Path) {
    for e in fs::read_dir(repo().join("bench/stress")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "py") {
            fs::copy(&p, dst.join(p.file_name().unwrap())).unwrap();
        }
    }
}

/// (scenario, edited text, expected (row, variable, hint label), row of the new error)
type DepEdit<'a> = (&'a str, String, Vec<(usize, &'a str, &'a str)>, usize);

fn one_run(run: usize, rec: &mut Recorder) {
    let tmp = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(tmp.path()).unwrap();
    copy_project(&root);
    let main_path = root.join("stress_main.py");
    let base = fs::read_to_string(&main_path).unwrap();
    assert!(base.is_ascii(), "generator must emit ASCII");
    let uri = format!("file://{}", main_path.display());
    let first = run == 0;

    // 1. cold start
    let mut srv = Server::spawn(&root);
    rec.sample(
        "cold_initialize",
        Sample {
            total: srv.init_ms,
            ..Default::default()
        },
    );
    let (mut doc, open) = open_doc(&mut srv, &uri, &base);
    rec.sample("cold_open_first_diagnostics", open);
    rec.rss_open_mb.push(srv.rss_mb());
    let base_diags = sorted_diags(doc.diags.clone());
    let (hints, ms) = full_hints(&mut srv, &doc);
    rec.sample(
        "inlay_full_first",
        Sample {
            total: ms,
            ..Default::default()
        },
    );
    let base_hints = sorted_hints(hints.clone());
    for _ in 0..3 {
        let (_, ms) = full_hints(&mut srv, &doc);
        rec.sample(
            "inlay_full_cached",
            Sample {
                total: ms,
                ..Default::default()
            },
        );
    }
    if first {
        let (checked, fails) = check_markers(&base, &doc.diags, &hints);
        rec.scalar("markers_checked_main", checked as f64);
        rec.marker_failures.extend(fails);
        rec.scalar("hint_count", hints.len() as f64);
        rec.scalar("diagnostic_count", doc.diags.len() as f64);
        rec.scalar("lines", line_count(&base) as f64);
    }

    // 1b. markers in the imported helper modules (checked once, in run 0)
    if first {
        for name in ["stress_torch.py", "stress_jax.py", "stress_common.py"] {
            let p = root.join(name);
            let text = fs::read_to_string(&p).unwrap();
            let u = format!("file://{}", p.display());
            let (d, _) = open_doc(&mut srv, &u, &text);
            let (h, _) = full_hints(&mut srv, &d);
            let (checked, fails) = check_markers(&text, &d.diags, &h);
            rec.scalar(&format!("markers_checked_{name}"), checked as f64);
            for mut f in fails {
                f["scenario"] = json!(name);
                rec.marker_failures.push(f);
            }
            srv.notify("textDocument/didClose", json!({"textDocument": {"uri": u}}));
        }
    }

    // 2. hover and goto-definition
    let hover_row = site_row(&base, "hover");
    let hover_line = base.lines().nth(hover_row).unwrap();
    let hover_col = hover_line.find("hover_me").unwrap();
    for _ in 0..10 {
        let (r, ms) = srv.request(
            "textDocument/hover",
            json!({"textDocument": {"uri": uri}, "position": {"line": hover_row, "character": hover_col + 2}}),
        );
        rec.sample(
            "hover",
            Sample {
                total: ms,
                ..Default::default()
            },
        );
        let v = r["contents"]["value"].as_str().unwrap_or("");
        if !v.contains("hover_me: Float[Array, \"sbatch sseq\"]") {
            rec.fail("hover", "wrong-hover", format!("got {r}"));
        }
    }
    let goto_row = site_row(&base, "goto");
    let goto_line = base.lines().nth(goto_row).unwrap();
    let first_col = goto_line.find("\"sbatch").unwrap() + 1;
    let second_col = goto_line.rfind("\"sbatch").unwrap() + 1;
    for _ in 0..10 {
        let (r, ms) = srv.request(
            "textDocument/definition",
            json!({"textDocument": {"uri": uri}, "position": {"line": goto_row, "character": second_col}}),
        );
        rec.sample(
            "goto_definition",
            Sample {
                total: ms,
                ..Default::default()
            },
        );
        let (l, c) = (
            r["range"]["start"]["line"].as_u64(),
            r["range"]["start"]["character"].as_u64(),
        );
        if l != Some(goto_row as u64) || c != Some(first_col as u64) {
            rec.fail(
                "goto_definition",
                "wrong-definition",
                format!("got {r}, want {goto_row}:{first_col}"),
            );
        }
    }

    // 3. debounce floor: change a 1-line document
    let tiny_uri = format!("file://{}/tiny.py", root.display());
    let (mut tiny, _) = open_doc(&mut srv, &tiny_uri, "x = 1\n");
    for k in 0..5 {
        let s = change_doc(&mut srv, &mut tiny, format!("x = {k}\n"));
        rec.sample("floor_tiny_doc_change", s);
    }
    srv.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": tiny_uri}}),
    );

    // 4. single-character edits at start, middle, end
    let last_row = rows(base.trim_end_matches('\n')).len() - 1;
    let char_edits: Vec<(&str, String, Option<usize>)> = vec![
        ("edit_char_start", format!("#{base}"), None),
        (
            "edit_char_middle",
            append_to_line(&base, hover_row, "x"),
            Some(hover_row),
        ),
        ("edit_char_end", append_to_line(&base, last_row, "#"), None),
    ];
    for (name, text, shifted_row) in char_edits {
        let s = change_doc(&mut srv, &mut doc, text);
        rec.sample(name, s);
        let (h, _) = full_hints(&mut srv, &doc);
        if sorted_diags(doc.diags.clone()) != base_diags {
            rec.fail(
                name,
                "diagnostics-changed",
                "neutral edit changed diagnostics".into(),
            );
        }
        if let Some(row) = shifted_row {
            let was = base_hints
                .iter()
                .find(|x| x["position"]["line"] == row)
                .map(|x| x["position"]["character"].as_u64());
            let now = h
                .iter()
                .find(|x| x["position"]["line"] == row)
                .map(|x| x["position"]["character"].as_u64());
            if was.flatten().map(|c| c + 1) != now.flatten() {
                rec.fail(
                    name,
                    "hint-position",
                    format!("hint column was {was:?}, now {now:?}"),
                );
            }
        }
        oracle(rec, name, &root, &doc, &h);
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample(&format!("{name}_revert"), s);
    }

    // 5. paste and delete a 500-line block
    let payload = fs::read_to_string(repo().join("bench/stress/paste_block.txt")).unwrap();
    let paste_row = site_row(&base, "paste-here");
    let pasted = insert_after_row(&base, paste_row, &payload);
    let s = change_doc(&mut srv, &mut doc, pasted.clone());
    rec.sample("paste_500_lines", s);
    let (h, _) = full_hints(&mut srv, &doc);
    if first {
        let (checked, fails) = check_markers(&pasted, &doc.diags, &h);
        rec.scalar("markers_checked_paste", checked as f64);
        for mut f in fails {
            f["scenario"] = json!("paste_500_lines");
            rec.marker_failures.push(f);
        }
    }
    oracle(rec, "paste_500_lines", &root, &doc, &h);
    let s = change_doc(&mut srv, &mut doc, base.clone());
    rec.sample("delete_500_lines", s);
    let (h, _) = full_hints(&mut srv, &doc);
    if sorted_diags(doc.diags.clone()) != base_diags || sorted_hints(h.clone()) != base_hints {
        rec.fail(
            "delete_500_lines",
            "not-restored",
            "diagnostics or hints differ from the cold-open baseline".into(),
        );
    }
    oracle(rec, "delete_500_lines", &root, &doc, &h);

    // 6. add a shape error, then revert
    let err_row = site_row(&base, "error-insert");
    let def_row = base
        .lines()
        .position(|l| l.starts_with("def site_error("))
        .unwrap();
    for (name, line, want_row, word, need_related) in [
        (
            "error_insert_concat",
            "    bad_cat = jnp.concatenate([a, b], axis=0)",
            err_row + 1,
            "concatenate",
            false,
        ),
        (
            "error_insert_matmul",
            "    bad_mm = a @ b",
            def_row,
            "matmul",
            true,
        ),
    ] {
        let s = change_doc(&mut srv, &mut doc, insert_after_row(&base, err_row, line));
        rec.sample(name, s);
        let (h, _) = full_hints(&mut srv, &doc);
        // Inserting a line shifts every later diagnostic down by one row.
        let key = |d: &Value, shift: bool| {
            let row = d["range"]["start"]["line"].as_u64().unwrap_or(0) as usize;
            (
                d["message"].to_string(),
                if shift && row > err_row { row + 1 } else { row },
            )
        };
        let base_keys: Vec<_> = base_diags.iter().map(|d| key(d, true)).collect();
        let new: Vec<&Value> = doc
            .diags
            .iter()
            .filter(|d| !base_keys.contains(&key(d, false)))
            .collect();
        let ok = doc.diags.len() == base_diags.len() + 1
            && new.len() == 1
            && diag_rows(&[new[0].clone()]) == vec![want_row]
            && new[0]["message"].as_str().unwrap_or("").contains(word)
            && (!need_related || new[0]["relatedInformation"].is_array());
        if !ok {
            rec.fail(
                name,
                "wrong-diagnostic",
                format!(
                    "want one new `{word}` diagnostic on row {want_row}; got {} total, new: {:?}",
                    doc.diags.len(),
                    new
                ),
            );
        }
        oracle(rec, name, &root, &doc, &h);
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample(&format!("{name}_revert"), s);
        if sorted_diags(doc.diags.clone()) != base_diags {
            rec.fail(
                name,
                "diagnostic-not-removed",
                "diagnostics differ from baseline after revert".into(),
            );
        }
    }

    // 7. keystroke burst: 50 changes faster than the debounce
    {
        let from = srv.pubs.len();
        let v0 = doc.version;
        let texts: Vec<String> = (1..=50)
            .map(|k| append_to_line(&base, hover_row, &"x".repeat(k)))
            .collect();
        let bodies: Vec<Vec<u8>> = texts
            .iter()
            .enumerate()
            .map(|(k, t)| {
                serde_json::to_vec(
                    &json!({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
                "textDocument": {"uri": uri, "version": v0 + 1 + k as i64},
                "contentChanges": [{"text": t}]}}),
                )
                .unwrap()
            })
            .collect();
        let t_start = Instant::now();
        for b in &bodies {
            write_body(&srv.stdin, b);
        }
        let t_sent = Instant::now();
        let last = v0 + 50;
        let i = srv.wait_pub(&uri, last, from);
        let at = srv.pubs[i].at;
        srv.settle(600);
        rec.sample(
            "burst_50_changes",
            Sample {
                total: at.saturating_duration_since(t_sent).as_secs_f64() * 1000.0,
                compute: srv.compute_ms(t_start, at),
                debounce: DEBOUNCE_MS,
            },
        );
        rec.scalar(
            "burst_send_ms",
            t_sent.duration_since(t_start).as_secs_f64() * 1000.0,
        );
        rec.scalar(
            "burst_analyses",
            srv.analyses_between(t_start, Instant::now()) as f64,
        );
        let versions: Vec<i64> = srv.pubs[from..]
            .iter()
            .filter(|p| p.uri == uri)
            .filter_map(|p| p.version)
            .collect();
        rec.scalar("burst_published_versions", versions.len() as f64);
        if versions.windows(2).any(|w| w[1] < w[0]) || versions.last() != Some(&last) {
            rec.fail(
                "burst_50_changes",
                "stale-publication",
                format!("published versions {versions:?}, last must be {last}"),
            );
        }
        if versions.len() > 1 {
            rec.fail("burst_50_changes", "intermediate-publications", format!(
                "{} publications and {} full analyses for one burst; only version {last} may count (published {versions:?})",
                versions.len(), srv.analyses_between(t_start, Instant::now())));
        }
        doc.version = last;
        doc.text = texts[49].clone();
        doc.diags = srv.pubs[i].diags.clone();
        let (h, _) = full_hints(&mut srv, &doc);
        if sorted_diags(doc.diags.clone()) != base_diags {
            rec.fail(
                "burst_50_changes",
                "diagnostics-changed",
                "neutral typing changed diagnostics".into(),
            );
        }
        oracle(rec, "burst_50_changes", &root, &doc, &h);
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample("burst_50_changes_revert", s);
    }

    // 7b. realistic typing: one key every 100 ms (slower than the debounce).
    // Editors ask for inlay hints after every change; the control run does not.
    for (name, with_inlay) in [
        ("typing_plain", false),
        ("typing_with_inlay_requests", true),
    ] {
        let from = srv.pubs.len();
        let v0 = doc.version;
        let keys = 12;
        let t_start = Instant::now();
        let mut pending = Vec::new();
        let mut t_last = t_start;
        let end_line = line_count(&base) + 1;
        for k in 1..=keys {
            let text = append_to_line(&base, hover_row, &"x".repeat(k));
            t_last = srv.send_timed(
                &json!({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
                "textDocument": {"uri": uri, "version": v0 + k as i64},
                "contentChanges": [{"text": text}]}}),
            );
            if with_inlay {
                pending.push(srv.request_send(
                    "textDocument/inlayHint",
                    json!({"textDocument": {"uri": uri},
                           "range": {"start": {"line": 0, "character": 0},
                                     "end": {"line": end_line, "character": 0}}}),
                ));
            }
            srv.settle(100);
        }
        let last = v0 + keys as i64;
        let i = srv.wait_pub(&uri, last, from);
        let at = srv.pubs[i].at;
        for (id, t0) in pending {
            srv.request_wait("textDocument/inlayHint", id, t0);
        }
        srv.settle(700);
        let analyses = srv.analyses_between(t_start, Instant::now()) as f64;
        let published = srv.pubs[from..].iter().filter(|p| p.uri == uri).count() as f64;
        rec.scalar(&format!("{name}_analyses"), analyses);
        rec.scalar(&format!("{name}_publications"), published);
        rec.sample(
            name,
            Sample {
                total: at.saturating_duration_since(t_last).as_secs_f64() * 1000.0,
                compute: srv.compute_ms(t_start, at),
                debounce: DEBOUNCE_MS,
            },
        );
        if analyses > 1.0 {
            rec.fail(name, "debounce-bypassed", format!(
                "{keys} keystrokes 100 ms apart caused {analyses} full analyses and {published} publications; one is enough"));
        }
        doc.version = last;
        doc.text = append_to_line(&base, hover_row, &"x".repeat(keys));
        doc.diags = srv.pubs[i].diags.clone();
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample(&format!("{name}_revert"), s);
    }

    // 8. edits that change dependent shapes
    let probe_h = site_row(&base, "probe-h");
    let probe_out = site_row(&base, "probe-out");
    let child_h = site_row(&base, "child-h");
    let child_o = site_row(&base, "child-o");
    let helper_call = site_row(&base, "helper-call");
    let helper_dep = site_row(&base, "helper-dependent");
    let want = |rec: &mut Recorder, sc: &str, h: &[Value], row: usize, name: &str, label: &str| {
        let got = hint_label(h, row, name);
        if got.as_deref() != Some(label) {
            rec.fail(
                sc,
                "wrong-dependent-shape",
                format!("row {row} `{name}`: want {label}, got {got:?}"),
            );
        }
    };
    want(rec, "baseline", &base_hints, probe_h, "h", ": [sp, 64]");
    want(rec, "baseline", &base_hints, child_h, "h", ": [sb, 32]");
    want(
        rec,
        "baseline",
        &base_hints,
        helper_call,
        "r",
        ": [sh1, sh2]",
    );
    want(
        rec,
        "baseline",
        &base_hints,
        helper_dep,
        "s",
        ": [sh1, sh2]",
    );
    let dep_edits: Vec<DepEdit> = vec![
        (
            "edit_init_layer",
            replace_in_line(
                &base,
                site_row(&base, "init-layer"),
                "nn.Linear(32, 64)",
                "nn.Linear(32, 48)",
            ),
            vec![(probe_h, "h", ": [sp, 48]")],
            probe_out,
        ),
        (
            "edit_base_class_init_layer",
            replace_in_line(
                &base,
                site_row(&base, "base-init-layer"),
                "nn.Linear(16, 32)",
                "nn.Linear(16, 24)",
            ),
            vec![(child_h, "h", ": [sb, 24]")],
            child_o,
        ),
        (
            "edit_unannotated_helper_body",
            replace_in_line(
                &base,
                site_row(&base, "helper-return"),
                "u.sum(axis=-1)",
                "u.sum(axis=0)",
            ),
            vec![
                (helper_call, "r", ": [sh2, sh3]"),
                (helper_dep, "s", ": [sh2, sh3]"),
            ],
            usize::MAX,
        ),
    ];
    for (name, text, wants, err_line) in dep_edits {
        let s = change_doc(&mut srv, &mut doc, text);
        rec.sample(name, s);
        let (h, _) = full_hints(&mut srv, &doc);
        for (row, var, label) in &wants {
            want(rec, name, &h, *row, var, label);
        }
        if err_line != usize::MAX {
            let n = diag_rows(&doc.diags)
                .iter()
                .filter(|r| **r == err_line)
                .count();
            if n != 1 || doc.diags.len() != base_diags.len() + 1 {
                rec.fail(
                    name,
                    "missing-dependent-error",
                    format!(
                        "want exactly one new diagnostic on row {err_line}; rows {:?}",
                        diag_rows(&doc.diags)
                    ),
                );
            }
        }
        oracle(rec, name, &root, &doc, &h);
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample(&format!("{name}_revert"), s);
        let (h, _) = full_hints(&mut srv, &doc);
        if sorted_diags(doc.diags.clone()) != base_diags || sorted_hints(h) != base_hints {
            rec.fail(
                name,
                "not-restored",
                "diagnostics or hints differ from baseline after revert".into(),
            );
        }
    }

    // 9. edit to an imported helper file
    {
        let helper_path = root.join("stress_common.py");
        let helper_uri = format!("file://{}", helper_path.display());
        let orig = fs::read_to_string(&helper_path).unwrap();
        let old_sig = "def project_0(x: Float[Array, \"b d\"], w: Float[Array, \"d k\"]) -> Float[Array, \"b k\"]:";
        assert!(orig.contains(old_sig));
        let modified = orig.replace(old_sig, &old_sig.replace("\"b k\"", "\"k b\""));
        let cross_row = site_row(&base, "cross-call");
        let y_of = |h: &[Value]| hint_label(h, cross_row, "y");
        let y_base = y_of(&base_hints);
        let (mut hd, s) = open_doc(&mut srv, &helper_uri, &orig);
        rec.sample("helper_didOpen", s);
        let s = change_doc(&mut srv, &mut hd, modified.clone());
        rec.sample("helper_didChange_unsaved", s);

        let s = change_doc(&mut srv, &mut doc, append_to_line(&base, hover_row, "x"));
        rec.sample("main_after_helper_unsaved_edit", s);
        let (h, _) = full_hints(&mut srv, &doc);
        let y_unsaved = y_of(&h);
        oracle(rec, "helper_unsaved_edit", &root, &doc, &h);

        fs::write(&helper_path, &modified).unwrap();
        srv.notify(
            "textDocument/didSave",
            json!({"textDocument": {"uri": helper_uri}}),
        );
        srv.notify(
            "workspace/didChangeWatchedFiles",
            json!({"changes": [{"uri": helper_uri, "type": 2}]}),
        );
        let s = change_doc(&mut srv, &mut doc, append_to_line(&base, hover_row, "xx"));
        rec.sample("main_after_helper_saved_to_disk", s);
        let (h, _) = full_hints(&mut srv, &doc);
        let y_saved = y_of(&h);
        let mut fresh = Server::spawn(&root);
        let (fd, _) = open_doc(&mut fresh, &uri, &doc.text);
        let (fh, _) = full_hints(&mut fresh, &fd);
        let y_fresh = y_of(&fh);
        fresh.kill();
        if first {
            rec.note("helper_edit.baseline_y", format!("{y_base:?}"));
            rec.note("helper_edit.after_unsaved_buffer_change", format!("{y_unsaved:?} (server reads imports from disk, so the unsaved buffer has no effect)"));
            rec.note(
                "helper_edit.after_save_running_server",
                format!("{y_saved:?}"),
            );
            rec.note(
                "helper_edit.after_save_fresh_server",
                format!("{y_fresh:?}"),
            );
        }
        oracle(rec, "helper_saved_to_disk", &root, &doc, &h);

        srv.notify(
            "textDocument/didClose",
            json!({"textDocument": {"uri": helper_uri}}),
        );
        fs::write(&helper_path, &orig).unwrap();
        let s = change_doc(&mut srv, &mut doc, base.clone());
        rec.sample("main_after_helper_restored", s);
    }

    rec.rss_peak_mb.push(srv.kill());
}

// ---------------------------------------------------------------- report

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn lsp_stress_benchmark() {
    let runs = env_usize("NDIM_STRESS_RUNS", 5);
    let mut rec = Recorder::default();
    for run in 0..runs {
        let t = Instant::now();
        one_run(run, &mut rec);
        eprintln!(
            "run {}/{} done in {:.1}s",
            run + 1,
            runs,
            t.elapsed().as_secs_f64()
        );
    }

    // A latency sample with no server `analysis:` log did no analysis, so it
    // measured nothing (for example a skipped or failed analysis).
    let analysed = [
        "edit_",
        "paste_",
        "delete_",
        "error_",
        "burst_",
        "typing_",
        "main_after_",
    ];
    let silent: Vec<String> = rec
        .samples
        .iter()
        .filter(|(n, v)| {
            analysed.iter().any(|p| n.starts_with(p)) && v.iter().all(|s| s.compute == 0.0)
        })
        .map(|(n, _)| n.clone())
        .collect();
    for n in silent {
        rec.fail(
            &n,
            "no-analysis-log",
            "no `analysis:` log line in the measured window".into(),
        );
    }

    let mut scenarios = serde_json::Map::new();
    println!("\n=== lsp stress: {runs} runs, debounce {DEBOUNCE_MS} ms (ms) ===");
    println!(
        "{:<38} {:>3} {:>24} {:>8} {:>8}",
        "scenario", "n", "total p50/p95/max", "compute", "other"
    );
    for (name, v) in &rec.samples {
        let tot: Vec<f64> = v.iter().map(|s| s.total).collect();
        let cmp: Vec<f64> = v.iter().map(|s| s.compute).collect();
        let oth: Vec<f64> = v.iter().map(|s| s.total - s.debounce - s.compute).collect();
        println!(
            "{:<38} {:>3} {:>8.1}/{:>6.1}/{:>6.1} {:>8.1} {:>8.1}",
            name,
            v.len(),
            pct(&tot, 0.5),
            pct(&tot, 0.95),
            pct(&tot, 1.0),
            pct(&cmp, 0.5),
            pct(&oth, 0.5)
        );
        scenarios.insert(
            name.clone(),
            json!({
                "n": v.len(), "total_ms": stats(&tot), "server_compute_ms": stats(&cmp),
                "debounce_ms": v[0].debounce, "other_ms": stats(&oth),
            }),
        );
    }
    println!(
        "peak RSS MB (sampled): open {:?} peak {:?}",
        rec.rss_open_mb.iter().map(|x| r1(*x)).collect::<Vec<_>>(),
        rec.rss_peak_mb.iter().map(|x| r1(*x)).collect::<Vec<_>>()
    );
    for (k, v) in &rec.scalars {
        println!("  {k}: {:?}", v.iter().map(|x| r1(*x)).collect::<Vec<_>>());
    }
    for (k, v) in &rec.notes {
        println!("  note {k}: {v}");
    }

    let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for f in &rec.marker_failures {
        groups.entry(group_key(f)).or_default().push(f);
    }
    println!("\n=== correctness ===");
    println!(
        "marker failures: {} in {} groups",
        rec.marker_failures.len(),
        groups.len()
    );
    for (k, v) in &groups {
        println!(
            "  {:>4}x {k}   e.g. line {} `{}`",
            v.len(),
            v[0]["line"],
            v[0]["source"].as_str().unwrap_or("")
        );
    }
    for f in &rec.failures {
        println!("  FAIL {} [{}]: {}", f.scenario, f.kind, f.detail);
    }

    if let Ok(out) = std::env::var("NDIM_BENCH_OUT") {
        let body = json!({
            "meta": {
                "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                "runs": runs, "debounce_ms": DEBOUNCE_MS,
                "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
                "os": std::env::consts::OS,
                "rss_note": "sampled with ps every 25 ms; lower bound on true peak",
            },
            "scenarios": scenarios,
            "scalars": rec.scalars.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "peak_rss_mb": {"after_open": stats(&rec.rss_open_mb), "peak": stats(&rec.rss_peak_mb)},
            "notes": rec.notes,
            "marker_failure_groups": groups.iter().map(|(k, v)| json!({"group": k, "count": v.len(), "example": v[0]})).collect::<Vec<_>>(),
            "marker_failures": rec.marker_failures,
            "failures": rec.failures.iter().map(|f| json!({"scenario": f.scenario, "kind": f.kind, "detail": f.detail})).collect::<Vec<_>>(),
        });
        fs::write(&out, serde_json::to_string_pretty(&body).unwrap() + "\n").unwrap();
        println!("wrote {out}");
    }
    assert!(
        rec.marker_failures.is_empty() && rec.failures.is_empty(),
        "{} marker failures and {} scenario failures (see report above)",
        rec.marker_failures.len(),
        rec.failures.len()
    );
}

// ---------------------------------------------------------------- golden

fn normalize(v: Value, root: &Path) -> Value {
    let s = v
        .to_string()
        .replace(&format!("file://{}/", root.display()), "<root>/");
    serde_json::from_str(&s).unwrap()
}

fn golden_group(dir: &Path, files: &[PathBuf], out: &mut serde_json::Map<String, Value>) {
    let root = fs::canonicalize(dir).unwrap();
    let mut srv = Server::spawn(&root);
    for path in files {
        let text = fs::read_to_string(path).unwrap();
        let uri = format!("file://{}", fs::canonicalize(path).unwrap().display());
        let (doc, _) = open_doc(&mut srv, &uri, &text);
        let (hints, _) = full_hints(&mut srv, &doc);
        srv.notify(
            "textDocument/didClose",
            json!({"textDocument": {"uri": uri}}),
        );

        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(&text, None).unwrap();
        let cache = ndim_lsp::new_resolution_cache();
        let a = ndim_lsp::analyze_layer_shapes(
            tree.root_node(),
            &text,
            std::slice::from_ref(&root),
            |p: &PathBuf| fs::read_to_string(p).ok(),
            8,
            Some(&cache),
        )
        .unwrap();
        let mut assigns: Vec<Value> = a
            .assignment_shapes
            .iter()
            .map(|r| json!([r.line, r.name, r.shape]))
            .collect();
        assigns.sort_by_key(|x| x.to_string());
        let rng = |r: &tree_sitter::Range| {
            json!([
                r.start_point.row,
                r.start_point.column,
                r.end_point.row,
                r.end_point.column
            ])
        };
        let mut errors: Vec<Value> = a
            .errors
            .iter()
            .map(|e| {
                json!({
                    "variable": e.variable, "message": e.message, "kind": format!("{:?}", e.kind),
                    "range": rng(&e.range),
                    "related": e.related.as_ref().map(|(r, m)| json!([rng(r), m])),
                    "has_fix": e.fix.is_some(),
                })
            })
            .collect();
        errors.sort_by_key(|x| x.to_string());
        let key = path
            .strip_prefix(repo())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        out.insert(
            key,
            normalize(
                json!({
                    "assignment_shapes": assigns,
                    "errors": errors,
                    "diagnostics": sorted_diags(doc.diags),
                    "inlay_hints": sorted_hints(hints),
                }),
                &root,
            ),
        );
    }
    srv.kill();
}

fn py_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "py"))
        .collect();
    v.sort();
    v
}

fn build_golden() -> String {
    let mut out = serde_json::Map::new();
    golden_group(
        &repo().join("corpus"),
        &py_files(&repo().join("corpus")),
        &mut out,
    );
    golden_group(
        &repo().join("bench"),
        &[repo().join("bench/benchmark_large.py")],
        &mut out,
    );
    golden_group(
        &repo().join("bench/stress"),
        &py_files(&repo().join("bench/stress")),
        &mut out,
    );
    serde_json::to_string_pretty(&Value::Object(out)).unwrap() + "\n"
}

#[test]
#[ignore]
fn golden_dump() {
    let path = std::env::var("NDIM_GOLDEN_OUT").expect("set NDIM_GOLDEN_OUT");
    let a = build_golden();
    let b = build_golden();
    assert!(
        a == b,
        "two golden runs differ: the dump is not deterministic"
    );
    fs::write(&path, a.as_bytes()).unwrap();
    println!("wrote {path} ({} bytes)", a.len());
}
