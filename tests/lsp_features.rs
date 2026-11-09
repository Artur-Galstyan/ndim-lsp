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
        assert_eq!(start_of(&diags[0]["range"]), pos(&text, "k\"], b"), "{text:?}");

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
