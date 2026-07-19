//! wasm-bindgen bindings for the fxtranslate engine.
//!
//! Exposes a [`Translator`] to JavaScript: constructed from in-memory model,
//! vocabulary, and (optional) shortlist buffers supplied by the host, it wraps a
//! lean/scalar [`fxtranslate::Engine`] built via [`Engine::from_bytes`]. The host
//! (Node or a browser) reads the `.bin`/`.spm`/`lex` files and hands their bytes
//! in as `Uint8Array`s; nothing here touches the filesystem, the network, or a
//! C++ toolchain — the wasm build is a validation + npm-distribution artifact, not
//! a production target (the native Rust engine is what ships).
//!
//! [`Engine::from_bytes`]: fxtranslate::Engine::from_bytes

use fxtranslate::cache::verify_and_decompress as core_verify_and_decompress;
use fxtranslate::engine::{Engine, Phase};
use fxtranslate::remote::{pairs as core_pairs, parse_records as core_parse_records, Record};
use fxtranslate::route::{catalog as core_catalog, resolve_route as core_resolve_route, Route};
use fxtranslate::segment::{IcuSegmenter, Segmenter};
use wasm_bindgen::prelude::*;

/// A translation engine callable from JavaScript.
///
/// Owns its [`Engine`] (and, transitively, owned copies of the model bytes — the
/// engine borrows nothing from the `Uint8Array`s passed to the constructor), so
/// it stays valid across `translate` calls without the host holding the buffers.
#[wasm_bindgen]
pub struct Translator {
    engine: Engine,
}

#[wasm_bindgen]
impl Translator {
    /// Build a translator from model + vocabulary bytes, with an optional lexical
    /// shortlist.
    ///
    /// Pass byte-identical `src_vocab` and `trg_vocab` for shared-vocab pairs
    /// (most pairs, including en→fr) and distinct buffers for split-vocab (CJK).
    /// `shortlist` may be `null`/`undefined`; when present it restricts the output
    /// vocabulary per sentence (required for exact reference parity).
    #[wasm_bindgen(constructor)]
    pub fn new(
        model: &[u8],
        src_vocab: &[u8],
        trg_vocab: &[u8],
        shortlist: Option<Box<[u8]>>,
    ) -> Result<Translator, JsError> {
        // Route panics to `console.error` with a legible message instead of an
        // opaque wasm trap. Idempotent, so calling it per-constructor is fine.
        console_error_panic_hook::set_once();

        let engine =
            Engine::from_bytes(model, src_vocab, trg_vocab).map_err(|e| JsError::new(&e))?;
        let engine = match shortlist {
            Some(bytes) => engine.with_shortlist_bytes(&bytes),
            None => engine,
        };
        Ok(Translator { engine })
    }

    /// Translate a single sentence-unit `text` and return the target text.
    pub fn translate(&self, text: &str) -> String {
        self.engine.translate(text)
    }

    /// Translate `text` of arbitrary length: split into sentences with ICU4X
    /// (UAX #29, the same engine the native CLI uses) and translate each within
    /// the model's context window, rejoined with the original whitespace.
    pub fn translate_long(&self, text: &str) -> String {
        self.engine.translate_long(text)
    }

    /// Translate one block (its sentences, one per line in `block`) and report
    /// per-phase boundaries to the host so it can time each phase itself.
    ///
    /// wasm has no usable `std::time::Instant` (it panics on
    /// `wasm32-unknown-unknown`), so the native `--timing` spans don't work here.
    /// Instead the host passes `on_phase`, a JS function the engine calls at the
    /// encode/decode/first-token boundaries; the host records `performance.now()`
    /// on each call and derives TTFT + decode tok/s the same way `perf.py` does
    /// from the native `Instant` spans. `on_phase` receives the phase name as a
    /// string: `"encode_start"`, `"decode_start"`, `"first_token"`, `"decode_end"`.
    ///
    /// Returns the block translation (sentences rejoined with `\n`) followed by a
    /// trailing line of the shape `sentences\tsrc_tokens\ttokens` so the host has
    /// the counts (words are host-counted from the source) without a second call.
    #[wasm_bindgen(js_name = translateBlockPhased)]
    pub fn translate_block_phased(&self, block: &str, on_phase: &js_sys::Function) -> String {
        let lines: Vec<&str> = block.split('\n').filter(|l| !l.trim().is_empty()).collect();
        let this = JsValue::NULL;
        let (outs, counts) = self.engine.translate_batch_phased(&lines, |phase| {
            let name = match phase {
                Phase::EncodeStart => "encode_start",
                Phase::DecodeStart => "decode_start",
                Phase::FirstToken => "first_token",
                Phase::DecodeEnd => "decode_end",
            };
            // Fire the host clock. Ignore the (unit) return; a throwing callback
            // would surface as a wasm trap, which is fine for a perf harness.
            let _ = on_phase.call1(&this, &JsValue::from_str(name));
        });
        let mut joined = outs.join("\n");
        joined.push('\n');
        joined.push_str(&format!(
            "{}\t{}\t{}",
            counts.sentences, counts.src_tokens, counts.tokens
        ));
        joined
    }

    /// Current wasm linear-memory size in bytes (pages x 64 KiB). Polled by the
    /// perf harness after each block to report the linear-memory high-water mark —
    /// mostly the ~150 MB owned model buffer plus activations. This is a *different*
    /// metric from native settled/peak RSS (no shared file-backed pages, no mmap),
    /// so the writeup compares them with that caveat, not as identical.
    #[wasm_bindgen(js_name = linearMemoryBytes)]
    pub fn linear_memory_bytes(&self) -> f64 {
        #[cfg(target_arch = "wasm32")]
        {
            // memory_size returns the page count of linear memory 0; a wasm page
            // is 64 KiB. f64 so JS gets it exactly past 2^32 bytes.
            (core::arch::wasm32::memory_size(0) as f64) * 65536.0
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            0.0
        }
    }

    /// The active int8 GEMM backend, so a silent scalar build can't be mistaken
    /// for a SIMD one. Reports `"wasm-simd128"` when the crate was built for
    /// `wasm32` with `-C target-feature=+simd128` (the pure-Rust SIMD128 kernel is
    /// then live), and `"scalar"` for a plain scalar wasm build.
    pub fn backend(&self) -> String {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            fxtranslate::gemm::backend().to_string()
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            "scalar".to_string()
        }
    }
}

/// The pure model-discovery surface, exposed to JavaScript as free functions.
///
/// These mirror the core `remote`/`route`/`segment`/`cache` functions the JS shell
/// needs for the read-only paths (`list`, `models`, download verify). Structured
/// results cross the wasm boundary as JSON strings the host parses — this keeps the
/// wasm build free of serde/serde-wasm-bindgen and matches the crate's minimal-dep
/// philosophy. The JS orchestration does its own async HTTP + filesystem and only
/// ever calls these synchronous, pure functions.
pub mod discovery {
    use super::*;

    /// Append `s` to `out` as a JSON string literal (quotes + minimal escaping).
    /// Covers what Remote Settings language tags / names / hashes actually contain
    /// (`"`, `\`, control chars); no serde needed for these shallow shapes.
    fn push_json_str(out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
    }

    /// A JSON array of string literals.
    fn json_str_array(items: &[String]) -> String {
        let mut out = String::from("[");
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            push_json_str(&mut out, item);
        }
        out.push(']');
        out
    }

    /// One record as a JSON object with the fields [`Record`] carries. `null` for
    /// absent optionals (`architecture`, `decompressedHash`).
    fn record_json(out: &mut String, r: &Record) {
        out.push_str("{\"name\":");
        push_json_str(out, &r.name);
        out.push_str(",\"fileType\":");
        push_json_str(out, &r.file_type);
        out.push_str(",\"sourceLanguage\":");
        push_json_str(out, &r.src);
        out.push_str(",\"targetLanguage\":");
        push_json_str(out, &r.trg);
        out.push_str(",\"version\":");
        push_json_str(out, &r.version);
        out.push_str(",\"architecture\":");
        match &r.architecture {
            Some(a) => push_json_str(out, a),
            None => out.push_str("null"),
        }
        out.push_str(",\"decompressedHash\":");
        match &r.decompressed_hash {
            Some(h) => push_json_str(out, h),
            None => out.push_str("null"),
        }
        out.push_str(",\"location\":");
        push_json_str(out, &r.location);
        out.push('}');
    }

    /// Parse a Remote Settings `records` response body into the model-file records
    /// fxtranslate understands, returned as a JSON array.
    ///
    /// The host fetches the collection JSON itself (async `fetch`) and passes the
    /// body in; this normalizes it to the fields routing needs (name, fileType,
    /// languages, version, architecture, decompressedHash, location), skipping
    /// records that lack the required fields — same rule as the native parser.
    /// Returns the JSON array on success, or throws with the parse error.
    #[wasm_bindgen(js_name = parseRecords)]
    pub fn parse_records(body: &str) -> Result<String, JsError> {
        let records = core_parse_records(body).map_err(|e| JsError::new(&e))?;
        let mut out = String::from("[");
        for (i, r) in records.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            record_json(&mut out, r);
        }
        out.push(']');
        Ok(out)
    }

    /// Resolve `src`→`trg` against a Remote Settings `records` body to the route
    /// that realizes it: a direct model, or a two-leg pivot through a hub.
    ///
    /// Takes the raw collection JSON (the same body [`parseRecords`] accepts) so the
    /// host doesn't have to round-trip a records representation back across the
    /// boundary. Returns one of
    /// `{"kind":"direct","src":..,"trg":..}` or
    /// `{"kind":"pivot","src":..,"pivot":..,"trg":..}`,
    /// or throws when neither a direct model nor a pivot exists.
    #[wasm_bindgen(js_name = resolveRoute)]
    pub fn resolve_route(records_json: &str, src: &str, trg: &str) -> Result<String, JsError> {
        let records = core_parse_records(records_json).map_err(|e| JsError::new(&e))?;
        let route = core_resolve_route(&records, src, trg).map_err(|e| JsError::new(&e))?;
        let mut out = String::new();
        match route {
            Route::Direct { src, trg } => {
                out.push_str("{\"kind\":\"direct\",\"src\":");
                push_json_str(&mut out, &src);
                out.push_str(",\"trg\":");
                push_json_str(&mut out, &trg);
                out.push('}');
            }
            Route::Pivot { src, pivot, trg } => {
                out.push_str("{\"kind\":\"pivot\",\"src\":");
                push_json_str(&mut out, &src);
                out.push_str(",\"pivot\":");
                push_json_str(&mut out, &pivot);
                out.push_str(",\"trg\":");
                push_json_str(&mut out, &trg);
                out.push('}');
            }
        }
        Ok(out)
    }

    /// Classify every language reachable through `hub` (e.g. `"en"`) by the
    /// directions it supports, from a Remote Settings `records` body — the data the
    /// `list` command renders.
    ///
    /// Takes the raw collection JSON. Returns
    /// `{"bidirectional":[..],"sourceOnly":[..],"targetOnly":[..]}`, each a sorted
    /// language-tag array. Throws only on a JSON parse error.
    #[wasm_bindgen(js_name = catalog)]
    pub fn catalog(records_json: &str, hub: &str) -> Result<String, JsError> {
        let records = core_parse_records(records_json).map_err(|e| JsError::new(&e))?;
        let cat = core_catalog(&records, hub);
        let mut out = String::from("{\"bidirectional\":");
        out.push_str(&json_str_array(&cat.bidirectional));
        out.push_str(",\"sourceOnly\":");
        out.push_str(&json_str_array(&cat.source_only));
        out.push_str(",\"targetOnly\":");
        out.push_str(&json_str_array(&cat.target_only));
        out.push('}');
        Ok(out)
    }

    /// The unique, version-gated `src → trg` model pairs from a Remote Settings
    /// `records` body — the raw one-way models the `list --all` view enumerates.
    ///
    /// Takes the raw collection JSON. Returns a JSON array of `[src, trg]` pairs,
    /// already sorted and deduplicated and filtered to the supported model major —
    /// so the JS shell renders exactly what `translate` could load without
    /// reimplementing the version gate. The shell does its own prefix filtering on
    /// this list (a shallow, safe-to-duplicate concern). Throws on a JSON parse error.
    #[wasm_bindgen(js_name = modelPairs)]
    pub fn model_pairs(records_json: &str) -> Result<String, JsError> {
        let records = core_parse_records(records_json).map_err(|e| JsError::new(&e))?;
        let ps = core_pairs(&records);
        let mut out = String::from("[");
        for (i, (s, t)) in ps.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('[');
            push_json_str(&mut out, s);
            out.push(',');
            push_json_str(&mut out, t);
            out.push(']');
        }
        out.push(']');
        Ok(out)
    }

    /// Split `text` into sentence units with ICU4X (UAX #29) and return their
    /// trimmed content as a JSON string array.
    ///
    /// This is the same [`IcuSegmenter`] the wasm `Translator.translate_long`
    /// drives, exposed so the JS shell can segment before batching and get the
    /// exact boundaries the translate path will use. Whitespace between sentences
    /// is not preserved here (the CLI reassembles from the source with
    /// `reassemble`); this yields the sentence contents in order.
    #[wasm_bindgen(js_name = segmentSentences)]
    pub fn segment_sentences(text: &str) -> String {
        let spans = IcuSegmenter::new().sentences(text);
        let sentences: Vec<String> = spans.iter().map(|s| s.of(text).to_string()).collect();
        json_str_array(&sentences)
    }

    /// Decompress a zstd model attachment and verify the decompressed bytes against
    /// an expected hex SHA-256 (a record's `decompressedHash`), returning the
    /// decompressed bytes.
    ///
    /// The JS shell downloads the `.zst` attachment itself and calls this to decode
    /// + verify before writing it to its cache — so it never reimplements zstd or
    /// SHA-256. Pass `expected_sha256_hex` = `null`/`undefined` to decompress
    /// without verifying (records that carry no hash). Throws on a decode failure or
    /// a hash mismatch.
    #[wasm_bindgen(js_name = verifyAndDecompress)]
    pub fn verify_and_decompress(
        compressed: &[u8],
        expected_sha256_hex: Option<String>,
    ) -> Result<Vec<u8>, JsError> {
        core_verify_and_decompress(compressed, expected_sha256_hex.as_deref())
            .map_err(|e| JsError::new(&e))
    }
}
