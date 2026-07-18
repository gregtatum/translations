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

use fxtranslate::engine::Engine;
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

    /// Translate `text` of arbitrary length: split into sentences (the built-in
    /// segmenter — the wasm build is icu-free) and translate each within the
    /// model's context window, rejoined with the original whitespace.
    pub fn translate_long(&self, text: &str) -> String {
        self.engine.translate_long(text)
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
