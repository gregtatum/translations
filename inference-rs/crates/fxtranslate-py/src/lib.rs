//! PyO3 bindings for the fxtranslate engine.
//!
//! Exposes a [`Translator`] to Python two ways: a byte-for-byte mirror of the wasm
//! constructor (host supplies model + vocab + optional shortlist buffers, wrapping
//! [`Engine::from_bytes`]), and the native-only [`Translator::load`] classmethod —
//! the batteries-included discover → download → cache → translate path over
//! [`loader::load_translation`], which the wasm build cannot offer (it needs
//! `net`/`mmap`). A [`discovery`] submodule mirrors the wasm free functions but
//! returns native Python objects (dicts/lists) rather than hand-rolled JSON strings.
//!
//! The native build turns on the gemmology SIMD kernel, so [`Translator::backend`]
//! reports a real NEON/i8mm/AVX2 backend rather than the wasm `"scalar"`/`simd128`.
//!
//! Two wasm methods are intentionally absent (see the docs on [`Translator`]):
//! `linear_memory_bytes` (measures wasm linear-memory pages — meaningless natively)
//! and `translate_block_phased` (a callback shim that exists only because wasm has
//! no usable `std::time::Instant`; native code has real timing).
//!
//! [`Engine::from_bytes`]: fxtranslate::engine::Engine::from_bytes
//! [`loader::load_translation`]: fxtranslate::loader::load_translation

use fxtranslate::cache::{verify_and_decompress as core_verify_and_decompress, Cache};
use fxtranslate::engine::{Engine, Translation};
use fxtranslate::fetch::NetworkFetch;
use fxtranslate::loader::load_translation;
use fxtranslate::remote::{pairs as core_pairs, parse_records as core_parse_records, Record};
use fxtranslate::route::{catalog as core_catalog, resolve_route as core_resolve_route, Route};
use fxtranslate::segment::{IcuSegmenter, Segmenter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule};

/// Either constructor holds a different core type — the BYO path holds an
/// [`Engine`], the `load` path holds a pivot-aware [`Translation`] — but both expose
/// the same `translate`/`translate_long`/`backend` surface. This enum forwards to
/// whichever is inside so the Python surface is identical across both.
enum Inner {
    /// A single engine, from `Translator(model, src_vocab, trg_vocab, shortlist)`.
    Engine(Engine),
    /// A (possibly two-leg pivot) translation, from `Translator.load(src, trg)`.
    Translation(Translation),
}

impl Inner {
    fn translate(&self, text: &str) -> String {
        match self {
            Inner::Engine(e) => e.translate(text),
            Inner::Translation(t) => t.translate(text),
        }
    }

    fn translate_long(&self, text: &str) -> String {
        match self {
            Inner::Engine(e) => e.translate_long(text),
            Inner::Translation(t) => t.translate_long(text),
        }
    }
}

/// A translation engine callable from Python.
///
/// Owns its engine(s) (and, transitively, owned copies of the model bytes — the
/// engine borrows nothing from the `bytes` passed to the constructor), so it stays
/// valid across `translate` calls without the caller holding the buffers.
///
/// Two wasm-surface methods are deliberately omitted:
/// - `linear_memory_bytes` — it measures wasm linear-memory pages, which is
///   meaningless for a native process; use the OS / `resource` module for RSS.
/// - `translate_block_phased` — the wasm callback shim only exists because
///   `wasm32-unknown-unknown` has no usable `std::time::Instant`. Native code has
///   real timing; a Pythonic `translate_timed` can be added later if wanted.
/// `unsendable`: the gemmology-backed [`Engine`] holds a raw pointer to native
/// weight memory and is neither `Send` nor `Sync`, so a `Translator` is bound to the
/// thread that created it — PyO3 raises if it's touched from another thread.
#[pyclass(unsendable)]
struct Translator {
    inner: Inner,
}

#[pymethods]
impl Translator {
    /// Build a translator from model + vocabulary bytes, with an optional lexical
    /// shortlist — the exact byte-for-byte mirror of the wasm constructor.
    ///
    /// Pass byte-identical `src_vocab` and `trg_vocab` for shared-vocab pairs (most
    /// pairs, including en→es) and distinct buffers for split-vocab (CJK). When
    /// `shortlist` is given it restricts the output vocabulary per sentence
    /// (required for exact reference parity). A malformed model surfaces as
    /// `ValueError`, not a panic.
    #[new]
    #[pyo3(signature = (model, src_vocab, trg_vocab, shortlist = None))]
    fn new(
        model: &[u8],
        src_vocab: &[u8],
        trg_vocab: &[u8],
        shortlist: Option<&[u8]>,
    ) -> PyResult<Translator> {
        let engine =
            Engine::from_bytes(model, src_vocab, trg_vocab).map_err(PyValueError::new_err)?;
        let engine = match shortlist {
            Some(bytes) => engine.with_shortlist_bytes(bytes),
            None => engine,
        };
        Ok(Translator {
            inner: Inner::Engine(engine),
        })
    }

    /// Discover, download+cache (verified), and build a translator for `src`→`trg`
    /// in one call — the native, batteries-included path.
    ///
    /// Wraps [`load_translation`](fxtranslate::loader::load_translation): it fetches
    /// the Remote Settings records with the built-in HTTP client, resolves a direct
    /// model or a two-leg pivot, downloads+verifies every file into the cache, and
    /// builds the engine(s). Files already cached are reused, and if Remote Settings
    /// is unreachable it falls back to whatever is already cached — so a previously
    /// downloaded pair keeps translating offline.
    ///
    /// `cache_dir` overrides the platform-native cache location
    /// ([`Cache::locate`](fxtranslate::cache::Cache::locate), e.g.
    /// `~/Library/Caches/fxtranslate/models` on macOS). A resolution/download/build
    /// failure surfaces as `ValueError`.
    #[classmethod]
    #[pyo3(signature = (src, trg, cache_dir = None))]
    fn load(
        _cls: &Bound<'_, pyo3::types::PyType>,
        src: &str,
        trg: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Translator> {
        let cache = match cache_dir {
            Some(dir) => Cache::with_root(dir),
            None => Cache::locate(),
        };
        let fetch = NetworkFetch::new();
        let translation =
            load_translation(&fetch, &cache, src, trg).map_err(PyValueError::new_err)?;
        Ok(Translator {
            inner: Inner::Translation(translation),
        })
    }

    /// Translate a single sentence-unit `text` and return the target text.
    fn translate(&self, text: &str) -> String {
        self.inner.translate(text)
    }

    /// Translate `text` of arbitrary length: split into sentences with ICU4X (UAX
    /// #29, the same engine the native CLI uses) and translate each within the
    /// model's context window, rejoined with the original whitespace.
    fn translate_long(&self, text: &str) -> String {
        self.inner.translate_long(text)
    }

    /// The active int8 GEMM (matrix-multiply) backend, so a silent scalar build
    /// can't be mistaken for a SIMD one.
    ///
    /// Reports the gemmology SIMD backend name (e.g. `"i8mm+neon64"` on aarch64,
    /// `"avx2"` on x86_64) when the core crate's `build.rs` could compile the C++
    /// shim, else `"scalar"` — valuable for PyPI, where a source build on an
    /// unsupported target may legitimately fall back to scalar.
    ///
    /// Delegates to [`fxtranslate::gemm::backend`], which resolves the live backend
    /// *inside the core crate* where the `gemmology_simd` cfg is actually set — a
    /// `#[cfg(gemmology_simd)]` here would never fire, since that cfg is emitted only
    /// for the crate whose `build.rs` compiled the shim.
    fn backend(&self) -> String {
        fxtranslate::gemm::backend().to_string()
    }
}

/// The pure model-discovery surface, exposed to Python as a submodule.
///
/// Mirrors the wasm `discovery` free functions (`remote`/`route`/`segment`/`cache`),
/// but returns native Python objects — dicts, lists, tuples, `bytes` — instead of
/// the JSON strings the wasm code hand-builds to keep serde out of the wasm graph.
/// PyO3 has no such constraint, so these are idiomatic Python values. The input
/// contract is unchanged from wasm (the functions take the raw Remote Settings
/// records body), so behavior is identical; only the return type is Pythonic.
mod discovery {
    use super::*;
    use pyo3::types::{PyDict, PyTuple};

    /// One [`Record`] as a Python dict, with the same keys the wasm JSON uses.
    /// Absent optionals (`architecture`, `decompressedHash`) map to `None`.
    fn record_dict<'py>(py: Python<'py>, r: &Record) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("name", &r.name)?;
        d.set_item("fileType", &r.file_type)?;
        d.set_item("sourceLanguage", &r.src)?;
        d.set_item("targetLanguage", &r.trg)?;
        d.set_item("version", &r.version)?;
        d.set_item("architecture", &r.architecture)?;
        d.set_item("decompressedHash", &r.decompressed_hash)?;
        d.set_item("location", &r.location)?;
        Ok(d)
    }

    /// Parse a Remote Settings `records` response body into the model-file records
    /// fxtranslate understands, returned as a `list[dict]`.
    ///
    /// Normalizes to the fields routing needs (name, fileType, languages, version,
    /// architecture, decompressedHash, location), skipping records that lack the
    /// required fields — the same rule as the native parser. Raises `ValueError` on
    /// a parse error.
    #[pyfunction]
    fn parse_records<'py>(py: Python<'py>, body: &str) -> PyResult<Bound<'py, PyList>> {
        let records = core_parse_records(body).map_err(PyValueError::new_err)?;
        let dicts: Vec<Bound<'py, PyDict>> = records
            .iter()
            .map(|r| record_dict(py, r))
            .collect::<PyResult<_>>()?;
        PyList::new(py, dicts)
    }

    /// Resolve `src`→`trg` against a Remote Settings `records` body to the route
    /// that realizes it: a direct model, or a two-leg pivot through a hub.
    ///
    /// Takes the raw collection JSON (the same body `parse_records` accepts).
    /// Returns `{"kind": "direct", "src": .., "trg": ..}` or
    /// `{"kind": "pivot", "src": .., "pivot": .., "trg": ..}`. Raises `ValueError`
    /// when neither a direct model nor a pivot exists.
    #[pyfunction]
    fn resolve_route<'py>(
        py: Python<'py>,
        records_json: &str,
        src: &str,
        trg: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let records = core_parse_records(records_json).map_err(PyValueError::new_err)?;
        let route = core_resolve_route(&records, src, trg).map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        match route {
            Route::Direct { src, trg } => {
                d.set_item("kind", "direct")?;
                d.set_item("src", src)?;
                d.set_item("trg", trg)?;
            }
            Route::Pivot { src, pivot, trg } => {
                d.set_item("kind", "pivot")?;
                d.set_item("src", src)?;
                d.set_item("pivot", pivot)?;
                d.set_item("trg", trg)?;
            }
        }
        Ok(d)
    }

    /// Classify every language reachable through `hub` (e.g. `"en"`) by the
    /// directions it supports, from a Remote Settings `records` body — the data the
    /// `list` command renders.
    ///
    /// Returns `{"bidirectional": [..], "sourceOnly": [..], "targetOnly": [..]}`,
    /// each a sorted language-tag list. Raises `ValueError` only on a JSON parse
    /// error.
    #[pyfunction]
    fn catalog<'py>(
        py: Python<'py>,
        records_json: &str,
        hub: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let records = core_parse_records(records_json).map_err(PyValueError::new_err)?;
        let cat = core_catalog(&records, hub);
        let d = PyDict::new(py);
        d.set_item("bidirectional", cat.bidirectional)?;
        d.set_item("sourceOnly", cat.source_only)?;
        d.set_item("targetOnly", cat.target_only)?;
        Ok(d)
    }

    /// The unique, version-gated `src → trg` model pairs from a Remote Settings
    /// `records` body — the raw one-way models the `list --all` view enumerates.
    ///
    /// Returns a `list[tuple[str, str]]` of `(src, trg)` pairs, already sorted,
    /// deduplicated, and filtered to the supported model major. Raises `ValueError`
    /// on a JSON parse error.
    #[pyfunction]
    fn model_pairs<'py>(py: Python<'py>, records_json: &str) -> PyResult<Bound<'py, PyList>> {
        let records = core_parse_records(records_json).map_err(PyValueError::new_err)?;
        let ps = core_pairs(&records);
        let tuples: Vec<Bound<'py, PyTuple>> = ps
            .iter()
            .map(|(s, t)| PyTuple::new(py, [s, t]))
            .collect::<PyResult<_>>()?;
        PyList::new(py, tuples)
    }

    /// Split `text` into sentence units with ICU4X (UAX #29) and return their
    /// trimmed content as a `list[str]`.
    ///
    /// This is the same [`IcuSegmenter`] `Translator.translate_long` drives, exposed
    /// so a caller can segment before batching and get the exact boundaries the
    /// translate path will use. Whitespace between sentences is not preserved here
    /// (the CLI reassembles from the source); this yields the sentence contents in
    /// order.
    #[pyfunction]
    fn segment_sentences<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let spans = IcuSegmenter::new().sentences(text);
        let sentences: Vec<&str> = spans.iter().map(|s| s.of(text)).collect();
        PyList::new(py, sentences)
    }

    /// Decompress a zstd model attachment and verify the decompressed bytes against
    /// an expected hex SHA-256 (a record's `decompressedHash`), returning the
    /// decompressed `bytes`.
    ///
    /// Pass `expected_sha256_hex` = `None` to decompress without verifying (records
    /// that carry no hash). Raises `ValueError` on a decode failure or a hash
    /// mismatch.
    #[pyfunction]
    #[pyo3(signature = (compressed, expected_sha256_hex = None))]
    fn verify_and_decompress<'py>(
        py: Python<'py>,
        compressed: &[u8],
        expected_sha256_hex: Option<&str>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = core_verify_and_decompress(compressed, expected_sha256_hex)
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Register the `discovery` submodule and its functions on `parent`.
    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = parent.py();
        let m = PyModule::new(py, "discovery")?;
        // Fully-qualified name so tracebacks/tooling see the real dotted path rather
        // than a bare `discovery`.
        m.setattr("__name__", "fxtranslate._fxtranslate.discovery")?;
        m.add_function(wrap_pyfunction!(parse_records, &m)?)?;
        m.add_function(wrap_pyfunction!(resolve_route, &m)?)?;
        m.add_function(wrap_pyfunction!(catalog, &m)?)?;
        m.add_function(wrap_pyfunction!(model_pairs, &m)?)?;
        m.add_function(wrap_pyfunction!(segment_sentences, &m)?)?;
        m.add_function(wrap_pyfunction!(verify_and_decompress, &m)?)?;
        parent.add_submodule(&m)?;
        // Make `from fxtranslate._fxtranslate.discovery import ...` importable, not
        // just attribute access — register the submodule in `sys.modules`.
        py.import("sys")?
            .getattr("modules")?
            .set_item("fxtranslate._fxtranslate.discovery", &m)?;
        Ok(())
    }
}

/// The compiled extension module. Users import the friendly `fxtranslate` package
/// (`python/fxtranslate/__init__.py`), which re-exports this private submodule.
#[pymodule]
fn _fxtranslate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Translator>()?;
    discovery::register(m)?;
    Ok(())
}
