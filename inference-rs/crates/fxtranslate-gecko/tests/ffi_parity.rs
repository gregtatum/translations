//! Layer-1 correctness (01-binding-and-build.md §5): the C ABI must be
//! **byte-exact** to the standalone `fxtranslate` engine on the real en-fr model,
//! in the same process on the same arch with the same feature set. A single diff
//! is an ABI bug — truncation, wrong vocab side, or UTF-8 mishandling — not a
//! tolerable float divergence (same libm, same GEMM as the direct call).
//!
//! Also proves the hardening contract: malformed model/shortlist bytes yield a
//! null handle + a last-error, and do NOT panic or abort across the boundary.
//!
//! The crate is built with `crate-type = ["staticlib", "rlib"]`, so this test
//! links the rlib and calls the exact same `#[no_mangle] extern "C"` functions
//! Firefox's C++ links against — the C ABI is exercised directly, in Rust.

use std::path::Path;
use std::ptr;

use fxtranslate::engine::Engine;
use fxtranslate_gecko::{
    fxtranslate_aligned_free, fxtranslate_backend, fxtranslate_engine_free, fxtranslate_engine_new,
    fxtranslate_last_error, fxtranslate_string_free, fxtranslate_translate,
    fxtranslate_translate_aligned, FxAligned, FxEngine,
};

// Same layout / skip-if-absent convention the oracle tests use (paths are
// relative to the crate manifest dir, which is crates/fxtranslate-gecko).
const MODEL_DIR: &str = "../../../data/models/enfr";

fn model_path(file: &str) -> std::path::PathBuf {
    Path::new(MODEL_DIR).join(file)
}

/// Load the en-fr model/vocab/shortlist bytes, or `None` if the model tree is
/// absent (so the test skips cleanly on a machine without the data, exactly like
/// the oracle tests).
fn load_enfr_bytes() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let model = model_path("model.enfr.intgemm.alphas.bin");
    let vocab = model_path("vocab.enfr.spm");
    let shortlist = model_path("lex.50.50.enfr.s2t.bin");
    if !model.exists() || !vocab.exists() || !shortlist.exists() {
        return None;
    }
    Some((
        std::fs::read(model).expect("model reads"),
        std::fs::read(vocab).expect("vocab reads"),
        std::fs::read(shortlist).expect("shortlist reads"),
    ))
}

/// Retrieve the thread-local last error via the C ABI (query length, then fill).
fn last_error() -> String {
    // SAFETY: query-then-fill protocol; null query is always valid.
    let len = unsafe { fxtranslate_last_error(ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len];
    // SAFETY: buf has `len` writable bytes.
    let written = unsafe { fxtranslate_last_error(buf.as_mut_ptr(), buf.len()) };
    assert_eq!(written, len, "last_error length is stable across calls");
    String::from_utf8_lossy(&buf).into_owned()
}

/// Drive the full C ABI (engine_new + translate + string_free + engine_free) on
/// `text`, returning the target string. `engine` is built fresh per call so the
/// test also exercises the load path each time.
///
/// # Safety
/// Requires valid model/vocab/shortlist byte slices.
unsafe fn c_abi_translate(
    model: &[u8],
    vocab: &[u8],
    shortlist: &[u8],
    text: &str,
) -> String {
    // Shared vocab: en-fr ships one vocab.enfr.spm used as both src and trg.
    let engine: *mut FxEngine = fxtranslate_engine_new(
        model.as_ptr(),
        model.len(),
        vocab.as_ptr(),
        vocab.len(),
        vocab.as_ptr(),
        vocab.len(),
        shortlist.as_ptr(),
        shortlist.len(),
    );
    assert!(
        !engine.is_null(),
        "engine_new returned null on the real model: {}",
        last_error()
    );

    let mut out_ptr: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    let rc = fxtranslate_translate(
        engine,
        text.as_ptr(),
        text.len(),
        &mut out_ptr,
        &mut out_len,
    );
    assert_eq!(rc, 0, "translate returned {rc}: {}", last_error());
    assert!(!out_ptr.is_null(), "translate gave a null buffer on rc==0");

    let bytes = std::slice::from_raw_parts(out_ptr, out_len).to_vec();
    let out = String::from_utf8(bytes).expect("C ABI output is valid UTF-8");

    fxtranslate_string_free(out_ptr, out_len);
    fxtranslate_engine_free(engine);
    out
}

/// Build the engine directly (no C ABI) — the standalone reference the binding
/// must match byte-for-byte, using the identical from_bytes + with_shortlist_bytes
/// + translate_long chain.
fn direct_translate(model: &[u8], vocab: &[u8], shortlist: &[u8], text: &str) -> String {
    let engine = Engine::from_bytes(model, vocab, vocab)
        .expect("direct from_bytes")
        .with_shortlist_bytes(shortlist);
    engine.translate_long(text)
}

const SENTENCES: &[&str] = &[
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Firefox is a web browser developed by Mozilla.",
    "This is a longer paragraph. It has multiple sentences. Each one is translated within the model's context window and then rejoined with the original whitespace.",
    "Numbers like 42 and punctuation — em dashes, quotes \"like this\" — should round-trip.",
];

#[test]
fn c_abi_is_byte_exact_to_standalone_engine() {
    let Some((model, vocab, shortlist)) = load_enfr_bytes() else {
        eprintln!("skipping ffi_parity: en-fr model absent at {MODEL_DIR}");
        return;
    };

    for &text in SENTENCES {
        // SAFETY: valid buffers.
        let via_c = unsafe { c_abi_translate(&model, &vocab, &shortlist, text) };
        let direct = direct_translate(&model, &vocab, &shortlist, text);
        assert_eq!(
            via_c, direct,
            "C ABI diverged from standalone engine for input {text:?}\n  C ABI: {via_c:?}\n  direct: {direct:?}"
        );
        assert!(!via_c.is_empty(), "empty translation for {text:?}");
    }
}

#[test]
fn c_abi_no_shortlist_matches_standalone() {
    let Some((model, vocab, _shortlist)) = load_enfr_bytes() else {
        eprintln!("skipping ffi_parity (no-shortlist): en-fr model absent");
        return;
    };
    let text = "Hello, world!";

    // shortlist_len == 0 => no shortlist (null ptr path).
    let via_c = unsafe {
        let engine = fxtranslate_engine_new(
            model.as_ptr(),
            model.len(),
            vocab.as_ptr(),
            vocab.len(),
            vocab.as_ptr(),
            vocab.len(),
            ptr::null(),
            0,
        );
        assert!(!engine.is_null(), "no-shortlist engine_new null: {}", last_error());
        let mut out_ptr = ptr::null_mut();
        let mut out_len = 0usize;
        let rc = fxtranslate_translate(engine, text.as_ptr(), text.len(), &mut out_ptr, &mut out_len);
        assert_eq!(rc, 0, "no-shortlist translate rc {rc}: {}", last_error());
        let out = String::from_utf8(std::slice::from_raw_parts(out_ptr, out_len).to_vec()).unwrap();
        fxtranslate_string_free(out_ptr, out_len);
        fxtranslate_engine_free(engine);
        out
    };

    let direct = Engine::from_bytes(&model, &vocab, &vocab)
        .expect("direct from_bytes")
        .translate_long(text);
    assert_eq!(via_c, direct, "no-shortlist C ABI diverged from standalone");
}

#[test]
fn malformed_model_returns_null_not_panic() {
    let Some((model, vocab, shortlist)) = load_enfr_bytes() else {
        eprintln!("skipping ffi_parity (malformed): en-fr model absent");
        return;
    };

    // Truncated model buffer: from_bytes / weights parsing must not panic across
    // the boundary — engine_new returns null with a last-error set.
    let truncated_model = &model[..model.len() / 2];
    let engine = unsafe {
        fxtranslate_engine_new(
            truncated_model.as_ptr(),
            truncated_model.len(),
            vocab.as_ptr(),
            vocab.len(),
            vocab.as_ptr(),
            vocab.len(),
            shortlist.as_ptr(),
            shortlist.len(),
        )
    };
    assert!(engine.is_null(), "truncated model must yield null, not a handle");
    assert!(!last_error().is_empty(), "truncated model must set a last-error");

    // Truncated shortlist: Shortlist::from_bytes panics (try_into().unwrap on a
    // short slice); catch_unwind must turn that into a null + last-error.
    let truncated_shortlist = &shortlist[..8]; // < one full header
    let engine2 = unsafe {
        fxtranslate_engine_new(
            model.as_ptr(),
            model.len(),
            vocab.as_ptr(),
            vocab.len(),
            vocab.as_ptr(),
            vocab.len(),
            truncated_shortlist.as_ptr(),
            truncated_shortlist.len(),
        )
    };
    assert!(
        engine2.is_null(),
        "truncated shortlist must yield null, not a handle"
    );
    assert!(
        !last_error().is_empty(),
        "truncated shortlist must set a last-error"
    );
}

#[test]
fn null_and_empty_inputs_are_clean_errors() {
    // Null model with a nonzero length: a caller bug, must be a clean error not a
    // segfault. (Length nonzero so slice_from_raw actually rejects the null ptr.)
    let engine = unsafe {
        fxtranslate_engine_new(ptr::null(), 16, ptr::null(), 0, ptr::null(), 0, ptr::null(), 0)
    };
    assert!(engine.is_null(), "null model ptr must yield null");
    assert!(!last_error().is_empty(), "null model must set a last-error");

    // Null out-params on translate: rc != 0, no crash.
    let rc = unsafe {
        fxtranslate_translate(ptr::null_mut(), ptr::null(), 0, ptr::null_mut(), ptr::null_mut())
    };
    assert_ne!(rc, 0, "null out-params must be a nonzero rc");

    // Freeing null is a no-op.
    unsafe {
        fxtranslate_string_free(ptr::null_mut(), 0);
        fxtranslate_engine_free(ptr::null_mut());
    }
}

/// The aligned C ABI matches the standalone engine and returns a well-formed
/// struct-of-arrays: text equals `translate_aligned().target_text` (and, for a
/// single sentence, `translate_long`), the token counts match the matrix
/// dimensions, and every alignment row is a distribution.
#[test]
fn c_abi_aligned_matches_standalone_and_is_well_formed() {
    let Some((model, vocab, shortlist)) = load_enfr_bytes() else {
        eprintln!("skipping ffi_parity (aligned): en-fr model absent");
        return;
    };

    // translate_aligned is the single-sequence path, so use single sentences.
    let inputs = &[
        "Hello, world!",
        "Firefox is a web browser developed by Mozilla.",
    ];

    for &text in inputs {
        let direct = Engine::from_bytes(&model, &vocab, &vocab)
            .expect("direct from_bytes")
            .with_shortlist_bytes(&shortlist)
            .translate_aligned(text);

        // Single-sentence: the aligned target equals the plain translate_long.
        let plain = direct_translate(&model, &vocab, &shortlist, text);
        assert_eq!(
            direct.target_text, plain,
            "aligned target_text must equal translate_long for a single sentence: {text:?}"
        );

        // SAFETY: valid buffers; drive the aligned C ABI end to end.
        unsafe {
            let engine = fxtranslate_engine_new(
                model.as_ptr(),
                model.len(),
                vocab.as_ptr(),
                vocab.len(),
                vocab.as_ptr(),
                vocab.len(),
                shortlist.as_ptr(),
                shortlist.len(),
            );
            assert!(!engine.is_null(), "aligned engine_new null: {}", last_error());

            let mut out: *mut FxAligned = ptr::null_mut();
            let rc = fxtranslate_translate_aligned(engine, text.as_ptr(), text.len(), &mut out);
            assert_eq!(rc, 0, "translate_aligned rc {rc}: {}", last_error());
            assert!(!out.is_null(), "translate_aligned gave null on rc==0");

            let a = &*out;

            // text == standalone target_text.
            let ffi_text =
                String::from_utf8(std::slice::from_raw_parts(a.text_ptr, a.text_len).to_vec())
                    .expect("aligned text is UTF-8");
            assert_eq!(ffi_text, direct.target_text, "aligned C ABI text diverged: {text:?}");

            // source_normalized round-trips.
            let ffi_src_norm = String::from_utf8(
                std::slice::from_raw_parts(a.src_norm_ptr, a.src_norm_len).to_vec(),
            )
            .expect("aligned src_norm is UTF-8");
            assert_eq!(ffi_src_norm, direct.source_normalized, "src_norm diverged: {text:?}");

            // Token counts match the standalone shape and the matrix dims.
            assert_eq!(a.src_tokens_len, direct.source_tokens.len(), "src token count");
            assert_eq!(a.trg_tokens_len, direct.target_tokens.len(), "trg token count");
            assert_eq!(a.align_rows, a.trg_tokens_len, "rows == trg tokens");
            assert_eq!(a.align_cols, a.src_tokens_len, "cols == src tokens");

            // Token spans mirror the engine (usize -> u32).
            let src_toks = std::slice::from_raw_parts(a.src_tokens_ptr, a.src_tokens_len);
            for (ffi, eng) in src_toks.iter().zip(&direct.source_tokens) {
                assert_eq!(ffi.id, eng.id, "src token id");
                assert_eq!(ffi.begin as usize, eng.begin, "src token begin");
                assert_eq!(ffi.end as usize, eng.end, "src token end");
            }
            let trg_toks = std::slice::from_raw_parts(a.trg_tokens_ptr, a.trg_tokens_len);
            for (ffi, eng) in trg_toks.iter().zip(&direct.target_tokens) {
                assert_eq!(ffi.id, eng.id, "trg token id");
                assert_eq!(ffi.begin as usize, eng.begin, "trg token begin");
                assert_eq!(ffi.end as usize, eng.end, "trg token end");
            }

            // Every alignment row is a valid distribution summing to ~1, and the
            // flat matrix matches the standalone nested rows exactly.
            let flat = std::slice::from_raw_parts(a.align_ptr, a.align_rows * a.align_cols);
            for r in 0..a.align_rows {
                let row = &flat[r * a.align_cols..(r + 1) * a.align_cols];
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "alignment row {r} sum {sum} not ~1 for {text:?}"
                );
                assert_eq!(row, direct.alignments[r].as_slice(), "flat row {r} != nested");
            }

            fxtranslate_aligned_free(out);
            fxtranslate_engine_free(engine);
        }
    }
}

/// The aligned path fails cleanly on bad inputs: a null out-pointer and a null
/// engine both return nonzero without panicking, and freeing null is a no-op.
#[test]
fn c_abi_aligned_bad_inputs_are_clean_errors() {
    // Null out-pointer: nonzero rc, no crash.
    let rc = unsafe { fxtranslate_translate_aligned(ptr::null_mut(), ptr::null(), 0, ptr::null_mut()) };
    assert_ne!(rc, 0, "null out must be a nonzero rc");

    // Null engine with a valid out-pointer: nonzero rc, *out left null.
    let mut out: *mut FxAligned = 1 as *mut FxAligned;
    let rc = unsafe {
        fxtranslate_translate_aligned(ptr::null_mut(), ptr::null(), 0, &mut out)
    };
    assert_ne!(rc, 0, "null engine must be a nonzero rc");
    assert!(out.is_null(), "null engine must leave *out null");

    // Freeing null is a no-op.
    unsafe {
        fxtranslate_aligned_free(ptr::null_mut());
    }
}

#[test]
fn backend_is_reported_and_ideally_simd() {
    // SAFETY: query-then-fill.
    let len = unsafe { fxtranslate_backend(ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len];
    let written = unsafe { fxtranslate_backend(buf.as_mut_ptr(), buf.len()) };
    assert_eq!(written, len);
    let backend = String::from_utf8(buf).expect("backend name is UTF-8");
    assert!(!backend.is_empty(), "backend name must be non-empty");
    eprintln!("fxtranslate_backend => {backend:?}");

    // Soft assertion: on this arm64 host we expect a real SIMD kernel. Keep it a
    // logged warning rather than a hard failure so a scalar-only test build (no
    // C++ toolchain, or --features portable) still passes.
    if backend == "scalar" {
        eprintln!(
            "WARNING: GEMM backend is 'scalar' — SIMD kernel not compiled in this build. \
             Set FXTRANSLATE_REQUIRE_SIMD=1 to hard-fail the engine build if the kernel is expected."
        );
    } else {
        assert_ne!(backend, "scalar");
    }
}
