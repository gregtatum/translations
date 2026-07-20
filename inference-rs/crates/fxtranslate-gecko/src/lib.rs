//! C ABI over the native `fxtranslate` engine, for Firefox/Gecko C++ to call.
//!
//! This is the **third binding** of the same one-shape engine surface — a direct
//! sibling of `fxtranslate-wasm` (`Translator::new` at
//! `crates/fxtranslate-wasm/src/lib.rs`) and `fxtranslate-py`
//! (`Translator::new` at `crates/fxtranslate-py/src/lib.rs`). All three wrap the
//! same byte-path entry points ([`Engine::from_bytes`] +
//! [`Engine::with_shortlist_bytes`] + [`Engine::translate_long`]); this one
//! exposes them as an opaque-handle `extern "C"` surface so a WebIDL C++ shim
//! (`FxTranslator`) can hold a `void*` and pass model/vocab/shortlist bytes
//! straight from the worker's `ArrayBuffer`s.
//!
//! # Safety contract with C++
//!
//! Every entry point that runs engine logic wraps its body in
//! [`std::panic::catch_unwind`]: a Rust panic unwinding across the FFI boundary
//! into C++ is undefined behavior, so a caught panic is turned into a clean
//! null/error-code return plus a thread-local last-error message. **All** incoming
//! buffers are treated as untrusted — pointers are null/len-checked before a slice
//! is ever formed, so malformed or truncated model/vocab/shortlist bytes become a
//! clean error, never a panic, an out-of-bounds read, or silent garbage.
//!
//! Ownership: strings returned by [`fxtranslate_translate`] are heap-allocated by
//! Rust and MUST be freed by the caller via [`fxtranslate_string_free`] with the
//! exact `(ptr, len)` pair returned. The engine handle MUST be freed exactly once
//! via [`fxtranslate_engine_free`]. The engine owns its model bytes (it copies out
//! of the constructor slices — see [`Engine::from_bytes`]), so the caller may free
//! its `ArrayBuffer`s immediately after `fxtranslate_engine_new` returns.
//!
//! [`Engine::from_bytes`]: fxtranslate::engine::Engine::from_bytes
//! [`Engine::with_shortlist_bytes`]: fxtranslate::engine::Engine::with_shortlist_bytes
//! [`Engine::translate_long`]: fxtranslate::engine::Engine::translate_long

use std::cell::RefCell;
use std::os::raw::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use fxtranslate::engine::Engine;

/// Opaque handle the C++ side holds as a `void*`. Never dereferenced by C++.
///
/// A `#[repr(C)]` newtype around the owned [`Engine`] so the header can name the
/// type (`FxEngine`) without exposing the Rust layout. The engine holds a raw
/// pointer into native weight memory and is neither `Send` nor `Sync`, so the C++
/// side must confine one handle to the thread that created it (the same rule the
/// PyO3 binding enforces via `unsendable`).
#[repr(C)]
pub struct FxEngine {
    engine: Engine,
}

thread_local! {
    /// The last error message, per thread. Set whenever an entry point returns
    /// null / a nonzero code, so C++ can retrieve *why* (see
    /// [`fxtranslate_last_error`]). Thread-local because the engine handle is
    /// thread-confined; each worker thread sees only its own failures.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Record `msg` as this thread's last error (replacing any previous one).
fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

/// Clear this thread's last error (called at the start of a fallible op so a stale
/// message can't be mistaken for a fresh failure).
fn clear_last_error() {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
}

/// Turn a `catch_unwind` payload into a human-readable string. `panic!("msg")`
/// and `panic!("{}", x)` land as `&str`/`String`; anything else is opaque.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

/// Form a `&[u8]` from an untrusted `(ptr, len)` pair, or return `Err(msg)`.
///
/// A zero length yields an empty slice (a null pointer is fine when `len == 0` —
/// the C++ side passes `null, 0` for "no shortlist"). A null pointer with a
/// nonzero length is a caller bug and becomes a clean error, not a segfault.
fn slice_from_raw<'a>(ptr: *const u8, len: usize, what: &str) -> Result<&'a [u8], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(format!("{what}: null pointer with non-zero length {len}"));
    }
    // SAFETY: ptr is non-null and len > 0 (checked above). The caller (the WebIDL
    // C++ shim) contracts to pass a valid buffer of at least `len` bytes that
    // stays alive for the duration of the call; the engine copies out of it before
    // returning, so no borrow escapes.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Build an engine from in-memory model + vocab bytes, with an optional lexical
/// shortlist. Returns a heap-allocated handle, or null on error (retrieve the
/// reason with [`fxtranslate_last_error`]).
///
/// - `shortlist_len == 0` (or `shortlist_ptr == null`) => no shortlist.
/// - Pass the **same** vocab buffer as both `src_vocab` and `trg_vocab` for
///   shared-vocab pairs (most pairs, including en→fr, which ship one
///   `vocab.enfr.spm`); distinct buffers for split-vocab (CJK).
///
/// Mirrors `Translator::new` in the wasm/py crates. Malformed bytes (truncated
/// model, bad shortlist header, garbage vocab) surface as a null return + a
/// last-error, never a panic across the boundary.
///
/// # Safety
///
/// Each `(ptr, len)` pair must either be `(null, 0)` or point to a readable buffer
/// of at least `len` bytes that lives for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_engine_new(
    model_ptr: *const u8,
    model_len: usize,
    src_vocab_ptr: *const u8,
    src_vocab_len: usize,
    trg_vocab_ptr: *const u8,
    trg_vocab_len: usize,
    shortlist_ptr: *const u8,
    shortlist_len: usize,
) -> *mut FxEngine {
    clear_last_error();

    // AssertUnwindSafe: on a panic we discard everything and return null; nothing
    // partially-built escapes, so there is no broken-invariant hazard to guard.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let model = slice_from_raw(model_ptr, model_len, "model")?;
        let src_vocab = slice_from_raw(src_vocab_ptr, src_vocab_len, "src_vocab")?;
        let trg_vocab = slice_from_raw(trg_vocab_ptr, trg_vocab_len, "trg_vocab")?;

        // from_bytes already returns Result on a structurally-wrong model. Any
        // deeper panic in weight parsing (weights.rs expect/panic on truncated or
        // structurally-wrong buffers) is caught by the surrounding catch_unwind.
        let engine = Engine::from_bytes(model, src_vocab, trg_vocab)?;

        // Shortlist::from_bytes returns a plain Shortlist (not a Result) and
        // panics on a truncated/garbage buffer (try_into().unwrap on a short
        // slice). The catch_unwind converts that panic into the Err below.
        let engine = if shortlist_len == 0 {
            engine
        } else {
            let shortlist = slice_from_raw(shortlist_ptr, shortlist_len, "shortlist")?;
            engine.with_shortlist_bytes(shortlist)
        };

        Ok::<Engine, String>(engine)
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(FxEngine { engine })),
        Ok(Err(msg)) => {
            set_last_error(msg);
            std::ptr::null_mut()
        }
        Err(payload) => {
            set_last_error(format!(
                "panic while building engine (malformed model/vocab/shortlist bytes): {}",
                panic_message(&*payload)
            ));
            std::ptr::null_mut()
        }
    }
}

/// Translate `text` (a UTF-8 text block) and write the target text to `*out_ptr` /
/// `*out_len`. Returns 0 on success, nonzero on error.
///
/// Uses [`Engine::translate_long`] — in-engine ICU4X (UAX#29) segmentation — since
/// the worker feeds a text block, not a pre-split sentence. On success the caller
/// owns the returned buffer and MUST free it with [`fxtranslate_string_free`]
/// using the returned `(ptr, len)`. On error `*out_ptr` is set null and `*out_len`
/// zero, and the reason is retrievable via [`fxtranslate_last_error`].
///
/// # Safety
///
/// `engine` must be a live handle from [`fxtranslate_engine_new`]. `text_ptr`/
/// `text_len` must describe a readable buffer (or be `(null, 0)`). `out_ptr` and
/// `out_len` must be non-null, writable pointers.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_translate(
    engine: *mut FxEngine,
    text_ptr: *const u8,
    text_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    clear_last_error();

    if out_ptr.is_null() || out_len.is_null() {
        set_last_error("translate: null out_ptr/out_len");
        return 1;
    }
    // Default the out-params to empty so an early error leaves them well-defined.
    // SAFETY: non-null checked directly above.
    unsafe {
        *out_ptr = std::ptr::null_mut();
        *out_len = 0;
    }

    if engine.is_null() {
        set_last_error("translate: null engine handle");
        return 1;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: non-null checked above; the caller contracts the handle is live
        // and thread-confined. We borrow, never take ownership.
        let engine = unsafe { &(*engine).engine };
        let bytes = slice_from_raw(text_ptr, text_len, "text")?;
        let text = std::str::from_utf8(bytes).map_err(|e| format!("text is not UTF-8: {e}"))?;
        Ok::<String, String>(engine.translate_long(text))
    }));

    match result {
        Ok(Ok(out)) => {
            // Hand the String's buffer to C as (ptr, len). We deliberately shrink
            // to exact capacity so `fxtranslate_string_free` can reconstruct the
            // Vec with cap == len. Leak it; the caller frees via string_free.
            let mut bytes = out.into_bytes();
            bytes.shrink_to_fit();
            debug_assert_eq!(bytes.len(), bytes.capacity());
            let len = bytes.len();
            let ptr = bytes.as_mut_ptr();
            std::mem::forget(bytes);
            // SAFETY: out_ptr/out_len non-null checked above.
            unsafe {
                *out_ptr = ptr;
                *out_len = len;
            }
            0
        }
        Ok(Err(msg)) => {
            set_last_error(msg);
            2
        }
        Err(payload) => {
            set_last_error(format!("panic during translate: {}", panic_message(&*payload)));
            3
        }
    }
}

/// Free a string buffer returned by [`fxtranslate_translate`]. `(ptr, len)` must be
/// exactly the pair that call produced (or `(null, 0)`, a no-op). Double-free or a
/// mismatched len is undefined behavior — the standard C free contract.
///
/// # Safety
///
/// See above: `ptr`/`len` must originate from a single [`fxtranslate_translate`]
/// call and not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_string_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: reconstruct the exact Vec<u8> we forgot in fxtranslate_translate
    // (cap == len because we shrank to fit before forgetting). Dropping it frees
    // the buffer through the same global allocator that allocated it.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }));
}

/// Free an engine handle from [`fxtranslate_engine_new`]. Passing null is a no-op.
/// Freeing the same handle twice is undefined behavior.
///
/// # Safety
///
/// `engine` must be a handle from [`fxtranslate_engine_new`] not already freed.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_engine_free(engine: *mut FxEngine) {
    if engine.is_null() {
        return;
    }
    // SAFETY: reclaim the Box we leaked in engine_new. catch_unwind guards a
    // (very unlikely) panic in Engine's Drop from crossing the boundary.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(engine));
    }));
}

/// Write this thread's last error message (UTF-8, NOT NUL-terminated) to a
/// caller-provided buffer. Returns the number of bytes the message occupies.
///
/// Call with `out_ptr == null` (or a too-small `out_cap`) to query the length,
/// then again with a buffer of that size. Writes `min(msg_len, out_cap)` bytes;
/// the return value is always the full message length so the caller can detect
/// truncation. Never panics.
///
/// # Safety
///
/// If `out_ptr` is non-null it must point to at least `out_cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_last_error(out_ptr: *mut u8, out_cap: usize) -> usize {
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        let bytes = msg.as_bytes();
        let full_len = bytes.len();
        if !out_ptr.is_null() && out_cap > 0 {
            let n = full_len.min(out_cap);
            // SAFETY: out_ptr has out_cap writable bytes (caller contract); n <= out_cap.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, n);
            }
        }
        full_len
    })
}

/// Write the active int8 GEMM backend name (UTF-8, NOT NUL-terminated) to a
/// caller-provided buffer, returning the full byte length. Same query-then-fill
/// protocol as [`fxtranslate_last_error`].
///
/// Lets the C++ side / tests assert a non-scalar kernel is live: `"i8mm+neon64"`
/// on aarch64, `"avx2"` on x86_64, `"scalar"` on a fallback build. Delegates to
/// [`fxtranslate::gemm::backend`], resolved inside the engine crate where the
/// `gemmology_simd` cfg is actually set.
///
/// # Safety
///
/// If `out_ptr` is non-null it must point to at least `out_cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fxtranslate_backend(out_ptr: *mut u8, out_cap: usize) -> usize {
    let name = fxtranslate::gemm::backend();
    let bytes = name.as_bytes();
    let full_len = bytes.len();
    if !out_ptr.is_null() && out_cap > 0 {
        let n = full_len.min(out_cap);
        // SAFETY: out_ptr has out_cap writable bytes (caller contract); n <= out_cap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, n);
        }
    }
    full_len
}

// ============================================================================
// M2-ready shape — RESERVED, NOT IMPLEMENTED (see 01-binding-and-build.md §4
// "M2-ready shape"). Defined here as commented-out signatures so the seam
// (engine ↔ JS HTML handler = plain text + alignments) is not designed out. M2
// needs `translate(text) -> { text, alignments }` for the DOMParser HTML layer;
// whether fxtranslate can populate FxAlignment at all is Workstream B's gate
// (cross-attention head vs greenfield). Do NOT build this now.
//
// /// Token-level source<->target alignment for one translated unit. Placeholder
// /// shape gated by Workstream B (does fxtranslate emit alignments today?).
// #[repr(C)]
// pub struct FxAlignment {
//     pub src_token_start: u32,
//     pub src_token_len: u32,
//     pub trg_token_start: u32,
//     pub trg_token_len: u32,
//     // + probability/weight if the cross-attention head yields one.
// }
//
// /// Translate + emit alignments. out_text_* freed via fxtranslate_string_free;
// /// out_align_* freed via a future fxtranslate_alignments_free(ptr, count).
// #[no_mangle]
// pub unsafe extern "C" fn fxtranslate_translate_aligned(
//     engine: *mut FxEngine,
//     text_ptr: *const u8, text_len: usize,
//     out_text_ptr: *mut *mut u8, out_text_len: *mut usize,
//     out_align_ptr: *mut *mut FxAlignment, out_align_count: *mut usize,
// ) -> c_int;
// ============================================================================
