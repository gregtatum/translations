/* fxtranslate.h — C ABI for the native fxtranslate translation engine.
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * This header is the C++/Firefox-facing surface of the `fxtranslate-gecko` glue
 * crate. The WebIDL C++ shim (`FxTranslator`) #includes this and holds an
 * `FxEngine*` as its opaque handle, passing model/vocab/shortlist bytes straight
 * from the worker's ArrayBuffers.
 *
 * Hand-written to avoid pulling cbindgen into the build; keep it in lockstep with
 * the `#[no_mangle] extern "C"` functions in src/lib.rs.
 *
 * Safety / ownership contract (see src/lib.rs for the full rationale):
 *   - No Rust panic ever unwinds across this boundary: every entry point that runs
 *     engine logic is wrapped in catch_unwind and reports failure as a null/error
 *     return plus a thread-local last-error message.
 *   - All incoming buffers are treated as untrusted: (ptr,len) pairs are
 *     null/len-checked before any slice is formed, so malformed model/vocab/
 *     shortlist bytes become a clean error, not a crash or silent garbage.
 *   - Strings returned by fxtranslate_translate are heap-allocated by Rust and
 *     MUST be freed with fxtranslate_string_free using the exact (ptr,len) pair.
 *   - An FxEngine* MUST be freed exactly once via fxtranslate_engine_free.
 *   - An FxEngine holds native weight memory and is thread-confined: use one handle
 *     only from the thread that created it.
 */

#ifndef FXTRANSLATE_H
#define FXTRANSLATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque translation-engine handle. Never dereferenced by C/C++. */
typedef struct FxEngine FxEngine;

/*
 * Build an engine from in-memory model + vocab bytes, with an optional lexical
 * shortlist. Returns a heap-allocated handle, or NULL on error (retrieve the
 * reason with fxtranslate_last_error).
 *
 *   - shortlist_len == 0 (or shortlist_ptr == NULL) => no shortlist.
 *   - Pass the SAME vocab buffer as both src_vocab and trg_vocab for shared-vocab
 *     pairs (most pairs, incl. en->fr's single vocab.enfr.spm); distinct buffers
 *     for split-vocab (CJK).
 *
 * The engine copies out of these buffers, so the caller may free its ArrayBuffers
 * as soon as this returns. Malformed bytes yield NULL + a last-error, never a
 * panic across the boundary.
 */
FxEngine* fxtranslate_engine_new(const uint8_t* model_ptr, size_t model_len,
                                 const uint8_t* src_vocab_ptr, size_t src_vocab_len,
                                 const uint8_t* trg_vocab_ptr, size_t trg_vocab_len,
                                 const uint8_t* shortlist_ptr, size_t shortlist_len);

/*
 * Translate a UTF-8 text block. On success returns 0 and writes an owned target
 * buffer to *out_ptr / *out_len (free it with fxtranslate_string_free). On error
 * returns nonzero, sets *out_ptr = NULL / *out_len = 0, and records a last-error.
 *
 * Uses in-engine ICU4X (UAX#29) segmentation (Engine::translate_long), so the
 * caller feeds a whole text block rather than a pre-split sentence.
 */
int32_t fxtranslate_translate(FxEngine* engine, const uint8_t* text_ptr, size_t text_len,
                              uint8_t** out_ptr, size_t* out_len);

/*
 * Free a buffer returned by fxtranslate_translate. (ptr,len) must be exactly the
 * pair that call produced; NULL is a no-op.
 */
void fxtranslate_string_free(uint8_t* ptr, size_t len);

/* Free an engine handle. NULL is a no-op. Do not free the same handle twice. */
void fxtranslate_engine_free(FxEngine* engine);

/*
 * Copy this thread's last error message (UTF-8, NOT NUL-terminated) into out_ptr,
 * writing at most out_cap bytes. Returns the full message length so the caller can
 * size a buffer / detect truncation. Pass (NULL, 0) to query the length. Used when
 * fxtranslate_engine_new returns NULL or fxtranslate_translate returns nonzero.
 */
size_t fxtranslate_last_error(uint8_t* out_ptr, size_t out_cap);

/*
 * Copy the active int8 GEMM backend name (UTF-8, NOT NUL-terminated) into out_ptr;
 * same query-then-fill protocol as fxtranslate_last_error. E.g. "i8mm+neon64"
 * (aarch64), "avx2" (x86_64), or "scalar" (fallback). Lets C++/tests assert a
 * non-scalar kernel is live.
 */
size_t fxtranslate_backend(uint8_t* out_ptr, size_t out_cap);

/*
 * ===========================================================================
 * M2 token-alignment shape (S8a). translate_aligned emits, per translated unit,
 * the target text, the SPM-normalized source, both token arrays, and the soft
 * cross-attention alignment matrix — everything the JS HTML tag-transfer layer
 * needs. The whole struct-of-arrays result is one allocation freed in one call.
 * ===========================================================================
 */

/*
 * One token: vocab id plus a [begin, end) span in UTF-16 code units. Source-token
 * spans index into FxAligned.src_norm_ptr; target-token spans index into
 * FxAligned.text_ptr. (Offsets are u32; a DOM string is bounded well under
 * UINT32_MAX code units — the Rust side saturates rather than wraps.)
 */
typedef struct FxToken {
  uint32_t id;
  uint32_t begin; /* UTF-16 code units, inclusive */
  uint32_t end;   /* UTF-16 code units, exclusive */
} FxToken;

/*
 * Struct-of-arrays result of fxtranslate_translate_aligned. Every pointer is a
 * Rust heap allocation owned by this struct; the whole thing is reclaimed in one
 * call to fxtranslate_aligned_free. Strings are UTF-8, NOT NUL-terminated (use the
 * paired _len). The alignment matrix is flat row-major:
 *   align_ptr[r * align_cols + c] = P(source_tokens[c] | target_tokens[r])
 * with align_rows == trg_tokens_len and align_cols == src_tokens_len.
 */
typedef struct FxAligned {
  uint8_t* text_ptr; /* target text (UTF-8) */
  size_t text_len;
  uint8_t* src_norm_ptr; /* SPM-normalized source (UTF-8) */
  size_t src_norm_len;
  FxToken* src_tokens_ptr; /* source tokens, incl. trailing EOS */
  size_t src_tokens_len;
  FxToken* trg_tokens_ptr; /* target tokens, incl. terminal EOS */
  size_t trg_tokens_len;
  float* align_ptr; /* flat [align_rows * align_cols], [trg][src] */
  size_t align_rows;
  size_t align_cols;
} FxAligned;

/*
 * Translate a UTF-8 unit and emit token alignments. On success returns 0 and
 * writes an owned FxAligned* to *out (free it with fxtranslate_aligned_free). On
 * error returns nonzero, sets *out = NULL, and records a last-error.
 *
 * Uses the single-sequence aligned path (Engine::translate_aligned) — NO sentence
 * segmentation, unlike fxtranslate_translate. The HTML layer segments upstream and
 * calls this once per unit.
 */
int32_t fxtranslate_translate_aligned(FxEngine* engine, const uint8_t* text_ptr,
                                      size_t text_len, FxAligned** out);

/*
 * Free an FxAligned* returned by fxtranslate_translate_aligned: releases both
 * strings, both token arrays, the alignment matrix, and the struct itself. NULL is
 * a no-op. Do not free the same pointer twice.
 */
void fxtranslate_aligned_free(FxAligned* ptr);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FXTRANSLATE_H */
