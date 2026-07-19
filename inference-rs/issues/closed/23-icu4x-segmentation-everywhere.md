# Use ICU4X sentence segmentation everywhere

**Closed, done.** Today the native CLI and the wasm/npm package segment with two
*different* engines, so identical input can produce different sentence boundaries
across the two artifacts. Unify on **one** engine — ICU4X (`icu_segmenter`) — for
both, and delete the wiring that only exists to route around wasm being icu-free.
Keep [`BasicSegmenter`](../../crates/fxtranslate/src/segment.rs) available as an
opt-out for size-sensitive or dependency-free builds. This is the foundation for
[24-abbreviation-suppression.md](../24-abbreviation-suppression.md); the prefix-table
follow-up is [25-segmenter-prefix-tables.md](../25-segmenter-prefix-tables.md).

**Resolution.** wasm now enables the `icu-segmenter` feature, so `translate_long`
and `segmentSentences` both route through ICU4X — the same engine as the CLI. The
icu-free comments are gone from `lib.rs`/`segment.rs`/the Cargo manifests; the
single-engine + ICU4X≡Firefox rationale is recorded in `architecture.md` and both
READMEs. A native≡wasm segmentation equality check over a corpus (exercising
`U.S.`, `...`, CJK, and abbreviations where the engines diverge) lives in
`crates/fxtranslate-wasm/tests/discovery_parity.rs`, proven under both `cargo test`
and `wasm-pack test --node`. `BasicSegmenter` remains the `icu-segmenter`-off
opt-out and that build links no ICU (verified via `cargo tree`). wasm module grew
330 KB → 348 KB (+18 KB).

## Current state: two engines, one product

- **Native CLI** (`fxtranslate-cli`, feature `icu-segmenter` on) →
  `IcuSegmenter`, ICU4X `SentenceSegmenter` with `SentenceBreakInvariantOptions`
  (`segment.rs:172`). Rule-based UAX #29, bundled `compiled_data`, no dictionary
  data needed (sentence breaking is rule-based; the large word/line dictionaries
  are for space-less-script *word* breaking, which we don't do).
- **wasm/npm** → `BasicSegmenter`, the dependency-free punctuation splitter
  (`segment.rs:111`), because the wasm build is icu-free. `Engine::translate_long`
  falls back to it (`engine.rs:307`) and `lib/translate.js` drives
  `translate_long`.

`BasicSegmenter` diverges from ICU4X on abbreviations, initials, and ellipses
(e.g. it splits `U.S.`, `z. B.`, and `...`), so the npm package and the native
CLI disagree on where sentences break. The conformance harness misses it because
`scripts/conformance_translate.py` feeds input line-by-line (already segmented).

## Why ICU4X, and why it's cheap in wasm

**Firefox's `Intl.Segmenter` is backed by ICU4X** (Rust) — the *same* library
family as our `IcuSegmenter`. (Note: Node/Chrome `Intl.Segmenter` is ICU4C via
V8 — a *different* implementation, so "just call the host `Intl.Segmenter`" would
mean matching whatever ICU the runtime ships, not Firefox. Compiling ICU4X into
our own artifact pins the engine to Firefox's family on every host.)

Measured cost of bundling ICU4X into the wasm build: **330 KB → 349 KB, +19 KB**
(wasm-opt'd). `SentenceSegmenter` only references the small rule-based
sentence-break tables; the heavy word/line dictionaries are dead-code-eliminated.
Against a ~150 MB model download, 19 KB buys one deterministic segmentation
engine across native, Node, Chrome, and Firefox.

## Work

1. **Enable ICU4X in wasm.** Add `icu-segmenter` to the `fxtranslate` features in
   `crates/fxtranslate-wasm/Cargo.toml`. `Engine::translate_long` then routes to
   `IcuSegmenter` automatically (`engine.rs:303`) — no wasm API change for the
   direct/pivot path; `lib/translate.js` keeps calling `translate_long`.
2. **Point `discovery::segment_sentences` at ICU4X** (`fxtranslate-wasm/src/lib.rs:328`),
   so any JS-side segmentation preview matches the translate path.
3. **Remove the icu-free workaround wiring.** Drop the "the wasm build is icu-free"
   branches/comments in `lib.rs` (e.g. the `translate_long` doc at `lib.rs:67`)
   and anywhere the code apologizes for wasm using `BasicSegmenter`.
4. **Keep `BasicSegmenter` as a compile-time opt-out, don't delete it.** It stays
   the `icu-segmenter`-off fallback in `Engine::translate_long` (`engine.rs:307`).
   The opt-out is the existing Cargo feature and nothing more — building with
   `icu-segmenter` off selects `BasicSegmenter` *and* drops the ICU dependency +
   its ~19 KB of data. A runtime selector is deliberately **not** added: it would
   link ICU in regardless, defeating the only reason to opt out (shaving the
   dependency bytes).

## Documentation

Update the docs wherever the two-engine story is currently explained, so the code
reflects "ICU4X everywhere, Basic is an opt-out":

- `crates/fxtranslate/src/segment.rs` module docs (`segment.rs:1`) — currently say
  the CLI enables ICU and the wasm build is icu-free.
- `crates/fxtranslate-wasm/src/lib.rs` (`lib.rs:9`, `lib.rs:67`, `lib.rs:320`) —
  remove "the wasm build is icu-free" framing.
- `architecture.md` — record the single-engine decision and the ICU4X ≡ Firefox
  rationale.
- npm `README.md` if the default-on ICU4X segmentation is worth calling out.

## Done when

- Native CLI and npm/wasm produce identical sentence boundaries on the same input
  (add a native≡wasm segmentation equality check across a corpus).
- `BasicSegmenter` is still reachable by building with `icu-segmenter` off, and
  that build links no ICU dependency.
- No remaining code or comments describe the wasm build as icu-free.
