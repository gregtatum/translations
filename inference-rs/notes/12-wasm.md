# wasm build: run the engine in Node (and a browser) for validation + npm

## Goal

Compile the `fxtranslate` engine to WebAssembly and drive it from a **Node-based
tool** that loads a model and translates a sentence — proving the whole pipeline
(tokenize → encode → SSRU decode → project → detokenize) runs correctly in a wasm
runtime. Then cross-check it in a **real browser via a Firefox WebDriver** harness,
and package the artifact for **npm**.

The wasm build is a **validation + distribution artifact, not a production target.**
The Firefox Translations replacement ships the Rust code *natively*; wasm is never
integrated into Firefox. Its jobs are (1) correctness + performance numbers from
Node and a browser, and (2) npm reach as a general-purpose package. So we target a
**general `wasm32` build via `wasm-bindgen`**, not Firefox-internal Emscripten /
MozIntGEMM linkage.

## Why this is mostly already done

The crate was designed feature-first with wasm in mind. Every platform-specific
piece is optional and off by default (`crates/fxtranslate/Cargo.toml`):

- `mmap`, `download`/`net` (HTTP + TLS), `threads`/`gemm-threads`, `icu-segmenter`
  are all opt-in — a lean build drops file I/O, sockets, threads, and the C++
  toolchain.
- `build.rs` already **falls back to the portable scalar kernel without failing**
  on any arch it has no kernel for (wasm included) — it only hard-errors when
  `FXTRANSLATE_REQUIRE_SIMD` is set. So a lean wasm build compiles today.
- Byte-oriented constructors already exist where wasm needs them:
  `Model::from_bytes` (`model.rs:241`), `SpmVocab::from_bytes` (`spm.rs:75`),
  `Shortlist::from_bytes` (`shortlist.rs:40`). Model bytes come from JS, not a path.

The one thing that does **not** exist for wasm is a fast int8 kernel: `gemmology`
is a C++ shim compiled via `cc` for aarch64/x86_64 only. On wasm we get the scalar
Rust kernel — correct, but slow. See §SIMD.

## Scope decisions (narrow, on purpose)

- **Target `wasm32-unknown-unknown` + `wasm-bindgen`.** npm-friendly, standalone, no
  Firefox coupling. Not wasi, not Emscripten.
- **Node driver first.** A small Node tool that loads model bytes from disk, hands
  them to the wasm module, and prints the translation. This is the "prove it works"
  milestone. Browser/WebDriver comes after.
- **Scalar first for bring-up, then a wasm SIMD128 kernel — both in scope.** Prove
  correctness on the portable scalar path first (baseline number), then add the
  pure-Rust wasm SIMD128 int8 kernel (§SIMD) in the same effort. Scalar gates the
  first *run*; SIMD gates representative *numbers*.
- **Single-threaded.** `threads`/`gemm-threads` stay off (they already are on wasm).
- **Model bytes supplied by the host.** No `download`/`net` — the Node/JS side reads
  or fetches the `.bin` / `.spm` / `lex` files and passes `Uint8Array`s in. Reuse the
  existing `Fetch` trait story only if we later want in-wasm downloading (out of scope
  now).
- **One language pair to start** (en→fr, the trace/parity workhorse). Pivot and
  multi-pair come free from the engine but aren't the validation target.

## Feature set for the wasm build

```
cargo build --target wasm32-unknown-unknown -p fxtranslate \
    --no-default-features --features lean-embed
```

`lean-embed` (pure Rust) keeps the load/retained-memory win without pulling `mmap`.
No `fast`, no `gemmology`, no `net`. This is the config the plan builds the
wasm-bindgen layer on top of.

## Component design

### 1. A wasm-bindgen binding layer

The engine's public surface is path-based (`Engine::load` takes `&Path`,
`engine.rs:115`); wasm needs a bytes-based entry. Two thin additions:

1. `Engine::from_bytes(model: &[u8], src_vocab: &[u8], trg_vocab: &[u8])` (and an
   optional `with_shortlist_bytes`) — wraps the existing `*::from_bytes`
   constructors + `Engine::new`. `Weights` currently only has `load(path)`
   (`weights.rs:259`), which wraps `Model::load`; add a `Weights::from_bytes`
   wrapping `Model::from_bytes`. Small, pure plumbing; no math touched.
2. A new crate `crates/fxtranslate-wasm` (kept out of the published core, like the
   oracle crate) with `#[wasm_bindgen]` wrappers: a `Translator` object holding an
   `Engine`, constructed from `Uint8Array`s, exposing `translate(text) -> String`
   and `translate_long(text) -> String`. `wasm-bindgen` handles the string/bytes
   marshaling. `console_error_panic_hook` for legible panics.

Keeping the bindings in their own crate means the core `fxtranslate` gains no
`wasm-bindgen` dependency and stays a clean library.

### 2. The Node driver

A `wasm/` (or `crates/fxtranslate-wasm/js/`) Node tool that:
- loads the wasm module (built with `wasm-pack build --target nodejs`, or raw
  `wasm-bindgen` + a hand-written loader),
- reads model files from `data/models/enfr/` as buffers,
- constructs the `Translator`, translates a sentence, prints it.

This is the milestone: `node run.js "Hello world."` → French, from wasm.

### 3. Browser + Firefox WebDriver harness (after Node works)

- `wasm-pack build --target web`, a minimal HTML page that fetches the model and
  translates.
- A WebDriver script (geckodriver) that loads the page, runs a fixed sentence set,
  and reports outputs + timing. Reuses the same corpus the native `task rs:parity`
  uses so numbers are comparable.

### 4. npm packaging

`wasm-pack` emits a package with the `.wasm`, JS glue, and `.d.ts`. Decide bundling
of model files: **exclude them** (the package ships the engine, models are fetched/
provided by the consumer) to keep the package small and match the library-first
stance.

## Validation strategy

The point of the whole exercise is parity, so lean on the existing oracle:

1. **Correctness vs native.** Same model + same sentences through the native engine
   (`task rs:translate`) and the wasm build; assert identical (or within the
   established rtol/atol tie tolerance) output text. The scalar kernel is already
   bit-identical to native scalar (`tests/gemm_parity.rs`), so wasm-scalar should
   match native-scalar exactly; any diff is a wasm-specific bug (float/rounding,
   marshaling), which is exactly what we want to surface.
2. **Corpus parity in Node + browser.** Run the `task rs:parity` corpus through both
   Node-wasm and Firefox-wasm; compare exact-match rate to the native baseline.
3. **Numbers.** TTFT + tok/s from Node and from Firefox (WebDriver), scalar first;
   re-measure after the SIMD phase. Note the runtime (Node version, Firefox version)
   with each number.

Note in output which kernel ran (scalar vs simd) so a silent scalar build doesn't
get mistaken for a representative speed measurement.

### Validation results (steps 1–6 done)

The full en→fr corpus (`corpora/nllb-en-fr.blocks.txt`, 1306 lines, shortlist on)
was run through the native scalar oracle (`--no-default-features --features
lean-embed`) and through the wasm build under **Node v22.18.0**. Both sides are
internally deterministic (two runs, byte-identical each). The wasm SIMD128 build
and the plain wasm scalar build produce **bit-for-bit identical output over the
whole corpus** — both are exact-family int8 backends — so the faster SIMD build
stands in for "wasm output" everywhere below.

**Exact-match rate: 1302 / 1306 = 99.69% (native-scalar vs wasm).** The four
diverging lines are all long sentences where a near-tie greedy argmax flips one
token:

| line | nature of the divergence |
|---|---|
| 9   | single word swap (`rencontres` → `réunions`) |
| 27  | two-token region (apostrophe glyph + `une` → `la`), resolves back |
| 251 | one word swap (`en fait` → `réellement`), one-token length cascade |
| 531 | word-order/insertion (`gratuitement` placement), one-token cascade |

No blank lines diverge; no short sentence diverges; there is no runaway cascade.
These are exactly the "whole-word argmax flips on long sentences" the plan
anticipated.

**Proven root cause: f32 transcendental last-bit differences, ≤ 1 ULP.** A
throwaway spike routed *every* transcendental the engine uses — `exp` (softmax +
highway sigmoid, `ops.rs`), `sqrt` (layernorm, `ops.rs`), and `sin` + `powf`
(positional encoding, `engine.rs:191`/`:1121`) — through the pure-Rust `libm`
crate on **both** native and wasm, rebuilt both, and re-ran the full corpus:
**divergences collapsed from 4 to 0 — 100.00% exact match.** Same library on both
sides ⇒ identical output, which is the proof. A standalone ULP sweep quantifies
the gap between macOS arm64's *system* libm (what native `f32::` calls) and the
*portable* libm (what `wasm32-unknown-unknown` std uses, since wasm has no system
libm): **exp, sin, and powf each differ by at most 1 ULP; sqrt is 0 ULP**
(hardware IEEE sqrt on both). On wasm, `f32::` and the `libm` crate are identical
(both are the portable libm); on native they differ by that 1 ULP. Pure f32
add/mul/dot and the int8 GEMM accumulation are **bit-identical** across the two
targets (verified on transcendental-free inputs); the int8 GEMM is integer-exact
by construction.

Note the subtlety that made this worth proving rather than asserting: swapping
only the `ops.rs` transcendentals (`exp`/`sqrt`) does **not** fix it — the
dominant contributor is the positional-encoding `sin`/`powf`, whose 1-ULP wobble
is baked into every token's input embedding and propagates through the network.
The collapse to zero only happens once *all four* are shared.

**Decision on record:** accept these small, fully-explained divergences. Do
**not** force a shared libm in production — it would touch the hot math path
(softmax/layernorm/PE) for a cosmetic bit-parity gain we don't need, since wasm is
a validation + npm artifact and production ships native Rust. The int8 GEMM stays
held to the strict bit-identical bar (`gemm_parity.rs`); it already clears it.
Forcing full wasm↔native determinism *is* possible if ever required — the libm
swap demonstrates it reaches 100% — but it's off the table by default.

## SIMD int8 kernel for wasm (in scope)

Scalar int8 runs one multiply-add per instruction, and the matmul is ~2/3 of all
work (`notes/11`), so a scalar-only wasm build is far slower than the native SIMD
path. Representative numbers require a **wasm SIMD128 int8 kernel**, and it is **in
scope for this effort** — brought up right after the scalar pipeline is proven end
to end in Node (see build order), not deferred to some future.

### It's "accelerated kernel N+1", not a new operator

The `int8shiftAlphaAll` operator already lives in scalar Rust: `ops::intgemm_affine`
(`ops.rs:532`), with `prepare_a`'s +127 shift (`ops.rs:456`) and `prepare_bias`'s
shift-correction term (`ops.rs:497`). gemmology (C++, ARM i8mm / x86 AVX2) is a SIMD
*twin* of that operator, reached through `PreparedB` (`gemm.rs`) — it supplies speed,
not the algorithm. The wasm kernel is the same pattern: a **third `PreparedB`
implementation** computing the identical math with wasm intrinsics. `engine.rs` /
`weights.rs` call `PreparedB::{new, matmul, read_row}` and don't change.

### Implementation

- **Pure Rust via `core::arch::wasm32`** — no emscripten, no C++. `#[target_feature(
  enable = "simd128")]` on the kernel; build with `-C target-feature=+simd128` (Node
  ≥ 16 and current Firefox enable simd128 by default). The register is 128-bit = **16
  int8 lanes**, so `PreparedB::new` returns `None` unless `k % 16 == 0` — the same
  register-width gate the NEON/SSE path already has (`gemm.rs:84`); the transformer's
  `k = 384/1536` qualify.
- **Accumulate exactly into i32.** `A` is the shifted `u8` activation, `B` is `i8`.
  Widen each 16-byte block to two `i16x8` vectors (u8 zero-extend, i8 sign-extend) and
  accumulate with **`i32x4.dot_i16x8_s`**, whose products are formed at i32 precision
  — *no int16 intermediate, no saturation*. Then the scalar tail: `unquant · acc +
  prepared_bias`.
- **We own the packed layout.** Unlike the FFI path (gemmology's opaque tiled pack),
  the Rust kernel chooses `B`'s block layout, so `read_row` — used by `lean-embed` to
  serve embedding rows out of the packed weights (`gemm.rs:148`) — is an inverse we
  control, not a reimplementation of someone else's tiling.
- **Dispatch / gating.** Add a third arm to the `gemm` module: `gemmology_simd` (FFI)
  → wasm arm (`cfg(all(target_arch = "wasm32", target_feature = "simd128"))`) →
  scalar stub. `backend()` reports `"wasm-simd128"`. The module compiles only under
  the `gemmology` feature today (`lib.rs:51`); widen that to also compile on
  wasm+simd128 so the kernel is available *without* pulling the C++-toolchain feature.
  `portable` still forces the scalar path.

### Parity: this is an *exact* backend (better than AVX2)

int8 kernels split by accumulator width (`gemm_parity.rs:167`, `gemm-backends.md`):
**exact** — accumulate straight into i32 (scalar, ARM `usdot`, x86 VNNI) — vs
**saturating** — accumulate through an int16 `maddubs` lane that saturates at ±32767
(x86 AVX2/SSE). Because the design above accumulates via `i32x4.dot_i16x8_s` at full
i32 precision, **the wasm kernel is in the exact family**: it matches the scalar
oracle bit-close on *any* input, not just the model's benign weight range. It slots
straight into `gemm_parity.rs` — `check()` already asserts exact backends match, and
adding `"wasm-simd128"` with `backend_saturates(...) == false` gives it the strict
full-range assertion in `full_range_matches_only_on_exact_backends` (a *stronger*
guarantee than AVX2, which is only reported there).

**Correction to an earlier assumption:** wasm does *not* force the saturating path.
Firefox's shipping WASM engine saturates only because intgemm's SSE `maddubs` code was
lowered to wasm via simd-everywhere; a fresh native-intrinsics kernel using
`dot_i16x8_s` is exact. If we ever specifically wanted to *reproduce* Firefox-WASM
numerics we would deliberately emulate `maddubs` instead — but the project's parity
bar is against the scalar oracle, which the exact kernel matches best, so exact is the
default. Bit-matching Firefox's WASM engine is out of scope (we don't ship into
Firefox).

### Validating the wasm kernel in CI

Run `gemm_parity.rs` under a **wasm runtime**, not just native. With
`wasm-bindgen-test` + a Node (or wasmtime) runner and `FXTRANSLATE_REQUIRE_SIMD` set,
a green wasm run proves the SIMD128 kernel is actually live (`backend() != "scalar"`)
and bit-close to the scalar reference — the same cheat-proof gate the native arches
use (`gemm_parity.rs:104`, `simd_backend_is_live_when_required`), extended to wasm.

### Explicitly not now

- **relaxed-simd** (`i32x4.relaxed_dot_i8x16_i7x16_add_s`, the closest wasm has to
  VNNI/i8mm fused dot-accumulate) needs a **7-bit** operand and the newer relaxed-simd
  feature; our weights span full int8 `[-127, 127]`, so it doesn't apply without
  rescaling. A possible future speedup once support is universal.
- **wasm threads** (SharedArrayBuffer): stays out. Single-threaded, matching the
  native default and the "wasm stays single-threaded" constraint in `notes/11`.

## Perf + binary size (comparable to the existing numbers)

Both new metrics must line up with what the project already reports, so wasm numbers
drop straight into the existing tables (`notes/11`, the crate README perf section)
rather than being a one-off.

### Perf — same metrics, same corpus, same model

Reuse the exact metric definitions `scripts/perf.py` emits so the columns are
comparable: **words/s** over a corpus (the fair cross-engine metric — source words ÷
per-block compute time), plus **TTFT (ms)** and **decode tok/s**, median + IQR over
`--runs` after `--warmup`. Same model + same corpus as the native baselines
(`corpora/frankenstein-en.blocks.txt` / `nllb-en-fr.blocks.txt`), single-threaded,
shortlist-off — the README single-thread baseline (~1267 words/s en→ru base) and the
`notes/11` tables are the reference points.

Target table (one row per build, all on the same machine + corpus):

| build | words/s | TTFT (ms) | decode tok/s | rel. to native SIMD |
|---|---|---|---|---|
| native scalar (`portable`) | | | | |
| native SIMD (`fast`/gemmology) | | | | 1.00× |
| wasm scalar (Node) | | | | |
| wasm SIMD128 (Node) | | | | |
| wasm SIMD128 (Firefox, WebDriver) | | | | |

wasm-specific measurement notes:

- **Time from the host, not `std::time::Instant`** — `Instant` is unavailable on
  `wasm32-unknown-unknown` (the `--timing` spans in `engine.rs:277`/`784` rely on it,
  so they don't work in wasm). Measure wall-clock in the Node driver / WebDriver
  script with `performance.now()` around the translate call. words/s is fully
  host-measurable; for TTFT / decode-tok/s comparable to the native `--timing` spans,
  expose a thin phase hook the host times (encode vs decode-loop) rather than porting
  `Instant`.
- **Record the runtime** with each number (Node version + V8, Firefox version) — a
  wasm number without its runtime isn't comparable.
- **The wasm build enables a *faithful* one-off marian comparison** that
  `translator-cli` can't give (perf.py already flags translator-cli's batch number as
  only an upper-bound reference); note that where relevant.

### Memory

Native peak RSS is already measured (`/usr/bin/time -l`; `scripts/final_comparison.py`
reads RSS via `ps`). For wasm report the **linear-memory high-water** (max
`WebAssembly.Memory.buffer.byteLength`, or track `memory.grow`) — mostly the ~150 MB
owned model buffer plus activations. Flag that wasm linear memory and native settled
RSS are *different* metrics (no shared file-backed pages, no `mmap`), so compare with
that caveat, not as if identical.

### Binary size — native vs wasm

The native side already has size characterization via `scripts/release_build.py`
(`task rs:release`); extend it, don't replace it.

- **Native:** the lean engine as a stripped `cdylib` (and/or the release binary), with
  a `cargo bloat` top-contributors breakdown — the existing tool.
- **wasm:** the `.wasm` module size, **before and after `wasm-opt -Oz`**, reported
  **raw + gzip + brotli** (npm/CDN and Firefox both deliver compressed, so compressed
  size is the real number). Use **`twiggy top`** for the code-size breakdown — the
  wasm analog of `cargo bloat`.
- **Apples-to-apples:** compare the *same lean feature set* (`--no-default-features
  --features lean-embed`, no `net`/`mmap`/`icu`) on both, and separate **code** size
  from the **model** (which the artifact excludes) so the comparison is engine-code vs
  engine-code. Also report the **scalar vs SIMD128 wasm delta** (the SIMD kernel adds a
  little code) and the `console_error_panic_hook`/wasm-bindgen glue overhead.

Target: a small table of native-cdylib vs wasm (raw / gz / br) at the same feature
set, so "how big is the engine as wasm vs native" has one clear answer.

## Build order

1. `Weights::from_bytes` + `Engine::from_bytes` (+ shortlist-bytes). Unit-test on
   native that the byte path yields an engine identical to the path-based `load`.
2. Confirm the lean crate compiles to `wasm32-unknown-unknown`
   (`--no-default-features --features lean-embed`); fix any std-only spot the lean
   feature set doesn't already gate. Scalar kernel at this point.
3. `crates/fxtranslate-wasm` bindgen layer (`Translator`, panic hook).
4. `wasm-pack build --target nodejs` + the Node driver; translate one sentence
   (scalar).
5. Correctness vs native on the traced sentence, then the parity corpus (Node). This
   is the "it runs in wasm, correctly" milestone.
6. **wasm SIMD128 kernel** (§SIMD): the third `PreparedB` arm in the `gemm` module,
   `i32x4.dot_i16x8_s` exact accumulation, `-C target-feature=+simd128`; extend
   `gemm_parity.rs` (`backend_saturates("wasm-simd128") == false`) and run it under a
   wasm runtime in CI with `FXTRANSLATE_REQUIRE_SIMD`.
7. Perf (§Perf): host-timed words/s (+ TTFT/tok/s via a phase hook) in Node, scalar
   vs SIMD128, into the existing perf-table format; and linear-memory high-water.
8. Binary size (§Binary size): extend `release_build.py` with the `.wasm` size
   (raw/gz/br, `wasm-opt -Oz`, `twiggy top`) vs the native lean `cdylib` at the same
   feature set.
9. `--target web` + Firefox WebDriver harness; corpus parity + numbers in-browser
   (SIMD128 build), runtime version recorded.
10. npm packaging (models excluded); dry-run publish.

Steps 1–5 are the narrow "prove it runs in Node" milestone; 6 makes it fast and pins
the SIMD kernel to the oracle; 7–8 produce the comparable perf + size numbers; 9–10
add the browser and distribution.

## Open questions / risks

- **Tooling choice:** `wasm-pack` vs raw `wasm-bindgen` + `wasm-opt`. `wasm-pack`
  gives npm packaging and multi-target glue for free; lean toward it unless it
  fights the workspace layout.
- **Model size in wasm memory.** ~150 MB of weights live as an owned buffer (no
  `mmap`); `lean-embed` drops the resident f32 embedding tables. Confirm the wasm
  linear-memory / `maximum` pages budget and how Node/Firefox handle a buffer that
  large. Likely the biggest practical risk.
- **Scalar throughput** may be slow enough that browser numbers are painful before
  SIMD128 lands (step 6) — acceptable for the step-5 correctness pass; call it out.
- **No `std::time::Instant` on wasm** — the engine's `--timing` spans won't work in
  wasm; perf must be host-timed (`performance.now()`), and TTFT/tok/s need a thin
  phase hook rather than the native span path. Don't report a wasm number without its
  runtime version (§Perf).
- **`from_bytes` lifetimes / ownership** — the engine must own the model bytes (no
  borrowing a JS buffer across calls). `Model::Owned(Vec<u8>)` already supports this;
  make the bindgen `Translator` take ownership of the passed arrays.
- **String marshaling** for non-Latin scripts across the wasm boundary (UTF-8/UTF-16)
  — validate with a CJK pair once the Latin path works.

## Status

Build order steps 1–6 done: byte constructors (`Weights`/`Engine::from_bytes`),
lean `wasm32-unknown-unknown` build, the `fxtranslate-wasm` bindgen crate, the Node
driver (`js/run.js` + `js/batch.js`), and the live wasm SIMD128 int8 kernel
(`backend() == "wasm-simd128"`, exact family). Validated: full-corpus
native-scalar vs wasm parity at 99.69% exact match, with the four divergences
proven to be ≤ 1-ULP transcendental (exp/sin/powf) differences between arm64
system libm and wasm portable libm (see "Validation results"). Next: step 7
(host-timed perf numbers) and step 8 (`.wasm` size vs native cdylib).
