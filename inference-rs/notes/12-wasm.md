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
| native scalar (`lean-embed`) | 565 | 30.9 | 1265 | 0.29× |
| native SIMD (`fast`/gemmology) | 1925 | 8.0 | 4048 | 1.00× |
| wasm scalar (Node) | 80 | 220.5 | 179 | 0.04× |
| wasm SIMD128 (Node) | 365 | 44.9 | 799 | 0.19× |
| wasm SIMD128 (Firefox, WebDriver) | _(step 9)_ | | | |

### Perf results (step 7 — filled)

All four rows above were measured on the **same machine, same model, same
corpus, single-threaded, shortlist-off** so they drop straight into the project's
`perf.py --blocks` format:

- **Model + corpus:** en→fr `base` model (`data/models/enfr/`,
  `model.enfr.intgemm.alphas.bin`) over `corpora/nllb-en-fr.blocks.txt` (**307
  blocks / 15,856 source words**) — the same corpus the step-1–5 wasm parity pass
  used, so the native and wasm rows are apples-to-apples with each other. (The
  README/notes-11 canonical single-thread baseline is en→ru `base` +
  *Frankenstein* at ~1267–1276 words/s; the numbers here are a different pair and
  corpus, so treat that as the external reference point, not a row in this table.)
- **Metric definitions (identical to `scripts/perf.py`'s `--blocks` mode):**
  **words/s** = source words ÷ sum of per-block compute time (encode + decode,
  model load excluded); **TTFT (ms)** = per-block encode + first-decode-step,
  median across blocks; **decode tok/s** = generated tokens ÷ total decode
  seconds. All reported as median + IQR over measured runs after a warmup.
- **Runs:** native rows and the wasm SIMD128 row are 5 measured runs after 1
  warmup; the wasm scalar row is 3 measured after 1 warmup (each scalar-wasm pass
  is ~3 min at 80 words/s, so fewer runs — its per-run spread is tiny, 80–81, so
  the median is solid). Every row was measured on a **quiet machine** (no other
  heavy job running), which matters: an earlier native-scalar run taken while the
  wasm jobs were still in flight read a contaminated ~1900 words/s; the clean
  quiet-machine value is 565.

**Runtime (record with every wasm number):** **Node v22.18.0**, **V8
12.4.254.21-node.27**. Kernel per wasm row (so a silent scalar build can't pass as
a SIMD number): the "wasm scalar" row reports `backend() == "scalar"`; the "wasm
SIMD128" row reports `backend() == "wasm-simd128"` (the live pure-Rust
`i32x4.dot_i16x8_s` kernel). Both were the `wasm-pack build --target nodejs
--no-default-features` lean build; the SIMD row additionally passed
`RUSTFLAGS="-C target-feature=+simd128"`.

**Reading the numbers.**

- **wasm SIMD128 is ~4.6× the wasm scalar kernel** (365 vs 80 words/s) — the
  SIMD128 kernel is doing its job; a scalar-only wasm build would be badly
  unrepresentative, which is exactly why the plan flags the kernel per row.
- **wasm SIMD128 lands at ~0.19× native SIMD** and **~0.65× native *scalar***
  (365 vs 565). The native scalar kernel is plain Rust, but LLVM autovectorizes
  its int8 dot-product loop to NEON on aarch64, so "native scalar" is not slow;
  the honest wasm-vs-native gap is against that autovectorized native scalar and
  the i8mm gemmology SIMD, and ~0.19× of the fast native path is the expected
  order for a 128-bit `dot_i16x8_s` kernel under a JIT vs. i8mm hardware.
- **TTFT** tracks throughput: 8.0 ms native SIMD → 44.9 ms wasm SIMD128 → 220.5 ms
  wasm scalar. These are host-timed via the phase hook, not native `Instant`
  spans (see below).

**Linear-memory high-water (a different metric from native RSS).** Max wasm
linear memory (`core::arch::wasm32::memory_size(0)` × 64 KiB, polled per block):

| build | linear-memory high-water |
|---|---|
| wasm scalar (Node) | 68.8 MiB |
| wasm SIMD128 (Node) | 93.2 MiB |

**Caveat, stated explicitly:** wasm linear memory is **not** the same metric as
native settled/peak RSS (native en→fr settles at ~150 MB, mostly the resident
weights). wasm has **no shared file-backed pages and no mmap** — the model lives
as one owned `Vec<u8>` in linear memory plus activation scratch, and linear memory
only ever grows (never returns pages to the OS), so its high-water is the model
buffer + peak activations, measured against a byte-length counter, not against a
resident-set-size counter. Compare the two with that caveat, not as identical.
(The two wasm rows differ because they are separate processes with independent
allocation histories; both are dominated by the ~31 MB on-disk model expanding to
its in-memory int8 form + scratch, well under the earlier ~150 MB worst-case worry
for a fully-resident f32 build — `lean-embed` keeps the embedding table int8.)

**How TTFT / decode-tok/s are measured on wasm (no `Instant`).** `std::time::Instant`
panics on `wasm32-unknown-unknown`, so the native `--timing` spans
(`engine.rs` `translate_batch_timed`, `Instant` markers) don't run in wasm. The
harness instead:

1. times the whole per-block `translate` call with `performance.now()` on the
   host (words/s is fully host-measurable this way); and
2. for the encode-vs-decode split and first-token latency, uses a **phase-boundary
   hook**: a new native-safe `Engine::translate_batch_phased(texts, on_phase)`
   (core engine) that invokes a closure at `EncodeStart` / `DecodeStart` /
   `FirstToken` / `DecodeEnd` — the exact boundaries the native `Instant` spans
   measure. The wasm binding
   `Translator.translateBlockPhased(block, onPhase)` forwards those to a JS
   callback, and `js/perf.js` records `performance.now()` on each, deriving TTFT
   and decode-tok/s identically to how `perf.py` derives them from the native
   spans. **Native is untouched:** `translate_batch_timed` and the `--timing` path
   still use `Instant`; the phased method is additive and only the wasm crate
   calls it.

Reproduce:

```
# native rows (writes [block] spans that carry encode_ms/decode_ms/ttft_ms/tokens)
cargo build --release -p fxtranslate-oracle --features fast              # SIMD
cargo build --release -p fxtranslate-oracle --no-default-features --features lean-embed  # scalar
target/release/fxtranslate-oracle translate data/models/enfr/model.enfr.intgemm.alphas.bin \
    data/models/enfr/vocab.enfr.spm data/models/enfr/vocab.enfr.spm \
    --blocks corpora/nllb-en-fr.blocks.txt --timing

# wasm rows (Node)
cd crates/fxtranslate-wasm
wasm-pack build --target nodejs --no-default-features                                    # scalar
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target nodejs --no-default-features  # SIMD128
node js/perf.js --runs 5 --warmup 1        # words/s + TTFT + decode tok/s + linear-memory high-water
```

### Perf results (step 7)

Measured on macOS arm64 (Apple Silicon), en→fr `data/models/enfr/`, block corpus
`corpora/nllb-en-fr.blocks.txt`, single-threaded, shortlist **off** (the
production baseline). All four measured rows use the *same* metric definitions the
native `scripts/perf.py --blocks` path emits: **words/s** = source words ÷ Σ
per-block compute time (encode+decode, model load excluded); **TTFT (ms)** =
median per-block time-to-first-token (encode + first decode step); **decode tok/s**
= generated tokens ÷ Σ decode time. Each number is the median over 4 measured runs
after 1 warmup run; IQRs were tight (native ≤ 3%, wasm < 1%).

The wasm rows are host-timed (`performance.now()`) via `crates/fxtranslate-wasm/js/
perf.js` — wasm has no usable `std::time::Instant` — through the engine phase hook
`Engine::translate_batch_phased` (mirrors `translate_batch_timed`; the host records
the clock at each `Phase` boundary). The native rows are the oracle's `[block]`
spans aggregated with the identical formulas, so they line up column-for-column.

| build | words/s | TTFT (ms) | decode tok/s | rel. to native SIMD |
|---|---|---|---|---|
| native scalar (`lean-embed`) | 563 | 31.1 | 1261 | 0.29× |
| native SIMD (`fast`/gemmology) | 1934 | 8.0 | 4072 | 1.00× |
| wasm scalar (Node) | 80 | 230.4 | 177 | 0.041× |
| wasm SIMD128 (Node) | 346 | 47.2 | 755 | 0.18× |
| wasm SIMD128 (Firefox, WebDriver) | *(step 9)* | | | |

Notes on the numbers:

- **Runtime** for both wasm rows: **Node v22.18.0 / V8 12.4.254.21-node.27**. Kernel
  is self-reported per row via `Translator.backend()` — the scalar build reports
  `"scalar"`, the SIMD build reports `"wasm-simd128"` — so a silent scalar build
  can't be mistaken for a SIMD measurement.
- **wasm-scalar sample is bounded.** Scalar wasm runs at ~80 words/s, so the full
  307-block / 15 856-word corpus is ~3.3 min *per pass* — too slow for a
  warmup + 4 runs. The scalar row is therefore measured on a **bounded 40-block /
  2 036-word head sample** (`blocks[:40]`), ~25 s per pass. words/s, TTFT, and
  decode tok/s are rates, so the smaller sample is directly comparable; the sample
  was byte-stable across runs (IQR collapsed to a point). All other rows use the
  full corpus. The wasm-SIMD128 row *does* run the full corpus (~46 s/pass).
- **SIMD128 speedup:** the pure-Rust `i32x4.dot_i16x8_s` kernel is ~4.3× the scalar
  wasm build (346 vs 80 words/s), and lands at 0.18× native SIMD — the expected gap
  between a 128-bit (16-int8-lane) wasm kernel and native ARM i8mm/AVX2 plus
  wasm-runtime overhead. Native scalar-vs-SIMD is 3.4× (563 → 1934), consistent
  with the matmul being ~2/3 of the work (`notes/11`).

**Linear-memory high-water** (max `WebAssembly.Memory.buffer.byteLength`, captured
by `perf.js` after each block): **68.8 MiB** for wasm scalar, **93.2 MiB** for wasm
SIMD128. Both are dominated by the ~31 MB owned model buffer plus per-block
activation scratch (the SIMD path's larger high-water reflects the full-corpus run
touching bigger blocks, not a kernel cost). **Caveat:** wasm linear memory and
native settled/peak RSS are *different* metrics — wasm has no `mmap`, no shared
file-backed pages, and the model lives as an owned heap `Vec<u8>` — so compare with
that caveat, not as if the two were the same measurement.

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
system libm and wasm portable libm (see "Validation results").

Build order step 7 done: host-timed perf numbers for all four native/wasm
scalar/SIMD rows are in "Perf results (step 7)" above, measured through the
`Engine::translate_batch_phased` host-clock hook and `js/perf.js`, plus the
wasm linear-memory high-water. Next: step 8 (`.wasm` size vs native cdylib) and
step 9 (the Firefox WebDriver row).
