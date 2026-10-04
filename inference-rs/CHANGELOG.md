# Changelog

Notable changes to the `fxtranslate` engine and the `fxtranslate-cli` binary,
which are versioned and published together. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Decoder first-step seeding.** The decoder embedded its seed token (`eos`) at
  position 0, but marian zero-pads there: it builds the decoder input by shifting
  the target embeddings right and zero-padding the vacated first slot
  (`shift(embeddings, {0, 1, 0})`), so the first step sees only the positional
  encoding. Embedding `Wemb[eos]` injected a vector of L2 norm 30.5 against
  `PE(0)`'s 16.0 — nearly 2× the only signal marking a sentence start — so the
  model picked the right word in the wrong case, and word order drifted from
  there. Greedy exact-match against `translator-cli` (shortlist off,
  `corpora/dev-en.txt`):

  | pair | before | after |
  |---|---|---|
  | en-ru int8 | 0/20 | 18/20 |
  | en-ru float32 | 0/20 | **20/20** |
  | en-fr int8 | 15/20 | 19/20 |
  | en-es int8 | 6/20 | 19/20 |

  **This changes output for every model and pair.** The remaining int8 mismatches
  are legitimate quantization near-ties (they vanish at float32). Pinned by
  `tests/decoder_seed.rs`. The trace replay could not catch it — the step-0
  embedding and PE are `const` leaves, so it passes them through — and
  `onnx/numpy_ref.py` shared the old convention. All three ports are fixed and
  re-validated: `onnx/` (`decode_step.onnx` gained an `embed_gate` input, since
  its embedding lookup is inside the graph), `ggml/marian_ggml.cpp` (G1), and
  llama.cpp's `LLM_ARCH_MARIAN` decoder graph (G2) — the last in the external
  `~/src/llama.cpp` checkout, left uncommitted there. reference, fxtranslate
  (int8 + float32), numpy_ref, ONNX (float + int8), and both ggml ports now
  produce identical output. See `notes/23-float-model-support.md`.

### Added

- Float32 (non-quantized) model support. `marian-conv --gemm-type float32`
  containers now load and run alongside the shipped `*.intgemm.alphas.bin`
  int8 models, with no API change and no flag — `Weights` detects the precision
  at load and reports it via `Weights::precision()`. This makes an
  apples-to-apples int8-vs-float comparison possible within one engine, which is
  what it was built for; it is a correctness/reference facility, not a fast path.
  - The float container stores an affine weight `[K, N]` where the int8 one
    stores it `[N, K]` (marian's float save branch neither packs nor
    transposes), and carries no `*_QuantMultA` tensors. `Wemb` is the exception:
    `[vocab, dim]` in both. `ops::affine_f32` / `ops::project_f32_into` take
    their weight in the float orientation explicitly so the two cannot be
    confused, and `tests/float_model.rs` pins it — a transposed read scores
    cosine -0.03 against the correct one, versus 0.9998 for int8-vs-float on the
    same checkpoint.
  - A float model always uses resident f32 embedding tables: under `lean-embed`
    there is nothing to save (the tables are the model's own data) and no quant
    multiplier for the int8 output projection, so the embedding representation is
    now chosen at load from the dtype rather than purely by feature.
  - Verified against `onnx/numpy_ref.py`, an independent float implementation of
    the same checkpoint: encoder max abs diff 1.9e-06 and first-step logits
    4.2e-05, with an identical top-5 — versus 0.168 / 4.74 for the int8 path.
  - `--float32` on `task rs:translate`, `rs:translate-reference`, and `rs:parity`
    runs the `<pair>-f32` conversion, so both engines can be compared at either
    precision through the same interface.

- Optional CPU threading, both off by default (the default build, wasm, and
  reproducible builds stay single-threaded and deterministic):
  - `threads` — data-parallel batch translation. `Engine::greedy_batch` /
    `translate_batch` spread a batch's independent sentences across worker threads
    that share one read-only copy of the weights (each keeps its own thread-local
    scratch); `Engine::with_threads(n)` sets the worker count. Pure Rust
    (`std::thread::scope`), no new dependencies. Measured ~6.6× throughput on an
    18-core machine, with peak memory growing only by per-worker activation scratch
    (~19 MiB/worker) rather than a full model copy per thread.
  - `gemm-threads` — intra-op parallelism inside one int8 matmul (the full-vocab
    output projection and the encoder layers), via a persistent thread pool in the
    gemmology shim sized by `FXT_GEMM_THREADS`. Speeds up single-sentence latency
    (~1.68× decode) at essentially no extra memory; implies `gemmology`.
  - Both are bit-identical to the single-thread path (they partition independent
    sentences / disjoint output columns), so the oracle-parity and batch-invariance
    suites still hold. See `notes/11-threading-opportunities.md`.

## [0.4.0] - 2026-07-09

### Fixed

- Translation is now resilient offline: when Remote Settings can't be reached,
  `translate` falls back to an already-cached model instead of failing at the records
  fetch. A direct cached pair, or a pivot whose legs are both cached, still works with no
  network; only a genuinely absent pair errors (now naming both the missing pair and the
  discovery failure). Backed by `Cache::cached_model`, which rebuilds the file set from
  the cache directory without any records.

## [0.3.0] - 2026-07-09

### Added

- `fxtranslate models`, a subcommand for managing the local model cache, so models
  are no longer only fetched as a side effect of translating:
  - `models list` — the cache location, every cached pair with its size, and the total.
  - `models add <src> <trg>` — pre-download a pair (both legs of a pivot) without translating.
  - `models rm <pair> | --all` — delete a pair, or the whole cache, reporting the space reclaimed.
  - `models info <pair>` — a pair's files, their sizes, and their on-disk paths.
- Library APIs backing the above: `Cache::{list_cached, pair_files, remove_pair}` and a
  `dir_size` helper (pure filesystem, temp-file aware), plus a pivot-aware, engine-free
  `loader::ensure_route_files` that resolves a route and caches every file it needs.

## [0.2.0] - 2026-07-09

### Added

- Pivot translation: any reachable pair now translates, routing through a hub language
  (English) and running two legs when no direct model exists — Marian has no pivot logic,
  so fxtranslate orchestrates it around the engine.
- `list` presents languages rather than raw models — fully-supported languages (usable to
  and from any other) separately from single-direction, one-way models — with `--all` for
  the raw `src → trg` pairs. Every shipped language renders a display name instead of a code.
- Long or multi-sentence input is split and translated per sentence — a built-in
  punctuation splitter plus an optional ICU (UAX #29) backend for CJK — instead of being
  silently truncated at the model's context window; pivots segment on each leg.

## [0.1.0] - 2026-07-05

Initial release: a pure-Rust inference engine for Firefox Translations models
(`fxtranslate`) and a batteries-included CLI (`fxtranslate-cli`). Discovers, downloads,
and verifies models from Remote Settings into a local cache (timeouts, retry with backoff,
HTTP Range resume, streaming progress), runs a portable scalar engine with an opportunistic
SIMD (gemmology) fast path and optional memory-mapping, and translates from arguments,
piped stdin, or an interactive prompt.

[0.3.0]: https://github.com/gregtatum/translations/compare/fxtranslate-v0.2.0...fxtranslate-v0.3.0
[0.2.0]: https://github.com/gregtatum/translations/compare/fxtranslate-v0.1.0...fxtranslate-v0.2.0
[0.1.0]: https://github.com/gregtatum/translations/releases/tag/fxtranslate-v0.1.0
