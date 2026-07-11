# Changelog

Notable changes to the `fxtranslate` engine and the `fxtranslate-cli` binary,
which are versioned and published together. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
