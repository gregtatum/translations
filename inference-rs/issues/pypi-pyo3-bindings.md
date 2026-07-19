# PyPI packaging via PyO3 + maturin: the `fxtranslate-py` crate

**Open, foundational. Blocks [pypi-publish-integration.md](pypi-publish-integration.md);
that in turn precedes [pypi-wheel-matrix-ci.md](pypi-wheel-matrix-ci.md) and
[python-cli-conformance.md](python-cli-conformance.md).** This is the Python analogue of
the npm work ([closed/26-npm-publish-checklist.md](closed/26-npm-publish-checklist.md)):
a new native-extension crate that binds the `fxtranslate` engine to Python and is packaged
as a wheel for PyPI, on the same lockstep version as the crates and the npm package. Where
npm compiles the engine to wasm (a validation/distribution artifact, host supplies bytes),
the Python binding is **native compiled Rust** — so it turns on the batteries the wasm build
cannot: the gemmology SIMD kernel, mmap, and built-in networked model management.

## Background: why native, not a wasm reuse

The npm package embeds the engine as wasm because the browser is a real target for it. Python
has no such constraint: a `pip install` runs native code, so the Python binding should be the
*fast, batteries-included* face of the engine. This is the same reasoning that motivated the
Rust CLI package ([closed/14-rust-only-package.md](closed/14-rust-only-package.md)): discover
→ download → cache → translate, all in one artifact, no reimplementation of a fetch/cache shell
in the host language. PyO3 + maturin is the standard, well-supported path to ship compiled Rust
to PyPI, and it lets us depend on the core crate directly (a normal `path` dependency), the way
`fxtranslate-cli` and `fxtranslate-wasm` already do.

The name `fxtranslate` is free on PyPI (404 as of 2026-07-18) and is the distribution name, so
the ecosystem reads identically across crates.io, npm, and PyPI.

## Design

### The crate

New workspace member `crates/fxtranslate-py`, added to `[workspace].members` in the root
`Cargo.toml`. Structure mirrors `crates/fxtranslate-wasm`:

```
crates/fxtranslate-py/
  Cargo.toml
  pyproject.toml            maturin build config (the PyPI project metadata)
  src/lib.rs                the PyO3 module: Translator + discovery + convenience
  README.md                 PyPI-consumer README (specced in the publish issue)
  DEVELOPMENT.md            dev-facing companion (specced in the publish issue)
  python/fxtranslate/       .pyi stub package for type hints + (later) the CLI shell
  tests/                    offline packaging tests (pytest), fixtures below
    fixtures/               recorded Remote Settings snapshot + tiny fixture blobs
```

`Cargo.toml`:

- `name = "fxtranslate-py"`, `version = "0.4.1"` (lockstep), `edition = "2021"`,
  `license = "MPL-2.0"`, `repository = "git+https://github.com/gregtatum/translations.git"`.
- `publish = false` — a cargo directive only. Keeps the crate off crates.io (like the wasm and
  oracle crates); PyPI upload is via maturin/twine, independent of cargo. The `publish.py`
  workspace loader already treats `publish = false` as "guarded" and never `cargo publish`es it
  (verified against the current loader).
- `[lib] crate-type = ["cdylib"]`. PyO3 needs a `cdylib` to produce the extension module. Unlike
  the wasm crate we do **not** need `rlib` — nothing in the workspace depends on this crate, and
  the offline binding tests run through Python (pytest), not `cargo test`. (Open question below
  on whether to keep an `rlib` for a thin native `cargo test`.)
- Dependencies:
  - `fxtranslate = { path = "../fxtranslate", default-features = false, features = ["lean-embed", "gemmology", "mmap", "icu-segmenter", "net"] }`.
    `net` implies `download` implies `discovery`, so this one line gives us the SIMD kernel,
    mmap, Unicode segmentation, Remote Settings discovery, the verified cache, and the built-in
    `ureq`/rustls HTTP client. This is deliberately the *native fast + batteries* config the
    wasm crate can't have (`gemmology`/`mmap`/`net` are all native-only).
  - `pyo3 = { version = "0.22", features = ["extension-module", "abi3-py38"] }`. `abi3-py38`
    builds against the stable ABI, so **one wheel per platform serves all CPython ≥ 3.8**
    instead of one per minor version. `extension-module` omits linking libpython (correct for a
    wheel). Pin to whatever PyO3 line is current-stable and abi3-py38-capable at implementation
    time; the surface uses only stable PyO3 constructs.
- `[build-dependencies]`: none needed here — the C++ `cc` build-dependency lives in the core
  crate's `build.rs` and is pulled transitively by the `gemmology` feature. Verified:
  `crates/fxtranslate/build.rs` gates all C++ compilation behind the `gemmology` feature and
  falls back to the portable scalar kernel when no SIMD kernel/toolchain exists, so a source
  build on an unsupported target still succeeds (scalar). abi3 constrains only the Python C-API
  surface, not the Rust/C++ engine, so there is no abi3 conflict with the vendored kernel.

### `pyproject.toml` (maturin)

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "fxtranslate"
requires-python = ">=3.8"
license = { text = "MPL-2.0" }
description = "Firefox Translations neural machine-translation engine — native (compiled Rust), batteries-included model management."
readme = "README.md"
dynamic = ["version"]

[project.urls]
Repository = "https://github.com/gregtatum/translations"

[tool.maturin]
module-name = "fxtranslate._fxtranslate"
features = ["pyo3/extension-module"]
python-source = "python"
```

Decisions encoded here:

- **Distribution name `fxtranslate`** (matches crates.io + npm), **import name `fxtranslate`**.
  The compiled module is `fxtranslate._fxtranslate` and a thin `python/fxtranslate/__init__.py`
  re-exports its public surface, so users write `from fxtranslate import Translator`. This split
  (a `python-source` package around a private compiled submodule) is the maturin-recommended
  layout and is what makes room for the `.pyi` stubs and, later, a pure-Python CLI entry point
  ([python-cli-conformance.md](python-cli-conformance.md)) without a second wheel.
- **Version is `dynamic`**, sourced from the crate's `Cargo.toml` by maturin — so the lockstep
  bump in `publish.py` only has to touch `Cargo.toml`, not a second version string. (Open
  question: confirm the pinned maturin reads the crate version for `dynamic = ["version"]`; if
  not, `publish.py` bumps a `[project] version` in `pyproject.toml` too, exactly as it already
  bumps `npm/package.json`. This is the one cross-cutting unknown with the publish issue.)

### The binding surface (`src/lib.rs`)

Mirror the wasm surface (`crates/fxtranslate-wasm/src/lib.rs`) as closely as the language allows.
A `#[pymodule]` named `_fxtranslate` exposing:

**`Translator` (`#[pyclass]`)** — two ways in:

1. **BYO-model constructor**, mirroring wasm exactly:
   `Translator(model: bytes, src_vocab: bytes, trg_vocab: bytes, shortlist: bytes | None = None)`.
   Wraps `fxtranslate::engine::Engine::from_bytes(model, src_vocab, trg_vocab)` then
   `.with_shortlist_bytes(..)` when `shortlist` is given (both verified present). Map the engine's
   `Err(String)` to `PyValueError`.
2. **Batteries-included classmethod** (the native advantage — see the recommendation in Open
   questions): `Translator.load(src: str, trg: str, cache_dir: str | None = None) -> Translator`.
   Wraps `fxtranslate::loader::load_translation(&NetworkFetch::new(), &cache, src, trg)` (verified
   present at `loader.rs:80`; it is the pivot-aware entry — returns the core's `Translation` enum,
   handling direct vs. two-leg pivot). `cache` is `fxtranslate::cache::Cache::locate()`
   (platform-native dir, verified) unless `cache_dir` overrides it; `NetworkFetch::new()` is the
   built-in `ureq`/rustls client (verified). This is discover→download→cache→translate in one
   call, with the core's offline-cache fallback already built in — what makes
   `pip install fxtranslate` genuinely batteries-included without Python reimplementing a fetch
   shell.

Because the two paths hold different core types (`Engine` for BYO, `Translation` for `load`), the
`#[pyclass]` wraps an enum over the two (or stores a boxed `dyn`-like translate closure); both
expose the same `translate`/`translate_long` methods (the `Translation` enum forwards to the
underlying engine(s)). Implementer's choice; keep the Python surface identical across both.

Methods on `Translator`:

- `translate(self, text: str) -> str`.
- `translate_long(self, text: str) -> str` (ICU4X segmentation).
- `backend(self) -> str` — report the active int8 GEMM (matrix-multiply) backend. Native
  equivalent of the wasm `backend()`: return `fxtranslate::gemm::backend()` under
  `#[cfg(gemmology_simd)]`, else `"scalar"`. This keeps a silent scalar wheel from being mistaken
  for a SIMD one — valuable for PyPI where a source build may legitimately fall back to scalar.

**Method fates from the wasm surface (spell out explicitly):**

- `linear_memory_bytes` — **omit.** It measures wasm linear-memory pages; meaningless natively.
- `translate_block_phased(block, on_phase)` — **omit from the first release.** It exists only
  because wasm has no usable `std::time::Instant`; natively the engine has real timing. If a timed
  API is ever wanted, expose a Pythonic `translate_timed(text) -> (str, dict)` rather than the
  callback shim. Follow-up, not a blocker.

**`discovery` submodule** — mirror the wasm free functions, but return **native Python objects,
not JSON strings**. The wasm code serializes to JSON by hand only to avoid pulling serde into the
wasm graph; PyO3 has no such constraint and idiomatic Python wants dicts/lists. Expose as a
submodule `fxtranslate.discovery`:

- `parse_records(body: str) -> list[dict]` → `remote::parse_records`, each `Record` as a dict
  with the same keys the wasm JSON uses (`name`, `fileType`, `sourceLanguage`, `targetLanguage`,
  `version`, `architecture`, `decompressedHash`, `location`).
- `resolve_route(records_json: str, src, trg) -> dict` → `route::resolve_route`;
  `{"kind": "direct", ...}` or `{"kind": "pivot", ...}`.
- `catalog(records_json: str, hub: str) -> dict` → `route::catalog`;
  `{"bidirectional": [...], "sourceOnly": [...], "targetOnly": [...]}`.
- `model_pairs(records_json: str) -> list[tuple[str, str]]` → `route::pairs`.
- `segment_sentences(text: str) -> list[str]` → `IcuSegmenter::new().sentences`.
- `verify_and_decompress(compressed: bytes, expected_sha256_hex: str | None = None) -> bytes`
  → `cache::verify_and_decompress`.

Keep the wasm functions' input contract (they take the raw records JSON body) so behavior is
identical; only the *return* type is Pythonic.

### Type stubs

Ship a `python/fxtranslate/__init__.pyi` (and `discovery.pyi`) describing the surface, plus a
`py.typed` marker, so editors and mypy see types even though the module is compiled. This is the
Python analogue of npm's hand-maintained `.d.ts`.

### Offline packaging tests (pytest)

Same stance as [closed/14-rust-only-package.md](closed/14-rust-only-package.md) and the npm
parity harness: **prove the packaging, not the engine.** Tests must not hit the network.

**Fixture reality (verified — read before writing tests):** the core's `tests/fixtures/` holds
`rs-models-v2.json` (a trimmed real Remote Settings records snapshot, 7 records) and
`tiny.bin.zst`. But `tiny.bin.zst` is a **download/cache-plumbing** fixture — `packaging.rs`
states "No network, no engine" and only feeds it through verify/decompress/cache, never through
`Engine::from_bytes`. So it is **not** a loadable, translatable engine. This splits the suite:

**Hermetic suite (always runs, no network, no real model):**

- **Binding-loads test.** `import fxtranslate; fxtranslate.Translator` — proves the wheel loads
  and the module initializes on the running interpreter (the core value of an abi3 wheel).
- **`backend()` test.** Assert it returns a non-empty string that is `"scalar"` or a known SIMD
  backend — guards the silent-scalar-wheel risk.
- **discovery test.** Feed `rs-models-v2.json` through `parse_records` / `resolve_route` /
  `catalog` / `model_pairs` and assert the dict/list shapes and a known pivot resolution; run
  `verify_and_decompress` over `tiny.bin.zst` and assert it decodes (and that a wrong hash
  raises). No network — the body is a fixture. Copy the minimal fixtures into
  `crates/fxtranslate-py/tests/fixtures/`.
- **Error-mapping test.** `Translator(b"garbage", b"", b"")` raises a Python `ValueError` (not a
  panic/abort) — proves the `Err(String)` → exception mapping.

**Opt-in translate-parity test (needs a real model, mirrors Pass B / `conformance-corpus`):**

- Guard behind an env flag / pytest marker (e.g. `FXTRANSLATE_MODEL_DIR` set, else `skip`), the
  way `task rs:conformance-corpus` is opt-in and needs models. Build a `Translator` from a real
  en→es model triple (model + vocab + vocab, **no shortlist**, matching `Engine::load` and the npm
  translate parity test) and **pin the output to the Rust CLI oracle** over the same input
  (transitively oracle-validated) — not a hardcoded self-referential string. The oracle is the
  `fxtranslate` binary at `target/conformance/debug/fxtranslate` (see `scripts/conformance.py`
  `RUST_BIN`). This keeps the anti-cheat validation stance without putting a ~150 MB model in the
  hermetic path.
- **Cache-wiring test (opt-in, no live download).** Point `Translator.load(..., cache_dir=tmp)` at
  a pre-seeded cache directory containing a real model, with the network fetch unreachable, and
  assert it loads from cache (exercises the offline-cache fallback). A live download is a manual
  smoke check, mirroring issue 14's "REPL is a manual smoke feature" stance.

### New Taskfile tasks (`rs:*`)

Add to `Taskfile.yml` under the `rs` namespace, matching the `rs:build-wasm` style:

- `rs:build-py`: `maturin develop -m crates/fxtranslate-py/Cargo.toml` for a local editable
  install into the active venv/toolchain, or `maturin build` for a wheel. Include the same
  "not installed" guard `rs:test` uses for `cargo-nextest` — print an install hint
  (`pipx install maturin`) rather than a bare command-not-found, since maturin is confirmed
  **not** installed locally.
- `rs:test-py`: `maturin develop` then `pytest crates/fxtranslate-py/tests` (hermetic suite).
  Deps on `build-cli` so the Rust oracle binary exists when the opt-in parity test is enabled.

## Acceptance criteria

- `crates/fxtranslate-py` exists as a `publish = false` workspace member at the lockstep version,
  depending on the core with `["lean-embed", "gemmology", "mmap", "icu-segmenter", "net"]`.
- `maturin develop` produces an importable `fxtranslate` module; `from fxtranslate import
  Translator` and `from fxtranslate import discovery` work.
- `Translator` exposes the BYO-model constructor (bytes-parity with wasm) **and**
  `Translator.load(src, trg, cache_dir=None)`; methods `translate`, `translate_long`, `backend`.
  `linear_memory_bytes` and `translate_block_phased` are intentionally absent (documented).
- `discovery` exposes `parse_records`, `resolve_route`, `catalog`, `model_pairs`,
  `segment_sentences`, `verify_and_decompress`, returning native Python objects.
- abi3 (`abi3-py38`): one built wheel imports on ≥ 2 CPython minors (e.g. 3.10 and 3.13) without
  rebuild.
- The hermetic pytest suite passes with no network and no real model; the opt-in translate test
  pins to the Rust CLI oracle when a model is provided.
- `task rs:build-py` and `task rs:test-py` exist and guard the missing-maturin case.

## Open questions

- **`Translator.load` convenience vs. mirroring npm's split shell — recommendation: ship
  `load`.** Because the binding is native and pulls `net`, the whole discover→download→cache pivot
  flow already exists in `loader::load_translation` (verified); exposing it is a thin wrapper. The
  npm package kept the shell in JS only because wasm can't do `net` — that constraint does not
  apply here. Recommend `load` as a classmethod **in addition to** the BYO constructor. Confirm
  the API shape (`classmethod` vs a module-level `load_translator` function) at implementation.
- Does the pinned maturin honor `dynamic = ["version"]` from the crate `Cargo.toml`, or must
  `publish.py` also rewrite a `[project] version` in `pyproject.toml`? Verify; if the latter, the
  publish issue's version-lockstep step must cover it (it already handles `npm/package.json`).
- Keep a `cargo test`-able `rlib` alongside the `cdylib` (the wasm crate does), or rely solely on
  pytest? Leaning pytest-only, since the surface is a thin wrapper.
- Confirm the `Translator` internal representation for the two constructors (enum over
  `Engine` / `Translation`, or a common translate closure) once the `Translation` API is read in
  full.
