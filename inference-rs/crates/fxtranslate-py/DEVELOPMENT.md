# Developing `fxtranslate` (Python)

Developer notes for the Python package: what it is, how it's structured, how it's built and tested. For *using* the package, see [README.md](./README.md); for the release process, see [RELEASING.md](../../RELEASING.md).

## What this package is

One PyPI package — the core `fxtranslate` engine bound to Python with [PyO3] and packaged as a wheel with [maturin]. Unlike the npm/wasm build, it ships the fast, batteries-included native engine: the gemmology SIMD int8 GEMM kernel, `mmap`, ICU4X sentence segmentation, and built-in networked model discovery/download/caching.

- **Library** — `from fxtranslate import Translator` (the `Translator.load` batteries-included path + the bring-your-own-model constructor).
- **Discovery surface** — `from fxtranslate import discovery` (the pure model-discovery/routing helpers, returning native Python objects).
- **CLI** — a `fxtranslate` console script (and `python -m fxtranslate`), byte-identical to the Rust `fxtranslate-cli`.

It is a `publish = false` cargo crate (kept off crates.io, like the wasm and oracle crates); the wheel is built and uploaded to PyPI independently of cargo.

## Layout

```
crates/fxtranslate-py/
  Cargo.toml           crate metadata + deps; the version source
  pyproject.toml       maturin build config / PyPI metadata (version is dynamic from Cargo.toml)
  src/lib.rs           the PyO3 module: Translator + Cache + the discovery submodule
  python/fxtranslate/  the friendly package — __init__.py re-export + .pyi stubs + py.typed + the CLI shell
  tests/               hermetic pytest suite + fixtures
```

Two conventions here are deliberate, following maturin's guidance:

- The compiled module is the private `fxtranslate._engine`, and `python/fxtranslate/__init__.py` re-exports its surface so users write `from fxtranslate import Translator`. Wrapping a compiled submodule in a Python source package is what makes room for `.pyi` stubs and the CLI shell without a second wheel. (The lib is named `_engine`, not `fxtranslate`, so it doesn't collide with the core crate's `libfxtranslate.dylib` — see `Cargo.toml`.)
- `pyproject.toml` declares `dynamic = ["version"]`, so maturin reads the version straight from `Cargo.toml` — the workspace publisher bumps that one place.

The wheel is built with PyO3's `abi3-py38`, so one wheel per platform serves every CPython ≥ 3.8. The `pyo3/extension-module` feature is set **only** through maturin (`[tool.maturin] features`), never in `Cargo.toml`, so a plain `cargo build -p fxtranslate-py` still links (see the `Cargo.toml` comment).

## Build & test

```sh
task rs:build-py                 # build a wheel into target/wheels/
task rs:build-py -- develop      # editable install into the active venv
task rs:test-py                  # maturin develop + hermetic pytest
```

Both tasks guard the missing-maturin case with an install hint (`pipx install maturin`).

Manual, from a fresh venv:

```sh
python3 -m venv /tmp/fxpy && source /tmp/fxpy/bin/activate
pip install maturin pytest
maturin develop -m crates/fxtranslate-py/Cargo.toml
pytest crates/fxtranslate-py/tests
```

## Test strategy: hermetic by default, oracle-pinned parity opt-in

A plain `task rs:test-py` needs no network and no real model:

- **Hermetic (always runs)** — the wheel loads and initializes, the discovery surface runs over recorded fixtures, and error paths map to the right Python exceptions.
- **Translate parity (opt-in)** — skipped unless `FXTRANSLATE_MODEL_DIR` points at a directory with a real en→es model triple. When enabled it pins output to the Rust CLI oracle, **not** a hardcoded string, so the binding stays byte-for-byte with the native CLI rather than against a frozen expectation.

## The CLI

The `fxtranslate` console script is held byte-identical to the Rust `fxtranslate-cli` — same subcommands, flags, help/usage text, error strings, exit codes, and formatters. It re-implements only the *interface* (arg parsing, help/error text, formatters); everything else — translate, discovery/routing, and cache storage — delegates to the **compiled** engine rather than being re-ported. (This is the opposite of the npm shell, which must re-port that logic in JS because its wasm core can't touch the network or filesystem.)

It is registered as the third entry in the cross-CLI conformance harness alongside the Rust reference and npm:

```sh
task rs:conformance                    # rust + npm + python vs the goldens
task rs:conformance -- --only python   # just the python binding
```

Python is a *soft* dep of the harness: with no maturin build available it SKIPs cleanly (like npm without its wasm build) rather than failing.

## Releasing

The Python package is published **in lockstep with the crates.io crates and the npm package**, on one shared version, by `scripts/publish.py`. Don't `twine upload` by hand for a release — run the workspace publisher so all three registries move together under one git tag. See [RELEASING.md](../../RELEASING.md).

[PyO3]: https://pyo3.rs
[maturin]: https://www.maturin.rs
```
