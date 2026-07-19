# Developing `fxtranslate` (Python)

Developer notes for the Python package: what it is, how it's structured, how it's built and tested. For *using* the package, see [README.md](./README.md); for the release process, see [RELEASING.md](../../RELEASING.md).

## What this package is

One PyPI package — the core `fxtranslate` engine bound to Python with [PyO3] and packaged as a wheel with [maturin]. Unlike the npm/wasm build, it ships the fast, batteries-included native engine: the gemmology SIMD int8 GEMM kernel, `mmap`, ICU4X sentence segmentation, and built-in networked model discovery/download/caching.

- **Library** — `from fxtranslate import Translator` (the `Translator.load` batteries-included path + the bring-your-own-model constructor).
- **Discovery surface** — `from fxtranslate import discovery` (the pure model-discovery/routing helpers, returning native Python objects).

It is a `publish = false` cargo crate (kept off crates.io, like the wasm and oracle crates); the wheel is built and uploaded to PyPI independently of cargo.

## Layout

```
crates/fxtranslate-py/
  Cargo.toml           crate metadata + the fxtranslate (native) + pyo3 deps; the version source
  pyproject.toml       maturin build config / PyPI project metadata (version is dynamic from Cargo.toml)
  src/lib.rs           the PyO3 module: Translator + the discovery submodule
  python/fxtranslate/  the friendly package — __init__.py re-export + .pyi stubs + py.typed
  tests/               hermetic pytest suite + fixtures (copied from the core crate)
  dist/                maturin/twine build output (gitignored; the publisher builds here)
```

The compiled module is the private `fxtranslate._fxtranslate`; the thin `python/fxtranslate/__init__.py` re-exports its surface so users write `from fxtranslate import Translator`. This python-source-package-around-a-compiled-submodule split is maturin's recommended layout and makes room for the `.pyi` stubs (and, later, a pure-Python CLI) without a second wheel.

The wheel is built with PyO3's `abi3-py38`, so one wheel per platform serves every CPython ≥ 3.8 — the built wheel filename contains `abi3` (e.g. `cp38-abi3-...`). `pyproject.toml` declares `dynamic = ["version"]`, so maturin reads the version straight from `Cargo.toml`; the workspace publisher bumps that one place and the wheel/sdist track it for free.

The `pyo3/extension-module` feature (which omits linking libpython — correct for a loadable extension) is turned on **only** through maturin (`[tool.maturin] features`), not in `Cargo.toml`. That lets a plain `cargo build -p fxtranslate-py` link against libpython for a dev/check build while the shipped wheel still omits it — enabling it unconditionally would break `cargo build` with undefined `_Py*` symbols.

## Build & test

```sh
task rs:build-py                 # build a wheel into target/wheels/
task rs:build-py -- develop      # editable install into the active venv
task rs:test-py                  # maturin develop + hermetic pytest
```

Both tasks guard the missing-maturin case with an install hint (`pipx install maturin`), matching `rs:test`'s nextest guard.

Manual, from a fresh venv:

```sh
python3 -m venv /tmp/fxpy && source /tmp/fxpy/bin/activate
pip install maturin pytest
maturin develop -m crates/fxtranslate-py/Cargo.toml
python -c "from fxtranslate import Translator, discovery; print(Translator, discovery)"
pytest crates/fxtranslate-py/tests
```

## The test strategy: hermetic by default, oracle-pinned parity opt-in

The suite is split so a plain `task rs:test-py` needs no network and no real model:

- **Hermetic (always runs)** — the wheel loads and initializes; `backend()` reports a known value; the discovery surface runs over the recorded `rs-models-v2.json` fixture and `verify_and_decompress` over `tiny.bin.zst`; and `Translator(b"garbage", b"", b"")` maps to a `ValueError`. No network, no real model.
- **Translate parity (opt-in)** — skipped unless `FXTRANSLATE_MODEL_DIR` points at a directory with a real en→es model triple. When enabled it pins the output to the Rust CLI oracle (`target/conformance/debug/fxtranslate`), **not** a hardcoded string, so the Python binding stays byte-for-byte with the native CLI rather than testing against a frozen expectation.

## Releasing

The Python package is published **in lockstep with the crates.io crates and the npm package**, on one shared version, by `scripts/publish.py`. Don't `twine upload` by hand for a release — run the workspace publisher so all three registries move together and get one git tag. By default it does everything except the PyPI push (it builds the sdist + local wheel, runs `twine check`, and prints the finish commands); pass `--pypi-upload` to include the upload. See [RELEASING.md](../../RELEASING.md).

[PyO3]: https://pyo3.rs
[maturin]: https://www.maturin.rs
