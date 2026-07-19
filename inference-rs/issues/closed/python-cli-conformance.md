# Python CLI + conformance REGISTRY entry (secondary)

**Open, secondary. Depends on [pypi-pyo3-bindings.md](pypi-pyo3-bindings.md); can land after the
first PyPI release.** A `fxtranslate` console-script for the Python package, byte-parity with the
Rust `fxtranslate-cli`, added to the cross-CLI conformance harness as one more REGISTRY entry.
This is desirable-but-secondary — it must **not** block getting the library on PyPI, so it is its
own issue after [pypi-publish-integration.md](pypi-publish-integration.md).

## Background

The project already holds two CLIs byte-identical to the Rust oracle: the Rust binary itself and
the npm `bin/fxtranslate.js`, both registered in `scripts/conformance.py`'s `REGISTRY` and checked
by `task rs:conformance` (Pass A, hermetic byte-exact) and `task rs:conformance-corpus` (Pass B,
tolerant translation parity). Adding a language binding is, by design, "one entry in the
REGISTRY." A Python CLI is the natural third entry.

Unlike npm — where the CLI shell (arg grammar, help, cache, formatters) is hand-ported JS over a
wasm core — the Python package is **native and already batteries-included** (`net` + `loader` +
`Cache`). So the Python CLI can be a thin shell over the same compiled engine surface
(`Translator.load`, `discovery`), not a reimplementation of discovery/routing.

## Design

- **Console script**: a `[project.scripts]` entry in `crates/fxtranslate-py/pyproject.toml`
  binding `fxtranslate = "fxtranslate.cli:main"`, so `pip install fxtranslate` also installs the
  `fxtranslate` command. The CLI code lives in the `python/fxtranslate/` source package (cli.py,
  usage.py, run.py, format.py, lang.py — mirroring the npm `lib/` split), a thin shell over the
  compiled `Translator`/`discovery` surface.
- **Byte-parity contract**: same subcommands, flags, help/usage text, error strings, exit codes,
  and list/models formatters as the Rust `fxtranslate-cli` oracle — the same contract the npm
  DEVELOPMENT doc describes. Port the arg grammar, `USAGE`/`MODELS_USAGE`/`LIST_USAGE`, and the
  formatters byte-for-byte from `crates/fxtranslate-cli/src`.
- **Conformance REGISTRY**: add a third `Cli` to `scripts/conformance.py`'s `REGISTRY`, e.g.
  `Cli(name="py", prefix=["python", "-m", "fxtranslate"])`. Add a `PY_BIN`/entry-point resolution
  next to `RUST_BIN`/`NPM_BIN`. The Pass-A hermetic harness (loopback records server, throwaway
  cache dirs, committed goldens) and Pass-B corpus parity then cover the Python CLI automatically
  — same goldens, same asserts.
- **Taskfile**: extend `rs:conformance`'s deps so the Python entry point is available (a
  `maturin develop` step, mirroring how `build-wasm` prepares the npm binding), and document
  `task rs:conformance -- --only py`.
- **Translate parity**: like npm, hold a numeric tolerance on the translate path (build the engine
  from model + two vocabs, no shortlist — matching `Engine::load`), byte-identity on everything
  else (help/errors/list/models).

## Acceptance criteria

- `pip install fxtranslate` installs a `fxtranslate` console script whose help/usage/error output
  is byte-identical to the Rust `fxtranslate-cli`.
- A `py` entry in `scripts/conformance.py`'s `REGISTRY`; `task rs:conformance` and
  `task rs:conformance-corpus` run it and it passes (byte-exact Pass A; within tolerance Pass B).
- `task rs:conformance -- --only py` works; the harness builds the Python entry point via a
  maturin dep step.
- Landing this does not change or block the library-only first PyPI release.

## Open questions

- Invocation registered in the harness: the installed `fxtranslate` script vs. `python -m
  fxtranslate` — prefer `python -m fxtranslate` in CI so it works from the built wheel/editable
  install without PATH assumptions.
- Does the native `Translator.load` path make the hermetic Pass-A cache/`models` tests simpler
  than npm's (which reimplements the cache in JS), since the Python CLI uses the core `Cache`
  directly? Likely yes.
- Which CLI error strings are produced by the core (shared, free) vs. the Rust shell (must be
  re-ported in Python)? Audit `crates/fxtranslate-cli/src` at implementation time.
