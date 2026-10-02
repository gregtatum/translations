# inference-rs

A Rust reimplementation of the Firefox Translations inference engine, validated against the
C++ engine it replaces (`inference/build/src/app/translator-cli`). One engine ships to four
places: crates.io, npm, PyPI, and Firefox.

This file is the orientation for working *in* the workspace. For *using* the engine or the CLI
— installation, the library API, performance — see [`crates/fxtranslate/README.md`](./crates/fxtranslate/README.md).

- [DEVELOPMENT.md](./DEVELOPMENT.md) — how correctness is established, how the four bindings
  relate, conformance, CI, and the conventions this directory follows.
- [RELEASING.md](./RELEASING.md) — shipping all four artifacts on one shared version.

## Get going

You need a [Rust toolchain](https://rustup.rs), [go-task](https://taskfile.dev),
[cargo-nextest](https://nexte.st), and a C++ compiler — the default `fast` build compiles a
small SIMD shim (`gemmology_shim.cpp`) through `cc`, and `--features portable` is the
toolchain-less fallback. Building the reference C++ engine (`task inference-build`) needs its
own toolchain; `wasm-pack` and `maturin` are needed only for the npm and Python bindings. Tasks
print an install hint when a tool they need is missing, so you can add them as you go.

```bash
task rs:check                                 # the gate: lints, tests, conformance, C++ engine build
task rs:download-model -- en es               # fetch a model triple into data/models/
task rs:fxtranslate -- translate en es "The weather is nice today."
```

`task rs:check` is the one to run after any code change — it is the same set of checks CI runs,
rendered as a compact pass/fail board.

**The Taskfile is the source of truth for how to run things**, and this file does not duplicate
it. There are 48 tasks:

```bash
task --list | grep '^\* rs:'       # every task with its one-line description
task rs:<name> --summary           # the long form: what it does, what it needs, examples
```

## Map

Every top-level entry in this directory:

| path | what it is |
|---|---|
| `crates/fxtranslate` | The engine library. Published to crates.io. |
| `crates/fxtranslate-cli` | The batteries-included CLI (`fxtranslate` binary). Published to crates.io, and the reference every other CLI is held to. |
| `crates/fxtranslate-wasm` | wasm-bindgen binding. Built by `wasm-pack` and copied into `npm/`; not published to crates.io. |
| `crates/fxtranslate-py` | PyO3 binding. Shipped to PyPI as a maturin wheel; not published to crates.io. |
| `crates/fxtranslate-gecko` | C-ABI binding, vendored into Firefox. Not published anywhere. |
| `crates/fxtranslate-oracle` | Dev-only validation harness and raw diagnostic binary — see its [README](./crates/fxtranslate-oracle/README.md). |
| `npm/` | The npm package: a JS CLI and library over the copied wasm core. |
| `onnx/` | Clean-room ONNX export evaluation — see its [README](./onnx/README.md). |
| `ggml/` | Bare-libggml port evaluation — see its [README](./ggml/README.md). |
| `scripts/` | The Python harnesses the tasks drive (checks, parity, perf, conformance, publish). Run them through `task`, not directly. |
| `corpora/` | Committed text and token-id fixtures the harnesses translate, with `.sha256` companions. |
| `notes/` | Numbered design and analysis history, oldest to newest. The highest numbers are the current thinking. |
| `issues/` | Task specs, moved to `issues/closed/` when done. |
| `artifacts/` | Generated output — traces, profiles, goldens. Entirely gitignored; nothing here is a source of truth. |
| `architecture.md` | The model itself: why the decoder is an SSRU and not a Transformer decoder, and which `model.yml` keys lie. |
| `gemm-backends.md` | The int8 GEMM backends, and where their numbers can legitimately diverge. |
| `pivot-translations.md` | How pairs with no direct model (`es → fr`) route through a hub language. |
| `Taskfile.yml` | Every task, included by the repo-root Taskfile under the `rs:` namespace. |
| `CHANGELOG.md` | Release history. |

Downloaded models live outside this directory, in the repo-root `data/models/<src><trg>/`
(gitignored), shared with the C++ engine.
