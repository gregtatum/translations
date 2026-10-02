# inference-rs

This folder contains a Rust reimplementation of the Firefox Translations inference engine. It's built to be memory-safe and embeddable, and is optimized the lightweight and CPU-optimized models that Firefox uses for private on-device translations.

This file serves as the entry-point and orientation into the umbrella project. For the `fxtranslate` library itself, see [`crates/fxtranslate/README.md`](./crates/fxtranslate/README.md).

- [DEVELOPMENT.md](./DEVELOPMENT.md)
- [RELEASING.md](./RELEASING.md)

## Getting started

All of the commands are managed with the [`Taskfile.yml`](Taskfile.yml) (see [taskfile.dev](https://taskfile.dev/)). This file is the source of truth for documentation and all of the commands can be listed with `task --list`.

```bash
# A lightning fast correctness check for testing and linting.
task rs:check

# The fxtranslate-cli crate commands are exposed via:
task rs:fxtranslate -- --help

# For example to automatically download the model and translate from English to Spanish:
task rs:fxtranslate -- translate en es "The weather is nice today."
```

If you run `task` from the repo root, you must use the `rs:` prefix. If you run from `./inference-rs` the `rs:` prefix can be dropped, for example `task fxtranslate -- --help`.

## Map

Every top-level entry in this directory:

| path | what it is |
|---|---|
| `crates/fxtranslate` | The engine library. Published to crates.io. |
| `crates/fxtranslate-cli` | The batteries-included CLI (`fxtranslate` binary). Published to crates.io, and the reference every other CLI is held to. |
| `crates/fxtranslate-wasm` | wasm-bindgen binding. Built by `wasm-pack` and copied into `npm/` |
| `crates/fxtranslate-py` | PyO3 binding. Shipped to PyPI as a maturin wheel |
| `crates/fxtranslate-gecko` | C-ABI binding, for vendoring into Firefox. |
| `crates/fxtranslate-oracle` | Dev-only validation harness and raw diagnostic binary — see its [README](./crates/fxtranslate-oracle/README.md). |
| `npm/` | The npm package: a JS CLI and library over the copied wasm core. |
| `onnx/` | ONNX export evaluation — see its [README](./onnx/README.md). |
| `ggml/` | Bare-libggml port evaluation — see its [README](./ggml/README.md). |
| `scripts/` | The Python harnesses the tasks drive (checks, parity, perf, conformance, publish). Run them through `task`, not directly. |
| `corpora/` | Committed text and token-id fixtures the harnesses translate |
| `notes/` | Numbered design and analysis history, oldest to newest. AI-generated notes. |
| `issues/` | Task specs, moved to `issues/closed/` when done. |
| `artifacts/` | Generated output — traces, profiles, goldens. Entirely gitignored. Nothing here is a source of truth. |
| `architecture.md` | The model itself: why the decoder is an SSRU and not a Transformer decoder, and which `model.yml` keys lie. |
| `gemm-backends.md` | The int8 GEMM backends, and where their numbers can legitimately diverge. |
| `pivot-translations.md` | How pairs with no direct model (`es → fr`) route through a hub language. |
| `Taskfile.yml` | Every task, included by the repo-root Taskfile under the `rs:` namespace. |
| `CHANGELOG.md` | Release history. |

Script downloaded models live outside this directory, in the repo-root `data/models/<src><trg>/`
(gitignored), and shared with the C++ engine. `fxtranslate-cli` uses the OS-specific cache location.
