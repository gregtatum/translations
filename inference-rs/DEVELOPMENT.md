# Developing inference-rs

This document serves as a deeper dive compared to the [README.md](./README.md) into developing `inference-rs`. It explains how the Rust library is built, the oracle system, and the binding layers, as well as other general development standards.

## The check gate

```sh
task rs:check
```

Run this after any code change. It parallelizes and schedules the tasks for the quickest wall clock time possible. These checks match what run in CI.

When run interactively it provides an ergonomic view into the tasks running. When running non-interactively (in an AI agent) the output is compact and token friendly.

To fix all formatting errors run:

```sh
task rs:lint-fix
```

## The model is validated against a Marian oracle

The translations training pipeline runs [Marian](https://github.com/marian-nmt/marian-dev/) at training time. In `fxtranslate-rs` a vendored fork of it used as the reference inference engine. This project validates the implementation using an oracle system that inspects the internal states of the Marian system.

For comparing fxtranslate against Marian, strict byte-equality checks break because of the backend's implementation details (e.g. non-associative float addition, and accumulators with different precision/behavior). Instead we compare tensors within a tight tolerance.

The Marian fork was modified to allow `task inference-build` to be built without Docker on Apple Silicon machines.

The engine is validated at the **op-level** with intermediate tensors recorded from reference. It is also evaluated **end-to-end** against reference translations (`task rs:parity`).

### Recording a reference trace

The C++ engine can record every intermediate tensor of one translation:

```bash
task rs:translate-reference -- en fr --text "Hello world." --cpu-threads 1 --trace
```


This writes `artifacts/<src><trg>.trace` — one record per graph node in forward-execution order
(`{id, op, name, dtype, shape, child ids, raw bytes}`) — plus a `.trace.txt` manifest of the same
nodes with shapes only. The binary format is documented at the top of
[`trace_recorder.h`](../inference/marian-fork/src/graph/trace_recorder.h). Under the hood this is
the `MARIAN_TRACE` env var, a no-op on normal runs; `--trace` just sets it.

Keep `--text` short and `--cpu-threads 1`, because static model parameters are re-recorded on
every decoding step: one short sentence is already ~170 MB.

On the Rust side, [`trace.rs`](./crates/fxtranslate/src/trace.rs) parses a trace into per-node
fixtures with typed views over the tensor bytes, and
[`compare.rs`](./crates/fxtranslate-oracle/src/compare.rs) holds the within-tolerance float
comparator the parity tests assert against. To inspect one by hand (also a smoke check that the
reader survives a real full-size trace):

```bash
cargo run -p fxtranslate-oracle -- trace artifacts/enfr.trace 20
```

## The bindings to the engine

The engine is built as the Rust `fxtranslate` library, and is shared across different programming ecosystems. The core engine is shared, but each library must handle particulars for that ecosystem.

- **`fxtranslate-cli`** ([crates.io](https://crates.io/crates/fxtranslate-cli)) This is the reference implementation that other CLIs must match.
- **`fxtranslate-gecko`** exposes a bare C ABI over the engine for embedding, no CLI.
- **`fxtranslate-py`** Re-implements the `fxtranslate-cli` interface, and ships built wheels for the underlying library.
- **`fxtranslate-wasm`** Builds the wasm core library – wasm32 SIMD128 rather than native SIMD int8.
- **`npm` Builds the CLI for use in node.js, and copies the artifacts from `fxtranslate-wasm`.

Each CLI is kept in sync with a conformance test that checks that they have the same behavior byte-for-byte to stdout, stderr, and the exit code. `task rs:conformance` runs all of them, and `task rs:conformance -- --only {npm,python,rust}` narrows the check. This allows for agentic updating of the codepaths so we can ship in multiple ecosystems.

## Where each ecosystem's tasks live

`Taskfile.yml` is the source of truth and carries a description and examples for every task;
this is only a map of which cluster to look in.

| area | task cluster |
|---|---|
| Rust engine and CLI | `rs:test`, `rs:release`, `rs:fxtranslate` |
| npm / wasm | the `npm/` package scripts — `build:wasm`, `typecheck`, `check:parity` |
| Python | `rs:build-py`, `rs:test-py` |
| cross-CLI conformance | `rs:conformance`, `rs:conformance-corpus` |
| the C++ oracle | `inference-build`, `rs:translate-reference`, `rs:parity` |
| perf and memory | `rs:perf`, `rs:kernel-micro`, `rs:spec-probe`, `rs:spec-bench` |
| fixtures and goldens | `rs:download-model`, `rs:sample-corpus`, `rs:make-blocks`, `rs:spm-goldens`, `rs:capture-bergamot-goldens` |
| runtime evaluations | `rs:onnx-*`, `rs:ggml-*` (see [onnx/](./onnx/README.md), [ggml/](./ggml/README.md)) |
| lint and release | `rs:lint-black`, `rs:lint-rust`, `rs:lint-ci`, `rs:publish` |

## CI

[`.github/workflows/inference-rs.yml`](../.github/workflows/inference-rs.yml) runs the same `task rs:<name>` commands as `task rs:check`. In addition, per-platform checks are run in CI as well which gets additional test coverage.

## Conventions

- **`notes/`** is numbered, append-only AI design and analysis history. Nothing in it is edited to stay current. Read the highest numbers for present thinking and treat low numbers as a record of what was believed then.
- **`issues/`** holds task specs, moved to `issues/closed/` when done. Closed specs are kept because they document why something is shaped the way it is. These were used to drive agentic workflows.
- **`artifacts/`** is entirely gitignored and holds generated output — traces, profiles, goldens. Nothing in it is a source of truth.
- **`corpora/`** holds committed fixtures with `.sha256` companions, so a changed corpus is a visible change rather than a silently different benchmark.
- **`scripts/`** is driven through `task`, not run directly. `statusboard.py` is shared by `rs:check` and the release publisher so the two boards read as one family.
- Comments and docs explain *why* rather than restating what the code does.
