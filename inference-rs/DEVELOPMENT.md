# Developing inference-rs

How correctness is established here, how the four bindings relate to each other, and the
conventions this workspace follows. For orientation and the task list see
[README.md](./README.md); for shipping see [RELEASING.md](./RELEASING.md).

## The check gate

```bash
task rs:check
```

Run this after any code change. It is the single gate, and CI runs the same checks: Black and
rustfmt lints, a CI-drift lint, the workspace Rust tests, CLI conformance, and a build of the
reference C++ engine. The check list lives in `CHECKS` at the top of `scripts/check.py` — add
to that list, not to a second place.

Two scheduling details explain what you see. Light checks run in parallel, but the three heavy
ones (Rust tests, conformance, the C++ build) each saturate every core, so they are serialized
— measured at ~12s serialized versus ~19s racing each other on six performance cores. And an
interactive run renders a live board while a non-TTY run stops at the first failure, which is
what makes it usable in a pipe or from an agent.

`task rs:lint-ci` is the reason local and CI can't silently diverge: it fails if
`.github/workflows/inference-rs.yml` stops matching the `CHECKS` list. The heavy C++
`inference-build` is the one deliberate exemption — it stays local.

`task rs:test` is one `cargo nextest` run across the whole workspace in a single parallel pool,
followed by a `cargo test --doc` pass because nextest does not run doctests. Feature unification
pulls in `net` through the CLI crate and `fast` is already the default, so `fxtranslate` compiles
exactly once while every feature-gated test still runs.

## Correctness comes from an oracle, not from expectations

The reference C++ engine is the source of truth. Almost nothing here is asserted against a
hardcoded expectation — tests are pinned to what the reference engine actually does, so the port
cannot drift by agreeing with a stale guess.

That is affordable because the reference builds and runs natively on Apple Silicon, no Docker:

```bash
task inference-build        # builds inference/build/src/app/translator-cli
```

On ARM that build uses the gemmology int8 backend, which runs the same `int8shiftAlphaAll`
algorithm as the shipped WASM models — so a native build is a faithful oracle, not an
approximation of one. See [gemm-backends.md](./gemm-backends.md) for how the int8 backends line
up per architecture, and where two CPUs can legitimately disagree.

The engine is validated on two axes: **op-level**, against intermediate tensors recorded from the
reference, and **end-to-end**, against reference translations (`task rs:parity`).

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

The real-trace tests skip when no trace is present, so `task rs:test` passes without one — which
means a green test run does *not* prove the trace path still works. Record a trace when you touch
op-level code. [`crates/fxtranslate-oracle/README.md`](./crates/fxtranslate-oracle/README.md)
documents the harness internals: the comparator, the replay bisector that finds the first
diverging node, and the diagnostic binary.

## Four bindings, one engine

The engine is written once and reaches four ecosystems. The bindings differ along one axis that
explains nearly everything else: **how much of the work happens outside the compiled engine.**

| binding | engine form | what the binding re-implements | held to the engine by | ships as |
|---|---|---|---|---|
| `fxtranslate-cli` | native SIMD int8 | nothing — it *is* the reference | it is the oracle | crates.io |
| `fxtranslate-wasm` → `npm/` | wasm32 + SIMD128 | the entire async shell in JS: arg grammar, help and error text, discovery, download, cache, formatters | `npm run check:parity` locally, plus conformance Pass A | npm |
| `fxtranslate-py` | native SIMD int8 | the CLI *interface* only — arg parsing, help text, formatters | conformance Pass A; opt-in translate parity | PyPI wheel (abi3) |
| `fxtranslate-gecko` | native SIMD int8 | nothing — a C ABI over the engine | in-process `tests/ffi_parity.rs` | vendored into Firefox |

**The wasm core cannot reach the network or the filesystem.** That single constraint is why the
npm binding re-ports so much more than the others: everything outside pure computation — model
discovery, downloading, the on-disk cache, progress output — has to be written again in JS and
then held to the Rust original by tests. Python and Gecko run the native engine, so they delegate
all of it and re-implement at most the surface text. When you are weighing where a change is
risky, that is the ranking: npm carries real re-implementation risk, Python carries interface
risk, Gecko carries almost none.

[`notes/13-cli-parity.md`](./notes/13-cli-parity.md) is the full design rationale for that split —
why the shell is native per language while only the pure decision logic is shared wasm.

The Gecko crate builds as both `staticlib` and `rlib` for a specific reason: `staticlib` is what
Firefox links through `gkrust`, while `rlib` lets `tests/ffi_parity.rs` call the same C ABI
in-process and prove it byte-exact against the standalone engine. Its feature selection is
deliberate and annotated in its own `Cargo.toml` — no `mmap` because Firefox feeds bytes rather
than paths, no `net`/`download` because Firefox owns HTTP and Remote Settings, but
`icu-segmenter` on, because Firefox no longer sentence-splits in JS and segmentation has to ship
in the engine.

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

## Conformance: one harness, two passes

Every CLI — Rust, npm, Python — is registered in one harness, and the Rust binary is the oracle.

**Pass A — `task rs:conformance`** asserts byte-identical stdout, stderr, and exit code across
all three for help, errors, `list`, and `models list/info/rm`. It is hermetic: a loopback records
server and throwaway cache directories, no network, no model downloads. That is what lets it sit
in the default `task rs:check`. Goldens are regenerated from the oracle with
`scripts/conformance.py --update-goldens`, and `task rs:conformance -- --only python` narrows to
one binding.

**Pass B — `task rs:conformance-corpus`** feeds a corpus through each CLI's `translate` and
reports a line-by-line exact-match rate against a committed Rust golden, passing above a
documented tolerance floor. It needs real models (~150 MB per pair) and is slow, so it is in
neither `rs:check` nor CI — run it deliberately.

The split between the two passes is a real statement about what is guaranteed, not a
convenience: **the interface is byte-identical, the translations are not.** The npm-wasm path
diverges on a handful of ≤1-ULP libm differences that flip an argmax, while the int8 GEMM itself
stays bit-identical. So interface drift is a bug and translation drift inside tolerance is
expected. The npm package's own `translate` path likewise holds a numeric tolerance by design: it
builds the engine from model plus two vocabs and no shortlist, matching Rust's `Engine::load`.

Both bindings are *soft* dependencies of the harness. With no maturin build available the Python
binding SKIPs cleanly, as npm does without its wasm build, rather than failing the run.

## Ecosystem gotchas

The things that cost a day if you don't know them.

**npm**
- `npm/wasm/` is a **copy** of the `fxtranslate-wasm` build output, not a reference to the
  sibling crate. That is what keeps the published tarball self-contained, so a consumer gets the
  engine without the Rust workspace. It is also gitignored, so a fresh `npm run build:wasm` is a
  prerequisite for packing or publishing — `prepublishOnly` enforces it. `task rs:conformance`
  rebuilds it for you as a dependency, which is usually how it stays current.
- The JS ships exactly as authored: JSDoc, no transpile step in the publish path.
  `npm run typecheck` is `tsc --noEmit`. `lib/index.d.ts` re-exports the wasm-pack-generated
  `.d.ts` rather than restating it, so the engine's types cannot drift from the build.
- `npm run check:parity` needs the oracle built first: `cargo build -p fxtranslate-cli`.

**Python**
- The compiled module is the private `fxtranslate._engine`, re-exported by a Python source
  package. Wrapping it that way is what makes room for `.pyi` stubs and the CLI shell without a
  second wheel. It is named `_engine` so it cannot collide with the core crate's
  `libfxtranslate.dylib`.
- `pyo3/extension-module` is set **only** through maturin's `[tool.maturin] features`, never in
  `Cargo.toml` — otherwise a plain `cargo build -p fxtranslate-py` stops linking.
- `abi3-py38` means one wheel per platform covers every CPython ≥ 3.8, rather than one wheel per
  Python minor version.
- `pyproject.toml` declares `dynamic = ["version"]` and reads the version from `Cargo.toml`, so
  the release tooling bumps one place.

**All non-published crates** (`-wasm`, `-py`, `-gecko`, `-oracle`) are `publish = false`. That
keeps them off crates.io; it does **not** stop `wasm-pack` or `maturin` from building their
artifacts.

## CI

[`.github/workflows/inference-rs.yml`](../.github/workflows/inference-rs.yml) runs the same
`task rs:<name>` commands as the local gate, which `task rs:lint-ci` enforces. Two jobs then add
coverage a single developer machine cannot give you:

- **`test`** runs the same `rs:test` across a two-row matrix — `ubuntu-24.04-arm` for the exact
  i8mm backend and `ubuntu-latest` for x86 AVX2 — and reports which SIMD backend was actually
  live, so a silent scalar fallback can't pass as a SIMD run. It also builds and tests the
  `threads` and `gemm-threads` features, which are off by default and so absent from `rs:check`.
- **`test-portable`** builds with no `task` setup and no `g++` at all, proving the scalar
  `portable` kernel path still works for users without a C++ toolchain.

The x86 runners top out at AVX2, so the exact VNNI backends cannot be validated in CI at all —
see [gemm-backends.md](./gemm-backends.md) for what that leaves unproven.

## Conventions

- **`notes/`** is numbered, append-only design and analysis history. Nothing in it is edited to
  stay current; read the highest numbers for present thinking and treat low numbers as a record
  of what was believed then.
- **`issues/`** holds task specs, moved to `issues/closed/` when done. Closed specs are kept
  because they document why something is shaped the way it is.
- **`artifacts/`** is entirely gitignored and holds generated output — traces, profiles,
  goldens. Nothing in it is a source of truth, and it is large (traces run to hundreds of MB).
- **`corpora/`** holds committed fixtures with `.sha256` companions, so a changed corpus is a
  visible change rather than a silently different benchmark.
- **`scripts/`** is driven through `task`, not run directly. `statusboard.py` is shared by
  `rs:check` and the release publisher so the two boards read as one family.
- Comments and docs explain *why*, for long-term maintenance, rather than restating what the
  code does.

## TODO

Known documentation gaps, recorded rather than quietly left out:

- **Cargo feature matrix.** There is no single statement of which feature combinations must
  compile and pass — `fast`, `portable`, `mmap`, `download`, `net`, `icu-segmenter`,
  `lean-embed`, `gemmology`, `threads`, `gemm-threads`. The knowledge is currently spread across
  the user-facing table in `crates/fxtranslate/README.md`, ad-hoc `cargo test --features` lines
  in the CI workflow, and the annotated dependency block in `crates/fxtranslate-gecko/Cargo.toml`.
- **The Gecko binding has no prose documentation** beyond the comments in its `Cargo.toml` — in
  particular, nothing describes building or testing it against a real Gecko tree.
- **Model triple layout.** `task rs:download-model` writes `data/models/<src><trg>/`, but the
  naming conventions inside it — shared versus split vocabularies, the `lex.*.s2t.bin` shortlist
  — are only described incidentally, in the published package READMEs.
