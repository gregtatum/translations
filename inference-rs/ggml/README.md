# Bare-libggml translation engine (G1 of the ggml/llama.cpp port eval)

A from-scratch converter + compiled ggml engine that runs the Bergamot/Marian student model
on **libggml**, so its speed/memory/quality sit in the same `final_comparison.py` table as
`inference-rs`, marian, and ONNX. This is **G1** — a bare-libggml binary (own encoder+decoder
`ggml_cgraph` + host greedy loop), the closest honest analog to the ONNX eval's clean-room
route. It is *not* a full `LLM_ARCH_MARIAN` in llama.cpp (that is G2, the product path). See
`../notes/18-ggml-port-design.md` for the full design, gates, and results, and `SPEC.md` in
`../onnx/` for the architecture (the model is identical).

## Why a compiled binary

ggml has no "stock runtime runs my clean-room graph" path the way ONNX does — the graph is
C++. Making it a compiled binary (not Python ggml bindings) means its RSS is just ggml
arenas + weights, a *fair* memory peer to the native tools (and far below ONNX's whole-Python
process). Tokenization stays in Python (the shared SPM); the binary speaks source ids in /
target ids out.

## Pipeline

```
float .npz  →  convert_marian_gguf.py  →  marian.float.gguf (F32 scaffold)  ─ Gate 2 vs numpy_ref
                                       →  marian.q8_0.gguf  (shipped)        ─ Gate 1 vs inference-rs int8
                                                                             ─ perf/mem vs all engines
```

## Files

- `convert_marian_gguf.py` — float `.npz` → GGUF (reuses `../onnx/model_npz.py`; bakes
  the positional-encoding table with the `../onnx/numpy_ref.py` formula). Writes an F32
  float scaffold and a Q8_0 artifact; prints the size table.
- `marian_ggml.cpp` — the engine. Modes: `decode` (ids in/out), `blockbench --blocks F`
  (emits `[block] {json}` spans like the other engines), `dump ID…` (encoder context +
  first-step logits → `testdata/`, for the gates).
- `build.sh` — builds libggml (CPU-only) from `GGML_DIR` (default `~/src/ggml`) + compiles
  the engine. Run via `task rs:ggml-build`.
- `pretokenize.py` — block corpus → source-id blocks for `blockbench` (keeps the sampled
  process the binary alone).
- `translate.py` — human CLI: tokenize → binary `decode` → detokenize.
- `validate_ggml.py` — Gate 2 (float vs numpy golden) + Gate 1 (Q8_0 vs inference-rs int8).

## Run

```sh
task rs:ggml-build                    # libggml + engine
task rs:ggml-convert                  # -> models/marian.{float,q8_0}.gguf
task rs:onnx-dump                     # (optional) inference-rs int8 reference for Gate 1
task rs:ggml-validate                 # Gate 2 + Gate 1
task rs:ggml-translate -- "Hello, world."
task rs:ggml-translate -- --float "Hello, world."
task rs:ggml-perf                     # apples-to-apples table (add --onnx for the 5-way)
```

`GGML_DIR` overrides the ggml checkout location (a sibling checkout like `marian-dev` /
`gemmology`).

## Results (en-ru base v3.0, 1 thread; see notes/18 for full detail)

**Gates.** Gate 2 PASS — float ggml vs numpy golden: encoder abs max 7.3e-7, logits 1.9e-5.
Gate 1 PASS — Q8_0 vs inference-rs int8: mutual top-5 argmax (int8 near-tie swap). Q8_0 sits
**2.76%** from the float golden vs intgemm's **10.59%** — block-wise Q8_0 is ~4× closer to
float than intgemm shifted-int8.

**Perf/memory** (Frankenstein blocks, model load excluded). The decoder is block-batched
with row compaction; the headline eval rows are single-thread for fairness with the native
tools, but ggml threads well and that is its real advantage:

| engine | words/s | settled MiB |
|---|---|---|
| **ggml (q8_0, 4 threads)** | **2003** | 150 |
| marian native (1t) | 1317 | 298 |
| inference-rs (rust, 1t) | 1252 | 130 |
| ggml (q8_0, 1t) | 779 | 150 |
| Firefox Wasm | 419 | 355 |

Single-thread ggml is 0.6× inference-rs; at 4 threads it is **1.6× inference-rs and 1.5×
marian** (both single-thread by design) at a third of ONNX's memory. ggml scales ~2.5–3× to
4–6 threads where ORT gained only ~9% (notes/16). Decoder batching is only ~+5% on this
per-sentence-shaped corpus (1.38×/token for uniform multi-sentence batches); threads are the
lever. See `../notes/18-ggml-port-design.md` for the full optimization pass.
