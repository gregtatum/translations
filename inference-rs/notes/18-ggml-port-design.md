# Porting the translation models to ggml/llama.cpp — design + G1 results

Companion to `notes/14-ggml-llamacpp-port.md` (the original feasibility survey) and a
sibling to the ONNX evaluation (`notes/15`, `notes/16`). This note is the *apples-to-apples*
build: a bare-libggml engine that runs the Bergamot student model on ggml so its speed,
memory, and quality sit in the same `final_comparison.py` table as inference-rs, marian, and
ONNX. Built and validated 2026-08-03.

The question this feeds: **maintenance burden of three consolidation options** — keep the
bespoke memory-safe Rust engine (`inference-rs`), reuse ONNX Runtime (already in Gecko via
transformers.js), or ride llama.cpp/ggml (already a shipping Gecko backend). This note makes
the ggml leg measurable.

## The core asymmetry vs the ONNX eval (read this first)

The ONNX eval could be a clean-room Python converter driving *stock* ORT because **an ONNX
graph is data**: you compose primitives (`MatMul`, `Sigmoid`, `Gather`, …) with
`onnx.helper`, hand the file to an unmodified runtime, and it executes — the SSRU is just
more nodes plus a host loop. No fork.

**ggml has no equivalent.** GGUF is a weights+metadata container; the *graph* is C++
compiled into llama.cpp and selected by an `enum llm_arch`. There is no path where you emit
a file and an unmodified shipping binary runs your architecture. Every ggml route writes
graph code in C/C++ (or via FFI). This is not an inconvenience — it *is* the maintenance
axis being compared: "how much code must live in the runtime."

So the ggml eval has three measurement subjects of increasing product fidelity, not one:

| | what it is | measures | ONNX analog |
|---|---|---|---|
| **G0** | swap only inference-rs's GEMM to a libggml cgraph; keep the Rust loop | is ggml's Q8_0 ARM GEMM competitive with gemmology i8mm | — (kernel bake-off) |
| **G1** | bare-libggml binary: own encoder+decoder cgraph + driver, GGUF weights | ggml runtime end-to-end, fair standalone RSS | route B (clean-room graph, near-stock runtime lib) |
| **G2** | real `LLM_ARCH_MARIAN` in patched/upstreamed llama.cpp + driver | the shipping runtime's process, tokenizer parity, upstream/patch burden | route A (heavy, product fidelity) |

The ONNX eval sits between G1 and G2 (clean-room graph on a *stock* runtime). **G1 is the
closest honest analog and the built subject here.** G2's cost *is* the maintenance burden
under evaluation — you pay it to ship, not to get a number.

Decision (confirmed with Greg): build **G1 first, Q8_0**, Rust/Python driving a compiled
C++ ggml engine. Proceed to G2 only if G1 clears the perf/quality gates.

## What was built (`inference-rs/ggml/`, driven by `rs:ggml-*` tasks)

A one-for-one mirror of `onnx/`, reusing the ONNX eval's validated pieces as the single
source of truth for the architecture (so the two evals can't silently disagree on the model):

| file | role | reuse |
|---|---|---|
| `convert_marian_gguf.py` | float `.npz` → `marian.float.gguf` (F32 scaffold) + `marian.q8_0.gguf` (shipped) | imports `onnx/model_npz.py`; bakes PE via the `onnx/numpy_ref.py` formula |
| `marian_ggml.cpp` | the engine: builds the encoder + SSRU-decoder cgraphs on libggml, greedy loop in the driver | — |
| `build.sh` | builds libggml (CPU-only) from `GGML_DIR` + compiles the engine | — |
| `pretokenize.py` | block corpus → source-id blocks for blockbench (fair-RSS: the sampled process is the binary) | `onnx/tokenizer.py` |
| `translate.py` | tokenize → binary `decode` → detokenize (human CLI) | `onnx/tokenizer.py` |
| `validate_ggml.py` | Gate 2 (vs numpy golden) + Gate 1 (vs inference-rs int8) | `onnx/numpy_ref.py`, `onnx/testdata/inferrs_*.f32` |
| `scripts/final_comparison.py --ggml` | adds ggml as a peer row in the shared perf harness | the existing harness |

### Engine design decisions (converter and engine are one codebase, so the tensor contract is ours)

- **Weight orientation.** The `.npz` stores each linear logically `[in, out]`.
  `ggml_mul_mat(W, x)` contracts on `ne0`, so the converter stores `W.T` (numpy `(out, in)`
  → ggml `ne0=in, ne1=out`), giving `y[o] = Σ_i W[i,o]·x[i]`. FFN (non-square) shape-errors
  if this is wrong; the gates catch a transposed square matrix.
- **Tied embedding, emitted twice** — `token_embd.weight` (F16, for the `get_rows` lookup)
  and `output.weight` (Q8_0, for the tied output projection `logits = ggml_mul_mat(Wemb,u)`).
  This is exactly ONNX's split (float Gather, quantized projection) and is why the size
  ratio is capped.
- **Quantization.** Q8_0 (block-wise, per-32 scale) on every linear matmul weight + the
  output projection; F16 embeddings; F32 biases/LayerNorm and the baked PE table.
- **PE baked as an F32 constant** via the exact `numpy_ref` rotor formula — never a live
  `Sin` (the known marian-exporter divergence). F32 so it adds cleanly to F32 activations.
- **SSRU is elementwise per step**, exactly as `notes/14` predicted: `c_t = g·c_prev +
  (1−g)·cand`, implemented as `cand + g⊙(c_prev − cand)` with `ggml_sigmoid/sub/mul/add` and
  `ggml_relu`. No `ggml_ssm_scan`, no custom op — decode is one token at a time. State is one
  `[dim]` vector per layer, threaded in the driver (read `state_out`, feed `state_in`).
- **Cross-attention K/V folded once per sentence** in the encoder graph, pulled to host,
  fed to the decode graph — the T5 pattern, done by hand.
- **Decode graph rebuilt per token.** A single-token step is tiny next to the matmuls, and
  rebuilding sidesteps a gallocr input/scratch aliasing bug that corrupted the SSRU state
  across reused computes (first token was correct, then it diverged into garbage). A
  cross-step graph-reuse optimization is a named follow-up.
- **Tokenization stays in Python** (the shared SPM); the binary speaks ids in / ids out.
  This keeps tokenization identical across every engine and defers llama.cpp SPM parity to
  the G2 tokenizer gate.

## Validation — both gates PASS (`task rs:ggml-validate`)

Transitively cheat-proof against the marian oracle, per the project's validation doctrine.

**Gate 2 — FLOAT ggml vs `numpy_ref` golden** (graph correctness; must be ~1e-4):
encoder abs max **7.3e-7**, logits abs max **1.9e-5**, first-step argmax agrees (273). The
graph is exact in F32. (An early all-F16 float file was off by ~0.27 mean — pure F16
accumulation over the depth, which is *why* the shipped path is Q8_0, not F16, for matmuls.)

**Gate 1 — Q8_0 ggml vs inference-rs intgemm int8** (int8 sanity; mutual top-K argmax):
first-step argmax ggml=273 sits in the intgemm top-5 and vice-versa (intgemm argmax=5051) —
the classic int8 near-tie swap. **Finding:** the ~11% ggml-vs-intgemm encoder divergence is
intgemm being the noisy one — Q8_0 is **2.76%** from the float golden while intgemm is
**10.59%**. Q8_0 (block-wise) tracks float ~4× closer than intgemm (per-tensor, shifted),
corroborating `notes/14`'s hypothesis and the ONNX finding that block/QDQ int8 beats
intgemm on fidelity. End-to-end the Q8_0 translation of the fixed sentence is coherent and
arguably better than the float greedy output (`здравствуйте, мир…` vs truncated `здрав…`).

## Perf/memory — the apples-to-apples number (`task rs:ggml-perf`)

en-ru **base v3.0** (dim 512, ffn 2048, 6-enc/2-dec SSRU), Frankenstein blocks (103 blocks,
9513 words), single-thread, shortlist off, model load excluded, RSS sampled every 20 ms:

| engine | words/s | tokens/s | translate s | settled MiB | peak MiB |
|---|---|---|---|---|---|
| marian block-bench (native) | 1332 | 1868 | 7.15 | 298 | 298 |
| inference-rs (rust, fast) | 1285 | 1801 | 7.41 | 129 | 148 |
| **ggml (q8_0, 1t)** | **744** | **1043** | **12.79** | **119** | **119** |
| Firefox Wasm (Full-Page) | 419 | 567 | 22.85 | 355 | 355 |

- ggml is **0.58× inference-rs / 0.56× marian / 1.78× Firefox Wasm** on speed, at the
  **lowest memory of any engine (119 MiB)** — the fair-RSS win a compiled binary buys (vs
  ONNX's 2357 wps but 444 MiB whole-Python-process from `notes/16`).
- The 744 wps is a **floor, not a ceiling**: it carries two un-optimized handicaps the
  batched rs/marian rows don't — (1) the decode graph is rebuilt every token (no cross-step
  reuse), (2) the encoder runs once per sentence (not block-batched). Both are known levers.

So the three-way tradeoff is now on one page: **ONNX** = fastest but heaviest memory and its
SSRU glue lives in transformers.js JS; **ggml** = leanest memory, currently slowest but with
clear decode-glue headroom, and its SSRU glue lives in C++ arch code; **inference-rs** =
balanced and already tuned, but a bespoke engine to carry.

## Feasibility verdict (the encoder vs the SSRU decoder — Greg's original worry)

- **Encoder: zero risk.** Standard post-norm transformer; every op is stock ggml. The only
  non-vanilla bits are graph-wiring (post-norm order, baked PE, `sqrt(d)` scale on a tied
  `Wemb`). Proven exact in Gate 2.
- **SSRU decoder: the feared part was a false alarm.** Elementwise per step, no scan, no
  custom op. Proven exact in Gate 2 and coherent in Q8_0. The state slot is one vector per
  layer, which in G2 is exactly what `llama-memory-recurrent` stores.

The real cost was never the math — it's the G2 integration surface (see `notes/14` §"three
gates"): a novel `LLM_ARCH_MARIAN` combining cross-attention with recurrent memory (the
pieces exist in T5 + Jamba/RWKV but nothing combines them), SPM tokenizer parity, and
upstreamability.

## Optimization pass (2026-08-03, after the first G1 measurement)

Started from 744 wps single-thread and the two named handicaps. Measured, not guessed
(notes/17 doctrine): the span split is **38% encoder / 62% decoder**, and a bandwidth
budget shows decode at m=1 is **~90% weight-streaming compute** (the 512×32000 Q8_0
projection is ~17.5 MB streamed per token → ~0.35 ms of 0.567 ms/token), ~10% glue. So the
per-token graph rebuild is not the cost; streaming weights to make one token at a time is.

**Decoder batching (the predicted big lever) — implemented, gated, but only ~+5% here.**
Added a batched lockstep decoder (leading batch dim, `[dim,Smax,B]` cross K/V + a shared
additive `cross_bias` mask, per-row eos/length retirement), gated by **batch-invariance**
(Gate 3: a B=5 mixed-length batch is token-identical to B=1 — `decode` vs `decode solo`).

- Naive full-batch lockstep was a **regression** (744→550 wps, +41 MiB): a block runs to its
  *longest* sentence, so ragged lengths waste compute on already-retired rows, and at B≈13
  the 32000-vocab projection tips compute-bound (losing the bandwidth amortization batching
  gives ORT).
- A controlled **uniform-length** batch (B=8, equal length) *does* win **1.38×/token**
  (0.501→0.364 ms) — the amortization is real; ragged waste was the whole regression.
- Fixed with **row compaction** (retired rows drop out each step; total row-steps == Σ
  per-sentence lengths). Net **779 wps (+5% over per-sentence), 150 MiB**. The win is small
  *because the corpus is per-sentence-shaped* — many blocks are 1 sentence (B=1, no benefit),
  and the encoder (38%) is still per-sentence. This matches Greg's prior that batching isn't
  large for this workload; the 1.38× is only there for uniform multi-sentence traffic.

**Threads — the real lever (~2.5–3×), and a genuine ggml advantage.**

| threads | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| wps | 792 | 1258 | 1964 | 2354 | 2349 |

ggml scales cleanly because the m=1 projection is bandwidth-bound and Apple Silicon adds
bandwidth per core, and ggml parallelizes the big matmul — where notes/16 found ORT threads
added only ~9%. **At 4 threads ggml is 2003 wps / 150 MiB — 1.6× inference-rs and 1.5× marian
(both single-thread by design), still a third of ONNX's memory:**

| engine | words/s | settled MiB |
|---|---|---|
| ggml (q8_0, 4t) | **2003** | 150 |
| marian native (1t) | 1317 | 298 |
| inference-rs (1t) | 1252 | 130 |
| Firefox Wasm | 419 | 355 |

**Revised conclusion:** ggml is *not* slower — single-thread it's 0.6× inference-rs, but it
threads to 1.5–1.8× using cores inference-rs doesn't touch, at a third of ONNX's memory.
For the maintenance question this is the key point: riding llama.cpp/ggml gives
multi-threading (and batching, scheduling, state management) for free, whereas matching it in
inference-rs is unstarted threading work (notes/11). The eval's headline rows stay
single-thread for fairness; the threaded row is "what the runtime actually does."

## Recommended next steps

Decoder batching (with compaction) and threading are **done**; the remaining levers:

1. **Kernel bake-off (G0)** — the clean ceiling question now that glue is handled: microbench
   ggml's Q8_0 ARM matmul vs gemmology i8mm at the decode shapes (m=1…B, k=512, n=32000/2048/
   512), the analog of `rs:kernel-micro`. The 1t gap to inference-rs (0.6×) should be mostly
   kernel + the per-sentence encoder; this isolates it.
2. **Batch the encoder** (38% of time, still per-sentence) — padded + masked self-attention
   across the block, same batch-invariance gate. Bounded upside on this per-sentence corpus.
3. **Cross-step graph reuse** (~10% of decode) — `ggml_backend_sched` to drop the per-token
   rebuild and the per-step cross-K/V re-gather. Cheap, modest.
4. **Quality at scale**: chrF/token-overlap of ggml-Q8_0 vs float and vs inference-rs
   (a `quality_ggml.py`, mirroring `onnx/quality.py`).
5. **Only if the numbers justify consolidation → G2**: `LLM_ARCH_MARIAN` in a patched
   llama.cpp + a driver, then the SPM-parity and upstream-appetite gates from `notes/14`.
   Note G2 inherits batching/threading/scheduling from the runtime — so the threading result
   above is a preview of what G2 gives for free.

## Reproduce

```sh
task rs:ggml-build          # libggml (CPU-only) from GGML_DIR=~/dev/ggml + compile engine
task rs:ggml-convert        # float .npz -> marian.{float,q8_0}.gguf
task rs:onnx-dump           # (optional) inference-rs int8 reference for Gate 1
task rs:ggml-validate       # Gate 2 + Gate 1
task rs:ggml-translate -- "Hello, world."
task rs:ggml-perf           # 4-way table (add --onnx for the full 5-way)
```
