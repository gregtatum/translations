# G2 — `LLM_ARCH_MARIAN` in llama.cpp (the product path)

Spec for **G2** of the ggml/llama.cpp port evaluation. Builds on `notes/14` (feasibility
survey), `notes/18` (G1 bare-libggml, built + validated), and the ONNX sibling
(`notes/15/16`). G1 answered "can ggml run this and how fast" (yes; threads to 1.6×
inference-rs / 1.5× marian at a third of ONNX's memory). **G2 answers the actual product
question: can the model run on the *shipping* llama.cpp runtime so we stop carrying a bespoke
engine?**

## What G2 is — and is not

G2 is **C++ architecture code in llama.cpp** (a new `LLM_ARCH_MARIAN`) plus a **converter
revision**, so the model is built and run by the shipping runtime (`llama-cli`, and Gecko's
`LlamaRunner`) rather than by our hand-written G1 graph + driver.

- **Not a "final GGUF build."** The GGUF is a shared weights container that *both* G1 and G2
  consume. G2 changes where the *graph/runtime code* lives, not the weights format. The
  converter is *revised* for G2 (naming + embedded vocab, below), but the GGUF is not the
  deliverable that defines G2.
- **Only route that reduces runtimes.** Upstreamed (route A), marginal maintenance → ~0;
  carried as a patch (route C), a rebase burden but still one fewer engine than shipping
  `inference-rs` alongside llama.cpp. G1/route-B stay bespoke and don't reduce runtimes.

**What G2 inherits for free** (why it's worth the integration cost): batching, threading,
graph scheduling, recurrent/KV state management, and sampling all come from llama.cpp. G1's
~2.5–3× CPU thread scaling (notes/18) is a preview of what the runtime gives without us
writing any of it. So G2 deletes the G1 driver glue rather than porting it.

## Scope / non-goals (inherited from notes/14, unchanged)

CPU-first (Metal off — confirmed a non-goal). Full-vocab, shortlist off (matches
production). Re-quantize from the float student to Q8_0; **no retrain**. Bit-exactness with
the marian oracle is explicitly given up; validate by numeric parity to the G1 engine + chrF.

## Deliverable 1 — converter revision (GGUF for llama.cpp)

Reuse `ggml/convert_marian_gguf.py`'s proven math (weight orientation `[in,out]→ggml ne0=in`,
Q8_0 quant set, F16 `token_embd`, baked sinusoidal PE). Change only the **packaging** so
llama.cpp — not our code — can load and tokenize it:

1. **Tensor names → llama.cpp conventions.** Emit the `LLM_TENSOR_*` names the new arch
   registers (not our self-describing `enc.0.self.wq` scheme). Model on T5's enc/dec tensor
   set + new SSRU tensors.
2. **Hparams under standard KV** (`*.embedding_length`, `*.attention.head_count`,
   `*.block_count` for enc + `*.decoder.block_count`, `*.feed_forward_length`, …) plus
   arch-specific keys (SSRU decoder cell, tied-embeddings-all, sinusoidal-pos, post-norm).
3. **Embed the SPM vocab in GGUF vocab format** — `tokenizer.ggml.model` (unigram/spm),
   token list + scores + types, and the Marian **`precompiled_charsmap`/normalization** if
   present — so llama.cpp tokenizes natively. Today G1 keeps tokenization in Python and the
   binary takes ids; G2 must move it into the GGUF. **This embedding is exactly what the
   tokenizer-parity gate tests, and it is the highest-risk part of G2.**
4. Author with `gguf-py`, registered like `convert_hf_to_gguf.py`'s model classes.

## Deliverable 2 — `LLM_ARCH_MARIAN` in llama.cpp

Integration surface, verified against `~/dev/llama.cpp` @ `af97976c` (re-verify before
coding — line numbers drift):

- **`src/llama-arch.h`** — add `LLM_ARCH_MARIAN` to `enum llm_arch` (near `LLM_ARCH_T5`,
  line ~91; `LLM_ARCH_JAMBA` ~69, `LLM_ARCH_RWKV6` ~101 are the recurrent neighbors).
- **`src/llama-arch.cpp`** — `LLM_ARCH_NAMES` entry; `LLM_TENSOR_NAMES` for the enc/dec/SSRU
  tensors; `LLM_TENSOR_INFOS` op kinds; **add `MARIAN` to `llm_arch_is_recurrent()`**
  (line 943).
- **`src/models/marian.cpp`** (new) — the model class + graph:
  - **Encoder** (post-norm transformer, sinusoidal PE, tied embed): model on
    `src/models/t5.cpp`'s encoder graph (`LLM_GRAPH_TYPE_ENCODER`, stateless).
  - **Decoder step**: SSRU (elementwise `sigmoid/sub/mul/add/relu` — the exact ops proven in
    G1's `build_decoder`) over **recurrent memory**, then **cross-attention** to encoder
    output (T5's `build_attn_inp_cross` / `build_inp_cross_embd`), then FFN. Post-norm
    throughout.
  - **SSRU state slot** = `llama-memory-recurrent` `get_s_l(il)` per layer, one `[d]` vector
    per layer per sequence, written back with `ggml_cpy` — the Jamba/RWKV pattern
    (`src/models/jamba.cpp` exists as the recurrent-in-decoder precedent).
- **`src/llama-model.cpp`** — register load/build; add `MARIAN` to `llama_model_has_encoder()`
  (line 2746, T5 case at 2748) **and** to the recurrent branch of `create_memory()`
  (line 2051; recurrent selected at 2151, hybrid at 2161). **Verified: `create_memory` picks
  recurrent independently of `has_encoder`, and no assertion forbids a model that is both** —
  so the encoder-decoder + recurrent combination is allowed. The **novelty/risk is the
  wiring, not a blocker**: T5 = cross-attn + KV-cache; Jamba = recurrent, no encoder; nothing
  yet combines cross-attention *with* recurrent memory.
- **Two-phase API** — `llama_encode` (`include/llama.h:960`) runs the encoder once;
  `llama_decode` (`:976`) runs each decode step. T5 already wires this path.

## Deliverable 3 — driver / Gecko `LlamaRunner`

1. First, a minimal driver (or `llama-cli`) to bench + validate the arch in isolation.
2. Then Gecko: `toolkit/components/ml` → `LlamaCppPipeline.mjs` → native `LlamaRunner` is
   today a **decoder-only text-generation** path. Riding it for translation needs
   `LlamaRunner` taught the `llama_encode`→`llama_decode` two-phase call — Gecko-side work
   *on top of* the arch (flagged in notes/14). Scope this separately from the arch PR.

## Gates (sharpened from notes/14; transitive to G1 + the marian oracle)

1. **Numeric parity vs the G1 engine.** G1 is oracle-validated (notes/18 Gate 1/2). Diff
   llama.cpp's float graph against G1's encoder context + first-step logits (~1e-4), then
   Q8_0 top-K. Reuse `ggml/testdata/` + the numpy golden. **Batch-invariance is free** —
   llama.cpp owns batching.
2. **Tokenizer parity (HARD gate — do FIRST).** llama.cpp's embedded SPM vs `spm.rs` over a
   corpus, token-diff to zero. Marian normalization / `precompiled_charsmap` is the risk that
   most likely bites; a divergence silently changes translations. Fail fast here before
   writing arch code.
3. **Perf, full-vocab, threaded.** Should meet or beat G1's 4-thread 2003 wps in the same
   `final_comparison.py` harness (llama.cpp scheduling ≥ our hand loop). Add a `llama.cpp`
   row next to `--ggml`/`--onnx`.
4. **Upstreamability.** Gauge OPUS-MT/Marian PR appetite → route A (upstream, ~0 maintenance)
   vs route C (carry patch, rebase burden).

## Validation reuse

`ggml/testdata/` (inference-rs int8 + numpy golden), the fixed sentence, the chrF/quality
corpus, and `final_comparison.py` all carry over. Add a thin llama.cpp translate wrapper that
emits target ids so the existing id-level diff harness applies unchanged.

## Milestones (cheapest-first; fail fast on the hard gate)

- **M0 — converter revision + tokenizer-parity gate.** The hard gate; if Marian SPM can't be
  reproduced by llama.cpp's loader, G2 stalls regardless of the arch. Do this before arch code.
- **M1 — encoder-only arch** + numeric parity vs the G1 encoder.
- **M2 — SSRU decoder + cross-attn + recurrent memory**, end-to-end greedy; numeric + chrF gates.
- **M3 — perf** (threaded) vs G1 / inference-rs / ONNX in the shared harness.
- **M4 — upstream PR (route A) or carry-patch (route C) decision;** then Gecko `LlamaRunner`
  two-phase.

## Risks

- **Tokenizer parity** (`precompiled_charsmap` / Marian normalization) — highest; gated M0.
- **cross-attn + recurrent wiring** — novel in llama.cpp; pieces exist (T5 + Jamba), the
  combination is new. Integration risk, not a math risk (G1 proved the math).
- **Upstream appetite** — if route A stalls, route C's rebase burden is the fallback cost.
- **`LlamaRunner` is decoder-only text-gen** — the encode/decode two-phase is Gecko work
  atop the arch, not part of the upstream PR.

## Effort

Roughly a T5-sized model class (~350–400 lines) + the simpler SSRU + the converter revision +
the tokenizer work. Weeks, **dominated by tokenizer parity and the upstream cycle, not the
math** — G1 already de-risked the graph and quantization. If M0 (tokenizer) and M3 (perf)
pass, G2 is the consolidation win; if the upstream cycle is slow, route C bridges.
