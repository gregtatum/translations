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
Q8_0 quant set, F16 `token_embd`, baked sinusoidal positional encoding). Change only
the **packaging** so llama.cpp — not our code — can load and tokenize it:

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
  - **Encoder** (post-norm transformer, sinusoidal positional encoding, tied embed): model on
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

## M3 results — perf in the shared harness (measured)

Machine: Apple Silicon, 18 logical cores (6 performance + 12 efficiency). CPU-only build
(`GGML_METAL=OFF`, `GGML_BLAS=OFF`, `GGML_NATIVE=OFF`); OpenMP was absent so llama.cpp uses its
own ggml threadpool. Driver: `ggml/marian_llama_blockbench.cpp` (mirrors G1's `blockbench`
byte-for-byte; same pretokenized source ids from `pretokenize.py`; `--llama` row in
`final_comparison.py`). Q8_0 GGUF (`marian-llama.q8_0.gguf`). Numbers are medians of the clean
sweep (no concurrent RSS sampling — the truer compute figure).

### Full table, 1 thread (all engines, `final_comparison.py`)

| engine                 | words/s | tokens/s | translate s | init ms | peak MiB |
|------------------------|--------:|---------:|------------:|--------:|---------:|
| ONNX ORT (int8, 1t)    |    2398 |     3363 |        3.97 |     206 |      445 |
| marian block-bench     |    1330 |     1864 |        7.15 |      57 |      298 |
| inference-rs (fast)    |    1280 |     1795 |        7.43 |      67 |      150 |
| **llama.cpp (q8_0)**   |     832 |     1167 |       11.43 |      86 |  **195** |
| ggml G1 (q8_0)         |     758 |     1063 |       12.55 |      52 |      150 |
| Firefox Wasm           |     419 |      567 |       22.85 |     135 |      355 |

At **1 thread llama.cpp beats G1** (832 vs 758 wps, 1.10×) — its scheduler is already better
than G1's hand loop. RSS 195 MiB vs G1 150 MiB (context/KV overhead), still well under ONNX's
445 MiB and half of Firefox's 355.

### Thread sweep — the headline, with the encode/decode split

| threads | llama wps | llama enc s | llama dec s | G1 wps | G1 enc s | G1 dec s |
|--------:|----------:|------------:|------------:|-------:|---------:|---------:|
| 1t      |       751 |        2.14 |       10.55 |    759 |        — |        — |
| 2t      |      1326 |        1.07 |        6.11 |   1246 |        — |        — |
| 3t      |      1639 |        0.78 |        5.02 |      — |        — |        — |
| 4t      |      1819 |        0.62 |        4.61 |   1865 |     1.51 |     3.60 |
| 5t      |      1852 |        0.54 |        4.60 |      — |        — |        — |
| 6t      |      1775 |        0.50 |        4.86 |   2178 |     1.14 |     3.22 |
| 8t      |      1665 |        0.53 |        5.19 |   2289 |        — |        — |

**Gate: FAIL (narrowly).** The 2003 wps/4t bar isn't met — llama.cpp does **1819 wps at 4t**
(vs G1's 1865) and **peaks at ~1852 wps (5t)**, then *regresses* (1775 @ 6t, 1665 @ 8t). G1
keeps climbing to 2178 @ 6t / 2289 @ 8t.

**Diagnosed bottleneck — the decoder, and it's a driver/runtime artifact, not the arch math:**
- llama.cpp's **encoder is 2.4× faster than G1's** at 4t (0.62s vs 1.51s): it runs the whole
  source sequence as one `llama_encode` graph, where the G1 driver rebuilds a fresh encoder
  graph per sentence. Clear win for the shipping runtime.
- llama.cpp's **decoder is the drag** (4.61s vs G1's 3.60s at 4t) and **stops scaling past ~4-5
  threads**, then regresses. Root cause: the two-phase driver is **single-sequence, one token
  per `llama_decode`** — ~13k decode calls, each paying llama.cpp's fixed per-decode overhead
  (batch build/free, graph reservation, output-buffer plumbing, threadpool fan-out/join). At
  m=1 the actual matmul is tiny, so that fixed cost dominates and the threadpool spends more
  time synchronizing than computing; past 6 threads it spills onto the E-cores and net
  throughput drops. G1's hand loop has none of that per-step wrapping and keeps scaling.

**This is fixable and not inherent to `LLM_ARCH_MARIAN`:** the levers are batching multiple
sentences per decode (G1 found batching only ~+5% on this per-sentence corpus, but it directly
amortizes llama.cpp's per-decode overhead, which is exactly what's hurting here) and reusing a
persistent batch/graph across steps instead of `llama_batch_init`/`free` per token. The M2
agent's flagged per-sentence encoder-output host copy is *not* the bottleneck — the encoder is
llama.cpp's strong suit here.

### M4 read (upstreamability as a property — NOT a plan to upstream)

**Guardrail: do NOT open a pull request against llama.cpp.** Whether the arch is ever
upstreamed is a human decision for Greg/the team to make and act on directly; nothing in this
work should submit, draft-for-submission, or otherwise action an upstream PR. The notes below
assess *how upstreamable the change is* as a maintenance-model data point for that decision —
they are not a step toward opening one.

On that assessment: perf is **neutral-to-mildly-positive**, not a blocker. At single thread
llama.cpp already beats G1, the encoder is materially faster, and memory (195 MiB) stays a
third of ONNX and half of Firefox — the memory story is a genuine selling point. The 4t gap is
a **greedy-driver limitation** (one-token single-sequence decode), not an arch cost, and lives
in Gecko's `LlamaRunner` two-phase glue rather than in the arch code itself. The honest
framing: the arch is competitive and memory-lean today; closing the last ~10% to G1's thread
scaling is decoder-driver batching work, deferrable and orthogonal to the arch code.

## M3b — block-batched decode: tried, measured, REGRESSED (do not ship)

Hypothesis (from M3): the single-sequence one-token-per-`llama_decode` driver pays llama.cpp's
fixed per-decode overhead ~13k times, so batching B sentences per decode step should amortize
it and close the gap to G1. Built it end to end, validated correctness, measured — and it is a
**large regression, not a win**. Root cause is architectural, and worth writing down so nobody
re-attempts the flat-stash version.

### What was built
- **Arch (llama.cpp `marian.cpp`), minimal + correct:** added per-sequence masking so a block
  of sentences can share one enc-dec pass.
  - Encoder self-attention now takes the block-diagonal `build_attn_inp_no_cache` mask, so a
    packed multi-sentence `llama_encode` never lets one sentence attend to another.
  - Decoder cross-attention now takes the `build_attn_inp_cross` mask (0 / -inf from
    `cross->seq_ids_enc`, the T5 mechanism) so each decoder token attends only to its own
    source rows. Relaxed the `!equal_seqs()` assert in `llm_graph_input_attn_cross::set_input`
    (its fill is per-token, correct for the recurrent equal-seqs split).
  - Both masks are all-zero for a single sequence, so **M1/M2 stay bit-identical** (re-ran:
    encoder 1.0e-6, decoder Gate 1/2 PASS, id-identical to G1).
- **Driver (`marian_llama_blockbench.cpp`):** one packed `llama_encode` per block (each
  sentence its own `seq_id`), then lockstep greedy `llama_decode` over live sequences, retiring
  on EOS/length. `n_seq_max` sized to the largest block.

### Correctness — PASS (this part is genuinely good)
- **Float, block-batched == solo == G1 float: byte-identical on all 403 frankenstein sentences
  (chrF 100.00).** The masked multi-sequence enc-dec math is exactly right — batch-invariant in
  exact arithmetic.
- Q8_0 block-batched vs Q8_0 solo = 93.3% exact-id (chrF vs G1 float 97.72). NOT a batching
  bug: B=1-batched vs solo is **100%** id-identical at Q8_0, so the divergence is purely
  llama.cpp's Q8_0 GEMM taking a different (wider-matrix) accumulation path at m>1 — ~1-ULP
  logit noise that occasionally flips a greedy argmax on a near-tie, then the recurrence
  amplifies it. The documented "explained float divergence" class, negligible chrF cost.

### Perf — FAIL, and worse than M3 single-sequence at every thread count
Q8_0, frankenstein blocks (103 blocks, up to 13 sentences/block), median of 3:

| threads | batched wps | batched dec s | M3 single-seq wps | G1 wps |
|--------:|------------:|--------------:|------------------:|-------:|
| 1t      |         302 |         28.88 |               751 |    759 |
| 2t      |         540 |         16.24 |              1326 |   1246 |
| 4t      |         873 |         10.09 |              1819 |   1865 |
| 5t      |         945 |          9.34 |              1852 |      — |
| 6t      |         990 |          9.02 |              1775 |   2178 |
| 8t      |        1044 |          8.59 |              1665 |   2289 |

Batched **peaks at 1044 wps (8t)** — roughly **half** the single-sequence peak (1852) and
**~0.52× the 2003 wps gate**. Decode time roughly doubled.

### Why — B² cross-attention from the flat encoder stash (the load-bearing finding)
Isolated the largest block (13 sentences, 564 total src tokens), 4 threads:
- **B=13 batched: 783.6 ms decode.** B=1-each (same sentences solo): **186.3 ms** summed.
  **4.2× slower batched.**

llama.cpp stashes the encoder output as ONE flat `cross.v_embd` of shape
`[n_embd, n_enc_total]`, where `n_enc_total` = the *concatenation* of every sentence's source
tokens in the block (~564 here). Cross-attention builds K/V from that whole flat stash, so each
of the B lockstep decoder tokens computes scores against **all** `n_enc_total` keys and then the
per-sequence mask zeroes out the ~B-1/B of them that belong to other sentences. Work scales as
`n_enc_total × B ≈ (mean_len × B) × B = B²·mean_len`, versus solo's `mean_len × 1` per step.
For a block of B short sentences that is ~B× the cross-attention FLOPs, all thrown away by the
mask. The SSRU/FFN parts do batch cleanly (B× weight reuse, the intended amortization), but
cross-attention's B² term swamps that win on this corpus (short sentences, big vocab).

This is exactly the shape G1 avoids: G1's `cross_attention_batched` carries per-row K/V
`[dim, Smax, B]` where `Smax` is the block's *max* source length (~24, not the ~564 sum) and
packs each row's OWN source, so its cross-attn is `Smax×B`, never `n_enc_total×B`. llama.cpp's
single-stash enc-dec design has no equivalent — the flat stash is baked into `llama_cross` /
`build_inp_cross_embd`, shared with T5. Making llama.cpp per-sequence-scope the cross K/V (a
`[dim, Smax, B]` gather + a real ragged mask) is a **substantial change to shared enc-dec graph
infrastructure**, not a driver tweak — and precisely the "STOP and report" boundary in the M3b
brief.

### Verdict / what to keep
- **KEEP the arch masking change.** It is correct, bit-identical for single-sequence, and is the
  prerequisite for ANY future correct multi-sequence enc-dec (it's also just more-correct than
  the unmasked M2 graph). Low risk, upstream-friendly.
- **REVERT the driver to single-sequence** as the shipping/benchmark path — M3's 1852-peak
  numbers stand as the llama.cpp result. Block batching via the flat stash is a pessimization on
  this workload; don't ship it, don't re-attempt it without first fixing the stash to per-seq
  `Smax`-scoped K/V.
- **M4 / Gecko `LlamaRunner`:** do NOT batch sentences into one enc-dec pass on stock llama.cpp.
  The per-decode overhead M3 diagnosed is real but batching to amortize it costs more than it
  saves because of the B² cross-attn. The right levers remain (a) reusing a persistent
  batch/graph across greedy steps instead of `llama_batch_init`/`free` per token, and (b)
  thread count — not multi-sequence batching. Multi-sequence batching only pays once llama.cpp's
  enc-dec stash is per-sequence-scoped, which is upstream infra work.
