# Porting the translation models to GGUF / llama.cpp

## Why this note exists

We're evaluating whether the bespoke Marian/Bergamot translation models can run on
**llama.cpp** (with weights in **GGUF** format) instead of the `inference-rs` engine
in this repo.

The motivation is **maintenance reduction, not speed.** The Gecko AI Runtime is
already integrating llama.cpp for other AI features. If translation can ride on that
same runtime, we may not need to ship the bespoke Rust engine (`fxtranslate` crate +
its C ABI + build integration + wasm/bindings + ongoing sync) into Firefox at all.
The question this note answers is: *"What would it take, and is it worth it?"*

Two things are explicitly **off the table** and one is explicitly **on the table**:

- **Not** retraining the models.
- **Not** chasing GPU/Metal. The workload is per-sentence translation with small
  batches; GPU offload is unlikely to beat straight CPU here, and it adds a lot of
  surface. This investigation is **CPU-first**.
- **On the table:** re-quantizing. Quantization is a training-pipeline step, and we
  likely still have the *pre-quantization float student models*. So we can convert
  those to GGUF and let ggml quantize (Q8_0/F16) — we do **not** have to
  reverse-engineer the shipped `intgemm` blob.

## The model, briefly (what we're porting)

Tiny transformer **encoder–decoder** (Bergamot/Marian student):

- Encoder: 6 layers, `d_emb=384`, 8 heads (`dk=48`), `d_ffn=1536`, ReLU FFN,
  **post-norm** residuals, sinusoidal positional encoding, tied embeddings.
- Decoder: **4 layers, and here is the non-standard part** — each layer replaces
  transformer self-attention with an **SSRU** (Simpler Simple Recurrent Unit),
  followed by cross-attention to the encoder and an FFN.
- Vocab: SentencePiece **unigram** (~32k–50k), shared or split (CJK).
- Shipped weights are `intgemm` shifted-int8 (per-tensor `alpha`, B stored
  transposed); today decoded by a `gemmology` i8mm kernel (~0.50× marian).

Reference implementation in this repo:

| Component            | File                                       |
|----------------------|--------------------------------------------|
| Binary model parser  | `crates/fxtranslate/src/model.rs`          |
| Weight loader / GEMM  | `crates/fxtranslate/src/weights.rs`, `gemm.rs` |
| Encoder / decoder    | `crates/fxtranslate/src/engine.rs`         |
| SSRU + ops           | `crates/fxtranslate/src/ops.rs` (`highway`) |
| Tokenizer            | `crates/fxtranslate/src/spm.rs`            |

### The SSRU — and why it's *not* the scary part

Feared as the blocker, it turns out to be the easy part. The recurrence
(`engine.rs` decode step, `ops.rs::highway`) is:

```
cand = W_rnn · u                 # no bias
f    = W_f · u + b_f
c_t  = σ(f) ⊙ c_{t-1} + (1 − σ(f)) ⊙ cand     # highway/forget gate
h_t  = ReLU(c_t)
```

The state `c` is **one vector per decoder layer per sequence**, zero-initialized.

Crucially, **translation decodes one token at a time — there is no decoder
prefill** (the decoder is seeded with EOS and runs autoregressively; it never
processes a multi-token prompt). So the recurrence is never a parallel-in-time
scan. Per step it is plain elementwise `mul/sub/add/sigmoid/relu`.

Consequence for ggml: **no `ggml_ssm_scan`, no custom op, no Metal kernel.** The
SSRU state slot is exactly what `llama-memory-recurrent` already stores. This is
strictly simpler than Mamba/RWKV, which need scan machinery *because* they prefill.

## What llama.cpp already gives us (and what it doesn't)

Explored against `~/dev/llama.cpp` (July 2026). Adding a new architecture is a
known, bounded path:

- **Arch registration:** `src/llama-arch.h` (`enum llm_arch`), `src/llama-arch.cpp`
  (`LLM_ARCH_NAMES`, `LLM_TENSOR_NAMES`, `LLM_TENSOR_INFOS`, and
  `llm_arch_is_recurrent()` at ~line 943).
- **Model class + graph:** `src/models/models.h`, `src/models/<arch>.cpp`,
  registered in `src/llama-model.cpp` (`create_memory()` at ~line 2051 picks
  recurrent vs hybrid vs KV-cache).
- **Encoder–decoder precedent = T5:** `src/models/t5.cpp` — encoder graph
  (`LLM_GRAPH_TYPE_ENCODER`, stateless), decoder graph with cross-attention
  (`build_inp_cross_embd`, `build_attn_inp_cross`), and the two-phase
  `llama_encode` / `llama_decode` API.
- **Recurrent-in-decoder precedent = Jamba / Granite-hybrid / RWKV / Mamba:**
  `src/llama-memory-recurrent.cpp`, `src/models/rwkv6-base.cpp`.
- **GGUF writer:** `gguf-py/gguf/gguf_writer.py`; converters like
  `convert_hf_to_gguf.py`.

**Key finding — the combination is novel but not forbidden.** `create_memory()`
selects the memory type (recurrent / hybrid / KV) independently of
`llama_model_has_encoder()`; there is no assertion blocking a model that is *both*
an encoder-decoder *and* recurrent. But **no existing arch combines
cross-attention with recurrent memory** — T5 has cross-attn with a KV-cache
decoder; Jamba has recurrent state but no encoder/cross-attn. Our decoder wants
recurrent SSRU state **and** cross-attention to encoder output. The pieces all
exist; the *wiring* is new. That's the integration risk — not the math.

## Options considered

- **A. Upstream a real `LLM_ARCH_MARIAN` into llama.cpp.** The only version that
  actually *deletes* maintenance rather than relocating it. Gecko already ships
  llama.cpp; translation rides along; if upstreamed, marginal maintenance → ~0.
- **B. Bare `libggml` bespoke graph** (our own encoder/decoder graphs, keep the
  Rust tokenizer/shortlist/loop). **Rejected for this goal:** it's still a bespoke
  engine to maintain — it does not reduce the number of runtimes, which is the
  whole point.
- **C. Carry `LLM_ARCH_MARIAN` as a patch** on the vendored llama.cpp (if upstream
  won't take it). Fallback to A — smaller than a second engine, but a rebase
  burden.
- **D. Keep `inference-rs`** (status quo) — most duplication if llama.cpp is also
  shipping.

**Direction: pursue A (upstream-first), fall back to C.** B only makes sense if the
goal were control/novelty, which it isn't.

## The three feasibility gates (do these BEFORE writing arch code)

These, not the SSRU, decide whether we can drop the Rust engine. Any one failing
changes the answer to "keep `inference-rs`, at least for now."

1. **CPU perf, full-vocab.** Both runtimes do a full-vocab output projection every
   decode step (`384 × ~32–50k` matmul per token) — **production runs the shortlist
   OFF** (see below), so this is an apples-to-apples comparison, not a regression.
   The open question is simply whether llama.cpp's CPU GEMM (Q8_0, ARM
   i8mm/dotprod kernels) keeps up with the `gemmology` i8mm path (~0.50× marian),
   including per-single-token graph/dispatch overhead. **Spike:** convert one model,
   benchmark stock llama.cpp CPU (Q8_0) vs `inference-rs` (words/sec) on a fixed
   sentence set.
2. **Tokenizer parity.** Marian SPM normalization (`precompiled_charsmap`,
   byte-fallback) may not match llama.cpp's SPM loader exactly. Divergence here
   silently changes translations. **Spike:** extract vocab → GGUF, tokenize a
   corpus with both llama.cpp and `spm.rs`, diff. Hard gate for deleting the Rust
   path.
3. **Upstreamability.** `LLM_ARCH_MARIAN` (encoder + recurrent-SSRU decoder +
   cross-attn) is novel. **Spike:** check for existing OPUS-MT/Marian PRs/issues
   and gauge maintainer appetite. Determines A (→0 maintenance) vs C (patch).

The pre-quant float student artifacts **are** archived and retrievable — see the
next section.

## Where to get the models

Canonical doc: **`artifacts/marian-mac/model-registry.md`** (buckets, registry,
layout, `gsutil` usage). The converter-specific facts that doc doesn't emphasize:

- **Two public (read) buckets:** prod
  `gs://moz-fx-translations-data--303e-prod-translations-data`, dev
  `gs://releng-translations-dev`. Layout is
  `models/{langpair}/{experiment}_{task_group_id}/{model_type}/`.
- Find models via the **registry**, not blind `gsutil ls`:
  <https://mozilla.github.io/translations/model-registry/> (web) or the JSON
  catalog `.../moz-fx-translations-data--303e-prod-translations-data/db/models.json`.
- **Use the float `.npz`, not the shipped `.bin`.** The converter must read the
  *pre-quantization* float checkpoint from the **`student-finetuned/`** dir — the
  output of the `distillation-student-model-finetune` stage — not
  `exported/…intgemm.alphas.bin`. Confirmed edge:
  `taskcluster/kinds/distillation-student-model-quantize/kind.yml:22,113` fetches
  `final.model.npz.best-{best_model}.npz` from finetune and emits
  `model.intgemm.alphas.bin` (`pipeline/quantize/quantize.sh:25`).

  ```
  gs://.../models/{langpair}/{exp}_{tgid}/student-finetuned/
      final.model.npz.best-*.npz          # float weights (Marian .npz = numpy zip)
      vocab.{src}.spm  vocab.{trg}.spm
  ```
  ```bash
  gsutil cp 'gs://.../models/{langpair}/{exp}_{tgid}/student-finetuned/final.model.npz.best-*.npz' ./
  ```
- That float model is **quantization-aware-finetuned** (trained to survive int8),
  so it's the right source to requantize to Q8_0 — not a naive fp32 model.
- The `vocab.spm` (in `student-finetuned/` or `exported/`) feeds the tokenizer
  spike; the `lex.50.50.*.s2t.bin` shortlist lives under `exported/`.

## Build shape (once the gates pass)

1. **Converter** `float Marian → GGUF`: read the `student-finetuned/…best-*.npz`
   (Marian `.npz` = a numpy zip — **not** the `.intgemm.alphas.bin` that `model.rs`
   parses, so this wants a numpy reader → a Python converter using `gguf-py`).
   Remap tensor names (use `model.rs` / `weights.rs` for names+shapes), write
   hparams (enc/dec depth, heads, `d_ffn`, tied-embeds, sinusoidal-pos flag), embed
   the SPM vocab, quantize to Q8_0.
2. **`LLM_ARCH_MARIAN`** in llama.cpp: enum + tensor maps; model class with an
   encoder graph (post-norm, sinusoidal pos) and a decoder graph (cross-attention
   like T5 + **elementwise** SSRU over recurrent memory); mark recurrent; wire
   `llama_encode`/`llama_decode`. Roughly T5-sized (~350 lines) plus the simpler
   SSRU.
3. **Shortlist: out of scope.** Ship full-vocab, matching production (see below).
4. **Validation:** tensor-diff the encoder against `inference-rs`; then chrF/BLEU +
   token-overlap on a real test set vs marian. We are **giving up bit-exactness
   with the marian oracle** for the ggml target (Q8_0 ≠ intgemm shifted-int8 —
   actually block-wise Q8_0 is usually *more* accurate); validate by quality, and
   document the divergence the way the wasm-parity work does.

## The lexical shortlist (out of scope; future speed path)

The shortlist restricts the output projection to a small candidate set per
sentence — a big decode speedup — but **it is turned OFF in production**: it hurt
translation quality. The code is left in place (in `inference-rs` and in the
exported artifacts) as a switch we could flip back on, but flipping it on for real
would require **document-level shortlisting** (building the candidate set across a
document rather than per sentence) to recover the quality, and possibly further
work beyond that.

Implications for this port:

- **Initial scope ships full-vocab**, exactly matching production. llama.cpp having
  no shortlist concept is therefore *not* a regression, and gate #1 is a fair
  full-vocab-vs-full-vocab comparison.
- **Path forward (later):** if document-level shortlisting is ever done, a shortlist
  would be worth adding to the llama.cpp side as a custom output-projection over a
  candidate subset — a self-contained speed optimization, orthogonal to the arch
  work. Treat it as a follow-up with its own quality bar, not a launch blocker.

## Recommended next step

Run the spikes, cheapest-first. Concrete starting point: **spike #1** — write the
`float → GGUF` converter for a single language pair and benchmark stock llama.cpp
CPU vs `inference-rs`. That single number tells us whether this is a viable
consolidation or whether the Rust engine earns its keep.

## Related

- `notes/05-fx-model-architecture.md` — full model architecture.
- `notes/08-perf-analysis.md`, `notes/10-decoder-optimizations.md` — the perf bar
  llama.cpp CPU has to clear.
- `notes/09-final-comparison.md` — validation methodology vs the marian oracle.
