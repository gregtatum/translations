# Speeding up inference-rs — targets, methodology, and levers

## Why this note exists

The ONNX perf study ([notes/16](./16-onnx-perf-comparison.md)) turned up an uncomfortable
fact: on the en-ru base model, block-batched and **single-threaded**, ONNX Runtime int8 does
**2367 words/s vs inference-rs's ~1279** — inference-rs is **0.54× ORT** (ORT is 1.85× us). The
threading A/B ruled out the easy explanation: ORT multithreading adds only ~9%, so the gap is
*single-threaded kernel/graph efficiency*, not threads. Both engines lower int8 matmuls to the
**same** Apple-Silicon i8mm instruction family (gemmology's `i8mm<neon64>` kernel; ORT's MLAS
`MatMulInteger`), so the gap is implementation — small-`m` kernel efficiency, per-call overhead,
and op fusion — not a capability we lack.

This note is the plan to close that gap **without giving up the memory win** (inference-rs is
~131 MiB settled vs ORT's ~444 MiB, and single-threaded by design). Memory is the product
differentiator; every speed experiment here is measured on *both* axes and a change that
inflates settled RSS past the current budget is a regression, not a win.

## The target, quantified

| engine (en-ru base, 1 thread, block-batched) | words/s | settled MiB |
|---|---:|---:|
| inference-rs (rust, fast) | ~1279 | ~131 |
| marian block-bench (native) | ~1325 | 298 |
| ONNX ORT int8 (1 thread) | 2367 | 444 |

Two references, two different lessons:
- **marian** (~1325) is the "same-family C++ kernel" ceiling we already nearly match (0.96×).
- **ORT** (2367) is 1.85× *both* of us — so ORT is doing something on this small autoregressive
  workload that neither the Rust engine nor marian's kernel does. That's the interesting target.

Goal: move inference-rs toward ORT's throughput while holding settled RSS at ~131 MiB and
staying token-identical to the marian oracle. Realistic framing: MLAS is a very mature kernel
library; we may not fully match it, but the levers below are concrete and independently useful.

## Profile update (2026-08-03, en-ru base) — what it changed

I ran `scripts/perf.py en ru --samply --blocks` and symbolicated the rs profile (9881 samples,
single thread, nllb block corpus). It **partly confirms and partly overturns** the priorities I
first sketched from the older en-fr profile. Self-time:

| self-time | symbol | note |
|---:|---|---|
| **73.2%** | `gemmology …Shift::Multiply<i8mm<neon64>>` | the int8 kernel — confirms it dominates |
| **~12%** | `LocalKey::with` (affine closure wrapper) | inlined `prepare_a_into` activation-quant, *not* TLS — see below |
| 4.5% | `Engine::encode_batch` | encoder glue (packing/copies) |
| 3.9% | `ops::layer_normalization` | scalar, SIMD-able |
| 1.5% | `Engine::decode_step_batch` | |

Inclusive regions (disjoint call subtrees): **encoder 43.5%**, **output projection 28.8%**,
**decoder layers 14.9%**. So decode-side ≈ 44% (projection 28.8% + layers 14.9%) and encoder ≈
44% — roughly **half the wall-clock is the encoder**, and within the decode side the single
32000-wide projection costs ~2× the two SSRU layers combined (dec-depth 2).

Three revisions to the plan:

1. **The kernel is the lever (73%), across *both* large-`m` encoder and small-`m` decode.** My
   first draft framed the kernel gap as a small-`m` decoder problem. It isn't — the encoder is
   large-`m` and also ~44% of time, all in the same gemmology kernel. So **H2 (kernel efficiency
   vs MLAS) is the primary lever**, and the microbench must span large `m` too, not just `m`=1–4.

2. **~12% self-time is the activation-quantization pass — *not* thread-local overhead (I was
   wrong at first).** The profile shows ~12% in `std::thread::local::LocalKey::with`, which
   reads like macOS TLS cost. It isn't: that frame is the `with_borrow_mut` *closure wrapper*
   around the affine body — `prepare_a`/`matmul_into` inline into it (0% self of their own) and
   the gemmology kernel is its *child* (74%). I tested the TLS hypothesis by making the
   thread-local `const`-initialized (removes any lazy-init guard); it moved the number by ~0 and
   throughput by ~1.6% (noise). So the 12% is the inlined **`prepare_a_into`** work — float→
   shifted-u8 activation quantization done once per affine — and the lever is to **fuse it into
   the matmul or SIMD it (H3/H6)**, not to touch the thread-local. (Lesson worth keeping: a hot
   `LocalKey::with` frame is almost always the closure body, not the TLS resolve.)

3. **Demote H1 (fused argmax).** On this model `project_argmax` barely registers (~0%); the
   projection's 28.8% is the GEMM itself, not the argmax writeback. H1's premise was the older
   profile's 6% argmax bucket. Still maybe worth the logit-writeback bandwidth, but the ceiling
   is small here — do it opportunistically, not first.

Net: the profile **did change my mind** — the biggest levers are the kernel (73%, via the
microbench) and the activation-quant pass (~12%, H0/H3), not fused argmax. Everything below
stands, reordered accordingly (see the revised sequence).

## Kernel microbench (2026-08-03) — decoder kernel wins, encoder trails MLAS ~1.4×

The plan hinged on whether the ~1.85× gap to ORT is the kernel (→ H2) or the graph (→
H0/H3/H4). I built shape-exact microbenches — `crates/fxtranslate/examples/kernel_micro.rs`
(gemmology, `task rs:kernel-micro`) and `onnx/kernel_micro.py` (ORT `MatMulInteger`, `task
rs:onnx-kernel-micro`) — at the decoder (m=1–4) and encoder (m=64/256) shapes, k=512, n up to
32000, single-thread. gemmology's kernel:

| shape | gemmology GFLOP/s |
|---|---:|
| m=1, n=512 / 2048 / 32000 | 238 / 250 / 201 |
| m=4, n=512 / 2048 / 32000 | 295 / 299 / 282 |
| m=64, n=512 / 2048 | 326 / 316 |
| m=256, n=512 / 2048 | 326 / 328 |

Clean MLAS numbers (ORT `MatMulInteger`, single-thread, **IOBinding** so no per-run feed/output
marshalling — this barely changed the figures, confirming they're the kernel, not dispatch):

| shape | gemmology (incl. dequant+bias) | MLAS (raw int matmul) | winner |
|---|---:|---:|---|
| m=1, n=512 (decode) | 238 | 35 | **gemmology 6.8×** |
| m=4, n=512 | 295 | 126 | **gemmology 2.3×** |
| m=4, n=32000 (proj) | 282 | 111 | **gemmology 2.5×** |
| m=64, n=2048 (encode) | 316 | 371 | MLAS 1.2× |
| m=256, n=512 (encode) | 326 | 465 | MLAS 1.4× |
| m=256, n=2048 (encode) | 328 | 455 | MLAS 1.4× |

Corrected read (my earlier "near roofline everywhere" was too strong): gemmology **beats** MLAS
at the **decoder** shapes by 2–7× — and it's carrying the dequant+bias epilogue MLAS's raw
`MatMulInteger` isn't — so the decoder has *no* kernel headroom. MLAS pulls ahead only at
**large-m encoder** shapes, ~1.2–1.4× (and even that gap shrinks once you charge MLAS for the
dequant it skips). So the one bit-identical kernel lever is matching MLAS's large-m tiling on
the encoder (~44% of time): ceiling ≈ `0.44·(1 − 1/1.4) ≈ 13%` overall, and less after the
dequant caveat. Effortful C++, modest and uncertain payoff.

This also means the end-to-end 1.85× is **not** a decoder-kernel deficit (we win there) — it's
the encoder kernel (~1.4× on 44%) plus structural differences in ORT's graph/quant that we
can't reproduce bit-identically.

So the kernel headroom is confined to the encoder (large m); the decoder kernel and the glue
have little to give bit-identically. The levers below were tried on that basis.

## Micro-levers tried — the bit-identical avenue is nearly exhausted (2026-08-03)

I prototyped the two cheapest glue levers, each bit-identical (parity tests green) and A/B'd on
the same corpus:

| lever | result | verdict |
|---|---|---|
| SIMD `prepare_a_into` (NEON, H0/H6) | **+1.7%** (1570→1596) | reverted — not worth carrying `unsafe` for ~1.7% |
| dedupe shared-activation quant across QKV + SSRU (H0/H4) | **+0.5%** (noise) | reverted — adds per-call allocs, no measurable gain |

Both target the ~12% quant; both barely move because the quant is **memory-bound** (stream f32
in, u8 out) and small next to the 73% kernel. On **fused QKV/SSRU GEMMs (H4):** the weights have
very different per-tensor scales (`Wq/Wk/Wv` qB = 61/65/299 — 389% spread; SSRU 165%), so
concatenating them under gemmology's *scalar* unquant would need a shared-scale requant that is
badly lossy → chrF would tank. It *could* be made bit-identical with a **per-column-unquant**
kernel callback (each column keeps its own scale) — but that's C++ shim work for a win capped by
the fact that **fusion doesn't reduce bytes streamed**, and the kernel is bandwidth-bound, so the
ceiling is ~the 0.5% the quant-dedup already showed. Not worth it.

### chrF-equivalence experiment: shortlist (2026-08-03)

Tested the biggest work-cutting lever directly — the lexical shortlist shrinks the 32000-wide
projection to a few hundred candidate tokens. **Speed: +42% (1590 → 2259 words/s)**, block
latency 28.5 → 19.6 ms — nearly ORT's 2367, at 131 MiB. **But it is not chrF-equivalent:**
against the shortlist-off ground truth (300 en-ru sentences), **chrF 94.16, with 117/300 (39%)
sentences changed** and real mistranslations (`продавать мою` → `продавать мины` "my [iron]" →
"mines"; `создали` → `вложили` "created" → "invested"). So the shortlist is a genuine speed↔
quality trade, not a free win — it fails the equivalence bar.

**Conclusion.** No *bit-identical* win of size remains: the decoder kernel already beats MLAS,
the encoder kernel is ~1.4× behind (≤~13% overall, effortful C++), fusion is bandwidth-capped
(~0.5%), and the glue is memory-bound and minor. The one *big* lever — the shortlist — buys +42%
but drops chrF to ~94 with visible errors. So the strategic fork is:

- **(a) Encoder kernel** — match MLAS's large-m tiling in gemmology. Bit-identical, no quality
  cost, but modest (≤~13%) and real kernel work. The only sizeable *equivalent* lever.
- **(b) Shortlist** — +42%, chrF ~94. A product decision (Firefox ships it), not equivalent.
- **(c) Accept the position** — 0.96× marian, ~3× Wasm, ⅓ ORT's memory.

My recommendation: **(a) if a bit-identical speedup is wanted** (it's the only equivalent one
left, and it's bounded); otherwise **(c)**. The shortlist (b) is a separate quality call. More
bit-identical micro-opts are not productive — this section is the evidence for stopping that line.

## Where the time goes (older en-fr profile, for shape context)

From [notes/08](./08-perf-analysis.md), profiled via samply on the block path. GEMM is ~96% of
self-time. The dominant shapes per decode step (`engine.rs` `decode_step_batch` →
`weights::full_logits`, `ops::intgemm_affine`, `gemm.rs` `PreparedB::matmul`):

| GEMM | m | k | n | frequency | note |
|---|---:|---:|---:|---|---|
| **output projection** (`full_logits`) | 1–4 | 512 | **32000** | every step | streams the ~16 MB int8 embedding once per step for only 1–4 rows |
| attention Q/K/V/O | 1–4 | 512 | 512 | every layer, every step | `ops::intgemm_affine` |
| SSRU cand + gate | 1–4 | 512 | 512 | every layer, every step | two affines on the *same* input |
| FFN W1/W2 | 1–4 | 512 | 2048/512 | every layer, every step | |

Two structural facts make the output projection the prime suspect:
1. **It's bandwidth-bound.** Arithmetic intensity ≈ `2·m·k·n / (k·n) = 2m` ≈ 2–8 FLOP/byte at
   `m`=1–4 — the kernel streams the whole 16 MB weight matrix to do a handful of rows. It was
   **61.4% of self-time** in the scalar baseline and remains the single biggest matmul.
2. **The en-ru base decoder is only `dec-depth = 2` layers.** With just two SSRU layers between
   embedding and the 32000-wide projection, the projection dominates a decode step even more
   than on the 4-layer en-fr student notes/08 profiled. Fixing it moves the needle most.

## Methodology — how to attack this without fooling ourselves

**1. Re-profile on the actual target first.** notes/08's profile is en-fr student (dec-depth 4).
Get a fresh en-ru base (dec-depth 2) profile before optimizing:
```
scripts/perf.py en ru --samply --blocks        # -> artifacts/perf-blocks-rs-enru.json.gz
```
Open in the Firefox Profiler, split **encoder vs decode-loop** self-time (the encoder is one
large-`m` pass and probably efficient; confirm, don't assume), and get the per-GEMM breakdown
for the base shapes.

**2. Isolate kernel from graph with a shape-exact microbenchmark.** The 1.85× could be the
kernel (gemmology weak at small `m`) or the graph (too many separate quant/dequant passes and
FFI crossings). Disambiguate with a criterion bench that runs *only* the GEMM at the exact
decoder shapes — `m ∈ {1,2,4}`, `k=512`, `n ∈ {512, 2048, 32000}` — for:
- gemmology `PreparedB::matmul` (our path),
- ORT's `MatMulInteger` called standalone (via a tiny one-node ONNX graph, single-thread),
so we get a clean per-shape kernel ratio. If gemmology ≈ MLAS per-shape, the gap is graph
overhead (levers H3/H4); if MLAS wins per-shape, it's the kernel (H2).

**3. Think in rooflines, not vibes.** For each GEMM, compute arithmetic intensity and label it
bandwidth- or compute-bound. Bandwidth-bound GEMMs (the projection) are helped by *reducing
bytes moved* (fused argmax, packed layout, shortlist), not by faster arithmetic.

**4. Measure both axes, every time.** `scripts/perf.py en ru --blocks` for words/s (median +
IQR of 4 runs, model load excluded) **and** `final_comparison.py` for settled/peak RSS. A speed
patch that adds >~5 MiB settled RSS needs a memory justification or it doesn't land.

**5. Correctness is a hard gate, not a nicety.** Every change must keep green:
`tests/gemm_parity.rs` (SIMD == scalar, bit-for-bit), `tests/int8_parity.rs` (scalar == marian
`int8shiftAlphaAll`), `tests/real_trace.rs` (full trace), `tests/batched_decode.rs`
(batch-invariance), and the 103-block Frankenstein output hash / 13,192-token count. Kernel and
fusion changes that merely *reorder the same MACs* must stay bit-identical; anything that can't
must be justified against the oracle and the parity stance ([wasm-parity]).

## Levers (roughly in priority order)

Each: mechanism → how to measure → expected payoff → memory/correctness risk.

### H0 — Fuse/SIMD the activation-quantization pass (profile-confirmed ~12%)
`prepare_a_into` (float activation → shifted-u8) runs once per affine and profiles at ~12%
(inlined into the affine closure; see the profile-update note for how I mis-read this as a
thread-local cost first). Two ways at it, ideally both:
- **SIMD it:** the clamp/shift/cast loop in `ops.rs:prepare_a_into` is scalar; NEON-vectorize
  the `f32 → clamp(-127,127) → +127 → u8` conversion.
- **Fuse it into the kernel (overlaps H3):** gemmology can quantize the activation as it packs
  `A`, avoiding a separate full pass over the activation before the matmul.
- **Measure:** the `LocalKey::with`/closure self-time should shrink; re-run words/s.
- **Payoff:** up to the measured ~12%, across every affine in encoder and decoder.
- **Risk:** must stay bit-identical to the current shifted-u8 quantization (int8_parity,
  gemm_parity). Memory-neutral.

### H1 — Fuse argmax into the output projection (kill the 16 MB×float writeback)
Greedy decoding needs only `argmax` over the 32000 logits, but `full_logits` materializes the
whole `[m, 32000]` float logit buffer and `project_argmax` then streams it again (6% self-time).
Fuse the max-reduce into the projection epilogue: as each output column is computed, update a
running `(max, argidx)` per row; never write the full logit vector.
- **Measure:** microbench the projection with vs without materialization; profile
  `project_argmax` share (should go to ~0).
- **Payoff:** removes a `m·32000·4`-byte write and a second full read per step — pure bandwidth
  on the hottest, most bandwidth-bound GEMM. Plausibly the highest-ROI single change.
- **Risk:** none to memory (drops a buffer). Bit-identical argmax → parity safe. Only affects
  greedy; a future beam/shortlist path still needs full logits, so keep both.

### H2 — Small-`m` kernel specialization
gemmology's i8mm kernel is tiled for throughput at moderate `m`; at `m`=1–4 it may re-load
weights per tiny row-tile, wasting the i8mm MAC density. MLAS keeps dedicated skinny-`m`
(GEMV-ish) quantized kernels.
- **Measure:** the H-#2 microbench per-shape ratio at `m`=1 vs 4 vs 8.
- **Payoff:** if MLAS beats gemmology at small `m` by ~1.5–2×, a specialized skinny kernel (or
  gemmology tuning) recovers most of the gap on *every* decode GEMM at once.
- **Risk:** kernel work is the most effortful/uncertain lever; gate hard on `gemm_parity`.

### H3 — Fuse activation-quant → matmul → dequant+bias
Each affine today is three passes: `prepare_a` (float→shifted-uint8, `ops.rs:456`), the FFI
matmul, then a separate dequant+bias. ORT fuses the whole chain (`MatMulIntegerToFloat`). At
`m`=1–4 the non-matmul passes and the FFI boundary are a real fraction (the ~4.9% "iterator"
self-time is quant/dequant). Fold the `unquant_mult` scale + prepared bias into the gemmology
kernel epilogue, and quantize activations inline.
- **Measure:** self-time of the quant/dequant iterators before/after; microbench affine end-to-
  end vs matmul-only.
- **Payoff:** removes per-affine passes across *all* ~8 affines/layer — compounding with H2.
- **Risk:** epilogue must reproduce `unquant_mult·acc + prepared_bias` exactly (int8_parity).

### H4 — Fewer, wider GEMMs (fused QKV, fused SSRU gate+cand)
Attention projects Q/K/V as three affines on the same input; SSRU computes `cand` (rnn_W) and
`gate` (rnn_Wf) as two affines on the same input. Concatenate the weight matrices at load
(`[k, 3·out]` / `[k, 2·out]`), do one matmul, slice the result.
- **Measure:** GEMM-count per step before/after; words/s.
- **Payoff:** amortizes per-call overhead (H3) and gives the kernel a larger `n` to work with
  (helps H2). Standard transformer fusion.
- **Risk:** memory neutral (same weights, concatenated once at load). Bit-identical → parity
  safe. Slightly more weight-load memory if we keep both layouts — concat in place.

### H5 — Shortlist for the projection (biggest lever, with an asterisk)
The projection's `n=32000` is the root cost. A lexical shortlist cuts it to a few hundred
candidate tokens (`engine.rs:645 project_int8`, m=1). It's **off by default** because it
perturbs output and the reference shortlist is imperfect. But it's the single largest possible
win on the dominant GEMM.
- **Measure:** words/s and chrF with shortlist on vs off; quantify the quality delta vs the
  oracle.
- **Payoff:** potentially large (projection is 60%+ of a step).
- **Risk:** **changes output** — breaks token-identity, so it's a product/quality decision, not
  a free optimization. Document the chrF cost; keep off by default unless the quality delta is
  acceptable. Listed for completeness, not as a stealth default.

### H6 — SIMD the quant/dequant scalar passes
`prepare_a` and dequant are scalar `map` iterators. NEON-vectorize the float→int8 clamp/shift and
the int32→float scale.
- **Payoff:** small (the ~4.9% iterator bucket), but cheap and memory-free. Do it if H3 doesn't
  already subsume it.

### H7 — Optional multithreading (deprioritized, and why)
The ORT A/B showed threads buy only ~9% on this small, latency-bound autoregressive model. A
threaded inference-rs would likely see a similar small gain at a real memory cost (per-thread
arenas/scratch) — the opposite of the product goal. **Not a near-term lever.** If a future
single-document latency use-case needs it, expose it as an explicit opt-in, off by default, and
measure the RSS cost — but expect ~9%, not the 1.85×.

## Suggested sequence (revised after the microbench + micro-lever results)

1. **Kernel microbench — DONE (IOBinding-clean).** gemmology beats MLAS at decoder shapes (2–7×);
   MLAS leads ~1.4× only at large-m encoder shapes. Kernel headroom exists **only on the encoder**.
2. **Bit-identical glue micro-opts — DONE, negative.** SIMD quant (+1.7%, unsafe) and shared-
   activation dedup (+0.5%, noise) both reverted. Fusion is bandwidth-capped (~0.5%) even done
   losslessly. **The glue avenue is exhausted at the ~1–2% level.**
3. **chrF-equivalence experiment (shortlist) — DONE, negative on equivalence.** +42% speed but
   chrF **94.16** vs the shortlist-off ground truth (39% of sentences change, real errors). A
   speed↔quality trade, not an equivalent win.
4. **Remaining fork:** **(a)** encoder-kernel work (match MLAS large-m tiling; bit-identical,
   ≤~13%, real C++); **(b)** ship the shortlist as a product quality decision (+42%, chrF ~94);
   **(c)** accept the position (0.96× marian, ~3× Wasm, ⅓ ORT's memory). Recommend **(a)** only if
   a bit-identical speedup is specifically wanted; else **(c)**.

The H0–H6 lever descriptions below are kept for reference; H0/H4 and the shortlist were tried
(steps 2–3), so treat them as background, not a to-do list.

## Guardrail summary (pin this)

- **Speed metric:** `scripts/perf.py en ru --blocks` — words/s, median + IQR, model load
  excluded, single thread, shortlist off.
- **Memory metric:** `final_comparison.py` settled/peak RSS @ 20 ms; budget ≈ current 131 MiB.
- **Correctness:** `gemm_parity`, `int8_parity`, `real_trace`, `batched_decode`, Frankenstein
  output hash + 13,192-token count — all must stay green; reordering-only changes stay
  bit-identical.
- **Profiling:** `scripts/perf.py en ru --samply --blocks` → Firefox Profiler; `--features
  dhat-heap` for allocation attribution.

[wasm-parity]: ./12-wasm.md
