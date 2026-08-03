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
| **~11.8%** | `std::thread::local::LocalKey::with` | **pure overhead**, not compute (see below) |
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

2. **New top lever, wasn't in the draft: ~11.8% self-time is macOS thread-local access.**
   `LocalKey::with` is called from `Weights::affine` (47.8% of that time), `full_logits_batch_into`
   (28.8%), and `select_active` (10.6%) — i.e. the reusable activation-quant scratch buffer is a
   `thread_local!`, and macOS's `tlv_get_addr` (lazy-init guard per access) is slow. At small `m`
   with ~8 affines/layer this is hit constantly. **Fix: hold the scratch as an `Engine` field (or
   thread it explicitly) instead of a thread-local.** Zero numerical risk, ~10%+ on the table —
   the highest ROI, lowest risk item, and it's not even a GEMM change.

3. **Demote H1 (fused argmax).** On this model `project_argmax` barely registers (~0%); the
   projection's 28.8% is the GEMM itself, not the argmax writeback. H1's premise was the older
   profile's 6% argmax bucket. Still maybe worth the logit-writeback bandwidth, but the ceiling
   is small here — do it opportunistically, not first.

Net: the profile **did change my mind** — start with the thread-local overhead (free ~10%) and
the kernel microbench (which gates the biggest 73% lever), not with fused argmax. Everything
below stands, reordered accordingly (see the revised sequence).

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

### H0 — Kill the macOS thread-local overhead in the affine hot path (measured 11.8%)
The reusable activation-quant scratch buffer is reached through a `thread_local!`
(`LocalKey::with` shows 11.8% self-time, from `Weights::affine`, `full_logits_batch_into`,
`select_active`). macOS resolves `__thread` via `tlv_get_addr` with a per-access lazy-init
guard — cheap on Linux, not here — and the decode step hits it on every affine.
- **Mechanism:** move the scratch to an `Engine`/`Weights` field (or pass it down explicitly) so
  the buffer is a plain field access, not a thread-local resolve. Keep it single-owner (the
  engine is used single-threaded).
- **Measure:** `LocalKey::with` self-time should go to ~0; re-run words/s.
- **Payoff:** up to ~10% for free — highest ROI, lowest risk, not a GEMM change.
- **Risk:** none numerically (same buffer, same math). Watch that it doesn't add a per-call
  allocation; it should be a reused field.

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

## Suggested sequence (revised after the profile)

1. **H0 — kill the thread-local overhead.** Free ~10%, zero risk, do it first.
2. **Kernel microbench, large *and* small `m`.** gemmology vs standalone ORT `MatMulInteger` at
   the encoder shapes (large `m`, k=512, n=512/2048) and the decode shapes (`m`=1–4, n up to
   32000). This gates the 73% lever: if MLAS wins per-shape it's the kernel (**H2**); if it
   ties, the remaining gap is graph overhead (**H3/H4**).
3. **H4 — fused QKV / SSRU GEMMs.** Bit-identical, memory-neutral; fewer, wider GEMMs help both
   the kernel (bigger `n`) and per-call overhead. Applies to encoder self-attention too (~44%).
4. **H3 — fused quant/dequant+bias epilogue.** Compounds across every affine in both encoder
   and decoder.
5. **H2 — kernel work** (skinny-`m` and/or better packing) if the microbench indicts gemmology.
   This is the effortful lever but it's 73% of time and spans encoder + decoder.
6. **H1 / H6** opportunistically — fused argmax (small ceiling here) and SIMD layernorm (3.9%)
   + quant passes.
7. Re-run the four-way (`task rs:onnx-perf`) and update notes/16's table; keep settled RSS ≤
   ~131 MiB throughout.

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
