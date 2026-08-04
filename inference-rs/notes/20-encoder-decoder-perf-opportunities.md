# Encoder & decoder perf — exploration opportunities (fxtranslate + llama.cpp)

Context note for a **new agent** picking up perf work on the translation engines. It captures
the opportunity space discovered while building the G2 llama.cpp port (`notes/19`) and the G1
optimization pass (`notes/18`). **The encoder and decoder are different physics and get
different levers — treat them as two independent tracks.**

Scope: CPU-first, en→ru **base v3.0** student (dim=512, heads=8, head_dim=64, enc-depth=6,
dec-depth=2, ffn=2048, vocab=32000, SSRU decoder + cross-attn, post-norm, sinusoidal PE,
tied-embeddings-all). Two engines in play: **fxtranslate** (the Rust port, `crates/fxtranslate`)
and **llama.cpp** (`LLM_ARCH_MARIAN` on the local `~/dev/llama.cpp` branch `marian-arch`). Do
not chase bit-exactness with the marian oracle (given up in G2); validate by numeric parity to
the validated float path + chrF, and by the harness in `scripts/final_comparison.py`.

## The one insight that organizes everything

Measured split (notes/18 + notes/19 M3): decode is ~62% of wall time, encode ~38%, and they are
opposite regimes:

- **Encoder = compute-bound.** Big matmuls (seq × dim × ffn); many independent output rows →
  splitting work across cores is efficient. In G2, llama.cpp's encoder ran **2.4× faster than
  G1** at 4 threads (0.62s vs 1.51s) and kept scaling — because it runs the whole source as one
  `llama_encode` graph and threads the matmuls, where G1 rebuilt an encoder graph per sentence.
- **Decoder = bandwidth-bound.** m=1 autoregressive; the dominant op is the tied output
  projection **512×32000 in Q8_0 ≈ ~17 MB streamed *every token*** (~70–90% of decode bytes).
  The arithmetic is trivial; you are moving weights through the memory bus. Adding cores just
  makes them contend for the same bus — measured: llama.cpp decode went 4.61→4.60→4.86s at
  4/5/6 threads (flat, then *worse*), while encode kept dropping. **Threading does not fix a
  bandwidth-bound op.**

So: **threading is an encoder lever; the decoder needs a different family of moves (shrink /
amortize / skip the weight stream).** Any "make the decoder multi-threaded" plan hits the same
wall we already measured.

## What ALREADY EXISTS (do not reinvent these)

**fxtranslate** (`crates/fxtranslate/src/engine.rs`, `Cargo.toml`):
- **Lexical shortlist** — `src/shortlist.rs`, `Engine.shortlist: Option<Shortlist>`. Implemented
  but **OFF by default** because it hurt quality (see [[inference-rs-validation]]). It restricts
  the output vocab per sentence from a source-derived candidate table. Keep it off as a
  *decision-maker*; it is reusable as a speculative *draft* (below).
- **Coarse data-parallel path** — `threads` feature, `Engine.threads`, `Engine::with_threads`,
  `encode_batch`. This is **thread-per-sentence**: the `Engine` is `Sync`, weights are shared
  read-only, each worker uses thread-local scratch (`thread_local!` full-vocab logits buffer).
  So the maintainable coarse parallelism already has scaffolding — the opportunity is tuning /
  extending it, not building it.
- **Batched encoder** — `encode_batch` (padded, mask-attention, `[batch, seq, dim]`).
- **gemmology int8 GEMM** — `gemmology` feature; vendored SIMD kernel via FFI (`gemm::PreparedB`,
  shifted int8 affine), see `src/lib.rs`. `fast = ["lean-embed","gemmology","mmap"]`. Single
  sentence is **single-threaded per matmul** (no intra-op threading).
- Timing hooks: `Timing{encode_ms, first_token_ms, decode_ms}`, `Phase::{EncodeStart,
  DecodeStart, FirstToken, DecodeEnd}` — the encode/decode split the harness reports.

**llama.cpp** (`~/dev/llama.cpp` @ branch `marian-arch`): the full `LLM_ARCH_MARIAN` arch runs
the model byte-identical to G1 (notes/19). Threading is ggml's own threadpool (intra-op
fork-join, below). Block-batching the decoder there **regressed** (notes/19 M3b) because the
encoder output is stashed as one flat `[n_embd, Σ source-len]` tensor shared with T5 →
cross-attn cost becomes B²·mean_len; a real batching win needs per-sequence-scoped cross K/V,
which is upstream shared-graph infra work.

## Baseline (from notes/19 M3, `final_comparison.py`, Apple Silicon 6P+12E, CPU-only, 1 thread)

| engine | words/s | tokens/s | peak RSS |
|---|--:|--:|--:|
| ONNX (ORT int8) | 2398 | 3363 | 445 MiB (Python proc — not a clean RSS peer) |
| marian block-bench | 1330 | 1864 | 298 MiB |
| fxtranslate (fast) | 1280 | 1795 | **150 MiB** |
| llama.cpp (Q8_0) | 832 | 1167 | 195 MiB |

llama.cpp threads to ~1852 wps @ 5t (then regresses); fxtranslate & marian are 1t-by-design
here. fxtranslate already **beats llama.cpp single-threaded** — gemmology's kernel is better
than ggml's Q8_0 single-thread path.

---

# ENCODER track — compute-bound, threading-friendly

Goal: cut per-sentence encoder latency (time-to-first-block) and/or raise throughput. The
encoder parallelizes well; this is where "pull in llama.cpp's threading model" actually pays.

### E1. Intra-op threading of the encoder matmuls (fxtranslate)
The mechanism llama.cpp uses is **intra-op data parallelism with fork-join**: a persistent
threadpool of spin-waiting workers; per matmul, split the output rows across cores (dynamic
work-stealing via an atomic chunk counter); a barrier between ops. fxtranslate today threads
only *across sentences* (`threads` feature), not *within* a sentence — so a single sentence's
encoder is serial. Adding intra-op threading to the encoder GEMMs (FFN + attention projections)
cuts single-sentence latency and is the direct analog of llama.cpp's 2.4× encoder win.
- **Files (fxtranslate):** `src/engine.rs` (encoder pass), the gemmology call site (`src/lib.rs`,
  `gemm::PreparedB`) — the kernel must be callable on an **output-row range** so each worker
  takes a chunk. Consider `rayon` for the encoder only (its per-`par_iter` join overhead is fine
  for big encoder ops, too high for tiny decode ops) OR a hand-rolled persistent spin-pool.
- **Reference (llama.cpp / ggml threading model):** `ggml/src/ggml-cpu/ggml-cpu.c` —
  `struct ggml_threadpool` (~L480), `current_chunk` atomic (~L491), `ggml_barrier` (~L575), the
  mul_mat dynamic-chunk loop (~L1424, `ggml_threadpool_chunk_add`), driven from
  `ggml_graph_compute`. This is the whole "model": row-split + barrier + persistent workers.
- **Expected win:** encoder is 38% of time; a ~2–3× on it is a ~15–20% aggregate + big TTFT win.
- **Risk/gotcha:** **P/E-core asymmetry.** A static split waits on the slowest (E-core) straggler
  at the barrier — this is *why 6–8 threads regressed* in G2. Use dynamic chunking (atomic
  work-stealing, as ggml does) or pin to the 6 P-cores. Watch false sharing on the chunk counter.
- **Validate:** encoder output parity vs `ggml/validate_llama_encoder.py`-style golden (or the
  fxtranslate encoder golden), then `final_comparison.py` wps + a TTFT read.

### E2. Better use of the existing batched encoder for throughput
`encode_batch` already exists. For full-page throughput (many blocks), encoding a block's
sentences together amortizes setup. This is throughput (blocks/s), not per-block latency.
- **Files:** `src/engine.rs::encode_batch`; harness `scripts/final_comparison.py`.
- Note: G1 found aggregate batching only ~+5% on this corpus because it's per-sentence-shaped
  (many 1-sentence blocks) and ragged; don't expect a large win here on its own.

### E3. (Strategy) thread-per-sentence vs intra-op — pick by latency vs throughput
The `threads` feature (thread-per-sentence) maximizes **page throughput** with near-zero sync
and shares read-only weights (memory stays flat). Intra-op (E1) minimizes **per-block latency**.
For Firefox full-page UX, a **hybrid** — thread-per-sentence for throughput + a little intra-op
on the encoder for first-block latency — is likely best. Thread-per-sentence is the more
maintainable default and already exists.

---

# DECODER track — bandwidth-bound; shrink / amortize / skip the weight stream

The tied output projection (~17 MB/token) dominates. Shortlisting is only *one* strategy (skip).
The three families:

### MEASURE FIRST — is it DRAM-bound or last-level-cache-bound?
The projection is reused *every* token, and Apple Silicon's System Level Cache is ~8–24 MB —
17 MB sits right on that boundary. If it's largely SLC-resident across steps, you're bounded by
cache bandwidth (much higher), the "wall" is softer than the notes/18 DRAM estimate assumed,
and **threading the decoder helps more than we thought**. If it's truly hitting DRAM each token,
the shrink/amortize/skip moves below are the only levers. Cheap probe: artificially shrink the
vocab (or the projection width) and see whether decode tok/s scales inversely — if it does,
you're bandwidth-bound on that matrix. **Do this before building anything; it picks the track.**

### Shrink the stream

**D1. Q4 (mixed-precision) output projection.** Drop *only* the `output` matrix to 4-bit; keep
everything else Q8. Halves the dominant stream (~2× on ~70% of decode). It's the most
quality-sensitive matrix (sets the argmax), so gate on chrF.
- **Files (converter):** `ggml/convert_marian_llama.py` — the `output.weight` quant (it's already
  emitted separately from `token_embd`, so you can quantize it independently). For the fxtranslate
  path, the equivalent projection weight load in `src/engine.rs` / weight loader.
- **Validate:** `ggml/quality_llama_decoder.py` / chrF on the corpus; compare argmax-flip rate.
- **Cost:** cheapest thing to try — a converter change + a measurement. Do this early to get the
  quality number even if you don't ship it.

### Amortize the stream (stream the weights once for more than one token)

**D2. Speculative decoding — "shortlist-as-draft, full-vocab-as-verifier."** A cheap draft
proposes K tokens; the full model verifies all K in **one** m=K pass, streaming the 17 MB
projection **once for K positions** instead of K times. The elegant fit here: the shortlist was
rejected as a *decision-maker* for quality, but as a **draft it cannot hurt quality** — the
full-vocab model verifies every token and rejects wrong guesses, so greedy output is bit-exact
to full-vocab while you get the shortlist's bandwidth savings on accepted tokens. You already
have `src/shortlist.rs`. (Self-speculative alternative: a Q4 copy of the model as draft, Q8 as
verifier — no shortlist table to carry.)
- **Files (fxtranslate):** `src/engine.rs` decode loop + `src/shortlist.rs`; needs a
  draft→verify accept/reject loop and an m=K verify projection. The greedy accept rule is simple:
  accept the longest prefix where draft token == verifier argmax at that position.
- **llama.cpp path:** has speculative infra (`common/`, `examples/speculative`) but wiring it to
  the MARIAN two-phase enc-dec is real work — prefer prototyping in fxtranslate where you own the
  decode loop.
- **Expected win:** proportional to acceptance rate; 1.5–3× is typical when the draft is decent.

**D3. Batch the projection across sentences (m=B) + thread-per-sentence.** Decoding B independent
sentences in lockstep streams the projection once for B tokens → converts it from bandwidth- to
compute-bound (G1 saw B≈13 tip it compute-bound). G1's block-batched decode with **row
compaction** (retired rows drop out) exists; net was only +5% aggregate there because of ragged
block lengths + unbatched encoder, but the projection-amortization itself is real. Pairs with the
existing `threads` path.
- **Files (fxtranslate):** `src/engine.rs` (the batched decode path / `encode_batch` peer),
  `threads` feature, the `thread_local!` logits buffer.
- **Note:** do NOT port this to llama.cpp as batch-in-one-graph — it regressed (M3b, flat cross
  stash). fxtranslate owns its decoder, so it can do per-row `[dim, Smax, B]` cross K/V like G1.

### Skip the stream (don't touch most of the matrix)

**D4. Approximate argmax / MIPS over the vocab (the principled "beyond shortlist").** Greedy needs
the *winner*, not all 32k logits: `argmax_v (W_v · u)` is a Maximum-Inner-Product-Search over
32k static dim-512 vectors. Build an index over the (fixed) vocab embedding matrix once; per token
do an approximate top-K search, then exact-score only those candidates. Unlike the static
shortlist, the candidate set is **input-adaptive** (from the actual hidden state) and
**exact-verifiable** (re-score candidates) — which is why it needn't cost quality the way the
source-derived shortlist did.
- **Files (fxtranslate):** new module for the index + a hook in `src/engine.rs`'s projection step;
  reuses the vocab embedding (tied `output`).
- **Risk:** MIPS structures (HNSW etc.) are pointer-chasing / latency-bound, not bandwidth-
  friendly — may not be a clean win on a memory-bound machine, and it's a chunky bespoke component
  (cuts against the maintenance-reduction goal). Rank below D1–D3; pursue only if those fall short.

---

## Strategic framing (keep in view)

The llama.cpp port's motivation is **maintenance reduction (one runtime), not raw speed** — see
[[ggml-llamacpp-port-eval]]. Anything that adds a bespoke runtime/threadpool/index to fxtranslate
pushes the opposite way. Ranking by effort-to-maintainability: **thread-per-sentence (exists) >
Q4-output (converter) > speculative (reuses shortlist) > intra-op encoder threading > MIPS
(most bespoke).** llama.cpp's intra-op model is optimized for LLM prefill / big batch — the
opposite shape from per-sentence autoregressive MT — so don't assume its threading maps cleanly
onto the decoder.

## File map — llama.cpp (`~/dev/llama.cpp`, branch `marian-arch`; line numbers drift, symbols don't)

- **`src/models/marian.cpp`** — the arch graph. `marian_self_attn` (hand-built encoder attn,
  avoids `build_attn` pad noise), `marian_cross_attn`, recurrent SSRU via `build_rs_inp()` /
  `build_rs(rs_inp, get_s_l(il), n_embd_s(), n_seqs)` + `ggml_cpy` store-back, cross-attn reads
  `build_inp_cross_embd()`. Decoder layers stored at `layers[n_layer + il]`. SSRU state sized to
  n_embd via `ssm_d_inner = n_embd` (`ssm_d_state=1`, `ssm_d_conv=0`).
- **`src/llama-context.cpp:~1550`** — the T5+MARIAN encoder-output stash into `cross.v_embd`
  (`memcpy` of the encoder embeddings). This flat `[n_embd, Σlen]` layout is the batching blocker.
- **`src/llama-arch.{h,cpp}`** — `LLM_ARCH_MARIAN`, `LLM_TENSOR_NAMES` (`enc.blk.N.*`,
  `dec.blk.N.{rnn,rnn_f,rnn_norm,cross_attn_*,ffn_*}`, `token_embd`, `position_embd`, `output`),
  `LLM_TENSOR_INFOS`, `llm_arch_is_recurrent`.
- **`src/llama-model.{cpp,h}`** — hparam load + `build_graph`; `llama_model_has_encoder/decoder`;
  `create_memory` recurrent branch; `llama_layer` SSRU (`ssru_*`) + cross-attn bias fields.
- **`src/llama-graph.cpp`** — `build_attn_inp_no_cache` (encoder block-diagonal mask),
  `build_attn_inp_cross` (cross mask, `seq_ids_enc`), `build_rs_inp`, `build_inp_cross_embd`.
- **`src/llama-vocab.cpp`** — UGM tokenizer + the byte-fallback branch added for Marian (~L1035–66).
- **`include/llama.h`** — two-phase `llama_encode` / `llama_decode`.
- **`ggml/src/ggml-cpu/ggml-cpu.c`** — the threading model: `ggml_threadpool` (~L480),
  `current_chunk` (~L491), `ggml_barrier` (~L575), mul_mat chunk loop (~L1424), `ggml_graph_compute`.
- **Precedents:** `src/models/t5.cpp` (enc-dec + cross-attn + the stash), `src/models/jamba.cpp`
  / `rwkv6.cpp` (recurrent-in-decoder, the `get_s_l`/`build_rs` pattern).

## File map — fxtranslate (`inference-rs`)

- **`crates/fxtranslate/src/engine.rs`** — encoder pass, greedy decode loop, `encode_batch`,
  `threads` data-parallel path, `Timing`/`Phase` split, `thread_local!` logits buffer.
- **`crates/fxtranslate/src/shortlist.rs`** — the lexical shortlist (off; reuse as D2 draft).
- **`crates/fxtranslate/src/lib.rs`** — gemmology FFI (`gemm::PreparedB`, `fast_gemm`/`gemmology`).
- **`crates/fxtranslate/Cargo.toml`** — features: `fast`, `gemmology`, `threads`, `lean-embed`, `mmap`.
- **`ggml/convert_marian_llama.py`** — G2 converter (Q4-output experiment lives here).
- **`scripts/final_comparison.py`** — the shared harness (add rows / thread sweeps here).
- **`ggml/validate_llama_{encoder,decoder}.py`, `ggml/quality_llama_decoder.py`** — parity + chrF gates.

## Related notes
`notes/18` (G1 bare-libggml + the threads-are-the-lever optimization pass, the encode/decode
split, batching+row-compaction), `notes/19` (G2 `LLM_ARCH_MARIAN`, M3 perf table + M3b batching
regression + the DRAM-bandwidth diagnosis), `notes/16` (ONNX perf), `notes/17` (inference-rs
kernel perf ceiling).
