# Speculative decoding with the shortlist as draft — implementation plan

Context + plan for a **new agent** implementing speculative decoding in **fxtranslate**
(`crates/fxtranslate`). Builds on the opportunity map in `notes/20` (decoder track, item D2).
The idea is the standout decoder lever because it reuses an asset we already built and shelved —
the lexical shortlist — and it is **provably lossless** (output stays byte-identical to
full-vocab greedy).

**Do this in fxtranslate, not llama.cpp.** It needs per-sentence cross-attention against each
sentence's own encoder output, which fxtranslate's decoder owns cleanly but llama.cpp's flat
shared encoder-output stash cannot batch (the M3b regression, `notes/19`).

## The idea in one paragraph

The decoder is slow because the tied output projection (~17 MB in Q8_0) is streamed through
memory **once per token**, and that haul dominates decode (`notes/18/20`). The shortlist
(`src/shortlist.rs`, off by default because it hurt quality as a *decision-maker*) restricts the
output vocab to a small per-sentence candidate set — cheap to project, but sometimes wrong. So
use it as a **fast guesser**, not a decider: guess the next K tokens with the shortlisted
projection, then **verify all K in one full-vocab projection haul** and keep only the tokens the
full model agrees with. The shortlist never decides the output — the full model confirms every
token — so the result is exactly full-vocab greedy, just with far fewer expensive hauls.

## Why it saves time (the one fact)

Streaming the 17 MB projection **once can score K positions for almost the same cost as scoring
one** — the haul is the expense; multiplying it against K hidden vectors instead of 1 is a
matrix-times-matrix instead of matrix-times-vector, weights streamed once. So the win is:

- **Baseline:** K tokens = **K expensive projection hauls** (K × 17 MB).
- **Speculative:** K cheap shortlisted projections (draft) + **1 expensive projection haul**
  (verify, scoring all K positions at once).

The decoder *layers* (SSRU + cross-attn + FFN) run K times either way — they're cheap; nothing
changes there. The whole saving is turning K full-projection hauls into one. Rough ceiling: if
the projection is ~80% of decode and acceptance averages A tokens/round, decode ≈
`0.2 + 0.8/A` of baseline. A≈3 → decode ~2×, ~1.5× overall; A≈4 → ~1.9× overall. **The probe in
Phase 0 measures A before any code.**

## The precise mechanic (get this exactly right — it's the correctness core)

Terminology: consuming a token produces (a) the updated per-layer SSRU **state** and (b) a
**hidden vector** `u`, which the projection turns into the next token. `s_j` = state after
consuming the token at draft step j; `u_j` = hidden vector at step j.

**Draft phase (K cheap autoregressive steps, one sentence):**
- Start from the current carried state `s_{-1}` and last emitted token `t_{-1}`.
- Step j = 0..K-1: consume the input token with `s_{j-1}` → produce `s_j` and `u_j`; project
  `u_j` over **only the shortlist candidates** → draft token `d_j`; feed `d_j` as the next input.
- Save every `u_j` and `d_j` (and the states `s_j`). **Do not throw the hidden vectors away —
  verify reuses them, so the layers are not re-run.**

**Verify phase (one full-vocab haul):**
- Apply the **full 512×32000 projection** to `u_0..u_{K-1}` in **one** matmul (weights streamed
  once, K columns) → full-vocab argmax `f_j` at each position.
- `f_j` is exactly what plain full-vocab greedy would emit at position j given the same history,
  because it's the same hidden vector through the same projection.

**Accept / correct / roll back:**
- Find the first position `m` where `d_m != f_m`.
- Accept `d_0..d_{m-1}` (all matched the full model) **plus the correction `f_m`** = `m+1` new
  tokens. Discard `d_{m+1}..` and their hidden/state.
- If **all K matched** (no mismatch), accept `d_0..d_{K-1}` = K tokens.
- **State carry:** keep state `s_m` (valid — it was computed consuming accepted tokens), and
  carry the correction `f_m` as the next round's input token. (Full-accept case: carry `s_{K-1}`
  and `d_{K-1}`.) Next round's first draft step consumes the carried token from the carried state.

**Why hidden vectors past `m` are invalid:** `u_{m+1}` was computed consuming `d_m`, which the
full model rejected, so its input was wrong — hence discarded. Everything up to and including `m`
consumed accepted tokens, so it's valid. This is the entire "roll back and guess again."

**Correctness invariants (assert these in tests):**
1. A guessed token is emitted **only if** it equals the full-vocab argmax at its position.
2. At a mismatch, the emitted token is the full-vocab argmax (the correction) — never the guess.
3. Output token ids are **byte-identical** to plain full-vocab greedy. This is the primary gate.
4. Acceptance rate = fraction of positions where full-vocab argmax ∈ shortlist. Nothing else
   affects correctness — only speed.

## Batching + ragged ends (see notes/20 discussion)

Each sentence is an **independent row**: own position, own SSRU state, own carried token, own
pending K-guess. One round = one shared full-projection haul over **all live rows × their K
positions**; then each row independently finds its own mismatch, accepts its own prefix +
correction, rolls back only its own leftover guesses, and re-guesses from its new tail. Finished
rows (hit EOS) retire and the batch compacts (G1's row-compaction pattern). **There is no global
rollback** — one row's bad guess never touches another. Bonus: speculation covers batching's
weak spot — even a lone straggler at the ragged tail still gets multi-token-per-haul from its own
guesses, so the tail stops being a throughput cliff.

## Plan — phased, cheapest and most decisive first

### Phase 0 — Acceptance probe (NO speculative code; do this first, it decides everything)
Run ordinary full-vocab greedy on the chrF corpus. At each decode step, before taking the full
argmax, check whether that argmax token is in the sentence's shortlist candidate set. Record:
- **overall acceptance** = fraction of steps whose full argmax ∈ shortlist (this is A's driver),
- **run-length distribution** of consecutive in-shortlist steps (predicts avg tokens/round for a
  given K — e.g. how often you'd get 4-in-a-row).
This estimates the speedup ceiling for the cost of a few greedy runs. If acceptance is low (say
<70%), stop — speculation won't pay, and that's a cheap thing to learn. Also fold in the
`notes/20` DRAM-vs-cache probe so you know the projection is really the bottleneck.
- **Files:** a small harness around `src/engine.rs`'s greedy loop + `src/shortlist.rs`
  (expose the per-sentence candidate id set and a membership check).

### Phase 1 — Single-sentence speculative loop
Implement draft → verify → accept/rollback for one sentence, reusing hidden vectors between draft
and verify (do not re-run layers). Fixed K (start K=4–6). The only genuinely new compute is the
shortlisted projection per draft step + the batched full projection over K in verify.
- **Gate:** output ids **byte-identical** to full-vocab greedy on the corpus (invariant 3).
- **Measure:** avg tokens accepted/round (A), decode wps vs baseline, and confirm chrF unchanged.
- **Files:** `src/engine.rs` (a speculative variant of the greedy loop; a "project over a
  candidate-id subset" path alongside the full projection; SSRU state carry/rollback),
  `src/shortlist.rs` (candidate ids + membership).

### Phase 2 — Batched + ragged
Extend to multiple independent rows: per-row accept/rollback/retire, row compaction, per-sentence
cross-attention K/V (fxtranslate already does per-sentence encoder output — reuse `encode_batch`
and the `threads`/batched-decode scaffolding). Compose with the existing thread-per-sentence path
if useful.
- **Gate:** batched speculative output **byte-identical** to per-sentence speculative and to
  plain greedy (batch-invariance, the G1 `solo`-vs-batched gate pattern).
- **Files:** `src/engine.rs` batched decode path, the `thread_local!` logits buffer, row
  compaction.

### Phase 3 — Tune + wire into the harness
Sweep K (guess length) and shortlist size S (bigger S → higher acceptance but costlier drafts;
find the knee). Add a `--fxtranslate-spec` row (or a feature/flag) to
`scripts/final_comparison.py` and report vs marian / ONNX / llama.cpp / baseline fxtranslate.
- **Gate:** net speedup with output still byte-identical; pick default K, S.
- **Files:** `scripts/final_comparison.py`, a `spec-decode` cargo feature or runtime flag in
  `Cargo.toml`/`src/engine.rs`.

## Design knobs & risks
- **K (guess length):** too small → little amortization; too large → wasted draft steps past the
  likely mismatch. Start 4–6; tune in Phase 3. Consider adaptive K later (grow K when acceptance
  is high).
- **Shortlist size S:** the acceptance-vs-draft-cost tradeoff. The probe (Phase 0) tells you what
  S buys in acceptance.
- **Draft overhead is the only downside:** on a low-acceptance sentence you still do 1 full haul
  per round (no worse than baseline in hauls) but paid K cheap draft steps. Never a quality risk,
  only a mild speed risk — which the probe rules out up front.
- **No new float divergence:** draft and verify share the same layer math and the same projection
  weights for candidate rows, so the accept comparison and the corrections are exact — hence the
  byte-identical guarantee. Do not introduce a separate lower-precision draft here (that would be
  the *self-speculative* variant, `notes/20` D2 alt — a different, non-exact-by-construction path;
  keep this plan to shortlist-as-draft, which is exact).

## Validation summary
Primary gate everywhere: **speculative output ids == plain full-vocab greedy ids, exactly** (and
chrF therefore identical). Secondary: acceptance rate A, decode wps, aggregate wps vs baselines in
`final_comparison.py`. Reuse `ggml/quality_llama_decoder.py`-style chrF tooling and the corpora.

## Phase 0 results — GO (measured 2026-08-04)

Implemented and run. The probe is `Engine::acceptance_probe` (`crates/fxtranslate/src/engine.rs`)
+ `Shortlist::candidate_set` (`src/shortlist.rs`), driven by `examples/acceptance_probe.rs`
(`task rs:spec-probe`). It decodes plain full-vocab greedy — the shortlist never decides a token —
and records per step whether the full-vocab argmax is in the sentence's shortlist candidate set.
Gate test `acceptance_probe_mirrors_full_vocab_greedy` pins that the probe walks the exact
full-vocab greedy path (step count == shortlist-free `greedy` length + 1 for EOS).

Corpus: `corpora/nllb-en-fr.txt`, 1000 sentences, 24 052 decode steps, en→fr int8 model with the
shipped `lex.50.50` shortlist.

- **Overall acceptance: 99.2%** (23 863 / 24 052 steps; only 189 mismatches). Far above the 70%
  go/no-go bar. This is the key finding: the shortlist is a near-perfect *guesser* even though it
  was a quality-losing *decider* — the ~0.8% it gets wrong is exactly what the full-vocab verifier
  now corrects for free ([[inference-rs-validation]]).
- **Shortlist size:** mean 705, median 696, max 1344 candidates/sentence — the draft-projection
  width (S). Small vs the 32 000 full vocab, so drafts are cheap.
- **Run lengths:** in-shortlist runs are long (mode ~14–26 consecutive correct guesses), so
  acceptance barely falls off with K — larger K keeps paying.
- **Simulated A (tokens accepted / verify haul) and predicted decode speedup** (P = 0.8; opt =
  note's ceiling with free layer re-runs, real = layers re-run K×/round):

  | K | A | hauls/tok | decode↑ opt | decode↑ real |
  |---|-----|-----|------|------|
  | 2 | 1.95 | 0.512 | 1.64× | 1.63× |
  | 3 | 2.85 | 0.350 | 2.08× | 2.04× |
  | 4 | 3.72 | 0.269 | 2.41× | 2.32× |
  | 5 | 4.54 | 0.220 | 2.66× | 2.52× |
  | 6 | 5.32 | 0.188 | 2.86× | 2.66× |
  | 8 | 6.80 | 0.147 | 3.15× | 2.83× |

**Verdict: GO.** Acceptance is high enough that A grows almost linearly in K (mismatches are rare),
so the note's `K=4–6` start is conservative — adaptive/large K is worth trying in Phase 3. Proceed
to Phase 1 (single-sentence speculative loop). Two caveats before trusting the *absolute* decode
speedup: (1) confirm P≈0.8 with the `notes/20` DRAM-vs-cache probe — the table scales with P;
(2) the model excludes the cheap draft-projection cost (S≈705 columns × K steps/round), a mild
real-world haircut the Phase 1 wps measurement will show directly.

## Phases 1–3 built + measured — NEGATIVE RESULT: it regresses (2026-08-04)

Speculative decoding is now implemented end-to-end and **proven correct**, but on measurement it is
**a throughput regression on every workload tested**, so it is not enabled and gets no row in the
`notes/20` comparison table. The Phase 0 projection below was wrong — this section supersedes it.

**Built (correct, byte-identical, gated):**
- `Engine::greedy_speculative(src, k)` — single-sentence draft→verify→accept/rollback, reusing the
  drafted hidden vectors in verify.
- `Engine::greedy_batch_speculative(sentences, k)` — Phase 2 batched: per-row guess/accept/retire,
  ragged compaction, one shared full-vocab haul per round over Σ live-rows×K.
- Gates: `translate.rs::speculative_matches_full_vocab_greedy` (K=1..8) and
  `batched_decode.rs::speculative_batch_matches_full_vocab_greedy` (K=1..6, per row vs full-vocab
  batched greedy *and* single greedy). The correctness argument holds exactly as designed: the
  emitted token is always the full-vocab argmax, so output is byte-identical regardless of the
  draft — confirmed over the whole en-ru corpus in the bench's pre-timing check.

**Measured (`task rs:spec-bench`, en-ru base, Frankenstein, median of 4, 1 thread):**

| K | block-batched wps | vs baseline | single-sentence wps | vs baseline |
|---|--:|--:|--:|--:|
| baseline (full-vocab greedy) | 1256 | 1.00× | 877 | 1.00× |
| 2 | 1094 | 0.87× | 794 | 0.91× |
| 4 | 1044 | 0.83× | 757 | 0.86× |
| 6 | 985 | 0.78× | — | — |
| 8 | 963 | 0.77× | — | — |

Slower everywhere, and monotonically worse with K. **Why the plan's premise was wrong:**

1. **The projection is not DRAM-bandwidth-bound here** — the falsified core assumption. If it were
   (notes/18), collapsing K per-token hauls into one would win big at batch=1; instead batch=1
   *regresses* (0.91× at K=2). The 16.4 MiB int8 `Wemb` evidently stays largely cache-resident on
   this M-series part, so "streaming it once per token" is a cache read, not a DRAM haul — there is
   no haul to collapse. **This is exactly the notes/20 DRAM-vs-cache probe the plan said to run
   first, and I skipped it.** The whole speedup model rested on P≈0.8 (projection = 80% of decode);
   that was never confirmed and is false for this workload.
2. **Batching already amortizes the projection.** The baseline `greedy_batch` streams the vocab
   weight once per *step* shared across all active rows; speculation streams it once per *round*.
   With the weight cache-resident anyway, that buys almost nothing — and speculation *adds* the
   draft's per-row candidate-gather (`project_int8` re-gathers ~965 rows every draft step) + SSRU
   snapshot clones + rollback bookkeeping, which is pure overhead. Hence worse with larger K.

**Disposition.** Keep the code (it is correct, tested, and a real capability) but do not enable it.
Do not chase draft-side optimizations (cache the per-sentence draft weight, drop the snapshot
clones) until a direct **projection-share probe** confirms the projection is actually the decode
bottleneck on the target — the end-to-end regression says it is not, and measure-first discipline
(the lesson of this whole exercise) says confirm that before spending more. The real decode levers
remain in `notes/20`: encoder threading (E1) and reducing the eager-allocation churn.

### Profiles (samply, Firefox Profiler)

CPU profiles on the en-ru base / Frankenstein workload (`task rs:spec-bench --profile`, recorded
with `--unstable-presymbolicate`), the regression made visible. Files + full write-up in the
gitignored `artifacts/speculative-profiles.md`.

| profile | shared | leaf self-time headline |
|---|---|---|
| baseline `greedy_batch` (full-vocab) | https://share.firefox.dev/4wHJCaG | gemmology i8mm GEMM **69.1%**, encode_batch 8.9%, layernorm 4.1% |
| speculative `greedy_batch_speculative` K=5 | https://share.firefox.dev/4xi9ABl | GEMM **57.8%** + draft overhead the baseline lacks: `gemmology_read_row` 5.2% (candidate gather) + `intgemm_affine` 3.4% (un-batched draft GEMM) + `Vec::spec_from_iter` 2.2% (snapshot clones) + `output_wemb_int8_row`/`project_argmax` 1.5% |

Side-by-side: speculation adds **~13%** of runtime in draft machinery (candidate gather + un-batched
draft GEMM + snapshot/alloc churn) while shrinking the full-vocab-projection GEMM only modestly — so
the GEMM *share* falls 69→58% yet wall-clock rises. The engine is GEMM-bound (same as notes/09/17);
speculation adds a second GEMM path rather than removing work, which is why it regresses.

### (superseded) Phase 0 en→ru probe + projection

Kept for the record; the measurement above overrides it. Re-running the probe on the en→ru base
model gave **95.4% acceptance** (13 595 steps, mean shortlist 965) and a *projected* decode↑ of
~2.2–2.6× → a *projected* ~1.35–1.41× overall (1280 → ~1730–1800 wps) via an Amdahl calc on the
measured 52.6%/47.4% encode/decode split. That projection assumed the decode↑ was real; it was not,
because assumption (1) above fails. The encode/decode split itself is a valid measurement and is
recorded in `notes/20`.

## Cross-refs
`notes/20` (encoder/decoder opportunity map; this is decoder item D2, plus the DRAM-vs-cache
measure-first), `notes/18` (decode is bandwidth-bound on the projection; batching + row
compaction), `notes/19` (why this belongs in fxtranslate, not llama.cpp — the flat cross stash),
[[inference-rs-validation]] (why the shortlist is off as a decider — exactly the cases the
verifier now catches for free).
