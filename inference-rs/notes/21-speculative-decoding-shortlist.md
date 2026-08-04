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

## Cross-refs
`notes/20` (encoder/decoder opportunity map; this is decoder item D2, plus the DRAM-vs-cache
measure-first), `notes/18` (decode is bandwidth-bound on the projection; batching + row
compaction), `notes/19` (why this belongs in fxtranslate, not llama.cpp — the flat cross stash),
[[inference-rs-validation]] (why the shortlist is off as a decider — exactly the cases the
verifier now catches for free).
