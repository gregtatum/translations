# Pivot translations

Firefox Translations ships **one-way** models that all translate to or from a single hub
language — English in production (`en → es` and `es → en` are two separate models). To
translate a pair where neither side is the hub, say `es → fr`, you **pivot**: run `es → en`,
then feed that English text into `en → fr`. This is how a linear number of models covers the
cartesian product of language pairs — with ~50 languages you ship ~100 one-way models instead of
~2500 direct pairs.

This document describes how fxtranslate resolves and runs pivots, what it costs in memory, how the
`list` command presents it, and how the behavior is validated cheat-proof against the marian
oracle.

## Marian does not pivot — we mirror how Firefox does

There is no pivot logic inside Marian (grep `inference/marian-fork/src` for "pivot" — nothing). A
Marian process loads one encoder–decoder and translates one direction. Firefox orchestrates the
pivot *outside* the engine: it runs the `es → en` model, takes the **detokenized** output text,
and runs it through the `en → fr` model. Two independent inferences, plain text handed between
them.

fxtranslate mirrors that exactly. A pivot is two [`Engine`](./crates/fxtranslate/src/engine.rs)
runs chained through their public `translate(&str) -> String` API — the first engine's decoded
string is the second engine's input. Nothing about the transformer or the greedy decoder changes;
a pivot is pure data-flow composition of the direct path, so it stays faithful to the reference by
construction.

## Resolving a pair to a route

[`route::resolve_route`](./crates/fxtranslate/src/route.rs) maps a requested `src → trg` to a
`Route`:

- **`Direct { src, trg }`** — a single model translates the pair. A direct model **always wins**.
- **`Pivot { src, pivot, trg }`** — no direct model, so bridge through a hub `P` that has models
  for both `src → P` and `P → trg`.

The hub is **not hardcoded to English**. `resolve_route` collects every language that bridges the
pair and only *prefers* `en` (`route::PREFERRED_HUB`) as a tie-breaker when several qualify,
falling back to the lexicographically-first bridge otherwise. So a future collection with a
different hub keeps working — this is covered by a unit test that pivots through a non-English hub
(`pivots_through_a_non_english_hub` in `route.rs`).

[`loader::load_translation`](./crates/fxtranslate/src/loader.rs) is the batteries-included entry
point: it fetches the Remote Settings records once, resolves the route, downloads+verifies the
model files for each leg through the existing `cache::ensure_model`, and returns an
[`engine::Translation`](./crates/fxtranslate/src/engine.rs) — `Direct(Engine)` or
`Pivot { pivot, first, second }`. `Translation::translate` runs the direct engine, or chains the
two legs. The CLI is transparent to all of this; `translate es fr` just works, and the status line
names the hop (`ready (es→en→fr, pivot)`).

## Memory: what is settled vs. retained

A `Translation` is built once and reused for every line of a session (pipe/REPL). So a pivot holds
**both** engines resident for the life of the session — we do **not** reload per line (that would
be catastrophic in pipe/REPL mode, where `translate()` runs many times). The trade-off is
confirmed by design: hold both, pay the memory, skip the reloads.

- **Settled (persistent for the session):** each leg's loaded model — int8 weights, biases,
  layernorm params, and (unless the `lean-embed` feature is on) the dequantized embedding table;
  under `gemmology`, the register-blocked prepared weights. A shared-vocab model peaks around
  **~89 MB** (see [`notes/06-memory-approach.md`](./notes/06-memory-approach.md)); a pivot holds
  two, so peak settled memory is roughly **~178 MB** for a shared-vocab pivot (more for a
  split-vocab/CJK leg, which carries a second embedding table).
- **Retained (transient, per call):** each engine keeps its own reusable scratch buffers
  (`RefCell<Vec<…>>` for quantized activations, logits, etc.), pre-allocated at load and resized in
  place — no steady-state allocation during decode. The one pivot-specific transient is the
  **intermediate hub-language string** between the two legs: a single `String` produced by leg 1
  and consumed by leg 2, then dropped. It is O(sentence length), negligible against the models.

There is no shared workspace or buffer pool across the two engines — each owns its scratch, and the
peak is dominated by the two resident models, reached at load time, not during decode.

## The `list` command

The default `list` view is **language-oriented**, not model-oriented, because pivoting makes most
languages fully interoperable:

- **Fully supported** — a language with a model both to and from the hub (`en → L` *and*
  `L → en`). It can translate to/from any other fully-supported language, directly or by pivoting.
  Listed once, as itself: `Spanish (es)`.
- **Single-direction only** — a language that ships a model in only one direction (e.g. only
  `en → nn`). It cannot pivot both ways, so it is listed separately with the direction it supports
  (rendered as the actual one-way pair).

`list --all` drops to the raw, per-direction model pairs (`en → es`, `es → en`, …) — the literal
download units. A `[lang]` argument filters either view by prefix (`zh` catches `zh-Hans` and
`zh-Hant`). The classification lives in [`route::catalog`](./crates/fxtranslate/src/route.rs); the
rendering in [`cli::write_languages` / `write_pairs`](./crates/fxtranslate-cli/src/cli.rs).

## Cheat-proof audit

Pivoting is validated two ways against the marian oracle, both auto-detecting the pivot the same
way the shipped resolver does (via `translate_common.resolve_route`, the harnesses' on-disk
equivalent of `route::resolve_route`).

- **Correctness — [`scripts/parity.py`](./scripts/parity.py):** `parity.py es fr` compares our
  `es → en → fr` pivot against the reference `translator-cli` doing the *same* two steps. Crucially
  each engine feeds its **own** leg-1 output into leg 2 (our English into our `en → fr`, marian's
  into marian's). That keeps it non-tautological: leg 2 runs on machine-English (out of the
  dev-corpus distribution), so the check exercises our pivot orchestration — the detokenized text
  handoff — end to end, not just each leg in isolation.
- **Performance — [`scripts/perf.py`](./scripts/perf.py):** `perf.py es fr` times the end-to-end
  pivot (leg 1 over the corpus, its output through leg 2), reporting per-leg and combined
  wall-clock / sentences-per-second for both engines. It logs plainly that the run is a pivot so
  the numbers aren't misread as a direct pair. (`--samply`/`--blocks` are per-model paths and are
  refused for pivots — profile each leg directly.)
- **Memory:** capture a pivot session's peak with the engine's `dhat-heap` feature and confirm it
  matches the "two resident models" picture above.

Both harnesses need the legs downloaded (`task rs:download-model -- es en`, `-- en fr`) and, for
the baseline, the native `translator-cli` (`task inference-build`).
