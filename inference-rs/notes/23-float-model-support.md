# Float32 model support, and what it says about the en-ru divergence

Motivation: `fxtranslate` and the reference `translator-cli` disagree on en-ru,
with the Rust output differing from the reference largely by capitalization. The
question was whether that divergence is caused by **int8 quantization** — which
can only be answered by running the same engine on the same model unquantized.

So `marian-conv --gemm-type float32` was used to convert the production en-ru
student, and `fxtranslate` grew float32 support (`data/models/enru-f32/`).

## Answer: no, quantization does not cause it

The divergence is precision-independent. Over all 20 sentences of
`corpora/dev-en.txt` (parity.py's default corpus), shortlist off:

| | exact match vs reference | mismatch is leading-capital only | genuinely different wording |
|---|---|---|---|
| int8 | 0/20 | 12/20 | 8/20 |
| float32 | 0/20 | 13/20 | 7/20 |

Running float32 changes *which* sentences are worded differently (quantization
does shift word choice — and shifts it on both engines equally) but does not
close the gap at all. Single-sentence illustration:

```
fxtranslate   int8   сегодня погода хорошая.
fxtranslate   f32    сегодня погода хорошая.
translator-cli int8  Погода сегодня хорошая.
translator-cli f32   Погода сегодня хорошая.
```

## Root cause: the decoder's first step embedded a token marian does not

**marian zero-pads the decoder input at position 0.** It builds the decoder input
by shifting the target embeddings right and zero-padding the vacated first slot
(`shift(embeddings, {0, 1, 0})` in `DecoderTransformer::step`), so at the first
step there is no previous token to embed — only the positional encoding.

`fxtranslate` seeded `prev = eos` and embedded it, the BOS convention other
toolkits use. `Wemb[eos]` has 476 of its 512 entries non-zero, and scaled by
`√512` it has L2 norm **30.5** against `PE(0)`'s **16.0** — so the first decode
step was receiving a spurious vector ~1.9× the size of the only signal that says
"this is the start of a sentence."

Confirmed directly in the reference trace: the decoder step-0 embedding node
(`id=397`) is all 512 elements zero, and the step-0 PE node (`id=399`) is exactly
`PE(0)` in marian's block layout (256 zeros, then 256 ones). `decoder_start_state_1`
is also all zeros, so the SSRU cell seeding was already right.

The symptom was systematic. Step-0 logits over the same 280 shortlist candidates:

```
translator-cli: ▁Погода(18.79) ▁Сегодня(15.80) ▁По(13.83) ▁У(11.18) ▁На(11.06) ▁В(11.02)
fxtranslate   : ▁сегодня(15.22) ▁погода(11.78) ▁на(10.28)  ▁и(10.25)  ▁у(9.87)  ▁сейчас(9.35)
```

The same words, in systematically the opposite case. The model picked the right
word and the wrong capitalization; once the first token differed, word order and
agreement drifted downstream, which is why some sentences looked reordered or
truncated rather than merely lowercased.

### Impact of the fix

Greedy exact-match against `translator-cli`, shortlist off, `corpora/dev-en.txt`:

| pair | before | after |
|---|---|---|
| en-ru int8 | 0/20 | 18/20 |
| en-ru **float32** | 0/20 | **20/20** |
| en-fr int8 | 15/20 | 19/20 |
| en-es int8 | 6/20 | 19/20 |

float32 reaching 20/20 is the clean confirmation: with the seeding fixed,
`fxtranslate` reproduces the reference exactly, and the handful of remaining int8
mismatches are the legitimate near-tie reshuffles quantization causes (they are
single-word synonym swaps — `как ты`/`как дела`, `и`/`а` — and they disappear at
float32).

Pinned by `tests/decoder_seed.rs`, which fails on both counts without the fix.

### Why nothing caught this

Three layers of existing verification all had the same blind spot:

- **The trace replay** (`oracle replay`) recomputes each node from its children
  *in the trace*. The decoder's step-0 embedding and PE are `const` leaf nodes, so
  they are passed through, never recomputed. The replay reports 635 nodes
  recomputed and no real divergence even with the bug present.
- **`onnx/numpy_ref.py`** seeds `prev = tok.eos_id` — the same convention, hence
  the same bug. Its agreement with `fxtranslate` validated the ops and weight
  layout but not the seeding. The ONNX and ggml ports inherit this too, since
  they were built against the same reference.
- **The en-fr anchor test** (`Hello world.` → `Bonjour le monde.`) is one of the
  sentences that happens to be robust to the perturbation, so it passed either way.

The float32 work is what made this findable: it ruled out quantization in one
step, which left the driver loop as the only place to look.

## The float path itself

`Weights` detects precision at load from the container's dtype and reports it via
`Weights::precision()`. No new API, no flag, no change above `Weights`.

The three container differences are in `architecture.md` "The two containers".
The one that matters: an affine weight is `[K, N]` in the float container and
`[N, K]` in the int8 one, because marian's float save branch is a bare
`val->get(item, pName)` with no pack and no transpose, while the int8 branch runs
`PrepareBTransposed`. `Wemb` is the exception — `[vocab, dim]` in both, since its
declared shape is already intgemm's `Bᵀ` form.

This is a silent failure mode: a square weight read in the wrong orientation
still has a valid shape and yields fluent, wrong output. `ops::affine_f32` takes
its weight in the float orientation explicitly, and `tests/float_model.rs` pins
the behaviour against hand-computed values on a non-square weight.

`lean-embed` also had to change. It assumed int8 unconditionally
(`quant_mult()` on load, `output_wemb_qmult()` always `Some`). A float model now
always uses resident f32 tables — under `lean-embed` there is nothing to save,
because the tables *are* the model's own data — so the embedding representation
is chosen at load from the dtype rather than purely by feature. Without this, a
float model only ran under `--no-default-features`, which also drops the SIMD
kernel and so would not have been a like-for-like comparison.

### Verification

Against `onnx/numpy_ref.py` on the same sentence and the same source ids:

| | encoder max abs diff | first-step logits max abs diff | top-5 |
|---|---|---|---|
| int8 | 0.168 | 4.742 | reshuffled, argmax disagreed |
| **float32** | **1.9e-06** | **4.2e-05** | **identical, in order** |

That residual is f32 accumulation-order noise. Orientation is separately pinned
by cosine similarity: int8-vs-float on the same weight scores 0.9998–0.99998,
while a deliberately transposed read scores **-0.031**.

### Not a fast path

Float support is a reference/comparison facility. The f32 GEMM is a plain
`axpy`-ordered scalar loop (`ops::affine_f32`) with no packing, blocking, or
SIMD — it is cache-friendly and auto-vectorizes, but it is not competitive with
the gemmology int8 kernel and is not meant to be.

### A caveat that reframes any int8-vs-float quality claim

The shipped students are **quantization-aware finetuned**, so their float weights
already sit on the int8 grid. Per-tensor correlation between the float checkpoint
and the dequantized int8 model is 1.000000, with max abs diff ~1e-4; the residual
off the grid averages 0.0015 of a quantization step where honest rounding would
average 0.25. Biases and layernorms are bit-identical.

So int8-vs-float on these models measures **activation** quantization and
accumulation order, not weight quantization — there is almost no weight error
left to measure. Any quality number drawn from this comparison needs to say so.
