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

## Where the divergence does look like it lives

Three independent implementations of the forward pass agree with each other and
disagree with `translator-cli`:

- `fxtranslate` f32
- `fxtranslate` int8
- `onnx/numpy_ref.py` — a clean-room numpy implementation read from the `.npz`

Tokenization is also exonerated: `fxtranslate`'s source ids match `sentencepiece`
exactly (`[287, 6751, 279, 8097, 1168, 264, 0]` for "The weather is nice today."),
and the reference trace's input node has the same length.

What remains, and the shape of the evidence, points at the **first decode step**
rather than the arithmetic. The wording-divergent cases look like a displaced or
dropped first token, not like noise:

```
src : I love programming.
ref : Я люблю программирование.
rust: люблю программирование.            <- leading "Я" absent

src : Please close the door.
ref : Пожалуйста, закройте дверь.
rust: , пожалуйста, закрой дверь.        <- output begins with a comma
```

A sentence cannot legitimately begin with a comma, so step 0 is emitting
something it should not. That is consistent with the capitalization symptom
having the same root cause rather than being a separate post-processing gap.

**Important caveat on the evidence.** `numpy_ref.greedy` seeds the decoder with
`prev = eos`, `pos = 0`, zero SSRU states — the same convention
`Engine::greedy` uses. So numpy_ref's agreement validates the ops and the weight
layout (which is what certifies the float path below); it does **not**
independently validate the decoder start convention. If the seeding is what
differs from marian, both implementations would share the error. That is the
next thing to check, and it is not checked here.

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
| **float32** | **1.9e-06** | **5.8e-05** | **identical, in order** |

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
