# Clean-room ONNX exporter — consolidated spec

Authoritative reference for the Python clean-room ONNX converter (route B in
`notes/15-onnx-port.md`). Derived from three source-of-truth passes over the
`inference-rs` engine; the engine is ground truth, not any external exporter.

## Goal & pipeline

```
float .npz  →  build ONNX graph (float32)  →  ORT quantize_dynamic  →  int8 ONNX  →  ship
              └── validate: onnx-float vs numpy-float (bit-close ~1e-5) ──┘
              └── cross-check: numpy-float vs inference-rs int8 (argmax agree, ~1% off) ──┘
                                                                          └ validate: quality (chrF/token-overlap)
```

The float32 graph is a dev scaffold; the shipped artifact is the quantized int8 graph.
Validate in float first so quantization is not a confounding variable.

## Model (en-fr student-finetuned)

- float weights: `data/models/en-fr/student-finetuned/final.model.npz.best-chrf.npz` (float32, 183 arrays)
- source SPM: `data/models/en-fr/student-finetuned/vocab.en.spm`
- target SPM: `data/models/en-fr/student-finetuned/vocab.fr.spm`
- int8 model (for cross-check via inference-rs): `data/models/enfr/model.enfr.intgemm.alphas.bin`

## Config constants

| symbol | value | note |
|---|---|---|
| dim (d) | 384 | model / embedding dim |
| heads (h) | 8 | |
| head_dim (dk) | 48 | d/h; attention scale = 1/sqrt(48) ≈ 0.144338 |
| enc_depth | 6 | encoder layers, names `encoder_l1..l6` (1-based) |
| dec_depth | 4 | decoder layers, names `decoder_l1..l4` (1-based) |
| ffn dim | 1536 | |
| vocab (V) | 32000 | |
| eps | 1e-6 | LayerNorm, inside sqrt, biased (÷d) variance |
| activation | ReLU | FFN and SSRU cell output |
| norm | POST-norm | `LayerNorm(sublayer(x) + x)` — NOT pre-norm |
| positional encoding | sinusoidal rotor | precompute as constant; do NOT emit in-graph Sin |
| embeddings | tied-all | `Wemb` is source emb, target emb, and (transposed) output projection |

## Weight orientation (CRITICAL)

The **`.npz` stores each weight logically `[in, out]`** (e.g. `_self_Wq` is `(384,384)`,
`_ffn_W1` is `(384,1536)`, `_ffn_W2` is `(1536,384)`). So from the npz the op is a plain
`MatMul(x[.,in], W[in,out]) -> [.,out]`, **no transpose**. (The transpose convention you
may see in `model.rs` is the int8 `.bin` storage layout, irrelevant to the npz path.)

Square (384,384) matrices won't error if transposed wrong — the inference-rs int8
cross-check is what catches an orientation bug there. Non-square (FFN, output proj) will
shape-error if wrong.

## Embedding + positions

```
x_t = sqrt(384) * Wemb[id_t] + PE(pos_t)          # scale BEFORE adding the positional encoding
sqrt(384) ≈ 19.5959
```
No embedding LayerNorm. Positional encoding (rotor form, T = d/2 = 192):
```
freq[c] = 1e-4 ^ ((c mod 192) / 191)              # c in 0..384
offs[c] = floor(c / 192) * (pi/2)                 # 0 for c<192 (sin), pi/2 for c>=192 (cos)
PE(pos)[c] = sin(pos * freq[c] + offs[c])
```
Precompute the whole `[max_seq, 384]` positional-encoding table as an f32 constant/initializer and add a
slice; or pass a precomputed positional-encoding vector as a graph input. Never emit a live `Sin` op
(known marian-exporter divergence bug).

## Encoder layer (post-norm), for L in 1..6, prefix `encoder_l{L}`

```
# self-attention (bidirectional, NO mask for a single unpadded sentence)
q = x @ {p}_self_Wq + {p}_self_bq        # [seq,384]
k = x @ {p}_self_Wk + {p}_self_bk
v = x @ {p}_self_Wv + {p}_self_bv
# per-head: reshape [seq,8,48]; scores = (q_h @ k_h^T) * (1/sqrt(48)); softmax over key axis
# ctx_h = softmax @ v_h; join heads -> [seq,384]
attn = ctx @ {p}_self_Wo + {p}_self_bo
x = LayerNorm(attn + x; {p}_self_Wo_ln_scale, {p}_self_Wo_ln_bias)   # POST-norm
# FFN
h = relu(x @ {p}_ffn_W1 + {p}_ffn_b1)    # [seq,1536]
f = h @ {p}_ffn_W2 + {p}_ffn_b2          # [seq,384]
x = LayerNorm(f + x; {p}_ffn_ffn_ln_scale, {p}_ffn_ffn_ln_bias)     # note doubled ffn_ffn
```
No final encoder LayerNorm. Output `context` = `[seq, 384]`.

## Decoder step (post-norm), for L in 1..4, prefix `decoder_l{L}`. `u` = layer input.

```
# SSRU (replaces self-attention); persistent state = pre-ReLU cell c, one [384] per layer, init zeros
cand = u @ {p}_rnn_W                      # NO bias
gate = u @ {p}_rnn_Wf + {p}_rnn_bf
g    = sigmoid(gate)                      # the ONLY sigmoid in the model
c_t  = g * c_prev + (1 - g) * cand        # highway; g weights OLD cell, (1-g) the candidate
hcell= relu(c_t)
x_self = LayerNorm(hcell + u; {p}_rnn_ffn_ln_scale, {p}_rnn_ffn_ln_bias)
# cross-attention: Q from decoder, K/V from ENCODER context (precompute K/V ONCE, reuse each step)
q = x_self @ {p}_context_Wq + {p}_context_bq              # [1,384]
# cross_k = context @ {p}_context_Wk + {p}_context_bk     (precomputed, [seq,384])
# cross_v = context @ {p}_context_Wv + {p}_context_bv     (precomputed, [seq,384])
# per-head, scale 1/sqrt(48), softmax over source axis; join
attn = ctx @ {p}_context_Wo + {p}_context_bo
x_ctx = LayerNorm(attn + x_self; {p}_context_Wo_ln_scale, {p}_context_Wo_ln_bias)
# FFN
h = relu(x_ctx @ {p}_ffn_W1 + {p}_ffn_b1)
f = h @ {p}_ffn_W2 + {p}_ffn_b2
x = LayerNorm(f + x_ctx; {p}_ffn_ffn_ln_scale, {p}_ffn_ffn_ln_bias)
# x -> next layer's u; after layer 4, x is decoder top
logits = x @ Wemb^T + decoder_ff_logit_out_b             # [1,32000]; Wemb is [32000,384]
```
State thread: `decoder_state_i = c_t` (pre-ReLU), NOT relu(c_t). SSRU residual adds `u`
(layer input), not `hcell`.

## Tokenizer

Python `sentencepiece` reproduces ids exactly. Source: `sp.encode(text)` then **append
eos_id (0)**; **no BOS**. Decoder seeds with target EOS (id 0) at pos 0. Stop on eos.

## decode_step graph contract

- run ONCE: encoder -> context; then per-layer cross_k_i, cross_v_i (each [seq,dim]).
- INPUTS/step: prev_token (int64), pos (or precomputed positional-encoding vector), cross_k_i, cross_v_i,
  decoder_state_i (init zeros). Layer count is dec-depth (2 for en-ru base, 4 for the en-fr
  student) — read from the model config, not hardcoded.
- OUTPUTS/step: logits, new decoder_state_i (new c_t).
- **Batched graph:** every step tensor carries a leading batch dim B (prev_token [B],
  decoder_state_i [B,dim], cross_k_i/cross_v_i [B,Smax,dim], logits [B,vocab]), plus a shared
  additive source mask `cross_bias` [B,Smax] (0 for real tokens, -inf for padding). pe_vec
  stays [dim] — the whole batch decodes the same position in lockstep. B=1 with an all-zero
  bias is the per-sentence case. Rows are independent (per-row SSRU state, masked cross-attn),
  so a block-batched decode is token-identical to decoding each sentence alone.
- driver loop: greedy argmax per row, feed states back, cap max_len = min(ceil(2*seq)+4, 256)
  per row; a row retires on eos or its cap.

## Reference extraction (inference-rs int8 cross-check)

inference-rs has no tensor dump today. All needed methods are `pub` (`Engine::encode`,
`decode_step`, `project`). Add a convenience `pub fn dump_reference(&self, text) ->
(context, seq, first_logits)` on `Engine` + a `dump` subcommand to `fxtranslate-oracle`
(dispatch match at `crates/fxtranslate-oracle/src/main.rs:49-53`). Emit context [seq,384],
first-step logits [32000], and the src token ids used, so the Python side feeds identical
ids. int8-precision reference: expect argmax agreement + ~1% tensor diff vs numpy-float.

## Fixed test sentence

`"Hello, world. This is a test of the translation engine."`

## Layout (`inference-rs/onnx/`)

- `model_npz.py`  — load npz, config, named weight access (returns numpy [in,out])
- `tokenizer.py`  — sentencepiece wrap (encode + eos, no bos)
- `numpy_ref.py`  — float32 forward: encoder, decode_step, greedy loop (the bit-close golden)
- `export_encoder.py`, `export_decoder.py` — ONNX graph builders
- `engine.py`  — ORT sessions + greedy generation (the ONNX translation engine)
- `quantize.py`   — ORT quantize_dynamic
- `validate.py`   — numeric diffs + quality
- `testdata/`     — reference tensors from inference-rs
- venv: `inference-rs/.venv` (numpy, sentencepiece, onnx 1.22, onnxruntime 1.28, sacrebleu)
