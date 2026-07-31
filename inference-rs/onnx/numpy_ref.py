#!/usr/bin/env python3
"""Float32 numpy reference forward pass (the bit-close golden).

Implements the SPEC exactly: tied+scaled embeddings with a precomputed rotor PE
table, 6 post-norm encoder layers, and a 4-layer SSRU decoder step with cross
attention and the tied output projection. See ``onnx/SPEC.md``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import model_npz as npz
import tokenizer as tok

_TESTDATA = Path(__file__).resolve().parent / "testdata"


# --- primitives ---------------------------------------------------------------


def layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Marian LayerNorm: biased (÷d) variance, eps inside sqrt."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + npz.EPS) * scale + bias


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Plain MatMul in stored [in,out] orientation, optional bias add."""
    y = x @ w
    if b is not None:
        y = y + b
    return y


def multihead(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention over ``[tq,d]`` q and ``[tk,d]`` k/v.

    Softmax is taken over the key axis; heads are split/joined internally.
    Returns ``[tq, d]``.
    """
    tq = q.shape[0]
    tk = k.shape[0]
    h, dk = npz.HEADS, npz.HEAD_DIM
    qh = q.reshape(tq, h, dk).transpose(1, 0, 2)  # [h,tq,dk]
    kh = k.reshape(tk, h, dk).transpose(1, 0, 2)  # [h,tk,dk]
    vh = v.reshape(tk, h, dk).transpose(1, 0, 2)  # [h,tk,dk]
    scores = np.einsum("htd,hsd->hts", qh, kh) * npz.ATTN_SCALE  # [h,tq,tk]
    attn = softmax(scores, axis=-1)
    ctx = np.einsum("hts,hsd->htd", attn, vh)  # [h,tq,dk]
    return ctx.transpose(1, 0, 2).reshape(tq, h * dk)


# --- positional encoding (rotor form, precomputed constant) -------------------


def build_pe(max_seq: int) -> np.ndarray:
    d = npz.DIM
    half = d // 2  # 192
    c = np.arange(d)
    freq = np.power(1e-4, (c % half) / (half - 1))  # (c mod 192)/191
    offs = np.floor(c / half) * (math.pi / 2.0)  # 0 for c<192, pi/2 for c>=192
    pos = np.arange(max_seq)[:, None]
    return np.sin(pos * freq[None, :] + offs[None, :]).astype(np.float32)


PE = build_pe(256)


def embed(ids: list[int], start_pos: int = 0) -> np.ndarray:
    """sqrt(d)*Wemb[id] + PE(pos), scaling BEFORE adding PE."""
    x = npz.EMBED_SCALE * npz.wemb()[ids]  # [seq,384]
    pe = PE[start_pos : start_pos + len(ids)]
    return (x + pe).astype(np.float32)


# --- encoder ------------------------------------------------------------------


def _enc_layer(x: np.ndarray, p: str) -> np.ndarray:
    def w(n: str) -> np.ndarray:
        return npz.weight(f"{p}_{n}")

    q = linear(x, w("self_Wq"), w("self_bq"))
    k = linear(x, w("self_Wk"), w("self_bk"))
    v = linear(x, w("self_Wv"), w("self_bv"))
    attn = linear(multihead(q, k, v), w("self_Wo"), w("self_bo"))
    x = layer_norm(attn + x, w("self_Wo_ln_scale"), w("self_Wo_ln_bias"))

    h = relu(linear(x, w("ffn_W1"), w("ffn_b1")))
    f = linear(h, w("ffn_W2"), w("ffn_b2"))
    x = layer_norm(f + x, w("ffn_ffn_ln_scale"), w("ffn_ffn_ln_bias"))
    return x


def encode(src_ids: list[int]) -> np.ndarray:
    """Encode source ids to context ``[seq, 384]`` (no final LN, no mask)."""
    x = embed(list(src_ids))
    for layer in range(1, npz.ENC_DEPTH + 1):
        x = _enc_layer(x, f"encoder_l{layer}")
    return x


# --- decoder cross-attention K/V precompute -----------------------------------


def precompute_cross_kv(context: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """K/V per decoder layer from the encoder context; computed once."""
    kv = []
    for layer in range(1, npz.DEC_DEPTH + 1):
        p = f"decoder_l{layer}"
        k = linear(context, npz.weight(f"{p}_context_Wk"), npz.weight(f"{p}_context_bk"))
        v = linear(context, npz.weight(f"{p}_context_Wv"), npz.weight(f"{p}_context_bv"))
        kv.append((k, v))
    return kv


# --- decoder step -------------------------------------------------------------


def decode_step(
    prev_token: int,
    pos: int,
    cross_kv: list[tuple[np.ndarray, np.ndarray]],
    states: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """One autoregressive step.

    ``states[i]`` is the pre-ReLU SSRU cell ``c`` for layer i (init zeros).
    Returns ``(logits[32000], new_states)``.
    """
    u = embed([prev_token], start_pos=pos)[0]  # [384]
    new_states: list[np.ndarray] = []

    for layer in range(1, npz.DEC_DEPTH + 1):
        p = f"decoder_l{layer}"

        def w(n: str) -> np.ndarray:
            return npz.weight(f"{p}_{n}")

        # SSRU cell: highway gate weights OLD cell with g, candidate with (1-g).
        cand = u @ w("rnn_W")  # no bias
        g = sigmoid(linear(u, w("rnn_Wf"), w("rnn_bf")))
        c_prev = states[layer - 1]
        c_t = g * c_prev + (1.0 - g) * cand
        new_states.append(c_t)
        hcell = relu(c_t)
        x_self = layer_norm(hcell + u, w("rnn_ffn_ln_scale"), w("rnn_ffn_ln_bias"))

        # cross-attention: Q from decoder, K/V precomputed from encoder context.
        q = linear(x_self[None, :], w("context_Wq"), w("context_bq"))  # [1,384]
        k, v = cross_kv[layer - 1]
        attn = linear(multihead(q, k, v), w("context_Wo"), w("context_bo"))[0]
        x_ctx = layer_norm(attn + x_self, w("context_Wo_ln_scale"), w("context_Wo_ln_bias"))

        # FFN
        h = relu(linear(x_ctx, w("ffn_W1"), w("ffn_b1")))
        f = linear(h, w("ffn_W2"), w("ffn_b2"))
        u = layer_norm(f + x_ctx, w("ffn_ffn_ln_scale"), w("ffn_ffn_ln_bias"))

    logits = u @ npz.wemb().T + npz.logit_bias()  # tied output projection
    return logits.astype(np.float32), new_states


# --- greedy loop --------------------------------------------------------------


def greedy(src_ids: list[int]) -> list[int]:
    """Greedy decode loop: seed prev=eos, states=zeros, stop on eos."""
    context = encode(src_ids)
    cross_kv = precompute_cross_kv(context)
    states = [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)]

    seq = len(src_ids)
    max_len = min(math.ceil(2 * seq) + 4, 256)

    out: list[int] = []
    prev = tok.eos_id
    for pos in range(max_len):
        logits, states = decode_step(prev, pos, cross_kv, states)
        token = int(np.argmax(logits))
        if token == tok.eos_id:
            break
        out.append(token)
        prev = token
    return out


def translate(text: str) -> str:
    """Encode, greedily decode, and detokenize."""
    return tok.decode_ids(greedy(tok.encode_source(text)))


# --- CLI ----------------------------------------------------------------------


def _main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="numpy float reference translation")
    ap.add_argument("text", nargs="?", default=None, help="source text")
    ap.add_argument("--ids", default=None, help='explicit source ids, e.g. "0 1 2"')
    args = ap.parse_args(argv)

    if args.ids is not None:
        src_ids = [int(x) for x in args.ids.split()]
        text = f"<ids: {args.ids}>"
    else:
        text = args.text or "Hello, world. This is a test of the translation engine."
        src_ids = tok.encode_source(text)

    print(f"text:    {text}")
    print(f"src_ids: {src_ids}")

    context = encode(src_ids)
    logits0, _ = decode_step(
        tok.eos_id,
        0,
        precompute_cross_kv(context),
        [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)],
    )

    out_ids = greedy(src_ids)
    print(f"out_ids: {out_ids}")
    print(f"translation: {tok.decode_ids(out_ids)}")

    _TESTDATA.mkdir(exist_ok=True)
    np.save(_TESTDATA / "numpy_encoder.npy", context)
    np.save(_TESTDATA / "numpy_logits.npy", logits0)
    print(f"wrote {_TESTDATA/'numpy_encoder.npy'} {context.shape}")
    print(f"wrote {_TESTDATA/'numpy_logits.npy'} {logits0.shape}")


if __name__ == "__main__":
    _main(sys.argv[1:])
