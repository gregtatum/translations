#!/usr/bin/env python3
"""Build and save ``models/encoder.onnx``.

The encoder graph embeds+scales source ids, adds a precomputed rotor PE, runs
six post-norm transformer layers, and folds in the per-decoder-layer cross
attention K/V so the whole pre-loop pass is a single ORT run. Weights enter as
initializers in the stored ``[in,out]`` orientation, so every linear is a plain
``MatMul(x, W)`` with no transpose. Mirrors ``numpy_ref.encode`` op for op.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import model_npz as npz
from numpy_ref import build_pe

_MODELS = Path(__file__).resolve().parent / "models"
_MAX_SEQ = 256


class _Graph:
    """Accumulates nodes/initializers and hands out unique names."""

    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: list = []
        self._seen: set[str] = set()
        self._n = 0

    def name(self, hint: str) -> str:
        self._n += 1
        return f"{hint}_{self._n}"

    def const(self, name: str, arr: np.ndarray) -> str:
        if name not in self._seen:
            self.inits.append(numpy_helper.from_array(arr.astype(arr.dtype), name))
            self._seen.add(name)
        return name

    def weight(self, w_name: str) -> str:
        return self.const(w_name, npz.weight(w_name))

    def add(self, op: str, inputs: list[str], hint: str, **attrs) -> str:
        out = self.name(hint)
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out


def _linear(g: _Graph, x: str, w_name: str, b_name: str | None, hint: str) -> str:
    y = g.add("MatMul", [x, g.weight(w_name)], hint)
    if b_name is not None:
        y = g.add("Add", [y, g.weight(b_name)], hint + "_b")
    return y


def _layer_norm(g: _Graph, x: str, scale: str, bias: str, hint: str) -> str:
    return g.add(
        "LayerNormalization",
        [x, g.weight(scale), g.weight(bias)],
        hint,
        axis=-1,
        epsilon=npz.EPS,
    )


def _multihead(g: _Graph, q: str, k: str, v: str, seq: str, hint: str) -> str:
    """Per-head scaled dot-product attention; q/k/v are ``[seq,384]``.

    Reshape to ``[seq,8,48]`` then transpose to ``[8,seq,48]``, score with
    scale 1/sqrt(48), softmax over the key axis, join heads back to ``[seq,384]``.
    """
    h, dk = npz.HEADS, npz.HEAD_DIM
    # shape [seq, 8, 48]
    hd_shape = g.const(f"{hint}_hdshape", np.array([-1, h, dk], dtype=np.int64))
    perm = dict(perm=[1, 0, 2])

    def heads(t: str, tag: str) -> str:
        r = g.add("Reshape", [t, hd_shape], f"{hint}_{tag}_r")
        return g.add("Transpose", [r], f"{hint}_{tag}_t", **perm)

    qh = heads(q, "q")  # [8,seq,48]
    kh = heads(k, "k")
    vh = heads(v, "v")
    kh_t = g.add("Transpose", [kh], f"{hint}_kt", perm=[0, 2, 1])  # [8,48,seq]
    scores = g.add("MatMul", [qh, kh_t], f"{hint}_scores")  # [8,seq,seq]
    scale = g.const("attn_scale", np.array(npz.ATTN_SCALE, dtype=np.float32))
    scores = g.add("Mul", [scores, scale], f"{hint}_scaled")
    attn = g.add("Softmax", [scores], f"{hint}_softmax", axis=-1)
    ctx = g.add("MatMul", [attn, vh], f"{hint}_ctx")  # [8,seq,48]
    ctx = g.add("Transpose", [ctx], f"{hint}_ctx_t", perm=[1, 0, 2])  # [seq,8,48]
    join = g.const(f"{hint}_join", np.array([-1, h * dk], dtype=np.int64))
    return g.add("Reshape", [ctx, join], f"{hint}_join_r")


def _enc_layer(g: _Graph, x: str, p: str, seq: str) -> str:
    q = _linear(g, x, f"{p}_self_Wq", f"{p}_self_bq", f"{p}_q")
    k = _linear(g, x, f"{p}_self_Wk", f"{p}_self_bk", f"{p}_k")
    v = _linear(g, x, f"{p}_self_Wv", f"{p}_self_bv", f"{p}_v")
    ctx = _multihead(g, q, k, v, seq, f"{p}_mha")
    attn = _linear(g, ctx, f"{p}_self_Wo", f"{p}_self_bo", f"{p}_attn")
    res = g.add("Add", [attn, x], f"{p}_attn_res")
    x = _layer_norm(g, res, f"{p}_self_Wo_ln_scale", f"{p}_self_Wo_ln_bias", f"{p}_attn_ln")

    h = _linear(g, x, f"{p}_ffn_W1", f"{p}_ffn_b1", f"{p}_ffn1")
    h = g.add("Relu", [h], f"{p}_ffn_relu")
    f = _linear(g, h, f"{p}_ffn_W2", f"{p}_ffn_b2", f"{p}_ffn2")
    res = g.add("Add", [f, x], f"{p}_ffn_res")
    x = _layer_norm(g, res, f"{p}_ffn_ffn_ln_scale", f"{p}_ffn_ffn_ln_bias", f"{p}_ffn_ln")
    return x


def build() -> onnx.ModelProto:
    g = _Graph()
    src_ids = "src_ids"  # int64 [seq]

    # embedding: sqrt(d) * Wemb[id] + PE(pos)
    wemb = g.const("Wemb", npz.wemb())
    emb = g.add("Gather", [wemb, src_ids], "emb", axis=0)  # [seq,384]
    scale = g.const("embed_scale", np.array(npz.EMBED_SCALE, dtype=np.float32))
    emb = g.add("Mul", [emb, scale], "emb_scaled")

    # seq = shape(src_ids)[0]
    shape = g.add("Shape", [src_ids], "src_shape")
    zero = g.const("i0", np.array([0], dtype=np.int64))
    seq = g.add("Slice", [shape, zero, g.const("i1", np.array([1], dtype=np.int64)), zero], "seq")

    pe_table = g.const("pe_table", build_pe(_MAX_SEQ))  # [max_seq,384]
    pe = g.add("Slice", [pe_table, zero, seq, zero], "pe_slice")  # [seq,384]
    x = g.add("Add", [emb, pe], "x0")

    for layer in range(1, npz.ENC_DEPTH + 1):
        x = _enc_layer(g, x, f"encoder_l{layer}", seq)
    context = x

    outputs = [helper.make_tensor_value_info("context", TensorProto.FLOAT, ["seq", npz.DIM])]
    output_names = [("context", context)]

    # fold per-decoder-layer cross K/V
    for layer in range(npz.DEC_DEPTH):
        p = f"decoder_l{layer + 1}"
        k = _linear(g, context, f"{p}_context_Wk", f"{p}_context_bk", f"{p}_ck")
        v = _linear(g, context, f"{p}_context_Wv", f"{p}_context_bv", f"{p}_cv")
        output_names.append((f"cross_k_{layer}", k))
        output_names.append((f"cross_v_{layer}", v))
        outputs.append(
            helper.make_tensor_value_info(f"cross_k_{layer}", TensorProto.FLOAT, ["seq", npz.DIM])
        )
        outputs.append(
            helper.make_tensor_value_info(f"cross_v_{layer}", TensorProto.FLOAT, ["seq", npz.DIM])
        )

    # rename internal outputs to the public names via Identity
    for public, internal in output_names:
        g.nodes.append(helper.make_node("Identity", [internal], [public]))

    inputs = [helper.make_tensor_value_info("src_ids", TensorProto.INT64, ["seq"])]
    graph = helper.make_graph(g.nodes, "encoder", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def main() -> None:
    _MODELS.mkdir(exist_ok=True)
    model = build()
    onnx.checker.check_model(model)
    out = _MODELS / "encoder.onnx"
    onnx.save(model, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
