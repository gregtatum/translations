#!/usr/bin/env python3
"""Build and save ``models/decode_step.onnx``.

One autoregressive decoder step per the SPEC decode_step contract: embed the
previous token (host-supplied PE added in-graph), run four post-norm SSRU layers
with cross attention over the precomputed encoder K/V, then the tied output
projection. The pre-ReLU SSRU cell ``c`` is threaded as persistent state.
Mirrors ``numpy_ref.decode_step`` op for op.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import model_npz as npz

_MODELS = Path(__file__).resolve().parent / "models"


class _Graph:
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
            self.inits.append(numpy_helper.from_array(arr, name))
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


def _cross_attn(g: _Graph, q: str, k: str, v: str, hint: str) -> str:
    """Cross attention: q is ``[1,384]``, k/v are ``[seq,384]``.

    Softmax over the source axis; scale 1/sqrt(48); returns ``[1,384]``.
    """
    h, dk = npz.HEADS, npz.HEAD_DIM
    q_shape = g.const("cross_q_shape", np.array([1, h, dk], dtype=np.int64))
    kv_shape = g.const("cross_kv_shape", np.array([-1, h, dk], dtype=np.int64))

    qh = g.add("Reshape", [q, q_shape], f"{hint}_q_r")
    qh = g.add("Transpose", [qh], f"{hint}_q_t", perm=[1, 0, 2])  # [8,1,48]
    kh = g.add("Reshape", [k, kv_shape], f"{hint}_k_r")
    kh = g.add("Transpose", [kh], f"{hint}_k_t", perm=[1, 2, 0])  # [8,48,seq]
    vh = g.add("Reshape", [v, kv_shape], f"{hint}_v_r")
    vh = g.add("Transpose", [vh], f"{hint}_v_t", perm=[1, 0, 2])  # [8,seq,48]

    scores = g.add("MatMul", [qh, kh], f"{hint}_scores")  # [8,1,seq]
    scale = g.const("attn_scale", np.array(npz.ATTN_SCALE, dtype=np.float32))
    scores = g.add("Mul", [scores, scale], f"{hint}_scaled")
    attn = g.add("Softmax", [scores], f"{hint}_softmax", axis=-1)
    ctx = g.add("MatMul", [attn, vh], f"{hint}_ctx")  # [8,1,48]
    ctx = g.add("Transpose", [ctx], f"{hint}_ctx_t", perm=[1, 0, 2])  # [1,8,48]
    join = g.const("cross_join", np.array([1, h * dk], dtype=np.int64))
    return g.add("Reshape", [ctx, join], f"{hint}_join_r")


def _dec_layer(g: _Graph, u: str, p: str, layer: int) -> tuple[str, str]:
    """One decoder layer. ``u`` is ``[1,384]``; returns (next_u, new_state[1,384])."""
    # SSRU: cand=u@rnn_W (no bias); gate=sigmoid(u@rnn_Wf+bf);
    # c = g*c_prev + (1-g)*cand;  h = relu(c);  LN(h+u)
    cand = _linear(g, u, f"{p}_rnn_W", None, f"{p}_cand")
    gate = _linear(g, u, f"{p}_rnn_Wf", f"{p}_rnn_bf", f"{p}_gate")
    gsig = g.add("Sigmoid", [gate], f"{p}_gsig")
    c_prev = f"decoder_state_{layer}"
    one = g.const("one", np.array(1.0, dtype=np.float32))
    one_minus_g = g.add("Sub", [one, gsig], f"{p}_1mg")
    old = g.add("Mul", [gsig, c_prev], f"{p}_old")
    new = g.add("Mul", [one_minus_g, cand], f"{p}_new")
    c_t = g.add("Add", [old, new], f"{p}_ct")  # [1,384], new state
    hcell = g.add("Relu", [c_t], f"{p}_hcell")
    res = g.add("Add", [hcell, u], f"{p}_self_res")
    x_self = _layer_norm(g, res, f"{p}_rnn_ffn_ln_scale", f"{p}_rnn_ffn_ln_bias", f"{p}_self_ln")

    # cross attention
    q = _linear(g, x_self, f"{p}_context_Wq", f"{p}_context_bq", f"{p}_cq")
    attn = _cross_attn(g, q, f"cross_k_{layer}", f"cross_v_{layer}", f"{p}_cross")
    attn = _linear(g, attn, f"{p}_context_Wo", f"{p}_context_bo", f"{p}_co")
    res = g.add("Add", [attn, x_self], f"{p}_ctx_res")
    x_ctx = _layer_norm(
        g, res, f"{p}_context_Wo_ln_scale", f"{p}_context_Wo_ln_bias", f"{p}_ctx_ln"
    )

    # FFN
    h = _linear(g, x_ctx, f"{p}_ffn_W1", f"{p}_ffn_b1", f"{p}_ffn1")
    h = g.add("Relu", [h], f"{p}_ffn_relu")
    fout = _linear(g, h, f"{p}_ffn_W2", f"{p}_ffn_b2", f"{p}_ffn2")
    res = g.add("Add", [fout, x_ctx], f"{p}_ffn_res")
    next_u = _layer_norm(g, res, f"{p}_ffn_ffn_ln_scale", f"{p}_ffn_ffn_ln_bias", f"{p}_ffn_ln")
    return next_u, c_t


def build() -> onnx.ModelProto:
    g = _Graph()

    # embed prev_token: sqrt(d)*Wemb[id] + pe_vec  -> [1,384]
    wemb = g.const("Wemb", npz.wemb())
    emb = g.add("Gather", [wemb, "prev_token"], "emb", axis=0)  # [1,384] (prev_token is [1])
    scale = g.const("embed_scale", np.array(npz.EMBED_SCALE, dtype=np.float32))
    emb = g.add("Mul", [emb, scale], "emb_scaled")
    # pe_vec is [384]; broadcast-add
    u = g.add("Add", [emb, "pe_vec"], "u0")

    new_states = []
    for layer in range(npz.DEC_DEPTH):
        u, c_t = _dec_layer(g, u, f"decoder_l{layer + 1}", layer)
        new_states.append(c_t)

    # tied output projection: u @ Wemb^T + logit_bias  -> [1,32000]
    wemb_t = g.const("Wemb_T", np.ascontiguousarray(npz.wemb().T))  # [384,32000]
    logits = g.add("MatMul", [u, wemb_t], "logits_mm")
    logits = g.add("Add", [logits, g.const("logit_bias", npz.logit_bias())], "logits_b")
    # squeeze [1,32000] -> [32000]
    sq_axes = g.const("sq0", np.array([0], dtype=np.int64))
    g.nodes.append(helper.make_node("Squeeze", [logits, sq_axes], ["logits"]))

    # emit new states squeezed to [384]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [npz.VOCAB])]
    for layer, c_t in enumerate(new_states):
        g.nodes.append(helper.make_node("Squeeze", [c_t, sq_axes], [f"new_decoder_state_{layer}"]))
        outputs.append(
            helper.make_tensor_value_info(
                f"new_decoder_state_{layer}", TensorProto.FLOAT, [npz.DIM]
            )
        )

    inputs = [
        helper.make_tensor_value_info("prev_token", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("pe_vec", TensorProto.FLOAT, [npz.DIM]),
    ]
    for i in range(npz.DEC_DEPTH):
        inputs.append(
            helper.make_tensor_value_info(f"cross_k_{i}", TensorProto.FLOAT, ["seq", npz.DIM])
        )
        inputs.append(
            helper.make_tensor_value_info(f"cross_v_{i}", TensorProto.FLOAT, ["seq", npz.DIM])
        )
    for i in range(npz.DEC_DEPTH):
        inputs.append(
            helper.make_tensor_value_info(f"decoder_state_{i}", TensorProto.FLOAT, [npz.DIM])
        )

    graph = helper.make_graph(g.nodes, "decode_step", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def main() -> None:
    _MODELS.mkdir(exist_ok=True)
    model = build()
    onnx.checker.check_model(model)
    out = _MODELS / "decode_step.onnx"
    onnx.save(model, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
