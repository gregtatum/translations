#!/usr/bin/env python3
"""Convert the float Marian/Bergamot student .npz to GGUF for the bare-libggml engine.

This is the ggml analog of ``onnx/export_{encoder,decoder}.py`` (route B of
``notes/15-onnx-port.md``): a clean-room converter that reads the *pre-quantization*
float ``.npz`` and emits weights in a container the ggml engine loads. It reuses
``onnx/model_npz.py`` for config + named weights so both evaluations share one source
of truth for the architecture, and it bakes the sinusoidal positional-encoding table
exactly as ``onnx/numpy_ref.py`` computes it, so PE is never a divergence source.

Two artifacts are written into ``ggml/models/``:
  * ``marian.float.gguf`` — F16 linears (the Gate-2 dev scaffold, diffed vs numpy_ref)
  * ``marian.q8_0.gguf``  — Q8_0 linears (the shipped/measured artifact, Gate 1 + perf)

Weight orientation: the ``.npz`` stores each linear logically ``[in, out]``. ggml's
``ggml_mul_mat(W, x)`` contracts on ``ne0`` (the fastest axis), so we store the
transpose ``W.T`` with numpy shape ``(out, in)`` → ggml ``ne0=in, ne1=out``, giving
``y[o] = sum_i W[i,o] * x[i]``. The tied embedding ``Wemb`` (vocab, dim) is emitted
twice: ``token_embd.weight`` (F16, for the ``get_rows`` embedding lookup) and
``output.weight`` (Q8_0, for the tied output projection ``logits = Wemb @ u``) — the
same split ONNX uses (float Gather, quantized projection).

Run via ``task rs:ggml-convert``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ONNX = _HERE.parent / "onnx"
sys.path.insert(0, str(_ONNX))  # reuse the ONNX eval's validated model loader

import model_npz as npz  # noqa: E402

import gguf  # noqa: E402

_MODELS = _HERE / "models"
_ARCH = "marian"
_MAX_SEQ = 256


def _build_pe(max_seq: int) -> np.ndarray:
    """Sinusoidal rotor PE table, byte-for-byte the formula in numpy_ref.build_pe."""
    d = npz.DIM
    half = d // 2
    c = np.arange(d)
    freq = np.power(1e-4, (c % half) / (half - 1))
    offs = np.floor(c / half) * (math.pi / 2.0)
    pos = np.arange(max_seq)[:, None]
    return np.sin(pos * freq[None, :] + offs[None, :]).astype(np.float32)


class _Emit:
    """Accumulates tensors so a float and a Q8_0 GGUF can be written from one pass.

    ``linear`` weights are transposed to (out, in) once; ``store_q8`` marks whether the
    Q8_0 file quantizes that tensor (linears + the output projection) or keeps it F16
    (embeddings, PE) — everything else (biases, LayerNorm) stays F32 in both files.
    """

    def __init__(self) -> None:
        self.float_tensors: list[tuple[str, np.ndarray]] = []
        self.q8_tensors: list[tuple[str, np.ndarray, bool]] = []

    def linear(self, name: str, npz_name: str) -> None:
        """A matmul weight: npz [in,out] -> ggml (out,in). F32 in the float scaffold (so
        Gate 2 isolates graph bugs from precision), Q8_0 in the shipped file."""
        w = npz.weight(npz_name)
        assert w.ndim == 2, f"{npz_name} is not a matrix: {w.shape}"
        wt = np.ascontiguousarray(w.T)  # (out, in) -> ggml ne0=in, ne1=out
        self.float_tensors.append((name, wt.astype(np.float32)))
        self.q8_tensors.append((name, wt.astype(np.float32), True))

    def raw(self, name: str, arr: np.ndarray, f16: bool = False) -> None:
        """A non-matmul tensor (bias, LayerNorm, embedding, PE): never Q8_0."""
        a = arr.astype(np.float16 if f16 else np.float32)
        self.float_tensors.append((name, a))
        self.q8_tensors.append((name, a.astype(np.float32) if not f16 else a, False))


def _collect() -> _Emit:
    e = _Emit()

    wemb = npz.wemb().astype(np.float32)  # (vocab, dim)
    # get_rows lookup (ne0=dim, ne1=vocab): F32 in the float scaffold, F16 in the shipped file.
    e.float_tensors.append(("token_embd.weight", wemb.astype(np.float32)))
    e.q8_tensors.append(("token_embd.weight", wemb.astype(np.float16), False))
    # tied output projection: logits = ggml_mul_mat(Wemb, u); Wemb (vocab,dim) -> ne0=dim.
    e.float_tensors.append(("output.weight", wemb.astype(np.float32)))
    e.q8_tensors.append(("output.weight", wemb, True))
    e.raw("output.bias", npz.logit_bias())
    e.raw("pos_enc", _build_pe(_MAX_SEQ))  # F32: added to F32 embeddings, keep it clean

    def ln(prefix_out: str, npz_prefix: str) -> None:
        e.raw(f"{prefix_out}.ln.scale", npz.weight(f"{npz_prefix}_ln_scale"))
        e.raw(f"{prefix_out}.ln.bias", npz.weight(f"{npz_prefix}_ln_bias"))

    # Encoder layers (npz names are 1-based; GGUF names 0-based).
    for l in range(npz.ENC_DEPTH):
        p, o = f"encoder_l{l+1}", f"enc.{l}"
        for x in ("q", "k", "v", "o"):
            e.linear(f"{o}.self.w{x}", f"{p}_self_W{x}")
            e.raw(f"{o}.self.b{x}", npz.weight(f"{p}_self_b{x}"))
        ln(f"{o}.self", f"{p}_self_Wo")
        e.linear(f"{o}.ffn.w1", f"{p}_ffn_W1")
        e.raw(f"{o}.ffn.b1", npz.weight(f"{p}_ffn_b1"))
        e.linear(f"{o}.ffn.w2", f"{p}_ffn_W2")
        e.raw(f"{o}.ffn.b2", npz.weight(f"{p}_ffn_b2"))
        ln(f"{o}.ffn", f"{p}_ffn_ffn")

    # Decoder layers: SSRU + cross-attention + FFN.
    for l in range(npz.DEC_DEPTH):
        p, o = f"decoder_l{l+1}", f"dec.{l}"
        e.linear(f"{o}.rnn.w", f"{p}_rnn_W")  # no bias
        e.linear(f"{o}.rnn.wf", f"{p}_rnn_Wf")
        e.raw(f"{o}.rnn.bf", npz.weight(f"{p}_rnn_bf"))
        ln(f"{o}.rnn", f"{p}_rnn_ffn")
        for x in ("q", "k", "v", "o"):
            e.linear(f"{o}.cross.w{x}", f"{p}_context_W{x}")
            e.raw(f"{o}.cross.b{x}", npz.weight(f"{p}_context_b{x}"))
        ln(f"{o}.cross", f"{p}_context_Wo")
        e.linear(f"{o}.ffn.w1", f"{p}_ffn_W1")
        e.raw(f"{o}.ffn.b1", npz.weight(f"{p}_ffn_b1"))
        e.linear(f"{o}.ffn.w2", f"{p}_ffn_W2")
        e.raw(f"{o}.ffn.b2", npz.weight(f"{p}_ffn_b2"))
        ln(f"{o}.ffn", f"{p}_ffn_ffn")

    return e


def _write_meta(w: gguf.GGUFWriter) -> None:
    w.add_uint32("marian.dim", npz.DIM)
    w.add_uint32("marian.heads", npz.HEADS)
    w.add_uint32("marian.head_dim", npz.HEAD_DIM)
    w.add_uint32("marian.enc_depth", npz.ENC_DEPTH)
    w.add_uint32("marian.dec_depth", npz.DEC_DEPTH)
    w.add_uint32("marian.ffn_dim", npz.FFN_DIM)
    w.add_uint32("marian.vocab", npz.VOCAB)
    w.add_uint32("marian.max_seq", _MAX_SEQ)
    w.add_uint32("marian.eos_id", npz.EOS_ID)
    w.add_float32("marian.eps", npz.EPS)
    w.add_float32("marian.embed_scale", npz.EMBED_SCALE)
    w.add_float32("marian.attn_scale", float(npz.ATTN_SCALE))


def _write_float(e: _Emit, path: Path) -> None:
    w = gguf.GGUFWriter(str(path), _ARCH)
    _write_meta(w)
    for name, arr in e.float_tensors:
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _write_q8(e: _Emit, path: Path) -> None:
    w = gguf.GGUFWriter(str(path), _ARCH)
    _write_meta(w)
    for name, arr, do_q8 in e.q8_tensors:
        if do_q8:
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _size_table() -> None:
    fp = _MODELS / "marian.float.gguf"
    qp = _MODELS / "marian.q8_0.gguf"
    fmb, qmb = fp.stat().st_size / 1e6, qp.stat().st_size / 1e6
    print(f"\n{'artifact':<22}{'MB':>8}")
    print(f"{'marian.float.gguf':<22}{fmb:>8.1f}")
    print(f"{'marian.q8_0.gguf':<22}{qmb:>8.1f}")
    print(f"{'ratio (float/q8)':<22}{fmb/qmb:>8.2f}x")


def main() -> None:
    _MODELS.mkdir(exist_ok=True)
    print(
        f"[convert] {npz.ARCHITECTURE} dim={npz.DIM} heads={npz.HEADS} "
        f"enc={npz.ENC_DEPTH} dec={npz.DEC_DEPTH} ffn={npz.FFN_DIM} vocab={npz.VOCAB}"
    )
    e = _collect()
    _write_float(e, _MODELS / "marian.float.gguf")
    _write_q8(e, _MODELS / "marian.q8_0.gguf")
    print(f"[convert] wrote {len(e.float_tensors)} tensors")
    _size_table()


if __name__ == "__main__":
    main()
