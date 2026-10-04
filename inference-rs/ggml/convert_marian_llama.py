#!/usr/bin/env python3
"""Convert the float Marian/Bergamot student .npz to a llama.cpp ``marian`` GGUF.

This is the llama.cpp converter: unlike ``convert_marian_gguf.py`` (bespoke tensor names
for the bare-libggml engine), this packages the *same* proven weight math into the shape
llama.cpp's ``LLM_ARCH_MARIAN`` expects — standard KV keys, ``enc.blk.N.*`` / ``dec.blk.N.*``
tensor names, and the SentencePiece vocab embedded exactly as ``convert_spm_gguf.py`` does
(so llama.cpp selects the UGM tokenizer, ``tokenizer.ggml.model = "t5"``).

The llama.cpp arch wires the full two-phase encoder + SSRU decoder; this emits the complete
encoder + decoder (SSRU + cross-attention) tensor set, ``output.bias``, and hparams, in both
a float GGUF (numeric parity gate) and a Q8_0 GGUF (top-K + chrF + perf).

Weight orientation is unchanged from the bare-libggml converter: each linear is stored
transposed to numpy shape
``(out, in)`` → ggml ``ne0=in, ne1=out`` so ``ggml_mul_mat(W, x)`` gives ``y[o]=sum_i W[i,o]*x[i]``.
The sinusoidal absolute positional encoding is baked into ``position_embd`` (indexed by
absolute position in the graph) — the same table ``numpy_ref.build_positional_encoding``
computes, so the positional encoding is never a divergence source.

Tensor-name scheme (must match src/llama-arch.cpp LLM_TENSOR_NAMES exactly):
    token_embd.weight, position_embd.weight, output.weight, output.bias
    enc.blk.N.attn_q/attn_k/attn_v/attn_o     (.weight + .bias)
    enc.blk.N.attn_norm                       (.weight + .bias)  post-attention LayerNorm
    enc.blk.N.ffn_up/ffn_down                 (.weight + .bias)
    enc.blk.N.ffn_norm                        (.weight + .bias)  post-FFN LayerNorm
    dec.blk.N.rnn                             (.weight)          SSRU candidate (no bias)
    dec.blk.N.rnn_f                           (.weight + .bias)  SSRU forget gate
    dec.blk.N.rnn_norm                        (.weight + .bias)  SSRU cell LayerNorm
    dec.blk.N.cross_attn_q/k/v/o              (.weight + .bias)
    dec.blk.N.cross_attn_norm                 (.weight + .bias)
    dec.blk.N.ffn_up/ffn_down                 (.weight + .bias)
    dec.blk.N.ffn_norm                        (.weight + .bias)

Run via ``task rs:ggml-llama-convert``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_HERE = Path(__file__).resolve().parent
_ONNX = _HERE.parent / "onnx"
sys.path.insert(0, str(_ONNX))  # reuse the ONNX eval's validated model loader

import model_npz as npz  # noqa: E402

from sentencepiece import SentencePieceProcessor  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as spm_model  # noqa: E402

import gguf  # noqa: E402

_MODELS = _HERE / "models"
_ARCH = "marian"
_MAX_SEQ = 256


def _build_positional_encoding(max_seq: int) -> np.ndarray:
    """Sinusoidal rotor positional-encoding table, byte-for-byte the formula in
    numpy_ref.build_positional_encoding.

    Returns shape (max_seq, DIM); GGUF stores it row-major as ne0=DIM, ne1=max_seq, so
    ggml_get_rows(position_embd, pos) yields the DIM-vector for absolute position `pos`.
    """
    d = npz.DIM
    half = d // 2
    c = np.arange(d)
    freq = np.power(1e-4, (c % half) / (half - 1))
    offs = np.floor(c / half) * (math.pi / 2.0)
    pos = np.arange(max_seq)[:, None]
    return np.sin(pos * freq[None, :] + offs[None, :]).astype(np.float32)


def _linear(npz_name: str) -> np.ndarray:
    """A matmul weight: npz [in,out] -> ggml (out,in)."""
    w = npz.weight(npz_name)
    assert w.ndim == 2, f"{npz_name} is not a matrix: {w.shape}"
    return np.ascontiguousarray(w.T).astype(np.float32)  # (out, in) -> ggml ne0=in, ne1=out


# A collected tensor: (name, float array, quantizable). `quant=True` marks a matmul weight the
# Q8_0 GGUF quantizes; everything else (biases, LayerNorm, embeddings, positional
# encoding) stays F32/F16.
def _collect(include_decoder: bool = False) -> list[tuple[str, np.ndarray, bool]]:
    """Encoder-only by default. `include_decoder` bakes in the SSRU decoder."""
    out: list[tuple[str, np.ndarray, bool]] = []

    def lin(name: str, npz_name: str) -> None:
        out.append((name, _linear(npz_name), True))

    def raw(name: str, arr: np.ndarray) -> None:
        out.append((name, arr.astype(np.float32), False))

    wemb = npz.wemb().astype(np.float32)  # (vocab, dim)
    raw("token_embd.weight", wemb)  # get_rows lookup (F16 is a later perf tweak)
    # baked sinusoidal positional encoding
    raw("position_embd.weight", _build_positional_encoding(_MAX_SEQ))
    out.append(("output.weight", wemb, True))  # tied projection: quantized like the linears
    if include_decoder:
        raw("output.bias", npz.logit_bias())  # decoder logit bias

    def ln(name: str, npz_prefix: str) -> None:
        raw(f"{name}.weight", npz.weight(f"{npz_prefix}_ln_scale"))
        raw(f"{name}.bias", npz.weight(f"{npz_prefix}_ln_bias"))

    # encoder layers (npz names are 1-based; GGUF names 0-based).
    for l in range(npz.ENC_DEPTH):
        p, o = f"encoder_l{l+1}", f"enc.blk.{l}"
        for src, dst in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_o")):
            lin(f"{o}.{dst}.weight", f"{p}_self_W{src}")
            raw(f"{o}.{dst}.bias", npz.weight(f"{p}_self_b{src}"))
        ln(f"{o}.attn_norm", f"{p}_self_Wo")  # post-attention LayerNorm
        lin(f"{o}.ffn_up.weight", f"{p}_ffn_W1")
        raw(f"{o}.ffn_up.bias", npz.weight(f"{p}_ffn_b1"))
        lin(f"{o}.ffn_down.weight", f"{p}_ffn_W2")
        raw(f"{o}.ffn_down.bias", npz.weight(f"{p}_ffn_b2"))
        ln(f"{o}.ffn_norm", f"{p}_ffn_ffn")  # post-FFN LayerNorm

    # decoder layers: SSRU + cross-attention + FFN.
    for l in range(npz.DEC_DEPTH) if include_decoder else range(0):
        p, o = f"decoder_l{l+1}", f"dec.blk.{l}"
        lin(f"{o}.rnn.weight", f"{p}_rnn_W")  # candidate, no bias
        lin(f"{o}.rnn_f.weight", f"{p}_rnn_Wf")
        raw(f"{o}.rnn_f.bias", npz.weight(f"{p}_rnn_bf"))
        ln(f"{o}.rnn_norm", f"{p}_rnn_ffn")
        for src, dst in (
            ("q", "cross_attn_q"),
            ("k", "cross_attn_k"),
            ("v", "cross_attn_v"),
            ("o", "cross_attn_o"),
        ):
            lin(f"{o}.{dst}.weight", f"{p}_context_W{src}")
            raw(f"{o}.{dst}.bias", npz.weight(f"{p}_context_b{src}"))
        ln(f"{o}.cross_attn_norm", f"{p}_context_Wo")
        lin(f"{o}.ffn_up.weight", f"{p}_ffn_W1")
        raw(f"{o}.ffn_up.bias", npz.weight(f"{p}_ffn_b1"))
        lin(f"{o}.ffn_down.weight", f"{p}_ffn_W2")
        raw(f"{o}.ffn_down.bias", npz.weight(f"{p}_ffn_b2"))
        ln(f"{o}.ffn_norm", f"{p}_ffn_ffn")

    return out


def _toktype(sp: SentencePieceProcessor, i: int) -> int:
    if sp.IsUnknown(i):
        return int(gguf.TokenType.UNKNOWN)
    if sp.IsControl(i):
        return int(gguf.TokenType.CONTROL)
    if sp.IsUnused(i):
        return int(gguf.TokenType.UNUSED)
    if sp.IsByte(i):
        return int(gguf.TokenType.BYTE)
    return int(gguf.TokenType.NORMAL)


def _add_vocab(w: gguf.GGUFWriter, spm_path: Path) -> None:
    """Embed the SentencePiece vocab exactly as convert_spm_gguf.py (UGM tokenizer path)."""
    mp = spm_model.ModelProto()
    mp.ParseFromString(spm_path.read_bytes())
    ns = mp.normalizer_spec
    assert mp.trainer_spec.model_type == 1, "expected a UNIGRAM SentencePiece model"

    sp = SentencePieceProcessor()
    sp.LoadFromFile(str(spm_path))
    n = sp.vocab_size()

    w.add_tokenizer_model("t5")  # -> LLAMA_VOCAB_TYPE_UGM
    w.add_tokenizer_pre("default")
    w.add_token_list([sp.IdToPiece(i).encode("utf-8") for i in range(n)])
    w.add_token_scores([sp.GetScore(i) for i in range(n)])
    w.add_token_types([_toktype(sp, i) for i in range(n)])

    assert ns.escape_whitespaces is True, "Marian uses escaped whitespace; UGM default matches"
    assert mp.trainer_spec.treat_whitespace_as_suffix is False, "UGM prepends the space marker"
    w.add_add_space_prefix(ns.add_dummy_prefix)
    w.add_remove_extra_whitespaces(ns.remove_extra_whitespaces)
    if ns.precompiled_charsmap:
        w.add_precompiled_charsmap(ns.precompiled_charsmap)

    if sp.unk_id() >= 0:
        w.add_unk_token_id(sp.unk_id())
    if sp.eos_id() >= 0:
        w.add_eos_token_id(sp.eos_id())
    if sp.bos_id() >= 0:
        w.add_bos_token_id(sp.bos_id())
    if sp.pad_id() >= 0:
        w.add_pad_token_id(sp.pad_id())
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    return n


def _write_meta(w: gguf.GGUFWriter, ftype: gguf.LlamaFileType) -> None:
    w.add_name("marian-en-ru")
    # standard KV keys
    w.add_context_length(_MAX_SEQ)
    w.add_embedding_length(npz.DIM)
    w.add_block_count(npz.ENC_DEPTH)  # encoder depth (n_layer)
    w.add_decoder_block_count(npz.DEC_DEPTH)  # SSRU decoder depth
    w.add_feed_forward_length(npz.FFN_DIM)
    w.add_head_count(npz.HEADS)
    w.add_head_count_kv(npz.HEADS)
    w.add_key_length(npz.HEAD_DIM)
    w.add_value_length(npz.HEAD_DIM)
    w.add_layer_norm_eps(npz.EPS)  # {arch}.attention.layer_norm_epsilon
    w.add_embedding_scale(npz.EMBED_SCALE)  # {arch}.embedding_scale = sqrt(dim)
    # 1/sqrt(head_dim); the graph derives the same if this key is absent.
    w.add_key_value("marian.attention.scale", float(npz.ATTN_SCALE), gguf.GGUFValueType.FLOAT32)
    w.add_decoder_start_token_id(npz.EOS_ID)  # Marian starts decoding from </s>
    w.add_file_type(ftype)


def _write(tensors, out: Path, quant: bool) -> None:
    w = gguf.GGUFWriter(str(out), _ARCH)
    _write_meta(w, gguf.LlamaFileType.MOSTLY_Q8_0 if quant else gguf.LlamaFileType.ALL_F32)
    n = _add_vocab(w, npz.SRC_SPM_PATH)
    for name, arr, quantizable in tensors:
        if quant and quantizable:
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    mb = out.stat().st_size / 1e6
    print(f"[convert-llama] wrote {len(tensors)} tensors + {n}-piece vocab -> {out} ({mb:.1f} MB)")


def main() -> None:
    _MODELS.mkdir(exist_ok=True)
    print(
        f"[convert-llama] {npz.ARCHITECTURE} dim={npz.DIM} heads={npz.HEADS} "
        f"enc={npz.ENC_DEPTH} dec={npz.DEC_DEPTH} ffn={npz.FFN_DIM} vocab={npz.VOCAB}"
    )
    tensors = _collect(include_decoder=True)
    _write(tensors, _MODELS / "marian-llama.float.gguf", quant=False)  # Gate-1/2 numeric parity
    _write(tensors, _MODELS / "marian-llama.q8_0.gguf", quant=True)  # top-K + chrF + perf


if __name__ == "__main__":
    main()
