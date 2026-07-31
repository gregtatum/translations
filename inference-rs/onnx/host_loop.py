"""Greedy translation driving the exported ONNX graphs via onnxruntime.

Runs ``encoder.onnx`` once to get the context and folded cross K/V, then loops
``decode_step.onnx`` with host-computed PE vectors and state feedback until EOS.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

import model_npz as M
import tokenizer as T
from numpy_ref import _PE

_MODELS = Path(__file__).resolve().parent / "models"


class Translator:
    """Holds both ORT sessions and runs the greedy loop.

    ``int8=True`` selects the ``*.int8.onnx`` graphs produced by ``quantize.py``;
    the default is the float32 graphs (the high-quality reference path).
    """

    def __init__(self, int8: bool = False) -> None:
        opts = ort.SessionOptions()
        prov = ["CPUExecutionProvider"]
        suffix = ".int8.onnx" if int8 else ".onnx"
        self.encoder = ort.InferenceSession(str(_MODELS / f"encoder{suffix}"), opts, providers=prov)
        self.decoder = ort.InferenceSession(str(_MODELS / f"decode_step{suffix}"), opts, providers=prov)

    def greedy(self, src_ids: list[int]) -> list[int]:
        src = np.asarray(src_ids, dtype=np.int64)
        enc = {o.name: v for o, v in zip(self.encoder.get_outputs(), self.encoder.run(None, {"src_ids": src}))}

        cross = {}
        for i in range(M.DEC_DEPTH):
            cross[f"cross_k_{i}"] = enc[f"cross_k_{i}"]
            cross[f"cross_v_{i}"] = enc[f"cross_v_{i}"]

        states = {f"decoder_state_{i}": np.zeros(M.DIM, dtype=np.float32) for i in range(M.DEC_DEPTH)}

        seq = len(src_ids)
        max_len = min(math.ceil(2 * seq) + 4, 256)

        out: list[int] = []
        prev = T.eos_id
        for pos in range(max_len):
            feed = {
                "prev_token": np.array([prev], dtype=np.int64),
                "pe_vec": _PE[pos].astype(np.float32),
                **cross,
                **states,
            }
            res = {o.name: v for o, v in zip(self.decoder.get_outputs(), self.decoder.run(None, feed))}
            tok = int(np.argmax(res["logits"]))
            if tok == T.eos_id:
                break
            out.append(tok)
            prev = tok
            states = {f"decoder_state_{i}": res[f"new_decoder_state_{i}"] for i in range(M.DEC_DEPTH)}
        return out

    def translate(self, text: str) -> str:
        return T.decode_ids(self.greedy(T.encode_source(text)))


_TRANSLATORS: dict[bool, Translator] = {}


def translate(text: str, int8: bool = False) -> str:
    """Module-level convenience wrapper reusing a Translator per precision."""
    if int8 not in _TRANSLATORS:
        _TRANSLATORS[int8] = Translator(int8=int8)
    return _TRANSLATORS[int8].translate(text)


def _main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="ONNX greedy translation")
    ap.add_argument("text", nargs="?",
                    default="Hello, world. This is a test of the translation engine.")
    ap.add_argument("--int8", action="store_true", help="use the quantized int8 graphs")
    args = ap.parse_args(argv)
    print(translate(args.text, int8=args.int8))


if __name__ == "__main__":
    _main(sys.argv[1:])
