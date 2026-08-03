#!/usr/bin/env python3
"""Greedy ONNX translation engine: drive the exported graphs via onnxruntime.

Runs ``encoder.onnx`` once to produce the encoder context and the folded
per-decoder-layer cross-attention K/V, then loops ``decode_step.onnx`` — feeding
back the SSRU state and a precomputed positional-encoding vector each step —
until EOS. The generation loop lives here in the driver, not inside the graph
(no ONNX Loop/Scan), exactly as ``numpy_ref`` and the ``inference-rs`` engine do.

Usually driven via ``task rs:onnx-translate -- --int8 "…"``, which translates a
positional argument or reads stdin line by line; ``--int8`` selects the quantized
graphs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import model_npz as npz
import tokenizer as tok
from numpy_ref import PE

_MODELS = Path(__file__).resolve().parent / "models"
_DEFAULT_TEXT = "Hello, world. This is a test of the translation engine."


class Engine:
    """Both ORT sessions plus the greedy decode loop.

    ``int8=True`` selects the ``*.int8.onnx`` graphs produced by ``quantize.py``;
    the default is the float32 graphs (the high-quality reference path).
    """

    def __init__(self, int8: bool = False, threads: int = 1) -> None:
        # Pin ORT to a fixed thread count (default 1) so perf numbers are comparable to the
        # single-threaded native rs/marian baselines rather than ORT's machine-dependent default.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        suffix = ".int8.onnx" if int8 else ".onnx"
        self.encoder = ort.InferenceSession(
            str(_MODELS / f"encoder{suffix}"), opts, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(_MODELS / f"decode_step{suffix}"), opts, providers=providers
        )

    def greedy_timed(self, src_ids: list[int]) -> tuple[list[int], float, float]:
        """Greedy decode returning (ids, encode_ms, decode_ms).

        encode_ms is the single encoder pass; decode_ms is the whole autoregressive loop.
        Both exclude model load (the sessions are already built), matching the encode/decode
        split the inference-rs and block-bench `[block]` spans report.
        """
        src = np.asarray(src_ids, dtype=np.int64)
        t = time.perf_counter()
        enc_out = self.encoder.run(None, {"src_ids": src})
        encode_ms = (time.perf_counter() - t) * 1000.0
        enc = {o.name: v for o, v in zip(self.encoder.get_outputs(), enc_out)}

        cross = {}
        for i in range(npz.DEC_DEPTH):
            cross[f"cross_k_{i}"] = enc[f"cross_k_{i}"]
            cross[f"cross_v_{i}"] = enc[f"cross_v_{i}"]

        states = {
            f"decoder_state_{i}": np.zeros(npz.DIM, dtype=np.float32) for i in range(npz.DEC_DEPTH)
        }

        seq = len(src_ids)
        max_len = min(math.ceil(2 * seq) + 4, 256)

        out: list[int] = []
        prev = tok.eos_id
        t = time.perf_counter()
        for pos in range(max_len):
            feed = {
                "prev_token": np.array([prev], dtype=np.int64),
                "pe_vec": PE[pos].astype(np.float32),
                **cross,
                **states,
            }
            res = {
                o.name: v for o, v in zip(self.decoder.get_outputs(), self.decoder.run(None, feed))
            }
            token = int(np.argmax(res["logits"]))
            if token == tok.eos_id:
                break
            out.append(token)
            prev = token
            states = {
                f"decoder_state_{i}": res[f"new_decoder_state_{i}"] for i in range(npz.DEC_DEPTH)
            }
        decode_ms = (time.perf_counter() - t) * 1000.0
        return out, encode_ms, decode_ms

    def greedy(self, src_ids: list[int]) -> list[int]:
        return self.greedy_timed(src_ids)[0]

    def translate(self, text: str) -> str:
        return tok.decode_ids(self.greedy(tok.encode_source(text)))


_ENGINES: dict[bool, Engine] = {}


def translate(text: str, int8: bool = False) -> str:
    """Module-level convenience wrapper reusing one Engine per precision."""
    if int8 not in _ENGINES:
        _ENGINES[int8] = Engine(int8=int8)
    return _ENGINES[int8].translate(text)


def _main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Greedy translation through the exported ONNX graphs"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to translate. If omitted, translate stdin line by line (a sample sentence on a TTY).",
    )
    parser.add_argument("--int8", action="store_true", help="Use the quantized int8 graphs")
    args = parser.parse_args(argv)

    # Load the sessions once, then translate the positional text, or each stdin
    # line in turn — the same interface as `fxtranslate-oracle translate`.
    engine = Engine(int8=args.int8)
    if args.text is not None:
        print(engine.translate(args.text))
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line:
                print(engine.translate(line))
    else:
        print(engine.translate(_DEFAULT_TEXT))


if __name__ == "__main__":
    _main(sys.argv[1:])
