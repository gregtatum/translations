#!/usr/bin/env python3
"""Numeric gate: ONNX graphs vs the numpy float golden.

Checks encoder context, the first decode steps' logits and states, and the
end-to-end greedy translation string against ``numpy_ref``. All tensor diffs
must be <= 1e-4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import model_npz as npz
import numpy_ref as ref
import tokenizer as tok
from engine import Engine

_MODELS = Path(__file__).resolve().parent / "models"
_TOL = 1e-4

_SENTENCES = [
    "Hello, world. This is a test of the translation engine.",
    "The quick brown fox jumps over the lazy dog near the river.",
]


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _encoder_sessions():
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(_MODELS / "encoder.onnx"), providers=providers),
        ort.InferenceSession(str(_MODELS / "decode_step.onnx"), providers=providers),
    )


def check_encoder(enc, text: str) -> bool:
    src = np.asarray(tok.encode_source(text), dtype=np.int64)
    outs = {o.name: v for o, v in zip(enc.get_outputs(), enc.run(None, {"src_ids": src}))}
    expected = ref.encode(list(src))
    diff = np.abs(outs["context"] - expected)
    ok = diff.max() <= _TOL
    print(f"  encoder context  max={diff.max():.3e} mean={diff.mean():.3e}  [{_status(ok)}]")
    return ok


def check_decode_steps(enc, dec, text: str, n_steps: int = 3) -> bool:
    src = list(np.asarray(tok.encode_source(text), dtype=np.int64))
    context = ref.encode(src)
    cross_kv = ref.precompute_cross_kv(context)
    enc_out = {
        o.name: v
        for o, v in zip(
            enc.get_outputs(), enc.run(None, {"src_ids": np.asarray(src, dtype=np.int64)})
        )
    }
    # Feed the batched decoder graph with a batch of one (a leading axis of 1 and an all-zero
    # cross_bias — no padding), then drop that axis to compare against the [DIM]/[VOCAB] golden.
    cross = {}
    for i in range(npz.DEC_DEPTH):
        cross[f"cross_k_{i}"] = enc_out[f"cross_k_{i}"][None]  # [1,seq,DIM]
        cross[f"cross_v_{i}"] = enc_out[f"cross_v_{i}"][None]
    cross["cross_bias"] = np.zeros((1, len(src)), dtype=np.float32)

    onnx_states = {
        f"decoder_state_{i}": np.zeros((1, npz.DIM), dtype=np.float32)
        for i in range(npz.DEC_DEPTH)
    }
    ref_states = [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)]

    ok = True
    prev = tok.eos_id
    for pos in range(n_steps):
        feed = {
            "prev_token": np.array([prev], dtype=np.int64),
            "pe_vec": ref.PE[pos].astype(np.float32),
            # Position 0 takes no embedding, only the PE (see export_decoder.build).
            "embed_gate": np.array(0.0 if pos == 0 else 1.0, dtype=np.float32),
            **cross,
            **onnx_states,
        }
        res = {o.name: v for o, v in zip(dec.get_outputs(), dec.run(None, feed))}
        ref_logits, ref_states = ref.decode_step(prev, pos, cross_kv, ref_states)

        ld = np.abs(res["logits"][0] - ref_logits).max()
        sd = max(
            np.abs(res[f"new_decoder_state_{i}"][0] - ref_states[i]).max()
            for i in range(npz.DEC_DEPTH)
        )
        step_ok = ld <= _TOL and sd <= _TOL
        ok = ok and step_ok
        print(f"  decode step {pos}  logits_max={ld:.3e} state_max={sd:.3e}  [{_status(step_ok)}]")

        onnx_states = {
            f"decoder_state_{i}": res[f"new_decoder_state_{i}"] for i in range(npz.DEC_DEPTH)
        }
        prev = int(np.argmax(ref_logits))  # drive both identically off the golden argmax
    return ok


def check_batch_invariance(engine: Engine, texts: list[str]) -> bool:
    """Decoding sentences together must equal decoding each alone — the property that makes
    the block-batched perf path trustworthy. Compares the batched output to the per-sentence
    output (the latter is itself a batch of one through the same graph)."""
    batch_ids = [tok.encode_source(t) for t in texts]
    batched = engine.greedy_batch(batch_ids)
    ok = True
    for text, ids, got in zip(texts, batch_ids, batched):
        alone = engine.greedy(ids)
        same = got == alone
        ok &= same
        print(
            f"  batched vs alone [{_status(same)}]  {text[:40]!r} ({len(got)} vs {len(alone)} tok)"
        )
    return ok


def check_e2e(engine: Engine, text: str) -> bool:
    onnx_str = engine.translate(text)
    ref_str = ref.translate(text)
    ok = onnx_str == ref_str
    print(f"  onnx: {onnx_str}")
    print(f"  ref : {ref_str}")
    print(f"  end-to-end string match  [{_status(ok)}]")
    return ok


def main() -> int:
    onnx.checker.check_model(onnx.load(str(_MODELS / "encoder.onnx")))
    onnx.checker.check_model(onnx.load(str(_MODELS / "decode_step.onnx")))
    print("onnx.checker: both graphs OK")

    enc, dec = _encoder_sessions()
    engine = Engine()

    all_ok = True
    for text in _SENTENCES:
        print(f"\nsentence: {text!r}")
        all_ok &= check_encoder(enc, text)
        all_ok &= check_decode_steps(enc, dec, text)
        all_ok &= check_e2e(engine, text)

    print("\nbatch-invariance (block-batched decode == per-sentence):")
    all_ok &= check_batch_invariance(engine, _SENTENCES)

    print(f"\nOVERALL: {_status(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
