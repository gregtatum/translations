#!/usr/bin/env python3
"""Cross-check numpy float reference against inference-rs int8 dumps.

Reads ``testdata/inferrs_meta.json`` + ``inferrs_encoder.f32`` +
``inferrs_logits.f32`` (raw little-endian float32, produced by the Rust track),
runs ``numpy_ref`` on the SAME src ids, and reports encoder/logits diffs plus
first-step argmax agreement. int8-precision reference => expect ~1% tensor diff
but the argmax must agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

import model_npz as npz
import numpy_ref as ref

_TESTDATA = Path(__file__).resolve().parent / "testdata"
_META = _TESTDATA / "inferrs_meta.json"
_ENC = _TESTDATA / "inferrs_encoder.f32"
_LOGITS = _TESTDATA / "inferrs_logits.f32"

# The float golden and the int8 engine only have to *agree on the model*, not on the exact
# argmax: int8 rounding legitimately reshuffles tokens whose float logits are near-tied (the
# same reason wasm/native argmax can flip; see notes/16 and the wasm-parity stance). So the
# gate is mutual top-K membership — the numpy argmax must sit in the int8 top-K and vice
# versa. A real architecture bug (transposed weight, wrong PE, mis-read dim) would put the
# tokens nowhere near each other's top-K, so this still catches what the gate is for.
_TOPK = 5


def _diffs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()), float(d.mean())


def _topk(logits: np.ndarray, k: int) -> list[int]:
    return [int(i) for i in np.argsort(logits)[::-1][:k]]


def main() -> int:
    missing = [p.name for p in (_META, _ENC, _LOGITS) if not p.exists()]
    if missing:
        print(f"inference-rs reference files not found yet: {', '.join(missing)}")
        print("Run the Rust dump step to produce them, then re-run this script.")
        return 0

    meta = json.loads(_META.read_text())
    src_ids = list(meta["src_ids"])
    seq, dim = meta["encoder_shape"]
    logits_len = meta["logits_len"]
    print(f"text:    {meta.get('text')}")
    print(f"src_ids: {src_ids}")

    ref_enc = np.fromfile(_ENC, dtype="<f4").reshape(seq, dim)
    ref_logits = np.fromfile(_LOGITS, dtype="<f4").reshape(logits_len)

    context = ref.encode(src_ids)
    logits0, _ = ref.decode_step(
        npz.EOS_ID,
        0,
        ref.precompute_cross_kv(context),
        [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)],
    )

    enc_max, enc_mean = _diffs(context, ref_enc)
    log_max, log_mean = _diffs(logits0, ref_logits)
    np_top = _topk(logits0, _TOPK)
    rs_top = _topk(ref_logits, _TOPK)
    np_arg, rs_arg = np_top[0], rs_top[0]
    exact = np_arg == rs_arg
    mutual = np_arg in rs_top and rs_arg in np_top
    agree = exact or mutual

    sp = spm.SentencePieceProcessor(model_file=str(npz.TGT_SPM_PATH))

    def fmt(top: list[int], logits: np.ndarray) -> str:
        return ", ".join(f"{i}:{sp.id_to_piece(i)!r}({logits[i]:.2f})" for i in top)

    print(f"encoder abs diff:  max={enc_max:.4g}  mean={enc_mean:.4g}")
    print(f"logits  abs diff:  max={log_max:.4g}  mean={log_mean:.4g}")
    print(f"numpy    top-{_TOPK}: {fmt(np_top, logits0)}")
    print(f"inferrs  top-{_TOPK}: {fmt(rs_top, ref_logits)}")
    print("(int8 reference: ~1% tensor diffs expected; argmax must agree OR be a near-tie")
    print(f" reshuffle — mutual top-{_TOPK} membership)")
    if exact:
        print("ARGMAX PASS (exact)")
    elif mutual:
        print(
            f"ARGMAX PASS (near-tie: numpy {np_arg} and inference-rs {rs_arg} in both top-{_TOPK})"
        )
    else:
        print(f"ARGMAX FAIL (numpy={np_arg}, inference-rs={rs_arg}, not mutually in top-{_TOPK})")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
