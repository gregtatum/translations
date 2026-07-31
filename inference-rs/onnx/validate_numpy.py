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

import model_npz as npz
import numpy_ref as ref

_TESTDATA = Path(__file__).resolve().parent / "testdata"
_META = _TESTDATA / "inferrs_meta.json"
_ENC = _TESTDATA / "inferrs_encoder.f32"
_LOGITS = _TESTDATA / "inferrs_logits.f32"


def _diffs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()), float(d.mean())


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
    np_arg = int(np.argmax(logits0))
    rs_arg = int(np.argmax(ref_logits))
    agree = np_arg == rs_arg

    print(f"encoder abs diff:  max={enc_max:.4g}  mean={enc_mean:.4g}")
    print(f"logits  abs diff:  max={log_max:.4g}  mean={log_mean:.4g}")
    print(f"first-step argmax: numpy={np_arg}  inference-rs={rs_arg}")
    print("(int8 reference: ~1% tensor diffs are expected; argmax must agree)")
    print(f"ARGMAX {'PASS' if agree else 'FAIL'}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
