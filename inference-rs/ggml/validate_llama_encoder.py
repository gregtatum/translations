#!/usr/bin/env python3
"""M1 gate — the llama.cpp ``marian`` encoder vs the numpy_ref float golden.

Feeds the fixed sentence's EXACT source ids (the same ids G1 and numpy_ref use) through
``ggml/marian_encoder_dump`` (a libllama driver that runs LLM_GRAPH_TYPE_ENCODER with
pooling NONE + embeddings, then reads llama_get_embeddings) and diffs the full per-token
encoder context [seq, dim] against ``numpy_ref.encode``. PASS = abs max < 1e-4.

Run via ``task rs:ggml-llama-encoder`` (builds the CPU libllama, driver, and GGUF first).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ONNX = _HERE.parent / "onnx"
sys.path.insert(0, str(_ONNX))

import model_npz as npz  # noqa: E402
import numpy_ref as ref  # noqa: E402
import tokenizer as tok  # noqa: E402

_BIN = _HERE / "marian_encoder_dump"
_MODEL = _HERE / "models" / "marian-llama.float.gguf"
_SENT = "Hello, world. This is a test of the translation engine."
_TOL = 1e-4


def main() -> int:
    if not _BIN.exists():
        sys.exit(f"[validate] driver not built at {_BIN} (run: task rs:ggml-llama-encoder)")
    if not _MODEL.exists():
        sys.exit(f"[validate] GGUF not found at {_MODEL} (run the converter first)")

    src_ids = tok.encode_source(_SENT)
    print(f"text:    {_SENT}\nsrc_ids: {src_ids}\n")

    out = _HERE / "testdata" / "llama_encoder.bin"
    out.parent.mkdir(exist_ok=True)
    subprocess.run(
        [str(_BIN), str(_MODEL), str(out), *[str(i) for i in src_ids]],
        check=True, capture_output=True,
    )

    enc_llama = np.fromfile(out, dtype=np.float32).reshape(len(src_ids), npz.DIM)
    ctx_golden = ref.encode(src_ids)  # numpy_ref float golden [seq, dim]

    d = np.abs(enc_llama.astype(np.float64) - ctx_golden.astype(np.float64))
    amax, amean = d.max(), d.mean()
    rel = amean / (np.abs(ctx_golden).mean() + 1e-12) * 100
    passed = amax < _TOL

    print("M1 gate — llama.cpp marian encoder vs numpy_ref golden (must be < 1e-4):")
    print(f"    encoder [{len(src_ids)},{npz.DIM}]  abs max={amax:.3e}  mean={amean:.3e}  (rel {rel:.3f}%)")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} (abs max {amax:.3e} vs tol {_TOL:.0e})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
