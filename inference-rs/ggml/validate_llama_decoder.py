#!/usr/bin/env python3
"""Decoder gate — the llama.cpp ``marian`` SSRU decoder end-to-end vs the numpy_ref/G1 goldens.

Two gates on the fixed sentence (the same source ids the encoder gate uses), driven through
``ggml/marian_decoder_dump`` (a libllama two-phase driver: llama_encode -> greedy llama_decode,
threading the SSRU recurrent cell state via llama.cpp's recurrent memory):

  Gate 1 — first-step float logits: with the recurrent state zero, the [vocab] logits must
    match numpy_ref.decode_step to ~1e-3 with an identical argmax. Validates the decoder graph
    wiring (SSRU s_prev=0, cross-attn, FFN) BEFORE recurrence is exercised. Note the first
    decoder input is decoder_start (EOS) but its *embedding* is gated off: marian zero-pads
    the shifted target embeddings at position 0, so only the PE enters step 0 and the start
    token is inert (see notes/23-float-model-support.md).

  Gate 2 — end-to-end greedy: the full greedy loop's output ids must match G1's float greedy
    output for the fixed sentence. Exercises the recurrent state threading across steps.

Run via ``task rs:ggml-llama-decoder`` (builds libllama, the driver, and the GGUF first).
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

_DEC = _HERE / "marian_decoder_dump"
_G1 = _HERE / "marian_ggml"
_MODEL = _HERE / "models" / "marian-llama.float.gguf"
_G1_MODEL = _HERE / "models" / "marian.float.gguf"
_SENT = "Hello, world. This is a test of the translation engine."
_LOGITS_TOL = 1e-3  # first-step logits are O(10); 1e-3 abs is a tight graph-correctness bar


def main() -> int:
    if not _DEC.exists():
        sys.exit(f"[validate] driver not built at {_DEC} (run: task rs:ggml-llama-decoder)")
    if not _MODEL.exists():
        sys.exit(f"[validate] GGUF not found at {_MODEL} (run the converter first)")

    src_ids = tok.encode_source(_SENT)
    print(f"text:    {_SENT}\nsrc_ids: {src_ids}\n")

    # --- Gate 1: first-step float logits vs numpy_ref golden ---
    out_bin = _HERE / "testdata" / "llama_logits.bin"
    out_bin.parent.mkdir(exist_ok=True)
    subprocess.run(
        [str(_DEC), "dump", str(_MODEL), str(out_bin), *[str(i) for i in src_ids]],
        check=True,
        capture_output=True,
    )
    log_llama = np.fromfile(out_bin, dtype=np.float32)

    ctx_g = ref.encode(src_ids)
    log_g, _ = ref.decode_step(
        tok.eos_id,
        0,
        ref.precompute_cross_kv(ctx_g),
        [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)],
    )
    d = np.abs(log_llama.astype(np.float64) - log_g.astype(np.float64))
    argmatch = int(log_llama.argmax()) == int(log_g.argmax())
    gate1 = d.max() < _LOGITS_TOL and argmatch
    print("Gate 1 — first-step float logits vs numpy_ref (must be < 1e-3, argmax match):")
    print(f"    logits [{log_llama.size}]  abs max={d.max():.3e}  mean={d.mean():.3e}")
    print(
        f"    argmax llama={log_llama.argmax()} golden={log_g.argmax()}  "
        f"{'PASS' if gate1 else 'FAIL'}\n"
    )

    # --- Gate 2: end-to-end greedy output ids vs G1 float ---
    llama_out = subprocess.run(
        [str(_DEC), "decode", str(_MODEL), "--", *[str(i) for i in src_ids]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    g1_out = "?"
    have_g1 = _G1.exists() and _G1_MODEL.exists()
    if have_g1:
        g1_out = subprocess.run(
            [str(_G1), str(_G1_MODEL), "decode"],
            input=" ".join(str(i) for i in src_ids) + "\n",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        gate2 = llama_out == g1_out
    else:
        gate2 = None

    print("Gate 2 — end-to-end greedy output ids vs G1 float:")
    print(f"    llama.cpp: {llama_out}")
    if have_g1:
        print(f"    G1:        {g1_out}")
        print(f"    {'PASS (id-identical)' if gate2 else 'FAIL (ids differ)'}\n")
    else:
        print(
            "    (G1 engine/GGUF not built; skipping the id comparison — run task rs:ggml-build)\n"
        )

    ok = gate1 and (gate2 in (True, None))
    print(
        f"RESULT: Gate 1 {'PASS' if gate1 else 'FAIL'}, "
        f"Gate 2 {'PASS' if gate2 else ('SKIP' if gate2 is None else 'FAIL')}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
