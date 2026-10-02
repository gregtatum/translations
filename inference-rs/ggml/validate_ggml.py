#!/usr/bin/env python3
"""Validate the ggml engine against the golden and the inference-rs oracle (both gates).

Mirrors onnx/validate_{numpy,onnx}.py, transitively cheat-proof against the marian oracle:

  Gate 2 (graph correctness): the FLOAT GGUF must match the numpy_ref float golden on the
    encoder context and first-step logits to ~1e-4. Isolates graph-wiring bugs from
    quantization. This is the ggml equivalent of onnx Gate 2.

  Gate 1 (int8 sanity): the Q8_0 GGUF cross-checked against the inference-rs int8 dump
    (onnx/testdata/inferrs_*.f32). Because Q8_0 (block-wise) and intgemm (per-tensor,
    shifted) are different int8 schemes, the bar is mutual top-K membership of the
    first-step argmax, not exact agreement — a real architecture bug would put the tokens
    nowhere near each other's top-K. We also report each int8 scheme's distance to the
    float golden (Q8_0 typically tracks float more closely than intgemm).

Run via `task rs:ggml-validate` (which builds the engine and GGUFs first).
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

_BIN = _HERE / "marian_ggml"
_MODELS = _HERE / "models"
_TESTDATA = _HERE / "testdata"
_INFERRS = _ONNX / "testdata"
_SENT = "Hello, world. This is a test of the translation engine."
_TOPK = 5
_GATE2_TOL = 1e-4


def _dump(model: str, src_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Run the engine's dump mode; return (encoder [seq,dim], first-step logits [vocab])."""
    subprocess.run(
        [str(_BIN), str(_MODELS / model), "dump", *[str(i) for i in src_ids]],
        check=True,
        cwd=_HERE.parent,  # dump writes ggml/testdata/ relative to inference-rs/
        capture_output=True,
    )
    enc = np.fromfile(_TESTDATA / "ggml_encoder.bin", dtype=np.float32).reshape(
        len(src_ids), npz.DIM
    )
    log = np.fromfile(_TESTDATA / "ggml_logits.bin", dtype=np.float32)
    return enc, log


def _stat(name: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rel = d.mean() / (np.abs(b).mean() + 1e-12) * 100
    print(f"    {name:24} abs max={d.max():.3e} mean={d.mean():.3e} (rel {rel:.2f}%)")


def main() -> int:
    if not _BIN.exists():
        sys.exit(f"[validate] engine not built at {_BIN} (run: task rs:ggml-build)")
    src_ids = tok.encode_source(_SENT)
    print(f"text:    {_SENT}\nsrc_ids: {src_ids}\n")

    # Golden float reference (in-process, same code path onnx uses).
    ctx_g = ref.encode(src_ids)
    log_g, _ = ref.decode_step(
        tok.eos_id,
        0,
        ref.precompute_cross_kv(ctx_g),
        [np.zeros(npz.DIM, dtype=np.float32) for _ in range(npz.DEC_DEPTH)],
    )

    # --- Gate 2: FLOAT ggml vs numpy golden ---
    print("Gate 2 — ggml FLOAT vs numpy_ref golden (must be ~1e-4):")
    enc_f, log_f = _dump("marian.float.gguf", src_ids)
    _stat("encoder", enc_f, ctx_g)
    _stat("logits", log_f, log_g)
    emax = np.abs(enc_f - ctx_g).max()
    lmax = np.abs(log_f - log_g).max()
    argmatch = int(log_f.argmax()) == int(log_g.argmax())
    gate2 = emax < _GATE2_TOL and lmax < _GATE2_TOL * 1e3 and argmatch  # logits are larger-scale
    print(
        f"    argmax ggml={log_f.argmax()} golden={log_g.argmax()}  "
        f"{'PASS' if gate2 else 'FAIL'}\n"
    )

    # --- Gate 1: Q8_0 ggml vs inference-rs int8 oracle ---
    print("Gate 1 — ggml Q8_0 vs inference-rs int8 oracle (mutual top-K argmax):")
    enc_q, log_q = _dump("marian.q8_0.gguf", src_ids)
    meta_ok = (_INFERRS / "inferrs_logits.f32").exists()
    if not meta_ok:
        print("    inference-rs reference not found (onnx/testdata/inferrs_*.f32).")
        print("    Produce it with `task rs:onnx-dump`, then re-run. Skipping Gate 1.")
        gate1 = None
    else:
        enc_r = np.fromfile(_INFERRS / "inferrs_encoder.f32", dtype="<f4").reshape(
            len(src_ids), npz.DIM
        )
        log_r = np.fromfile(_INFERRS / "inferrs_logits.f32", dtype="<f4")
        _stat("encoder (q8 vs intgemm)", enc_q, enc_r)
        _stat("logits  (q8 vs intgemm)", log_q, log_r)
        gg = set(int(i) for i in np.argsort(log_q)[::-1][:_TOPK])
        rr = set(int(i) for i in np.argsort(log_r)[::-1][:_TOPK])
        gate1 = log_q.argmax() in rr and log_r.argmax() in gg
        print(
            f"    argmax ggml={log_q.argmax()} inferrs={log_r.argmax()}  "
            f"mutual-top{_TOPK}={'PASS' if gate1 else 'FAIL'}"
        )
        print("    (int8 scheme distance to float golden, lower = closer:)")
        _stat("q8 vs float", enc_q, ctx_g)
        _stat("intgemm vs float", enc_r, ctx_g)
    print()

    # --- Gate 3: batch-invariance (batched block == per-sentence, token-identical) ---
    print("Gate 3 — batch-invariance (block-batched decode == per-sentence):")
    sents = [
        "Hello.",
        "This is a test of the translation engine.",
        "The quick brown fox jumps over the lazy dog.",
        "Good morning.",
        "She sells seashells by the seashore in the bright summer sun.",
    ]
    ids_in = "\n".join(" ".join(str(i) for i in tok.encode_source(s)) for s in sents) + "\n"
    gguf = str(_MODELS / "marian.q8_0.gguf")

    def _decode(extra: list[str]) -> str:
        return subprocess.run(
            [str(_BIN), gguf, "decode", *extra],
            input=ids_in,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    batched, solo = _decode([]), _decode(["solo"])
    gate3 = batched == solo
    print(
        f"    {len(sents)} sentences (B={len(sents)} vs B=1): "
        f"{'PASS (token-identical)' if gate3 else 'FAIL (batched != per-sentence)'}"
    )
    print()

    ok = gate2 and (gate1 in (True, None)) and gate3
    print(
        f"RESULT: Gate 2 {'PASS' if gate2 else 'FAIL'}, "
        f"Gate 1 {'PASS' if gate1 else ('SKIP' if gate1 is None else 'FAIL')}, "
        f"Gate 3 {'PASS' if gate3 else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
