#!/usr/bin/env python3
"""M2 Gate 3 — chrF parity of the llama.cpp ``marian`` decoder vs G1 on a corpus.

Translates a slice of the shared English corpus through both engines (en->ru), detokenizes
with the target SPM, and reports corpus chrF (llama.cpp hypothesis vs the G1 float anchor)
plus the exact output-id match rate. Both are the SAME weight math, so the FLOAT vs FLOAT
run should be near-identical; the Q8_0 llama.cpp run is compared to G1 Q8_0 as a quantized
sanity check (chrF on par = the SSRU + cross-attn + recurrence combination is sound at scale).

Run via ``task rs:ggml-llama-quality`` (defaults to 100 sentences; pass -n N to change).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sacrebleu.metrics import CHRF

_HERE = Path(__file__).resolve().parent
_ONNX = _HERE.parent / "onnx"
sys.path.insert(0, str(_ONNX))

import tokenizer as tok  # noqa: E402

_DEC = _HERE / "marian_decoder_dump"
_G1 = _HERE / "marian_ggml"
_CORPUS = _HERE.parent / "corpora" / "nllb-en-fr.txt"


def _g1_decode(model: Path, id_lines: list[list[int]]) -> list[list[int]]:
    inp = "\n".join(" ".join(str(i) for i in ids) for ids in id_lines) + "\n"
    out = subprocess.run(
        [str(_G1), str(model), "decode"], input=inp,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [[int(x) for x in line.split()] for line in out]


def _llama_decode(model: Path, id_lines: list[list[int]]) -> list[list[int]]:
    outs = []
    for ids in id_lines:
        r = subprocess.run(
            [str(_DEC), "decode", str(model), "--", *[str(i) for i in ids]],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        outs.append([int(x) for x in r.split()] if r else [])
    return outs


def _report(name: str, hyp: list[list[int]], ref: list[list[int]]) -> None:
    hyp_txt = [tok.decode_ids(h) for h in hyp]
    ref_txt = [tok.decode_ids(r) for r in ref]
    chrf = CHRF().corpus_score(hyp_txt, [ref_txt]).score
    exact = sum(1 for h, r in zip(hyp, ref) if h == r) / max(1, len(ref)) * 100
    print(f"    {name:26} chrF={chrf:6.2f}  exact-id={exact:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="number of corpus sentences")
    args = ap.parse_args()

    if not _DEC.exists():
        sys.exit(f"[quality] driver not built at {_DEC} (run: task rs:ggml-llama-build)")

    lines = _CORPUS.read_text(encoding="utf-8").splitlines()[: args.n]
    id_lines = [tok.encode_source(s) for s in lines]
    print(f"corpus: {_CORPUS.name}  sentences={len(id_lines)}\n")

    models_ll = _HERE / "models"
    models_g1 = _HERE / "models"

    print("Gate 3 — chrF (llama.cpp hypothesis vs G1 anchor), en->ru:")
    # FLOAT vs FLOAT: same weight math, should be ~identical.
    ll_f = _llama_decode(models_ll / "marian-llama.float.gguf", id_lines)
    g1_f = _g1_decode(models_g1 / "marian.float.gguf", id_lines)
    _report("llama float vs G1 float", ll_f, g1_f)

    # Q8_0 sanity if both quantized GGUFs exist.
    ll_q = models_ll / "marian-llama.q8_0.gguf"
    g1_q = models_g1 / "marian.q8_0.gguf"
    if ll_q.exists() and g1_q.exists():
        _report("llama q8_0 vs G1 q8_0", _llama_decode(ll_q, id_lines), _g1_decode(g1_q, id_lines))
        _report("llama q8_0 vs G1 float", _llama_decode(ll_q, id_lines), g1_f)
    else:
        print("    (Q8_0 GGUFs not present; skipping the quantized rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
