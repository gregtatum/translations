#!/usr/bin/env python3
"""Quality gate: float ONNX vs int8 ONNX vs inference-rs intgemm int8.

The float ONNX path (``engine``) is the high-quality reference. We measure how
much int8 quantization degrades it and whether ORT's dynamic int8 is competitive
with inference-rs's production intgemm int8.

No parallel en->fr reference set exists in the repo (``corpora/nllb-en-fr.txt`` is
English source only), so this is an engine-vs-engine study, not absolute BLEU: we
report chrF and overlap of each engine pair, treating float ONNX as the anchor.

Metrics per engine pair:
  * chrF (sacrebleu, corpus-level) — hypothesis vs reference engine.
  * exact-match rate — fraction of sentences that are byte-identical.
  * mean token overlap — mean Jaccard over whitespace tokens.

A rough per-sentence wall-clock is also reported (indicative only, ORT defaults,
single-threaded).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from sacrebleu.metrics import CHRF

import engine
from model_npz import _REPO_ROOT, SRC_SPM_PATH, TGT_SPM_PATH

_ORACLE = _REPO_ROOT / "inference-rs" / "target" / "release" / "fxtranslate-oracle"
# The intgemm int8 bin for the same production model the ONNX graphs are built from
# (en-ru base v3.0); its vocab is the resolved model's vocab (shared for en-ru).
_INT8_BIN = _REPO_ROOT / "data" / "models" / "enru" / "model.enru.intgemm.alphas.bin"
_SRC_SPM = SRC_SPM_PATH
_TGT_SPM = TGT_SPM_PATH
_CORPUS = _REPO_ROOT / "inference-rs" / "corpora" / "nllb-en-fr.txt"

_N = 50


def load_dev_set() -> list[str]:
    """Return ~50 varied English source sentences from the NLLB corpus."""
    lines = [ln.strip() for ln in _CORPUS.read_text().splitlines() if ln.strip()]
    # Spread the pick across the file for variety rather than the first N.
    step = max(1, len(lines) // _N)
    return lines[::step][:_N]


def onnx_translate(sentences: list[str], int8: bool) -> tuple[list[str], float]:
    out, t0 = [], time.perf_counter()
    for s in sentences:
        out.append(engine.translate(s, int8=int8))
    return out, (time.perf_counter() - t0) / len(sentences)


def inferrs_translate(sentences: list[str]) -> tuple[list[str], float]:
    """Batch all sentences through one oracle process over stdin."""
    cmd = [str(_ORACLE), "translate", str(_INT8_BIN), str(_SRC_SPM), str(_TGT_SPM)]
    payload = "\n".join(sentences) + "\n"
    t0 = time.perf_counter()
    res = subprocess.run(cmd, input=payload, capture_output=True, text=True, check=True)
    dt = (time.perf_counter() - t0) / len(sentences)
    out = [ln for ln in res.stdout.splitlines()]
    if len(out) != len(sentences):
        raise RuntimeError(f"oracle returned {len(out)} lines, expected {len(sentences)}")
    return out, dt


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def pair_stats(hyp: list[str], ref: list[str]) -> tuple[float, float, float]:
    chrf = CHRF().corpus_score(hyp, [ref]).score
    exact = sum(h == r for h, r in zip(hyp, ref)) / len(hyp)
    overlap = sum(_jaccard(h, r) for h, r in zip(hyp, ref)) / len(hyp)
    return chrf, exact, overlap


def main() -> int:
    if not _ORACLE.exists():
        print(
            f"oracle binary not found: {_ORACLE}\nbuild it: cargo build --release -p fxtranslate-oracle"
        )
        return 1

    src = load_dev_set()
    print(f"dev set: {len(src)} English sentences from {_CORPUS.name}\n")

    float_onnx, t_float = onnx_translate(src, int8=False)
    int8_onnx, t_int8 = onnx_translate(src, int8=True)
    inferrs, t_rs = inferrs_translate(src)

    pairs = [
        ("int8-ONNX vs float-ONNX", int8_onnx, float_onnx),
        ("inference-rs vs float-ONNX", inferrs, float_onnx),
        ("int8-ONNX vs inference-rs", int8_onnx, inferrs),
    ]

    print(f"{'pair':<30}{'chrF':>8}{'exact':>8}{'overlap':>9}")
    for name, hyp, ref in pairs:
        chrf, exact, overlap = pair_stats(hyp, ref)
        print(f"{name:<30}{chrf:>8.2f}{exact * 100:>7.0f}%{overlap * 100:>8.0f}%")

    print("\nperf (mean wall-clock per sentence, single-threaded, ORT defaults; indicative only):")
    print(f"  float ONNX   {t_float * 1000:>8.1f} ms/sent")
    print(f"  int8 ONNX    {t_int8 * 1000:>8.1f} ms/sent")
    print(f"  inference-rs {t_rs * 1000:>8.1f} ms/sent  (includes process + model load)")

    fixed = "Hello, world. This is a test of the translation engine."
    print(f"\nfixed sentence:\n  src : {fixed}")
    print(f"  flt : {engine.translate(fixed, int8=False)}")
    print(f"  int8: {engine.translate(fixed, int8=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
