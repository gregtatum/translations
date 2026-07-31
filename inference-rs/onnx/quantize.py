#!/usr/bin/env python3
"""Dynamic int8 quantization of the exported float32 ONNX graphs.

Runs ``onnxruntime.quantization.quantize_dynamic`` over ``encoder.onnx`` and
``decode_step.onnx`` to produce int8 versions. Dynamic quantization is the right
tool for a transformer with no calibration set: activation scales are computed
per-inference at run time and only the weights (MatMul/Gemm) are stored int8.

Per ORT transformer guidance we quantize weights to QInt8 per-tensor. The op
set is restricted to MatMul so the graph's Gather (embedding lookup), LayerNorm
sub-ops, Softmax, etc. stay float; only the big linear weights shrink, which is
where nearly all the size is.
"""

from __future__ import annotations

import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

_MODELS = Path(__file__).resolve().parent / "models"

_GRAPHS = [
    ("encoder.onnx", "encoder.int8.onnx"),
    ("decode_step.onnx", "decode_step.int8.onnx"),
]


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def quantize_one(src: Path, dst: Path) -> tuple[float, float]:
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=False,
        reduce_range=False,
    )
    return _mb(src), _mb(dst)


def main() -> int:
    total_before = total_after = 0.0
    print(f"{'graph':<24}{'float MB':>12}{'int8 MB':>12}{'ratio':>10}")
    for src_name, dst_name in _GRAPHS:
        src, dst = _MODELS / src_name, _MODELS / dst_name
        if not src.exists():
            print(f"missing {src}; run the export step first")
            return 1
        before, after = quantize_one(src, dst)
        total_before += before
        total_after += after
        print(f"{dst_name:<24}{before:>12.1f}{after:>12.1f}{before / after:>9.2f}x")

    print(
        f"{'TOTAL':<24}{total_before:>12.1f}{total_after:>12.1f}{total_before / total_after:>9.2f}x"
    )
    print(f"size reduction: {100 * (1 - total_after / total_before):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
