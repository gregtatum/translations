#!/usr/bin/env python3
"""Microbenchmark ORT's MatMulInteger (MLAS) at the same shapes as the Rust kernel bench.

The companion to crates/fxtranslate/examples/kernel_micro.rs — together they isolate the
int8 GEMM kernel from the rest of each engine, so notes/17 can decide whether ONNX's lead is
the kernel (MLAS beats gemmology per-shape) or the graph (they tie and the gap is elsewhere).

Single-threaded, CPU, GFLOP/s = 2·m·k·n / time. NOTE: this is the *raw* integer matmul
(MatMulInteger → int32); the Rust bench's matmul_into also does dequant+bias, so read the
comparison with that epilogue difference in mind.

Run: task rs:onnx-kernel-micro   (or: .venv/bin/python3 onnx/kernel_micro.py)
"""

from __future__ import annotations

import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def build(m: int, k: int, n: int) -> bytes:
    a = helper.make_tensor_value_info("A", TensorProto.UINT8, [m, k])
    b = helper.make_tensor_value_info("B", TensorProto.INT8, [k, n])
    y = helper.make_tensor_value_info("Y", TensorProto.INT32, [m, n])
    node = helper.make_node("MatMulInteger", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "mmi", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def bench(m: int, k: int, n: int) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(build(m, k, n), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(m, k), dtype=np.uint8)
    b = rng.integers(-64, 64, size=(k, n)).astype(np.int8)
    feed = {"A": a, "B": b}
    for _ in range(8):
        sess.run(None, feed)
    budget, start, iters = 0.4, time.perf_counter(), 0
    while time.perf_counter() - start < budget:
        for _ in range(16):
            sess.run(None, feed)
        iters += 16
    per = (time.perf_counter() - start) / iters
    gflops = (2.0 * m * k * n) / per / 1e9
    print(f"  m={m:<4} k={k} n={n:<6}  {per * 1e6:>9.2f} us/matmul  {gflops:>7.1f} GFLOP/s")


def main() -> None:
    print("ORT MatMulInteger (MLAS), single-thread — raw int matmul (no dequant)\n")
    print("decoder shapes (small m):")
    for m in (1, 2, 4):
        for n in (512, 2048, 32000):
            bench(m, 512, n)
    print("\nencoder shapes (large m = batch*seq):")
    for m in (64, 256):
        for n in (512, 2048):
            bench(m, 512, n)


if __name__ == "__main__":
    main()
