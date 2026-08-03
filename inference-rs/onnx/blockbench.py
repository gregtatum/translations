#!/usr/bin/env python3
"""Block-benchmark the ONNX engine, emitting the same `[block]` spans as the native tools.

The perf harness (scripts/final_comparison.py) compares inference-rs, marian block-bench,
and this ONNX engine on the same block corpus. Each engine loads its model once and prints
one `[block] {...}` line per block to stderr with the per-block compute time (model load
excluded); the harness sums those for words/s and samples RSS around the process. This makes
the ONNX path a peer subject rather than a separate, non-comparable measurement.

Each block's sentences are decoded as one lockstep batch (padded cross K/V + a source mask,
like inference-rs and block-bench), so `decode_ms` is the batched cost. The encoder still runs
once per sentence, so a block isn't batched fully end to end — `encode_ms` is the summed
per-sentence encoder cost. ORT is pinned single-threaded to match the native baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import engine as onnx_engine
import tokenizer as tok


def read_blocks(path: str | None) -> list[list[str]]:
    """Blocks are blank-line-separated; each non-empty line in a block is a sentence."""
    text = Path(path).read_text() if path else sys.stdin.read()
    blocks = []
    for chunk in text.split("\n\n"):
        sentences = [ln for ln in chunk.splitlines() if ln.strip()]
        if sentences:
            blocks.append(sentences)
    return blocks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Block-benchmark the ONNX engine")
    parser.add_argument(
        "--blocks", default=None, help="block file (blank-line separated); else stdin"
    )
    parser.add_argument("--int8", action="store_true", help="use the quantized int8 graphs")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ORT intra-op threads (default 1; 0 = ORT default / multithreaded)",
    )
    args = parser.parse_args(argv)

    blocks = read_blocks(args.blocks)
    eng = onnx_engine.Engine(int8=args.int8, threads=args.threads)

    for i, sentences in enumerate(blocks):
        batch_ids = [tok.encode_source(s) for s in sentences]
        outs, encode_ms, decode_ms = eng.greedy_batch_timed(batch_ids)
        span = {
            "block": i,
            "sentences": len(sentences),
            # spm subwords + the appended EOS per sentence, as the rs spans count
            "src_tokens": sum(len(ids) for ids in batch_ids),
            "tokens": sum(len(o) for o in outs),
            "encode_ms": encode_ms,
            "decode_ms": decode_ms,
        }
        print("[block] " + json.dumps(span), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
