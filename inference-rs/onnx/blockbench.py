#!/usr/bin/env python3
"""Block-benchmark the ONNX engine, emitting the same `[block]` spans as the native tools.

The perf harness (scripts/final_comparison.py) compares inference-rs, marian block-bench,
and this ONNX engine on the same block corpus. Each engine loads its model once and prints
one `[block] {...}` line per block to stderr with the per-block compute time (model load
excluded); the harness sums those for words/s and samples RSS around the process. This makes
the ONNX path a peer subject rather than a separate, non-comparable measurement.

Caveat, stated so the numbers aren't over-read: this drives the block's sentences through the
greedy engine one at a time — there is no within-block batching yet (inference-rs and
block-bench batch a block's sentences into one padded decode). So `encode_ms + decode_ms` is
the honest compute cost of the current ONNX engine, but the gap to the batched engines is
partly the missing batching, not just the runtime. ORT is pinned single-threaded to match.
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
    parser.add_argument("--threads", type=int, default=1, help="ORT intra-op threads (default 1)")
    args = parser.parse_args(argv)

    blocks = read_blocks(args.blocks)
    eng = onnx_engine.Engine(int8=args.int8, threads=args.threads)

    for i, sentences in enumerate(blocks):
        encode_ms = decode_ms = 0.0
        src_tokens = out_tokens = 0
        for sentence in sentences:
            ids = tok.encode_source(sentence)
            out, enc_ms, dec_ms = eng.greedy_timed(ids)
            encode_ms += enc_ms
            decode_ms += dec_ms
            src_tokens += len(ids)  # spm subwords + the appended EOS, as the rs spans count
            out_tokens += len(out)
        span = {
            "block": i,
            "sentences": len(sentences),
            "src_tokens": src_tokens,
            "tokens": out_tokens,
            "encode_ms": encode_ms,
            "decode_ms": decode_ms,
        }
        print("[block] " + json.dumps(span), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
