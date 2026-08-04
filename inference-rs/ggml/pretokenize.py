#!/usr/bin/env python3
"""Pre-tokenize a block corpus into source ids for the ggml engine's blockbench mode.

The ggml engine is a compiled binary whose RSS we sample directly (the whole point of
G1's memory fairness), so it must not embed a tokenizer process. Instead this step runs
once, before timing, using the SAME SentencePiece model as the ONNX eval (appending EOS,
no BOS — production behavior), and writes an ids-block file the binary consumes:

    blocks are blank-line separated; each non-empty line is one sentence's
    space-separated source ids (including the trailing EOS).

Usage: pretokenize.py <blocks.txt>  > <blocks.ids>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "onnx"))
import tokenizer as tok  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pretokenize.py <blocks.txt>", file=sys.stderr)
        return 1
    text = Path(sys.argv[1]).read_text()
    out: list[str] = []
    for chunk in text.split("\n\n"):
        sentences = [ln for ln in chunk.splitlines() if ln.strip()]
        if not sentences:
            continue
        for s in sentences:
            out.append(" ".join(str(i) for i in tok.encode_source(s)))
        out.append("")  # blank line separates blocks
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
