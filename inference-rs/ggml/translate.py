#!/usr/bin/env python3
"""Translate text through the ggml engine: tokenize -> binary decode -> detokenize.

The compiled engine speaks source ids in / target ids out (tokenization lives in Python,
the shared SPM). This wrapper closes the loop for humans, mirroring onnx/engine.py's CLI.

Usage:
    translate.py [--float] "Hello, world."
    echo "Hello, world." | translate.py [--float]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "onnx"))
import tokenizer as tok  # noqa: E402

_BIN = _HERE / "marian_ggml"
_DEFAULT = "Hello, world. This is a test of the translation engine."


def translate(lines: list[str], precision: str) -> list[str]:
    gguf = _HERE / f"models/marian.{precision}.gguf"
    ids_in = "\n".join(" ".join(str(i) for i in tok.encode_source(ln)) for ln in lines) + "\n"
    r = subprocess.run(
        [str(_BIN), str(gguf), "decode"],
        input=ids_in,
        capture_output=True,
        text=True,
        check=True,
    )
    out = []
    for line in r.stdout.splitlines():
        ids = [int(x) for x in line.split()] if line.strip() else []
        out.append(tok.decode_ids(ids))
    return out


def main(argv: list[str]) -> int:
    precision = "float" if "--float" in argv else "q8_0"
    argv = [a for a in argv if a != "--float"]
    if argv:
        lines = [argv[0]]
    elif not sys.stdin.isatty():
        lines = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    else:
        lines = [_DEFAULT]
    for t in translate(lines, precision):
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
