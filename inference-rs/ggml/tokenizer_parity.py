#!/usr/bin/env python3
"""Tokenizer-parity gate: llama.cpp UGM vs the Marian SentencePiece oracle.

For each (vocab GGUF, corpus, golden-ids) triple this tokenizes every corpus line
with ``llama-tokenize`` (vocab-only, no bos/eos) and diffs the id sequence against
the golden **exactly**. The goldens are either upstream ``spm_encode --output_format=id``
output (en-fr, en-ja) or the shipping Rust reference ``spm.rs`` (en-ru, generated
on the fly here since no upstream golden exists for that vocab).

PASS = 100% exact id match on every corpus. Any mismatch is printed with the raw
line, llama.cpp ids, and golden ids so divergences can be root-caused (normalizer
flag vs charsmap vs byte-fallback vs eos convention).

Run via ``task rs:ggml-tok-parity``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_INFERENCE_RS = _HERE.parent
_TRANSLATIONS = _INFERENCE_RS.parent
_MODELS = _HERE / "models"
_CORPORA = _INFERENCE_RS / "corpora"

_LLAMA_TOKENIZE = Path(
    __import__("os").environ.get(
        "LLAMA_TOKENIZE",
        str(Path.home() / "dev/llama.cpp/build/bin/llama-tokenize"),
    )
)


@dataclass
class Case:
    name: str
    vocab_gguf: Path
    corpus: Path
    golden_ids: Path | None  # None => generate reference from spm.rs
    spm: Path  # source .spm (for spm.rs reference generation)


CASES = [
    Case(
        "enfr/dev-en",
        _MODELS / "enfr.vocab.gguf",
        _CORPORA / "dev-en.txt",
        _CORPORA / "dev-en.ids",
        _TRANSLATIONS / "data/models/enfr/vocab.enfr.spm",
    ),
    Case(
        "enfr/nllb-en-fr",
        _MODELS / "enfr.vocab.gguf",
        _CORPORA / "nllb-en-fr.txt",
        _CORPORA / "nllb-en-fr.ids",
        _TRANSLATIONS / "data/models/enfr/vocab.enfr.spm",
    ),
    Case(
        "enja-src/dev-en",
        _MODELS / "enja-src.vocab.gguf",
        _CORPORA / "dev-en.txt",
        _CORPORA / "dev-en.enja-src.ids",
        _TRANSLATIONS / "data/models/enja/srcvocab.enja.spm",
    ),
    Case(
        "enja-trg/dev-ja",
        _MODELS / "enja-trg.vocab.gguf",
        _CORPORA / "dev-ja.txt",
        _CORPORA / "dev-ja.enja-trg.ids",
        _TRANSLATIONS / "data/models/enja/trgvocab.enja.spm",
    ),
    Case(
        "enru/dev-en",
        _MODELS / "enru.vocab.gguf",
        _CORPORA / "dev-en.txt",
        None,  # no upstream golden; use spm.rs reference
        _TRANSLATIONS / "data/models/onnx/en-ru/vocab.spm",
    ),
]


def _read_lines(path: Path) -> list[str]:
    # Keep lines verbatim minus the line terminator; drop a trailing empty line.
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _llama_ids(vocab: Path, line: str) -> list[int]:
    """Tokenize one line with llama-tokenize (no bos/eos, no escape processing)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(line)  # exact bytes, no trailing newline
        tmp = f.name
    try:
        out = subprocess.run(
            [
                str(_LLAMA_TOKENIZE),
                "-m",
                str(vocab),
                "-f",
                tmp,
                "--ids",
                "--no-bos",
                "--no-escape",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        Path(tmp).unlink(missing_ok=True)
    # Output is the last "[a, b, c]" line; other lines are log warnings.
    for ln in reversed(out.splitlines()):
        ln = ln.strip()
        if ln.startswith("[") and ln.endswith("]"):
            body = ln[1:-1].strip()
            return [int(x) for x in body.split(",")] if body else []
    raise RuntimeError(f"no id list in llama-tokenize output:\n{out}")


def _spm_rs_reference(spm: Path, corpus: Path) -> list[list[int]]:
    """Generate golden ids from the shipping Rust reference tokenizer (spm.rs)
    via a tiny cargo example, so the en-ru vocab (no upstream golden) is still
    diffed against the proven reference rather than being self-referential."""
    lines = _read_lines(corpus)
    stdin = "\n".join(lines) + "\n"
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "fxtranslate", "--example", "spm_encode_ids", "--", str(spm)],
        cwd=str(_INFERENCE_RS),
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ref = []
    for ln in out.splitlines():
        ln = ln.strip()
        ref.append([int(x) for x in ln.split()] if ln else [])
    if len(ref) != len(lines):
        raise RuntimeError(f"spm.rs produced {len(ref)} lines, corpus has {len(lines)}")
    return ref


def run_case(c: Case) -> tuple[int, int, list[str]]:
    lines = _read_lines(c.corpus)
    if c.golden_ids is not None:
        golden = [
            [int(x) for x in g.split()] if g.strip() else [] for g in _read_lines(c.golden_ids)
        ]
        golden_src = c.golden_ids.name
    else:
        golden = _spm_rs_reference(c.spm, c.corpus)
        golden_src = "spm.rs"
    assert len(golden) == len(lines), f"{c.name}: {len(golden)} golden vs {len(lines)} lines"

    matches = 0
    diffs: list[str] = []
    for i, (line, g) in enumerate(zip(lines, golden)):
        got = _llama_ids(c.vocab_gguf, line)
        if got == g:
            matches += 1
        else:
            diffs.append(f"  line {i+1}: {line!r}\n    llama.cpp: {got}\n    golden   : {g}")
    print(f"[{c.name}] golden={golden_src}  {matches}/{len(lines)} exact")
    return matches, len(lines), diffs


def main() -> None:
    if not _LLAMA_TOKENIZE.exists():
        print(
            f"ERROR: llama-tokenize not found at {_LLAMA_TOKENIZE} "
            f"(set LLAMA_TOKENIZE or build it on branch marian-arch)",
            file=sys.stderr,
        )
        sys.exit(2)

    rows = []
    all_diffs: list[str] = []
    for c in CASES:
        if not c.vocab_gguf.exists():
            print(f"ERROR: missing {c.vocab_gguf}; run convert_spm_gguf.py first", file=sys.stderr)
            sys.exit(2)
        m, t, diffs = run_case(c)
        rows.append((c.name, m, t))
        if diffs:
            all_diffs.append(f"\n=== {c.name} mismatches ===\n" + "\n".join(diffs))

    print("\n" + "=" * 56)
    print(f"{'corpus':<24}{'exact':>10}{'total':>8}{'rate':>10}")
    print("-" * 56)
    total_m = total_t = 0
    for name, m, t in rows:
        total_m += m
        total_t += t
        print(f"{name:<24}{m:>10}{t:>8}{m / t * 100:>9.1f}%")
    print("-" * 56)
    print(f"{'ALL':<24}{total_m:>10}{total_t:>8}{total_m / total_t * 100:>9.1f}%")

    if all_diffs:
        print("\n".join(all_diffs))
        print(f"\nVERDICT: FAIL — {total_t - total_m} line(s) diverge")
        sys.exit(1)
    print("\nVERDICT: PASS — llama.cpp UGM is byte-exact with the Marian oracle")


if __name__ == "__main__":
    main()
