#!/usr/bin/env python3
"""
Capture plain-text translation goldens from the REAL Bergamot WASM engine.

This freezes "what Bergamot does today" as a diffable artifact so the native
Rust engine can be gated against it (plan 03 §2/§3b, gate M1.3). The goldens are
captured ONCE by running the actual shipping Bergamot engine (the C++/Marian
translator compiled to wasm, the same one vendored into Firefox as
bergamot-translator.js) through the committed legacy Node harness under
inference/wasm/tests. At test time Bergamot is gone: native/JS diff against the
committed JSONL.

Pipeline:
  1. Verify the built engine (inference/wasm/tests/generated/bergamot-translator.
     {js,wasm}) exists — build it with `inference/scripts/build-wasm.py` if not.
  2. Verify the model dir the harness reads (inference/wasm/tests/models/<pair>)
     resolves to the exact Remote-Settings artifacts (data/models/<pair>).
  3. Feed the SAME English source corpus that conformance Pass B uses
     (inference-rs/corpora/nllb-en-fr.txt, first GOLDEN_LINES lines) through the
     engine via corpus-translate.mjs, plain text (html:false), en->fr.
  4. Write inference-rs/artifacts/goldens/bergamot/plain/<src>-<trg>.jsonl with
     one {"src":..., "tgt":...} per line.
  5. Write a MODELS.lock next to the goldens recording the sha256 of every model
     file the goldens were captured against, so they are reproducible and we know
     when to re-capture (on a Remote-Settings model bump).

Nothing here is committed automatically; it just produces files.

Usage:
    scripts/capture_bergamot_goldens.py                 # en->fr, 200 lines
    scripts/capture_bergamot_goldens.py --lines 50      # quick slice
    scripts/capture_bergamot_goldens.py --source en --target fr --corpus nllb-en-fr.txt
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CRATE_DIR = SCRIPTS_DIR.parent  # inference-rs/
REPO_ROOT = CRATE_DIR.parent  # translations/
CORPORA_DIR = CRATE_DIR / "corpora"
GOLDEN_DIR = CRATE_DIR / "artifacts" / "goldens" / "bergamot" / "plain"
MODELS_LOCK = CRATE_DIR / "artifacts" / "goldens" / "bergamot" / "MODELS.lock"

WASM_TESTS_DIR = REPO_ROOT / "inference" / "wasm" / "tests"
DRIVER = WASM_TESTS_DIR / "corpus-translate.mjs"
WASM_JS = WASM_TESTS_DIR / "generated" / "bergamot-translator.js"
WASM_BIN = WASM_TESTS_DIR / "generated" / "bergamot-translator.wasm"
HARNESS_MODELS_DIR = WASM_TESTS_DIR / "models"
REMOTE_SETTINGS_MODELS = REPO_ROOT / "data" / "models"

# Keep the golden a modest checked-in file; matches conformance Pass B's slice so
# the two harnesses share input (plan 03 §2 "feed the SAME NLLB corpus").
DEFAULT_LINES = 200
DEFAULT_CORPUS = "nllb-en-fr.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def model_files_for(pair: str) -> list[Path]:
    """The three real model artifacts the harness loads for a language pair
    (source+target concatenated, no dash), e.g. 'enfr'."""
    d = REMOTE_SETTINGS_MODELS / pair
    return [
        d / f"model.{pair}.intgemm.alphas.bin",
        d / f"lex.50.50.{pair}.s2t.bin",
        d / f"vocab.{pair}.spm",
    ]


def pairs_used(source: str, target: str) -> list[str]:
    """Which model pairs the harness loads. Direct if one side is the 'en' pivot,
    otherwise both pivot legs (matches translations-engine.mjs PIVOT logic)."""
    if source == "en" or target == "en":
        return [f"{source}{target}"]
    return [f"{source}en", f"en{target}"]


def check_engine_built() -> None:
    if WASM_JS.exists() and WASM_BIN.exists():
        return
    sys.exit(
        f"Bergamot WASM engine not built:\n"
        f"  missing {WASM_JS if not WASM_JS.exists() else WASM_BIN}\n\n"
        "Build it (one-time, ~15-30 min) with emscripten 3.1.8:\n"
        "  cd inference/3rd_party/emsdk && ./emsdk install 3.1.8 && ./emsdk activate 3.1.8\n"
        "  CMAKE_POLICY_VERSION_MINIMUM=3.5 ALLOW_RUN_ON_HOST=1 \\\n"
        "    inference/scripts/build-wasm.py --clobber\n"
        "then copy build-wasm/bergamot-translator.{js,wasm} into\n"
        f"  {WASM_JS.parent}\n"
        "(the CMAKE_POLICY_VERSION_MINIMUM env var works around modern cmake\n"
        " dropping compatibility with the vendored Eigen's cmake_minimum_required)."
    )


def check_models(pairs: list[str]) -> None:
    for pair in pairs:
        link = HARNESS_MODELS_DIR / pair
        if not link.exists():
            sys.exit(
                f"model dir for '{pair}' not found at {link}\n"
                f"symlink the Remote-Settings artifacts into place:\n"
                f"  ln -sfn {REMOTE_SETTINGS_MODELS / pair} {link}"
            )
        for mf in model_files_for(pair):
            if not mf.exists():
                sys.exit(f"missing model artifact: {mf}")


def write_models_lock(pairs: list[str]) -> None:
    lines = [
        "# Model artifacts the Bergamot plain-text goldens were captured against.",
        "# Regenerate goldens (scripts/capture_bergamot_goldens.py) whenever these",
        "# hashes change (a Remote-Settings model bump). sha256  path",
        "",
    ]
    for pair in pairs:
        for mf in model_files_for(pair):
            rel = mf.relative_to(REPO_ROOT)
            lines.append(f"{sha256_file(mf)}  {rel}")
    MODELS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    MODELS_LOCK.write_text("\n".join(lines) + "\n")
    print(f"wrote {MODELS_LOCK.relative_to(CRATE_DIR)}")


def corpus_lines(corpus: str, lines: int) -> list[str]:
    path = CORPORA_DIR / corpus if not Path(corpus).is_absolute() else Path(corpus)
    if not path.exists():
        sys.exit(f"corpus not found: {path}")
    result = [l for l in path.read_text().splitlines() if l.strip()]
    return result[:lines] if lines else result


def run_capture(source: str, target: str, lines: list[str]) -> list[dict]:
    stdin = "\n".join(lines) + "\n"
    cmd = ["node", str(DRIVER), source, target]
    print(
        f"running the real Bergamot engine: {source}->{target}, {len(lines)} lines "
        f"(cwd={WASM_TESTS_DIR.relative_to(REPO_ROOT)})"
    )
    res = subprocess.run(
        cmd,
        input=stdin.encode("utf-8"),
        capture_output=True,
        cwd=WASM_TESTS_DIR,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", "replace"))
        sys.exit(f"corpus-translate.mjs exited {res.returncode}")
    out = []
    for jl in res.stdout.decode("utf-8").splitlines():
        if jl.strip():
            out.append(json.loads(jl))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="en")
    ap.add_argument("--target", default="fr")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="name under corpora/ or a path")
    ap.add_argument("--lines", type=int, default=DEFAULT_LINES, help="cap corpus lines (0 = all)")
    args = ap.parse_args()

    pairs = pairs_used(args.source, args.target)
    check_engine_built()
    check_models(pairs)

    lines = corpus_lines(args.corpus, args.lines)
    records = run_capture(args.source, args.target, lines)

    if len(records) != len(lines):
        print(
            f"WARNING: got {len(records)} outputs for {len(lines)} inputs "
            "(the engine may have collapsed or split lines)"
        )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_DIR / f"{args.source}-{args.target}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} goldens to {out_path.relative_to(CRATE_DIR)}")

    write_models_lock(pairs)


if __name__ == "__main__":
    main()
