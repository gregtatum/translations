#!/usr/bin/env python3
"""
Pass B of the centralized CLI conformance harness: translation parity, tolerant.

Where Pass A (scripts/conformance.py) asserts every fxtranslate CLI is
BYTE-IDENTICAL on the interface (help/errors/list/models), Pass B compares the
one thing that legitimately cannot be byte-identical across bindings: the
translated TEXT. The Rust CLI runs against the system libm; the npm CLI runs the
same int8 model through wasm's portable libm. The int8 GEMM stays bit-identical,
but the transcendental last bits (softmax/layernorm) differ by <=1 ULP, and on a
few sentences that flips an argmax to an equally-valid alternate word. So text
parity is measured, not asserted exact (see notes/13-cli-parity.md "Translation
conformance — tolerant" and [[wasm-parity-stance]]).

The check therefore is: feed a corpus through each registered CLI's `translate`
and compute its line-by-line exact-match rate against a committed golden. The
golden is generated once from the Rust CLI (the reference / oracle), so npm is
compared to something checked in, not just live to Rust. A CLI PASSES when its
rate meets TOLERANCE_FLOOR; a few divergences are printed so a human can confirm
they are benign word-choice flips, not structural corruption.

Why a floor and not 100%: the wasm build measures ~99.69% vs native on the
clean dev corpus, but on the noisier NLLB corpus this harness defaults to (quotes,
run-together numerals like "6.Remove", ellipses) the last-bit flips are more
frequent — a measured 195/200 = 97.5% en->fr, with all five diffs benign
(synonym swaps like reunions/rencontres, a preposition, and a curly-vs-straight
apostrophe). TOLERANCE_FLOOR is set just below that measured rate (see its
definition) so those known libm flips pass, while a real regression — a
routing/tokenization/pivot bug that corrupts many lines or shifts them
structurally — fails hard.

Pivot non-tautology: for a pair with no direct model (e.g. es->fr) each CLI
pivots es->en->fr INTERNALLY, feeding its OWN English into leg two. The harness
never cross-feeds one engine's intermediate into another's, so leg two is
exercised on each engine's own machine-English — the same non-tautological shape
as scripts/parity.py.

Registry & env plumbing are shared with Pass A (imported from conformance.py):
adding a binding is still one REGISTRY entry.

Usage:
    scripts/conformance_translate.py                    # every CLI vs the golden
    scripts/conformance_translate.py --limit 30         # first 30 lines (fast)
    scripts/conformance_translate.py --only npm         # just one binding
    scripts/conformance_translate.py --update-golden    # regenerate from the oracle

Needs real models (~150MB/pair) in the fxtranslate cache; it downloads on first
use. It is SLOW and NOT hermetic, so it is not part of the fast `task rs:check`
set — run it via `task rs:conformance-corpus`.

Exit 0 iff every runnable CLI met the tolerance floor.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import conformance as pa

SCRIPTS_DIR = Path(__file__).resolve().parent
CRATE_DIR = SCRIPTS_DIR.parent
CORPORA_DIR = CRATE_DIR / "corpora"
GOLDEN_DIR = SCRIPTS_DIR / "conformance" / "translate-goldens"

DEFAULT_CORPUS = "nllb-en-fr.txt"
DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "fr"

# How many corpus lines the committed golden covers. Kept small so the golden
# stays a modest checked-in text file; the Rust CLI regenerates it in ~15s.
GOLDEN_LINES = 200

# The minimum exact-match rate (against the golden) for a CLI to PASS. Measured:
# 195/200 = 97.5% for npm-wasm vs the Rust golden on this default NLLB en->fr
# slice, every diff a benign alternate word choice from <=1-ULP libm flips (per
# [[wasm-parity-stance]]); the cleaner dev corpus is ~99.69%. The floor sits just
# below the measured NLLB rate so those known flips pass while a structural
# regression — broken routing/tokenization/pivot corrupting many lines — fails
# hard. The Rust CLI vs its own golden is ~100% (sanity), so the floor only ever
# gates wasm. Raise it if the wasm libm parity improves.
TOLERANCE_FLOOR = 0.97

# How many example divergences to print, so a human can eyeball that they are
# alternate-word-choice flips and not garbage.
EXAMPLES = 3


def golden_path(source: str, target: str) -> Path:
    return GOLDEN_DIR / f"{source}-{target}.txt"


def corpus_lines(corpus: str, limit: int) -> list[str]:
    path = CORPORA_DIR / corpus if not Path(corpus).is_absolute() else Path(corpus)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if limit:
        lines = lines[:limit]
    return lines


def translate(cli: pa.Cli, source: str, target: str, lines: list[str]) -> list[str]:
    """Run one CLI's `translate <src> <trg>`, feeding the corpus on stdin (one
    sentence per line) and returning the translated stdout lines. Uses Pass A's
    hermetic base env so no inherited FXTRANSLATE_* / NO_COLOR leaks in; the real
    model cache (default platform dir) is used since Pass B needs live models."""
    env = pa.base_env(_TranslateEnvCase())
    env.update(cli.env)
    stdin = "\n".join(lines) + "\n"
    cmd = cli.prefix + ["translate", source, target]
    res = subprocess.run(cmd, input=stdin.encode(), capture_output=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"{cli.name}: translate exited {res.returncode}")
    return res.stdout.decode("utf-8").splitlines()


class _TranslateEnvCase:
    """Minimal stand-in for Pass A's Case so base_env() gives us the hermetic,
    NO_COLOR-forced environment. Pass B never exercises the colored-TTY path."""

    id = "translate"
    tty = (False, False, False)


def rate_and_diffs(
    golden: list[str], got: list[str]
) -> tuple[int, int, list[tuple[int, str, str]]]:
    n = min(len(golden), len(got))
    matches = 0
    diffs: list[tuple[int, str, str]] = []
    for i in range(n):
        if golden[i] == got[i]:
            matches += 1
        else:
            diffs.append((i, golden[i], got[i]))
    return matches, n, diffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help="corpus file (name under corpora/ or path)"
    )
    parser.add_argument("--limit", type=int, default=0, help="cap corpus lines (0 = whole golden)")
    parser.add_argument("--only", help="run just the named CLI (e.g. rust, npm)")
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help=f"regenerate the golden ({GOLDEN_LINES} lines) from the oracle, then exit",
    )
    args = parser.parse_args()

    gpath = golden_path(args.source, args.target)

    if args.update_golden:
        oracle = next(c for c in pa.REGISTRY if c.is_oracle)
        if not oracle.available():
            sys.exit(
                f"oracle binary missing: {oracle.prefix}\nbuild it with: cargo build -p fxtranslate-cli"
            )
        lines = corpus_lines(args.corpus, GOLDEN_LINES)
        print(
            f"regenerating golden from the oracle ({oracle.name}, {oracle.prefix[0]}): "
            f"{args.source}->{args.target}, {len(lines)} lines"
        )
        out = translate(oracle, args.source, args.target, lines)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        gpath.write_text("\n".join(out) + "\n")
        print(f"wrote {len(out)} lines to {gpath.relative_to(CRATE_DIR)}")
        return

    if not gpath.exists():
        sys.exit(
            f"no golden at {gpath.relative_to(CRATE_DIR)}\n"
            "generate it with: scripts/conformance_translate.py --update-golden"
        )
    golden = gpath.read_text().splitlines()

    # Compare against the first `limit` golden lines (or the whole golden). The
    # golden fixes both the corpus slice and the reference text.
    slice_n = args.limit if args.limit else len(golden)
    slice_n = min(slice_n, len(golden))
    golden = golden[:slice_n]
    lines = corpus_lines(args.corpus, slice_n)

    registry = pa.REGISTRY
    if args.only:
        registry = [c for c in pa.REGISTRY if c.name == args.only]
        if not registry:
            sys.exit(f"no CLI named {args.only!r} in the registry")

    print(
        f"translation parity {args.source}->{args.target} "
        f"({len(lines)} lines, golden={gpath.name}, floor={TOLERANCE_FLOOR:.0%})"
    )
    if len(lines) < 100:
        # The floor is calibrated against the full golden; on a small --limit slice
        # a single benign flip swings the rate several points, so a FAIL here may be
        # sampling noise, not a regression. Run the full golden to judge the floor.
        print(
            "  (small slice: one benign flip moves the rate several points — judge the floor on the full golden)"
        )

    overall_fail = 0
    for cli in registry:
        if not cli.available():
            print(f"\n{cli.name}: SKIP (binary missing: {cli.prefix})")
            continue
        runtime = "rust-native" if cli.is_oracle else "npm-wasm" if cli.name == "npm" else cli.name
        got = translate(cli, args.source, args.target, lines)
        matches, n, diffs = rate_and_diffs(golden, got)
        rate = matches / n if n else 0.0
        ok = rate >= TOLERANCE_FLOOR
        status = "PASS" if ok else "FAIL"
        print(
            f"\n[{cli.name} / {runtime}] {status}  "
            f"{matches}/{n} exact = {rate:.2%}  (floor {TOLERANCE_FLOOR:.0%}, "
            f"{len(diffs)} divergent)"
        )
        if cli.is_oracle and diffs:
            # The oracle vs its own golden should be identical; anything else means
            # the golden is stale or the build is nondeterministic.
            print(
                "  NOTE: oracle diverges from its own golden — stale golden? regenerate with --update-golden"
            )
        for i, g, a in diffs[:EXAMPLES]:
            print(f"  line {i + 1}:")
            print(f"    src   : {lines[i]}")
            print(f"    golden: {g}")
            print(f"    {runtime}: {a}")
        if not ok:
            overall_fail += 1

    sys.exit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()
