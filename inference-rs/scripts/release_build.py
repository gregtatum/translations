#!/usr/bin/env python3
"""
Release build + size characterization + cheat-proof validation.

Builds the release artifacts and reports their size, then — unless
`--skip-validation` — proves the engine build:

  1. the shippable CLI (`fxtranslate`, from fxtranslate-cli) is lean: the
     marian-oracle diagnostics (trace/replay) and the dhat allocator live in the
     separate fxtranslate-oracle crate, so they cannot leak into the product;
  2. the engine runs correctly: a committed corpus piped through the release
     oracle binary matches the debug build exactly (a faithful, non-miscompiled
     build of the oracle-validated engine), and its greedy output is compared to
     the reference `translator-cli` as a tracking metric. The parity number needs
     the C++ oracle built and the pair's model downloaded; otherwise it is
     skipped (use --skip-validation to bypass validation entirely).

Usage:
    inference-rs/scripts/release_build.py                 # build + validate (en-fr)
    inference-rs/scripts/release_build.py --pair en ru
    inference-rs/scripts/release_build.py --bloat         # + cargo bloat breakdown
    inference-rs/scripts/release_build.py --wasm-size     # + native-vs-wasm size table
    inference-rs/scripts/release_build.py --skip-validation
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import translate_common as common

CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CRATE_DIR.parent
MANIFEST = CRATE_DIR / "Cargo.toml"
TARGET = CRATE_DIR / "target"
TRANSLATOR_CLI = REPO_ROOT / "inference/build/src/app/translator-cli"
DEFAULT_CORPUS = CRATE_DIR / "corpora/dev-en.txt"
WASM_CRATE = CRATE_DIR / "crates/fxtranslate-wasm"

# The shippable product and the dev/validation binary.
CLI_BIN = TARGET / "release" / "fxtranslate"
ORACLE_BIN = TARGET / "release" / "fxtranslate-oracle"

# The lean feature set shared by the native cdylib and the wasm module, so the
# native-vs-wasm size comparison is engine-code vs engine-code (no net/mmap/icu/
# gemmology, scalar embedding kept int8). Matches the wasm crate's own default.
LEAN_FEATURES = ["--no-default-features", "--features", "lean-embed"]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB ({n:,} B)"


def build(pkg_args: list[str]) -> None:
    cmd = ["cargo", "build", "--release", "--manifest-path", str(MANIFEST)] + pkg_args
    print(f"[build] {' '.join(cmd)}", file=sys.stderr)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"[build] failed: {' '.join(cmd)}")


def size_report(bloat: bool) -> None:
    print("\n== size ==")
    for name, binp in (("fxtranslate (CLI)", CLI_BIN), ("fxtranslate-oracle", ORACLE_BIN)):
        if binp.exists():
            print(f"  {name:20} {mib(binp.stat().st_size)}")

    if bloat:
        if shutil.which("cargo-bloat"):
            print("\n== cargo bloat (fxtranslate-cli, top crates) ==")
            r = sh(
                [
                    "cargo",
                    "bloat",
                    "--release",
                    "--manifest-path",
                    str(MANIFEST),
                    "-p",
                    "fxtranslate-cli",
                    "--crates",
                    "-n",
                    "15",
                ]
            )
            print(r.stdout or r.stderr)
        else:
            print("\n[bloat] cargo-bloat not installed (cargo install cargo-bloat); skipping")


def _compressed(path: Path) -> tuple[int, int, int]:
    """(raw, gzip -9, brotli -q11) byte sizes for a file. Compressed sizes are the
    real deliverable number — npm/CDN and browsers ship the .wasm compressed."""
    data = path.read_bytes()
    return len(data), len(gzip.compress(data, 9)), _brotli_size(data)


def _brotli_size(data: bytes) -> int:
    """brotli -q11 size, via the CLI if present, else Node's zlib.brotliCompressSync.
    Either is fine; the method used is reported alongside the number."""
    if shutil.which("brotli"):
        r = subprocess.run(["brotli", "-q", "11", "-c"], input=data, stdout=subprocess.PIPE)
        return len(r.stdout)
    if shutil.which("node"):
        script = (
            "const z=require('zlib');const c=[];process.stdin.on('data',d=>c.push(d));"
            "process.stdin.on('end',()=>{const b=z.brotliCompressSync(Buffer.concat(c),"
            "{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}});process.stdout.write(String(b.length));});"
        )
        r = subprocess.run(["node", "-e", script], input=data, stdout=subprocess.PIPE, text=False)
        return int(r.stdout.decode())
    return -1


def _kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB" if n >= 0 else "n/a"


def _row(name: str, raw: int, gz: int, br: int) -> None:
    print(f"  {name:26} {_kib(raw):>10}  gz {_kib(gz):>10}  br {_kib(br):>10}")


def size_report_wasm() -> None:
    """Native lean cdylib vs wasm module, same feature set (LEAN_FEATURES), code
    only — the model (~150 MB) is excluded from both artifacts by design.

    The native number is the engine's own `.text` via `cargo bloat --filter
    fxtranslate` on a lean binary: a bare cdylib with no exported symbols is
    dead-stripped to an empty stub (nothing keeps the code alive), so the honest
    native code figure is the engine code that actually links into a binary using
    it. That is the true analog of `twiggy top`, which likewise measures code
    reachable from the wasm module's exports.

    The wasm numbers are the wasm-bindgen module before and after `wasm-opt -Oz`,
    each raw + gzip + brotli, for both the scalar and SIMD128 kernels; `twiggy
    top` gives the code-size breakdown."""
    for tool in ("wasm-pack", "wasm-opt"):
        if not shutil.which(tool):
            print(f"\n[wasm-size] {tool} not installed; skipping the wasm size report")
            return

    print("\n== native engine code (lean cdylib, cargo bloat --filter fxtranslate) ==")
    if shutil.which("cargo-bloat"):
        r = sh(
            [
                "cargo",
                "bloat",
                "--release",
                "--manifest-path",
                str(MANIFEST),
                "-p",
                "fxtranslate-oracle",
                *LEAN_FEATURES,
                "--filter",
                "fxtranslate",
                "-n",
                "12",
            ]
        )
        print(r.stdout or r.stderr)
    else:
        print("  cargo-bloat not installed (cargo install cargo-bloat); skipping breakdown")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        table = []  # (label, wasm_path)

        # Raw wasm-bindgen output requires disabling wasm-pack's own wasm-opt so we
        # can measure a clean before/after -Oz. Do it with a temporary metadata
        # overlay that is removed afterward, never committed.
        cargo = WASM_CRATE / "Cargo.toml"
        original = cargo.read_text()
        try:
            cargo.write_text(
                original + "\n[package.metadata.wasm-pack.profile.release]\nwasm-opt = false\n"
            )
            for label, rustflags in (
                ("scalar", {}),
                ("simd128", {"RUSTFLAGS": "-C target-feature=+simd128"}),
            ):
                env = dict(__import__("os").environ, **rustflags)
                subprocess.run(
                    [
                        "wasm-pack",
                        "build",
                        "--target",
                        "nodejs",
                        "--no-default-features",
                        "--out-dir",
                        str(out / label),
                    ],
                    cwd=WASM_CRATE,
                    env=env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raw_wasm = out / label / "fxtranslate_wasm_bg.wasm"
                oz_wasm = out / f"{label}-oz.wasm"
                simd_flag = ["--enable-simd"] if label == "simd128" else []
                subprocess.run(
                    ["wasm-opt", "-Oz", *simd_flag, str(raw_wasm), "-o", str(oz_wasm)], check=True
                )
                table.append((f"wasm {label} (raw bindgen)", raw_wasm))
                table.append((f"wasm {label} (-Oz)", oz_wasm))
        finally:
            cargo.write_text(original)

        print("\n== wasm module size (same lean feature set; code only, model excluded) ==")
        for label, path in table:
            _row(label, *_compressed(path))

        # twiggy top on the optimized SIMD module — the wasm analog of cargo bloat.
        simd_oz = out / "simd128-oz.wasm"
        if shutil.which("twiggy") and simd_oz.exists():
            # Keep the name section so symbols resolve.
            named = out / "simd128-oz-named.wasm"
            subprocess.run(
                [
                    "wasm-opt",
                    "-Oz",
                    "-g",
                    "--enable-simd",
                    str(out / "simd128" / "fxtranslate_wasm_bg.wasm"),
                    "-o",
                    str(named),
                ],
                check=True,
            )
            print("\n== twiggy top (wasm SIMD128 -Oz, code-size breakdown) ==")
            r = sh(["twiggy", "top", "-n", "15", str(named)])
            print(r.stdout or r.stderr)
        else:
            print("\n[twiggy] not installed (cargo install twiggy); skipping wasm breakdown")


def validate_lean() -> list[str]:
    """The product CLI must carry none of the oracle diagnostics or the dhat
    allocator. These live in the fxtranslate-oracle crate, so a no-args run of the
    CLI (which prints its usage) must never mention trace/replay or the dhat
    banner. A structural invariant now — this guards against it regressing."""
    fails = []
    out = sh([str(CLI_BIN)])  # no args -> usage on stderr
    blob = out.stdout + out.stderr
    if "replay" in blob or "trace <" in blob:
        fails.append("CLI usage lists trace/replay (oracle diagnostics leaked into the product)")
    if "[dhat]" in blob:
        fails.append("dhat allocator banner present in the product CLI")
    return fails


def validate_correct(src: str, trg: str, limit: int) -> list[str]:
    """Cheat-proof correctness of the engine artifact, two parts:

    - GATE: the release oracle binary reproduces the debug build's output exactly.
      The engine is validated against the marian oracle by the test suite; this
      proves the optimized release artifact is a *faithful* build of that engine
      (not broken/miscompiled). Deterministic 100% — the thing checked is the
      build, not the algorithm.
    - REPORT (not a gate): release-vs-`translator-cli` exact-match rate — the same
      oracle tracking metric parity.py reports (greedy output is not 100%
      identical to marian on every sentence, so this is informational)."""
    try:
        _s, _t, _langs, config = common.resolve_config(common.DEFAULT_MODELS_DIR, src, trg)
    except SystemExit as e:
        return [
            f"correctness: model for {src}-{trg} not downloaded ({e}); "
            "download it or pass --skip-validation"
        ]

    mc = common.parse_model_config(config)
    vocabs = mc["vocabs"]
    src_v = str(vocabs[0])
    trg_v = str(vocabs[1] if len(vocabs) > 1 else vocabs[0])
    lines = [l for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()][:limit]
    text = "\n".join(lines) + "\n"

    rel = sh(
        [str(ORACLE_BIN), "translate", str(mc["model"]), src_v, trg_v], input=text
    ).stdout.splitlines()
    dbg = sh(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "fxtranslate-oracle",
            "--features",
            "fast",
            "--manifest-path",
            str(MANIFEST),
            "--",
            "translate",
            str(mc["model"]),
            src_v,
            trg_v,
        ],
        input=text,
    ).stdout.splitlines()

    fails = []
    print(f"\n== correctness (artifact, {src}-{trg}) ==")
    if rel == dbg and len(rel) == len(lines):
        print(f"  GATE ok: release == debug on all {len(lines)} sentences (faithful build)")
    else:
        n = min(len(rel), len(dbg))
        diffs = [i for i in range(n) if rel[i] != dbg[i]]
        fails.append(
            f"release output differs from debug build "
            f"({len(diffs)} lines, or line-count {len(rel)} vs {len(dbg)} vs {len(lines)} in)"
        )

    # Oracle tracking metric (reported, never gates).
    if TRANSLATOR_CLI.exists():
        import tempfile

        kept = [
            l for l in config.read_text().splitlines() if not l.strip().startswith("shortlist:")
        ]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
        tmp.write("\n".join(kept) + "\n")
        tmp.close()
        try:
            ref = sh(
                [str(TRANSLATOR_CLI), "--model-config-paths", tmp.name, "--cpu-threads", "1"],
                input=text,
            ).stdout.splitlines()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        n = min(len(ref), len(rel))
        matches = sum(1 for i in range(n) if ref[i] == rel[i])
        print(f"  oracle parity (tracking, non-gating): {matches}/{n} exact vs translator-cli")
    else:
        print(f"  oracle parity: skipped (translator-cli not built at {TRANSLATOR_CLI})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pair", nargs=2, metavar=("SRC", "TRG"), default=["en", "fr"])
    ap.add_argument("--limit", type=int, default=20, help="validation corpus sentence cap")
    ap.add_argument("--bloat", action="store_true", help="add a cargo bloat crate breakdown")
    ap.add_argument(
        "--wasm-size",
        action="store_true",
        help="add the native-cdylib-vs-wasm size table (raw/gz/br, wasm-opt -Oz, twiggy top)",
    )
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    build(["-p", "fxtranslate-cli"])  # the shippable product (portable)
    build(["-p", "fxtranslate-oracle", "--features", "fast"])  # the native validation engine
    size_report(args.bloat)
    if args.wasm_size:
        size_report_wasm()

    if args.skip_validation:
        print("\n[validation] skipped (--skip-validation)")
        return

    fails = validate_lean()
    fails += validate_correct(args.pair[0], args.pair[1], args.limit)

    print("\n== validation ==")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("  PASS: product CLI is lean and the release engine is a faithful build")


if __name__ == "__main__":
    main()
