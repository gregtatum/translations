#!/usr/bin/env python3
"""
Three-way apples-to-apples comparison on the SAME data and SAME model:

  1. inference-rs      — the Rust engine (default fast: lean-embed + gemmology)
  2. marian block-bench — the native marian-fork reference (bergamot)
  3. Firefox Wasm       — Full-Page Translations (numbers pasted from a perftest run)

All three translate Firefox's benchmark page (a Frankenstein excerpt) with the
en→ru **base** model, block by block (one batched translate per paragraph on a
loaded engine — the production shape). Metrics mirror Firefox's TranslationsBencher:

  words/s      = source words ÷ translation seconds        (higher better)
  tokens/s     = source spm tokens ÷ translation seconds   (higher better)
  translate s  = translation wall time, model load EXCLUDED (lower better)
  init ms      = model load / engine init                  (lower better)
  peak RSS MiB = peak resident memory of the process       (lower better)

`translation seconds` excludes model load on all three (Firefox measures
engine-ready → done; the native tools sum per-block compute; init is measured separately).

Run:  inference-rs/scripts/final_comparison.py            (en→ru, base model)
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import translate_common as common

CRATE = Path(__file__).resolve().parent.parent
REPO = CRATE.parent
BIN = CRATE / "target/release/fxtranslate-oracle"
BLOCK_BENCH = REPO / "inference/build/src/app/block-bench"
DEFAULT_BLOCKS = CRATE / "corpora/frankenstein-en.blocks.txt"
# The ONNX engine runs under the project .venv (onnxruntime lives there, not in poetry);
# blockbench.py emits the same [block] spans as the native tools. See notes/16.
VENV_PY = CRATE / ".venv/bin/python3"
ONNX_BLOCKBENCH = CRATE / "onnx/blockbench.py"
ONNX_MODELS = CRATE / "onnx/models"
# The ggml engine is a compiled binary; its RSS is just ggml arenas + weights (no Python
# runtime), so it is the fairest memory peer. It consumes pre-tokenized source ids so the
# sampled process is the binary itself. See notes/18.
GGML_BIN = CRATE / "ggml/marian_ggml"
GGML_PRETOK = CRATE / "ggml/pretokenize.py"
GGML_MODELS = CRATE / "ggml/models"
# The llama.cpp LLM_ARCH_MARIAN engine is also a compiled binary (built by ggml/build_llama.sh),
# so its RSS is the fairest memory peer to G1. It consumes the SAME pretokenized source-id block
# file G1 uses and emits identical [block] spans. See notes/18 M3.
LLAMA_BIN = CRATE / "ggml/marian_llama_blockbench"
LLAMA_MODELS = CRATE / "ggml/models"

# Firefox "Full-Page Translations Base Model" (en→ru), medians of the 5-run
# perftest the user provided. wordCount/tokenCount are the page's source totals.
# `settled` = stabilized-inference-process-memory (retained during translation,
# what Activity Monitor shows); `peak` = peak-inference-process-memory.
FIREFOX = {
    "label": "Firefox Wasm (Full-Page)",
    "words_per_second": 418.982,
    "tokens_per_second": 566.884,
    "translate_s": 22.853,  # total-translation-time (engine-ready → done)
    "init_ms": 135.169,  # engine-init-time
    "settled_rss_mib": 355.361,  # stabilized-inference-process-memory-usage
    "peak_rss_mib": 355.361,  # peak-inference-process-memory-usage
    "peak_parent_mib": 384.928,
    "word_count": 9575,
    "token_count": 12955,
}


def med(xs):
    return statistics.median(xs)


def sample_rss_mib(pid: int):
    """Current resident set size of `pid` in MiB (macOS/Linux `ps` reports KiB)."""
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    v = r.stdout.strip()
    return int(v) / 1024.0 if v.isdigit() else None


def run_sampled(cmd, stdin_path, interval, env=None):
    """Run `cmd`, polling its RSS every `interval` s. Returns (wall_s, stderr,
    samples) where samples is [(t_since_start, rss_mib)]. stderr goes to a temp
    file (no pipe-buffer deadlock while we poll); stdin is fed from a file. `env`
    (if given) is merged over the current environment for the child."""
    err = tempfile.TemporaryFile()
    stdin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    child_env = None
    if env:
        import os

        child_env = {**os.environ, **env}
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.DEVNULL, stderr=err, env=child_env)
    samples = []
    while p.poll() is None:
        rss = sample_rss_mib(p.pid)
        if rss:
            samples.append((time.perf_counter() - t0, rss))
        time.sleep(interval)
    p.wait()
    wall = time.perf_counter() - t0
    if stdin is not subprocess.DEVNULL:
        stdin.close()
    err.seek(0)
    stderr = err.read().decode("utf-8", errors="replace")
    err.close()
    if p.returncode != 0:
        sys.exit(f"[final] command failed: {' '.join(cmd)}\n{stderr[-2000:]}")
    return wall, stderr, samples


def rss_settled_peak(samples, wall):
    """Peak = max RSS over the run (includes the load transient). Settled =
    median RSS over the second half of the run (steady-state translation, past
    the load ramp) — the retained working set, comparable to Activity Monitor."""
    if not samples:
        return 0.0, 0.0
    peak = max(r for _, r in samples)
    steady = [r for t, r in samples if t >= wall * 0.5] or [r for _, r in samples]
    return med(steady), peak


def parse_blocks(stderr):
    pre = "[block] "
    return [json.loads(l[len(pre) :]) for l in stderr.splitlines() if l.startswith(pre)]


def marian_config(config: Path) -> Path:
    """Temp config: drop shortlist, force ssplit-mode: sentence (pre-split input)."""
    kept = [
        l
        for l in config.read_text().splitlines()
        if not l.strip().startswith(("shortlist:", "ssplit-mode:"))
    ]
    kept.append("ssplit-mode: sentence")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
    tmp.write("\n".join(kept) + "\n")
    tmp.close()
    return Path(tmp.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", default="en")
    ap.add_argument("target", nargs="?", default="ru")
    ap.add_argument("--models-dir", default=common.DEFAULT_MODELS_DIR)
    ap.add_argument("--blocks", default=str(DEFAULT_BLOCKS))
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.02, help="rss sample interval (s)")
    ap.add_argument(
        "--onnx",
        action="store_true",
        help="add the ONNX engine (onnx/blockbench.py under .venv) as a fourth subject",
    )
    ap.add_argument(
        "--onnx-precision",
        choices=["int8", "float"],
        default="int8",
        help="ONNX graph precision to benchmark (default int8, comparable to rs/marian int8)",
    )
    ap.add_argument(
        "--onnx-threads",
        type=int,
        default=1,
        help="ORT intra-op threads for the ONNX row (default 1 = single, matching the native "
        "rows; 0 = ORT default/multithreaded — trades memory for decode speed)",
    )
    ap.add_argument(
        "--ggml",
        action="store_true",
        help="add the bare-libggml engine (ggml/marian_ggml, Q8_0) as a subject (see notes/18)",
    )
    ap.add_argument(
        "--ggml-precision",
        choices=["q8_0", "float"],
        default="q8_0",
        help="ggml GGUF precision to benchmark (default q8_0, comparable to rs/marian int8)",
    )
    ap.add_argument(
        "--ggml-threads",
        type=int,
        default=1,
        help="ggml CPU threads for the ggml row (default 1, matching the native rows)",
    )
    ap.add_argument(
        "--llama",
        action="store_true",
        help="add the llama.cpp LLM_ARCH_MARIAN engine (ggml/marian_llama_blockbench, Q8_0) as a "
        "subject (see notes/18 M3)",
    )
    ap.add_argument(
        "--llama-precision",
        choices=["q8_0", "float"],
        default="q8_0",
        help="llama.cpp GGUF precision to benchmark (default q8_0, comparable to rs/marian int8)",
    )
    ap.add_argument(
        "--llama-threads",
        type=int,
        default=1,
        help="llama.cpp context threads (n_threads / n_threads_batch) for the llama row "
        "(default 1, matching the native rows)",
    )
    args = ap.parse_args()

    _s, _t, _l, config = common.resolve_config(args.models_dir, args.source, args.target)
    mc = common.parse_model_config(config)
    model, vocabs = mc["model"], mc["vocabs"]
    srcv = vocabs[0]
    trgv = vocabs[1] if len(vocabs) > 1 else vocabs[0]

    blocks = Path(args.blocks)
    text = blocks.read_text()
    n_blocks = sum(1 for c in text.split("\n\n") if c.strip())
    src_words = sum(len(l.split()) for l in text.splitlines() if l.strip())

    print(f"[final] building release (native fast config)…", file=sys.stderr)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "fxtranslate-oracle",
            "--features",
            "fast",
            "--manifest-path",
            str(CRATE / "Cargo.toml"),
        ],
        check=True,
    )
    if not BLOCK_BENCH.exists():
        sys.exit(f"[final] block-bench not found at {BLOCK_BENCH}")

    engine_cmd = [
        str(BIN),
        "translate",
        str(model),
        str(srcv),
        str(trgv),
        "--blocks",
        str(blocks),
        "--timing",
    ]
    tmpcfg = marian_config(config)
    marian_cmd = [str(BLOCK_BENCH), "--model-config-paths", str(tmpcfg)]

    onnx_cmd = None
    if args.onnx:
        suffix = ".int8.onnx" if args.onnx_precision == "int8" else ".onnx"
        needed = [ONNX_MODELS / f"encoder{suffix}", ONNX_MODELS / f"decode_step{suffix}"]
        missing = [p.name for p in needed if not p.exists()]
        if not VENV_PY.exists():
            sys.exit(f"[final] .venv python not found at {VENV_PY} (run: task rs:onnx-export)")
        if missing:
            hint = (
                "task rs:onnx-quantize" if args.onnx_precision == "int8" else "task rs:onnx-export"
            )
            sys.exit(f"[final] ONNX graphs missing: {', '.join(missing)} (build them: {hint})")
        onnx_cmd = [str(VENV_PY), str(ONNX_BLOCKBENCH), "--blocks", str(blocks)]
        onnx_cmd += ["--threads", str(args.onnx_threads)]
        if args.onnx_precision == "int8":
            onnx_cmd.append("--int8")

    ggml_cmd = None
    ggml_env = None
    ggml_pretok = None
    if args.ggml:
        gguf = GGML_MODELS / f"marian.{args.ggml_precision}.gguf"
        if not GGML_BIN.exists():
            sys.exit(f"[final] ggml engine not built at {GGML_BIN} (build it: task rs:ggml-build)")
        if not gguf.exists():
            sys.exit(f"[final] ggml GGUF missing: {gguf} (build it: task rs:ggml-convert)")
        # Pre-tokenize the block corpus once (shared SPM), so the sampled process is the
        # binary alone — the fair-RSS requirement.
        ggml_pretok = tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False)
        pre = subprocess.run(
            [str(VENV_PY), str(GGML_PRETOK), str(blocks)], capture_output=True, text=True
        )
        if pre.returncode != 0:
            sys.exit(f"[final] ggml pretokenize failed:\n{pre.stderr}")
        ggml_pretok.write(pre.stdout)
        ggml_pretok.close()
        ggml_cmd = [str(GGML_BIN), str(gguf), "blockbench", "--blocks", ggml_pretok.name]
        ggml_env = {"FXT_GGML_THREADS": str(args.ggml_threads)}

    llama_cmd = None
    llama_env = None
    llama_pretok = None
    if args.llama:
        gguf = LLAMA_MODELS / f"marian-llama.{args.llama_precision}.gguf"
        if not LLAMA_BIN.exists():
            sys.exit(
                f"[final] llama.cpp engine not built at {LLAMA_BIN} (build it: "
                f"bash ggml/build_llama.sh)"
            )
        if not gguf.exists():
            sys.exit(
                f"[final] llama GGUF missing: {gguf} (build it: "
                f"task rs:ggml-llama-convert or python ggml/convert_marian_llama.py)"
            )
        # Same pretokenized source ids as G1 (shared SPM) — the sampled process is the binary alone.
        llama_pretok = tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False)
        pre = subprocess.run(
            [str(VENV_PY), str(GGML_PRETOK), str(blocks)], capture_output=True, text=True
        )
        if pre.returncode != 0:
            sys.exit(f"[final] llama pretokenize failed:\n{pre.stderr}")
        llama_pretok.write(pre.stdout)
        llama_pretok.close()
        llama_cmd = [str(LLAMA_BIN), str(gguf), "--blocks", llama_pretok.name]
        llama_env = {"FXT_LLAMA_THREADS": str(args.llama_threads)}

    # Accumulators across runs.
    keys = ["wps", "tps", "translate_s", "init_ms", "settled", "peak"]
    rs = {k: [] for k in keys}
    mar = {k: [] for k in keys}
    onx = {k: [] for k in keys}
    ggm = {k: [] for k in keys}
    llm = {k: [] for k in keys}
    src_tokens = None
    try:
        for i in range(args.warmup + args.runs):
            # inference-rs (reads --blocks; no stdin)
            wall, err, samples = run_sampled(engine_cmd, None, args.interval)
            spans = parse_blocks(err)
            compute_s = sum(s["encode_ms"] + s["decode_ms"] for s in spans) / 1000.0
            src_tokens = sum(s["src_tokens"] for s in spans)
            settled, peak = rss_settled_peak(samples, wall)
            if i >= args.warmup:
                rs["wps"].append(src_words / compute_s)
                rs["tps"].append(src_tokens / compute_s)
                rs["translate_s"].append(compute_s)
                rs["init_ms"].append((wall - compute_s) * 1000.0)
                rs["settled"].append(settled)
                rs["peak"].append(peak)

            # marian block-bench (block text on stdin)
            wall, err, samples = run_sampled(marian_cmd, str(blocks), args.interval)
            spans = parse_blocks(err)
            compute_s = sum(s["wall_ms"] for s in spans) / 1000.0
            settled, peak = rss_settled_peak(samples, wall)
            if i >= args.warmup:
                mar["wps"].append(src_words / compute_s)
                mar["tps"].append(src_tokens / compute_s)
                mar["translate_s"].append(compute_s)
                mar["init_ms"].append((wall - compute_s) * 1000.0)
                mar["settled"].append(settled)
                mar["peak"].append(peak)

            # ONNX engine (onnx/blockbench.py under .venv; reads --blocks, no stdin)
            if onnx_cmd:
                wall, err, samples = run_sampled(onnx_cmd, None, args.interval)
                spans = parse_blocks(err)
                compute_s = sum(s["encode_ms"] + s["decode_ms"] for s in spans) / 1000.0
                settled, peak = rss_settled_peak(samples, wall)
                if i >= args.warmup:
                    onx["wps"].append(src_words / compute_s)
                    onx["tps"].append(src_tokens / compute_s)
                    onx["translate_s"].append(compute_s)
                    onx["init_ms"].append((wall - compute_s) * 1000.0)
                    onx["settled"].append(settled)
                    onx["peak"].append(peak)

            # ggml engine (compiled binary; reads a pre-tokenized --blocks file, no stdin)
            if ggml_cmd:
                wall, err, samples = run_sampled(ggml_cmd, None, args.interval, env=ggml_env)
                spans = parse_blocks(err)
                compute_s = sum(s["encode_ms"] + s["decode_ms"] for s in spans) / 1000.0
                settled, peak = rss_settled_peak(samples, wall)
                if i >= args.warmup:
                    ggm["wps"].append(src_words / compute_s)
                    ggm["tps"].append(src_tokens / compute_s)
                    ggm["translate_s"].append(compute_s)
                    ggm["init_ms"].append((wall - compute_s) * 1000.0)
                    ggm["settled"].append(settled)
                    ggm["peak"].append(peak)

            # llama.cpp engine (compiled binary; reads a pre-tokenized --blocks file, no stdin)
            if llama_cmd:
                wall, err, samples = run_sampled(llama_cmd, None, args.interval, env=llama_env)
                spans = parse_blocks(err)
                compute_s = sum(s["encode_ms"] + s["decode_ms"] for s in spans) / 1000.0
                settled, peak = rss_settled_peak(samples, wall)
                if i >= args.warmup:
                    llm["wps"].append(src_words / compute_s)
                    llm["tps"].append(src_tokens / compute_s)
                    llm["translate_s"].append(compute_s)
                    llm["init_ms"].append((wall - compute_s) * 1000.0)
                    llm["settled"].append(settled)
                    llm["peak"].append(peak)
    finally:
        tmpcfg.unlink(missing_ok=True)
        if ggml_pretok:
            Path(ggml_pretok.name).unlink(missing_ok=True)
        if llama_pretok:
            Path(llama_pretok.name).unlink(missing_ok=True)

    print(
        f"\ncorpus: {blocks.stem} ({n_blocks} blocks, {src_words} source words, "
        f"{src_tokens} source tokens) | model: {model.parent.name} base | 1 thread | "
        f"shortlist off | runs={args.runs} warmup={args.warmup} | rss sampled every "
        f"{int(args.interval * 1000)}ms\n"
    )
    hdr = (
        f"{'engine':28}{'words/s':>9}{'tokens/s':>10}{'translate s':>13}"
        f"{'init ms':>9}{'settled MiB':>13}{'peak MiB':>10}"
    )
    print(hdr)

    def row(label, d):
        print(
            f"{label:28}{med(d['wps']):>9.0f}{med(d['tps']):>10.0f}"
            f"{med(d['translate_s']):>13.2f}{med(d['init_ms']):>9.0f}"
            f"{med(d['settled']):>13.0f}{med(d['peak']):>10.0f}"
        )

    onnx_tlabel = "ORT-default threads" if args.onnx_threads == 0 else f"{args.onnx_threads}t"
    row("inference-rs (rust, fast)", rs)
    row("marian block-bench (native)", mar)
    if args.onnx:
        row(f"ONNX ORT ({args.onnx_precision}, {onnx_tlabel})", onx)
    if args.ggml:
        row(f"ggml ({args.ggml_precision}, {args.ggml_threads}t)", ggm)
    if args.llama:
        row(f"llama.cpp ({args.llama_precision}, {args.llama_threads}t)", llm)
    print(
        f"{FIREFOX['label']:28}{FIREFOX['words_per_second']:>9.0f}"
        f"{FIREFOX['tokens_per_second']:>10.0f}{FIREFOX['translate_s']:>13.2f}"
        f"{FIREFOX['init_ms']:>9.0f}{FIREFOX['settled_rss_mib']:>13.0f}"
        f"{FIREFOX['peak_rss_mib']:>10.0f}"
    )

    rw, mw, fw = med(rs["wps"]), med(mar["wps"]), FIREFOX["words_per_second"]
    print(
        "\nspeedups (words/s):\n"
        f"  inference-rs vs Firefox Wasm : {rw / fw:.2f}x\n"
        f"  marian native vs Firefox Wasm: {mw / fw:.2f}x\n"
        f"  inference-rs vs marian native: {rw / mw:.2f}x"
    )
    if args.onnx:
        ow = med(onx["wps"])
        print(
            f"  ONNX ORT vs Firefox Wasm     : {ow / fw:.2f}x\n"
            f"  ONNX ORT vs inference-rs     : {ow / rw:.2f}x  (ONNX/rs; <1 = ONNX slower)"
        )
    if args.ggml:
        gw = med(ggm["wps"])
        print(
            f"  ggml vs Firefox Wasm         : {gw / fw:.2f}x\n"
            f"  ggml vs marian native        : {gw / mw:.2f}x\n"
            f"  ggml vs inference-rs         : {gw / rw:.2f}x  (ggml/rs; <1 = ggml slower)"
        )
    if args.llama:
        lw = med(llm["wps"])
        print(
            f"  llama.cpp vs Firefox Wasm    : {lw / fw:.2f}x\n"
            f"  llama.cpp vs marian native   : {lw / mw:.2f}x\n"
            f"  llama.cpp vs inference-rs    : {lw / rw:.2f}x  (llama/rs; <1 = llama slower)"
        )
        if args.ggml:
            print(
                f"  llama.cpp vs ggml (G1)       : {lw / med(ggm['wps']):.2f}x  "
                f"(llama/G1; >=1 = llama's scheduler meets/beats G1's hand loop)"
            )
    print(
        "\nnotes:\n"
        "  - Same en→ru BASE model (dim-emb 512, ffn 2048, SSRU dec) and same\n"
        "    Frankenstein text for all three; block = paragraph = one translate call.\n"
        "  - translate s excludes model load on all three (Firefox: engine-ready→done;\n"
        "    native: summed per-block compute). init ms is model load / engine init.\n"
        "  - settled MiB = retained working set during translation (median RSS over the\n"
        "    run's second half, past the load ramp — what Activity Monitor shows). peak\n"
        "    MiB = max RSS incl. the load transient. Native values are sampled RSS of the\n"
        "    whole process; Firefox is its inference *subprocess* (settled = stabilized-,\n"
        "    peak = peak-inference-process-memory). Firefox also runs a parent process\n"
        "    (~%d MiB peak) that the native tools have no equivalent of.\n"
        "  - words: whitespace (%d); Firefox counts %d via ICU. tokens: source spm\n"
        "    subwords (%d); Firefox counts %d. Same text, ~1%% counting differences.\n"
        "  - Firefox Wasm carries HTML parsing + process IPC per block that the native\n"
        "    harnesses do not; it is the shipping end-to-end path, not just the kernel."
        % (
            round(FIREFOX["peak_parent_mib"]),
            src_words,
            FIREFOX["word_count"],
            src_tokens,
            FIREFOX["token_count"],
        )
    )
    if args.onnx:
        print(
            "  - ONNX ORT caveats (read the row with these in mind):\n"
            f"    * threads: this ONNX row used {onnx_tlabel} (the native rows are always 1\n"
            "      thread). ORT multithreading trades memory (per-thread arenas) for decode\n"
            "      speed; toggle with --onnx-threads (0 = ORT default). inference-rs is\n"
            "      single-threaded and memory-optimized by design.\n"
            "    * the decoder is block-batched (padded + masked, like rs/marian); the\n"
            "      encoder still runs once per sentence, so the block isn't batched fully\n"
            "      end to end. int8 is ORT dynamic QDQ, not intgemm.\n"
            "    * RSS is the whole Python+onnxruntime process (interpreter + ORT arenas),\n"
            "      so settled/peak carry overhead the native single-binary tools don't.\n"
            "    * init ms includes Python startup + ORT session load, not just model load."
        )
    if args.ggml:
        print(
            "  - ggml caveats (read the row with these in mind):\n"
            "    * a compiled single binary like the native tools, so its RSS is a fair peer\n"
            "      (ggml arenas + weights) — unlike the Python+ORT ONNX row.\n"
            "    * Q8_0 is ggml's block-wise int8 (per-32 scale), NOT intgemm shifted-int8;\n"
            "      it tracks the float reference more closely (see notes/18 Gate 1).\n"
            "    * the decoder is block-batched with row compaction (retired rows drop out, so\n"
            "      no wasted compute on ragged blocks); the encoder still runs once per\n"
            "      sentence. Batching is ~+5% on this per-sentence-shaped corpus (1.38x only\n"
            "      for uniform multi-sentence batches); THREADS are the real lever — ggml\n"
            "      scales ~2.5-3x to 4-6 threads (see notes/18), unlike ORT's ~9%.\n"
            "    * token_embd stays F16 (like ONNX keeping the Gather float); the tied output\n"
            "      projection is Q8_0."
        )
    if args.llama:
        print(
            "  - llama.cpp caveats (read the row with these in mind):\n"
            "    * the LLM_ARCH_MARIAN engine inside upstream llama.cpp (marian-arch branch),\n"
            "      driven two-phase (llama_encode -> greedy llama_decode). A compiled single\n"
            "      binary like G1, so its RSS is a fair peer (ggml arenas + weights).\n"
            "    * threads set on the llama context (n_threads / n_threads_batch); llama.cpp\n"
            "      uses its own ggml threadpool (OpenMP was off at build).\n"
            "    * single-sequence: one sentence at a time (no block batching), same shape as\n"
            "      G1's per-sentence encoder path. Its own Q8_0 GGUF (marian-llama.q8_0.gguf),\n"
            "      whose block layout differs slightly from G1's, so a handful of argmaxes flip\n"
            "      (~0.2% output-length difference vs G1) — same source ids, same greedy math."
        )


if __name__ == "__main__":
    main()
