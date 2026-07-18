// Host-timed perf harness for the wasm fxtranslate engine (build-order step 7).
//
// Produces the SAME metrics as the native `scripts/perf.py --blocks` path so the
// wasm rows drop straight into the project's perf table:
//
//   words/s      = source words / sum of per-block compute time (encode+decode),
//                  model load excluded. The fair cross-engine metric.
//   TTFT (ms)    = per-block time-to-first-token (encode + first decode step),
//                  reported as median across blocks (as perf.py does per run).
//   decode tok/s = generated tokens / total decode time (excludes model load).
//
// All three are reported as median + IQR (25th-75th percentile) over `--runs`
// measured runs after `--warmup` discarded runs, matching perf.py's med_iqr.
//
// wasm has no usable `std::time::Instant` (it panics on wasm32-unknown-unknown),
// so we do NOT rely on the engine's `--timing` Instant spans. Instead:
//   - words/s wall time is measured with `performance.now()` on the host around
//     the per-block translate call;
//   - TTFT / decode-tok/s use a phase-boundary callback the engine invokes at the
//     encode/decode/first-token boundaries (`translateBlockPhased`), and the host
//     records `performance.now()` on each callback — the same boundaries the
//     native Instant spans measure.
//
// We also report the wasm linear-memory high-water mark (max
// WebAssembly.Memory.buffer.byteLength across the run) and the Node + V8 version,
// so a number is never reported without its runtime or its memory context.
//
// Usage:
//   node perf.js [blockfile] [--runs N] [--warmup N] [--shortlist]
// Defaults: corpora/nllb-en-fr.blocks.txt, runs=5, warmup=1, shortlist OFF
// (the production single-thread baseline, per the plan).

const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const { Translator } = require("../pkg/fxtranslate_wasm.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..", "..");
const CRATE_DIR = path.resolve(__dirname, "..", "..", "..");
const MODEL_DIR = path.join(REPO_ROOT, "data", "models", "enfr");
const DEFAULT_BLOCKS = path.join(CRATE_DIR, "corpora", "nllb-en-fr.blocks.txt");

function readModel(name) {
  return fs.readFileSync(path.join(MODEL_DIR, name));
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// Mirror perf.py's med_iqr: median plus the 25th/75th percentile band. Falls back
// to (min,max) for tiny samples, matching the Python helper.
function medIqr(xs) {
  const m = median(xs);
  if (xs.length < 2) return [m, xs[0], xs[0]];
  if (xs.length < 4) return [m, Math.min(...xs), Math.max(...xs)];
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => {
    const idx = p * (s.length + 1) - 1; // linear interpolation, ~statistics.quantiles
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo < 0) return s[0];
    if (hi >= s.length) return s[s.length - 1];
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  };
  return [m, q(0.25), q(0.75)];
}

// Split a blank-line-delimited block file into blocks of non-empty lines, exactly
// like the native oracle's `content.split("\n\n")` block loop.
function readBlocks(file) {
  const text = fs.readFileSync(file, "utf8");
  return text
    .split("\n\n")
    .map((chunk) => chunk.split("\n").filter((l) => l.trim() !== ""))
    .filter((b) => b.length > 0);
}

function main() {
  const args = process.argv.slice(2);
  const getOpt = (name, dflt) => {
    const i = args.indexOf(name);
    return i >= 0 && args[i + 1] ? Number(args[i + 1]) : dflt;
  };
  const runs = getOpt("--runs", 5);
  const warmup = getOpt("--warmup", 1);
  const useShortlist = args.includes("--shortlist");
  // A bare positional path is the block file; skip flags and their numeric values
  // (`--runs 5`, `--warmup 1`) so they aren't mistaken for the file path.
  const flagValues = new Set();
  ["--runs", "--warmup"].forEach((n) => {
    const i = args.indexOf(n);
    if (i >= 0 && args[i + 1]) flagValues.add(args[i + 1]);
  });
  const file =
    args.find((a) => !a.startsWith("--") && !flagValues.has(a)) || DEFAULT_BLOCKS;

  const model = readModel("model.enfr.intgemm.alphas.bin");
  const vocab = readModel("vocab.enfr.spm");
  const shortlist = useShortlist ? readModel("lex.50.50.enfr.s2t.bin") : null;

  const t = new Translator(model, vocab, vocab, shortlist);
  const kernel = t.backend();

  const blocks = readBlocks(file);
  const srcWords = blocks.reduce(
    (acc, b) => acc + b.reduce((a, l) => a + l.split(/\s+/).filter(Boolean).length, 0),
    0
  );

  let memHigh = t.linearMemoryBytes();

  // One measured pass over the whole corpus: for each block, time the whole
  // translate call (words/s numerator work) and the phase spans (TTFT, decode).
  function onePass() {
    let sumBlockMs = 0; // encode+decode compute time, summed over blocks
    let sumDecodeMs = 0;
    let sumTokens = 0;
    const ttfts = [];
    for (const block of blocks) {
      let tEncodeStart = 0;
      let tDecodeStart = 0;
      let tFirstToken = 0;
      let tDecodeEnd = 0;
      const onPhase = (name) => {
        const now = performance.now();
        if (name === "encode_start") tEncodeStart = now;
        else if (name === "decode_start") tDecodeStart = now;
        else if (name === "first_token") tFirstToken = now;
        else if (name === "decode_end") tDecodeEnd = now;
      };
      const result = t.translateBlockPhased(block.join("\n"), onPhase);
      // Last line is "sentences\tsrc_tokens\ttokens"; strip it off.
      const nl = result.lastIndexOf("\n");
      const counts = result.slice(nl + 1).split("\t").map(Number);
      const tokens = counts[2];

      const encodeMs = tDecodeStart - tEncodeStart;
      const decodeMs = tDecodeEnd - tDecodeStart;
      const firstTokenMs = tFirstToken - tDecodeStart; // decode step 0 latency
      sumBlockMs += encodeMs + decodeMs; // matches perf.py block_ms
      sumDecodeMs += decodeMs;
      sumTokens += tokens;
      ttfts.push(encodeMs + firstTokenMs); // TTFT = encode + first decode step
      const mem = t.linearMemoryBytes();
      if (mem > memHigh) memHigh = mem;
    }
    const totalS = sumBlockMs / 1000.0;
    return {
      wps: totalS ? srcWords / totalS : 0,
      ttft: median(ttfts),
      decodeTokps: sumDecodeMs ? sumTokens / (sumDecodeMs / 1000.0) : 0,
    };
  }

  const wpsRuns = [];
  const ttftRuns = [];
  const tokpsRuns = [];
  for (let i = 0; i < warmup + runs; i++) {
    const r = onePass();
    if (i < warmup) continue;
    wpsRuns.push(r.wps);
    ttftRuns.push(r.ttft);
    tokpsRuns.push(r.decodeTokps);
  }

  const [wm, wlo, whi] = medIqr(wpsRuns);
  const [tm, tlo, thi] = medIqr(ttftRuns);
  const [dm, dlo, dhi] = medIqr(tokpsRuns);

  console.log(
    `\nblocks: ${path.basename(file)} (${blocks.length} blocks, ${srcWords} source words) | ` +
      `1 thread | batched per block | runs=${runs} warmup=${warmup} | ` +
      `shortlist=${useShortlist ? "on" : "off"}`
  );
  console.log(`kernel: ${kernel} (wasm) | Node ${process.version} | V8 ${process.versions.v8}`);
  console.log("");
  console.log(`words/s        median ${wm.toFixed(0)}   IQR ${wlo.toFixed(0)}-${whi.toFixed(0)}`);
  console.log(`TTFT (ms)      median ${tm.toFixed(1)}   IQR ${tlo.toFixed(1)}-${thi.toFixed(1)}`);
  console.log(`decode tok/s   median ${dm.toFixed(0)}   IQR ${dlo.toFixed(0)}-${dhi.toFixed(0)}`);
  console.log(
    `linear-memory high-water   ${(memHigh / (1024 * 1024)).toFixed(1)} MiB ` +
      `(max WebAssembly.Memory.buffer.byteLength)`
  );
  console.log(
    "\nnote: wasm linear memory is NOT the same metric as native settled/peak RSS —\n" +
      "no shared file-backed pages, no mmap; it is the owned model buffer + activations.\n" +
      "words/s host-timed with performance.now(); TTFT/decode via the engine phase hook."
  );
}

main();
