// Batch driver: translate each line of a file through the wasm engine and print
// one output line per input line, so the output can be diffed against the native
// oracle's line-by-line stdin translation (scalar-vs-scalar parity check).
//
// Usage: node batch.js <lines-file> [--shortlist] [--no-shortlist]
// Shortlist is ON by default (matches the exact-parity reference path).

const fs = require("fs");
const path = require("path");
const { Translator } = require("../pkg/fxtranslate_wasm.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..", "..");
const MODEL_DIR = path.join(REPO_ROOT, "data", "models", "enfr");
const read = (name) => fs.readFileSync(path.join(MODEL_DIR, name));

function main() {
  const args = process.argv.slice(2);
  const file = args.find((a) => !a.startsWith("--"));
  const useShortlist = !args.includes("--no-shortlist");
  if (!file) {
    console.error("usage: node batch.js <lines-file> [--no-shortlist]");
    process.exit(2);
  }

  const model = read("model.enfr.intgemm.alphas.bin");
  const vocab = read("vocab.enfr.spm");
  const shortlist = useShortlist ? read("lex.50.50.enfr.s2t.bin") : null;

  const t = new Translator(model, vocab, vocab, shortlist);
  console.error(`kernel: ${t.backend()} (wasm), shortlist: ${useShortlist}`);

  // Match the native oracle's line-by-line stdin: translate every line,
  // including blanks, and keep them aligned. Drop only a trailing empty line
  // from the terminal newline of the file.
  const raw = fs.readFileSync(file, "utf8").split("\n");
  if (raw.length && raw[raw.length - 1] === "") raw.pop();

  const out = raw.map((l) => t.translate(l));
  process.stdout.write(out.join("\n") + "\n");
}

main();
