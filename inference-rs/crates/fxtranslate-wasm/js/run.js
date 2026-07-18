// Node driver for the wasm fxtranslate engine.
//
// Reads the en->fr model, vocab, and shortlist from `data/models/enfr/` as
// buffers, constructs the wasm `Translator`, translates the sentence given on the
// command line (or a default), and prints the result. This is the step-4/5
// milestone: `node run.js "Hello world."` -> French, produced by wasm.
//
// Usage:
//   node run.js "Hello world."          translate one sentence
//   node run.js --long "text ..."       use translate_long (segmented)
//   node run.js                         translate a default sentence

const fs = require("fs");
const path = require("path");
const { Translator } = require("../pkg/fxtranslate_wasm.js");

// Repo root is four levels up: crates/fxtranslate-wasm/js -> inference-rs -> repo.
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..", "..");
const MODEL_DIR = path.join(REPO_ROOT, "data", "models", "enfr");

function read(name) {
  return fs.readFileSync(path.join(MODEL_DIR, name));
}

function main() {
  const args = process.argv.slice(2);
  const long = args[0] === "--long";
  const rest = long ? args.slice(1) : args;
  const text = rest.length ? rest.join(" ") : "Hello world.";

  const model = read("model.enfr.intgemm.alphas.bin");
  const vocab = read("vocab.enfr.spm"); // shared src+trg vocab for en->fr
  const shortlist = read("lex.50.50.enfr.s2t.bin");

  const t = new Translator(model, vocab, vocab, shortlist);
  const kernel = t.backend();

  const out = long ? t.translate_long(text) : t.translate(text);

  console.log(`kernel: ${kernel} (wasm)`);
  console.log(`src: ${text}`);
  console.log(`trg: ${out}`);
}

main();
