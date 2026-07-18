// Library entry point for `import { ... } from "fxtranslate"`. Re-exports the
// wasm core surface (Translator + the pure discovery/routing/segment/verify
// functions exposed in build-order step 1) alongside the JS shell's argument
// grammar and runner. The wasm artifacts live under ./wasm (copied in by
// `npm run build:wasm`; see README).

const wasm = require("../wasm/fxtranslate_wasm.js");
const { parse, CliError } = require("./cli");
const { run } = require("./run");
const { processIo } = require("./io");

module.exports = {
  // Shared wasm core (JS-facing names, from fxtranslate-wasm step 1).
  Translator: wasm.Translator,
  parseRecords: wasm.parseRecords,
  resolveRoute: wasm.resolveRoute,
  catalog: wasm.catalog,
  segmentSentences: wasm.segmentSentences,
  verifyAndDecompress: wasm.verifyAndDecompress,
  // JS shell surface (CLI grammar + runner + host I/O contract).
  parse,
  run,
  CliError,
  processIo,
};
