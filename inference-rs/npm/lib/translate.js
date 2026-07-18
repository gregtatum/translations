// Translation as a swappable dependency — the JS mirror of the Rust `Translator`/
// `Session` seam (crates/fxtranslate-cli/src/translate.rs). `load` turns a `src`→`trg`
// pair into a `Session` that translates lines; `EngineTranslator` is the real one,
// tests substitute a fake.
//
// This is where the async-shell / sync-core seam lives for `translate`: the shell
// (here) does the async Remote Settings fetch and the async verified download+cache,
// then drives the SYNCHRONOUS wasm core — `resolveRoute` to pick direct-vs-pivot and
// the `Translator` engine to translate — never pushing orchestration into wasm. A
// pivot feeds leg-one's OWN output into leg two (non-tautological), matching Rust's
// pivot-aware `Translation`.

const fs = require("node:fs");

const wasm = require("../wasm/fxtranslate_wasm.js");
const { recordsUrl } = require("./fetch");
const { Cache, ensureModel } = require("./cache");
const { renderProgress } = require("./format");

/**
 * Resolves a `src`→`trg` model into a ready {@link Session}. The two-phase shape
 * (`load` once, then `translate` many lines) matches pipe/REPL usage: the model is
 * downloaded and the engine built a single time. Mirrors the Rust `Translator` trait.
 *
 * @typedef {object} Translator
 * @property {(src: string, trg: string, cacheDir: string | undefined) => Promise<Session>} load
 */

/**
 * A loaded model, ready to translate lines. Mirrors the Rust `Session` trait.
 *
 * @typedef {object} Session
 * @property {(text: string) => string} translate  Translate one line of text.
 * @property {() => string | undefined} pivot  The pivot language for a two-leg pivot, else undefined.
 */

/**
 * Read a cached file into a `Uint8Array` for the wasm engine. Vocab files are small;
 * the model is ~150 MB — the engine copies it to owned storage, so the buffer can be
 * dropped after construction (matches the wasm `from_bytes` contract).
 *
 * @param {string} p
 * @returns {Uint8Array}
 */
function readBytes(p) {
  return new Uint8Array(fs.readFileSync(p));
}

/**
 * Build a wasm {@link wasm.Translator} engine from resolved model files. The shortlist
 * (lex) is deliberately NOT attached — the Rust CLI's `Engine::load` builds the engine
 * from model + two vocabs only, so the npm CLI must match it to stay within the
 * documented translation tolerance. Shared-vocab pairs pass the same bytes twice;
 * split-vocab (CJK) pass the two halves.
 *
 * @param {import("./cache").ModelFiles} files
 * @returns {wasm.Translator}
 */
function engineFrom(files) {
  const model = readBytes(files.model);
  const srcVocab = readBytes(files.srcVocab);
  const trgVocab = files.trgVocab === files.srcVocab ? srcVocab : readBytes(files.trgVocab);
  return new wasm.Translator(model, srcVocab, trgVocab);
}

/**
 * A direct-model session: one engine, `translate_long` per line so multi-sentence
 * input is segmented and translated in full (matching the Rust `translate_long`).
 *
 * @param {wasm.Translator} engine
 * @returns {Session}
 */
function directSession(engine) {
  return {
    translate: (text) => engine.translate_long(text),
    pivot: () => undefined,
  };
}

/**
 * A pivot session: two engines chained through the hub. Leg one's OWN output text is
 * fed into leg two (non-tautological), transparently realizing `src→pivot→trg`.
 * Mirrors the Rust `Translation::Pivot`.
 *
 * @param {wasm.Translator} first
 * @param {wasm.Translator} second
 * @param {string} pivot
 * @returns {Session}
 */
function pivotSession(first, second, pivot) {
  return {
    translate: (text) => second.translate_long(first.translate_long(text)),
    pivot: () => pivot,
  };
}

/**
 * The production translator: Remote Settings discovery (async fetch) + verified cache
 * (async download → wasm decode/verify → atomic write) + the wasm engine, all driven
 * from the sync core `resolveRoute`. Mirrors Rust's `EngineTranslator`.
 *
 * @param {import("./fetch").Fetch} fetch
 * @param {boolean} showProgress  render the download progress line (stderr TTY only)
 * @returns {Translator}
 */
function engineTranslator(fetch, showProgress) {
  return {
    async load(src, trg, cacheDir) {
      const cache = new Cache(cacheDir);
      if (showProgress) {
        cache.withProgress((name, done, total) =>
          process.stderr.write(renderProgress(name, done, total)),
        );
      }
      const body = await fetch.get(recordsUrl());
      const records = JSON.parse(wasm.parseRecords(body));
      // The sync core owns the direct-vs-pivot decision; the shell owns the I/O.
      const route = JSON.parse(wasm.resolveRoute(body, src, trg));
      if (route.kind === "pivot") {
        const leg1 = await ensureModel(fetch, cache, records, route.src, route.pivot);
        const leg2 = await ensureModel(fetch, cache, records, route.pivot, route.trg);
        return pivotSession(engineFrom(leg1), engineFrom(leg2), route.pivot);
      }
      const files = await ensureModel(fetch, cache, records, route.src, route.trg);
      return directSession(engineFrom(files));
    },
  };
}

module.exports = { engineTranslator };
