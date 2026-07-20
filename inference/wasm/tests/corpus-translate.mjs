/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * Corpus driver for the real Bergamot WASM engine.
 *
 * Reads a source corpus (one sentence per line) on stdin, translates each line
 * through the *actual* shipping Bergamot engine via the legacy Node harness
 * (`engine/translations-engine.mjs` -> `generated/bergamot-translator.{js,wasm}`),
 * and writes one JSON object per line to stdout: {"src": ..., "tgt": ...}.
 *
 * This is the golden-capture driver referenced by
 * inference-rs/scripts/capture_bergamot_goldens.py. It intentionally reuses the
 * committed harness unchanged so the goldens come from the same code path the 14
 * committed `.test.mjs` cases exercise. Plain text only (isHTML = false), which
 * matches what Firefox ships for text nodes: {qualityScores:false, html:false}
 * (the harness worker also requests alignment, but only getTranslatedText() is
 * read, so the emitted text is the plain-text translation).
 *
 * Must be run with cwd = inference/wasm/tests so the engine's cwd-relative
 * ./generated and ./models paths resolve.
 *
 * Usage:
 *   node corpus-translate.mjs <sourceLang> <targetLang> < corpus.txt > out.jsonl
 */

import { TranslationsEngine } from "./engine/translations-engine.mjs";

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const [sourceLanguage, targetLanguage] = process.argv.slice(2);
  if (!sourceLanguage || !targetLanguage) {
    process.stderr.write(
      "usage: node corpus-translate.mjs <sourceLang> <targetLang> < corpus.txt\n"
    );
    process.exit(2);
  }

  const raw = await readStdin();
  // One sentence per line; drop blank lines and trailing newline, matching the
  // conformance harness (conformance_translate.py corpus_lines()).
  const lines = raw.split("\n").filter((l) => l.trim().length > 0);

  const engine = new TranslationsEngine(sourceLanguage, targetLanguage);
  try {
    for (const src of lines) {
      const tgt = await engine.translate(src, /* isHTML */ false);
      process.stdout.write(JSON.stringify({ src, tgt }) + "\n");
    }
  } finally {
    engine.terminate();
  }
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error) + "\n");
  process.exit(1);
});
