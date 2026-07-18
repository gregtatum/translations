# fxtranslate (WebAssembly)

WebAssembly build of the **fxtranslate** neural machine-translation engine — a
pure-Rust reimplementation of the Firefox Translations inference pipeline
(tokenize → encode → SSRU decode → project → detokenize), compiled to `wasm32`
via `wasm-bindgen`.

This package ships the **engine only**. Translation models are **not bundled** —
you supply the model, vocabulary, and (optional) shortlist as byte buffers at
construction time. This keeps the package small (the engine is ~140 KiB of
`.wasm`; models are tens of MB) and lets you fetch or bundle whichever language
pair you need.

- **Single-threaded** (no `SharedArrayBuffer` / wasm threads).
- **SIMD128** int8 GEMM kernel (the pure-Rust `i32x4.dot_i16x8_s` path). Enabled
  in Node ≥ 16 and current browsers by default; no flags needed at runtime.
- No filesystem, no network, no C++ toolchain — the host provides the model bytes.

> This is a validation + distribution artifact. The Firefox Translations
> replacement ships the Rust engine **natively**; this wasm build exists for npm
> reach and cross-runtime validation, not for Firefox integration.

## Install

```sh
npm install fxtranslate-wasm
```

(Exact published name/scope and version are set at publish time.)

## Usage

You provide four byte buffers as `Uint8Array`s:

- `model` — the `*.intgemm.alphas.bin` weights
- `srcVocab` — the source SentencePiece vocab (`*.spm`)
- `trgVocab` — the target SentencePiece vocab (pass the **same** buffer as
  `srcVocab` for shared-vocab pairs like en→fr; a distinct buffer for split-vocab
  pairs such as CJK)
- `shortlist` — optional lexical shortlist (`lex.*.s2t.bin`); pass `null` to disable

Where those bytes come from — `fetch()` in the browser, `fs.readFileSync` in Node,
or a bundler asset — is up to you.

### Browser (bundler target)

```js
import { Translator } from "fxtranslate-wasm";

const [model, vocab, shortlist] = await Promise.all([
  fetch("/models/enfr/model.enfr.intgemm.alphas.bin").then((r) => r.arrayBuffer()),
  fetch("/models/enfr/vocab.enfr.spm").then((r) => r.arrayBuffer()),
  fetch("/models/enfr/lex.50.50.enfr.s2t.bin").then((r) => r.arrayBuffer()),
]);

const translator = new Translator(
  new Uint8Array(model),
  new Uint8Array(vocab),
  new Uint8Array(vocab), // shared vocab: same buffer for src + trg
  new Uint8Array(shortlist), // or null
);

console.log(translator.translate("Hello, world."));
console.log(translator.translateLong("A longer paragraph. It has several sentences."));
console.log(translator.backend()); // "wasm-simd128"
```

### Node

```js
import { readFileSync } from "node:fs";
import { Translator } from "fxtranslate-wasm";

const dir = "models/enfr";
const translator = new Translator(
  readFileSync(`${dir}/model.enfr.intgemm.alphas.bin`),
  readFileSync(`${dir}/vocab.enfr.spm`),
  readFileSync(`${dir}/vocab.enfr.spm`),
  readFileSync(`${dir}/lex.50.50.enfr.s2t.bin`),
);

console.log(translator.translate("Hello, world."));
```

## API

- `new Translator(model, srcVocab, trgVocab, shortlist?)` — build an engine from
  the supplied byte buffers. The engine takes **ownership** of copies of the bytes;
  you may free the input buffers afterward.
- `translate(text: string): string` — translate a single sentence-unit.
- `translateLong(text: string): string` — split arbitrary-length text into
  sentences (built-in segmenter; this build is ICU-free) and translate each within
  the model's context window, rejoined with the original whitespace.
- `backend(): string` — the active int8 GEMM backend, `"wasm-simd128"` for the
  shipped SIMD build (so a silent scalar fallback can't be mistaken for it).

TypeScript definitions (`.d.ts`) are included.

## Which build (wasm-pack target)

This package is published from the `--target bundler` output, which works with
webpack, Rollup, and Vite. If you need a different runtime you can rebuild from
source with another `wasm-pack` target:

- `--target bundler` — general npm / bundlers (this package).
- `--target web` — native ES modules / `<script type="module">` without a bundler
  (you `await init()` the module yourself).
- `--target nodejs` — CommonJS `require()` in Node without a bundler.

All are built with `RUSTFLAGS="-C target-feature=+simd128"` for the SIMD128 kernel.

## License

MPL-2.0
