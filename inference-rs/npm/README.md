# fxtranslate

Translate with [Firefox Translations](https://mozilla.github.io/translations/firefox-models/) models from Node — a **command-line tool** and a **library**, powered by the same lightweight, CPU-only models Firefox ships for on-device translation. It discovers, downloads, and caches the models for you; no API keys, no network round-trips to a translation service, nothing leaves your machine after the model download.

The engine is the Rust [`fxtranslate`](https://crates.io/crates/fxtranslate) crate compiled to WebAssembly, so translations match the native Rust CLI. The command-line interface is byte-for-byte compatible with the Rust [`fxtranslate-cli`](https://crates.io/crates/fxtranslate-cli): same subcommands, flags, help text, and exit codes. Prefer a native binary or a Rust dependency? See [Related packages](#related-packages).

## Command-line tool

```console
$ npm install -g fxtranslate
```

```console
# List supported languages (or filter to one); --all shows the raw model pairs:
$ fxtranslate list
$ fxtranslate list es
$ fxtranslate list --all

# Translate a phrase. The model for the pair is discovered, downloaded, and
# cached on first use, then reused from disk on subsequent runs.
$ fxtranslate translate en es "The weather is nice today."
El clima es agradable hoy.

$ fxtranslate translate en de "Knowledge is power."
Wissen ist Macht.

$ fxtranslate translate es en "Buenos días, ¿cómo estás?"
Good morning, how are you?

# Neither side is English? It pivots through English automatically.
$ fxtranslate translate es fr "Buenos días."
Bonjour.
```

Every Firefox Translations model translates to or from English, so each direction is its own model (`en → es` and `es → en` are separate downloads). A pair where neither side is English — like `es → fr` — is served by **pivoting**: it runs `es → en` then `en → fr` automatically, so the full cartesian product of languages works. That's also why `list` shows *languages* by default rather than raw model pairs.

With no text argument, `translate` reads from stdin — one translation per line when piped, or an interactive prompt on a terminal:

```console
# Pipe mode: one line in, one translation out.
$ echo "The library opens at nine in the morning." | fxtranslate translate en fr
La bibliothèque ouvre à neuf heures du matin.

# Interactive prompt (Ctrl-D to quit).
$ fxtranslate translate en es
Interactive en→es. Type a sentence and press Enter; Ctrl-D to quit.
en→es» ...
```

Status lines (model resolution, download progress) are written to stderr, so piped stdout carries only the translations.

## Library

```console
$ npm install fxtranslate
```

```js
import { Translator } from "fxtranslate";
import { readFileSync } from "node:fs";

// You supply the model triple: the marian `.bin` and the source/target
// SentencePiece vocabularies (the same file for most pairs; only CJK pairs
// differ). Pass the source vocab twice for a shared-vocab pair.
const model = readFileSync("en-es/model.bin");
const vocab = readFileSync("en-es/vocab.spm");

const translator = new Translator(model, vocab, vocab);
console.log(translator.translate("The weather is nice today."));
// El clima es agradable hoy.

// translate_long segments multi-sentence / long input and translates each part
// within the model's context window, instead of truncating.
console.log(translator.translate_long(longArticleText));
```

The library also re-exports the pure discovery/routing helpers the CLI is built on — `resolveRoute`, `catalog`, `modelPairs`, `parseRecords`, `segmentSentences`, and `verifyAndDecompress` — for building your own model-management flow. Types ship in the box (`fxtranslate` is authored with JSDoc and a hand-maintained `.d.ts`).

## The models

`fxtranslate` uses the models Firefox ships. They're discovered through Mozilla's Remote Settings and downloaded from Firefox's CDN, then cached under the platform-native cache directory — `~/Library/Caches/fxtranslate/models` on macOS, `$XDG_CACHE_HOME/fxtranslate/models` on Linux, `%LOCALAPPDATA%\fxtranslate\models` on Windows — one subdirectory per language pair. Override the CLI's location with `--cache-dir <DIR>`.

**That hosting is provisioned for Firefox, not for third-party traffic**, and neither the endpoints nor the availability of any particular model are a stable contract for downstream projects. This package is a convenient way to try the engine; if you're shipping a product on top of it, mirror the model files you depend on, serve them from infrastructure you control, and load them with the `Translator` constructor.

## Related packages

The same engine is published across ecosystems on one shared version, so you can pick the artifact that fits:

- **[`fxtranslate`](https://crates.io/crates/fxtranslate)** (crates.io) — the Rust engine library. Native SIMD int8 kernel, optional built-in model management.
- **[`fxtranslate-cli`](https://crates.io/crates/fxtranslate-cli)** (crates.io) — the native Rust CLI. Same interface as this package's `fxtranslate` binary, installed with `cargo install fxtranslate-cli`.

## Contributing

Building the wasm core, the conformance strategy that keeps this package byte-identical to the Rust CLI, and how to run the parity harness are documented in [DEVELOPMENT.md](./DEVELOPMENT.md).

## License

[MPL-2.0](./LICENSE). The engine is a Rust port of the [Firefox Translations](https://github.com/mozilla/translations) inference engine.
