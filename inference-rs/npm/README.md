# fxtranslate

Translate using the [Firefox Translations](https://mozilla.github.io/translations/firefox-models/) models. These are high-quality, lightweight, CPU-optimized models that Firefox ships for on-device translation.

## Install

```console
# The CLI.
$ npm install -g fxtranslate

# The library.
$ npm install fxtranslate
```

Requires Node 18 or newer. The inference engine ships as a single WebAssembly artifact, so there are no prebuilt binaries to match, no native toolchain to install, and no compile step — it runs anywhere Node does.

## The CLI

```sh
# Discover what models are available.
$ fxtranslate list
$ fxtranslate list es
$ fxtranslate list --all

# Translate a phrase. The model for the pair is discovered, downloaded, and
# cached on first use, then reused from disk on subsequent runs.
$ fxtranslate translate en es "The weather is nice today."
> El clima es agradable hoy.

# You can switch languages.
$ fxtranslate translate en de "Translations are fun"
> Übersetzungen machen Spaß

# Changing the language order changes the translation direction.
$ fxtranslate translate es en "Buenos días, ¿cómo estás?"
> Good morning, how are you?

# When translating between languages where there is not a specific matching language pair,
# it translates through a "pivot language".
# Here Spanish to Russian pivots through a common English model: es → en → ru.
$ fxtranslate translate es ru "Buenos días, ¿cómo estás?"
> Доброе утро, как дела?

# Translate entire documents by piping text into the CLI.
$ cat document.txt | fxtranslate translate en es > document-es.txt

# Enter into an interactive translation mode.
$ fxtranslate translate en es

# Access the full CLI documentation.
$ fxtranslate --help
```

Status lines — model resolution, download progress — go to stderr, so a piped stdout carries only the translations.

## Model usage

The auto-discovery is powered by Firefox's internal model delivery service. This should not be used for production services. Please download and re-host the models. They can be downloaded through the CLI, or manually from the [mozilla/translations models dashboard](https://mozilla.github.io/translations/firefox-models/). The CLI has best-effort support for model downloads, but may break.

## Library examples

Load models from your local model store.

```js
import { readFileSync } from "node:fs";
import { Translator } from "fxtranslate";

// English-Spanish has a shared vocab file.
const vocab = readFileSync("models/en-es/vocab.enes.spm");

const enEs = new Translator(
  readFileSync("models/en-es/model.enes.intgemm.alphas.bin"),
  vocab,
  vocab,
);

console.log(enEs.translate_long("The weather is nice today. Don't you think so?"));
// El clima es agradable hoy. ¿No lo crees?

console.log(enEs.translate("The weather is nice today."));
// El clima es agradable hoy.
```

`translate_long` segments the text into sentences and translates each one, which is
the right default for input you haven't split yourself. `translate` treats its
argument as a single sentence and silently truncates anything past the model's
context size.

Some pairs have split vocabs, like English-Japanese. The source and target vocabs differ.

```js
const enJa = new Translator(
  readFileSync("models/en-ja/model.enja.intgemm.alphas.bin"),
  readFileSync("models/en-ja/srcvocab.enja.spm"),
  readFileSync("models/en-ja/trgvocab.enja.spm"),
);
```

A fourth, optional argument takes a lexical shortlist (`lex.50.50.enes.s2t.bin`), which restricts the output vocabulary per sentence and speeds up decoding.

Most supported languages can translate between each other. When no direct model exists for a pair, the translation routes through a pivot language; for 50 languages that means fewer than 100 models rather than the 2,450 a fully direct matrix would need. The CLI picks the route automatically — `resolveRoute` is the same decision exposed to you — but it can also be done manually.

```js
const esEn = new Translator(...);
const enRu = new Translator(...);

console.log(enRu.translate_long(esEn.translate_long("Buenos días, ¿cómo estás?")));
```

The discovery and routing helpers the CLI is built on are exported too, so you can
drive model management yourself. Each returns a JSON string, so parse what you need.

```js
import { parseRecords, modelPairs, resolveRoute, catalog } from "fxtranslate";

const url =
  "https://firefox.settings.services.mozilla.com" +
  "/v1/buckets/main/collections/translations-models-v2/records";
const body = await fetch(url).then((response) => response.text());

// Every one-way model pair, version-gated to what this engine can load.
const pairs = JSON.parse(modelPairs(body));
// [ ["af", "en"], ["ar", "en"], ["az", "en"], ["be", "en"], ... ]

// How a given pair is served: a direct model, or a two-leg pivot.
console.log(JSON.parse(resolveRoute(body, "es", "fr")));
// { kind: 'pivot', src: 'es', pivot: 'en', trg: 'fr' }

// Languages grouped by the directions they support, and the individual
// model-file records behind them.
const { bidirectional, sourceOnly, targetOnly } = JSON.parse(catalog(body, "en"));
const records = JSON.parse(parseRecords(body));
```

Fetch all model files once, to re-host them. This can be several gigabytes. Each
record carries a `location` to append to the attachments CDN root and a
`decompressedHash`; `verifyAndDecompress` does the zstd decode and the SHA-256
check in one step, the same way the CLI does.

```js
import { writeFileSync } from "node:fs";
import { parseRecords, verifyAndDecompress } from "fxtranslate";

const CDN_ROOT = "https://firefox-settings-attachments.cdn.mozilla.net";

for (const record of JSON.parse(parseRecords(body))) {
  const response = await fetch(`${CDN_ROOT}/${record.location}`);
  const compressed = new Uint8Array(await response.arrayBuffer());
  writeFileSync(record.name, verifyAndDecompress(compressed, record.decompressedHash));
}
```

Or let the CLI do it, which writes into the same verified cache that `translate` reads.

```sh
# Download a pair ahead of time. Both legs of a pivot are fetched.
$ fxtranslate models add es fr

# What's cached, how big it is, and where it lives.
$ fxtranslate models list
$ fxtranslate models info en-es

# Remove a pair, or the whole cache.
$ fxtranslate models rm en-es
$ fxtranslate models rm --all
```

Without a `--cache-dir`, models land in the platform-native cache directory:

 * **macOS** – `~/Library/Caches/fxtranslate/models`
 * **Linux** – `$XDG_CACHE_HOME/fxtranslate/models` or `~/.cache/fxtranslate/models`
 * **Windows** – `%LOCALAPPDATA%\fxtranslate\models`

## How this works

The underlying inference engine is a portable Rust library based on the [Marian](https://github.com/marian-nmt/marian-dev/) expression graph, compiled here to WebAssembly with a pure-Rust SIMD128 int8 kernel in place of the native build's [Gemmology matrix library](https://github.com/mozilla/gemmology). The Firefox models have a similar architecture to the traditional encoder/decoder [transformer models](https://arxiv.org/abs/1706.03762), but with a shallow RNN decoder based on the [SSRU described here](https://aclanthology.org/D19-5632/). These models come from [Mozilla's translation training program](https://github.com/mozilla/translations). They are student models distilled and quantized for CPU from larger transformer-based teacher models.

- **[`fxtranslate` on crates.io](https://crates.io/crates/fxtranslate)** – The Rust inference engine library
- **[`fxtranslate-cli` on crates.io](https://crates.io/crates/fxtranslate-cli)** – The Rust CLI
- **[`fxtranslate` on pypi](https://pypi.org/project/fxtranslate/)** – The Python bindings library and CLI

## License

MPL-2.0 from [Firefox Translations](https://github.com/mozilla/translations)
