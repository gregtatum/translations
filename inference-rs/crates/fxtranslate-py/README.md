# fxtranslate

Translate using the [Firefox Translations](https://mozilla.github.io/translations/firefox-models/) models. These are high-quality, lightweight, CPU-optimized models that Firefox ships for on-device translation.

## Install

```console
$ pip install fxtranslate
```

Requires Python 3.8 or newer. Prebuilt wheels cover Linux (x86_64 and aarch64), macOS (Apple silicon), and Windows (x86_64). Other platforms, install from the source distribution, which compiles the engine locally. You'll need a [Rust toolchain](https://rustup.rs) and a C++ compiler installed first.

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
# Here Spanish to French pivots through a common English model: es → en → fr.
$ fxtranslate translate es fr "Buenos días."
> bonjour.

# Translate entire documents by piping text into the CLI.
$ cat document.txt | fxtranslate translate en es > document-es.txt

# Enter into an interactive translation mode.
$ fxtranslate translate en es

# Access the full CLI documentation.
$ fxtranslate --help
```

## Model usage

The auto-discovery is powered by Firefox's internal model delivery service. This should not be used for production services. Please download and re-host the models. They can be downloaded through the CLI, or manually from the [mozilla/translations models dashboard](https://mozilla.github.io/translations/firefox-models/). The CLI has best-effort support for model downloads, but may break.

## Library examples

Load models from your local model store.

```py
from pathlib import Path
from fxtranslate import Translator

# English-Spanish has a shared vocab file.
model_dir = Path("models/en-es")
vocab = (model_dir / "vocab.enes.spm").read_bytes()

en_es = Translator(
    model=(model_dir / "model.enes.intgemm.alphas.bin").read_bytes(),
    src_vocab=vocab,
    trg_vocab=vocab,
)

print(en_es.translate_long("The weather is nice today. Don't you think so?"))
print(en_es.translate("The weather is nice today."))
```

`translate_long` segments the text into sentences and translates each one, which is
the right default for input you haven't split yourself. `translate` treats its
argument as a single sentence and silently truncates anything past the model's
context size.

Some pairs have split vocabs, like English-Japanese. The source and target vocabs differ.

```py
model_dir = Path("models/en-ja")
en_ja = Translator(
    model=(model_dir / "model.enja.intgemm.alphas.bin").read_bytes(),
    src_vocab=(model_dir / "srcvocab.enja.spm").read_bytes(),
    trg_vocab=(model_dir / "trgvocab.enja.spm").read_bytes(),
)
```

Most supported languages can translate between each other. When no direct model exists for a pair, the translation routes through a pivot language; for 50 languages that means fewer than 100 models rather than the 2,450 a fully direct matrix would need. The CLI and `Translator.load` pick the route automatically, but it can also be done manually.

```py
es_en = Translator(...)
en_fr = Translator(...)

print(en_fr.translate_long(es_en.translate_long("Buenos días.")))
```

Fetch all model files once, to re-host them. This can be several gigabytes.

```py
from fxtranslate import Cache, discovery

records = discovery.fetch_records_body()
pairs = discovery.model_pairs(records)

for src, trg in pairs:
    discovery.add_models(src, trg, cache_dir="./models", progress=True)

cache = Cache("./models")
print(cache.pair_files("en-es"))
# [ ('lex.50.50.enes.s2t.bin',        4198436,  '/path/to/models/en-es/lex.50.50.enes.s2t.bin'),
#   ('model.enes.intgemm.alphas.bin', 31561787, '/path/to/models/en-es/model.enes.intgemm.alphas.bin'),
#   ('vocab.enes.spm',                816054,   '/path/to/models/en-es/vocab.enes.spm')]
```

Or just specific ones.

```py
for src, trg in [("en", "es"), ("es", "en"), ("en", "de"), ("de", "en")]:
    discovery.add_models(src, trg, cache_dir="./models", progress=True)
```

Without a `cache_dir`, models land in the platform-native cache directory:

 * **macOS** – `~/Library/Caches/fxtranslate/models`
 * **Linux** – `$XDG_CACHE_HOME/fxtranslate/models` or `~/.cache/fxtranslate/models`
 * **Windows** – `%LOCALAPPDATA%\fxtranslate\models`

`Cache` allows for working with cached models.

```py
from fxtranslate import Cache

cache = Cache()
print(cache.root)
# /path/to/cache/fxtranslate/models

for entry in cache.list_cached():
    print(entry["name"], entry["bytes"])
  # en-de 36719532
  # en-es 36576277

# Removes the model.
cache.remove_pair("en-de")
```

## How this works

The underlying inference engine is a portable Rust library based on the [Marian](https://github.com/marian-nmt/marian-dev/) expression graph powered by the [Gemmology matrix library](https://github.com/mozilla/gemmology). The Firefox models have a similar architecture to the traditional encoder/decoder [transformer models](https://arxiv.org/abs/1706.03762), but with a shallow RNN decoder based on the [SSRU described here](https://aclanthology.org/D19-5632/). These models come from [Mozilla's translation training program](https://github.com/mozilla/translations). They are student models distilled and quantized for CPU from larger transformer-based teacher models.

- **[`fxtranslate` on crates.io](https://crates.io/crates/fxtranslate)** – The Rust inference engine library
- **[`fxtranslate-cli` on crates.io](https://crates.io/crates/fxtranslate-cli)** – The Rust CLI
- **[`fxtranslate` on npm](https://www.npmjs.com/package/fxtranslate)** – The Node.js bindings library and CLI
- **[`fxtranslate` on pypi](https://pypi.org/project/fxtranslate/)** – The Python bindings library and CLI

## License

MPL-2.0 from [Firefox Translations](https://github.com/mozilla/translations)
