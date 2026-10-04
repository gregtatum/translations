# fxtranslate-cli - A batteries included translation CLI

Translate using the [Firefox Translations](https://mozilla.github.io/translations/firefox-models/) models. These are high-quality, lightweight, CPU-optimized models that Firefox ships for on-device translation. This crate is the command-line front end, and it handles discovering, downloading, and caching the models for you. The engine underneath is [`fxtranslate`](https://crates.io/crates/fxtranslate), a Rust port of the translation engine in Firefox; see that crate for the developer API and the performance details.

Prefer Node or Python? The same CLI and engine ship as the [`fxtranslate`](https://www.npmjs.com/package/fxtranslate) npm package (`npm install -g fxtranslate`) and as the [`fxtranslate`](https://pypi.org/project/fxtranslate/) PyPI package (`pip install fxtranslate`), all on one shared version.

## Install

```console
$ cargo install fxtranslate-cli
```

Installs an `fxtranslate` binary. It uses the native SIMD kernel where one is wired (aarch64 i8mm, x86_64 AVX2) and a portable scalar fallback everywhere else, so the install never needs a C++ toolchain to succeed.

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

Every Firefox Translations model translates to or from English, so each direction is its own model (`en → es` and `es → en` are separate downloads). That's why `list` shows *languages* by default — each one usable to and from the others — rather than raw model pairs. See [pivot-translations.md](https://github.com/mozilla/translations/blob/main/inference-rs/pivot-translations.md) for how a pivot resolves, what it costs in memory, and how it's validated.

## Model usage

The auto-discovery is powered by Firefox's internal model delivery service. This should not be used for production services. Please download and re-host the models. They can be downloaded through the CLI, or manually from the [mozilla/translations models dashboard](https://mozilla.github.io/translations/firefox-models/). The CLI has best-effort support for model downloads, but may break.

## Managing models

Downloads go into the same verified cache that `translate` reads, so `models list` shows exactly what a translation would load.

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

## Using the library

To embed the engine in your own program, depend on the [`fxtranslate`](https://crates.io/crates/fxtranslate) crate directly. Its README covers loading models, the cargo features that control the SIMD kernel and model downloading, and re-hosting the model files.

## How this works

The underlying inference engine is a portable Rust library based on the [Marian](https://github.com/marian-nmt/marian-dev/) expression graph powered by the [Gemmology matrix library](https://github.com/mozilla/gemmology). The Firefox models have a similar architecture to the traditional encoder/decoder [transformer models](https://arxiv.org/abs/1706.03762), but with a shallow RNN decoder based on the [SSRU described here](https://aclanthology.org/D19-5632/). These models come from [Mozilla's translation training program](https://github.com/mozilla/translations). They are student models distilled and quantized for CPU from larger transformer-based teacher models.

- **[`fxtranslate` on crates.io](https://crates.io/crates/fxtranslate)** – The Rust inference engine library
- **[`fxtranslate` on npm](https://www.npmjs.com/package/fxtranslate)** – The Node.js bindings library and CLI
- **[`fxtranslate` on pypi](https://pypi.org/project/fxtranslate/)** – The Python bindings library and CLI

## Changelog

See [CHANGELOG.md](https://github.com/gregtatum/translations/blob/inference-rs/inference-rs/CHANGELOG.md) for the release history.

## License

MPL-2.0 from [Firefox Translations](https://github.com/mozilla/translations)
