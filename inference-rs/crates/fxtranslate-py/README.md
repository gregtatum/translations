# fxtranslate

Translate with [Firefox Translations](https://mozilla.github.io/translations/firefox-models/) models from Python — a **library** powered by the same lightweight, CPU-only models Firefox ships for on-device translation. The engine is native compiled Rust (fast, CPU-only, no GPU), and it discovers, downloads, and caches the models for you; no API keys, no network round-trips to a translation service, nothing leaves your machine after the model download.

Unlike the npm/wasm build, the Python wheel ships the fast, batteries-included engine: the [gemmology](https://github.com/mozilla/gemmology) SIMD int8 GEMM kernel, `mmap`, ICU4X sentence segmentation, and built-in networked model discovery/download/caching. Prefer a native binary, a Rust dependency, or Node? See [Related packages](#related-packages).

## Install

```console
$ pip install fxtranslate
```

The first release ships an sdist plus one platform wheel (built on the release machine). On other platforms `pip` compiles from source — that needs a Rust and C++ toolchain, and it falls back to the portable scalar kernel automatically if no SIMD backend is wired for your target — until the full binary-wheel matrix lands.

## Library, batteries-included

Lead with this — the native advantage over the wasm build. One call discovers the model for the pair, downloads and hash-verifies it into a local cache, builds the engine, and is ready to translate:

```python
from fxtranslate import Translator

t = Translator.load("en", "es")          # discover → download → cache → ready
print(t.translate("The weather is nice today."))
# El clima es agradable hoy.

# translate_long segments multi-sentence / long input and translates each part
# within the model's context window, instead of truncating.
print(t.translate_long(long_article))
```

## Library, bring-your-own model

If you already have the model files on disk (recommended for production — see [The models](#the-models)), pass the bytes directly. You supply the marian `.bin` model and the source/target SentencePiece vocabularies (the same file for most pairs; only CJK pairs differ) — pass the source vocab twice for a shared-vocab pair:

```python
from fxtranslate import Translator

model = open("en-es/model.bin", "rb").read()
vocab = open("en-es/vocab.spm", "rb").read()
t = Translator(model, vocab, vocab)       # src vocab twice for shared-vocab pairs
print(t.translate("The weather is nice today."))
```

## Discovery helpers

`fxtranslate.discovery` re-exports the pure model-discovery/routing surface the batteries-included path is built on — over a Remote Settings records body, returning native Python objects (no JSON strings) — for building your own model-management flow:

```python
from fxtranslate import discovery

route = discovery.resolve_route(records_json, "en", "es")  # {"kind": "direct", ...}
cat = discovery.catalog(records_json, "en")           # {"bidirectional": [...], ...}
pairs = discovery.model_pairs(records_json)           # list[(src, trg)]
recs = discovery.parse_records(records_json)          # list[dict]
sents = discovery.segment_sentences(text)             # list[str] (ICU4X UAX #29)
plain = discovery.verify_and_decompress(zst_bytes, sha256_hex)  # bytes
```

## The models

`fxtranslate` uses the models Firefox ships. They're discovered through Mozilla's Remote Settings and downloaded from Firefox's CDN, then cached under the platform-native cache directory — `~/Library/Caches/fxtranslate/models` on macOS, `$XDG_CACHE_HOME/fxtranslate/models` on Linux, `%LOCALAPPDATA%\fxtranslate\models` on Windows — one subdirectory per language pair.

**That hosting is provisioned for Firefox, not for third-party traffic**, and neither the endpoints nor the availability of any particular model are a stable contract for downstream projects. `Translator.load` is a convenient way to try the engine; if you're shipping a product on top of it, mirror the model files you depend on, serve them from infrastructure you control, and load them with the bring-your-own-model `Translator(...)` constructor.

## Related packages

The same engine is published across ecosystems on one shared version, so you can pick the artifact that fits:

- **[`fxtranslate`](https://crates.io/crates/fxtranslate)** (crates.io) — the Rust engine library. Native SIMD int8 kernel, optional built-in model management.
- **[`fxtranslate-cli`](https://crates.io/crates/fxtranslate-cli)** (crates.io) — the native Rust CLI, installed with `cargo install fxtranslate-cli`.
- **[`fxtranslate`](https://www.npmjs.com/package/fxtranslate)** (npm) — the Node CLI + library (WebAssembly core).

## License

MPL-2.0. The engine is a Rust port of the [Firefox Translations](https://github.com/mozilla/translations) inference engine.
