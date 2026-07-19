"""Firefox Translations neural machine-translation engine (native compiled Rust).

This package is a thin re-export of the compiled extension module
``fxtranslate._engine`` (built by maturin from the ``fxtranslate-py`` crate),
so consumers write::

    from fxtranslate import Translator, discovery

``Translator`` builds an engine either from in-memory model + vocab bytes
(``Translator(model, src_vocab, trg_vocab, shortlist=None)``) or the batteries-
included ``Translator.load(src, trg, cache_dir=None)`` classmethod, which discovers,
downloads, caches, and builds an engine for a language pair in one call. ``discovery``
exposes the pure model-discovery helpers (record parsing, route resolution, catalog,
segmentation, verify+decompress).
"""

from ._engine import Cache, Translator, discovery

__all__ = ["Cache", "Translator", "discovery"]
