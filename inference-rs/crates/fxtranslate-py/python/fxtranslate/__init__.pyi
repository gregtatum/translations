"""Type stubs for the fxtranslate package.

The runtime surface is compiled (PyO3), so these stubs describe it for editors and
mypy — the Python analogue of npm's hand-maintained ``.d.ts``.
"""

from typing import Dict, List, Optional, Tuple

from . import discovery as discovery

class Translator:
    """A translation engine.

    Two ways in: a byte-for-byte constructor (host supplies the model and vocab
    buffers) and the ``load`` classmethod (native discover/download/cache).
    """

    def __init__(
        self,
        model: bytes,
        src_vocab: bytes,
        trg_vocab: bytes,
        shortlist: Optional[bytes] = ...,
    ) -> None:
        """Build from model + vocabulary bytes, with an optional lexical shortlist.

        Pass byte-identical ``src_vocab`` and ``trg_vocab`` for shared-vocab pairs
        (most pairs) and distinct buffers for split-vocab (CJK). A malformed model
        raises :class:`ValueError`.
        """
        ...
    @classmethod
    def load(cls, src: str, trg: str, cache_dir: Optional[str] = ...) -> "Translator":
        """Discover, download+cache (verified), and build a translator for
        ``src``→``trg`` in one call. ``cache_dir`` overrides the platform-native
        cache location. Raises :class:`ValueError` on failure."""
        ...
    def translate(self, text: str) -> str:
        """Translate a single sentence-unit ``text``."""
        ...
    def translate_long(self, text: str) -> str:
        """Translate arbitrary-length ``text`` (ICU4X sentence segmentation)."""
        ...
    def backend(self) -> str:
        """The active int8 GEMM backend name (a SIMD name, or ``"scalar"``)."""
        ...

class Cache:
    """The verified model cache — read/prune the ``<src>-<trg>`` model directories
    the ``models`` CLI verbs act on, through the same core ``Cache`` the native CLI
    uses (not a re-port). Powers the ``fxtranslate.cli`` ``models`` subcommands."""

    def __init__(self, cache_dir: Optional[str] = ...) -> None:
        """Open the cache at ``cache_dir``, or the platform-native default when omitted."""
        ...
    @property
    def root(self) -> str:
        """The cache root directory as a string."""
        ...
    def list_cached(self) -> List[Dict[str, object]]:
        """Every cached pair as ``[{"name": str, "bytes": int}, ...]``, sorted by name."""
        ...
    def pair_files(self, name: str) -> List[Tuple[str, int, str]]:
        """A cached pair's files as ``[(name, bytes, path), ...]``, sorted by name."""
        ...
    def remove_pair(self, name: str) -> bool:
        """Delete the ``name`` pair directory. ``True`` if removed, ``False`` if absent."""
        ...
