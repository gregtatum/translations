"""Type stubs for the ``fxtranslate.discovery`` submodule.

Pure model-discovery helpers over Remote Settings records. Each takes the raw
records JSON body (same input contract as the wasm binding) and returns native
Python objects.
"""

from typing import Dict, List, Optional, Tuple

def parse_records(body: str) -> List[Dict[str, Optional[str]]]:
    """Parse a Remote Settings ``records`` body into model-file records.

    Each record is a dict with keys ``name``, ``fileType``, ``sourceLanguage``,
    ``targetLanguage``, ``version``, ``architecture``, ``decompressedHash``,
    ``location``; ``architecture`` and ``decompressedHash`` may be ``None``. Raises
    :class:`ValueError` on a parse error.
    """
    ...

def resolve_route(records_json: str, src: str, trg: str) -> Dict[str, str]:
    """Resolve ``src``→``trg`` to a route dict.

    ``{"kind": "direct", "src": .., "trg": ..}`` or
    ``{"kind": "pivot", "src": .., "pivot": .., "trg": ..}``. Raises
    :class:`ValueError` when neither a direct model nor a pivot exists.
    """
    ...

def catalog(records_json: str, hub: str) -> Dict[str, List[str]]:
    """Classify languages reachable through ``hub``.

    Returns ``{"bidirectional": [..], "sourceOnly": [..], "targetOnly": [..]}``.
    """
    ...

def model_pairs(records_json: str) -> List[Tuple[str, str]]:
    """The unique, version-gated ``(src, trg)`` model pairs."""
    ...

def segment_sentences(text: str) -> List[str]:
    """Split ``text`` into sentence units (ICU4X, UAX #29)."""
    ...

def verify_and_decompress(
    compressed: bytes, expected_sha256_hex: Optional[str] = ...
) -> bytes:
    """Decompress a zstd attachment and (optionally) verify its SHA-256.

    Raises :class:`ValueError` on a decode failure or hash mismatch.
    """
    ...
