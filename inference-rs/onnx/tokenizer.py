"""SentencePiece wrappers for the en-fr student model.

Source encoding appends EOS (id 0) and does NOT prepend BOS. The decoder seeds
generation with the target EOS itself, so target decoding just maps ids -> text.
"""

from __future__ import annotations

import functools

import sentencepiece as spm

from model_npz import EOS_ID, SRC_SPM_PATH, TGT_SPM_PATH

__all__ = ["eos_id", "encode_source", "decode_ids"]

eos_id = EOS_ID


@functools.lru_cache(maxsize=1)
def _src() -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(SRC_SPM_PATH))


@functools.lru_cache(maxsize=1)
def _tgt() -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(TGT_SPM_PATH))


def encode_source(text: str) -> list[int]:
    """Encode source text to ids, appending EOS (no BOS)."""
    ids = _src().encode(text, out_type=int)
    ids.append(EOS_ID)
    return ids


def decode_ids(ids: list[int]) -> str:
    """Decode target ids to a string (EOS is stripped by the caller)."""
    return _tgt().decode([int(i) for i in ids])
