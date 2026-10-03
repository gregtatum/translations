#!/usr/bin/env python3
"""Emit a vocab-only GGUF from a SentencePiece ``.spm`` for the tokenizer-parity gate.

This is the tokenizer half of the G2 llama.cpp Marian port (notes/18). It is
deliberately separate from ``convert_marian_gguf.py`` (which handles weights):
The gate proves, fail-fast, that llama.cpp's UGM tokenizer reproduces the Marian
SentencePiece tokenizer byte-for-byte before any architecture code is written.

We mirror ``llama.cpp/conversion/t5.py`` (the working T5 vocab converter): parse
the SentencePiece protobuf, copy the piece list / scores / types, and forward the
``normalizer_spec`` whitespace flags + ``precompiled_charsmap`` verbatim so
llama.cpp's UGM path applies the identical darts-clone XCDA normalization,
Viterbi segmentation, and ``<0xNN>`` byte fallback.

``general.architecture`` is set to ``t5`` so llama.cpp selects LLAMA_VOCAB_TYPE_UGM
(keyed off ``tokenizer.ggml.model == "t5"``) and loads cleanly in --vocab-only
mode without demanding T5 hparams. Marian's real arch metadata is irrelevant to
tokenization and is added by the weight converter in a later milestone.

Run via ``task rs:ggml-convert-vocab -- <name> <path.spm>`` or directly:
    python3 convert_spm_gguf.py <name> <path.spm>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from sentencepiece import SentencePieceProcessor  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as spm_model  # noqa: E402

import gguf  # noqa: E402

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE / "models"

# SentencePiece piece type -> llama.cpp LLAMA_TOKEN_ATTR_* (gguf TokenType enum).
_UNUSED = gguf.TokenType.UNUSED


def _toktype(sp: SentencePieceProcessor, i: int) -> gguf.TokenType:
    if sp.IsUnknown(i):
        return gguf.TokenType.UNKNOWN
    if sp.IsControl(i):
        return gguf.TokenType.CONTROL
    if sp.IsUnused(i):
        return gguf.TokenType.UNUSED
    if sp.IsByte(i):
        return gguf.TokenType.BYTE
    return gguf.TokenType.NORMAL


def convert(name: str, spm_path: Path) -> Path:
    mp = spm_model.ModelProto()
    mp.ParseFromString(spm_path.read_bytes())
    ns = mp.normalizer_spec
    assert mp.trainer_spec.model_type == 1, "expected a UNIGRAM SentencePiece model"

    sp = SentencePieceProcessor()
    sp.LoadFromFile(str(spm_path))
    n = sp.vocab_size()

    tokens = [sp.IdToPiece(i).encode("utf-8") for i in range(n)]
    scores = [sp.GetScore(i) for i in range(n)]
    toktypes = [int(_toktype(sp, i)) for i in range(n)]

    _MODELS.mkdir(exist_ok=True)
    out = _MODELS / f"{name}.vocab.gguf"

    # arch "t5" -> UGM tokenizer path; vocab-only load ignores the missing T5 hparams.
    w = gguf.GGUFWriter(str(out), "t5")
    w.add_name(f"{name}-spm-vocab")

    w.add_tokenizer_model("t5")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)

    # Whitespace handling: forward the .spm normalizer_spec verbatim. escape_whitespaces
    # (-> the U+2581 substitution) and treat_whitespace_as_suffix are UGM defaults in
    # llama-vocab.cpp (escape=true, suffix=false) and are not GGUF-configurable, so we
    # only assert they match Marian here rather than emitting them.
    assert ns.escape_whitespaces is True, "Marian uses escaped whitespace; UGM default matches"
    assert mp.trainer_spec.treat_whitespace_as_suffix is False, "UGM prepends the space marker"
    w.add_add_space_prefix(ns.add_dummy_prefix)
    w.add_remove_extra_whitespaces(ns.remove_extra_whitespaces)
    if ns.precompiled_charsmap:
        w.add_precompiled_charsmap(ns.precompiled_charsmap)

    # Special ids. Marian: eos=</s>, unk=<unk>; no bos, no pad, no sep.
    if sp.unk_id() >= 0:
        w.add_unk_token_id(sp.unk_id())
    if sp.eos_id() >= 0:
        w.add_eos_token_id(sp.eos_id())
    if sp.bos_id() >= 0:
        w.add_bos_token_id(sp.bos_id())
    if sp.pad_id() >= 0:
        w.add_pad_token_id(sp.pad_id())
    # UGM defaults add_eos=true; the spm_encode goldens carry no eos, so keep it off
    # for the parity diff. The real decoder appends </s> itself (Marian convention).
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(
        f"[convert-vocab] {name}: {n} pieces, charsmap={len(ns.precompiled_charsmap)}B, "
        f"add_space_prefix={ns.add_dummy_prefix} remove_extra_ws={ns.remove_extra_whitespaces} "
        f"-> {out}"
    )
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: convert_spm_gguf.py <name> <path.spm>", file=sys.stderr)
        sys.exit(2)
    convert(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
