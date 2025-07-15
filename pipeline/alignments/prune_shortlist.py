"""
Prune a lexicon of source to target tokens with their probabilities into a
lexical shortlist, that can provide the top probabilities for a translation.

This script filters a source-to-target alignment lexicon by:
1. Selecting up to `MAX` top target candidates (by alignment probability) per source token.
2. Appending a predefined list of globally frequent target tokens (the shortlist vocabulary).
3. Emitting only those entries that are actually observed in the alignment data.

Example usage:
  cat lex.s2t | grep -v NULL | python3 prune_shortlist.py 100 vocab.txt > lex.s2t.pruned

Inputs:
  - stdin: lines of the format "<target_token> <source_token> <probability>"
           typically produced by the Marian `extract_lex` tool.
  - MAX: maximum number of top target candidates to retain per source token.
  - vocab.txt: text-formatted vocabulary list of frequent target tokens (one per line),
               exported from SentencePiece via `spm_export_vocab`.

Output:
  - stdout: pruned lexicon in the same format, keeping only high-probability and shortlist entries.

Notes:
  - Lines with "NULL" as source or target are excluded early.
  - Vocabulary shortlist is padded by 2 to allow for overlap margin.
  - All input and output tokens are assumed to be SentencePiece tokenized.
"""

from pathlib import Path
import sys
import argparse
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--max_words",
        type=int,
        help="Maximum number of top target candidates to retain per source token.",
    )
    parser.add_argument(
        "--vocab_txt_file",
        type=Path,
        help=(
            "text-formatted vocabulary list of frequent target tokens (one per line), "
            "exported from SentencePiece via `spm_export_vocab`."
        ),
    )

    args = parser.parse_args()

    max_words: int = args.max_words
    vocab_txt_file: Path = args.vocab_txt_file

    top_vocab = []
    with vocab_txt_file.open() as f:
        for line in f:
            top_vocab.append(line.strip().split()[0])

    top_vocab = top_vocab[: max_words + 2]

    vocab_src: set[str] = set()
    # A mapping of the src->trg probabilities:
    # e.g. pairs[src][trg] = probability
    pairs: dict[str, dict[str, float]] = {}

    for line in sys.stdin:
        try:
            trg, src, probability_str = cast(tuple[str, str, str], line.strip().split())
        except ValueError:
            # some lines include empty items for zh-en, 63 from ~400k
            continue

        # If a token can't be found in SentencePiece then NULL is used by extract-lex.
        # Skip these entries. https://github.com/marian-nmt/extract-lex/blob/42fa605b53f32eaf6c6e0b5677255c21c91b3d49/src/extract-lex-main.cpp#L175C33-L175C49
        if trg == "NULL" or src == "NULL":
            continue

        vocab_src.add(src)

        probability = float(probability_str)
        if src in pairs:
            pairs[src][trg] = probability
        else:
            pairs[src] = {trg: probability}

    for src in vocab_src:
        trg_probabilities = pairs[src]

        l = {"a": 0}

        # Only retain up to the top probabilities.
        top_trg_tokens = sorted(
            trg_probabilities,
            key=lambda key: trg_probabilities[key],
            reverse=True,
        )
        top_trg_tokens = top_trg_tokens[:max_words]

        for trg in set(top_trg_tokens + top_vocab):
            if trg in trg_probabilities:
                probability = trg_probabilities[trg]
                print("{} {} {:.8f}".format(trg, src, probability))


if __name__ == "__main__":
    main()
