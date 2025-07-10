#!/bin/bash
##
# Generates a lexical shortlist for a corpus.
#
#
# It also generate SentencePiece tokenized alignments that are required for extract_lex
#

set -x
set -euo pipefail

echo "###### Generating alignments and shortlist"
[[ -z "${MARIAN}" ]] && echo "MARIAN is empty"
[[ -z "${BIN}" ]] && echo "BIN is empty"
[[ -z "${SRC}" ]] && echo "SRC is empty"
[[ -z "${TRG}" ]] && echo "TRG is empty"

corpus_prefix=$1
vocab_src=$2
vocab_trg=$3
output_dir=$4
threads=$5

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

cd "$(dirname "${0}")"

mkdir -p "${output_dir}"
tmp_dir="${output_dir}/tmp_shortlist"
mkdir -p "${tmp_dir}"

corpus_src="${corpus_prefix}.${SRC}.zst"
corpus_trg="${corpus_prefix}.${TRG}.zst"


echo "### Tokenize the source and target corpus with with SentencePiece"
zstdmt -dc "${corpus_src}" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_src}" \
   >"${tmp_dir}/corpus.spm.${SRC}"

zstdmt -dc "${corpus_trg}" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_trg}" \
   >"${tmp_dir}/corpus.spm.${TRG}"

echo "### Generate the corpus.aln alignments"
python3 align.py \
  --corpus_src="${tmp_dir}/corpus.spm.${SRC}" \
  --corpus_trg="${tmp_dir}/corpus.spm.${TRG}" \
  --output_path="${output_dir}/corpus.aln"

echo "### Extract the source-target and target-source lexicon from the corpus"
# https://github.com/marian-nmt/extract-lex
# This step takes the SentencePiece tokenized corpus witih aligments and extracts
# the statistical distribution of the source token to the target token, and vice versa.
# Both source->target and target->source are generated, but we only need source->target.
#
# The output format is:
#   <target_word> <source_word> <probability>
"${BIN}/extract_lex" \
  "${tmp_dir}/corpus.spm.${TRG}" \
  "${tmp_dir}/corpus.spm.${SRC}" \
  "${output_dir}/corpus.aln" \
  "${tmp_dir}/lex.s2t" \
  "${tmp_dir}/lex.t2s"

# Remove files that are no longer needed.
rm "${tmp_dir}lex.t2s" # The target to source probabilities are not needed.
rm "${tmp_dir}/corpus.spm.${TRG}"
rm "${tmp_dir}/corpus.spm.${SRC}"
rm "${output_dir}/corpus.aln"

echo "### Convert the vocab to the text format"
"${MARIAN}/spm_export_vocab" --model="${vocab_trg}" --output="${tmp_dir}/vocab.txt"

echo "### Build the shortlist by pruning the source to target lexicon to $max_words"
max_words=100
cat "${tmp_dir}/lex.s2t" |
  python3 "prune_shortlist.py" ${max_words} "${tmp_dir}/vocab.txt" |
  zstdmt >"${output_dir}/lex.s2t.pruned.zst"

echo "### Delete ${tmp_dir}"
rm -rf "${tmp_dir}"

echo "###### Done: Generating alignments and shortlist"
