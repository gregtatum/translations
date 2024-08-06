#!/bin/bash
##
# Scores a corpus with a reversed NMT model.
#


set -x
set -euo pipefail

echo "###### Scoring"
test -v MARIAN
test -v GPUS
test -v SRC
test -v TRG
test -v WORKSPACE

model=$1
vocab=$2
corpus_prefix=$3
output=$4

zstdmt --rm -d "${corpus_prefix}.${SRC}.zst"
zstdmt --rm -d "${corpus_prefix}.${TRG}.zst"

dir=$(dirname "${output}")
mkdir -p "${dir}"

"${MARIAN}/marian-scorer" \
  -m "${model}" \
  -v "${vocab}" "${vocab}" \
  -t "${corpus_prefix}.${TRG}.zst" "${corpus_prefix}.${SRC}.zst" \
  --mini-batch 32 \
  --mini-batch-words 1500 \
  --maxi-batch 1000 \
  --max-length 250 \
  --max-length-crop \
  --normalize \
  -d ${GPUS} \
  -w "${WORKSPACE}" \
  --log "${dir}/scores.txt.log" \
  >"${output}"

echo "###### Done: Scoring"
