#!/bin/bash
##
# Generates alignment and lexical shortlist for a corpus.
#

set -x
set -euo pipefail

echo "###### Generating alignments and shortlist"
test -v MARIAN
test -v BIN
test -v SRC
test -v TRG

corpus_prefix=$1
vocab_path=$2
output_dir=$3
threads=$4

COMPRESSION_CMD="${COMPRESSION_CMD:-pigz}"
ARTIFACT_EXT="${ARTIFACT_EXT:-gz}"

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

cd "$(dirname "${0}")"

mkdir -p "${output_dir}"
dir="${output_dir}/tmp"
mkdir -p "${dir}"

corpus_src="${corpus_prefix}.${SRC}.${ARTIFACT_EXT}"
corpus_trg="${corpus_prefix}.${TRG}.${ARTIFACT_EXT}"


echo "### Tokenize the source corpus with SentencePiece."
  test -s "${dir}/corpus.spm.${SRC}.${ARTIFACT_EXT}"                    \
                                                                        \
    `# De-compress the source corpus.`                                  \
    || ${COMPRESSION_CMD} -dc "${corpus_src}"                           \
                                                                        \
    `# Parallelize this tokenization.`                                  \
    | parallel --no-notice --pipe -k -j "${threads}" --block 50M        \
      `# Tokenize the corpus.`                                         \
      "${MARIAN}/spm_encode" --model "${vocab_path}"                    \
                                                                        \
    `# Save out the final tokenized corpus into a compressed file.`     \
    | ${COMPRESSION_CMD} >"${dir}/corpus.spm.${SRC}.${ARTIFACT_EXT}"

echo "### Tokenize the target corpus with SentencePiece."
  test -s "${dir}/corpus.spm.${TRG}.${ARTIFACT_EXT}"                    \
                                                                        \
    `# De-compress the target corpus.`                                  \
    || ${COMPRESSION_CMD} -dc "${corpus_trg}"                           \
                                                                        \
    `# Parallelize this tokenization.`                                  \
    | parallel --no-notice --pipe -k -j "${threads}" --block 50M        \
      `# Tokenize the corpus.`                                         \
      "${MARIAN}/spm_encode" --model "${vocab_path}"                    \
                                                                        \
    `# Save out the final tokenized corpus into a compressed file.`     \
    | ${COMPRESSION_CMD} >"${dir}/corpus.spm.${TRG}.${ARTIFACT_EXT}"


echo "### Merge the source and target corpus into one file"
  # Only re-generate these files if they don't exist.
  test -s "${output_dir}/corpus.aln.${ARTIFACT_EXT}" ||                         \
  test -s "${dir}/corpus"                            ||                         \
                                                                                \
    `# Join the source and target into the same line, using "\t" as delimiter`  \
    paste <(${COMPRESSION_CMD} -dc "${dir}/corpus.spm.${SRC}.${ARTIFACT_EXT}")  \
          <(${COMPRESSION_CMD} -dc "${dir}/corpus.spm.${TRG}.${ARTIFACT_EXT}")  \
                                                                                \
    `# Replace the "\" delimiter with the " ||| " delimiter. as fast_align`     \
    `# requires this as the delimiter`                                          \
    | sed 's/\t/ ||| /'                                                         \
                                                                                \
    `# Output this to the ./corpus file`                                        \
    > "${dir}/corpus"

echo "### Train the alignments using fast_align - https://github.com/clab/fast_align"
  # Only re-generate these files if they don't exist.
  test -s "${output_dir}/corpus.aln.${ARTIFACT_EXT}" ||                \
  test -s "${dir}/align.s2t.${ARTIFACT_EXT}"         ||                \
                                                                       \
    `# Run fast_align with the "source->target" alignment direction. ` \
    `# -vod is the recommended default configuration.`                 \
    "${BIN}/fast_align" -vod -i "${dir}/corpus"                        \
                                                                       \
    `# Save out the alignments to a compressed file `                  \
    | ${COMPRESSION_CMD} >"${dir}/align.s2t.${ARTIFACT_EXT}"

  # Only re-generate these files if they don't exist.
  test -s "${output_dir}/corpus.aln.${ARTIFACT_EXT}" ||                              \
  test -s "${dir}/align.t2s.${ARTIFACT_EXT}"         ||                              \
                                                                                     \
    `# Run fast_align with the (reversed -r) "target->source" alignment direction. ` \
    `# -vod is the recommended default configuration.`                               \
    "${BIN}/fast_align" -vodr -i "${dir}/corpus"                                     \
                                                                                     \
    `# Save out the alignments to a compressed file `                                \
    | ${COMPRESSION_CMD} >"${dir}/align.t2s.${ARTIFACT_EXT}"

echo "### Symmetrizing alignments"
  # fast_align generates asymmetric alignments, so the source-target and target-source
  # alignments can be different. This step symmetrizes the alignments so that they agree.
  # This will generate a final alignment where neither language is treated as the
  # "primary" language.

  # Decompress the alignment files.
  # Only re-generate these files if they don't exist.
  test -s "${output_dir}/corpus.aln.${ARTIFACT_EXT}" ||     \
  test -s "${dir}/align.t2s"                         ||     \
    `# Decompress the alignment files in place.`            \
    ${COMPRESSION_CMD} -d                                   \
      "${dir}/align.s2t.${ARTIFACT_EXT}"                    \
      "${dir}/align.t2s.${ARTIFACT_EXT}"

  # Only re-generate these files if they don't exist.
  test -s "${output_dir}/corpus.aln.${ARTIFACT_EXT}" ||                             \
                                                                                    \
    `# Use the atools from fast_align to symmetrize the results. This uses the `    \
    `# default "grow-diag-final-and" heuristic.`                                    \
    "${BIN}/atools"                                                                 \
      -i "${dir}/align.s2t"                                                         \
      -j "${dir}/align.t2s"                                                         \
      -c grow-diag-final-and                                                        \
                                                                                    \
    `# Save out the alignments to a compressed file `                               \
    | ${COMPRESSION_CMD} >"${output_dir}/corpus.aln.${ARTIFACT_EXT}"

echo "### Creating shortlist"
  # This step generates:
  # lex.s2t.zst
  # Only re-generate these files if they don't exist.
  test -s "${dir}/lex.s2t.${ARTIFACT_EXT}" ||
    if [ "${ARTIFACT_EXT}" = "zst" ]; then
      # extract_lex doesn't support zstd natively; we need to decrypt first.
      zstdmt --rm -d "${dir}/corpus.spm.${TRG}.${ARTIFACT_EXT}"
      zstdmt --rm -d "${dir}/corpus.spm.${SRC}.${ARTIFACT_EXT}"
      zstdmt --rm -d "${output_dir}/corpus.aln.${ARTIFACT_EXT}"

      "${BIN}/extract_lex"             \
        "${dir}/corpus.spm.${TRG}"     \
        "${dir}/corpus.spm.${SRC}"     \
        "${output_dir}/corpus.aln"     \
        "${dir}/lex.s2t"               \
        "${dir}/lex.t2s"
    else
      "${BIN}/extract_lex" \
        "${dir}/corpus.spm.${TRG}.${ARTIFACT_EXT}" \
        "${dir}/corpus.spm.${SRC}.${ARTIFACT_EXT}" \
        "${output_dir}/corpus.aln.${ARTIFACT_EXT}" \
        "${dir}/lex.s2t" \
        "${dir}/lex.t2s"
    fi

  # Only export the source->target direction, as it should be symmetrical.
  test -s "${dir}/lex.s2t" && ${COMPRESSION_CMD} "${dir}/lex.s2t"

echo "### Shortlist pruning"
  # Export the vocab from the SentencePiece model.
  test -s "${dir}/vocab.txt" ||
    "${MARIAN}/spm_export_vocab" --model="${vocab_path}" --output="${dir}/vocab.txt"

  test -s "${output_dir}/lex.s2t.pruned.${ARTIFACT_EXT}" ||
    ${COMPRESSION_CMD} -dc "${dir}/lex.s2t.${ARTIFACT_EXT}" |
    grep -v NULL |
    python3 "prune_shortlist.py" 100 "${dir}/vocab.txt" |
    ${COMPRESSION_CMD} >"${output_dir}/lex.s2t.pruned.${ARTIFACT_EXT}"

echo "### Deleting tmp dir"
rm -rf "${dir}"

echo "###### Done: Generating alignments and shortlist"
