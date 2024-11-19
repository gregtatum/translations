#!/bin/bash
##
# Cleans a bilingual corpus using bicleaner-ai, a tool for filtering noisy data.
#
# See:
#   docs/bicleaner.md

set -x
set -euo pipefail

echo "###### Bicleaner filtering"

test -v SRC
test -v TRG
test -v CUDA_DIR
test -v CUDNN_DIR

# Configure CUDA and cuDNN libraries
export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:${CUDNN_DIR}:${LD_LIBRARY_PATH:+LD_LIBRARY_PATH:}

# The prefix for the corpus files.
# e.g.
# For: $MOZ_FETCHES_DIR/ELRC_2682_v1
#      $MOZ_FETCHES_DIR/ELRC_2682_v1.da.zst
#      $MOZ_FETCHES_DIR/ELRC_2682_v1.en.zst
corpus_prefix=$1

# The prefix for the output files:
# e.g.
# For: $TASK_WORKDIR/artifacts/ELRC_2682_v1
#      $TASK_WORKDIR/artifacts/ELRC_2682_v1.da.zst
#      $TASK_WORKDIR/artifacts/ELRC_2682_v1.en.zst
output_prefix=$2

# The minimum bicleaner scoring threshold to retain, valued between 0.0 and 1.0.
# e.g. 0.75 will retain sentence pairs with a score of 0.75 or higher.
bicleaner_threshold=$3

# How many CPU threads to use if using a bicleaner with the CPU.
# e.g. "auto" or a number
threads=$4

# The directory that contains the bicleaner-ai pack.
# e.g. $MOZ_FETCHES_DIR/bicleaner-ai-da-en
pack_dir=$5

if [ "$threads" = "auto" ]; then
  threads=$(nproc)
fi

output_dir=$(dirname "${output_prefix}")
mkdir -p "${output_dir}"

if [ "${bicleaner_threshold}" == "0" ] || [ "${bicleaner_threshold}" == "0.0" ]; then
  echo "Threshold is 0, skipping filtering"
  cp "${corpus_prefix}.${SRC}.zst" "${output_prefix}.${SRC}.zst"
  cp "${corpus_prefix}.${TRG}.zst" "${output_prefix}.${TRG}.zst"
  exit 0
fi

# The bicleaner model has a source and target language, although the model direction
# doesn't affect the scoring. Make sure our language pair ordering matches that of
# bicleaner-ai. For example if our language pair is "en-ru", the bicleaner model
# can be: "en-ru", "ru-en", or "en-xx".
export src_column=1
export trg_column=2
model_source_lang=$(grep "source_lang" ${pack_dir}/*.yaml | awk '{print $2}')
model_target_lang=$(grep "target_lang" ${pack_dir}/*.yaml | awk '{print $2}')
if [ ${model_source_lang} == ${TRG} ] || [ ${model_target_lang} == ${SRC} ]; then
  # swap columns
  export src_column=2
  export trg_column=1
  fi

# Export cuda visible devices if empty or not set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader);
fi

echo "### Classifying"
if [ ${#CUDA_VISIBLE_DEVICES} -gt 1 ]; then
  # There is more than 1 GPU available, use GNU's parallel to run bicleaner-ai on multiple GPUs.

  # Convert CUDA_VISIBLE_DEVICES to an array
  export CUDA_VISIBLE_ARRAY=(${CUDA_VISIBLE_DEVICES//,/ })

  # Turn on tensorflow logging in bicleaner-ai
  export TF_CPP_MIN_LOG_LEVEL=0

  # This function expects a bicleaner yaml and a 1-based index into the CUDA_VISIBLE_ARRAY
  # Example: /mnt/nanna0/nbogoych/data/data/fr-en/fr-en-prod/biclean/pack/metadata.yaml index_in_CUDA_VISIBLE_ARRAY+1
  biclean() {
    export CUDA_VISIBLE_ARRAY=(${CUDA_VISIBLE_DEVICES//,/ })
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_ARRAY[$(($2-1))]}
    # The GPU devices have failed to be found, and bicleaner AI falls back
    # to operate on the CPU very slowly. To guard against this wasting expensive
    # GPU time, always check that it can find GPUs.
    python3 -c "import tensorflow; exit(0) if tensorflow.config.list_physical_devices('GPU') else exit(9001)"
    bicleaner-ai-classify \
      --disable_hardrules \
      --scol ${src_column} \
      --tcol ${trg_column} - - $1
  }
  export -f biclean

  # {%} is a 1-indexed job slot number from GNU parallel.  We use that as the 1-indexed offset
  # in CUDA_VISIBLE_ARRAY
  paste \
    <(zstdmt -dc "${corpus_prefix}.${SRC}.zst") \
    <(zstdmt -dc "${corpus_prefix}.${TRG}.zst") \
    | parallel -j ${#CUDA_VISIBLE_ARRAY[@]} \
      --pipe -k \
      --block 10M \
      biclean "${pack_dir}"/*.yaml {%}
    | zstdmt >"${output_prefix}.scored.zst"
else
  # Only 1 gpu is available, so do not parallelize.
  export BICLEANER_AI_THREADS=${threads}

  paste \
    <(zstdmt -dc "${corpus_prefix}.${SRC}.zst") \
    <(zstdmt -dc "${corpus_prefix}.${TRG}.zst")
  | bicleaner-ai-classify \
    --disable_hardrules \
    --scol ${src_column} \
    --tcol ${trg_column} \
    "${threads}" - - "${pack_dir}"/*.yaml
  | zstdmt >"${output_prefix}.scored.zst"
fi

echo "### Filter the dataset based on the bicleaner score."

# Each line in the scored dataset takes the form:
#   "{src}\t{trg}\t{score}"
#
# To filter these lines, they are piped to awk:
#
# Set the threshold variable:
#   `-v threshold=${bicleaner_threshold}`
#
# Treat the input line as field separated with a "\t" delimiter.
#   `-F"\t"`
#
# Include a threshold test that then just prints out the entire original line if it meets
# the threshold requirement
#   '{if ($3>threshold) {print $0}}'
#

sample_limit=1000

echo "### Filtering and writing output corpus"

kept=$(
  zstdmt -dc "${output_prefix}.scored.zst" |
    |
    awk -v threshold=${bicleaner_threshold} -F"\t" '{if ($3>threshold) {print $0}}' |
    tee >(cut -f1 | zstdmt >"${output_prefix}.${SRC}.zst") |
    tee >(cut -f2 | zstdmt >"${output_prefix}.${TRG}.zst") |
    tee >(shuf -n ${sample_limit} > "${output_prefix}.kept.sample.txt") |
    wc -l
)

echo "### Count and sample the filtered information"

filtered=$(
  zstdmt -dc --rm "${output_prefix}.scored.zst"                               \
    | awk                                                                     \
        -v threshold=${bicleaner_threshold}                                   \
        -F"\t"                                                                \
        '{if ($3<=threshold) {print $0}}'                                     \
    | tee >(shuf -n ${sample_limit} > "${output_prefix}.filtered.sample.txt") \
    | wc -l
)

echo "### Write the final statistics of filtered content."

cat > "${output_prefix}.stats.json" << EOF
{
  "description": "The amount of lines filtered by bicleaner",
  "threshold": ${bicleaner_threshold},
  "visited": $((kept + filtered)),
  "filtered": ${filtered},
  "kept": ${kept}
}
EOF

echo "###### Done: Bicleaner filtering"
