#!/bin/bash

##
# Builds Marian from the source, and packages the built binaries. This provides
# both the Marian binaries, and the SentencePiece binaries.
#
#  marian
#  marian-decoder
#  marian-scorer
#  marian-conv
#
#  spm_train
#  spm_encode
#  spm_export_vocab
#
# https://github.com/marian-nmt/marian
# https://github.com/google/sentencepiece

set -e
set -x
set -euo pipefail

pushd `dirname $0` &>/dev/null
MY_DIR=$(pwd)
popd &>/dev/null

patch=${1:-none}

export MARIAN_DIR=$MOZ_FETCHES_DIR/marian-source
export CUDA_DIR=$MOZ_FETCHES_DIR/cuda-toolkit

test -v CUDA_DIR

# Apply a patch file to the Marian source directory.
if [ "$patch" != "none" ]; then
  echo "###### Patching Marian"
  patch -d ${MARIAN_DIR} -p1 < ${MY_DIR}/${patch}
fi

echo "###### Compiling Marian"

mkdir -p "${MARIAN_DIR}/build"
cd "${MARIAN_DIR}/build"

cmake .. \
  -DUSE_SENTENCEPIECE=on \
  -DUSE_FBGEMM=on \
  -DCOMPILE_CPU=on \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_DIR}"

make -j "$(nproc)"

echo "###### Packaging Marian binaries"

cd $MARIAN_DIR/build
tar -cf $UPLOAD_DIR/marian.tar \
  "marian" \
  "marian-decoder" \
  "marian-scorer" \
  "marian-conv" \
  "spm_train" \
  "spm_encode" \
  "spm_export_vocab"

if [ -f "${MARIAN_DIR}/scripts/alphas/extract_stats.py" ]; then
  cd "${MARIAN_DIR}/scripts/alphas"
  tar -rf $UPLOAD_DIR/marian.tar extract_stats.py
fi

# Create marian.tar.zst
zstd --rm $UPLOAD_DIR/marian.tar
