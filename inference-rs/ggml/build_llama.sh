#!/usr/bin/env bash
# Build the CPU-only libllama for the Marian arch (G2/M1) and compile the encoder-parity
# driver (ggml/marian_encoder_dump) against it.
#
# The driver feeds exact source ids to LLM_ARCH_MARIAN's encoder graph and dumps the full
# [n_embd, seq] context for validate_llama_encoder.py. CPU-only + no BLAS keeps it a clean
# apples-to-apples float reference (Metal's embedded shader path is unavailable when linking
# libllama out-of-tree, and would only add GPU float noise anyway).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # inference-rs/ggml
LLAMA_DIR="${LLAMA_DIR:-$HOME/dev/llama.cpp}"
BUILD="$LLAMA_DIR/build-cpu"

if [[ ! -d "$LLAMA_DIR/src/models" ]]; then
  echo "[ggml-llama-build] llama.cpp not found at LLAMA_DIR=$LLAMA_DIR" >&2
  exit 1
fi

echo "[ggml-llama-build] configuring CPU-only libllama (Metal/BLAS off)"
cmake -S "$LLAMA_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF >/dev/null
echo "[ggml-llama-build] building libllama"
cmake --build "$BUILD" --target llama -j8 >/dev/null

for drv in marian_encoder_dump marian_decoder_dump marian_llama_blockbench; do
  echo "[ggml-llama-build] compiling $drv"
  clang++ -std=c++17 -O2 -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    "$HERE/$drv.cpp" \
    "$BUILD/bin/libllama.dylib" "$BUILD/bin/libggml.dylib" \
    "$BUILD/bin/libggml-base.dylib" "$BUILD/bin/libggml-cpu.dylib" \
    -Wl,-rpath,"$BUILD/bin" \
    -o "$HERE/$drv" 2>&1 | grep -v "ld: warning" || true
done
echo "[ggml-llama-build] done -> $HERE/marian_encoder_dump, $HERE/marian_decoder_dump, $HERE/marian_llama_blockbench"
