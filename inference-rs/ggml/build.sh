#!/usr/bin/env bash
# Build the bare-libggml translation engine (G1 of the ggml-port eval; see notes/18).
#
# Two idempotent steps:
#   1. Build libggml static libs (CPU-only, no Metal/BLAS/OpenMP — CPU-first per the eval
#      mandate) from a local ggml checkout ($GGML_DIR, default ~/dev/ggml, a sibling
#      checkout like marian-dev/gemmology) into ggml/build-ggml/.
#   2. Compile ggml/marian_ggml.cpp against those static libs -> ggml/marian_ggml.
#
# Re-runs are cheap: cmake --build no-ops when nothing changed; the engine relinks only
# when the source or libs are newer.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # inference-rs/ggml
GGML_DIR="${GGML_DIR:-$HOME/dev/ggml}"
BUILD="$HERE/build-ggml"
LIBS="$BUILD/src"

if [[ ! -d "$GGML_DIR/include" ]]; then
  echo "[ggml-build] ggml checkout not found at GGML_DIR=$GGML_DIR" >&2
  echo "             clone https://github.com/ggml-org/ggml there, or set GGML_DIR." >&2
  exit 1
fi

if [[ ! -f "$LIBS/libggml-base.a" ]]; then
  echo "[ggml-build] configuring libggml (CPU-only) from $GGML_DIR"
  cmake -S "$GGML_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
    -DGGML_STATIC=ON -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF >/dev/null
fi
echo "[ggml-build] building libggml static libs"
cmake --build "$BUILD" --target ggml ggml-base ggml-cpu -j8 >/dev/null

echo "[ggml-build] compiling marian_ggml"
clang++ -std=c++17 -O3 -mmacosx-version-min=15.0 -I"$GGML_DIR/include" \
  "$HERE/marian_ggml.cpp" \
  "$LIBS/libggml.a" "$LIBS/libggml-cpu.a" "$LIBS/libggml-base.a" \
  -framework Accelerate -framework Foundation \
  -o "$HERE/marian_ggml"
echo "[ggml-build] done -> $HERE/marian_ggml"
