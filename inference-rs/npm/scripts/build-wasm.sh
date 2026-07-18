#!/usr/bin/env bash
# Build the shared wasm core and copy its artifacts into this package.
#
# The core lives in the sibling Rust crate `crates/fxtranslate-wasm`. We build it
# with `wasm-pack --target nodejs` (CommonJS `require`, matching the CLI's Node
# runtime) and copy the generated `pkg/` into `npm/wasm/`. We COPY rather than
# reference the sibling so the published tarball is self-contained (`npm pack`
# includes `wasm/`) and so a consumer `npm install fxtranslate` gets the engine
# without the Rust workspace. `wasm/` is gitignored — it's a build artifact.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
crate="$repo/crates/fxtranslate-wasm"
dest="$here/wasm"

echo "building wasm core in $crate (target nodejs)…"
wasm-pack build --target nodejs "$crate"

echo "copying pkg → $dest"
rm -rf "$dest"
mkdir -p "$dest"
# Copy only the artifacts the package needs at runtime + for types.
cp "$crate/pkg/fxtranslate_wasm.js" \
   "$crate/pkg/fxtranslate_wasm_bg.js" \
   "$crate/pkg/fxtranslate_wasm.d.ts" \
   "$crate/pkg/fxtranslate_wasm_bg.wasm" \
   "$crate/pkg/fxtranslate_wasm_bg.wasm.d.ts" \
   "$dest/"

echo "wasm core ready in $dest"
