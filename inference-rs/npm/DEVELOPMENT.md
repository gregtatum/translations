# Developing `fxtranslate` (npm)

Developer notes for the npm package: how it's structured, how it stays byte-identical to the Rust CLI, and how to build and test it. For *using* the package, see [README.md](./README.md); for the full design rationale (why the shell is native per language and only the pure decision logic is shared wasm), see [`notes/13-cli-parity.md`](../notes/13-cli-parity.md) in the repo.

## What this package is

One npm package, two entry points over a shared WebAssembly inference core:

- **CLI** — `bin/fxtranslate.js`, exposed as the `fxtranslate` binary.
- **Library** — `import { Translator, resolveRoute, catalog, modelPairs, parseRecords, segmentSentences, verifyAndDecompress } from "fxtranslate"`.

The command-line interface is a byte-for-byte reimplementation of the Rust `fxtranslate-cli` (the oracle): same subcommands, flags, help text, error strings, and exit codes.

## Layout

```
npm/
  bin/fxtranslate.js     CLI entry (thin shim: argv + terminal Io → run). Mirrors main.rs.
  lib/
    index.js             library entry — re-exports the wasm core + JS shell surface
    index.d.ts           hand-maintained consumer types (re-exports the wasm .d.ts)
    cli.js               arg grammar + help routing + error strings (ports Rust `parse`)
    usage.js             USAGE / MODELS_USAGE / LIST_USAGE, byte-for-byte from cli.rs
    run.js               parse → dispatch → exit code (ports Rust `run`/`dispatch`)
    translate.js         the translate command: async discovery/download shell over the sync wasm engine
    io.js                the host I/O + terminal contract (mirrors Rust `Io`)
    fetch.js             the Node `Fetch` (global fetch) + Remote Settings records URL
    cache.js             the fs cache reader/writer (mirrors cache.rs)
    format.js            list/models view formatters, byte-for-byte from cli.rs
    lang.js              tag → display-name table, ported from lang.rs
  scripts/build-wasm.sh  builds the wasm core and copies pkg/ → wasm/
  test/parity.js         interface-parity check vs the Rust oracle (grammar + cache + live list)
  tsconfig.json          non-publishing `tsc --noEmit` config (JSDoc type check)
  wasm/                  copied-in wasm core artifacts (gitignored; run build:wasm)
```

## The wasm core (copied, not referenced)

The shared inference + discovery/routing/verify/segment core lives in the sibling Rust crate `crates/fxtranslate-wasm`. `npm run build:wasm` runs `wasm-pack build --target nodejs` there and **copies** the generated `pkg/` artifacts into `npm/wasm/`. Copying (rather than referencing the sibling) keeps the published tarball self-contained — `npm pack` bundles `wasm/`, and a consumer `npm install fxtranslate` gets the engine without the Rust workspace. `wasm/` is a build artifact and is gitignored, so **a fresh `build:wasm` is a prerequisite for packing or publishing** (the `prepublishOnly` script enforces this).

The `crates/fxtranslate-wasm` crate is `publish = false` — that keeps it off crates.io (its engine is already published as `fxtranslate`); it does **not** stop wasm-pack from building the npm artifact.

## The conformance strategy: byte-identical to the Rust CLI

The npm CLI is not an independent implementation that happens to look similar — it is held byte-identical to the Rust `fxtranslate-cli` by a two-layer contract:

- **The sync core is literally the same code.** Discovery, direct-vs-pivot routing, sentence segmentation, and archive verify/decompress are the Rust engine compiled to wasm, so their outputs can't drift from the native CLI — they *are* the native CLI's logic.
- **The async shell is ported and pinned by tests.** Argument grammar, help/usage text, error strings, exit codes, the cache layout, and the list/models formatters are hand-ported JS, then proven against the Rust binary as an oracle. `npm run check:parity` runs both CLIs over every case and asserts byte-identical stdout, stderr, and exit code:
  - hermetic grammar / help / error cases,
  - hermetic `models list` / `models info` against a built fixture cache,
  - live `list` against Remote Settings.

The harness needs the oracle built: `cargo build -p fxtranslate-cli`. The repo-level `scripts/conformance.py` (`task rs:conformance`) is the broader cross-CLI harness this feeds into.

The translate path holds a numeric tolerance rather than byte-identity, by design: it deliberately builds the engine from model + two vocabs only (no shortlist), matching the Rust `Engine::load`, so both stay within the documented translation tolerance.

## Types without a compile step

The `.js` is authored with JSDoc and **ships as written** — there is no TypeScript transpile step in the publish path.

- **Local checks:** `npm run typecheck` runs `tsc --noEmit` over the JSDoc + the hand-written `.d.ts`. It validates types (and catches `.d.ts` drift against the wasm-generated engine types) without emitting anything.
- **Packaged types:** `lib/index.d.ts` is the `types` entry. It re-exports the wasm-pack-generated `wasm/fxtranslate_wasm.d.ts` for the engine surface (so the engine types can't drift from the build) and hand-describes the small JS shell surface.

## Scripts

- `npm run build:wasm` — build + copy the wasm core into `wasm/`. Needs `wasm-pack` and the `wasm32-unknown-unknown` target.
- `npm run typecheck` — `tsc --noEmit` over the JSDoc (no transpile).
- `npm run check:parity` / `npm test` — run the Rust oracle and this CLI over every no-I/O case and assert byte-identical stdout, stderr, and exit code. Requires `cargo build -p fxtranslate-cli`.
- `prepublishOnly` — rebuilds the wasm core, typechecks, and runs the parity harness, so `npm publish` can never ship a stale or missing `wasm/`.

## Releasing

The npm package is published **in lockstep with the crates.io crates**, on one shared version, by `scripts/publish.py`. Don't `npm publish` by hand for a release — run the workspace publisher so the crates and the npm package move together and get one git tag. See [RELEASING.md](../RELEASING.md).
