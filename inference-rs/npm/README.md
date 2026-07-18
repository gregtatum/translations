# fxtranslate (npm)

Translate with Firefox Translations models — a Node **CLI** and **library** over
the shared wasm inference core. The command-line interface is byte-for-byte
compatible with the Rust `fxtranslate-cli` (the oracle): same subcommands, flags,
help text, error strings, and exit codes.

One package, two entry points:

- **CLI** — `bin/fxtranslate.js`, exposed as the `fxtranslate` binary.
- **Library** — `import { Translator, resolveRoute, catalog, modelPairs, parseRecords, segmentSentences, verifyAndDecompress } from "fxtranslate"`.

See `notes/13-cli-parity.md` in the repo for the full design (why the shell is
native per language and only the pure decision logic is shared wasm).

## Status

Build-order **step 3**: on top of the step-2 skeleton (argument grammar, help
text, error strings, exit codes), the **read-only command bodies** are now
implemented and proven byte-identical to the Rust CLI — `list [lang] [--all]`
(fetch Remote Settings → wasm `catalog`/`modelPairs` → native formatter),
`models list`, and `models info <pair>` (native `fs` cache reader over the same
on-disk layout and default root the Rust CLI uses). `npm run check:parity` runs
all three groups against the Rust oracle: hermetic grammar/help/error, hermetic
`models list`/`info` against a built fixture cache, and live `list` against
Remote Settings. The cache-*writing* / engine paths (`translate`, `models add`,
`models rm`) remain stubbed (note to stderr, exit 1) pending step 4.

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
    io.js                the host I/O + terminal contract (mirrors Rust `Io`)
    fetch.js             the Node `Fetch` (global fetch) + Remote Settings records URL
    cache.js             the fs cache reader (root/listCached/pairFiles); mirrors cache.rs
    format.js            list/models view formatters, byte-for-byte from cli.rs
    lang.js              tag → display-name table, ported from lang.rs
  scripts/build-wasm.sh  builds the wasm core and copies pkg/ → wasm/
  test/parity.js         interface-parity check vs the Rust oracle (grammar + cache + live list)
  tsconfig.json          non-publishing `tsc --noEmit` config (JSDoc type check)
  wasm/                  copied-in wasm core artifacts (gitignored; run build:wasm)
```

## The wasm core (copied, not referenced)

The shared inference + discovery/routing/verify/segment core lives in the sibling
Rust crate `crates/fxtranslate-wasm`. `npm run build:wasm` runs
`wasm-pack build --target nodejs` there and **copies** the generated `pkg/`
artifacts into `npm/wasm/`. Copying (rather than referencing the sibling) keeps
the published tarball self-contained — `npm pack` bundles `wasm/`, and a consumer
`npm install fxtranslate` gets the engine without the Rust workspace. `wasm/` is a
build artifact and is gitignored.

## Types without a compile step

The `.js` is authored with JSDoc and **ships as written** — there is no TypeScript
transpile step in the publish path. Two things give consumers and tooling types:

- **Local checks:** `npm run typecheck` runs `tsc --noEmit` over the JSDoc + the
  hand-written `.d.ts`. It validates types (and catches `.d.ts` drift against the
  wasm-generated engine types) without emitting anything.
- **Packaged types:** `lib/index.d.ts` is the `types` entry. It re-exports the
  wasm-pack-generated `wasm/fxtranslate_wasm.d.ts` for the engine surface (so the
  engine types can't drift from the build) and hand-describes the small JS shell
  surface (`parse`, `run`, `processIo`, `Command`, `Io`, `CliError`).

## Scripts

- `npm run build:wasm` — build + copy the wasm core into `wasm/`.
- `npm run typecheck` — `tsc --noEmit` over the JSDoc (no transpile).
- `npm run check:parity` / `npm test` — run the Rust oracle and this CLI over
  every no-I/O case and assert byte-identical stdout, stderr, and exit code.
  Requires the oracle: `cargo build -p fxtranslate-cli`.
