# Multi-language fxtranslate CLIs: one shared core, native shells, one validator

## Goal

Ship an npm package **`fxtranslate`** (unscoped, single package = library + `bin`
CLI) whose command-line interface is the *same* as the Rust `fxtranslate-cli`
(binary name `fxtranslate`) — same subcommands, flags, help text, output format,
stdin/REPL/pipe behavior, error strings, and exit codes. Do it in a way that
generalizes to a **third binding (Python) later** without tripling the work, and
back it with **one centralized validator** so the CLIs can't silently drift.

The `-cli` vs library split on the Rust side is a Cargo convention (a `bin` crate
depending on a `lib` crate). npm has no reason to copy it: one `fxtranslate`
package exposes both `main`/`module` (library) and a `bin` (CLI). Same intent for
a future Python package.

## The shape

Three layers, and the whole design is about drawing the line between them in the
right place:

```
                     ┌─────────────────────────────────────────┐
  per-language        │  Rust bin   │   npm (JS)   │  Python     │   ← the "shells"
  SHELLS (native,     │ fxtranslate │  fxtranslate │ (out of     │     duplication OK
  duplication OK)     │   -cli      │   bin/lib    │  scope)     │
                     └──────┬───────┴───────┬──────┴──────┬──────┘
  arg parse · help · output formatting · REPL/pipe · TTY/color ·
  HTTP (Fetch impl) · cache storage (fs) · orchestration glue
                            │               │             │
                     ┌──────┴───────────────┴─────────────┴──────┐
  SHARED CORE         │  one Rust crate, exposed three ways:      │   ← write once
  (write once,        │   native rlib · wasm-bindgen · pyo3       │
  compile many)       ├───────────────────────────────────────────┤
                     │ inference engine (Engine::from_bytes)      │
                     │ resolve_route / catalog (route.rs, pure)   │
                     │ parse_records (remote.rs, tinyjson, pure)  │
                     │ segmentation (segment.rs, pure)            │
                     │ verify+decompress (sha256 + zstd, pure)    │
                     └───────────────────────────────────────────┘
```

**Principle (the boundary rule).** The *interfacey bits* — argument grammar, help
text, output formatting, the REPL, and all host I/O (HTTP, filesystem, TTY) — are
written **natively in each language** and are allowed to be duplicated. Only the
*portable decision logic* — the stuff that is easy to get subtly wrong and must be
bit-identical everywhere — is compiled once in Rust and shared. That is the split
the codebase already supports (see next section); we are not inventing new seams so
much as exposing the ones that exist.

Why the shell is native and not compiled-to-wasm: the Rust `Fetch` trait is
**synchronous** (`fetch.rs:105`, ureq-backed), but browser/Node `fetch` is async.
Pushing orchestration into wasm would force blocking-on-async. Keeping
orchestration in the JS/Python shell means the shell does async fetch + fs itself
and only ever calls **pure, synchronous** core functions (`resolve_route`,
`parse_records`, `Translator.translate`). The impedance mismatch disappears.

## What is already shareable vs per-language (grounded in the code)

| Piece | Where | Shareable? | How each language uses it |
|---|---|---|---|
| Inference engine | `engine.rs` `Engine::from_bytes` | **yes** | already wasm (`fxtranslate-wasm`); pyo3 later |
| Route / pivot resolution | `route.rs:38` `resolve_route(records, src, trg)` | **yes — pure** | call it; don't re-derive pivots |
| Language catalog (for `list`) | `route.rs:89` `catalog(records, hub)` | **yes — pure** | call it; render natively |
| Remote Settings parse | `remote.rs:58` `parse_records(json)` (tinyjson) | **yes — pure** | call it; fetch the JSON natively |
| Sentence segmentation | `segment.rs` Basic + ICU | **yes — pure, wasm-safe** | call it (drives `translate_long`) |
| Model verify + decompress | `cache.rs` sha256 + zstd (ruzstd) | **yes — pure** | expose as a core fn; call after download |
| `Fetch` (HTTP) | `fetch.rs:105` trait, `&dyn Fetch` | **interface only** | each language implements over native HTTP |
| Cache **storage** | `cache.rs` — hard-wired `std::fs` | **no — reimplement** | JS `fs`/OPFS, Python `pathlib`, Rust `std::fs` |
| Cache **root** default | `cache.rs:130` `dirs::cache_dir()` | **convention** | must agree cross-language (see risks) |
| CLI host contract | `cli.rs:459` `Io { stdin, stdout, stderr, *_is_tty, no_color }` | **pattern** | each shell provides these |
| Arg parse / help / format | `cli.rs` (hand-rolled parser, pure) | **per-language** | native, duplicated on purpose |

The important finding: **cache storage is not trait-abstracted** — it is direct
`std::fs` (`read_dir`/`write`/`rename`/`remove_dir_all` in `cache.rs`). That is
fine under this design: filesystem plumbing is a shell concern and each language
does it idiomatically. But the *decision* parts of caching (on-disk layout naming,
sha256 verification, zstd decompress) are pure — those we extract into a shared
core function so JS/Python don't reimplement crypto/compression, only the raw
read/write.

## The npm `fxtranslate` package

- **Plain JS authored with JSDoc — no TypeScript compile step.** Ship the `.js`
  as written; provide consumer types via the wasm-pack-generated `.d.ts` for the
  engine plus a hand-maintained `.d.ts` for the CLI/library surface. No `tsc` in
  the publish path.
- **One package, two entry points.** `package.json` has `bin: { "fxtranslate":
  "bin/fxtranslate.js" }` and library `exports` (`main`/`module`) for
  `import { Translator, resolveRoute, ... } from "fxtranslate"`.
- **Node = batteries-included** (matches `fxtranslate-cli`): the JS shell owns the
  `Fetch` impl (`node` `fetch`), the cache (`fs` under the same platform dir the
  Rust CLI uses), the argparser/formatter, and the REPL. It calls the wasm core for
  `parse_records` / `resolve_route` / `catalog` / verify / `Translator`.
- **Browser = library only.** No `bin`, no cache, no `models` subcommand — the
  library takes model `Uint8Array`s the host supplies (the step 1–10 story). The
  CLI is inherently a Node artifact.
- **Models are not bundled** (library-first). The Node CLI downloads + caches them
  exactly like the Rust CLI.
- **Supersedes** the `@gregtatum/fxtranslate-wasm` placeholder name from
  `notes/12` step 10; the wasm build becomes the core inside `fxtranslate`.

The wasm core here is a slightly larger build than the lean `notes/12` one: it must
include the pure discovery/routing/verify code (tinyjson, ruzstd, sha2 — all
wasm-safe) while still excluding native HTTP (`ureq`/`net`) and `mmap`. Getting
those pure paths into the wasm build without transitively pulling `ureq` is a
feature-gating task (see risks).

## What "exact same interface" means — two tiers

A subtlety that must be explicit: the Rust CLI runs native (system libm) and the
npm CLI runs wasm (portable libm), so their **translated text** differs on the
handful of ≤1-ULP lines already documented in `notes/12` (see
`[[wasm-parity-stance]]`). So parity is checked in two tiers:

1. **Interface conformance — byte-exact.** Everything that is *not* model output:
   `--help`/subcommand help text, `list` tables, `models list/info/add/rm`
   output, error strings (`fxtranslate: {e}`), exit codes (0 / 1), argument
   grammar and validation, the `[fxtranslate] …` stderr status lines, and
   TTY/`NO_COLOR` behavior. These MUST be identical across every CLI. The Rust CLI
   is the reference (oracle).
2. **Translation conformance — tolerant.** The translated text, compared to a
   golden at the exact-match rate the wasm build already achieves (99.69% vs
   native, with the four known libm-divergent lines), per `[[wasm-parity-stance]]`.
   The int8 GEMM stays bit-identical; only transcendental last-bits may differ.

## Centralized validation (one Python harness)

One validator, orchestrated in **Python** (extends the existing `scripts/*.py`
family: `parity.py` already shells out via `subprocess`, captures
stdout/stderr/exit, and diffs line-by-line — that is the template). It has two
passes matching the two tiers, and it treats every CLI the same way: as a binary
you run with argv + stdin and observe.

**Pass A — interface conformance (hermetic, byte-exact).** A shared, checked-in
set of **scripted CLI cases**: each case is `(argv, stdin, injected-fetch fixture,
cache fixture, NO_COLOR, tty flags) → expected stdout + stderr + exit`. Run every
case against every CLI binary and assert byte-identical (translation text is
excluded from these cases, or uses a fixed mock engine). Hermetic = no network, no
real download: each shell must accept an **injected Fetch + fixture cache dir + a
fixed environment** so `list`/`models`/errors/help are deterministic. The Rust side
already has this in-process (`tests/common/mod.rs` `MockFetch` + `run_transcript()`
capturing interleaved stdout/stderr) — the Python harness generalizes it across
*binaries*. Golden transcripts are generated from the Rust CLI and committed once;
npm (and later Python) must reproduce them.

**Pass B — translation parity (corpus).** Feed a corpus (`corpora/nllb-en-fr.txt`
and the block variants) through each CLI's `translate` and compare to a golden with
the documented tolerance. This reuses the `parity.py` diff/reporting shape (and its
non-tautological pivot handling: each engine feeds its *own* intermediate into leg
two).

So there is exactly one source of truth for "how a fxtranslate CLI must behave":
the case set + goldens the Python harness runs. Adding a language = make its CLI
pass the same harness.

## Maintenance burden across N bindings

The question the shape has to answer: with three native shells, what stops them
drifting, and what does adding the third cost?

- **The dangerous-to-duplicate logic is not duplicated.** Pivot routing, catalog
  classification, Remote Settings record parsing, checksum/zstd verification,
  tokenization/segmentation, and inference are the parts where a subtle
  reimplementation bug would silently corrupt output. All of those are in the
  shared core, written once in Rust, compiled to native/wasm/pyo3. A routing or
  parsing fix lands in one place for all three.
- **The safe-to-duplicate logic is duplicated on purpose.** Arg grammar, help
  text, output formatting, and I/O plumbing are shallow, read best idiomatic per
  language, and — critically — are pinned by Pass A. If the npm formatter drifts a
  space, a golden case fails. Duplication without a conformance net would be a
  liability; with it, it's cheap.
- **Cost of adding Python (later, out of scope now).** Implement four thin things:
  a `Fetch` over `requests`/`urllib`, cache storage over `pathlib`, an `argparse`
  shell + formatter, and pyo3 bindings that re-export the same core functions
  already exposed to wasm. Then make it pass the existing Python harness. No new
  routing/parsing/verification/inference code. The shared-core surface
  (`Translator`, `resolve_route`, `catalog`, `parse_records`, `verify`,
  `segment`) is defined once and bound three times.
- **The lever to watch:** keep the shared-core surface *complete enough* that shells
  stay thin. Every time a shell reaches for logic that "feels core," prefer adding a
  pure core function over reimplementing it. The bright-line test: does it touch I/O
  or terminal presentation? If no, it belongs in the core.

## Scope decisions (narrow, on purpose)

- **Deliverable is the npm `fxtranslate` CLI + the Python conformance harness.**
- **Python CLI is out of scope** (explicit FYI from Greg). The shape must
  *accommodate* it — hence the core surface is defined language-neutrally and the
  validator drives opaque binaries — but we do not build it here.
- **Browser is library-only**; the CLI targets Node.
- **The shared core stays a clean Rust library**; the wasm/pyo3/bin layers depend
  on it and never leak back into it (same discipline as `fxtranslate-oracle` /
  `fxtranslate-wasm`).
- **Cache interoperability:** the Node CLI should read/write the *same* on-disk
  cache the Rust CLI uses (same root, same layout) so a user with both installed
  shares one model cache. That makes the cache root default part of the shared
  contract, not a per-shell whim.

## Build order

1. **Expose the pure core to wasm.** Add `#[wasm_bindgen]` wrappers for
   `parse_records`, `resolve_route`, `catalog`, segmentation, and a
   `verify_and_decompress` (sha256 + zstd) over `&[u8]`. Fix feature-gating so
   these compile into the wasm build without pulling `ureq`/`net`/`mmap`. Unit-test
   that wasm `resolve_route`/`catalog` match native on the same record set.
2. **npm package skeleton.** One `fxtranslate` package: `bin/fxtranslate.js`,
   library `exports`, JSDoc, hand-written `.d.ts` for the CLI/lib surface, the wasm
   core bundled. `bin` wired to a JS argparser that mirrors the Rust grammar.
3. **JS shell — read-only paths first.** Implement the JS `Fetch` (node `fetch`)
   and `list` end to end (fetch records → wasm `parse_records` → wasm `catalog` →
   native formatter). Then `models list/info` against a JS cache reader.
4. **JS shell — translate + cache-writing paths.** `translate` (args / stdin-pipe /
   REPL), `models add/rm`, download → wasm-`verify` → JS atomic write, pivot
   orchestration driving wasm `resolve_route`. Match `--cache-dir`, `NO_COLOR`, TTY.
5. **Python conformance harness — Pass A.** Port/adopt the Rust `run_transcript`
   case format; define the scripted case set + injected-fetch fixtures; generate
   goldens from the Rust CLI; run the npm CLI against them; assert byte-exact.
6. **Python conformance harness — Pass B.** Corpus translation parity across CLIs
   with the documented tolerance; reuse `parity.py` diff/report + pivot handling.
7. **Wire into `task`** (`rs:conformance` or similar) and CI, so every CLI is
   validated on each change. Record which runtime produced each result.
8. **npm packaging + dry-run** (as `notes/12` step 10, now for the combined
   lib+CLI `fxtranslate`), models excluded. Real publish awaits Greg's go-ahead.

Steps 1–4 produce a working npm CLI; 5–7 make its parity provable and continuous;
8 ships it.

## Open questions / risks

- **Feature-gating the pure discovery code into wasm.** `resolve_route`/`catalog`
  are pure, but `Record`/`parse_records` and verify may currently sit behind the
  `download`/`net` feature next to `ureq`. Splitting "portable discovery" from
  "native HTTP" so the former reaches wasm without the latter is the main core-side
  task. Verify the gate layout before step 1.
- **Cache root defaults must agree across languages.** Rust uses `dirs::cache_dir()`
  → `…/fxtranslate/models`. The JS shell must compute the *same* path per platform
  (no `dirs` crate in Node) or the caches diverge. Pin these in the shared contract
  and test them.
- **Async Fetch vs sync core — confirmed fine, but keep it fine.** As long as
  orchestration stays in the shell, wasm is only called synchronously. Don't add a
  core API that wants to call back into async host fetch.
- **Types without a compile step.** Hand-maintained `.d.ts` can drift from the
  JSDoc/JS. Decide whether a *non-publishing* `tsc --noEmit` check in CI (not a
  build step) is acceptable to catch drift — it validates types without transpiling
  shipped code.

(Author note: Here to clarify, I want local typescript checks to work to take advantange of tooling having typing checks, but to only use JSDoc so we have no compile step. Let's figure out a maintainable solve for the published types, that get packaged.)

- **Progress rendering + REPL nuances** (`\r` progress on a TTY, prompt on stderr,
  EOF handling) are easy to get almost-right; Pass A must include TTY-on cases, not
  just piped ones.
- **Help-text single-sourcing.** Help strings are duplicated per shell and pinned
  by goldens. If drift becomes annoying, consider a shared declarative spec the
  shells render — deferred; the conformance net is enough to start.
- **ICU segmenter size in wasm.** `icu-segmenter` embeds data; confirm the wasm
  size cost is acceptable versus the Basic segmenter, and which the npm build ships.

## Status

**Build-order steps 1–8 implemented.** The pure core is exposed to wasm
(`parse_records`/`resolve_route`/`catalog`/`verify_and_decompress`/`segment`,
step 1); the combined lib+CLI npm package lives at `npm/` (`bin/fxtranslate.js`,
`lib/*.js` + `lib/index.d.ts`, wasm core copied into `npm/wasm/` by
`npm run build:wasm`, steps 2–4); and the Python conformance harness
(`scripts/conformance.py`) proves the npm CLI's interface is **byte-exact** with
the Rust reference and translation parity holds at the documented tolerance
(steps 5–7). `npm run typecheck` (`tsc --noEmit`, JSDoc-only, no compile step) is
clean.

**Step 8 — packaging dry-run (validated; NO real publish).** `package.json`
finalized for publishing: `name` `fxtranslate` (unscoped; a `npm view fxtranslate`
returns 404, i.e. the name appears **available/unclaimed** on the registry),
`version` `0.4.0`, `license` `MPL-2.0`, `bin` `{ "fxtranslate":
"bin/fxtranslate.js" }`, `main`/`module` `lib/index.js`, `types`
`lib/index.d.ts`, library `exports` (`.` → types/import/require), `engines`
`node >=18` (global `fetch`), and a `files` allowlist `[bin/, lib/, wasm/,
README.md, LICENSE]`. A `LICENSE` (MPL-2.0) was added to `npm/`; `npm/.gitignore`
now also ignores `*.tgz` (alongside `wasm/`, `node_modules/`).

`npm pack --dry-run` manifest — **20 files, 182.3 kB packed / 467.0 kB
unpacked**:

```
📦  fxtranslate@0.4.0
 16.7 kB  LICENSE
  4.4 kB  README.md
  1.2 kB  bin/fxtranslate.js
  ~66 kB  lib/*.js (cache, cli, fetch, format, index, io, lang, run, translate, usage)
  3.2 kB  lib/index.d.ts
  1.0 kB  package.json
 11.9 kB  wasm/fxtranslate_wasm_bg.js
337.5 kB  wasm/fxtranslate_wasm_bg.wasm
  6.6 kB  wasm/fxtranslate_wasm.d.ts + 1.8 kB *_bg.wasm.d.ts
 20.2 kB  wasm/fxtranslate_wasm.js
package size (packed): 182.3 kB   unpacked: 467.0 kB   total files: 20
```

Included: `bin/`, `lib/` (`.js` + `.d.ts`), `wasm/` (core + `.d.ts`), `README.md`,
`LICENSE`, `package.json`. Excluded: `test/`, `node_modules/`, `tsconfig.json`,
`package-lock.json`, `scripts/`. **No model files** (`.bin`/`.spm`/`.lex`/…) — the
package is library-first and downloads/caches models at runtime like the Rust CLI.
`npm publish --dry-run` reported the same manifest and `+ fxtranslate@0.4.0`
(dry-run; not logged in, nothing published). A packed `.tgz` installed into a
throwaway temp dir resolved the `fxtranslate` bin and the wasm core from the
installed layout — `fxtranslate --help` and `list --help` ran correctly.

**The name, version, and the actual `npm publish` await Greg's go-ahead.** This is
a dry-run only.

Related: `[[multi-lang-binding-strategy]]`, `[[wasm-parity-stance]]`,
`[[fxtranslate-product-direction]]`.
