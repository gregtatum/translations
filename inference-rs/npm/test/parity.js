#!/usr/bin/env node
// Interface-parity check (Pass A, in miniature): run BOTH the Rust CLI (the
// oracle) and this JS CLI over each case and assert byte-identical stdout,
// stderr, and exit code. This pins the argument grammar, help text, error/exit
// behavior, and — as of step 3 — the read-only command bodies (`list`, `models
// list`, `models info`) to the Rust reference (crates/fxtranslate-cli). Rerun
// with `npm run check:parity`.
//
// Three groups:
//   * grammar/help/error — pure, no I/O (always run).
//   * cache-reading (`models list`/`info`) — HERMETIC: run against an on-disk
//     fixture cache this script builds, with known-size files, so both binaries
//     read the same bytes with no network. Covers the empty-cache and
//     unknown-pair edges too.
//   * `list` — needs LIVE Remote Settings (no injected-fetch flag on the Rust
//     binary yet; that arrives in step 5). Run against the network and diffed
//     byte-for-byte; skipped (not failed) when the network is unavailable.
//
// The still-stubbed cache-WRITING paths (`translate`, `models add`, `models rm`)
// are excluded — they land in step 4.

"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");

const repoRoot = path.resolve(__dirname, "..", "..");
const rustBin = path.join(repoRoot, "target", "debug", "fxtranslate");
const jsBin = path.join(__dirname, "..", "bin", "fxtranslate.js");

/**
 * Argv cases covering every no-I/O grammar/help/error path in the Rust CLI.
 * @type {string[][]}
 */
const grammarCases = [
  ["--help"],
  ["-h"],
  [],
  ["list", "--help"],
  ["models", "--help"],
  ["translate"],
  ["translate", "en"],
  ["models"],
  ["models", "bogus"],
  ["models", "add", "en"],
  ["models", "rm"],
  ["models", "info"],
  ["bogus"],
  // Extra routing cases from the Rust `--help` rules:
  ["translate", "--help"], // non-list command + --help → top-level help
  ["models", "list", "--help"], // --help short-circuits to models help
  ["--cache-dir"], // missing value → "--cache-dir needs a path"
  ["list", "-h"], // -h alias for list help
];

/** `list` cases — need LIVE Remote Settings (see the header note). @type {string[][]} */
const liveListCases = [["list"], ["list", "es"], ["list", "--all"], ["list", "es", "--all"]];

/**
 * @param {string} bin
 * @param {string[]} argv
 * @returns {{ stdout: Buffer, stderr: Buffer, status: number }}
 */
function runBin(bin, argv) {
  const isJs = bin.endsWith(".js");
  const cmd = isJs ? process.execPath : bin;
  const args = isJs ? [bin, ...argv] : argv;
  const res = spawnSync(cmd, args, {
    // Deterministic, non-TTY, uncolored environment for both — matches the
    // hermetic Pass A contract (help/errors are never colored).
    env: { ...process.env, NO_COLOR: "1" },
    encoding: "buffer",
    maxBuffer: 16 * 1024 * 1024,
  });
  return { stdout: res.stdout, stderr: res.stderr, status: res.status ?? -1 };
}

/**
 * Run one argv case against both binaries and report byte-equality. Returns
 * `true` when stdout, stderr, and exit code all match.
 *
 * @param {string[]} argv
 * @returns {boolean}
 */
function compareCase(argv) {
  const label = `fxtranslate ${argv.join(" ")}`.trim();
  const rust = runBin(rustBin, argv);
  const js = runBin(jsBin, argv);

  /** @type {string[]} */
  const diffs = [];
  if (rust.status !== js.status) {
    diffs.push(`exit: rust=${rust.status} js=${js.status}`);
  }
  if (!rust.stdout.equals(js.stdout)) {
    diffs.push(`stdout differs (rust=${rust.stdout.length}B js=${js.stdout.length}B)`);
  }
  if (!rust.stderr.equals(js.stderr)) {
    diffs.push(`stderr differs (rust=${rust.stderr.length}B js=${js.stderr.length}B)`);
  }

  if (diffs.length === 0) {
    console.log(`  ok   ${label}`);
    return true;
  }
  console.log(`  FAIL ${label}`);
  for (const d of diffs) console.log(`         ${d}`);
  if (!rust.stdout.equals(js.stdout)) {
    console.log(`       --- rust stdout ---\n${rust.stdout.toString()}`);
    console.log(`       --- js stdout ---\n${js.stdout.toString()}`);
  }
  if (!rust.stderr.equals(js.stderr)) {
    console.log(`       --- rust stderr ---\n${rust.stderr.toString()}`);
    console.log(`       --- js stderr ---\n${js.stderr.toString()}`);
  }
  return false;
}

/**
 * Build a throwaway fixture cache with a couple of `<src>-<trg>` pair dirs and
 * dummy files of known sizes — the hermetic input both `models list`/`info`
 * commands read. The sizes span the raw-bytes, KiB, and MiB paths of
 * `human_bytes`, and one dot-prefixed temp file proves it is skipped. Returns the
 * root path (passed to both binaries via `--cache-dir`).
 *
 * @returns {string}
 */
function buildFixtureCache() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "fxtranslate-parity-cache-"));
  /**
   * @param {string} pair
   * @param {string} name
   * @param {number} bytes
   */
  const write = (pair, name, bytes) => {
    const dir = path.join(root, pair);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, name), Buffer.alloc(bytes, 0x61));
  };
  write("en-es", "model.enes.bin", 31); // raw bytes
  write("en-es", "vocab.enes.spm", 31);
  write("en-fr", "model.enfr.bin", 5120); // 5.0 KiB
  write("es-en", "model.esen.bin", 1234567); // 1.2 MiB (decimal path)
  write("es-en", "vocab.esen.spm", 2048);
  fs.writeFileSync(path.join(root, "en-fr", ".model.enfr.bin.download"), Buffer.alloc(9999)); // must be skipped
  return root;
}

/** Whether the network can reach Remote Settings — a plain Rust `list` that exits 0. */
function networkAvailable() {
  const probe = runBin(rustBin, ["list"]);
  return probe.status === 0;
}

function main() {
  if (!fs.existsSync(rustBin)) {
    console.error(
      `oracle binary missing: ${rustBin}\n` + `build it with: cargo build -p fxtranslate-cli`,
    );
    process.exit(2);
  }

  let pass = 0;
  let fail = 0;
  /** @param {string[]} argv */
  const run = (argv) => {
    if (compareCase(argv)) pass++;
    else fail++;
  };

  console.log("grammar / help / error (hermetic):");
  for (const argv of grammarCases) run(argv);

  console.log("\nmodels list / info (hermetic, fixture cache):");
  const cache = buildFixtureCache();
  const emptyCache = fs.mkdtempSync(path.join(os.tmpdir(), "fxtranslate-parity-empty-"));
  try {
    run(["models", "list", "--cache-dir", cache]);
    run(["models", "list", "--cache-dir", emptyCache]); // empty-cache note
    run(["models", "info", "--cache-dir", cache, "en-es"]);
    run(["models", "info", "--cache-dir", cache, "es-en"]);
    run(["models", "info", "--cache-dir", cache, "en-fr"]);
    run(["models", "info", "--cache-dir", cache, "en", "es"]); // two-tag form
    run(["models", "info", "--cache-dir", cache, "zz-zz"]); // unknown pair note
  } finally {
    fs.rmSync(cache, { recursive: true, force: true });
    fs.rmSync(emptyCache, { recursive: true, force: true });
  }

  console.log("\nlist (LIVE Remote Settings — the Rust binary has no injected-fetch flag yet):");
  if (networkAvailable()) {
    for (const argv of liveListCases) run(argv);
  } else {
    console.log("  skip live `list` parity — network unavailable (fully hermetic in step 5).");
  }

  console.log(`\n${pass}/${pass + fail} cases byte-identical.`);
  process.exit(fail === 0 ? 0 : 1);
}

main();
