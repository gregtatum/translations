#!/usr/bin/env node
// Offline interface-parity check (Pass A, in miniature): run BOTH the Rust CLI
// (the oracle) and this JS CLI over every case that needs no network and no
// models, and assert byte-identical stdout, stderr, and exit code. This pins the
// argument grammar, help text, and error/exit-code behavior to the Rust
// reference (crates/fxtranslate-cli). Rerun with `npm run check:parity`.
//
// Cases that reach the network/cache/engine (e.g. a well-formed `list` or
// `models list`) are intentionally excluded here — their bodies are stubbed
// until steps 3–4, so they don't yet share output with the oracle. Every case
// below is pure grammar/help/error, which IS implemented and must match.

"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const repoRoot = path.resolve(__dirname, "..", "..");
const rustBin = path.join(repoRoot, "target", "debug", "fxtranslate");
const jsBin = path.join(__dirname, "..", "bin", "fxtranslate.js");

/**
 * Argv cases covering every no-I/O grammar/help/error path in the Rust CLI.
 * @type {string[][]}
 */
const cases = [
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
  });
  return { stdout: res.stdout, stderr: res.stderr, status: res.status ?? -1 };
}

function main() {
  if (!fs.existsSync(rustBin)) {
    console.error(
      `oracle binary missing: ${rustBin}\n` +
        `build it with: cargo build -p fxtranslate-cli`,
    );
    process.exit(2);
  }

  let pass = 0;
  let fail = 0;
  for (const argv of cases) {
    const label = `fxtranslate ${argv.join(" ")}`.trim();
    const rust = runBin(rustBin, argv);
    const js = runBin(jsBin, argv);

    /** @type {string[]} */
    const diffs = [];
    if (rust.status !== js.status) {
      diffs.push(`exit: rust=${rust.status} js=${js.status}`);
    }
    if (!rust.stdout.equals(js.stdout)) {
      diffs.push(
        `stdout differs (rust=${rust.stdout.length}B js=${js.stdout.length}B)`,
      );
    }
    if (!rust.stderr.equals(js.stderr)) {
      diffs.push(
        `stderr differs (rust=${rust.stderr.length}B js=${js.stderr.length}B)`,
      );
    }

    if (diffs.length === 0) {
      pass++;
      console.log(`  ok   ${label}`);
    } else {
      fail++;
      console.log(`  FAIL ${label}`);
      for (const d of diffs) console.log(`         ${d}`);
      // Show the first divergent stream inline to make failures debuggable.
      if (!rust.stdout.equals(js.stdout)) {
        console.log(`       --- rust stdout ---\n${rust.stdout.toString()}`);
        console.log(`       --- js stdout ---\n${js.stdout.toString()}`);
      }
      if (!rust.stderr.equals(js.stderr)) {
        console.log(`       --- rust stderr ---\n${rust.stderr.toString()}`);
        console.log(`       --- js stderr ---\n${js.stderr.toString()}`);
      }
    }
  }

  console.log(`\n${pass}/${pass + fail} cases byte-identical.`);
  process.exit(fail === 0 ? 0 : 1);
}

main();
