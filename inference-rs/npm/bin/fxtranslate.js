#!/usr/bin/env node
// The `fxtranslate` CLI entry point — the JS mirror of `fxtranslate-cli`'s
// `main.rs`. A thin shim: it lifts argv (dropping node + script), builds the
// real terminal `Io`, and hands off to `run`. All grammar/help/error logic lives
// in the library so it stays testable without a process or terminal.

"use strict";

const { run } = require("../lib/run");
const { processIo } = require("../lib/io");
const { nodeFetch } = require("../lib/fetch");

const args = process.argv.slice(2);
const io = processIo();

// The shell injects the real Node `Fetch`; `run` builds the real (engine-backed)
// translator from it on demand for `translate`, so both `list`/`models add` discovery
// and translation hit the same live Remote Settings endpoint. Read-only cache verbs
// ignore `deps` entirely.
const deps = { fetch: nodeFetch() };

run(args, io, deps)
  .then((code) => process.exit(code))
  .catch((err) => {
    // A defensive backstop: `run` already maps expected errors to a
    // `fxtranslate: …` line + exit 1, so reaching here means an unexpected throw.
    process.stderr.write(`fxtranslate: ${err && err.message ? err.message : err}\n`);
    process.exit(1);
  });
