#!/usr/bin/env node
// The `fxtranslate` CLI entry point — the JS mirror of `fxtranslate-cli`'s
// `main.rs`. A thin shim: it lifts argv (dropping node + script), builds the
// real terminal `Io`, and hands off to `run`. All grammar/help/error logic lives
// in the library so it stays testable without a process or terminal.

"use strict";

const { run } = require("../lib/run");
const { processIo } = require("../lib/io");

const args = process.argv.slice(2);
const io = processIo();

// Later steps (3–4) inject a real Fetch/Translator here; grammar + help + errors
// need neither, so `deps` is empty for now.
const code = run(args, io);
process.exit(code);
