// Parse argv, dispatch to the matching command, and execute it against `io`/`deps`
// — the JS mirror of the Rust `run`/`dispatch` (crates/fxtranslate-cli/src/cli.rs).
//
// Read-only paths are implemented here (step 3): `list [lang] [--all]`,
// `models list`, and `models info <pair>`, byte-for-byte against the Rust oracle.
// The cache-writing / engine paths (`translate`, `models add`, `models rm`) stay
// stubbed until step 4 — they emit a "not yet implemented" note and exit 1.

const wasm = require("../wasm/fxtranslate_wasm.js");
const { parse, CliError, USAGE, LIST_USAGE, MODELS_USAGE } = require("./cli");
const { recordsUrl, nodeFetch } = require("./fetch");
const { Cache } = require("./cache");
const {
  PREFERRED_HUB,
  humanBytes,
  palette,
  scalarLen,
  padEnd,
  padStart,
  writePairs,
  writeLanguages,
  prettyPair,
} = require("./format");

/**
 * The dependencies the shell injects — a `Fetch` for `list` (and, later, `models
 * add`) and a translator for `translate`. Mirrors the Rust `Deps` seam.
 *
 * @typedef {object} Deps
 * @property {import("./fetch").Fetch} [fetch]
 * @property {unknown} [translator]
 */

/**
 * A command whose body needs an engine or a cache write — not implemented until
 * step 4. Emits the placeholder note to stderr and signals exit 1.
 *
 * @param {import("./io").Io} io
 * @param {string} what
 * @returns {number} exit code (always 1)
 */
function notYetImplemented(io, what) {
  io.stderr(`fxtranslate: \`${what}\` is not yet implemented in the npm CLI (step 4)\n`);
  return 1;
}

/**
 * Parse and execute `args` against `io`/`deps`, returning the process exit code
 * (0 or 1). Errors are reported to stderr prefixed `fxtranslate: ` — the mirror of
 * the Rust `run`'s `Err(e) => writeln!(stderr, "fxtranslate: {e}")`.
 *
 * Async because `list` fetches Remote Settings; the sync commands resolve
 * immediately. The core it calls (`catalog`/`modelPairs`) is synchronous — only
 * the shell's HTTP is async, keeping the impedance mismatch out of the core.
 *
 * @param {string[]} args
 * @param {import("./io").Io} io
 * @param {Deps} [deps]
 * @returns {Promise<number>}
 */
async function run(args, io, deps = {}) {
  /** @type {import("./cli").Command} */
  let cmd;
  try {
    cmd = parse(args);
  } catch (e) {
    if (e instanceof CliError) {
      io.stderr(`fxtranslate: ${e.message}\n`);
      return 1;
    }
    throw e;
  }

  try {
    switch (cmd.kind) {
      case "help":
        io.stdout(`${USAGE}\n`);
        return 0;
      case "listHelp":
        io.stdout(`${LIST_USAGE}\n`);
        return 0;
      case "modelsHelp":
        io.stdout(`${MODELS_USAGE}\n`);
        return 0;
      case "list":
        return await runList(io, deps, cmd.query, cmd.all);
      case "modelsList":
        return runModelsList(io, cmd.cacheDir);
      case "modelsInfo":
        return runModelsInfo(io, cmd.name, cmd.cacheDir);
      case "translate":
        return notYetImplemented(io, "translate");
      case "modelsAdd":
        return notYetImplemented(io, "models add");
      case "modelsRemove":
        return notYetImplemented(io, "models rm");
      default: {
        /** @type {never} */
        const _never = cmd;
        void _never;
        return 1;
      }
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    io.stderr(`fxtranslate: ${message}\n`);
    return 1;
  }
}

/**
 * The `list` command: fetch the Remote Settings records, then render either the
 * language view (default) or the raw `--all` pair table via the wasm core. Mirrors
 * the Rust `dispatch`'s `Command::List` arm — color only on a TTY stdout honoring
 * `NO_COLOR`, and the `[N …]` trailer on stderr.
 *
 * @param {import("./io").Io} io
 * @param {Deps} deps
 * @param {string | undefined} query
 * @param {boolean} all
 * @returns {Promise<number>}
 */
async function runList(io, deps, query, all) {
  const fetch = deps.fetch || nodeFetch();
  const body = await fetch.get(recordsUrl());
  const color = io.stdoutIsTty && !io.noColor;

  if (all) {
    /** @type {[string, string][]} */
    const pairs = JSON.parse(wasm.modelPairs(body));
    const n = writePairs(pairs, query, color, io.stdout);
    io.stderr(`[${n} pairs]\n`);
  } else {
    const cat = JSON.parse(wasm.catalog(body, PREFERRED_HUB));
    const n = writeLanguages(cat, query, color, io.stdout);
    io.stderr(`[${n} languages]\n`);
  }
  return 0;
}

/**
 * `models list`: the cache location, a row per cached pair (`Source → Target`
 * label, `(dir-name)` id, size), then the total — or a friendly note for an empty
 * cache. Mirrors Rust's `run_models_list`.
 *
 * @param {import("./io").Io} io
 * @param {string | undefined} cacheDir
 * @returns {number}
 */
function runModelsList(io, cacheDir) {
  const cache = new Cache(cacheDir);
  io.stdout(`Cache: ${cache.root}\n`);

  const cached = cache.listCached();
  if (cached.length === 0) {
    io.stdout(
      "No models cached yet. Add one with `fxtranslate models add <src> <trg>`.\n",
    );
    io.stderr("[0 cached]\n");
    return 0;
  }

  const rows = cached.map((c) => {
    const pretty = prettyPair(c.name);
    const label = pretty ? `${pretty[0]} → ${pretty[1]}` : c.name;
    return { label, tag: `(${c.name})`, size: humanBytes(c.bytes) };
  });
  const wLabel = Math.max(0, ...rows.map((r) => scalarLen(r.label)));
  const wTag = Math.max(0, ...rows.map((r) => scalarLen(r.tag)));

  const color = io.stdoutIsTty && !io.noColor;
  const [cyan, , dim, reset] = palette(color);
  for (const r of rows) {
    const label = padEnd(r.label, wLabel);
    const tag = padEnd(r.tag, wTag);
    io.stdout(`  ${cyan}${label}${reset} ${dim}${tag}${reset} ${r.size}\n`);
  }
  const total = cached.reduce((sum, c) => sum + c.bytes, 0);
  io.stdout(`Total: ${humanBytes(total)}\n`);
  io.stderr(`[${cached.length} cached]\n`);
  return 0;
}

/**
 * `models info <pair>`: a cached pair's files — each with its size and full on-disk
 * path — plus the total, or a "not cached" note. Mirrors Rust's `run_models_info`.
 *
 * @param {import("./io").Io} io
 * @param {string} name
 * @param {string | undefined} cacheDir
 * @returns {number}
 */
function runModelsInfo(io, name, cacheDir) {
  const cache = new Cache(cacheDir);
  const files = cache.pairFiles(name);
  if (files.length === 0) {
    io.stdout(
      `${name} is not cached. Add it with \`fxtranslate models add <src> <trg>\`.\n`,
    );
    return 0;
  }

  const path = require("node:path");
  io.stdout(`${name} (${path.join(cache.root, name)})\n`);
  const wName = Math.max(0, ...files.map((f) => scalarLen(f.name)));
  const sizes = files.map((f) => humanBytes(f.bytes));
  const wSize = Math.max(0, ...sizes.map((s) => scalarLen(s)));
  for (let i = 0; i < files.length; i++) {
    const fname = padEnd(files[i].name, wName);
    const size = padStart(sizes[i], wSize);
    io.stdout(`  ${fname}  ${size}  ${files[i].path}\n`);
  }
  const total = files.reduce((sum, f) => sum + f.bytes, 0);
  io.stdout(`Total: ${humanBytes(total)}\n`);
  return 0;
}

module.exports = { run, notYetImplemented };
