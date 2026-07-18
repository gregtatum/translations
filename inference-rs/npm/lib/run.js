// Parse argv, dispatch to the matching command, and execute it against `io`/`deps`
// — the JS mirror of the Rust `run`/`dispatch` (crates/fxtranslate-cli/src/cli.rs).
//
// All commands are implemented: the read-only paths (`list`, `models list/info`)
// and — as of step 4 — the cache-writing / engine paths (`translate`, `models
// add`, `models rm`). The shell owns the async fetch + fs; the pure decisions
// (routing, catalog, decode+verify, inference) come from the synchronous wasm core.

const wasm = require("../wasm/fxtranslate_wasm.js");
const { parse, CliError, USAGE, LIST_USAGE, MODELS_USAGE } = require("./cli");
const { recordsUrl, nodeFetch } = require("./fetch");
const { Cache, ensureModel } = require("./cache");
const { engineTranslator } = require("./translate");
const {
  PREFERRED_HUB,
  humanBytes,
  renderProgress,
  palette,
  scalarLen,
  padEnd,
  padStart,
  writePairs,
  writeLanguages,
  prettyPair,
} = require("./format");

/**
 * The dependencies the shell injects — a `Fetch` (for `list`, `models add`, and the
 * default translator's discovery) and a `Translator` for `translate`. Mirrors the
 * Rust `Deps` seam; both default to real Node implementations.
 *
 * @typedef {object} Deps
 * @property {import("./fetch").Fetch} [fetch]
 * @property {import("./translate").Translator} [translator]
 */

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
        return await runTranslate(io, deps, cmd.src, cmd.trg, cmd.text, cmd.cacheDir);
      case "modelsAdd":
        return await runModelsAdd(io, deps, cmd.src, cmd.trg, cmd.cacheDir);
      case "modelsRemove":
        return runModelsRemove(io, cmd.name, cmd.all, cmd.cacheDir);
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

/**
 * The cache the `models`/`translate` verbs act on: an explicit `--cache-dir` or the
 * platform default, with the download-progress line wired to stderr only when
 * stderr is a TTY (the same gate Rust's `open_cache(..., io.stderr_is_tty)` uses).
 * Mirrors Rust's `open_cache`.
 *
 * @param {string | undefined} cacheDir
 * @param {boolean} progress
 * @param {import("./io").Io} io
 * @returns {Cache}
 */
function openCache(cacheDir, progress, io) {
  const cache = new Cache(cacheDir);
  if (progress) {
    cache.withProgress((name, done, total) => io.stderr(renderProgress(name, done, total)));
  }
  return cache;
}

/**
 * `models add <src> <trg>`: pre-download every file for the pair (both legs of a
 * pivot) into the cache, without building an engine. The shell drives the sync core
 * `resolveRoute` to decide direct-vs-pivot, then does the async download + verified
 * atomic write per leg (via {@link ensureModel}). Status goes to stderr and names
 * the resolved hop. Mirrors Rust's `run_models_add`.
 *
 * @param {import("./io").Io} io
 * @param {Deps} deps
 * @param {string} src
 * @param {string} trg
 * @param {string | undefined} cacheDir
 * @returns {Promise<number>}
 */
async function runModelsAdd(io, deps, src, trg, cacheDir) {
  const fetch = deps.fetch || nodeFetch();
  const cache = openCache(cacheDir, io.stderrIsTty, io);
  io.stderr(`[fxtranslate] downloading ${src}→${trg} model…\n`);

  const body = await fetch.get(recordsUrl());
  const records = JSON.parse(wasm.parseRecords(body));
  const route = JSON.parse(wasm.resolveRoute(body, src, trg));
  if (route.kind === "pivot") {
    await ensureModel(fetch, cache, records, route.src, route.pivot);
    await ensureModel(fetch, cache, records, route.pivot, route.trg);
    io.stderr(`[fxtranslate] cached (${src}→${route.pivot}→${trg}, pivot).\n`);
  } else {
    await ensureModel(fetch, cache, records, route.src, route.trg);
    io.stderr(`[fxtranslate] cached (${src}→${trg}).\n`);
  }
  return 0;
}

/**
 * `models rm <pair>` / `--all`: delete one cached pair (or every pair), each removal
 * reporting the space it reclaimed. Idempotent — removing an absent pair is a note,
 * not an error. Byte-for-byte with Rust's `run_models_remove`.
 *
 * @param {import("./io").Io} io
 * @param {string | undefined} name
 * @param {boolean} all
 * @param {string | undefined} cacheDir
 * @returns {number}
 */
function runModelsRemove(io, name, all, cacheDir) {
  const cache = new Cache(cacheDir);

  if (all) {
    const cached = cache.listCached();
    if (cached.length === 0) {
      io.stdout("No models cached; nothing to remove.\n");
      return 0;
    }
    let freed = 0;
    for (const c of cached) {
      cache.removePair(c.name);
      io.stdout(`Removed ${c.name} (${humanBytes(c.bytes)})\n`);
      freed += c.bytes;
    }
    io.stderr(`[${cached.length} removed, ${humanBytes(freed)} reclaimed]\n`);
    return 0;
  }

  // A specific pair: look it up first so we can report the reclaimed size.
  const found = cache.listCached().find((c) => c.name === name);
  if (found) {
    cache.removePair(/** @type {string} */ (name));
    io.stdout(`Removed ${name} (${humanBytes(found.bytes)})\n`);
  } else {
    io.stdout(`${name} is not cached.\n`);
  }
  return 0;
}

/**
 * `translate <src> <trg> [text…]`: resolve+load the session, then translate either
 * the arg text (one line), piped stdin (one translation per line), or an interactive
 * TTY REPL. Status lines go to stderr so piped stdout carries only translations.
 * Byte-for-byte with Rust's `run_translate` on the interface (status/prompt/EOF);
 * the translated TEXT is tolerant (wasm libm), per notes/13.
 *
 * @param {import("./io").Io} io
 * @param {Deps} deps
 * @param {string} src
 * @param {string} trg
 * @param {string} text
 * @param {string | undefined} cacheDir
 * @returns {Promise<number>}
 */
async function runTranslate(io, deps, src, trg, text, cacheDir) {
  const translator = deps.translator || engineTranslator(deps.fetch || nodeFetch(), io.stderrIsTty);
  io.stderr(`[fxtranslate] resolving ${src}→${trg} model…\n`);
  const session = await translator.load(src, trg, cacheDir);
  const pivot = session.pivot();
  if (pivot) {
    io.stderr(`[fxtranslate] ready (${src}→${pivot}→${trg}, pivot).\n`);
  } else {
    io.stderr(`[fxtranslate] ready (${src}→${trg}).\n`);
  }

  if (text !== "") {
    io.stdout(`${session.translate(text)}\n`);
    return 0;
  }

  if (io.stdinIsTty) {
    await repl(session, io, src, trg);
    return 0;
  }

  // Pipe mode: one translation per input line (marian-style).
  for await (const line of readLines(io.stdin)) {
    io.stdout(`${session.translate(line)}\n`);
  }
  return 0;
}

/**
 * Minimal interactive REPL: a prompt on stderr, a line in, its translation out,
 * until EOF (Ctrl-D). Blank lines are skipped. Mirrors Rust's `repl` — the prompt
 * (`src→trg» ` on stderr, no newline), the intro line, and the trailing newline at
 * EOF are all byte-identical.
 *
 * @param {import("./translate").Session} session
 * @param {import("./io").Io} io
 * @param {string} src
 * @param {string} trg
 * @returns {Promise<void>}
 */
async function repl(session, io, src, trg) {
  io.stderr(`Interactive ${src}→${trg}. Type a sentence and press Enter; Ctrl-D to quit.\n`);
  io.stderr(`${src}→${trg}» `);
  for await (const line of readLines(io.stdin)) {
    const t = line.trim();
    if (t !== "") {
      io.stdout(`${session.translate(t)}\n`);
    }
    io.stderr(`${src}→${trg}» `);
  }
  io.stderr("\n"); // EOF closes the final prompt line
}

/**
 * Yield `stdin` one line at a time, stripping only the trailing `\n` (and a
 * preceding `\r`) — matching Rust's `BufRead::read_line` line splitting. A final
 * line with no trailing newline is still yielded.
 *
 * @param {NodeJS.ReadableStream} stdin
 * @returns {AsyncGenerator<string>}
 */
async function* readLines(stdin) {
  let buf = "";
  stdin.setEncoding?.("utf8");
  for await (const chunk of stdin) {
    buf += chunk;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      let line = buf.slice(0, nl);
      if (line.endsWith("\r")) {
        line = line.slice(0, -1);
      }
      yield line;
      buf = buf.slice(nl + 1);
    }
  }
  if (buf.length > 0) {
    yield buf.endsWith("\r") ? buf.slice(0, -1) : buf;
  }
}

module.exports = { run };
