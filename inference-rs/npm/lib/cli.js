// The argument grammar, help routing, and error strings — a faithful port of the
// Rust CLI's pure `parse`/`parse_models` (crates/fxtranslate-cli/src/cli.rs). No
// I/O here: argv in, a `Command` object (or a thrown `CliError`) out, so the whole
// grammar is unit-testable and pinned byte-for-byte against the Rust oracle.

const { USAGE, MODELS_USAGE, LIST_USAGE } = require("./usage");

/**
 * A parse error carrying the exact message the Rust CLI would print after the
 * `fxtranslate: ` prefix. `parse` throws this; the shell renders it and exits 1,
 * matching `run`'s `Err(e) => writeln!(stderr, "fxtranslate: {e}")`.
 */
class CliError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = "CliError";
  }
}

/**
 * A parsed command line. A discriminated union on `kind`, mirroring the Rust
 * `Command` enum one-to-one.
 *
 * @typedef {(
 *   | { kind: "help" }
 *   | { kind: "listHelp" }
 *   | { kind: "list", query: string | undefined, all: boolean }
 *   | { kind: "translate", src: string, trg: string, text: string, cacheDir: string | undefined }
 *   | { kind: "modelsHelp" }
 *   | { kind: "modelsList", cacheDir: string | undefined }
 *   | { kind: "modelsAdd", src: string, trg: string, cacheDir: string | undefined }
 *   | { kind: "modelsRemove", name: string | undefined, all: boolean, cacheDir: string | undefined }
 *   | { kind: "modelsInfo", name: string, cacheDir: string | undefined }
 * )} Command
 */

/**
 * Parse argv (without the program name) into a {@link Command}. Pure — no I/O.
 * Throws {@link CliError} with the Rust-identical message on a grammar error.
 *
 * @param {string[]} args
 * @returns {Command}
 */
function parse(args) {
  /** @type {string | undefined} */
  let cacheDir = undefined;
  /** @type {string[]} */
  const positional = [];
  let help = false;
  let all = false;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "-h" || a === "--help") {
      help = true;
    } else if (a === "--all") {
      all = true;
    } else if (a === "--cache-dir") {
      const next = args[i + 1];
      if (next === undefined) {
        throw new CliError("--cache-dir needs a path");
      }
      cacheDir = next;
      i++;
    } else {
      positional.push(a);
    }
  }

  const first = positional[0];
  switch (first) {
    case "list":
      if (help) {
        return { kind: "listHelp" };
      }
      return { kind: "list", query: positional[1], all };
    case "models":
      return parseModels(positional, cacheDir, all, help);
    default:
      break;
  }

  // `--help` with a non-list (or no) command → top-level help.
  if (help) {
    return { kind: "help" };
  }
  if (first === undefined) {
    return { kind: "help" };
  }
  if (first === "translate") {
    // `translate` is explicit: `translate <src> <trg> [text…]`.
    if (positional.length < 3) {
      throw new CliError(
        `\`translate\` needs \`<src> <trg> [text…]\`; got \`${positional.join(" ")}\`\n\n${USAGE}`,
      );
    }
    return {
      kind: "translate",
      src: positional[1],
      trg: positional[2],
      text: positional.slice(3).join(" "),
      cacheDir,
    };
  }
  throw new CliError(
    `unknown command \`${first}\`; expected \`translate\`, \`list\`, or \`models\`\n\n${USAGE}`,
  );
}

/**
 * Parse a `models <cmd> …` invocation. `cacheDir`/`all` were already lifted out by
 * {@link parse}. Mirrors the Rust `parse_models`.
 *
 * @param {string[]} positional
 * @param {string | undefined} cacheDir
 * @param {boolean} all
 * @param {boolean} help
 * @returns {Command}
 */
function parseModels(positional, cacheDir, all, help) {
  if (help) {
    return { kind: "modelsHelp" };
  }
  const sub = positional[1];
  switch (sub) {
    case undefined:
      return { kind: "modelsHelp" };
    case "list":
      return { kind: "modelsList", cacheDir };
    case "add": {
      if (positional.length < 4) {
        throw new CliError(
          `\`models add\` needs \`<src> <trg>\`; got \`${positional.slice(1).join(" ")}\`\n\n${MODELS_USAGE}`,
        );
      }
      return { kind: "modelsAdd", src: positional[2], trg: positional[3], cacheDir };
    }
    case "rm": {
      if (all) {
        return { kind: "modelsRemove", name: undefined, all: true, cacheDir };
      }
      const name = pairName(positional.slice(2));
      if (name === "") {
        throw new CliError(
          `\`models rm\` needs a \`<pair>\` (e.g. \`en-es\` or \`en es\`) or \`--all\`\n\n${MODELS_USAGE}`,
        );
      }
      return { kind: "modelsRemove", name, all: false, cacheDir };
    }
    case "info": {
      const name = pairName(positional.slice(2));
      if (name === "") {
        throw new CliError(
          `\`models info\` needs a \`<pair>\` (e.g. \`en-es\` or \`en es\`)\n\n${MODELS_USAGE}`,
        );
      }
      return { kind: "modelsInfo", name, cacheDir };
    }
    default:
      throw new CliError(
        `unknown \`models\` subcommand \`${sub}\`; expected \`list\`, \`add\`, \`rm\`, or \`info\`\n\n${MODELS_USAGE}`,
      );
  }
}

/**
 * Build the `<src>-<trg>` cache-directory name from a pair argument. Both the
 * two-tag form (`["en", "es"]`) and the joined form (`["en-es"]`) collapse to
 * `en-es`. Mirrors the Rust `pair_name` (`tokens.join("-")`).
 *
 * @param {string[]} tokens
 * @returns {string}
 */
function pairName(tokens) {
  return tokens.join("-");
}

module.exports = { parse, parseModels, pairName, CliError, USAGE, MODELS_USAGE, LIST_USAGE };
