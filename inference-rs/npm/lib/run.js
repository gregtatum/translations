// Parse argv, dispatch to the matching command, and execute it against `io` —
// the JS mirror of the Rust `run`/`dispatch` (crates/fxtranslate-cli/src/cli.rs).
//
// Step 2 scope: the full argument grammar, all help/usage output, and every
// error string + exit code are wired byte-for-byte against the Rust oracle. The
// network/cache/translate command *bodies* (list, translate, models
// list/add/rm/info) are stubbed — they emit a "not yet implemented" note to
// stderr and exit 1 — pending steps 3–4. Help and grammar errors are complete.

const { parse, CliError, USAGE, LIST_USAGE, MODELS_USAGE } = require("./cli");

/**
 * The `Deps` the shell will inject in later steps (a Fetch for `list`/`models
 * add`, a Translator for `translate`). Stubbed now; kept as a parameter so the
 * dispatch shape matches the Rust `Deps` seam and steps 3–4 slot in without a
 * signature change.
 *
 * @typedef {object} Deps
 * @property {unknown} [fetch]
 * @property {unknown} [translator]
 */

/**
 * A command whose body needs network, cache, or an engine — not implemented in
 * this step. Emits the placeholder note to stderr and signals exit 1.
 *
 * @param {import("./io").Io} io
 * @param {string} what
 * @returns {number} exit code (always 1)
 */
function notYetImplemented(io, what) {
  io.stderr(`fxtranslate: \`${what}\` is not yet implemented in the npm CLI (steps 3–4)\n`);
  return 1;
}

/**
 * Parse and execute `args` against `io`/`deps`, returning the process exit code
 * (0 or 1). Errors are reported to stderr prefixed `fxtranslate: ` — the mirror
 * of the Rust `run`'s `Err(e) => writeln!(stderr, "fxtranslate: {e}")`.
 *
 * @param {string[]} args
 * @param {import("./io").Io} io
 * @param {Deps} [deps]
 * @returns {number}
 */
function run(args, io, deps = {}) {
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
      return notYetImplemented(io, "list");
    case "translate":
      return notYetImplemented(io, "translate");
    case "modelsList":
      return notYetImplemented(io, "models list");
    case "modelsAdd":
      return notYetImplemented(io, "models add");
    case "modelsRemove":
      return notYetImplemented(io, "models rm");
    case "modelsInfo":
      return notYetImplemented(io, "models info");
    default: {
      // Exhaustiveness: every `Command` kind is handled above, so `cmd` is
      // `never` here. Asserting that keeps the switch honest if a kind is added.
      /** @type {never} */
      const _never = cmd;
      void _never;
      return 1;
    }
  }
}

module.exports = { run, notYetImplemented };
