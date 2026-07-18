// The host I/O + terminal contract the CLI runs against — the JS mirror of the
// Rust `Io` struct (crates/fxtranslate-cli/src/cli.rs). Injected rather than
// probed from the process so tests can capture streams and set TTY/`NO_COLOR`
// facts explicitly (Pass A drives argv → transcript against fakes).

/**
 * @typedef {object} Io
 * @property {NodeJS.ReadStream | import("stream").Readable} stdin
 * @property {(s: string) => void} stdout  Write a chunk to stdout (no newline added).
 * @property {(s: string) => void} stderr  Write a chunk to stderr (no newline added).
 * @property {boolean} stdinIsTty   stdin is a terminal → `translate` uses the REPL.
 * @property {boolean} stdoutIsTty  stdout is a terminal → `list` may color.
 * @property {boolean} stderrIsTty  stderr is a terminal → `models add` draws progress.
 * @property {boolean} noColor      `NO_COLOR` is set in the environment.
 */

/**
 * Build an {@link Io} bound to the real process streams and terminal facts —
 * the mirror of `main.rs` wiring the real terminal into `run`. `NO_COLOR` is
 * honored the same way (presence of the variable, any value).
 *
 * @returns {Io}
 */
function processIo() {
  return {
    stdin: process.stdin,
    stdout: (s) => process.stdout.write(s),
    stderr: (s) => process.stderr.write(s),
    stdinIsTty: Boolean(process.stdin.isTTY),
    stdoutIsTty: Boolean(process.stdout.isTTY),
    stderrIsTty: Boolean(process.stderr.isTTY),
    noColor: process.env.NO_COLOR !== undefined,
  };
}

module.exports = { processIo };
