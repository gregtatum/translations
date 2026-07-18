// Hand-maintained consumer types for the `fxtranslate` library surface.
//
// Types strategy (per notes/13, "types without a compile step"):
//   * The .js is authored with JSDoc and ships as-is — NO transpile step.
//   * `npm run typecheck` (tsc --noEmit) validates the JSDoc locally so tooling
//     catches drift; it never emits or transpiles shipped code.
//   * Engine types are NOT re-typed by hand — they are re-exported straight from
//     the wasm-pack-generated ../wasm/fxtranslate_wasm.d.ts, so the engine surface
//     can't drift from the wasm build. Only the small JS shell surface (parser,
//     runner, host I/O) is described here.
//
// This file is the package's `types` entry and travels in the published tarball.

export {
  Translator,
  parseRecords,
  resolveRoute,
  catalog,
  segmentSentences,
  verifyAndDecompress,
} from "../wasm/fxtranslate_wasm";

/** A parsed command line — a discriminated union on `kind` (mirrors Rust `Command`). */
export type Command =
  | { kind: "help" }
  | { kind: "listHelp" }
  | { kind: "list"; query: string | undefined; all: boolean }
  | { kind: "translate"; src: string; trg: string; text: string; cacheDir: string | undefined }
  | { kind: "modelsHelp" }
  | { kind: "modelsList"; cacheDir: string | undefined }
  | { kind: "modelsAdd"; src: string; trg: string; cacheDir: string | undefined }
  | { kind: "modelsRemove"; name: string | undefined; all: boolean; cacheDir: string | undefined }
  | { kind: "modelsInfo"; name: string; cacheDir: string | undefined };

/** Error carrying the exact message the CLI prints after `fxtranslate: `. */
export class CliError extends Error {}

/** The host I/O + terminal contract the CLI runs against (mirrors Rust `Io`). */
export interface Io {
  stdin: NodeJS.ReadStream | import("stream").Readable;
  stdout: (s: string) => void;
  stderr: (s: string) => void;
  stdinIsTty: boolean;
  stdoutIsTty: boolean;
  stderrIsTty: boolean;
  noColor: boolean;
}

/** Dependencies injected by the shell in later steps (Fetch / Translator). */
export interface Deps {
  fetch?: unknown;
  translator?: unknown;
}

/** Parse argv (without the program name) into a {@link Command}; throws {@link CliError}. */
export function parse(args: string[]): Command;

/** Parse + execute `args` against `io`/`deps`, returning the exit code (0 or 1). */
export function run(args: string[], io: Io, deps?: Deps): number;

/** Build an {@link Io} bound to the real process streams and terminal facts. */
export function processIo(): Io;
