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
  modelPairs,
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

/** Transport the shell injects: text `get` (discovery) + streamed `download`. */
export interface Fetch {
  get(url: string): Promise<string>;
  download(
    url: string,
    onProgress: (done: number, total: number | undefined) => void,
  ): Promise<Uint8Array>;
}

/** A loaded model ready to translate lines (mirrors Rust `Session`). */
export interface Session {
  translate(text: string): string;
  pivot(): string | undefined;
}

/** Resolves a `src`→`trg` pair into a ready {@link Session} (mirrors Rust `Translator`). */
export interface Translator {
  load(src: string, trg: string, cacheDir: string | undefined): Promise<Session>;
}

/** Dependencies injected by the shell: a {@link Fetch} and a {@link Translator}. */
export interface Deps {
  fetch?: Fetch;
  translator?: Translator;
}

/** Parse argv (without the program name) into a {@link Command}; throws {@link CliError}. */
export function parse(args: string[]): Command;

/** Parse + execute `args` against `io`/`deps`, returning the exit code (0 or 1). */
export function run(args: string[], io: Io, deps?: Deps): Promise<number>;

/** Build an {@link Io} bound to the real process streams and terminal facts. */
export function processIo(): Io;
