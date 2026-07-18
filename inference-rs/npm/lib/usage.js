// The help/usage strings, reproduced byte-for-byte from the Rust CLI's `USAGE`,
// `MODELS_USAGE`, and `LIST_USAGE` constants (crates/fxtranslate-cli/src/cli.rs).
// Pass A conformance pins these; if the Rust text changes these must follow.

/** Top-level usage. Mirrors `cli.rs` `USAGE`. */
const USAGE = `fxtranslate — translate with Firefox Translations models

USAGE:
  fxtranslate list [lang] [--all]             List supported languages (or raw
                                              model pairs with --all; \`list --help\`)
  fxtranslate translate <src> <trg> [text…]   Translate: args if given, else stdin
                                              lines, else an interactive TTY prompt
  fxtranslate models <cmd>                    Manage locally cached models
                                              (\`models --help\`)

Non-English pairs pivot through English automatically (\`es → fr\` runs \`es → en\`
then \`en → fr\`); see pivot-translations.md.

OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)
  -h, --help          Show this help

EXAMPLES:
  fxtranslate list es
  echo "Hola mundo." | fxtranslate translate es fr   # pivots via English
  fxtranslate translate en es "Hello world."
  fxtranslate translate en es                 # interactive
  fxtranslate models list                     # what's cached, and how big`;

/** `models` usage. Mirrors `cli.rs` `MODELS_USAGE`. */
const MODELS_USAGE = `fxtranslate models — manage locally cached models

USAGE:
  fxtranslate models list                 Show the cache location and every cached
                                          model pair with its size and the total
  fxtranslate models add <src> <trg>      Download a pair ahead of time (both legs
                                          of a pivot, e.g. \`add es fr\` fetches
                                          es→en and en→fr) without translating
  fxtranslate models rm <pair> | --all    Delete a cached pair (or the whole cache),
                                          reporting the space reclaimed
  fxtranslate models info <pair>          Show a cached pair's files, their sizes,
                                          and their on-disk paths

A <pair> is either two tags (\`en es\`) or the joined directory name (\`en-es\`) shown
by \`models list\`. Downloads reuse the same verified cache as \`translate\`, so \`list\`
shows exactly what a translation would load.

OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)

EXAMPLES:
  fxtranslate models list
  fxtranslate models add es fr        # pre-fetch the es→en→fr pivot
  fxtranslate models info en-es
  fxtranslate models rm en-es
  fxtranslate models rm --all`;

/** `list` usage. Mirrors `cli.rs` `LIST_USAGE`. */
const LIST_USAGE = `fxtranslate list — list supported languages and models

USAGE:
  fxtranslate list [lang] [--all]

By default, \`list\` shows LANGUAGES, not raw models. A language is listed under
"Fully supported" when it can translate both to and from other languages — every
such pair works, directly or by pivoting through English. Languages that ship a
model in only one direction (e.g. only \`en → xx\`) work only that way, so they
appear under "Single-direction models" with the direction they support.

Underneath, every Firefox Translations model is a one-way pair to or from English
(\`en → es\` and \`es → en\` are separate models). Pass --all to list those raw pairs.

A [lang] argument filters by prefix, so \`zh\` catches \`zh-Hans\` and \`zh-Hant\`.
Display names are the standard BCP 47 language names; a tag with no known name shows
the code. See pivot-translations.md for how non-English pairs are served.

EXAMPLES:
  fxtranslate list                    # supported languages
  fxtranslate list es                 # just Spanish
  fxtranslate list --all              # every raw src → trg model pair
  fxtranslate list es --all           # both raw directions for Spanish`;

module.exports = { USAGE, MODELS_USAGE, LIST_USAGE };
