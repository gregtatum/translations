# Non-breaking-prefix suppression on top of ICU4X (fix the abbreviation regression)

**Open, scoped. Depends on [23-icu4x-segmentation-everywhere.md](23-icu4x-segmentation-everywhere.md).**
ICU4X (and every `Intl.Segmenter`) over-splits abbreviations: `Dr. Smith went to
Washington.` breaks into `["Dr.", "Smith went to Washington."]`. Add a thin,
data-driven suppression layer over ICU4X's UAX #29 boundaries that merges back a
break falling immediately after a known non-breaking prefix — restoring the
segmentation quality Moses/`ssplit` had before Firefox's WASM path dropped it.
Keep the modern UAX #29 base; treat abbreviation handling as a layer, not a
rewrite. Extending the prefix tables to all languages is the follow-up,
[25-segmenter-prefix-tables.md](25-segmenter-prefix-tables.md).

## Background: this recovers a real regression

Segmentation entered the project as **Moses**, via `ssplit-cpp` (browsermt's C++
reimplementation of the Moses sentence splitter), when `bergamot-translator` was
forked in as `./inference` (#867, 2024-10-01). `ssplit` uses per-language
non-breaking-prefix lists (`inference/3rd_party/ssplit-cpp/nonbreaking_prefixes/`,
copied from `moses-smt/mosesdecoder`) plus heuristics to avoid breaking after
abbreviations (`inference/3rd_party/ssplit-cpp/src/ssplit/ssplit.cpp:210`).

In **#945** (2024-12-03) the shipped Firefox WASM path switched from `ssplit` to
`Intl.Segmenter`; native builds kept `ssplit`. That was a deliberate
size/complexity trade that **regressed abbreviation segmentation** for the 25
languages with Moses prefix lists (`ca cs de el en es fi fr ga hu is it lt lv nl
pl pt ro ru sk sl sv ta yue zh`) — the high-traffic European set. Since fxtranslate
is Firefox-independent, we're free to *fix* this rather than inherit it.

`content_locale` does **not** rescue ICU here: probed `en/de/fr/es`, `Dr.`/`z. B.`/
`M.`/`Sr.` still split (only `ru т. е.` was suppressed). So the suppression has to
be ours.

## Design

A post-process over the boundary list ICU4X returns, mirroring `ssplit`'s decision
logic (`ssplit.cpp:200-226`) but far simpler because ICU4X already found the
candidate boundaries:

For each candidate boundary produced by ICU4X, **drop it** (merge the two spans)
when the sentence-final token is a suppressed prefix:

1. Take the word immediately before the terminating `.` (the last whitespace-
   delimited token of the left span).
2. If that word is in the active language's prefix list with class **1**
   (unconditional), suppress the break.
3. If it's class **2** (`#NUMERIC_ONLY#` — e.g. `No.`, `pp.`), suppress the break
   only when the next sentence starts with a digit.
4. Only `.` is suppressible this way; hard terminators (`? ! 。 ！ ？ …`) always
   break. Match `ssplit`'s "single uppercase initial" and `a.b.c` handling too
   (initials like `U.S.` — ICU4X already keeps these, so verify rather than
   re-implement).

The prefix list is selected by **source language**, which the shell/CLI already
knows (`src`). No list for a language → no suppression → identical to issue 23's
ICU4X output (graceful).

Toggle: suppression **on by default**; a flag/config value turns it off for exact
ICU4X / Firefox parity. (Independent of issue 23's compile-time `icu-segmenter`
opt-out — suppression only applies when ICU4X is compiled in.)

## Prefix tables + provenance

Source and copy the original Moses non-breaking-prefix files (the 25 above) into
the crate (e.g. `crates/fxtranslate/data/nonbreaking_prefixes/`), embedded at
compile time. Document provenance and license precisely, because they're
third-party data:

- Origin: `github.com/moses-smt/mosesdecoder` → `scripts/share/nonbreaking_prefixes`,
  via the `browsermt/ssplit-cpp` fork (see its `LICENSE.md` / `README.md`).
- Preserve the file format (`# NUMERIC_ONLY #` markers, comment lines) and a
  `PROVENANCE.md` noting source, commit/date copied, and license.

## Correctness proof

The logic is small and self-contained, so **curated per-language test cases are
the primary proof** — no oracle required. Cover, per language with a list:

- Unconditional prefix + uppercase follower → **no** break (`Dr. Smith`, `z. B.`,
  `M. Dupont`, `Sr. Gómez`).
- `#NUMERIC_ONLY#` prefix: break before a word, **no** break before a digit
  (`No. 5` stays, `No. Then` breaks).
- Initials / `a.b.c` (`U.S.`, `e.g.`) → **no** break (regression-guard on ICU4X).
- Decimals (`3.14`), ellipsis (`...`), hard terminators (`。！？`) → unaffected.
- A language with **no** list → output identical to bare ICU4X (issue 23).
- Whitespace round-trips through `reassemble` after a suppressed merge.

**Optional differential check (stretch, not required):** `ssplit-cpp` is in-repo
and builds; wire its CLI (`inference/3rd_party/ssplit-cpp/src/command/ssplit_main.cpp`)
as a segmentation oracle and report ICU4X+suppression vs `ssplit` boundary
divergence over a multilingual corpus. Use it to *quantify* how much of the #945
regression we recover, not as a gate (ICU4X and ssplit legitimately differ on
non-abbreviation cases).

## Done when

- The 25-language prefix tables are embedded with documented provenance.
- Suppression merges abbreviation boundaries per the design, defaulting on, with
  an off switch.
- Curated per-language tests pass; a no-list language is byte-identical to issue
  23's ICU4X output.
