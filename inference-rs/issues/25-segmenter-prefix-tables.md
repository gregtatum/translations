# Build non-breaking-prefix tables for all supported Firefox languages

**Open, future. Depends on [24-abbreviation-suppression.md](24-abbreviation-suppression.md).**
The Moses prefix lists cover only ~25 languages (the European set). Once the
suppression mechanism from issue 24 exists, extend the prefix tables to **every
Firefox-supported source language** so abbreviation handling isn't a two-tier
feature (good for `de/fr/es/…`, absent for the rest). This is a data-sourcing
problem, not an engine change.

## Scope

- Enumerate the source languages Firefox translations supports that lack a Moses
  prefix list (everything outside the 25 in issue 24).
- Produce a non-breaking-prefix table per language, in the same format issue 24
  consumes (plain prefixes + `# NUMERIC_ONLY #` markers), with provenance for each.
- Languages that don't use `.`-terminated abbreviations, or are space-less
  (Thai/Lao/Khmer/Burmese, CJK), likely need **no** table — sentence breaks there
  come from hard terminators or are absent; document that decision per language
  rather than shipping empty files.

## Approaches (to evaluate)

1. **Source existing lists online.** Candidates: CLDR/ULI sentence-break
   suppression data (the same data ICU's `content_locale` draws on), Moses-style
   lists maintained by other MT projects, language-specific abbreviation
   dictionaries. Pro: human-curated, citable provenance. Con: coverage is uneven
   and licenses vary — must be checked per source.
2. **Synthetic LLM generation.** Prompt an LLM for the common abbreviations in
   each language that are followed by `.` and shouldn't end a sentence (titles,
   units, ordinals, legal/academic shorthand), then normalize to the table format.
   Pro: uniform coverage. Con: needs validation — hallucinated or wrong-register
   entries would *cause* mis-merges; must be reviewed and tested, and provenance
   labeled as generated.

A hybrid is likely: prefer sourced lists where they exist and are adequately
licensed; fill gaps with reviewed synthetic generation.

## Validation

- Reuse issue 24's curated-test approach per newly-added language: a handful of
  real abbreviation cases + a numeric-only case + a negative (a real sentence
  boundary that must still break).
- Optionally, the `ssplit`/ICU4X differential harness from issue 24 as a
  regression signal on languages where a reference exists.
- Guard against over-suppression: a bad prefix entry silently *merges* sentences,
  which then truncate past the context window — worse than over-splitting. Tests
  should include boundaries that must survive.

## Provenance & licensing

Every table records its source (URL/dataset + date, or "LLM-generated, reviewed
by …") and license. Generated tables are marked as such so they can be revisited.

## Done when

- Every supported Firefox source language either has a documented prefix table or
  a documented reason it needs none.
- Each table has provenance + license and passes per-language suppression tests.
