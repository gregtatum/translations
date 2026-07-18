// The `list` and `models` view formatters — a byte-for-byte port of the Rust CLI's
// rendering (crates/fxtranslate-cli/src/cli.rs). Presentation is a shell concern
// (duplicated on purpose, per notes/13), but pinned against the Rust oracle: every
// column width, size unit, sort order, and trailer must match. The pure decisions
// these render (catalog classification, version-gated pairs) come from the wasm
// core, not from here.

const { displayName } = require("./lang");

/** The production hub. Mirrors `route::PREFERRED_HUB`. */
const PREFERRED_HUB = "en";

/**
 * Unicode scalar (code-point) count of `s` — what Rust's `chars().count()` and
 * `{:<width$}` measure. `String.length` counts UTF-16 units, so a spread is used
 * to count code points instead (matters for any astral char; harmless otherwise).
 *
 * @param {string} s
 * @returns {number}
 */
function scalarLen(s) {
  return [...s].length;
}

/**
 * Left-justify (pad right) `s` to `width` scalars — Rust `{:<width$}`.
 *
 * @param {string} s
 * @param {number} width
 * @returns {string}
 */
function padEnd(s, width) {
  const n = scalarLen(s);
  return n >= width ? s : s + " ".repeat(width - n);
}

/**
 * Right-justify (pad left) `s` to `width` scalars — Rust `{:>width$}`.
 *
 * @param {string} s
 * @param {number} width
 * @returns {string}
 */
function padStart(s, width) {
  const n = scalarLen(s);
  return n >= width ? s : " ".repeat(width - n) + s;
}

/**
 * `{:.1}`-style rounding matching Rust's float formatter (round half to even), so
 * the one-decimal KiB/MiB/GiB sizes agree byte-for-byte with the oracle.
 *
 * @param {number} value
 * @returns {string}
 */
function oneDecimal(value) {
  const scaled = value * 10;
  let rounded = Math.round(scaled);
  // Math.round rounds .5 up; Rust rounds half to even. Correct the exact-tie case.
  if (Math.abs(scaled - Math.trunc(scaled) - 0.5) < Number.EPSILON) {
    const floor = Math.floor(scaled);
    rounded = floor % 2 === 0 ? floor : floor + 1;
  }
  return (rounded / 10).toFixed(1);
}

/**
 * Human-readable byte count on the base-1024 scale — raw bytes under 1 KiB, then
 * one decimal of KiB/MiB/GiB. Mirrors Rust's `human_bytes`.
 *
 * @param {number} n
 * @returns {string}
 */
function humanBytes(n) {
  const KIB = 1024;
  const MIB = KIB * 1024;
  const GIB = MIB * 1024;
  const f = n;
  if (f >= GIB) {
    return `${oneDecimal(f / GIB)} GiB`;
  }
  if (f >= MIB) {
    return `${oneDecimal(f / MIB)} MiB`;
  }
  if (f >= KIB) {
    return `${oneDecimal(f / KIB)} KiB`;
  }
  return `${n} B`;
}

/**
 * ANSI palette `(cyan, green, dim, reset)`, or empty strings when color is off.
 * Mirrors Rust's `palette`.
 *
 * @param {boolean} color
 * @returns {[string, string, string, string]}
 */
function palette(color) {
  return color
    ? ["\x1b[36m", "\x1b[32m", "\x1b[2m", "\x1b[0m"]
    : ["", "", "", ""];
}

/**
 * Whether a `list` language view surfaces `lang` for `query`. Mirrors Rust's
 * `language_query_matches` (shell logic in cli.rs).
 *
 * @param {string} lang
 * @param {string} query
 * @returns {boolean}
 */
function languageQueryMatches(lang, query) {
  const dash = query.indexOf("-");
  if (dash >= 0) {
    const a = query.slice(0, dash);
    const b = query.slice(dash + 1);
    return lang.startsWith(a) || lang.startsWith(b);
  }
  return lang.startsWith(query);
}

/**
 * Whether a `list --all` query selects the pair `src → trg`. Mirrors Rust's
 * `remote::language_matches` — a shallow prefix match; the version gate that makes
 * this list correct lives in the wasm `modelPairs`.
 *
 * @param {string} src
 * @param {string} trg
 * @param {string} query
 * @returns {boolean}
 */
function languageMatches(src, trg, query) {
  const dash = query.indexOf("-");
  if (dash >= 0) {
    const qSrc = query.slice(0, dash);
    const qTrg = query.slice(dash + 1);
    return src.startsWith(qSrc) && trg.startsWith(qTrg);
  }
  return src.startsWith(query) || trg.startsWith(query);
}

/**
 * Render the aligned `src → trg` table for `rows` — the `--all` view. Names and the
 * source tag are padded before color-wrapping. Mirrors Rust's `render_pairs`.
 *
 * @param {[string, string][]} rows
 * @param {boolean} color
 * @param {(s: string) => void} out
 */
function renderPairs(rows, color, out) {
  const wSrc = Math.max(0, ...rows.map(([s]) => scalarLen(displayName(s))));
  const wStag = Math.max(0, ...rows.map(([s]) => scalarLen(s) + 2));
  const wTrg = Math.max(0, ...rows.map(([, t]) => scalarLen(displayName(t))));
  const [cyan, green, dim, reset] = palette(color);
  for (const [s, t] of rows) {
    const sname = padEnd(displayName(s), wSrc);
    const stag = padEnd(`(${s})`, wStag);
    const tname = padEnd(displayName(t), wTrg);
    out(
      `${cyan}${sname}${reset} ${dim}${stag}${reset} ${dim}→${reset} ${green}${tname}${reset} ${dim}(${t})${reset}\n`,
    );
  }
}

/**
 * Render single-direction rows as `source → target (src trg)`. Mirrors Rust's
 * `render_single_direction`.
 *
 * @param {[string, string][]} rows
 * @param {boolean} color
 * @param {(s: string) => void} out
 */
function renderSingleDirection(rows, color, out) {
  const wSrc = Math.max(0, ...rows.map(([s]) => scalarLen(displayName(s))));
  const wTrg = Math.max(0, ...rows.map(([, t]) => scalarLen(displayName(t))));
  const [cyan, green, dim, reset] = palette(color);
  for (const [s, t] of rows) {
    const sname = padEnd(displayName(s), wSrc);
    const tname = padEnd(displayName(t), wTrg);
    out(
      `${cyan}${sname}${reset} ${dim}→${reset} ${green}${tname}${reset} ${dim}(${s} ${t})${reset}\n`,
    );
  }
}

/**
 * The `list --all` view from the version-gated model pairs. Filters by `query`
 * (via {@link languageMatches}) and renders the raw `src → trg` table. Returns the
 * number of pairs shown, or throws with the Rust no-match message.
 *
 * @param {[string, string][]} allPairs  version-gated pairs from wasm `modelPairs`
 * @param {string | undefined} query
 * @param {boolean} color
 * @param {(s: string) => void} out
 * @returns {number}
 */
function writePairs(allPairs, query, color, out) {
  const shown = allPairs.filter(([s, t]) =>
    query === undefined ? true : languageMatches(s, t, query),
  );
  if (shown.length === 0) {
    throw new Error(
      `no model pairs match \`${query ?? ""}\` (${allPairs.length} pairs available; try \`fxtranslate list --all\`)`,
    );
  }
  renderPairs(shown, color, out);
  return shown.length;
}

/**
 * The default `list` view: languages, not raw pairs. Renders the "Fully supported"
 * and "Single-direction models" sections from the wasm catalog. Returns the number
 * of languages shown, or throws with the Rust no-match message. Mirrors Rust's
 * `write_languages`.
 *
 * @param {{ bidirectional: string[], sourceOnly: string[], targetOnly: string[] }} cat
 * @param {string | undefined} query
 * @param {boolean} color
 * @param {(s: string) => void} out
 * @returns {number}
 */
function writeLanguages(cat, query, color, out) {
  /** @param {string} l */
  const keep = (l) => (query === undefined ? true : languageQueryMatches(l, query));
  const bidi = cat.bidirectional.filter(keep);
  const sourceOnly = cat.sourceOnly.filter(keep);
  const targetOnly = cat.targetOnly.filter(keep);
  const total = bidi.length + sourceOnly.length + targetOnly.length;
  if (total === 0) {
    throw new Error(
      `no languages match \`${query ?? ""}\` (try \`fxtranslate list\`, or \`list --all\` for raw pairs)`,
    );
  }

  const [cyan, , dim, reset] = palette(color);

  if (bidi.length > 0) {
    out("Fully supported (translate to and from any other):\n");
    const w = Math.max(0, ...bidi.map((l) => scalarLen(displayName(l))));
    for (const l of bidi) {
      const name = padEnd(displayName(l), w);
      out(`  ${cyan}${name}${reset} ${dim}(${l})${reset}\n`);
    }
  }

  // Single-direction: hub→L (target-only) then L→hub (source-only).
  /** @type {[string, string][]} */
  const oneWay = [
    ...targetOnly.map((l) => /** @type {[string, string]} */ ([PREFERRED_HUB, l])),
    ...sourceOnly.map((l) => /** @type {[string, string]} */ ([l, PREFERRED_HUB])),
  ];
  if (oneWay.length > 0) {
    if (bidi.length > 0) {
      out("\n");
    }
    out("Single-direction models:\n");
    renderSingleDirection(oneWay, color, out);
  }

  return total;
}

/**
 * A cached pair's `Source → Target` label, or `null` when the directory name
 * doesn't carry the hub on one side. Mirrors Rust's `pretty_pair`.
 *
 * @param {string} name
 * @returns {[string, string] | null}
 */
function prettyPair(name) {
  const hub = PREFERRED_HUB;
  if (name.startsWith(`${hub}-`)) {
    return [displayName(hub), displayName(name.slice(hub.length + 1))];
  }
  if (name.endsWith(`-${hub}`)) {
    return [displayName(name.slice(0, name.length - hub.length - 1)), displayName(hub)];
  }
  return null;
}

module.exports = {
  PREFERRED_HUB,
  scalarLen,
  padEnd,
  padStart,
  humanBytes,
  palette,
  languageQueryMatches,
  languageMatches,
  writePairs,
  writeLanguages,
  prettyPair,
};
