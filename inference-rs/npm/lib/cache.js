// The JS cache reader — the Node counterpart to the Rust `Cache`
// (crates/fxtranslate/src/cache.rs). Cache *storage* is deliberately reimplemented
// per language (a shell concern), but the on-disk contract is shared: same root,
// same `<src>-<trg>` layout, same size accounting — so a user with both the Rust
// and Node CLIs installed shares one model cache. Only the read-only paths (`root`,
// `listCached`, `pairFiles`) live here; downloads/writes come in step 4.

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

/**
 * The platform-native cache base — the JS mirror of Rust's `dirs::cache_dir()`.
 * These MUST agree so both CLIs resolve the same default cache:
 *   * macOS:   `~/Library/Caches`
 *   * Windows: `%LOCALAPPDATA%` (FOLDERID_LocalAppData)
 *   * Linux/other: `$XDG_CACHE_HOME`, else `~/.cache`
 * Returns `undefined` when it can't be determined, matching `dirs`' `None`.
 *
 * @returns {string | undefined}
 */
function platformCacheDir() {
  const home = os.homedir();
  if (process.platform === "darwin") {
    return home ? path.join(home, "Library", "Caches") : undefined;
  }
  if (process.platform === "win32") {
    return process.env.LOCALAPPDATA || undefined;
  }
  // Linux and the other Unixes: XDG spec.
  const xdg = process.env.XDG_CACHE_HOME;
  if (xdg && xdg.length > 0) {
    return xdg;
  }
  return home ? path.join(home, ".cache") : undefined;
}

/**
 * The default cache root — the platform cache base with an `fxtranslate/models`
 * subtree, mirroring `Cache::locate()`. Falls back to a local `.fxtranslate-cache`
 * dir when the base can't be found (same fallback string as Rust).
 *
 * @returns {string}
 */
function defaultRoot() {
  const base = platformCacheDir() || ".fxtranslate-cache";
  return path.join(base, "fxtranslate", "models");
}

/**
 * A cache rooted at `root`, or the platform default when `cacheDir` is undefined.
 * Note the `--cache-dir` semantics match Rust's `open_cache`: the override points
 * at the cache *root* (where the `<src>-<trg>` dirs live) directly — it is NOT
 * re-suffixed with `fxtranslate/models`.
 */
class Cache {
  /** @param {string | undefined} cacheDir */
  constructor(cacheDir) {
    /** @type {string} */
    this.root = cacheDir !== undefined ? cacheDir : defaultRoot();
  }

  /**
   * A pair directory by its (joined) name, refusing anything that isn't a single
   * normal path component — mirroring Rust's `pair_path` guard so a stray `..` or
   * absolute path can never escape the cache. Returns the joined path.
   *
   * @param {string} name
   * @returns {string}
   */
  pairPath(name) {
    // A single normal component: no separators, not `.`/`..`, not absolute.
    if (
      name.length === 0 ||
      name === "." ||
      name === ".." ||
      name.includes("/") ||
      name.includes("\\") ||
      path.isAbsolute(name)
    ) {
      throw new Error(`invalid model name \`${name}\``);
    }
    return path.join(this.root, name);
  }

  /**
   * Enumerate cached pairs: each immediate sub-directory of the root (dot-prefixed
   * skipped), sorted by name, each with its computed size. A missing root is not an
   * error — nothing cached yet. Mirrors `Cache::list_cached`.
   *
   * @returns {{ name: string, dir: string, bytes: number }[]}
   */
  listCached() {
    if (!fs.existsSync(this.root)) {
      return [];
    }
    const out = [];
    for (const entry of fs.readdirSync(this.root, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }
      const name = entry.name;
      if (name.startsWith(".")) {
        continue;
      }
      const dir = path.join(this.root, name);
      out.push({ name, dir, bytes: dirSize(dir) });
    }
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }

  /**
   * The cached files in the `name` pair directory as `{ name, bytes, path }`,
   * sorted by name; dot-prefixed (temp) files skipped. An absent directory yields
   * an empty list. Mirrors `Cache::pair_files`.
   *
   * @param {string} name
   * @returns {{ name: string, bytes: number, path: string }[]}
   */
  pairFiles(name) {
    const dir = this.pairPath(name);
    if (!fs.existsSync(dir)) {
      return [];
    }
    const out = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fname = entry.name;
      if (fname.startsWith(".")) {
        continue;
      }
      const full = path.join(dir, fname);
      const st = fs.statSync(full);
      if (st.isFile()) {
        out.push({ name: fname, bytes: st.size, path: full });
      }
    }
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }
}

/**
 * Recursive sum of regular-file byte lengths under `dir`, skipping dot-prefixed
 * entries (which covers the cache's own in-flight temp files). A missing `dir` is
 * `0`, not an error. Mirrors Rust's `dir_size`.
 *
 * @param {string} dir
 * @returns {number}
 */
function dirSize(dir) {
  if (!fs.existsSync(dir)) {
    return 0;
  }
  let total = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) {
      continue;
    }
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      total += dirSize(full);
    } else if (entry.isFile()) {
      total += fs.statSync(full).size;
    }
  }
  return total;
}

module.exports = { Cache, defaultRoot, platformCacheDir, dirSize };
