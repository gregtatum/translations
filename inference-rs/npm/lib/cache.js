// The JS cache reader/writer — the Node counterpart to the Rust `Cache`
// (crates/fxtranslate/src/cache.rs). Cache *storage* is deliberately reimplemented
// per language (a shell concern), but the on-disk contract is shared: same root,
// same `<src>-<trg>` layout, same size accounting, same `.partial`/`.download` temp
// names — so a user with both the Rust and Node CLIs installed shares one model
// cache. The read-only paths (`root`, `listCached`, `pairFiles`, `removePair`,
// `cachedModel`) plus the download→verify→atomic-write path (`ensure`,
// `ensureModel`) all live here. Crypto/zstd is NOT reimplemented: `ensure` calls the
// wasm core's `verifyAndDecompress` (sha256 + zstd) to decode+verify each download.

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const wasm = require("../wasm/fxtranslate_wasm.js");

/** The settings-attachments CDN root — mirrors `remote::CDN_ROOT`. */
const CDN_ROOT = "https://firefox-settings-attachments.cdn.mozilla.net";

/**
 * The major model version this build is validated against — mirrors
 * `remote::SUPPORTED_MAJOR`. Records outside this major are ignored so a future
 * format bump can't be mistranslated.
 */
const SUPPORTED_MAJOR = 3;

/**
 * Dotted-version sort key (`"3.1"` → `[3, 1]`); non-numeric parts sort as 0.
 * Mirrors `remote::version_key`.
 *
 * @param {string} v
 * @returns {number[]}
 */
function versionKey(v) {
  return v.split(".").map((p) => {
    const n = parseInt(p, 10);
    return Number.isNaN(n) ? 0 : n;
  });
}

/**
 * Lexicographic compare of two dotted-version keys. Mirrors Rust's `Vec<u64>`
 * `cmp` (shorter-prefix-first when one is a prefix of the other).
 *
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
function compareVersionKey(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) {
      return a[i] < b[i] ? -1 : 1;
    }
  }
  return a.length - b.length;
}

/**
 * Whether this build can use a record of `version`: its major must equal
 * {@link SUPPORTED_MAJOR}. Mirrors `remote::version_supported`.
 *
 * @param {string} version
 * @returns {boolean}
 */
function versionSupported(version) {
  return versionKey(version)[0] === SUPPORTED_MAJOR;
}

/**
 * The latest supported-version record of `fileType` for the pair, or undefined.
 * Records outside the supported major are ignored; among the rest the latest minor
 * wins. Mirrors `remote::pick`. `records` are the objects `parseRecords` returns
 * (fields `fileType`, `sourceLanguage`, `targetLanguage`, `version`, …).
 *
 * @param {Record[]} records
 * @param {string} fileType
 * @param {string} src
 * @param {string} trg
 * @returns {Record | undefined}
 */
function pick(records, fileType, src, trg) {
  let best;
  let bestKey;
  for (const r of records) {
    if (
      r.fileType !== fileType ||
      r.sourceLanguage !== src ||
      r.targetLanguage !== trg ||
      !versionSupported(r.version)
    ) {
      continue;
    }
    const key = versionKey(r.version);
    if (best === undefined || compareVersionKey(key, /** @type {number[]} */ (bestKey)) > 0) {
      best = r;
      bestKey = key;
    }
  }
  return best;
}

/**
 * Full CDN URL of a record's (zstd) attachment. Mirrors `Record::cdn_url`.
 *
 * @param {Record} record
 * @returns {string}
 */
function cdnUrl(record) {
  return `${CDN_ROOT}/${record.location}`;
}

/**
 * A Remote Settings model-file record as `parseRecords` returns it.
 *
 * @typedef {object} Record
 * @property {string} name
 * @property {string} fileType
 * @property {string} sourceLanguage
 * @property {string} targetLanguage
 * @property {string} version
 * @property {string | null} architecture
 * @property {string | null} decompressedHash
 * @property {string} location
 */

/**
 * The verified model-file paths the engine loads. For shared-vocab pairs
 * `srcVocab === trgVocab`; split-vocab (CJK) pairs differ. `lex` is present only
 * when the pair ships a shortlist. Mirrors Rust's `ModelFiles`.
 *
 * @typedef {object} ModelFiles
 * @property {string} model
 * @property {string} srcVocab
 * @property {string} trgVocab
 * @property {string | undefined} lex
 */

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
    /**
     * Draw a `\r`-updated download progress line to stderr. `main` (the shell) sets
     * this from `stderr.isTTY`, so pipes/CI/tests stay quiet — the same TTY split as
     * `list`'s color. Off by default (read-only verbs never download).
     * @type {(name: string, done: number, total: number | undefined) => void}
     */
    this.onProgress = () => {};
    /** Whether a progress renderer is wired (so `ensure` closes the `\r` line). */
    this.showProgress = false;
  }

  /**
   * Enable a stderr download-progress callback (the mirror of Rust's
   * `Cache::with_progress`). The shell passes a renderer only when stderr is a TTY;
   * otherwise it passes nothing and downloads are silent. Returns `this` for
   * chaining, matching the Rust builder shape.
   *
   * @param {((name: string, done: number, total: number | undefined) => void) | undefined} onProgress
   * @returns {Cache}
   */
  withProgress(onProgress) {
    if (onProgress) {
      this.onProgress = onProgress;
      this.showProgress = true;
    }
    return this;
  }

  /**
   * Per-pair directory: `<root>/<src>-<trg>`. Mirrors `Cache::pair_dir`.
   *
   * @param {string} src
   * @param {string} trg
   * @returns {string}
   */
  pairDir(src, trg) {
    return path.join(this.root, `${src}-${trg}`);
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

  /**
   * Delete the `name` pair directory and everything in it. Idempotent: `false` if it
   * wasn't there, `true` if removed. Mirrors `Cache::remove_pair`.
   *
   * @param {string} name
   * @returns {boolean}
   */
  removePair(name) {
    const dir = this.pairPath(name);
    if (!fs.existsSync(dir)) {
      return false;
    }
    fs.rmSync(dir, { recursive: true, force: true });
    return true;
  }

  /**
   * Ensure `record`'s decompressed file is present and hash-verified, fetching (via
   * `fetch.download`) and decompressing it only on a miss or a hash mismatch. The
   * download→verify→atomic-write path: stream the zstd attachment to a `.download`
   * temp, decode+verify it with the wasm core (`verifyAndDecompress` = zstd +
   * sha256), then write the plaintext to a `.partial` temp and rename it into place —
   * so an interrupted run never leaves a partial file a later run would trust.
   * Mirrors `Cache::ensure`. Returns the on-disk path.
   *
   * @param {import("./fetch").Fetch} fetch
   * @param {Record} record
   * @returns {Promise<string>}
   */
  async ensure(fetch, record) {
    const dir = this.pairDir(record.sourceLanguage, record.targetLanguage);
    const dest = path.join(dir, record.name);

    // Cache hit: file present and (if we know the expected hash) matching.
    if (fs.existsSync(dest) && fs.statSync(dest).isFile()) {
      const expected = record.decompressedHash;
      if (expected === null || expected === undefined) {
        return dest; // no hash to check against; trust presence
      }
      const got = sha256Hex(fs.readFileSync(dest));
      if (got === expected) {
        return dest;
      }
      // Corrupt/partial/stale — fall through and re-fetch.
    }

    fs.mkdirSync(dir, { recursive: true });

    const url = cdnUrl(record);
    const download = path.join(dir, `.${record.name}.download`);
    const name = record.name;
    // Stream the (zstd) attachment to the `.download` temp, rendering progress when
    // enabled; the shell owns the async HTTP, the core owns decode+verify.
    let drew = false;
    const compressed = await fetch.download(url, (done, total) => {
      if (this.showProgress) {
        this.onProgress(name, done, total);
        drew = true;
      }
    });
    if (drew) {
      process.stderr.write("\n"); // finish the in-place progress line
    }
    // Persist the raw download so a later run could resume/inspect (matches Rust's
    // on-disk `.download`); then decode+verify entirely in the wasm core.
    fs.writeFileSync(download, compressed);
    /** @type {Buffer} */
    let bytes;
    try {
      const expected = record.decompressedHash ?? undefined;
      bytes = Buffer.from(wasm.verifyAndDecompress(compressed, expected));
    } catch (e) {
      fs.rmSync(download, { force: true });
      const msg = e instanceof Error ? e.message : String(e);
      // Name the record, matching Rust's `hash mismatch for {name} after download`.
      throw new Error(msg.includes(name) ? msg : `${msg} for ${name}`);
    }
    fs.rmSync(download, { force: true });

    // Atomic write: temp then rename.
    const tmp = path.join(dir, `.${record.name}.partial`);
    fs.writeFileSync(tmp, bytes);
    fs.renameSync(tmp, dest);
    return dest;
  }
}

/**
 * Hex SHA-256 of an already-decompressed cached file, for the cache-hit re-verify
 * against a record's `decompressedHash` (mirrors the hit branch of `Cache::ensure`).
 * The wasm core's `verifyAndDecompress` covers the download path (zstd + sha256);
 * here the bytes are already plaintext, so Node's `crypto` hashes them directly
 * rather than round-tripping through a zstd decoder.
 *
 * @param {Buffer} bytes
 * @returns {string}
 */
function sha256Hex(bytes) {
  const crypto = require("node:crypto");
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

/**
 * Resolve, download+cache (verified), and return every file needed to translate the
 * direct pair `src`→`trg` — model, vocab(s), and (optional) lex. Shared vocab ships
 * one `vocab`; split vocab (CJK) ships `srcvocab`/`trgvocab`. Mirrors Rust's
 * `ensure_model`. `records` are the parsed records the shell already fetched.
 *
 * @param {import("./fetch").Fetch} fetch
 * @param {Cache} cache
 * @param {Record[]} records
 * @param {string} src
 * @param {string} trg
 * @returns {Promise<ModelFiles>}
 */
async function ensureModel(fetch, cache, records, src, trg) {
  const model = pick(records, "model", src, trg);
  if (!model) {
    throw new Error(`no model for ${src}-${trg} in Remote Settings`);
  }
  const modelPath = await cache.ensure(fetch, model);

  let srcVocab;
  let trgVocab;
  const shared = pick(records, "vocab", src, trg);
  if (shared) {
    const p = await cache.ensure(fetch, shared);
    srcVocab = p;
    trgVocab = p;
  } else {
    const sv = pick(records, "srcvocab", src, trg);
    if (!sv) {
      throw new Error(`no vocab/srcvocab for ${src}-${trg}`);
    }
    const tv = pick(records, "trgvocab", src, trg);
    if (!tv) {
      throw new Error(`no trgvocab for ${src}-${trg}`);
    }
    srcVocab = await cache.ensure(fetch, sv);
    trgVocab = await cache.ensure(fetch, tv);
  }

  const lexRecord = pick(records, "lex", src, trg);
  const lex = lexRecord ? await cache.ensure(fetch, lexRecord) : undefined;

  return { model: modelPath, srcVocab, trgVocab, lex };
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

module.exports = {
  Cache,
  defaultRoot,
  platformCacheDir,
  dirSize,
  ensureModel,
  pick,
  cdnUrl,
  versionSupported,
  CDN_ROOT,
  SUPPORTED_MAJOR,
};
