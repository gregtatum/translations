// The JS `Fetch` shell — the Node counterpart to the Rust `Fetch` trait
// (crates/fxtranslate/src/fetch.rs) and `remote::fetch_records`
// (crates/fxtranslate/src/remote.rs). Orchestration stays in the shell: it does
// the async HTTP itself and hands the raw response body to the synchronous,
// pure wasm core (`parseRecords`/`catalog`/`modelPairs`). The Rust CLI hits the
// SAME endpoint, so a live diff of the two is apples-to-apples.

// Remote Settings model discovery, mirroring remote.rs constants exactly.
const PROD_ENDPOINT = "https://firefox.settings.services.mozilla.com";
// The `-v2` collection, not the legacy `translations-models`.
const COLLECTION = "translations-models-v2";

// Environment override for the records endpoint. Mirrors `remote::RECORDS_URL_ENV`:
// when set, `recordsUrl()` returns it verbatim, so the conformance harness can
// point both CLIs at a committed records fixture (served over http://127.0.0.1:PORT)
// and keep `list` hermetic + byte-exact. Unset = live production behavior.
const RECORDS_URL_ENV = "FXTRANSLATE_RECORDS_URL";

/**
 * The records endpoint for the model collection. Honors the `FXTRANSLATE_RECORDS_URL`
 * override when set; otherwise the live production URL. Mirrors `remote::records_url`.
 */
function recordsUrl() {
  const override = process.env[RECORDS_URL_ENV];
  if (override) {
    return override;
  }
  return `${PROD_ENDPOINT}/v1/buckets/main/collections/${COLLECTION}/records`;
}

/**
 * The `Fetch` the shell injects: `get(url)` returns a body as text (for `list`
 * record discovery); `download(url, onProgress)` streams a (zstd) attachment fully
 * into memory, firing `onProgress(done, total)` as bytes arrive — the transport half
 * of the `models add` / `translate` download path (the wasm core does decode+verify).
 * Orchestration stays in the shell: it does the async HTTP itself and hands the raw
 * bytes to the synchronous core.
 *
 * @typedef {object} Fetch
 * @property {(url: string) => Promise<string>} get  GET a URL, returning the body text.
 * @property {(url: string, onProgress: (done: number, total: number | undefined) => void) => Promise<Uint8Array>} download
 *   GET a URL, streaming the body to memory with progress; returns the full bytes.
 */

/**
 * Build the real Node `Fetch` over the global `fetch` (Node >= 18). It only does
 * transport; the pure record parsing / decode+verify happens in the wasm core.
 *
 * @returns {Fetch}
 */
function nodeFetch() {
  return {
    async get(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`GET ${url} failed: HTTP ${res.status}`);
      }
      return await res.text();
    },

    async download(url, onProgress) {
      const res = await fetch(url);
      if (!res.ok || !res.body) {
        throw new Error(`GET ${url} failed: HTTP ${res.status}`);
      }
      const lenHeader = res.headers.get("content-length");
      const total = lenHeader ? Number(lenHeader) : undefined;
      /** @type {Uint8Array[]} */
      const chunks = [];
      let done = 0;
      onProgress(done, total);
      for await (const chunk of res.body) {
        const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
        chunks.push(bytes);
        done += bytes.length;
        onProgress(done, total);
      }
      return Buffer.concat(chunks);
    },
  };
}

module.exports = { recordsUrl, nodeFetch, PROD_ENDPOINT, COLLECTION, RECORDS_URL_ENV };
