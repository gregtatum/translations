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

/** The records endpoint for the model collection. Mirrors `remote::records_url`. */
function recordsUrl() {
  return `${PROD_ENDPOINT}/v1/buckets/main/collections/${COLLECTION}/records`;
}

/**
 * A minimal `Fetch` the shell injects into `list`: `get(url)` returns the raw
 * response body as text. A later step adds streamed downloads for `models add`.
 *
 * @typedef {object} Fetch
 * @property {(url: string) => Promise<string>} get  GET a URL, returning the body text.
 */

/**
 * Build the real Node `Fetch` over the global `fetch` (Node >= 18). It only does
 * transport; the pure record parsing happens in the wasm core.
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
  };
}

module.exports = { recordsUrl, nodeFetch, PROD_ENDPOINT, COLLECTION };
