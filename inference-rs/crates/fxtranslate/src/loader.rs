//! End-to-end model loading: Remote Settings discovery + verified cache + engine
//! build, in one call. The batteries-included convenience an embedder reaches for
//! when it doesn't want to orchestrate [`remote`](crate::remote),
//! [`cache`](crate::cache), and [`Engine::load`] by hand — pass a [`Fetch`] (the
//! built-in [`NetworkFetch`](crate::fetch::NetworkFetch) under `net`, or your own
//! client) and a [`Cache`], get a ready [`Engine`].

use crate::cache::{ensure_model, Cache, ModelFiles};
use crate::engine::{Engine, Translation};
use crate::fetch::Fetch;
use crate::remote::fetch_records;
use crate::route::{resolve_route, Route};

/// Resolve, download (or cache-hit), and hash-verify every file needed to
/// translate `src`→`trg`, without building the engine — for callers that want the
/// on-disk paths (e.g. to `Engine::load_mmapped` themselves, or inspect them).
/// Fetches the Remote Settings records via `fetch`, then defers to
/// [`ensure_model`].
pub fn ensure_files(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<ModelFiles, String> {
    let records = fetch_records(fetch)?;
    ensure_model(fetch, cache, &records, src, trg)
}

/// Resolve `src`→`trg` and download+verify every file the route needs — including
/// **both** legs of a pivot — **without** building an engine. The pre-download path
/// behind `fxtranslate models add`: it mirrors [`load_translation`]'s discovery and
/// routing (so a pivot caches exactly the same two models a later `translate` would
/// load) but stops at the on-disk files. Returns the resolved [`Route`] (so the
/// caller can report the pivot hop) alongside one [`ModelFiles`] for a direct pair or
/// two — in run order — for a pivot.
pub fn ensure_route_files(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<(Route, Vec<ModelFiles>), String> {
    let records = fetch_records(fetch)?;
    let route = resolve_route(&records, src, trg)?;
    let files = match &route {
        Route::Direct { src, trg } => vec![ensure_model(fetch, cache, &records, src, trg)?],
        Route::Pivot { src, pivot, trg } => vec![
            ensure_model(fetch, cache, &records, src, pivot)?,
            ensure_model(fetch, cache, &records, pivot, trg)?,
        ],
    };
    Ok((route, files))
}

/// Discover, download+cache (verified), and build a ready [`Engine`] for
/// `src`→`trg`. The one-call path: `fetch_records` → [`ensure_model`] →
/// [`Engine::load`]. `fetch` supplies HTTP (the built-in `NetworkFetch` under
/// `net`, or an embedder's own [`Fetch`]); `cache` is where verified files land
/// (see [`Cache::locate`] for the platform default).
pub fn load_engine(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<Engine, String> {
    let files = ensure_files(fetch, cache, src, trg)?;
    Engine::load(&files.model, &files.src_vocab, &files.trg_vocab)
}

/// Discover, download+cache (verified), and build a ready [`Translation`] for
/// `src`→`trg` — the pivot-aware primary path. A direct model yields
/// [`Translation::Direct`]; a non-hub pair with no direct model is served by
/// resolving a pivot (see [`resolve_route`]) and loading **both** legs, held
/// resident for the session as [`Translation::Pivot`]. Records are fetched once
/// and reused across both legs.
pub fn load_translation(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<Translation, String> {
    let records = fetch_records(fetch)?;
    match resolve_route(&records, src, trg)? {
        Route::Direct { src, trg } => {
            let files = ensure_model(fetch, cache, &records, &src, &trg)?;
            let engine = Engine::load(&files.model, &files.src_vocab, &files.trg_vocab)?;
            Ok(Translation::Direct(engine))
        }
        Route::Pivot { src, pivot, trg } => {
            let leg1 = ensure_model(fetch, cache, &records, &src, &pivot)?;
            let leg2 = ensure_model(fetch, cache, &records, &pivot, &trg)?;
            let first = Engine::load(&leg1.model, &leg1.src_vocab, &leg1.trg_vocab)?;
            let second = Engine::load(&leg2.model, &leg2.src_vocab, &leg2.trg_vocab)?;
            Ok(Translation::Pivot {
                pivot,
                first,
                second,
            })
        }
    }
}
