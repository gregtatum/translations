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
use crate::route::{resolve_route, Route, PREFERRED_HUB};

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
    engine_from(&files)
}

/// Build an [`Engine`] from resolved [`ModelFiles`] (model + the two vocab paths).
fn engine_from(files: &ModelFiles) -> Result<Engine, String> {
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
    // Remote Settings first, so an online run always resolves the latest model version.
    // If discovery fails — offline, DNS down, Remote Settings unreachable — fall back to
    // whatever is already cached: a previously-downloaded pair still translates with no
    // network, instead of the whole command failing at the records fetch.
    let records = match fetch_records(fetch) {
        Ok(records) => records,
        Err(net) => {
            return load_cached_translation(cache, src, trg)
                .map_err(|miss| format!("{miss}; and Remote Settings was unreachable: {net}"))
        }
    };
    match resolve_route(&records, src, trg)? {
        Route::Direct { src, trg } => {
            let files = ensure_model(fetch, cache, &records, &src, &trg)?;
            Ok(Translation::Direct(engine_from(&files)?))
        }
        Route::Pivot { src, pivot, trg } => {
            let leg1 = ensure_model(fetch, cache, &records, &src, &pivot)?;
            let leg2 = ensure_model(fetch, cache, &records, &pivot, &trg)?;
            Ok(Translation::Pivot {
                pivot,
                first: engine_from(&leg1)?,
                second: engine_from(&leg2)?,
            })
        }
    }
}

/// Load `src`→`trg` from the local cache alone — the offline fallback for
/// [`load_translation`], with no network and no Remote Settings. A direct cached model
/// wins; otherwise a pivot through the hub ([`PREFERRED_HUB`]) works when both legs are
/// already cached (the routing every Firefox pair uses). Errs when nothing usable is on
/// disk, so the caller can report that alongside the discovery failure.
fn load_cached_translation(cache: &Cache, src: &str, trg: &str) -> Result<Translation, String> {
    if let Some(files) = cache.cached_model(src, trg) {
        return Ok(Translation::Direct(engine_from(&files)?));
    }
    let hub = PREFERRED_HUB;
    if src != hub && trg != hub {
        if let (Some(leg1), Some(leg2)) =
            (cache.cached_model(src, hub), cache.cached_model(hub, trg))
        {
            return Ok(Translation::Pivot {
                pivot: hub.to_string(),
                first: engine_from(&leg1)?,
                second: engine_from(&leg2)?,
            });
        }
    }
    Err(format!("no cached model for {src}→{trg}"))
}
