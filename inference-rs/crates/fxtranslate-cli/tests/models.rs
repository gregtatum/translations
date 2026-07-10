//! End-to-end `models` tests: real argv (`fxtranslate models …`) driven through
//! `cli::run` against a temp cache dir and the mockable `Fetch` trait — no network,
//! no engine.
//!
//! Like `list.rs`, each test is a **visible transcript snapshot** so a reviewer can
//! audit formatting (the cache header, the aligned size table, the `[N …]` trailer).
//! Because the cache lives at a unique temp path, transcripts are normalized: the
//! `--cache-dir` path is rewritten to `<CACHE>` before the snapshot compare, so the
//! non-deterministic temp path never leaks into the expected block.
//!
//! `models add` drives the pivot-aware pre-download: `rs-pivot.json` supplies es↔en
//! and en→fr models+vocabs (no hashes → the tiny zstd fixture is trusted), so
//! `add es fr` caches both legs and a follow-up `models list` shows them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use fxtranslate::remote::{parse_records, records_url};
use fxtranslate_cli::cli::Deps;

mod common;
use common::{assert_transcript, run_transcript, MockFetch, MockTranslator, Streams};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// A fresh, unique temp cache dir (no tempfile dep), mirroring the library's
/// `tmp_cache`. Returned as a string so it can be passed as `--cache-dir` and used to
/// normalize the transcript.
fn tmp_cache_dir() -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    std::env::temp_dir()
        .join(format!(
            "fxtranslate-cli-models-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
        .to_string_lossy()
        .into_owned()
}

/// A `MockFetch` wired for `rs-pivot.json`: the records at `records_url()` and every
/// attachment routed to the tiny (trusted) zstd fixture, so any pair/leg downloads.
fn pivot_fetch() -> MockFetch {
    let body = fixture("rs-pivot.json");
    let recs = parse_records(std::str::from_utf8(&body).unwrap()).unwrap();
    let mut mock = MockFetch::new().route(&records_url(), body);
    for r in &recs {
        mock = mock.route(&r.cdn_url(), fixture("tiny.bin.zst"));
    }
    mock
}

/// Run `models …` argv against `fetch` and a fresh temp cache dir, returning the
/// transcript with the temp path rewritten to `<CACHE>`.
fn models(fetch: &MockFetch, args: &[&str]) -> String {
    let dir = tmp_cache_dir();
    run_with_dir(fetch, args, &dir)
}

/// Like [`models`] but against an explicit `dir`, so two calls (e.g. `add` then
/// `list`) share one cache.
fn run_with_dir(fetch: &MockFetch, args: &[&str], dir: &str) -> String {
    let translator = MockTranslator::new(); // unused by `models`
    let deps = Deps {
        fetch,
        translator: &translator,
    };
    let mut argv: Vec<&str> = args.to_vec();
    argv.extend(["--cache-dir", dir]);
    let out = run_transcript(&argv, &deps, Streams::default());
    out.replace(dir, "<CACHE>")
}

/// `models list` on an untouched cache: the location, a friendly empty note, and a
/// `[0 cached]` trailer — no network touched.
#[test]
fn list_empty_cache() {
    assert_transcript(
        "models list (empty)",
        &models(&MockFetch::new(), &["models", "list"]),
        &[
            "Cache: <CACHE>",
            "No models cached yet. Add one with `fxtranslate models add <src> <trg>`.",
            "[0 cached]",
        ],
    );
}

/// `models add es fr` pivots through English: the status names the hop, and both legs
/// (`es→en`, `en→fr`) land in the cache — visible in a follow-up `models list`.
#[test]
fn add_pivot_then_list() {
    let fetch = pivot_fetch();
    let dir = tmp_cache_dir();

    assert_transcript(
        "models add es fr",
        &run_with_dir(&fetch, &["models", "add", "es", "fr"], &dir),
        &[
            "[fxtranslate] downloading es→fr model…",
            "[fxtranslate] cached (es→en→fr, pivot).",
        ],
    );

    // The two pivot legs are now cached (each: model + vocab = 62 B).
    assert_transcript(
        "models list (after pivot add)",
        &run_with_dir(&fetch, &["models", "list"], &dir),
        &[
            "Cache: <CACHE>",
            "  English → French  (en-fr) 62 B",
            "  Spanish → English (es-en) 62 B",
            "Total: 124 B",
            "[2 cached]",
        ],
    );
}

/// `models add en es` resolves directly (no pivot), and the status says so.
#[test]
fn add_direct() {
    assert_transcript(
        "models add en es",
        &models(&pivot_fetch(), &["models", "add", "en", "es"]),
        &[
            "[fxtranslate] downloading en→es model…",
            "[fxtranslate] cached (en→es).",
        ],
    );
}

/// `models info` lists a cached pair's files with sizes and full on-disk paths.
#[test]
fn info_lists_files() {
    let fetch = pivot_fetch();
    let dir = tmp_cache_dir();
    let _ = run_with_dir(&fetch, &["models", "add", "en", "es"], &dir);

    assert_transcript(
        "models info en-es",
        &run_with_dir(&fetch, &["models", "info", "en-es"], &dir),
        &[
            "en-es (<CACHE>/en-es)",
            "  model.enes.bin  31 B  <CACHE>/en-es/model.enes.bin",
            "  vocab.enes.spm  31 B  <CACHE>/en-es/vocab.enes.spm",
            "Total: 62 B",
        ],
    );
}

/// `models info` on an uncached pair is a note, not an error.
#[test]
fn info_uncached() {
    assert_transcript(
        "models info en-es (uncached)",
        &models(&MockFetch::new(), &["models", "info", "en-es"]),
        &["en-es is not cached. Add it with `fxtranslate models add <src> <trg>`."],
    );
}

/// `models rm <pair>` deletes one pair and reports the space reclaimed; a follow-up
/// `list` shows the sibling pair untouched.
#[test]
fn rm_one_pair() {
    let fetch = pivot_fetch();
    let dir = tmp_cache_dir();
    let _ = run_with_dir(&fetch, &["models", "add", "es", "fr"], &dir);

    assert_transcript(
        "models rm en-fr",
        &run_with_dir(&fetch, &["models", "rm", "en-fr"], &dir),
        &["Removed en-fr (62 B)"],
    );
    assert_transcript(
        "models list (after rm)",
        &run_with_dir(&fetch, &["models", "list"], &dir),
        &[
            "Cache: <CACHE>",
            "  Spanish → English (es-en) 62 B",
            "Total: 62 B",
            "[1 cached]",
        ],
    );
}

/// `models rm en es` accepts the two-tag form (same dir as `en-es`).
#[test]
fn rm_accepts_two_tags() {
    let fetch = pivot_fetch();
    let dir = tmp_cache_dir();
    let _ = run_with_dir(&fetch, &["models", "add", "en", "es"], &dir);
    assert_transcript(
        "models rm en es",
        &run_with_dir(&fetch, &["models", "rm", "en", "es"], &dir),
        &["Removed en-es (62 B)"],
    );
}

/// `models rm <pair>` on an uncached pair is a note, not an error.
#[test]
fn rm_uncached() {
    assert_transcript(
        "models rm en-es (uncached)",
        &models(&MockFetch::new(), &["models", "rm", "en-es"]),
        &["en-es is not cached."],
    );
}

/// `models rm --all` wipes every cached pair and reports the totals.
#[test]
fn rm_all() {
    let fetch = pivot_fetch();
    let dir = tmp_cache_dir();
    let _ = run_with_dir(&fetch, &["models", "add", "es", "fr"], &dir);

    assert_transcript(
        "models rm --all",
        &run_with_dir(&fetch, &["models", "rm", "--all"], &dir),
        &[
            "Removed en-fr (62 B)",
            "Removed es-en (62 B)",
            "[2 removed, 124 B reclaimed]",
        ],
    );
    assert_transcript(
        "models list (after rm --all)",
        &run_with_dir(&fetch, &["models", "list"], &dir),
        &[
            "Cache: <CACHE>",
            "No models cached yet. Add one with `fxtranslate models add <src> <trg>`.",
            "[0 cached]",
        ],
    );
}

/// `models rm --all` on an empty cache is a note.
#[test]
fn rm_all_empty() {
    assert_transcript(
        "models rm --all (empty)",
        &models(&MockFetch::new(), &["models", "rm", "--all"]),
        &["No models cached; nothing to remove."],
    );
}

/// `models` with no subcommand prints the models usage.
#[test]
fn models_help() {
    let got = models(&MockFetch::new(), &["models"]);
    assert!(
        got.starts_with("fxtranslate models — manage locally cached models"),
        "models help header, got: {got}"
    );
}
