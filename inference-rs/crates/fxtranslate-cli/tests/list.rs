//! End-to-end `list` tests: real argv (`fxtranslate list [lang] [--all]`) driven
//! through `cli::run` against checked-in Remote Settings snapshots via the
//! mockable `Fetch` trait — no network, no engine.
//!
//! Each test is a **visible transcript snapshot**: the expected block is the
//! literal CLI output (the language view or the raw `--all` pair table on stdout,
//! plus the `[N …]` trailer on stderr), so a reviewer can audit formatting —
//! columns, display names, sort order, fallbacks — by scanning the code. On a
//! mismatch the helper prints the actual output as a paste-ready array to drop in.
//!
//! Two fixtures: `rs-list.json` is fully bidirectional (normal pairs `es`/`fr`,
//! Chinese script tags, Norwegian incl. the `nn` code fallback), so it exercises
//! the default "fully supported" view and `--all`. `rs-single-direction.json` adds
//! a target-only (`en → nn`) and a source-only (`is → en`) language, so the
//! "single-direction only" section renders.

use std::path::PathBuf;

use fxtranslate::remote::records_url;
use fxtranslate_cli::cli::Deps;

mod common;
use common::{assert_transcript, run_transcript, MockFetch, MockTranslator, Streams};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// Run `list` argv against `fixture_name`, returning the combined transcript.
/// `color_tty` sets stdout as a terminal (the only thing that enables color).
fn list_against(fixture_name: &str, args: &[&str], color_tty: bool) -> String {
    let fetch = MockFetch::new().route(&records_url(), fixture(fixture_name));
    let translator = MockTranslator::new(); // unused by `list`
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };
    run_transcript(
        args,
        &deps,
        Streams {
            stdout_tty: color_tty,
            ..Default::default()
        },
    )
}

/// The common case: the fully-bidirectional `rs-list.json`.
fn list(args: &[&str], color_tty: bool) -> String {
    list_against("rs-list.json", args, color_tty)
}

/// The default (language-oriented) view.
mod languages {
    use super::*;

    /// Every language is bidirectional here, so all land under "Fully supported",
    /// code-sorted, one row each (English included as the hub). Names align to the
    /// widest ("Chinese (Traditional)"); the `nn` code fallback shows.
    #[test]
    fn fully_supported() {
        assert_transcript(
            "list",
            &list(&["list"], false),
            &[
                "Fully supported (translate to and from any other):",
                "  English               (en)",
                "  Spanish               (es)",
                "  French                (fr)",
                "  Norwegian Bokmål      (nb)",
                "  nn                    (nn)",
                "  Chinese (Simplified)  (zh-Hans)",
                "  Chinese (Traditional) (zh-Hant)",
                "[7 languages]",
            ],
        );
    }

    /// A bare language filters to just that language (prefix on the code).
    #[test]
    fn filter_one_language() {
        assert_transcript(
            "list es",
            &list(&["list", "es"], false),
            &[
                "Fully supported (translate to and from any other):",
                "  Spanish (es)",
                "[1 languages]",
            ],
        );
    }

    /// `zh` (prefix) surfaces both Chinese scripts.
    #[test]
    fn chinese_scripts() {
        assert_transcript(
            "list zh",
            &list(&["list", "zh"], false),
            &[
                "Fully supported (translate to and from any other):",
                "  Chinese (Simplified)  (zh-Hans)",
                "  Chinese (Traditional) (zh-Hant)",
                "[2 languages]",
            ],
        );
    }

    /// Both single-direction cases render: `en → nn` (target-only) and `is → en`
    /// (source-only), with `es` still fully supported. Single-direction rows pair
    /// the display names and close with a compact `(src trg)` tag.
    #[test]
    fn single_direction_sections() {
        assert_transcript(
            "list (single-direction fixture)",
            &list_against("rs-single-direction.json", &["list"], false),
            &[
                "Fully supported (translate to and from any other):",
                "  English (en)",
                "  Spanish (es)",
                "",
                "Single-direction models:",
                "English   → nn      (en nn)",
                "Icelandic → English (is en)",
                "[4 languages]",
            ],
        );
    }
}

/// The raw per-direction model pairs, behind `--all`.
mod all_pairs {
    use super::*;

    /// The whole table (no filter): sort order, every display name (incl. the `å`
    /// in Norwegian Bokmål, the `nn` code fallback, and the Chinese script names),
    /// five-column alignment, and the `[N pairs]` trailer — auditable at a glance.
    #[test]
    fn every_pair() {
        assert_transcript(
            "list --all",
            &list(&["list", "--all"], false),
            &[
                "English               (en)      → Spanish               (es)",
                "English               (en)      → French                (fr)",
                "English               (en)      → Norwegian Bokmål      (nb)",
                "English               (en)      → nn                    (nn)",
                "English               (en)      → Chinese (Simplified)  (zh-Hans)",
                "English               (en)      → Chinese (Traditional) (zh-Hant)",
                "Spanish               (es)      → English               (en)",
                "French                (fr)      → English               (en)",
                "Norwegian Bokmål      (nb)      → English               (en)",
                "nn                    (nn)      → English               (en)",
                "Chinese (Simplified)  (zh-Hans) → English               (en)",
                "Chinese (Traditional) (zh-Hant) → English               (en)",
                "[12 pairs]",
            ],
        );
    }

    /// A bare language surfaces both directions; equal-width names → no padding.
    #[test]
    fn language_both_directions() {
        assert_transcript(
            "list es --all",
            &list(&["list", "es", "--all"], false),
            &[
                "English (en) → Spanish (es)",
                "Spanish (es) → English (en)",
                "[2 pairs]",
            ],
        );
    }

    /// `zh-en`: the split query prefix-matches each half — src `zh*` (both
    /// scripts), trg `en` only.
    #[test]
    fn src_trg_pair() {
        assert_transcript(
            "list zh-en --all",
            &list(&["list", "zh-en", "--all"], false),
            &[
                "Chinese (Simplified)  (zh-Hans) → English (en)",
                "Chinese (Traditional) (zh-Hant) → English (en)",
                "[2 pairs]",
            ],
        );
    }
}

/// `list` non-happy paths: no-match error and color gating.
mod edges {
    use super::*;

    #[test]
    fn no_match_errors() {
        // No list, no trailer — just the error, on stderr.
        assert_transcript(
            "list xx",
            &list(&["list", "xx"], false),
            &[
                "fxtranslate: no languages match `xx` (try `fxtranslate list`, or `list --all` for raw pairs)",
            ],
        );
    }

    #[test]
    fn no_match_errors_all() {
        assert_transcript(
            "list xx --all",
            &list(&["list", "xx", "--all"], false),
            &["fxtranslate: no model pairs match `xx` (12 pairs available; try `fxtranslate list --all`)"],
        );
    }

    #[test]
    fn color_only_on_a_tty() {
        // Color is a pure add-on gated by stdout being a TTY (ANSI is illegible in a
        // snapshot, so this checks presence/absence rather than exact bytes).
        assert!(
            !list(&["list", "es"], false).contains('\x1b'),
            "no ANSI when stdout is not a TTY"
        );
        let colored = list(&["list", "es"], true);
        assert!(colored.contains("\x1b[36m"), "cyan language on a TTY");
        assert!(colored.contains("\x1b[0m"), "reset present");
    }
}

/// The model `version` gates the backend it's valid for: `list` only surfaces
/// languages this build can actually translate. The `rs-version-gate.json` fixture
/// pairs a supported `es → en` (v3) with a future-major `en → fr` (v100).
mod version_gate {
    use super::*;

    /// The v100 `en → fr` is gated out, so English is not bidirectional (it can
    /// only be a target here) and drops from "fully supported"; only the one-way
    /// `es → en` remains.
    #[test]
    fn hides_unsupported_major() {
        assert_transcript(
            "list version-gate",
            &list_against("rs-version-gate.json", &["list"], false),
            &[
                "Single-direction models:",
                "Spanish → English (es en)",
                "[1 languages]",
            ],
        );
    }
}
