//! Proves the wasm-exposed discovery wrappers produce the same routing / catalog /
//! record / **segmentation** decisions as the native core, on a fixed committed
//! record set and a small sentence corpus.
//!
//! Two runs, one fixture (`fixtures/rs-routing.json`, `include_str!`d so it works
//! under a wasm runtime that has no filesystem):
//!
//! - The **native** test (`cargo test -p fxtranslate-wasm`) checks the wrapper JSON
//!   against expectations derived *independently* from the core `resolve_route` /
//!   `catalog` structs — so it is not a tautology of "wrapper calls core", it pins
//!   that the JSON marshaling reflects the real routing decisions and the golden
//!   strings below are correct.
//! - The **wasm** test (`wasm-pack test --node`) asserts the wrappers, run live
//!   under wasm, emit those exact same golden JSON strings. Same golden on both
//!   targets is the wasm==native proof.

use fxtranslate_wasm::discovery::{
    catalog, model_pairs, parse_records, resolve_route, segment_sentences,
};

/// The committed record set: en↔es and en↔fr are bidirectional, en→nn is
/// target-only, is→en is source-only, and a v2 xx→yy record must be version-gated
/// out. Shared by both the native and the wasm test.
const FIXTURE: &str = include_str!("fixtures/rs-routing.json");

/// Golden JSON the wrappers must emit for this fixture. The native test proves
/// these describe the core's real decisions; the wasm test proves wasm reproduces
/// them byte-for-byte.
const ROUTE_DIRECT: &str = r#"{"kind":"direct","src":"en","trg":"es"}"#;
const ROUTE_PIVOT: &str = r#"{"kind":"pivot","src":"es","pivot":"en","trg":"fr"}"#;
const CATALOG: &str =
    r#"{"bidirectional":["en","es","fr"],"sourceOnly":["is"],"targetOnly":["nn"]}"#;
/// The version-gated, sorted, deduped `[src,trg]` pairs. The v2 `xx`→`yy` record is
/// gated out, so it does not appear — that gating is the whole reason `list --all`
/// calls the core rather than deriving pairs in JS.
const PAIRS: &str = r#"[["en","es"],["en","fr"],["en","nn"],["es","en"],["fr","en"],["is","en"]]"#;

/// The segmentation corpus and the ICU4X (UAX #29) JSON `segmentSentences` must
/// emit for each input. Chosen to exercise cases where ICU4X and the old
/// `BasicSegmenter` disagree — so this doubles as a regression guard that wasm now
/// segments with ICU4X, not the punctuation splitter:
///   - `U.S.` stays inside its sentence (Basic split it on the space after `S.`),
///   - `Wait...` keeps the ellipsis attached (Basic treated `…`/`...` as a hard stop),
///   - plus a plain multi-sentence Latin case, CJK fullwidth stops, and a
///     terminator-less tail that both engines handle the same.
/// The native test proves these goldens reproduce the core `IcuSegmenter`
/// (`segmentation_goldens_reflect_core_icu`); the wasm test proves wasm emits them
/// byte-for-byte — together, native ≡ wasm on the same input.
const SEG_CORPUS: [(&str, &str); 5] = [
    (
        "Hello there. How are you?",
        r#"["Hello there.","How are you?"]"#,
    ),
    (
        "The U.S. economy grew. Prices fell.",
        r#"["The U.S. economy grew.","Prices fell."]"#,
    ),
    ("Wait... really? Yes.", r#"["Wait... really?","Yes."]"#),
    ("你好世界。天气很好。", r#"["你好世界。","天气很好。"]"#),
    (
        "One sentence with no terminator",
        r#"["One sentence with no terminator"]"#,
    ),
];

fn route(src: &str, trg: &str) -> String {
    resolve_route(FIXTURE, src, trg)
        .ok()
        .expect("route resolves")
}

fn cat(hub: &str) -> String {
    catalog(FIXTURE, hub).ok().expect("catalog builds")
}

fn records() -> String {
    parse_records(FIXTURE).ok().expect("records parse")
}

fn pairs() -> String {
    model_pairs(FIXTURE).ok().expect("pairs build")
}

/// The direct pair `en`→`es` resolves to the direct-route JSON.
#[test]
fn resolve_route_direct_matches_golden() {
    assert_eq!(route("en", "es"), ROUTE_DIRECT);
}

/// The non-hub pair `es`→`fr` pivots through `en`, per the pivot-route JSON.
#[test]
fn resolve_route_pivot_matches_golden() {
    assert_eq!(route("es", "fr"), ROUTE_PIVOT);
}

/// The catalog classifies every language reachable through `en` by direction.
#[test]
fn catalog_matches_golden() {
    assert_eq!(cat("en"), CATALOG);
}

/// `parseRecords` returns a JSON array carrying the model records; the v2 `xx`→`yy`
/// record is version-gated out of routing (proven via the core API in
/// `goldens_reflect_core_decisions`, since the wrapper's error path builds a
/// `JsError` that only runs under wasm).
#[test]
fn parse_records_returns_json_array() {
    let json = records();
    assert!(json.starts_with('['));
    assert!(json.contains("\"en\""));
}

/// `modelPairs` returns the version-gated, sorted, deduped raw pairs — what
/// `list --all` renders. The v2 pair is gated out.
#[test]
fn model_pairs_matches_golden() {
    assert_eq!(pairs(), PAIRS);
}

/// `segmentSentences` splits every corpus input into the ICU4X golden JSON.
#[test]
fn segment_sentences_matches_corpus() {
    for (input, golden) in SEG_CORPUS {
        assert_eq!(segment_sentences(input), golden, "input: {input:?}");
    }
}

/// The golden strings are not hand-fabricated: rebuild the expectations straight
/// from the core routing API (bypassing the wasm wrappers) and confirm they agree
/// with what the wrappers emit. This is what makes the byte-exact wasm comparison
/// meaningful rather than circular. Native-only — it reaches into the core crate,
/// which the wasm test target reproduces via the identical wrapper output above.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn goldens_reflect_core_decisions() {
    use fxtranslate::remote::{pairs as core_pairs, parse_records as core_parse};
    use fxtranslate::route::{catalog as core_catalog, resolve_route as core_route, Route};

    let recs = core_parse(FIXTURE).expect("core parse");

    // The gated `xx`→`yy` v2 record is absent from the core pair set the wrapper mirrors.
    assert_eq!(
        core_pairs(&recs),
        vec![
            ("en".to_string(), "es".to_string()),
            ("en".to_string(), "fr".to_string()),
            ("en".to_string(), "nn".to_string()),
            ("es".to_string(), "en".to_string()),
            ("fr".to_string(), "en".to_string()),
            ("is".to_string(), "en".to_string()),
        ]
    );

    match core_route(&recs, "en", "es").expect("direct") {
        Route::Direct { src, trg } => {
            assert_eq!((src.as_str(), trg.as_str()), ("en", "es"));
        }
        other => panic!("expected direct, got {other:?}"),
    }
    match core_route(&recs, "es", "fr").expect("pivot") {
        Route::Pivot { src, pivot, trg } => {
            assert_eq!(
                (src.as_str(), pivot.as_str(), trg.as_str()),
                ("es", "en", "fr")
            );
        }
        other => panic!("expected pivot, got {other:?}"),
    }
    assert!(core_route(&recs, "es", "zu").is_err());

    let c = core_catalog(&recs, "en");
    assert_eq!(c.bidirectional, vec!["en", "es", "fr"]);
    assert_eq!(c.source_only, vec!["is"]);
    assert_eq!(c.target_only, vec!["nn"]);
}

/// The segmentation goldens aren't hand-fabricated: run the core `IcuSegmenter`
/// directly (bypassing the wasm wrapper) and confirm it produces exactly the
/// `SEG_CORPUS` goldens the wrapper is checked against. This is what makes the
/// wasm byte-exact comparison a real native≡wasm proof rather than circular.
/// Native-only — it reaches into the core crate; the wasm test reproduces the
/// identical wrapper output.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn segmentation_goldens_reflect_core_icu() {
    use fxtranslate::segment::{IcuSegmenter, Segmenter};

    for (input, golden) in SEG_CORPUS {
        let spans = IcuSegmenter::new().sentences(input);
        // Rebuild the same JSON-array shape the wrapper emits, independently.
        let parts: Vec<String> = spans.iter().map(|s| format!("{:?}", s.of(input))).collect();
        let rebuilt = format!("[{}]", parts.join(","));
        assert_eq!(rebuilt, golden, "core ICU output for {input:?}");
    }
}

/// The wasm run of the wrappers must emit the identical golden JSON the native run
/// pins above — that equality is the wasm==native guarantee for the discovery
/// surface. `wasm-pack test --node` executes this under a real wasm runtime.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn resolve_route_direct_matches_golden_wasm() {
        assert_eq!(route("en", "es"), ROUTE_DIRECT);
    }

    #[wasm_bindgen_test]
    fn resolve_route_pivot_matches_golden_wasm() {
        assert_eq!(route("es", "fr"), ROUTE_PIVOT);
    }

    #[wasm_bindgen_test]
    fn catalog_matches_golden_wasm() {
        assert_eq!(cat("en"), CATALOG);
    }

    #[wasm_bindgen_test]
    fn model_pairs_matches_golden_wasm() {
        assert_eq!(pairs(), PAIRS);
    }

    #[wasm_bindgen_test]
    fn segment_sentences_matches_corpus_wasm() {
        for (input, golden) in SEG_CORPUS {
            assert_eq!(segment_sentences(input), golden);
        }
    }

    #[wasm_bindgen_test]
    fn resolve_route_unreachable_errors_wasm() {
        assert!(resolve_route(FIXTURE, "es", "zu").is_err());
    }
}
