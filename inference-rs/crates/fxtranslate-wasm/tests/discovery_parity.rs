//! Proves the wasm-exposed discovery wrappers produce the same routing / catalog /
//! record decisions as the native core, on a fixed committed record set.
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

/// Segmentation splits multi-sentence input into its sentence contents.
#[test]
fn segment_sentences_splits_input() {
    let out = segment_sentences("Hello there. How are you?");
    assert_eq!(out, r#"["Hello there.","How are you?"]"#);
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
    fn segment_sentences_splits_input_wasm() {
        let out = segment_sentences("Hello there. How are you?");
        assert_eq!(out, r#"["Hello there.","How are you?"]"#);
    }

    #[wasm_bindgen_test]
    fn resolve_route_unreachable_errors_wasm() {
        assert!(resolve_route(FIXTURE, "es", "zu").is_err());
    }
}
