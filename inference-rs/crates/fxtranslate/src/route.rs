//! Language-pair routing: turn a requested `src`→`trg` into the model(s) that
//! realize it. Firefox Translations ships one-way models that all go to or from a
//! hub language (English in production), so a non-hub pair like `es`→`fr` has no
//! direct model — it is served by **pivoting**: run `es`→`en`, then feed that text
//! into `en`→`fr`. Marian has no pivot logic; Firefox orchestrates it outside the
//! engine, and [`Route`] is where fxtranslate makes that decision.
//!
//! The hub is discovered from the record set, not hardcoded: [`resolve_route`]
//! finds any language that bridges the pair and only *prefers* `en` when several
//! qualify, so a future collection with a different pivot keeps working.

use std::collections::BTreeSet;

use crate::remote::{pairs, pick, Record};

/// The hub language preferred when more than one could bridge a pair. English is
/// the production hub, but [`resolve_route`] falls back to any bridge, so this is a
/// tie-breaker, not an assumption.
pub const PREFERRED_HUB: &str = "en";

/// How to realize a `src`→`trg` request against the available models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// A single model translates the pair directly.
    Direct { src: String, trg: String },
    /// No direct model; pivot `src`→`pivot`→`trg` through two models.
    Pivot {
        src: String,
        pivot: String,
        trg: String,
    },
}

/// Resolve `src`→`trg` to a [`Route`] against `records`. A direct model always
/// wins; otherwise any hub `P` with models for both `src`→`P` and `P`→`trg` bridges
/// the pair (preferring [`PREFERRED_HUB`], else the lexicographically-first for
/// determinism). Errors when neither a direct model nor a pivot exists.
pub fn resolve_route(records: &[Record], src: &str, trg: &str) -> Result<Route, String> {
    if pick(records, "model", src, trg).is_some() {
        return Ok(Route::Direct {
            src: src.into(),
            trg: trg.into(),
        });
    }

    // Every candidate bridge language, sorted for determinism, minus the pair's
    // own endpoints. A hub bridges the pair when both legs have a model.
    let langs: BTreeSet<String> = pairs(records)
        .into_iter()
        .flat_map(|(s, t)| [s, t])
        .collect();
    let bridges: Vec<&String> = langs
        .iter()
        .filter(|p| p.as_str() != src && p.as_str() != trg)
        .filter(|p| {
            pick(records, "model", src, p).is_some() && pick(records, "model", p, trg).is_some()
        })
        .collect();

    let pivot = bridges
        .iter()
        .find(|p| p.as_str() == PREFERRED_HUB)
        .or_else(|| bridges.first())
        .ok_or_else(|| format!("no model or pivot route for {src}-{trg} in Remote Settings"))?;

    Ok(Route::Pivot {
        src: src.into(),
        pivot: (*pivot).clone(),
        trg: trg.into(),
    })
}

/// The languages a hub connects, split by which directions have a model. Drives
/// the `list` command's language-oriented view.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Catalog {
    /// Both `hub`→L and L→`hub` exist: fully capable — pivots to/from any other
    /// bidirectional language. Includes the hub itself.
    pub bidirectional: Vec<String>,
    /// Only L→`hub` exists: usable as a source, never a target.
    pub source_only: Vec<String>,
    /// Only `hub`→L exists: usable as a target, never a source.
    pub target_only: Vec<String>,
}

/// Classify every language reachable through `hub` (typically `en`) by the
/// directions it supports. Built from the supported `model` pairs, so it matches
/// what `translate` can actually load.
pub fn catalog(records: &[Record], hub: &str) -> Catalog {
    let ps = pairs(records);
    // L→hub (L can be a source) and hub→L (L can be a target).
    let into_hub: BTreeSet<&str> = ps
        .iter()
        .filter(|(_, t)| t == hub)
        .map(|(s, _)| s.as_str())
        .collect();
    let from_hub: BTreeSet<&str> = ps
        .iter()
        .filter(|(s, _)| s == hub)
        .map(|(_, t)| t.as_str())
        .collect();

    let mut cat = Catalog::default();
    // The hub is fully capable only when it genuinely translates both ways: some
    // hub→L (it can be a source) AND some L→hub (it can be a target). A gated-down
    // collection with only one direction must not advertise it as bidirectional.
    if !from_hub.is_empty() && !into_hub.is_empty() {
        cat.bidirectional.push(hub.to_string());
    }
    for lang in into_hub.union(&from_hub) {
        if *lang == hub {
            continue;
        }
        match (into_hub.contains(lang), from_hub.contains(lang)) {
            (true, true) => cat.bidirectional.push(lang.to_string()),
            (true, false) => cat.source_only.push(lang.to_string()),
            (false, true) => cat.target_only.push(lang.to_string()),
            (false, false) => unreachable!("lang came from a union of the two sets"),
        }
    }
    cat.bidirectional.sort();
    cat.source_only.sort();
    cat.target_only.sort();
    cat
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `model` record for a pair (routing ignores every other field).
    fn model(src: &str, trg: &str) -> Record {
        Record {
            name: format!("{src}{trg}"),
            file_type: "model".into(),
            src: src.into(),
            trg: trg.into(),
            version: "3.0".into(),
            architecture: None,
            decompressed_hash: None,
            location: String::new(),
        }
    }

    /// The English-hub world Firefox ships: en↔es, en↔fr.
    fn en_hub() -> Vec<Record> {
        vec![
            model("en", "es"),
            model("es", "en"),
            model("en", "fr"),
            model("fr", "en"),
        ]
    }

    #[test]
    fn direct_pair_resolves_directly() {
        let recs = en_hub();
        assert_eq!(
            resolve_route(&recs, "en", "es").unwrap(),
            Route::Direct {
                src: "en".into(),
                trg: "es".into()
            }
        );
    }

    #[test]
    fn non_hub_pair_pivots_through_english() {
        let recs = en_hub();
        assert_eq!(
            resolve_route(&recs, "es", "fr").unwrap(),
            Route::Pivot {
                src: "es".into(),
                pivot: "en".into(),
                trg: "fr".into()
            }
        );
    }

    #[test]
    fn direct_model_wins_over_an_available_pivot() {
        // Both a direct es→fr AND the pivot legs exist; the direct model must win.
        let mut recs = en_hub();
        recs.push(model("es", "fr"));
        assert_eq!(
            resolve_route(&recs, "es", "fr").unwrap(),
            Route::Direct {
                src: "es".into(),
                trg: "fr".into()
            }
        );
    }

    #[test]
    fn pivots_through_a_non_english_hub() {
        // No English anywhere: `a` and `b` only bridge through `x`. Proves the
        // resolver is generic in the hub, not hardcoded to English.
        let recs = vec![
            model("a", "x"),
            model("x", "a"),
            model("b", "x"),
            model("x", "b"),
        ];
        assert_eq!(
            resolve_route(&recs, "a", "b").unwrap(),
            Route::Pivot {
                src: "a".into(),
                pivot: "x".into(),
                trg: "b".into()
            }
        );
    }

    #[test]
    fn prefers_english_when_several_hubs_bridge() {
        // Both `en` and `zz` bridge es→fr; the preferred hub (en) is chosen.
        let recs = vec![
            model("es", "en"),
            model("en", "fr"),
            model("es", "zz"),
            model("zz", "fr"),
        ];
        match resolve_route(&recs, "es", "fr").unwrap() {
            Route::Pivot { pivot, .. } => assert_eq!(pivot, "en"),
            other => panic!("expected pivot, got {other:?}"),
        }
    }

    #[test]
    fn unreachable_pair_errors() {
        // `zu` has no model and bridges nothing.
        let err = resolve_route(&en_hub(), "es", "zu").unwrap_err();
        assert!(err.contains("es-zu"), "error names the pair: {err}");
    }

    #[test]
    fn catalog_classifies_directions() {
        // en↔es, en↔fr (bidirectional); en→nn only (target); is→en only (source).
        let recs = vec![
            model("en", "es"),
            model("es", "en"),
            model("en", "fr"),
            model("fr", "en"),
            model("en", "nn"),
            model("is", "en"),
        ];
        let cat = catalog(&recs, "en");
        assert_eq!(cat.bidirectional, vec!["en", "es", "fr"]);
        assert_eq!(cat.target_only, vec!["nn"]);
        assert_eq!(cat.source_only, vec!["is"]);
    }

    #[test]
    fn catalog_omits_hub_when_only_one_direction_exists() {
        // Only es→en: English can be a target but never a source, so it is NOT
        // advertised as fully supported; es is source-only.
        let cat = catalog(&[model("es", "en")], "en");
        assert!(
            cat.bidirectional.is_empty(),
            "hub not bidirectional one-way"
        );
        assert_eq!(cat.source_only, vec!["es"]);
        assert!(cat.target_only.is_empty());
    }
}
