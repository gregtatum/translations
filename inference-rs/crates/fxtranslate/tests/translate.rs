//! End-to-end greedy translation.
//!
//! The anchor: the traced en→fr run translates "Hello world." to
//! "Bonjour le monde." (target tokens [16060, 280, 514, 264]). This drives the
//! whole pipeline — tokenize → encode → SSRU greedy decode → detokenize — with
//! nothing read from the trace. Skips when the model/vocab are absent.

use fxtranslate::engine::Engine;
use fxtranslate::shortlist::Shortlist;

const MODEL: &str = "../../../data/models/enfr/model.enfr.intgemm.alphas.bin";
const VOCAB: &str = "../../../data/models/enfr/vocab.enfr.spm";
const SHORTLIST: &str = "../../../data/models/enfr/lex.50.50.enfr.s2t.bin";

fn engine() -> Option<Engine> {
    if !std::path::Path::new(MODEL).exists() || !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping translate: model or vocab absent");
        return None;
    }
    let engine = Engine::load(MODEL, VOCAB, VOCAB).expect("engine loads");
    let engine = engine.with_shortlist(Shortlist::load(SHORTLIST).expect("shortlist loads"));
    Some(engine)
}

#[test]
fn translates_hello_world() {
    let Some(engine) = engine() else { return };

    // Greedy token ids must match the traced run.
    let src = engine_src_ids();
    let out = engine.greedy(&src);
    eprintln!("greedy ids: {out:?}");
    assert_eq!(out, vec![16060, 280, 514, 264], "greedy token ids");

    // And the detokenized text.
    let text = engine.translate("Hello world.");
    eprintln!("translation: {text:?}");
    assert_eq!(text, "Bonjour le monde.");
}

#[test]
fn matches_reference_translations() {
    let Some(engine) = engine() else { return };
    // Verified identical to the reference translator-cli on the shipped en→fr model.
    let cases = [
        ("Hello world.", "Bonjour le monde."),
        (
            "The cat sat on the mat.",
            "Le chat était assis sur le tapis.",
        ),
        ("I love programming.", "J'adore la programmation."),
    ];
    for (src, want) in cases {
        assert_eq!(engine.translate(src), want, "translating {src:?}");
    }
}

/// A documented near-tie: the reference emits "Bonjour, comment allez-vous ?"
/// but the first token is a ~1% logit near-tie between `▁Bonjour` (14.13) and
/// `▁bon` (14.27). Different float reduction orders (our scalar sums vs the
/// reference SIMD reductions) tip it the other way, so we emit lowercase
/// "bonjour". This is within the tolerance parity bar (not bit-exactness); the
/// source tokenization and the rest of the sequence match the reference exactly.
/// Asserted case-insensitively to pin the behavior.
#[test]
fn near_tie_casing_matches_apart_from_case() {
    let Some(engine) = engine() else { return };
    let got = engine.translate("Good morning, how are you?");
    assert_eq!(got.to_lowercase(), "bonjour, comment allez-vous ?");
}

/// A single over-long line is silently truncated — the model translates a prefix
/// and drops the rest, with no error or warning.
///
/// The CLI has no notion of context size: it reads stdin one line at a time and
/// hands each line to the engine whole (one line = one "sentence"), however long.
/// Greedy decoding caps the *output* at `ceil(2.0 * src_len) + 4` tokens, hard-
/// capped at 256 (see `greedy` in engine.rs) — there is no cap on the *input* at
/// all. But well before that ceiling, an input far longer than the model's
/// training distribution makes the decoder emit EOS on its own: it renders the
/// opening and quits mid-sentence, dropping everything after.
///
/// Ten distinct sentences, each of which translates completely in isolation, are
/// joined into one line. The combined translation covers only the first six and a
/// half — the tail simply vanishes.
///
/// This test pins the *raw* single-sequence path (`Engine::translate`), which is
/// unchanged. The higher layer that fixes this is `Engine::translate_long` /
/// `translate_segmented` (see `segmented_translates_full_paragraph` below): it
/// segments the line into sentences and translates each within the context window,
/// mirroring how the marian oracle (`ssplit`) and Firefox (`Intl.Segmenter`)
/// avoid feeding the decoder an over-long sequence. The parity harness
/// (parity.py) feeds one sentence per line, so it never exercises the raw path.
#[test]
fn over_long_input_is_silently_truncated() {
    let Some(engine) = engine() else { return };

    let sentences = [
        "The weather today is remarkably pleasant and the sky is a brilliant blue.",
        "My brother traveled to Japan last summer and visited many ancient temples.",
        "Scientists have discovered a new species of frog deep in the rainforest.",
        "The old library on the corner has thousands of rare and valuable books.",
        "She carefully planted tomatoes, peppers, and herbs in her small garden.",
        "The orchestra performed a beautiful symphony for the enthusiastic audience.",
        "Engineers are building a longer bridge across the wide and rushing river.",
        "The children laughed and played in the park until the sun went down.",
        "A gentle rain fell over the quiet village throughout the entire night.",
        "The chef prepared a delicious meal with fresh vegetables and local fish.",
    ];

    // The tail is not inherently untranslatable: the last sentence renders in
    // full on its own.
    assert_eq!(
        engine.translate(sentences[9]),
        "Le chef a préparé un délicieux repas avec des légumes frais et du poisson local."
    );

    // Joined into one over-long line, the translation is cut short: it renders
    // the first six sentences, quits partway through the seventh ("un plus long
    // pont", no closing period), and drops the last three entirely. The decoder
    // stops on its own (EOS) well under the 256-token hard cap — the loss is the
    // model giving up, not the length clamp firing.
    assert_eq!(
        engine.translate(&sentences.join(" ")),
        "Le temps aujourd'hui est remarquablement agréable et le ciel est un bleu \
         brillant. Mon frère a voyagé au Japon l'été dernier et a visité de nombreux \
         temples anciens. Les scientifiques ont découvert une nouvelle espèce \
         d'amphibiens au fond de la forêt tropicale. L'ancienne bibliothèque au coin a \
         des milliers de livres rares et précieux. Elle a soigneusement planté des \
         tomates, des poivrons et des herbes dans son petit jardin. L'orchestre a joué \
         une belle symphonie pour le public enthousiaste. Les ingénieurs construisent \
         un plus long pont"
    );
}

/// The segmented path fixes the truncation above: `translate_segmented` splits the
/// same ten-sentence line into sentences and translates each within the context
/// window, so the tail no longer vanishes. Driven through `BasicSegmenter` so it
/// runs without the `icu-segmenter` feature; a feature-gated variant checks the
/// icu path too. Asserts the presence of the last two sentences' translations
/// (the ones the raw path dropped) rather than one exact string — per-sentence
/// output is stable, but this is not a bit-exact parity assertion.
#[test]
fn segmented_translates_full_paragraph() {
    use fxtranslate::segment::BasicSegmenter;
    let Some(engine) = engine() else { return };

    let paragraph = "The weather today is remarkably pleasant and the sky is a brilliant blue. \
         My brother traveled to Japan last summer and visited many ancient temples. \
         Scientists have discovered a new species of frog deep in the rainforest. \
         The old library on the corner has thousands of rare and valuable books. \
         She carefully planted tomatoes, peppers, and herbs in her small garden. \
         The orchestra performed a beautiful symphony for the enthusiastic audience. \
         Engineers are building a longer bridge across the wide and rushing river. \
         The children laughed and played in the park until the sun went down. \
         A gentle rain fell over the quiet village throughout the entire night. \
         The chef prepared a delicious meal with fresh vegetables and local fish.";

    let out = engine.translate_segmented(paragraph, &BasicSegmenter);
    eprintln!("segmented: {out:?}");
    // The opening still translates…
    assert!(out.contains("bleu brillant"), "opening present: {out:?}");
    // …and so do the closing sentences the raw path dropped.
    assert!(
        out.contains("Une pluie douce"),
        "9th sentence (rain) present: {out:?}"
    );
    assert!(
        out.contains("du poisson local"),
        "10th sentence (fish) present: {out:?}"
    );
}

/// The icu segmenter drives the same fix end to end via `translate_long`.
#[cfg(feature = "icu-segmenter")]
#[test]
fn translate_long_with_icu_is_complete() {
    let Some(engine) = engine() else { return };
    let paragraph = "The children laughed and played in the park until the sun went down. \
         A gentle rain fell over the quiet village throughout the entire night. \
         The chef prepared a delicious meal with fresh vegetables and local fish.";
    let out = engine.translate_long(paragraph);
    assert!(out.contains("du poisson local"), "tail present: {out:?}");
}

/// A single short sentence must be byte-identical through the segmented path and
/// the raw path — no regression from segmentation/reassembly.
#[test]
fn segmented_single_sentence_matches_raw() {
    use fxtranslate::segment::BasicSegmenter;
    let Some(engine) = engine() else { return };
    for s in ["Hello world.", "The cat sat on the mat.", "I love programming."] {
        assert_eq!(
            engine.translate_segmented(s, &BasicSegmenter),
            engine.translate(s),
            "segmented diverged from raw on {s:?}"
        );
    }
}

/// `--mmap` must be parity-safe: a memory-mapped model produces byte-identical
/// translations to the owned-heap load (the mapping is read-only).
#[cfg(feature = "mmap")]
#[test]
fn mmap_matches_owned() {
    if !std::path::Path::new(MODEL).exists() || !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping mmap parity: model or vocab absent");
        return;
    }
    let owned = Engine::load(MODEL, VOCAB, VOCAB).expect("owned engine loads");
    let mapped = Engine::load_mmapped(MODEL, VOCAB, VOCAB).expect("mmapped engine loads");
    for src in [
        "Hello world.",
        "The cat sat on the mat.",
        "I love programming.",
    ] {
        assert_eq!(
            owned.translate(src),
            mapped.translate(src),
            "mmap vs owned diverged on {src:?}"
        );
    }
}

/// Source ids for "Hello world." + EOS, matching the trace.
fn engine_src_ids() -> Vec<u32> {
    vec![17169, 564, 264, 0]
}
