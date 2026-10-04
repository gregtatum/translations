#![cfg(not(target_arch = "wasm32"))] // native-only (filesystem fixtures)
//! The decoder's first step takes no embedding — only the positional encoding.
//!
//! marian builds the decoder input by shifting the target embeddings right and
//! zero-padding the vacated first slot (`shift(embeddings, {0, 1, 0})`), so at
//! step 0 there is no previous token to embed. Embedding the seed token instead
//! — the BOS convention other toolkits use — injects a vector roughly 1.9× the
//! norm of the positional encoding at position 0 into that step. The model
//! still picks the right *word*, but prefers its lowercase form, so the symptom
//! is a sentence that reads correctly except for its capitalization (and, once
//! the first token shifts, occasional reordering downstream).
//!
//! This was worth 0/20 → 18/20 greedy exact-match against `translator-cli` on
//! en-ru, and 6/20 → 19/20 on en-es, so it is pinned here: the failure is quiet
//! and plausible-looking, never a crash.

use fxtranslate::engine::Engine;

const EN_RU_MODEL: &str = "../../../data/models/enru/model.enru.intgemm.alphas.bin";
const EN_RU_VOCAB: &str = "../../../data/models/enru/vocab.enru.spm";

fn en_ru() -> Option<Engine> {
    if !std::path::Path::new(EN_RU_MODEL).exists() {
        eprintln!("skipping decoder-seed check: en-ru model absent");
        return None;
    }
    Some(Engine::load(EN_RU_MODEL, EN_RU_VOCAB, EN_RU_VOCAB).expect("engine loads"))
}

/// Greedy output must match `translator-cli` on sentences whose first token is
/// where the two used to diverge. Each of these was wrong before the fix — in
/// capitalization, in word order, or by dropping the leading pronoun.
#[test]
fn matches_reference_on_sentence_initial_tokens() {
    let Some(engine) = en_ru() else { return };
    for (src, expect) in [
        // Was "сегодня погода хорошая." — right words, reordered and lowercased.
        ("The weather is nice today.", "Погода сегодня хорошая."),
        // Was "привет, мир." — a different, lowercase greeting.
        ("Hello world.", "Здравствуйте, мир."),
        // Was "люблю программирование." — leading pronoun dropped entirely.
        ("I love programming.", "Я люблю программирование."),
        // Was ", пожалуйста, закрой дверь." — output began with a comma.
        ("Please close the door.", "Пожалуйста, закройте дверь."),
        // Was "большое спасибо вам большое." — trailing repetition.
        ("Thank you very much.", "Большое спасибо."),
    ] {
        assert_eq!(engine.translate(src), expect, "greedy output for {src:?}");
    }
}

/// The mechanism, independent of any particular sentence and of any model's
/// vocabulary: if step 0 discards the embedding, then `decode_step` at position
/// 0 must produce *identical* output no matter which token is passed as `prev` —
/// the token simply is not read. At position 1 it must matter again.
///
/// This is the tightest statement of the change that the public API can make,
/// and it fails loudly if the seed embedding is ever folded back in.
#[test]
fn step_zero_ignores_the_previous_token_entirely() {
    let Some(engine) = en_ru() else { return };
    let src = engine.src_ids("The weather is nice today.");
    let seq = src.len();
    let context = engine.encode(&src);
    let dim = context.len() / seq;
    let depth = 8; // more SSRU cell slots than any shipped decoder needs

    let step = |prev: u32, pos: usize| {
        let mut cells = vec![vec![0.0f32; dim]; depth];
        engine.decode_step(prev, pos, &context, seq, &mut cells)
    };

    // Position 0: the token is never read, so every seed gives the same state.
    let from_eos = step(0, 0);
    for other in [1u32, 273, 1273, 27579] {
        assert_eq!(
            step(other, 0),
            from_eos,
            "decoder step 0 must ignore prev_id, but {other} changed the output \
             — the seed embedding is being folded in again"
        );
    }

    // Position 1: the previous token is embedded, so it must change the output.
    let a = step(1273, 1);
    let b = step(27579, 1);
    assert_ne!(a, b, "decoder step 1 must depend on the previous token");
}
