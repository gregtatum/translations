//! Alignment tensor parity against the marian reference trace (02 §5.1).
//!
//! Bergamot's token alignment is head 0 of the **last** decoder layer's
//! cross-attention softmax, `P(src | trg)`, captured per decode step
//! (`marian-fork/src/models/transformer.h`). [`Engine::translate_aligned`]
//! captures exactly that row into `Aligned::alignments[t]`. This test replays the
//! traced `Hello world.` run and asserts our captured rows match the trace's
//! corresponding `softmax` nodes:
//!
//! - **soft parity**: each row within a (loose, int8-aware) tolerance of the
//!   trace row — the strongest guarantee, catching head/layer/normalization
//!   mistakes before argmax collapses the row;
//! - **argmax parity**: the argmax source token per target token agrees exactly
//!   — the thing the HTML restorer (`html.cpp` `hardAlignments`) actually consumes.
//!
//! The trace records the cross-attention softmax as `[1, heads, q_len=1, kv_len]`
//! per decode step, `dec_depth` of them per step in layer order; the last of each
//! group is the last layer, and its first `kv_len` floats are head 0. Skips when
//! the model or trace are absent.

use fxtranslate::engine::Engine;
use fxtranslate::shortlist::Shortlist;
use fxtranslate::trace::Trace;
use fxtranslate_oracle::compare::{compare_f32, Tolerance};

const MODEL: &str = "../../../data/models/enfr/model.enfr.intgemm.alphas.bin";
const VOCAB: &str = "../../../data/models/enfr/vocab.enfr.spm";
const SHORTLIST: &str = "../../../data/models/enfr/lex.50.50.enfr.s2t.bin";
const TRACE_PATH: &str = "../../artifacts/enfr.trace";
/// The traced source sentence (see `real_trace.rs`).
const TRACE_TEXT: &str = "Hello world.";
/// Decoder depth of the en-fr model (4 SSRU layers).
const DEC_DEPTH: usize = 4;

fn engine() -> Option<Engine> {
    if !std::path::Path::new(MODEL).exists() || !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping alignment parity: model or vocab absent");
        return None;
    }
    let engine = Engine::load(MODEL, VOCAB, VOCAB)
        .expect("engine loads")
        .with_shortlist(Shortlist::load(SHORTLIST).expect("shortlist loads"));
    Some(engine)
}

/// Head-0 rows of the last decoder layer's cross-attention softmax, per decode
/// step, pulled from the trace. Cross-attention softmaxes have `q_len == 1`
/// (`shape[2] == 1`), distinguishing them from the encoder self-attention
/// (`q_len == seq`); they arrive `DEC_DEPTH` per step in layer order, so the last
/// of each group is the last layer. Head 0 is the first `kv_len` floats of the
/// `[1, heads, 1, kv_len]` tensor.
fn trace_last_layer_head0(trace: &Trace) -> Vec<Vec<f32>> {
    let cross: Vec<&fxtranslate::trace::TraceRecord> = trace
        .records
        .iter()
        .filter(|r| r.op_type == "softmax" && r.shape.get(2) == Some(&1))
        .collect();
    assert!(
        !cross.is_empty() && cross.len() % DEC_DEPTH == 0,
        "expected a whole number of {DEC_DEPTH}-layer decode steps, got {} cross softmaxes",
        cross.len()
    );
    cross
        .chunks(DEC_DEPTH)
        .map(|group| {
            let last = group.last().unwrap();
            let kv = *last.shape.last().unwrap() as usize;
            last.to_f32().expect("softmax is float32")[..kv].to_vec()
        })
        .collect()
}

#[test]
fn alignment_matches_trace_soft_and_argmax() {
    let Some(engine) = engine() else { return };
    if !std::path::Path::new(TRACE_PATH).exists() {
        eprintln!("skipping alignment parity: {TRACE_PATH} absent (record with task rs:translate-reference -- en fr --text \"Hello world.\" --cpu-threads 1 --trace)");
        return;
    }
    let trace = Trace::load(TRACE_PATH).expect("trace parses");

    let aligned = engine.translate_aligned(TRACE_TEXT);
    // Anchor: same translation and token sequence as the plain path / reference.
    assert_eq!(aligned.target_text, "Bonjour le monde.");
    // 4 source tokens incl. EOS (▁Hello ▁world . </s>) == the trace's kv_len.
    assert_eq!(aligned.source_tokens.len(), 4);

    let expected = trace_last_layer_head0(&trace);
    // Our rows include the terminal-EOS target row; marian records a softmax for
    // the EOS-producing step too, so the counts match.
    assert_eq!(
        aligned.alignments.len(),
        expected.len(),
        "decode-step count: ours {} vs trace {}",
        aligned.alignments.len(),
        expected.len()
    );

    // Soft parity. Each capture goes through the *same* `multihead` arithmetic
    // the graph-replay oracle already validates against this trace end-to-end, so
    // the row is trusted to op tolerance; the residual gap here is that the
    // softmax's exponential amplifies the int8 GEMM's sub-ULP score error into a
    // few percent of probability mass on the sharpest rows (peaks near 1.0). The
    // load-bearing quantity — the argmax and its dominant mass — matches tightly;
    // the whole-row atol just has to admit the amplified tail.
    let row_tol = Tolerance::new(2e-2, 2e-2);
    let mut worst = 0.0f32;
    for (t, (ours, want)) in aligned.alignments.iter().zip(&expected).enumerate() {
        // Each row is a valid distribution over the source (incl. EOS).
        let sum: f32 = ours.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {t} is not a probability distribution (sum {sum})"
        );

        let cmp = compare_f32(ours, want, row_tol).expect("row lengths match");
        worst = worst.max(cmp.max_abs_err);
        assert!(
            cmp.all_close(),
            "soft alignment row {t} diverged from trace: {cmp}\n  ours={ours:?}\n  want={want:?}"
        );

        // Argmax parity — the hard alignment the HTML restorer consumes — must be
        // exact, and its peak must agree tightly (it carries the alignment mass).
        let peak = argmax(want);
        assert_eq!(
            argmax(ours),
            peak,
            "argmax source token for target token {t}\n  ours={ours:?}\n  want={want:?}"
        );
        assert!(
            (ours[peak] - want[peak]).abs() <= 2.5e-2,
            "row {t} peak mass diverged: ours {} vs trace {}",
            ours[peak],
            want[peak]
        );
    }
    eprintln!(
        "alignment parity: {} rows, worst abs err {worst:e} (argmax exact on all rows)",
        aligned.alignments.len()
    );
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
        .unwrap_or(0)
}
