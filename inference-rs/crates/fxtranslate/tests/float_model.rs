#![cfg(not(target_arch = "wasm32"))] // native-only (filesystem fixtures)
//! Float32 (`marian-conv --gemm-type float32`) model support.
//!
//! The failure this guards against is silent: the float container stores an
//! affine weight `[K, N]` while the int8 container stores the same weight
//! `[N, K]` (marian's float save branch neither packs nor transposes). Reading
//! one as the other does not crash and does not produce obvious noise — it
//! produces fluent, wrong output. So the orientation is pinned here against
//! hand-computed values on a non-square weight, where the two readings cannot
//! coincide.
//!
//! [`float_matches_int8_on_the_same_model`] is the end-to-end counterpart and
//! skips unless both en-ru containers are present locally.

use fxtranslate::weights::{Precision, Weights};

const F32_MODEL: &str = "../../../data/models/enru-f32/model.enru.float32.bin";
const INT8_MODEL: &str = "../../../data/models/enru/model.enru.intgemm.alphas.bin";

/// Build a minimal float32 marian container in memory: a `[4, 2]` embedding, one
/// non-square `[2, 3]` affine weight, and its `[1, 3]` bias.
fn synthetic_float_model(w: &[f32], bias: &[f32]) -> Vec<u8> {
    let wemb: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
    let items: [(&str, &[i32], &[f32]); 3] = [
        ("Wemb", &[4, 2], &wemb),
        ("W", &[2, 3], w),
        ("b", &[1, 3], bias),
    ];

    let mut out = Vec::new();
    out.extend_from_slice(&1u64.to_le_bytes()); // version
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for (name, shape, data) in &items {
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(&0x404u64.to_le_bytes()); // Float32
        out.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        out.extend_from_slice(&((data.len() * 4) as u64).to_le_bytes());
    }
    for (name, ..) in &items {
        out.extend_from_slice(name.as_bytes());
    }
    for (_, shape, _) in &items {
        for &d in *shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
    out.extend_from_slice(&0u64.to_le_bytes()); // no alignment padding
    for (_, _, data) in &items {
        for &v in *data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[test]
fn detects_float_precision() {
    let bytes = synthetic_float_model(&[1.0; 6], &[0.0; 3]);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");
    assert_eq!(weights.precision(), Precision::Float32);
}

#[test]
fn affine_reads_the_weight_as_k_by_n() {
    // W is [K=2, N=3] row-major, so W[k, n] == w[k * 3 + n]:
    //     W = [[1, 2, 3],
    //          [4, 5, 6]]
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [10.0f32, 20.0, 30.0];
    let bytes = synthetic_float_model(&w, &bias);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");

    // x = [1, 1] over k=2, so each output is a column sum of W plus the bias.
    let out = weights.affine("W", &[1.0, 1.0], 1, Some("b"));
    assert_eq!(
        out,
        vec![1.0 + 4.0 + 10.0, 2.0 + 5.0 + 20.0, 3.0 + 6.0 + 30.0]
    );

    // x = [1, 0] selects row 0 of W — the reading that a [N, K] interpretation
    // would get wrong (it would pick up [1, 2] instead of [1, 2, 3]).
    let out = weights.affine("W", &[1.0, 0.0], 1, Some("b"));
    assert_eq!(out, vec![11.0, 22.0, 33.0]);

    // x = [0, 1] selects row 1.
    let out = weights.affine("W", &[0.0, 1.0], 1, Some("b"));
    assert_eq!(out, vec![14.0, 25.0, 36.0]);
}

#[test]
fn affine_batches_rows_independently() {
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [10.0f32, 20.0, 30.0];
    let bytes = synthetic_float_model(&w, &bias);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");

    let batched = weights.affine("W", &[1.0, 0.0, 0.0, 1.0], 2, Some("b"));
    assert_eq!(batched, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn affine_without_a_bias_name_adds_nothing() {
    // The int8 path uses a "fake bias" carrying the +127 shift correction; the
    // float path has no shift, so `None` really must mean a zero bias.
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = synthetic_float_model(&w, &[10.0, 20.0, 30.0]);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");

    let out = weights.affine("W", &[1.0, 0.0], 1, None);
    assert_eq!(out, vec![1.0, 2.0, 3.0]);
}

#[test]
fn embedding_rows_are_vocab_by_dim() {
    // `Wemb` is [vocab, dim] row-major in *both* containers — the one weight the
    // float path must NOT reorient.
    let bytes = synthetic_float_model(&[1.0; 6], &[0.0; 3]);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");

    // wemb is 0.00, 0.25, 0.50, ... as [4, 2], so row 2 starts at element 4.
    let mut row = [0.0f32; 2];
    weights.trg_embed_row_into(2, &mut row);
    assert_eq!(row, [1.0, 1.25]);
    weights.src_embed_row_into(0, &mut row);
    assert_eq!(row, [0.0, 0.25]);
}

#[test]
fn float_model_has_no_quant_multiplier() {
    // Drives the shortlist projection onto the float path rather than the int8
    // `SelectColumnsB` path, which a float model cannot run.
    let bytes = synthetic_float_model(&[1.0; 6], &[0.0; 3]);
    let weights = Weights::from_bytes(&bytes).expect("float model loads");
    assert_eq!(weights.output_wemb_qmult(), None);
}

/// End-to-end: the shipped en-ru int8 model and the float32 conversion of the
/// same checkpoint must agree. Their weights are the same to ~1e-4 (the student
/// was quantization-aware finetuned, so the float values already sit on the int8
/// grid), so the two affines differ only by *activation* quantization.
///
/// The gate is cosine similarity rather than an absolute tolerance. A transposed
/// weight yields an essentially uncorrelated vector (cosine ≈ 0), while the
/// quantization error that legitimately remains is a small perturbation along the
/// same direction (cosine ≈ 1) — and unlike an absolute bound, that holds
/// regardless of how the activation sits inside the quantizer's range.
#[test]
fn float_matches_int8_on_the_same_model() {
    let (Ok(f32_bytes), Ok(int8_bytes)) = (std::fs::read(F32_MODEL), std::fs::read(INT8_MODEL))
    else {
        eprintln!("skipping float/int8 cross-check: en-ru containers absent");
        return;
    };
    let fw = Weights::from_bytes(&f32_bytes).expect("float model loads");
    let iw = Weights::from_bytes(&int8_bytes).expect("int8 model loads");
    assert_eq!(fw.precision(), Precision::Float32);
    assert_eq!(iw.precision(), Precision::Int8);
    assert_eq!(fw.config().dim_emb, iw.config().dim_emb);

    // A fixed pseudo-random activation at a realistic post-layernorm scale
    // (±3). The int8 path's quantizer for this layer covers ±127/8.72 ≈ ±14.6
    // with a step of ~0.115, so this spans enough of the range that the result
    // is signal rather than rounding noise.
    let dim = fw.config().dim_emb;
    let x: Vec<f32> = (0..dim)
        .map(|i| (((i * 2_654_435_761_u64 as usize) % 2003) as f32 / 2003.0 - 0.5) * 6.0)
        .collect();

    for (base, bias) in [
        ("encoder_l1_self_Wq", "encoder_l1_self_bq"),
        ("encoder_l1_self_Wo", "encoder_l1_self_bo"),
        ("decoder_l1_context_Wk", "decoder_l1_context_bk"),
    ] {
        let fo = fw.affine(base, &x, 1, Some(bias));
        let io = iw.affine(base, &x, 1, Some(bias));
        assert_eq!(fo.len(), io.len(), "{base}: output lengths differ");

        let dot: f64 = fo.iter().zip(&io).map(|(a, b)| *a as f64 * *b as f64).sum();
        let nf: f64 = fo.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let ni: f64 = io.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (nf * ni);
        let rel = fo
            .iter()
            .zip(&io)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max)
            / nf
            * (dim as f64).sqrt();
        eprintln!("{base}: cosine {cosine:.6}, max rel err {rel:.4}");
        assert!(
            cosine > 0.999,
            "{base}: float and int8 affines are not the same transform \
             (cosine {cosine:.6}) — suspect a weight-orientation error"
        );
    }
}

/// The `[K, N]` float layout really is the transpose of the `[N, K]` int8 one:
/// reading a weight with the wrong orientation gives something *uncorrelated*,
/// not something subtly off. This is what makes the cosine gate above meaningful
/// rather than vacuous — and it is why a square weight is the dangerous case,
/// since there the wrong orientation still has a valid shape.
#[test]
fn transposed_orientation_would_be_caught() {
    let dim = 512;
    // A fixed pseudo-random weight and activation at the real model's dims.
    let lcg = |mut z: u64| {
        move || {
            z = z.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((z >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        }
    };
    let mut rw = lcg(0x5DEECE66);
    let w: Vec<f32> = (0..dim * dim).map(|_| rw() * 0.2).collect();
    let mut rx = lcg(12345);
    let x: Vec<f32> = (0..dim).map(|_| rx() * 6.0).collect();
    let zero = vec![0.0f32; dim];

    let correct = fxtranslate::ops::affine_f32(&x, 1, dim, &w, dim, &zero);

    let mut swapped = vec![0.0f32; dim * dim];
    for k in 0..dim {
        for n in 0..dim {
            swapped[n * dim + k] = w[k * dim + n];
        }
    }
    let transposed = fxtranslate::ops::affine_f32(&x, 1, dim, &swapped, dim, &zero);

    let dot: f64 = correct
        .iter()
        .zip(&transposed)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let nc: f64 = correct
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nt: f64 = transposed
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (nc * nt);
    eprintln!("transposed-vs-correct cosine: {cosine:.6}");
    assert!(
        cosine.abs() < 0.2,
        "a transposed read is too similar to the correct one (cosine {cosine:.6}) \
         for the orientation gate to be meaningful"
    );
}

/// `project_f32_into` reads `Wemb` as `[vocab, dim]` — row `v` is token `v`'s
/// embedding — and must match a hand-rolled dot product per row.
#[test]
fn float_projection_reads_wemb_as_vocab_by_dim() {
    let (dim, vocab) = (3usize, 4usize);
    let wemb: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
    let bias: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];
    let h: Vec<f32> = vec![1.0, 2.0, 3.0];

    let mut out = Vec::new();
    fxtranslate::ops::project_f32_into(&h, 1, dim, &wemb, vocab, &bias, &mut out);

    let expect: Vec<f32> = (0..vocab)
        .map(|v| (0..dim).map(|c| h[c] * wemb[v * dim + c]).sum::<f32>() + bias[v])
        .collect();
    assert_eq!(out, expect);
    // Row 1 is [3, 4, 5] -> 1*3 + 2*4 + 3*5 = 26, minus 0.5.
    assert_eq!(out[1], 25.5);
}
