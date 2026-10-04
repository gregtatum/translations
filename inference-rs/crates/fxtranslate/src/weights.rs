//! Model-weights view for dynamic execution.
//!
//! Wraps [`crate::model::Model`] and resolves parameters by marian's naming
//! convention, exposing exactly what the transformer forward needs:
//! - [`Weights::affine`] runs `y = x·W + bias` from a weight's base name. For an
//!   int8 model that is the shifted int8 affine end to end (quantize the
//!   activation, prepare the bias, integer GEMM, unquantize) — the
//!   `unquant = 1/(qA·qB)` multiplier is computed from the model's own quant
//!   multipliers, nothing from the trace. For a float32 model it is a plain f32
//!   GEMM with no quantization anywhere.
//! - [`Weights::src_embed_row_into`] / [`Weights::trg_embed_row_into`] write one
//!   embedding row, and [`Weights::full_logits`] the tied output projection. Their backing
//!   representation ([`Embed`]) is chosen at load from the model's dtype and the
//!   `lean-embed` feature.
//! - [`Weights::f32`] returns float parameters (biases, layernorm scale/bias).
//! - [`Config`] holds the architecture dims parsed from `special:model.yml`.
//!
//! # The two containers
//!
//! Firefox ships `*.intgemm.alphas.bin` (`marian-conv --gemm-type intgemm8`);
//! `marian-conv --gemm-type float32` writes the same model unquantized. The two
//! differ in exactly three ways, and [`Precision`] keys off the third:
//!
//! 1. **Weight dtype** — `intgemm8` (`0x4101`) vs `float32` (`0x404`). Biases and
//!    layernorm parameters are float32 and bit-identical in both.
//! 2. **No `*_QuantMultA`** — the float container has none (nor `none_QuantMultA`),
//!    so a float affine must not look for them.
//! 3. **Weight orientation** — the int8 branch runs `PrepareBTransposed`, storing a
//!    logically-`[K, N]` weight as `[N, K]`; the float branch is a bare
//!    `val->get(item, pName)` (`expression_graph_packable.h`) that neither packs
//!    nor transposes, leaving it `[K, N]`. `Wemb` is the exception: it is
//!    `[vocab, dim]` row-major in *both*, because its declared shape is already
//!    intgemm's `Bᵀ` form.
//!
//! Parameter names, shapes, and `special:model.yml` are otherwise identical, so
//! nothing above `Weights` has to know which container it got.

use std::collections::HashMap;

use crate::model::Model;
use crate::ops;
use crate::trace::DType;
// Unconditional: `FloatWeight` caches its bias in a `OnceLock` in every build.
use std::sync::OnceLock;
#[cfg(fast_gemm)]
use {crate::gemm::PreparedB, std::cell::RefCell};

/// Which numeric form the model's GEMM weights shipped in. Fixed at load and
/// reported by [`Weights::precision`] so a caller can state which path actually
/// ran rather than inferring it from the filename.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    /// `marian-conv --gemm-type intgemm8` — the shipped Firefox models.
    Int8,
    /// `marian-conv --gemm-type float32` — unquantized reference models.
    Float32,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Precision::Int8 => "int8",
            Precision::Float32 => "float32",
        })
    }
}

/// A gemmology-prepared affine weight. Built once at load; the raw int8 bytes are
/// then freed from the model (the packed form is all the GEMM needs), so each
/// weight is held once, not twice.
#[cfg(fast_gemm)]
struct AffineWeight {
    pb: PreparedB,
    /// Shift correction `-127·unquant·colsum(W)` (bias-independent), length `n`.
    correction: Vec<f32>,
    /// Full prepared bias `correction + raw_bias`, computed on first use (the bias
    /// name is known only at call time). A `OnceLock` rather than a `RefCell`-guarded
    /// `Option` so the cache is `Sync` — many threads may share these weights and
    /// race to initialize the bias; `get_or_init` makes that safe and idempotent
    /// (the value is a pure function of the immutable correction + raw bias).
    bias: OnceLock<Vec<f32>>,
    qa: f32,
    unquant: f32,
}

/// A float32 affine weight, decoded once at load into an owned `[k, n]`
/// row-major table — the orientation the float container stores (see the module
/// note on orientation; this is the transpose of the int8 layout).
///
/// Decoded eagerly rather than read per call because [`crate::model::Bytes`] may
/// be a memory-mapped view with no alignment guarantee, so the f32 values cannot
/// simply be reinterpreted in place. The model's raw bytes are freed once
/// decoded, so each weight stays resident exactly once.
struct FloatWeight {
    /// `[k, n]` row-major.
    w: Vec<f32>,
    k: usize,
    n: usize,
    /// Raw bias, resolved on first use (the bias name is known only at call time)
    /// and cached so a steady-state affine decodes nothing. `OnceLock` for the
    /// same reason as [`AffineWeight::bias`] — the weights are shared across
    /// threads and the value is a pure function of the immutable model.
    bias: OnceLock<Vec<f32>>,
}

/// Reusable scratch for the shifted int8 affine, so the hot path allocates no
/// per-call activation buffers. Lives in a thread-local (see `GEMM_SCRATCH`), not
/// in `Weights`, so the weights stay immutable/shareable and each worker thread
/// gets its own buffers with no contention.
#[cfg(fast_gemm)]
#[derive(Default)]
struct GemmScratch {
    a_u8: Vec<u8>,
    /// One int8 embedding row gathered out of the packed projection buffer, reused
    /// across lookups so the on-demand dequant path allocates nothing. Only the
    /// `lean-embed` on-demand dequant reads rows back out of the packed buffer.
    #[cfg(feature = "lean-embed")]
    wemb_row: Vec<i8>,
}

#[cfg(fast_gemm)]
thread_local! {
    /// Per-thread activation scratch for the affine/projection hot path. Thread-local
    /// (not a `Weights` field) so `Weights` is immutable and shareable across threads;
    /// each thread reuses its own buffers, keeping the single-thread zero-per-call-alloc
    /// behaviour while never sharing a buffer between threads.
    static GEMM_SCRATCH: RefCell<GemmScratch> = RefCell::new(GemmScratch::default());
}

/// Architecture hyperparameters read from the embedded `special:model.yml`.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim_emb: usize,
    pub heads: usize,
    pub enc_depth: usize,
    pub dec_depth: usize,
    pub dim_ffn: usize,
    pub vocab: usize,
}

impl Config {
    fn parse(yaml: &str) -> Config {
        let get = |key: &str, default: usize| -> usize {
            for line in yaml.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix(key) {
                    if let Some(v) = rest.trim().strip_prefix(':') {
                        if let Ok(n) = v.trim().parse() {
                            return n;
                        }
                    }
                }
            }
            default
        };
        Config {
            dim_emb: get("dim-emb", 384),
            heads: get("transformer-heads", 8),
            enc_depth: get("enc-depth", 6),
            dec_depth: get("dec-depth", 4),
            dim_ffn: get("transformer-dim-ffn", 1536),
            // dim-vocabs is a YAML list; fall back to the embedding row count.
            vocab: get("dim-vocabs", 0),
        }
    }
}

/// `lean-embed` + an int8 model: no resident f32 tables. Embedding rows are
/// dequantized on demand and the output projection runs full-vocab in int8, so
/// only the int8 table (already in `model`) is resident.
#[cfg(feature = "lean-embed")]
struct LeanInt8Embed {
    /// Source embedding param name (`== trg_wemb_param` for shared vocab).
    src_param: &'static str,
    src_inv_qmult: f32,
    trg_inv_qmult: f32,
    /// Prepared bias for the full-vocab int8 output projection (static, so it is
    /// precomputed once here rather than per decode step).
    proj_bias: Vec<f32>,
    proj_qa: f32,
    proj_unquant: f32,
    /// The output-projection `Wemb` packed for the full-vocab GEMM. Built once
    /// eagerly at load (see [`Weights::new`]); read-only afterwards, so a plain
    /// `Option` — no `RefCell`. The raw `Wemb` is dropped when the pack succeeds
    /// (the packed layout serves both projection and row lookups).
    #[cfg(fast_gemm)]
    proj_pb: Option<PreparedB>,
}

/// Resident `[vocab, dim]` f32 embedding tables.
struct ResidentEmbed {
    /// The target embedding; also the tied output weight.
    trg: Vec<f32>,
    /// Source embedding; `None` for shared vocab (reuses `trg`).
    src: Option<Vec<f32>>,
    /// `decoder_ff_logit_out_b`, decoded once. The projection runs every decode
    /// step, and this is `vocab` long — decoding it per call would allocate and
    /// convert 32k floats per step.
    proj_bias: Vec<f32>,
}

/// How the embedding tables are held, chosen at load from the model's dtype and
/// the `lean-embed` feature.
enum Embed {
    /// Resident f32 tables. Used for **every float32 model** — where the tables
    /// are just the model's own data, so there is nothing for `lean-embed` to
    /// save — and, without `lean-embed`, for int8 models (dequantized once at
    /// load, ~49 MB/table, for fast lookups and a float full-vocab projection).
    Resident(ResidentEmbed),
    /// `lean-embed` + an int8 model. Big memory win; see [`LeanInt8Embed`].
    #[cfg(feature = "lean-embed")]
    LeanInt8(Box<LeanInt8Embed>),
}

/// Loaded model weights + parsed config.
pub struct Weights {
    model: Model,
    config: Config,
    precision: Precision,
    trg_vocab: usize,
    dim: usize,
    /// Decoded layer-norm `scale`/`bias` params, keyed by the sublayer base name
    /// (`{base}_ln_scale` → `(scale, bias?)`). Cached once at load so `postnorm`
    /// borrows them instead of decoding a fresh `Vec` from the model every call.
    layer_norms: HashMap<String, (Vec<f32>, Option<Vec<f32>>)>,
    /// Model param name of the target embedding (`Wemb` shared, `decoder_Wemb`
    /// split); the int8 output projection reads it back from the model on demand.
    trg_wemb_param: &'static str,
    /// Embedding/output-projection representation.
    embed: Embed,

    /// Float32 affine weights, decoded at load. Empty unless
    /// `precision == Float32`.
    float_affines: HashMap<String, FloatWeight>,

    /// Affine weights packed into gemmology's layout at load, keyed by param name.
    /// Their raw int8 bytes are freed from `model` once packed (no double copy).
    /// Immutable after load — each `AffineWeight` caches its prepared bias in its
    /// own `OnceLock`, so the map itself needs no interior mutability.
    #[cfg(fast_gemm)]
    affine_cache: HashMap<String, AffineWeight>,
}

/// Is this item an affine weight (as opposed to a bias, a layernorm parameter, a
/// quant multiplier, or the embedding)?
///
/// Keyed on shape rather than on name: every affine weight is 2-D with both dims
/// > 1 (`[512,512]`, `[512,2048]`, `[2048,512]`), while every bias, layernorm
/// parameter, and `*_QuantMultA` is `[1, N]`. That holds in both containers and
/// needs no list of name suffixes to drift out of date.
fn is_affine_weight(it: &crate::model::ModelItem, embed_param: &str) -> bool {
    it.name != embed_param
        && it.shape.len() == 2
        && it.shape[0] > 1
        && it.shape[1] > 1
        && !it.name.starts_with("special:")
}

/// Decode every float32 affine weight into an owned `[k, n]` table and free the
/// raw bytes from `model`, so each weight is resident once. Returns an empty map
/// for an int8 model (it has no such items — all its weights are `intgemm8` and
/// its float items are all `[1, N]`), which is exactly how [`Precision`] is
/// decided.
fn prepare_float_affines(model: &mut Model, embed_param: &str) -> HashMap<String, FloatWeight> {
    let mut cache = HashMap::new();
    for it in model.items.iter_mut() {
        if it.dtype != DType::Float32 || !is_affine_weight(it, embed_param) {
            continue;
        }
        let (k, n) = (it.shape[0] as usize, it.shape[1] as usize);
        let w = match it.to_f32() {
            Ok(w) if w.len() == k * n => w,
            _ => continue,
        };
        cache.insert(
            it.name.clone(),
            FloatWeight {
                w,
                k,
                n,
                bias: OnceLock::new(),
            },
        );
        // Release the raw bytes (owned copy, or the mmap view's Arc handle); the
        // decoded table is all the GEMM needs.
        it.data = crate::model::Bytes::Owned(Vec::new());
    }
    cache
}

/// Pack every affine weight (those with a `{name}_QuantMultA` sibling, excluding
/// the embedding) into gemmology's layout, cache the shift correction, and free
/// the raw int8 bytes from `model` — so each affine weight is resident once
/// (packed) rather than twice (raw + packed). Weights gemmology can't take
/// (`k % 16 != 0`) are left raw for the scalar fallback.
///
/// Only called for an int8 model; a float model never reaches here (it would
/// `continue` past every weight for want of a `_QuantMultA` and silently produce
/// an empty cache).
#[cfg(fast_gemm)]
fn prepare_affines(model: &mut Model, embed_param: &str) -> HashMap<String, AffineWeight> {
    let names: Vec<String> = model.items.iter().map(|it| it.name.clone()).collect();
    let mut cache = HashMap::new();
    let mut dropped: Vec<String> = Vec::new();
    for name in &names {
        if name.ends_with("_QuantMultA") || name == embed_param {
            continue;
        }
        let qa_name = format!("{name}_QuantMultA");
        let it = match model.get(name) {
            Some(it) if it.shape.len() >= 2 => it,
            _ => continue,
        };
        let (k, n) = (it.shape[0] as usize, it.shape[1] as usize);
        let b = match it.int8_transposed() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let qb = match it.quant_mult() {
            Ok(q) => q,
            Err(_) => continue,
        };
        let qa = match model.get(&qa_name).and_then(|i| i.to_f32().ok()) {
            Some(v) if !v.is_empty() => v[0],
            _ => continue, // no activation quant-mult -> not an affine weight
        };
        let pb = match PreparedB::new(b, n, k) {
            Some(pb) => pb,   // k % 16 == 0
            None => continue, // keep raw for the scalar path
        };
        let unquant = 1.0 / (qa * qb);
        let correction = ops::prepare_bias(b, n, k, &vec![0.0; n], unquant);
        cache.insert(
            name.clone(),
            AffineWeight {
                pb,
                correction,
                bias: OnceLock::new(),
                qa,
                unquant,
            },
        );
        dropped.push(name.clone());
    }
    // Free the raw bytes of everything we packed (the packed copy is all the GEMM
    // needs; the correction covers the bias).
    for it in model.items.iter_mut() {
        if dropped.iter().any(|d| d == &it.name) {
            // Release the raw bytes (owned copy, or the mmap view's Arc handle).
            it.data = crate::model::Bytes::Owned(Vec::new());
        }
    }
    cache
}

/// Load an embedding parameter into a resident `[vocab, dim]` f32 table: decoded
/// as-is from a float model, dequantized from int8/quantMult otherwise. `Wemb` is
/// `[vocab, dim]` row-major in both containers, so neither branch transposes.
fn load_embedding(model: &Model, name: &str) -> Result<Vec<f32>, String> {
    let item = model
        .get(name)
        .ok_or_else(|| format!("model has no {name}"))?;
    match item.dtype {
        DType::Float32 => item.to_f32().map_err(|e| e.to_string()),
        _ => {
            let inv = 1.0 / item.quant_mult().map_err(|e| e.to_string())?;
            let raw = item.int8_transposed().map_err(|e| e.to_string())?;
            Ok(raw.iter().map(|&b| b as f32 * inv).collect())
        }
    }
}

/// Activation quant-mult (qA) for the tied output projection. Shared-vocab models
/// name that node "none" (`none_QuantMultA`, a plain float32 scalar). Split-vocab
/// (CJK) models name it `decoder_Wemb_QuantMultA` and store it as an `intgemm8`
/// scalar: a single int8 value (127) plus an appended quant multiplier, whose
/// *dequantized* value (`127 / quant_mult`) is the alpha.
///
/// Int8 models only — a float container has no `*_QuantMultA` at all.
fn read_output_qa(model: &Model) -> f32 {
    if let Some(v) = model.get("none_QuantMultA").and_then(|it| it.to_f32().ok()) {
        return v[0];
    }
    if let Some(it) = model.get("decoder_Wemb_QuantMultA") {
        let raw = it.int8_transposed().expect("intgemm8 alpha scalar")[0] as f32;
        let qmult = it.quant_mult().expect("intgemm8 alpha quant mult");
        return raw / qmult;
    }
    panic!("model has no output-projection QuantMultA");
}

impl Weights {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Weights, String> {
        let model = Model::load(path).map_err(|e| e.to_string())?;
        Weights::new(model)
    }

    /// Like [`Weights::load`] but parses the model from an in-memory buffer: the
    /// weight tensors are copied to owned heap storage ([`Model::from_bytes`]), so
    /// the resulting `Weights` borrows nothing from `bytes`. This is the byte-path
    /// entry the wasm build uses, where model bytes come from the host rather than
    /// a file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Weights, String> {
        let model = Model::from_bytes(bytes).map_err(|e| e.to_string())?;
        Weights::new(model)
    }

    /// Like [`Weights::load`] but memory-maps the model file: weight tensors are
    /// views into the mapping rather than owned heap copies (feature `mmap`).
    ///
    /// A float32 model still decodes its weights into owned tables at load (the
    /// mapped bytes have no alignment guarantee), so `mmap` saves less there than
    /// it does for int8.
    #[cfg(feature = "mmap")]
    pub fn load_mmapped(path: impl AsRef<std::path::Path>) -> Result<Weights, String> {
        let model = Model::load_mmapped(path).map_err(|e| e.to_string())?;
        Weights::new(model)
    }

    pub fn new(mut model: Model) -> Result<Weights, String> {
        let yaml = model
            .get("special:model.yml")
            .map(|it| String::from_utf8_lossy(&it.data).into_owned())
            .unwrap_or_default();
        let mut config = Config::parse(&yaml);

        // Shared-vocab models (tied-embeddings-all) ship a single `Wemb` used for
        // source, target, and the output projection. Split-vocab models (CJK)
        // ship separate `encoder_Wemb` (source) and `decoder_Wemb` (target +
        // output projection).
        let (trg_wemb_param, src_wemb_param): (&'static str, &'static str) =
            if model.get("Wemb").is_some() {
                ("Wemb", "Wemb")
            } else {
                ("decoder_Wemb", "encoder_Wemb")
            };
        let trg_item = model
            .get(trg_wemb_param)
            .ok_or_else(|| format!("model has no {trg_wemb_param}"))?;
        let dim = *trg_item.shape.last().ok_or("embedding has no shape")? as usize;
        let trg_vocab = trg_item.num_elements() / dim;
        if config.vocab == 0 {
            config.vocab = trg_vocab;
        }

        // Cache the layer-norm scale/bias params (small, ~72 KB total) so the hot
        // path borrows them instead of decoding a fresh Vec per postnorm call.
        let mut layer_norms: HashMap<String, (Vec<f32>, Option<Vec<f32>>)> = HashMap::new();
        for it in &model.items {
            if let Some(base) = it.name.strip_suffix("_ln_scale") {
                if let Ok(v) = it.to_f32() {
                    layer_norms.entry(base.to_string()).or_default().0 = v;
                }
            } else if let Some(base) = it.name.strip_suffix("_ln_bias") {
                if let Ok(v) = it.to_f32() {
                    layer_norms.entry(base.to_string()).or_default().1 = Some(v);
                }
            }
        }

        // Decoding the float weights is also how precision is detected: an int8
        // container has no 2-D float32 weights, so this comes back empty for it.
        let float_affines = prepare_float_affines(&mut model, trg_wemb_param);
        let precision = if float_affines.is_empty() {
            Precision::Int8
        } else {
            Precision::Float32
        };

        let raw_proj_bias = |model: &Model| {
            model
                .get("decoder_ff_logit_out_b")
                .and_then(|it| it.to_f32().ok())
                .unwrap_or_else(|| vec![0.0; trg_vocab])
        };

        // A float model always uses resident tables: they *are* the model's own
        // f32 data, so `lean-embed` has nothing to save and its int8 projection
        // has no quant multipliers to run on.
        let lean = cfg!(feature = "lean-embed") && precision == Precision::Int8;

        let embed = if !lean {
            let trg = load_embedding(&model, trg_wemb_param)?;
            let src = if src_wemb_param == trg_wemb_param {
                None
            } else {
                Some(load_embedding(&model, src_wemb_param)?)
            };
            Embed::Resident(ResidentEmbed {
                trg,
                src,
                proj_bias: raw_proj_bias(&model),
            })
        } else {
            #[cfg(not(feature = "lean-embed"))]
            unreachable!("lean is gated on the feature");
            #[cfg(feature = "lean-embed")]
            {
                let trg_item = model.get(trg_wemb_param).expect("target embedding");
                let qwemb = trg_item.quant_mult().map_err(|e| e.to_string())?;
                let src_inv_qmult = 1.0
                    / model
                        .get(src_wemb_param)
                        .ok_or_else(|| format!("model has no {src_wemb_param}"))?
                        .quant_mult()
                        .map_err(|e| e.to_string())?;
                let proj_qa = read_output_qa(&model);
                let proj_unquant = 1.0 / (proj_qa * qwemb);
                // Prepared bias is static — fold the shift correction once here.
                let raw = trg_item.int8_transposed().map_err(|e| e.to_string())?;
                let proj_bias =
                    ops::prepare_bias(raw, trg_vocab, dim, &raw_proj_bias(&model), proj_unquant);

                // Pack the target `Wemb` for the output projection now (eagerly) rather
                // than on first projection. Embedding lookups happen on the first encode,
                // *before* any projection, so building it here lets those lookups read
                // their rows back out of the packed buffer — and when the pack succeeds
                // (SIMD kernel present, `k % 16 == 0`) we drop the raw int8 copy, since
                // the packed layout is a lossless reblocking that serves both. That
                // removes the last ~15.6 MiB "held twice" duplication. If the pack fails
                // (scalar build), `proj_pb` stays `None` and the raw copy is kept.
                #[cfg(fast_gemm)]
                let proj_pb = {
                    let packed = {
                        let raw = model
                            .get(trg_wemb_param)
                            .expect("target embedding")
                            .int8_transposed()
                            .map_err(|e| e.to_string())?;
                        PreparedB::new(raw, trg_vocab, dim)
                    };
                    if packed.is_some() {
                        if let Some(it) =
                            model.items.iter_mut().find(|it| it.name == trg_wemb_param)
                        {
                            it.data = crate::model::Bytes::Owned(Vec::new());
                        }
                    }
                    packed
                };
                Embed::LeanInt8(Box::new(LeanInt8Embed {
                    src_param: src_wemb_param,
                    src_inv_qmult,
                    trg_inv_qmult: 1.0 / qwemb,
                    proj_bias,
                    proj_qa,
                    proj_unquant,
                    #[cfg(fast_gemm)]
                    proj_pb,
                }))
            }
        };

        // Int8 only: a float model has no `_QuantMultA` siblings, so this would
        // `continue` past every weight and hand back an empty cache.
        #[cfg(fast_gemm)]
        let affine_cache = if precision == Precision::Int8 {
            prepare_affines(&mut model, trg_wemb_param)
        } else {
            HashMap::new()
        };

        Ok(Weights {
            model,
            config,
            precision,
            trg_vocab,
            dim,
            layer_norms,
            trg_wemb_param,
            embed,
            float_affines,
            #[cfg(fast_gemm)]
            affine_cache,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// Which numeric form this model's GEMM weights shipped in.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// A float parameter by name (bias, layernorm scale/bias, …).
    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.model.get(name).and_then(|it| it.to_f32().ok())
    }

    /// Output vocabulary size (number of tied-projection rows).
    pub fn output_vocab(&self) -> usize {
        self.trg_vocab
    }

    /// Full-vocabulary output logits `h · Wemb^T + bias`.
    pub fn full_logits(&self, h: &[f32]) -> Vec<f32> {
        self.full_logits_batch(h, 1)
    }

    /// Batched tied output projection: `h` is `[m, dim]` (m decoder tops stacked
    /// row-major), the result is `[m, vocab]`. Projecting the whole minibatch in
    /// one GEMM streams the large vocab weight once per batch instead of once per
    /// row — the matrix×matrix reuse win at the output layer. Rows are
    /// independent, so per-row results match [`full_logits`] exactly.
    pub fn full_logits_batch(&self, h: &[f32], m: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.full_logits_batch_into(h, m, &mut out);
        out
    }

    /// [`full_logits_batch`] into a caller-owned buffer (resized to `[m, vocab]`),
    /// reused across decode steps to avoid a fresh full-vocab allocation each time.
    pub fn full_logits_batch_into(&self, h: &[f32], m: usize, out: &mut Vec<f32>) {
        match &self.embed {
            Embed::Resident(e) => {
                ops::project_f32_into(h, m, self.dim, &e.trg, self.trg_vocab, &e.proj_bias, out);
            }
            #[cfg(feature = "lean-embed")]
            Embed::LeanInt8(e) => {
                #[cfg(fast_gemm)]
                {
                    // `proj_pb` is packed eagerly at load (see `Weights::new`); when present
                    // it also backs embedding lookups, so the raw int8 copy is already gone.
                    if let Some(pb) = e.proj_pb.as_ref() {
                        GEMM_SCRATCH.with_borrow_mut(|s| {
                            ops::prepare_a_into(h, e.proj_qa, &mut s.a_u8);
                            pb.matmul_into(&s.a_u8, m, e.proj_unquant, &e.proj_bias, out);
                        });
                        return;
                    }
                }
                let raw = self
                    .model
                    .get(self.trg_wemb_param)
                    .expect("target embedding")
                    .int8_transposed()
                    .expect("int8 embedding");
                let a = ops::prepare_a(h, e.proj_qa);
                *out = ops::intgemm_affine(
                    &a,
                    m,
                    self.dim,
                    raw,
                    self.trg_vocab,
                    e.proj_unquant,
                    &e.proj_bias,
                );
            }
        }
    }

    /// Layer-norm `scale` and optional `bias` for a sublayer base name (e.g.
    /// `encoder_l1_ffn`), borrowed from the load-time cache — no per-call decode.
    pub fn layer_norm(&self, base: &str) -> Option<(&[f32], Option<&[f32]>)> {
        self.layer_norms
            .get(base)
            .map(|(g, b)| (g.as_slice(), b.as_deref()))
    }

    /// Write one source (encoder) embedding row into `dst` (length `dim`): copied
    /// from the resident f32 table (shared vocab reuses the target table), or
    /// dequantized on demand from the int8 tensor under `lean-embed`. No
    /// allocation — the caller owns `dst`.
    pub fn src_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        match &self.embed {
            Embed::Resident(e) => {
                let d = self.dim;
                let wemb = e.src.as_deref().unwrap_or(&e.trg);
                dst.copy_from_slice(&wemb[id as usize * d..(id as usize + 1) * d]);
            }
            #[cfg(feature = "lean-embed")]
            Embed::LeanInt8(e) => self.dequant_row_into(e.src_param, e.src_inv_qmult, id, dst),
        }
    }

    /// Write one target (decoder) embedding row into `dst` (length `dim`).
    pub fn trg_embed_row_into(&self, id: u32, dst: &mut [f32]) {
        match &self.embed {
            Embed::Resident(e) => {
                let d = self.dim;
                dst.copy_from_slice(&e.trg[id as usize * d..(id as usize + 1) * d]);
            }
            #[cfg(feature = "lean-embed")]
            Embed::LeanInt8(e) => {
                self.dequant_row_into(self.trg_wemb_param, e.trg_inv_qmult, id, dst)
            }
        }
    }

    /// Dequantize one embedding row from the int8 model tensor into `dst` (lean
    /// build) — `dst[c] = raw[c] · inv`, no allocation.
    #[cfg(feature = "lean-embed")]
    fn dequant_row_into(&self, param: &str, inv: f32, id: u32, dst: &mut [f32]) {
        let d = self.dim;
        // The target embedding is packed for the output projection and its raw int8
        // copy freed, so read the row back out of the packed buffer (a lossless
        // reblocking). The source embedding (split vocab) is never packed → raw.
        #[cfg(fast_gemm)]
        if param == self.trg_wemb_param {
            if let Embed::LeanInt8(e) = &self.embed {
                if let Some(pb) = e.proj_pb.as_ref() {
                    GEMM_SCRATCH.with_borrow_mut(|s| {
                        s.wemb_row.resize(d, 0);
                        pb.read_row(id as usize, &mut s.wemb_row);
                        for (o, &b) in dst.iter_mut().zip(&s.wemb_row) {
                            *o = b as f32 * inv;
                        }
                    });
                    return;
                }
            }
        }
        let raw = self
            .model
            .get(param)
            .expect("embedding param")
            .int8_transposed()
            .expect("int8 embedding");
        for (o, &b) in dst
            .iter_mut()
            .zip(&raw[id as usize * d..(id as usize + 1) * d])
        {
            *o = b as f32 * inv;
        }
    }

    /// Quant multiplier of the int8 target embedding, for the shortlist int8 output
    /// projection; `None` if the embedding is not int8 (a float model, or an int8
    /// model whose embedding shipped dequantized) — the caller then takes the
    /// float projection path.
    pub fn output_wemb_qmult(&self) -> Option<f32> {
        match &self.embed {
            // Under lean-embed the raw copy may be freed once packed, so use the
            // value captured at load (`trg_inv_qmult = 1/qwemb`).
            #[cfg(feature = "lean-embed")]
            Embed::LeanInt8(e) => Some(1.0 / e.trg_inv_qmult),
            // Resident tables are f32 regardless of what they were decoded from,
            // and a float model has no multiplier at all.
            Embed::Resident(_) => self.model.get(self.trg_wemb_param)?.quant_mult().ok(),
        }
    }

    /// Gather one int8 target-embedding row (length `dim`) into `out`, for the
    /// shortlist projection's candidate weight matrix. Reads from the packed
    /// projection buffer when the raw copy has been freed (lean-embed + SIMD),
    /// else from the raw int8 tensor. Returns `false` if the embedding is float.
    pub fn output_wemb_int8_row(&self, id: u32, out: &mut [i8]) -> bool {
        #[cfg(all(feature = "lean-embed", fast_gemm))]
        if let Embed::LeanInt8(e) = &self.embed {
            if let Some(pb) = e.proj_pb.as_ref() {
                pb.read_row(id as usize, out);
                return true;
            }
        }
        let d = self.dim;
        match self
            .model
            .get(self.trg_wemb_param)
            .and_then(|it| it.int8_transposed().ok())
        {
            Some(raw) => {
                out.copy_from_slice(&raw[id as usize * d..(id as usize + 1) * d]);
                true
            }
            None => false,
        }
    }

    /// Activation quant-mult (qA) for the tied output projection
    /// ([`read_output_qa`]).
    ///
    /// # Panics
    /// If the model has no output-projection `*_QuantMultA` — i.e. on a float
    /// model. Call sites reach this only after [`Weights::output_wemb_qmult`]
    /// returns `Some`, which a float model never does.
    pub fn output_qa(&self) -> f32 {
        read_output_qa(&self.model)
    }

    /// Run the affine `y = x·W + bias` from the weight's base name (e.g.
    /// `encoder_l1_self_Wq`). `x` is `[m, k]` row-major; the result is `[m, n]`.
    /// `bias_name` is the raw bias parameter, or `None` for the bias-less
    /// matmuls (SSRU's `W`).
    ///
    /// On an int8 model this is the shifted int8 affine, and the `None` bias uses
    /// the fake-bias correction only. On a float model it is a plain f32 GEMM with
    /// no quantization and no shift correction, so `None` really is a zero bias.
    ///
    /// # Panics
    /// If the weight (or, on an int8 model, its `*_QuantMultA`) is missing, or
    /// shapes are inconsistent.
    pub fn affine(&self, base: &str, x: &[f32], m: usize, bias_name: Option<&str>) -> Vec<f32> {
        // Float model: a plain f32 GEMM against the `[k, n]` table decoded at load.
        if self.precision == Precision::Float32 {
            let fw = self
                .float_affines
                .get(base)
                .unwrap_or_else(|| panic!("missing float weight {base}"));
            let bias = fw.bias.get_or_init(|| match bias_name {
                Some(bn) => self.f32(bn).unwrap_or_else(|| panic!("missing bias {bn}")),
                None => vec![0.0; fw.n],
            });
            return ops::affine_f32(x, m, fw.k, &fw.w, fw.n, bias);
        }

        // Fast path: weight packed at load. Build the full prepared bias once
        // (correction + raw bias), then reuse the activation scratch and let the
        // shim reuse its own — a steady-state affine allocates only its output.
        #[cfg(fast_gemm)]
        {
            if let Some(aw) = self.affine_cache.get(base) {
                // The full prepared bias (correction + raw bias) is a pure function of
                // this weight; compute it once and cache it in the `OnceLock`. The bias
                // name is fixed per weight at every call site, so the value is stable.
                let bias = aw.bias.get_or_init(|| {
                    let mut bias = aw.correction.clone();
                    if let Some(bn) = bias_name {
                        if let Some(rb) = self.f32(bn) {
                            for (b, r) in bias.iter_mut().zip(rb.iter()) {
                                *b += *r;
                            }
                        }
                    }
                    bias
                });
                return GEMM_SCRATCH.with_borrow_mut(|s| {
                    ops::prepare_a_into(x, aw.qa, &mut s.a_u8);
                    let mut out = Vec::new();
                    aw.pb.matmul_into(&s.a_u8, m, aw.unquant, bias, &mut out);
                    out
                });
            }
        }

        // Scalar path: raw weight (gemmology off, or `k % 16 != 0` so it wasn't
        // packed and its raw bytes were kept).
        let w = self
            .model
            .get(base)
            .unwrap_or_else(|| panic!("missing weight {base}"));
        // Stored logical shape is [K, N]; data is transposed to [N, K].
        let k = w.shape[0] as usize;
        let n = w.shape[1] as usize;
        let b = w.int8_transposed().expect("int8 weight");
        debug_assert_eq!(b.len(), n * k);
        let qb = w.quant_mult().expect("weight quant mult");

        let qa = self
            .f32(&format!("{base}_QuantMultA"))
            .unwrap_or_else(|| panic!("missing {base}_QuantMultA"))[0];
        let unquant = 1.0 / (qa * qb);

        let raw_bias = match bias_name {
            Some(bn) => self.f32(bn).unwrap_or_else(|| panic!("missing bias {bn}")),
            None => vec![0.0; n],
        };
        let prepared = ops::prepare_bias(b, n, k, &raw_bias, unquant);

        let a = ops::prepare_a(x, qa);
        ops::intgemm_affine(&a, m, k, b, n, unquant, &prepared)
    }
}
