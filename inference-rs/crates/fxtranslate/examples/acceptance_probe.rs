//! Phase-0 acceptance probe for shortlist-as-draft speculative decoding (notes/21).
//!
//! Runs ordinary full-vocab greedy over a corpus and, at each decode step, asks
//! whether the full-vocab argmax lies in the sentence's lexical-shortlist candidate
//! set — i.e. whether a shortlist *draft* would have guessed that token correctly.
//! No speculative code runs; this measures the speedup ceiling before any is written.
//!
//! It reports:
//!   - overall acceptance (mean in-shortlist rate — the driver of tokens/round),
//!   - the run-length distribution of consecutive in-shortlist steps,
//!   - for each guess length K, the simulated average tokens accepted per verify
//!     round A(K) and the predicted decode/overall speedup under the note's model,
//!   - a GO / STOP verdict against the note's ~70% acceptance threshold.
//!
//! Run: `cargo run --release --example acceptance_probe --features fast -p fxtranslate`
//! Options: `-- --corpus <path>` (default the bundled en-fr corpus), `--limit <n>`,
//! `--proj-frac <f>` (projection share of decode time, default 0.8 per notes/18/20).
//!
//! Model/vocab/shortlist default to `data/models/enfr/`; override the directory with
//! the `FXTRANSLATE_MODEL_DIR` env var.

use std::path::{Path, PathBuf};

use fxtranslate::engine::Engine;
use fxtranslate::shortlist::Shortlist;

/// The guess lengths K the speedup model is simulated at.
const K_VALUES: [usize; 6] = [2, 3, 4, 5, 6, 8];

fn main() {
    let opts = Options::parse();

    let model_dir = std::env::var("FXTRANSLATE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_path("data/models/enfr"));
    let model = model_dir.join("model.enfr.intgemm.alphas.bin");
    let vocab = model_dir.join("vocab.enfr.spm");
    let shortlist = model_dir.join("lex.50.50.enfr.s2t.bin");

    if !model.exists() || !vocab.exists() || !shortlist.exists() {
        eprintln!(
            "model/vocab/shortlist absent under {} — set FXTRANSLATE_MODEL_DIR",
            model_dir.display()
        );
        std::process::exit(2);
    }

    let engine = Engine::load(&model, &vocab, &vocab).expect("engine loads");
    let engine = engine.with_shortlist(Shortlist::load(&shortlist).expect("shortlist loads"));

    let corpus_path = opts
        .corpus
        .clone()
        .unwrap_or_else(|| repo_path("corpora/nllb-en-fr.txt"));
    let text = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("reading corpus {}: {e}", corpus_path.display()));
    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if let Some(limit) = opts.limit {
        lines.truncate(limit);
    }

    println!(
        "probe: {} sentences from {}  (proj-frac P = {:.2})",
        lines.len(),
        corpus_path.display(),
        opts.proj_frac
    );

    // Aggregate the raw per-step hit signal across the whole corpus.
    let mut total_steps = 0usize;
    let mut total_hits = 0usize;
    let mut cand_sizes: Vec<usize> = Vec::with_capacity(lines.len());
    // run_hist[len] = number of maximal in-shortlist runs of exactly `len` steps.
    let mut run_hist: Vec<usize> = Vec::new();
    // Per-K accumulators: (tokens, rounds) summed over sentences.
    let mut k_tokens = [0usize; K_VALUES.len()];
    let mut k_rounds = [0usize; K_VALUES.len()];

    for line in &lines {
        let src_ids = engine.src_ids(line);
        let probe = engine
            .acceptance_probe(&src_ids)
            .expect("shortlist attached");

        total_steps += probe.hits.len();
        total_hits += probe.hits.iter().filter(|&&h| h).count();
        cand_sizes.push(probe.candidate_count);
        tally_runs(&probe.hits, &mut run_hist);

        for (i, &k) in K_VALUES.iter().enumerate() {
            let (tokens, rounds) = simulate_rounds(&probe.hits, k);
            k_tokens[i] += tokens;
            k_rounds[i] += rounds;
        }
    }

    if total_steps == 0 {
        eprintln!("no decode steps — empty corpus?");
        std::process::exit(2);
    }

    let acceptance = total_hits as f64 / total_steps as f64;
    cand_sizes.sort_unstable();
    let cand_mean = cand_sizes.iter().sum::<usize>() as f64 / cand_sizes.len() as f64;
    let cand_med = cand_sizes[cand_sizes.len() / 2];

    println!(
        "\noverall acceptance: {:.1}%  ({total_hits}/{total_steps} steps in shortlist)",
        acceptance * 100.0
    );
    println!(
        "shortlist size: mean {cand_mean:.0}, median {cand_med}, max {} candidates/sentence",
        cand_sizes.last().copied().unwrap_or(0)
    );

    print_run_histogram(&run_hist);

    println!(
        "\nspeculative speedup by guess length K (P = {:.2}):",
        opts.proj_frac
    );
    println!("  (A = avg tokens accepted per verify round = full-vocab hauls saved)");
    println!(
        "  {:>3}  {:>6}  {:>10}  {:>12}  {:>11}",
        "K", "A", "hauls/tok", "decode↑ (opt)", "decode↑ (real)"
    );
    let p = opts.proj_frac;
    let mut best_real = 0.0f64;
    for (i, &k) in K_VALUES.iter().enumerate() {
        let a = k_tokens[i] as f64 / k_rounds[i] as f64;
        let hauls_per_tok = k_rounds[i] as f64 / k_tokens[i] as f64;
        // Optimistic (note's ceiling): decoder layers re-run for free, so decode
        // time is (1-P) fixed layers + P/A projection. Realistic: the layers run K
        // times per round (the whole draft) but yield only A accepted tokens, so
        // layer work scales by K/A. Draft-projection cost (cheap, small candidate
        // set) is excluded from both — this is a ceiling.
        let decode_opt = 1.0 / ((1.0 - p) + p / a);
        let decode_real = 1.0 / (((1.0 - p) * k as f64 + p) / a);
        best_real = best_real.max(decode_real);
        println!(
            "  {k:>3}  {a:>6.2}  {hauls_per_tok:>10.3}  {decode_opt:>11.2}x  {decode_real:>10.2}x"
        );
    }

    println!("\nverdict:");
    if acceptance >= 0.70 {
        println!(
            "  GO — acceptance {:.1}% ≥ 70%. Best realistic decode speedup ~{best_real:.2}x.",
            acceptance * 100.0
        );
        println!("  Proceed to Phase 1 (single-sentence speculative loop).");
    } else {
        println!(
            "  STOP — acceptance {:.1}% < 70%. Speculation won't pay; do not build Phases 1-3.",
            acceptance * 100.0
        );
    }
    println!(
        "  Note: the decode↑ model assumes the projection is P={:.2} of decode; confirm with the",
        p
    );
    println!("  notes/20 DRAM-vs-cache probe before trusting the absolute speedup figure.");
}

/// Simulate speculative rounds over one sentence's hit sequence for guess length
/// `k`, returning (tokens_emitted, verify_rounds). Every emitted token is counted
/// (the full path is fixed); `rounds` is the number of full-vocab verify hauls.
///
/// Per round from position `i`: count the leading in-shortlist run `r` (capped at
/// `k`). If `r == k` all K guesses matched — accept K, no correction. If the run
/// hits a mismatch at `i+r` (`i+r < len`), accept those `r` plus the 1 correction
/// the verifier supplies. If the run instead reaches the end of the sentence
/// (`i+r == len`), accept the `r` tail tokens with no correction — the sentence is
/// done. One verify haul per round either way.
fn simulate_rounds(hits: &[bool], k: usize) -> (usize, usize) {
    let len = hits.len();
    let mut i = 0usize;
    let mut rounds = 0usize;
    while i < len {
        rounds += 1;
        let mut r = 0usize;
        while r < k && i + r < len && hits[i + r] {
            r += 1;
        }
        if r == k {
            i += k;
        } else if i + r == len {
            i += r;
        } else {
            i += r + 1;
        }
    }
    (len, rounds)
}

/// Tally maximal runs of consecutive `true` (in-shortlist) steps into `hist`,
/// indexed by run length. `hist[0]` counts individual `false` (mismatch) steps.
fn tally_runs(hits: &[bool], hist: &mut Vec<usize>) {
    let mut run = 0usize;
    let bump = |hist: &mut Vec<usize>, idx: usize| {
        if idx >= hist.len() {
            hist.resize(idx + 1, 0);
        }
        hist[idx] += 1;
    };
    for &h in hits {
        if h {
            run += 1;
        } else {
            if run > 0 {
                bump(hist, run);
                run = 0;
            }
            bump(hist, 0);
        }
    }
    if run > 0 {
        bump(hist, run);
    }
}

fn print_run_histogram(hist: &[usize]) {
    let total_runs: usize = hist.iter().skip(1).sum();
    println!("\nin-shortlist run-length distribution (consecutive correct guesses):");
    if let Some(&misses) = hist.first() {
        println!("  mismatches (breaks): {misses}");
    }
    for (len, &count) in hist.iter().enumerate().skip(1) {
        if count == 0 {
            continue;
        }
        let pct = 100.0 * count as f64 / total_runs.max(1) as f64;
        let bar = "█".repeat((pct / 2.0).round() as usize);
        println!("  run={len:<3} {count:>6} ({pct:>4.1}%) {bar}");
    }
}

struct Options {
    corpus: Option<PathBuf>,
    limit: Option<usize>,
    proj_frac: f64,
}

impl Options {
    fn parse() -> Options {
        let mut corpus = None;
        let mut limit = None;
        let mut proj_frac = 0.8;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--corpus" => corpus = args.next().map(PathBuf::from),
                "--limit" => limit = args.next().and_then(|v| v.parse().ok()),
                "--proj-frac" => {
                    proj_frac = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .expect("--proj-frac needs a float")
                }
                other => {
                    eprintln!("unknown arg {other:?}");
                    std::process::exit(2);
                }
            }
        }
        Options {
            corpus,
            limit,
            proj_frac,
        }
    }
}

/// Resolve a repo-relative path from the crate manifest dir, so the example runs
/// the same regardless of the shell's working directory. The crate lives at
/// `<repo>/inference-rs/crates/fxtranslate`.
fn repo_path(rel: &str) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if rel.starts_with("corpora/") {
        // corpora/ live under inference-rs (two levels up from the crate).
        manifest.join("../..").join(rel)
    } else {
        // data/ lives at the repository root (three levels up).
        manifest.join("../../..").join(rel)
    }
}
