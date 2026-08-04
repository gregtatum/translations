//! Measured throughput of speculative decoding (notes/21) vs full-vocab greedy —
//! the real number that replaced the Phase 0 projection. (Outcome: it regresses on
//! this hardware/model; see the NEGATIVE RESULT section of notes/21.)
//!
//! Runs the same block corpus through two engines sharing the same weights:
//!   - **baseline**: shortlist-free batched greedy (`greedy_batch`) — full-vocab,
//!     the exact output the comparison table's "fxtranslate (fast)" row measures;
//!   - **speculative**: `greedy_batch_speculative(k)` with the shortlist as draft.
//! Both produce byte-identical output (asserted here over the whole corpus), so
//! this is apples-to-apples: same tokens, measured time. Reports words/s for each
//! and the speedup per guess length K. Model load is excluded (timed region is the
//! per-block `greedy_batch*` compute, as in `final_comparison.py`).
//!
//! Run: `cargo run --release --example speculative_bench --features fast -p fxtranslate`
//! Defaults to en-fr; the comparison-table workload is en-ru:
//!   `FXTRANSLATE_MODEL_DIR=../data/models/enru \`
//!   `cargo run --release --example speculative_bench --features fast -p fxtranslate -- \`
//!   `--corpus corpora/frankenstein-en.blocks.txt`
//! Options: `--corpus <path>`, `--runs <n>` (timed runs, median reported), `--ks 1,2,4,6,8`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fxtranslate::engine::Engine;
use fxtranslate::shortlist::Shortlist;

fn main() {
    let opts = Options::parse();

    let model_dir = std::env::var("FXTRANSLATE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_path("data/models/enfr"));
    let (Some(model), Some(vocab), Some(shortlist)) = (
        find_in_dir(&model_dir, ".intgemm.alphas.bin"),
        find_in_dir(&model_dir, ".spm"),
        find_in_dir(&model_dir, ".s2t.bin"),
    ) else {
        eprintln!(
            "model/vocab/shortlist not all found under {} — set FXTRANSLATE_MODEL_DIR",
            model_dir.display()
        );
        std::process::exit(2);
    };

    // Baseline is shortlist-free (full-vocab greedy); speculative attaches the
    // shortlist as its draft. Same weights, so the comparison is fair.
    let baseline = Engine::load(&model, &vocab, &vocab).expect("engine loads");
    let speculative = Engine::load(&model, &vocab, &vocab)
        .expect("engine loads")
        .with_shortlist(Shortlist::load(&shortlist).expect("shortlist loads"));

    let corpus_path = opts
        .corpus
        .clone()
        .unwrap_or_else(|| repo_path("corpora/frankenstein-en.blocks.txt"));
    let text = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("reading corpus {}: {e}", corpus_path.display()));
    // Blank-line-separated blocks, one sentence per line — one batched translate per
    // block (the production shape the comparison table uses).
    let blocks: Vec<Vec<Vec<u32>>> = text
        .split("\n\n")
        .map(|b| {
            b.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| baseline.src_ids(l))
                .collect::<Vec<_>>()
        })
        .filter(|b: &Vec<Vec<u32>>| !b.is_empty())
        .collect();
    let src_words: usize = text.split_whitespace().filter(|w| !w.is_empty()).count();
    let sentences: usize = blocks.iter().map(Vec::len).sum();

    println!(
        "bench: {} blocks, {sentences} sentences, {src_words} source words from {}",
        blocks.len(),
        corpus_path.display()
    );
    println!(
        "  model {}\n  runs {} (median reported)\n",
        model.display(),
        opts.runs
    );

    // Correctness gate over the WHOLE corpus before timing: every K's speculative
    // output must equal baseline full-vocab greedy, block by block.
    let baseline_out: Vec<Vec<Vec<u32>>> =
        blocks.iter().map(|b| baseline.greedy_batch(b)).collect();
    for &k in &opts.ks {
        for (bi, block) in blocks.iter().enumerate() {
            let spec = speculative.greedy_batch_speculative(block, k);
            assert_eq!(
                spec, baseline_out[bi],
                "byte-identity FAILED: block {bi}, K={k}"
            );
        }
    }
    println!(
        "byte-identity: OK — speculative == full-vocab greedy for all blocks, K∈{:?}\n",
        opts.ks
    );

    // Flattened single sentences, for the batch=1 comparison. Speculation targets
    // the per-token projection weight-haul; batched greedy already amortizes that
    // haul across the block's rows, so batch=1 is where speculation has the most to
    // win (the notes/18 bandwidth-bound regime).
    let flat: Vec<Vec<u32>> = blocks.iter().flatten().cloned().collect();

    if opts.per_sentence {
        let base_wps = median_wps(src_words, opts.runs, || {
            for s in &flat {
                std::hint::black_box(baseline.greedy(s));
            }
        });
        println!("baseline single-sentence (full-vocab greedy):  {base_wps:>7.0} wps");
        println!("\nspeculative (single-sentence, batch=1):");
        println!("  {:>3}  {:>9}  {:>8}", "K", "wps", "speedup");
        for &k in &opts.ks {
            let wps = median_wps(src_words, opts.runs, || {
                for s in &flat {
                    std::hint::black_box(speculative.greedy_speculative(s, k));
                }
            });
            println!("  {k:>3}  {wps:>9.0}  {:>7.2}x", wps / base_wps);
        }
        return;
    }

    let base_wps = median_wps(src_words, opts.runs, || {
        for b in &blocks {
            std::hint::black_box(baseline.greedy_batch(b));
        }
    });
    println!("baseline block-batched (full-vocab greedy):  {base_wps:>7.0} wps");
    println!("\nspeculative (block-batched):");
    println!("  {:>3}  {:>9}  {:>8}", "K", "wps", "speedup");
    for &k in &opts.ks {
        let wps = median_wps(src_words, opts.runs, || {
            for b in &blocks {
                std::hint::black_box(speculative.greedy_batch_speculative(b, k));
            }
        });
        println!("  {k:>3}  {wps:>9.0}  {:>7.2}x", wps / base_wps);
    }
}

/// Run `work` `runs` times (plus one warmup), returning the **median** words/s
/// (`src_words / compute_seconds`). Median, not mean, to shrug off a slow run.
fn median_wps(src_words: usize, runs: usize, mut work: impl FnMut()) -> f64 {
    work(); // warmup
    let mut secs: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let t = Instant::now();
            work();
            t.elapsed().as_secs_f64()
        })
        .collect();
    secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = secs[secs.len() / 2];
    src_words as f64 / med
}

struct Options {
    corpus: Option<PathBuf>,
    runs: usize,
    ks: Vec<usize>,
    per_sentence: bool,
}

impl Options {
    fn parse() -> Options {
        let mut corpus = None;
        let mut runs = 4;
        let mut ks = vec![1, 2, 3, 4, 5, 6, 8];
        let mut per_sentence = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--corpus" => corpus = args.next().map(PathBuf::from),
                "--per-sentence" => per_sentence = true,
                "--runs" => runs = args.next().and_then(|v| v.parse().ok()).expect("--runs n"),
                "--ks" => {
                    ks = args
                        .next()
                        .expect("--ks 1,2,4")
                        .split(',')
                        .map(|s| s.parse().expect("K must be a positive integer"))
                        .collect()
                }
                other => {
                    eprintln!("unknown arg {other:?}");
                    std::process::exit(2);
                }
            }
        }
        Options {
            corpus,
            runs,
            ks,
            per_sentence,
        }
    }
}

/// First entry in `dir` whose file name ends with `suffix` (sorted for
/// determinism); `None` if unreadable or no match. The three suffixes used here
/// (`.intgemm.alphas.bin`, `.spm`, `.s2t.bin`) are mutually disjoint.
fn find_in_dir(dir: &Path, suffix: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(suffix))
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Repo-relative path from the crate manifest dir, so the example runs regardless
/// of the shell's working directory. Crate is at `<repo>/inference-rs/crates/fxtranslate`.
fn repo_path(rel: &str) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if rel.starts_with("corpora/") {
        manifest.join("../..").join(rel)
    } else {
        manifest.join("../../..").join(rel)
    }
}
