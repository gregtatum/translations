# Threading opportunities — where more CPU cores can help

This note looks at how `inference-rs` / `fxtranslate` could use more than one CPU
core, what the wins and ceilings are, and which approach fits the project's
memory constraint (don't duplicate the model per thread). It is grounded in a
fresh, self-symbolicated profile of the current engine (see below), not the
historical numbers in [08-perf-analysis.md](./08-perf-analysis.md), which predate
the gemmology kernel swap and the base-model benchmark and are superseded here.

Everything below is single-threaded today: a translation run uses **one core**.

## Vocabulary (so the rest reads cleanly)

- **Matrix multiply (GEMM).** GEMM = "GEneral Matrix Multiply", the BLAS-era name
  for "multiply matrix A by matrix B (and add a bias)". Every linear layer in the
  model is one. The model spends almost all of its time here.
- **Affine / linear layer / "projection".** All the same operation: a vector times
  a weight matrix, plus a bias. In the code it is `Weights::affine`. The **output
  projection** is one specific, unusually large matrix multiply (below).
- **int8.** Weights are stored as 8-bit integers, not 32-bit floats, to save memory
  and speed up the multiply. So these are *integer* matrix multiplies.
- **SIMD / NEON / i8mm / neon64.** SIMD = "Single Instruction, Multiple Data" — one
  CPU instruction doing many multiply-adds at once. On ARM (Apple Silicon) the SIMD
  family is **NEON**; **i8mm** is a newer ARM extension with dedicated int8
  matrix-multiply instructions. The math library labels this combination
  `i8mm<neon64>` (`neon64` = 64-bit ARM NEON). Seeing `neon64` in a profile symbol
  is expected on an ARM Mac — it confirms the fast ARM kernel is live. An x86 build
  would show `avx2` (Intel's SIMD family) instead.
- **gemmology.** The small C++ math library (shared with Firefox) that performs one
  int8 matrix multiply as fast as possible with SIMD.

## What the model is doing, and where the matrix multiplies come from

The benchmark uses the en→ru **base** model: **512 numbers per token**, a **32,000**
word-piece vocabulary, **6 encoder layers**, **2 SSRU decoder layers**, tied 32k
embeddings.

**Encoder** — reads the whole source sentence at once and turns it into "meaning"
vectors. Runs **once** per sentence. Because it processes every source token
together, its matrix multiplies are *big* (a stack of many token-vectors × a weight
matrix). Per layer (×6): attention = 4 multiplies (~512×512 each); feed-forward
network = 2 multiplies (512→2048, then 2048→512).

**Decoder** — writes the translation **one word-piece at a time, in a loop**. Word 5
needs word 4 first, so the loop is inherently sequential. Per pass, per layer (×2):
the SSRU recurrent cell = 2 multiplies (512×512); cross-attention to the encoder =
query + output projections (512×512; key/value are precomputed once per sentence);
FFN = 2 multiplies (512→2048, 2048→512). These are *small* — one token's 512 numbers,
not the whole sentence.

**Output projection (the expensive per-step step).** After the decoder layers, the
512-number vector must score **all 32,000 possible next word-pieces** to choose one:
a single 512 × 32,000 matrix multiply, followed by an `argmax` scan of the 32,000
scores. This happens on *every* word the decoder emits.

## How the code reaches the SIMD kernel

Every matrix multiply funnels through one path:

```
engine.rs (builds the network)
  → Weights::affine()                 // weights.rs — looks up the packed int8 weight
    → gemm::PreparedB::matmul()       // gemm.rs — safe Rust wrapper
      → gemmology_multiply()          // gemmology_shim.cpp — thin C ABI shim
        → gemmology::Shift::Multiply  // the actual SIMD loop
```

The 32k output projection takes the same path, entered through
`Weights::full_logits_batch_into` instead of `affine`. That is why one kernel
accounts for two-thirds of the run.

## Current profile (block path, en→fr, single thread)

`artifacts/perf-blocks-rs-enfr.json.gz` (2026-07-10), symbolicated locally with
`atos` against `target/release/fxtranslate-oracle` (profiler-cli's symbol server
cannot resolve local Rust symbols; the profile carries raw offsets). 8.3 s wall,
96% CPU on one core. Shares of active time:

| region | share | what it is |
|---|---:|---|
| **`gemmology_multiply` (the SIMD matrix multiply)** | **~67%** | the hot kernel |
| ├─ encoder + decoder ordinary layers (`affine`) | ~40% | many multiplies |
| └─ output projection (`full_logits_batch_into`) | ~27% | 512×32,000, every decode step |
| **`argmax` over the 32k scores (`select_active` self)** | **~14%** | *not* a matrix multiply |
| normalization / softmax / int8 conversion / glue | remainder | each <~2% |

Two facts to carry forward:

1. The kernel symbol is
   `gemmology::Engine<i8mm<neon64>>::Shift::Multiply<UnquantizeAndAddBiasAndWrite,
   gemmology::SequentialExecutionEngine>` — the `SequentialExecutionEngine` type
   parameter means the multiply iterates its work on **one core**. gemmology has a
   built-in hook to iterate across threads; we are simply not using it.
2. The transformer affines (~40%) are now a *bigger* aggregate than the output
   projection (~27%). The old note's "61% in the projection" was the tiny model with
   a per-row projection; with batching + the base model the pile of ordinary layers
   dominates. This changes the threading math below.

Single-thread throughput is already ~0.96× native marian (README), so this is about
scaling *beyond* single-core parity, not catching up.

## Two ways to use more cores

### Option A — split ONE matrix multiply across cores (intra-op)

When gemmology multiplies the 512-number vector by the 512×32,000 weight, it walks
across the 32,000 output columns computing one score each. Those columns are
independent (score #5 doesn't depend on #4), so core 1 can take columns 0–8,000,
core 2 columns 8,000–16,000, etc. They all read the same weight and the same input
vector and write to different slots of the same output.

- **Memory cost: essentially zero.** No copy of the model — all cores read the one
  shared copy. This is the purest "no graph duplication" answer.
- gemmology already supports it: `Shift::Multiply` is templated on an *execution
  engine*; today we pass `SequentialExecutionEngine`, and it ships
  `StdThreadExecutionEngine(poolSize)` and `OpenMPExecutionEngine`
  (`vendor/gemmology/gemmology_fwd.h`). The parallel-for is over output columns in
  chunks of 8.
- **Ceiling ~2×.** ~67% of time is the matrix multiply, but ~33% (the 14% argmax
  plus normalization/softmax/int8-conversion) is not, and won't speed up. Amdahl:
  `1/(0.33 + 0.67/N)` → ~2.0× at 4 cores, ~2.4× at 8. And it is *optimistic*,
  because the decoder's per-step layers are tiny (one token × 512×512) — coordinating
  cores costs more than the work. Realistically A pays only on the 32k projection and
  the encoder layers (which are big because the encoder does the whole sentence at
  once), giving ~1.5–2×.
- **Do not use gemmology's shipped `StdThreadExecutionEngine` as-is** — it spawns and
  joins fresh `std::thread`s on *every* `Multiply` call, and the decoder calls the
  kernel many times per word. Wire a *persistent* pool into the shim, and gate it on
  the output-column count so only the big multiplies thread.
- **Unique value: single-sentence latency.** It is the only option that speeds up one
  short sentence, because the decoder writes one word at a time and cannot run ahead.

### Option B — translate several sentences at once (data parallel)

A paragraph is several sentences, and each is translated independently. Put sentence
1 on core 1, sentence 2 on core 2, etc. Each core runs the whole pipeline (encoder,
decoder loop, projection, argmax) for its own sentence.

- **Ceiling ~linear.** 100% of the work is spread across cores — not just the matrix
  multiplies — so 4 cores ≈ 4× throughput, until you run out of sentences to hand out
  or saturate memory bandwidth.
- **Memory fits the constraint.** The big thing in RAM is the model's weights (the
  embedding table + all weight matrices, ~150 MB), read-only after load, so **all
  cores share one copy**. Each core needs only its own scratch for the sentence it is
  on — current token vectors, the small SSRU cell-state, a 32k score buffer:
  kilobytes. So N cores ≈ *one* model + N × (a little scratch). This is "parallelize
  the structure without duplicating the graph."
- This mirrors marian's CPU model: one shared set of weights, one worker per
  `cpu-threads`, each translating a different sentence. (marian's own int8 kernel is
  single-threaded per call — its speed comes from this data parallelism, not from
  splitting one multiply.)
- **Obstacle is structural, not mathematical.** Today the per-sentence scratch lives
  *inside* the shared `Weights` (interior-mutability `RefCell`s: `affine_cache`,
  `scratch`, `logits_scratch`), and `PreparedB` holds a raw C++ pointer, so it is
  `!Send + !Sync`. The refactor: separate the immutable model (packed weights,
  prepared biases, embeddings — shareable via `Arc`) from the per-sentence scratch
  (owned per worker); pre-build the lazy affine biases at load; and `unsafe impl
  Sync` on the packed handle for reads, justified because the shim allocates its
  A/bias/output scratch per call, so concurrent `Multiply`s on one `PreparedB` only
  *read* the packed buffer. Results stay bit-for-bit identical, so the oracle-parity
  and batch-invariance tests still hold (see [inference-rs-validation]).

## Recommendation

- **Throughput (translate a document / many sentences faster): do Option B.** Highest
  ceiling, fits the memory constraint (one shared model + tiny per-thread scratch),
  and it is the proven approach. Work = the "model vs scratch" refactor + a thread
  pool handing out one sentence per core.
- **Single-sentence latency (one short sentence back fast, e.g. interactive): Option
  A**, on the two big multiplies only (the 32k projection and the encoder layers),
  with a *persistent* worker pool so we don't pay thread-startup on every decoder
  pass. Never on the small decoder layers.
- **Combine but don't nest.** If each sentence already owns a core (B), don't also
  have that core split its matrices (A) — they fight over the same cores. Pick one
  axis per workload, or size them jointly.

## Constraints to respect

- **wasm / default path stays single-threaded.** Firefox's shipping path is
  single-threaded wasm; feature-gate any pool so the default and wasm builds are
  unchanged.
- **Library-first.** Expose a thread-count knob rather than baking in a policy; an
  embedder may run its own parallelism across engine instances.
- **Bit-exactness.** Both options are bit-identical by construction (Option A
  partitions disjoint output columns with no cross-thread float reduction; Option B
  runs independent sentences), so the cheat-proof oracle-parity suite remains the
  gate. Avoid introducing any cross-thread floating-point reduction, which would
  break exactness.

## Next step

Prototype Option B: split `Model` from `Scratch`, share `Arc<Model>` across a small
pool, translate one sentence per core, and measure throughput vs the current
single-core numbers with `scripts/perf.py --blocks`.

## Plan

Both options ship, each behind its own cargo feature, both **off by default** so the
default build, the wasm path, and reproducible/audited builds stay single-threaded
and deterministic. The features are independent and compose; a runtime thread-count
knob (default = `available_parallelism`, `1` disables) controls how many cores are
actually used.

| feature | option | effect | new deps |
|---|---|---|---|
| `threads` | B | translate a batch's sentences across cores, one worker per core, sharing one copy of the weights | none (`std::thread::scope`) |
| `gemm-threads` | A | split each *large* int8 matrix multiply across cores | none (persistent pool in the C++ shim); requires `gemmology` |

### Phase 1 — make the engine shareable (single-threaded, no behaviour change)

The blocker to sharing one `Engine`/`Weights` across threads is interior mutability.
Remove it without changing any math (so every oracle-parity + batch-invariance test
stays bit-for-bit green):

1. `AffineWeight.bias: Option<Vec<f32>>` → `std::sync::OnceLock<Vec<f32>>`. The bias
   is still built lazily on first use (its name is only known at call time), but
   `OnceLock` is `Sync`, so the cache no longer needs `&mut`.
2. `affine_cache: RefCell<HashMap<..>>` → plain `HashMap<..>` (the only mutation was
   the lazy bias, now handled by `OnceLock`).
3. `proj_pb: RefCell<Option<PreparedB>>` → plain `Option<PreparedB>` (it is built
   once at load and only read afterwards).
4. Move the reusable activation scratch out of the shared object into thread-locals:
   `Weights`'s `GemmScratch` (`a_u8`, `wemb_row`) and `Engine`'s `logits_scratch`
   become `thread_local!` statics. Each thread reuses its own buffers, so the
   single-thread hot path keeps its zero-per-call-alloc behaviour and threads never
   share a buffer.
5. `unsafe impl Sync for PreparedB` in the SIMD module, justified: the packed weight
   is read-only after load, and the shim allocates its A/bias/output scratch per
   call, so concurrent `Multiply`s on one `PreparedB` only *read* the packed buffer.
6. Add a compile-time `assert_sync::<Engine>()` under the `threads` feature so the
   invariant can't silently regress.

### Phase 2 — Option B (`threads`)

7. Add the `threads` feature. Add a runtime knob (e.g. `Engine::with_threads(n)` /
   an env override in the CLI), default `available_parallelism`.
8. Parallel `greedy_batch` / `translate_batch`: partition the batch's sentences into
   `n` chunks and run each chunk on a `std::thread::scope` worker that borrows
   `&Engine` and calls the existing per-sentence path. Order-preserving. When `n == 1`
   or the feature is off, keep the current serial loop.
9. Validate: batched output must stay identical to the serial path (extends the
   existing batch-invariance test). Benchmark `perf.py --blocks`.

### Phase 3 — Option A (`gemm-threads`)

10. Add the `gemm-threads` feature (implies `gemmology`). Add a persistent thread
    pool in `gemmology_shim.cpp` that implements gemmology's execution-engine
    interface (`operator()(start, end, stride, fn)`): threads created once, parked on
    a condition variable, woken to run column ranges. No OpenMP / libomp.
11. Route only *large* multiplies through it — thresholded on the output-column count
    (`B_cols`) so the 32k projection and the encoder layers thread while the small
    per-step decoder multiplies stay sequential.
12. Validate against `gemm_parity` (bit-identical to the sequential kernel) + the full
    oracle suite. Benchmark.

### Phase 4 — report

13. Measure throughput (words/s, sent/s) and memory (settled/peak RSS; dhat live
    heap) for: single-thread baseline, Option B, Option A, and B+A, across a core
    sweep. Write the results back into this note and the crate README.

**Coordination (when both are on).** Prefer B across the sentences available, and
fall back to A only when there aren't enough sentences to fill the cores (e.g. a
1-sentence batch → A across all cores). This enforces "don't nest" automatically.

## Results (implemented)

Both options shipped as off-by-default features (`threads`, `gemm-threads`); the
default build, wasm, and reproducible builds are single-threaded and unchanged.
Every measurement below was checked **bit-identical to the single-thread output**
(`diff` of the full translation at each thread count), so the cheat-proof
oracle-parity and batch-invariance suites still hold — threading moves no logit.

**Setup.** en→ru `base` model (dim-emb 512, FFN 2048, 6 enc / 2 SSRU dec, tied 32k),
Firefox's *Frankenstein* corpus (`corpora/frankenstein-en.blocks.txt`: 103 blocks /
403 sentences / 9,513 source words), jemalloc allocator, release build. Machine: an
18-core Apple Silicon (**6 performance + 12 efficiency** cores) — the P/E split
shapes the curves. Throughput is source-words ÷ wall-clock over the whole corpus
(model load excluded); peak RSS is `/usr/bin/time -l` maximum resident set size;
median of 3 runs after a warmup.

### Option B — data-parallel across blocks (`threads`)

Whole blocks distributed across worker threads sharing one read-only `&Engine`
(bounded per-block batches; the production "translate a document" shape).

| threads | words/s | speedup | peak RSS |
|--------:|--------:|--------:|---------:|
| 1  | 1276 | 1.00× | 151 MiB |
| 2  | 2045 | 1.60× | 210 |
| 4  | 3163 | 2.48× | 270 |
| 6  | 4048 | 3.17× | 281 |
| 8  | 4951 | 3.88× | 367 |
| 12 | 6243 | 4.89× | 437 |
| 18 | 8448 | 6.62× | 491 |

- The `threads=1` figure (1276 words/s, 151 MiB) matches the README's single-thread
  baseline (1267 / 149), confirming the new path adds no serial overhead.
- **Near-linear through the 6 performance cores** (3.17× at 6), then a shallower
  climb as work spills onto the slower efficiency cores — still **6.6× at 18**. This
  is the shape to expect on a heterogeneous CPU; on 6 homogeneous cores you'd see
  ~5–6× flat.
- **Memory grows modestly and as predicted:** the ~130 MiB of resident weights are
  shared once; each extra in-flight block adds only its activation state (SSRU cells,
  cross-attention K/V, the 32k logits buffer). 18 workers → 491 MiB, i.e. **~19
  MiB/worker on top of one shared model**. The naïve alternative — one `Engine`
  (hence one copy of the weights) per thread — would cost ~18 × 151 ≈ **2.7 GiB**.
  Shared weights save ~2.2 GiB at 18 threads; that is the whole point of the Phase-1
  refactor.

### Option A — intra-op GEMM parallelism (`gemm-threads`)

The output-projection and encoder matmuls split across cores inside one `Multiply`
(pool size via `FXT_GEMM_THREADS`); the batch stays serial (`--threads 1`).

| gemm_threads | words/s | speedup | peak RSS |
|-------------:|--------:|--------:|---------:|
| 1 | 1256 | 1.00× | 151 MiB |
| 2 | 1658 | 1.31× | 155 |
| 4 | 2097 | 1.66× | 150 |
| 6 | 2087 | 1.66× | 156 |
| 8 | 2198 | 1.74× | 153 |

Single-sentence decode latency (m=1, the case Option B can't help): **32.2 ms → 19.2
ms at 4 cores = 1.68×**.

- **Caps at ~1.7×**, matching the Amdahl estimate: ~2/3 of the time is the matmul, but
  ~1/3 (argmax over 32k logits, layernorm, softmax, int8 conversion) doesn't
  parallelize, and the decoder's small per-token matmuls stay sequential by design.
  It plateaus at 4–6 cores — beyond that the serial third dominates.
- **Peak RSS is flat (~151–156 MiB)** across all pool sizes: intra-op threading adds
  only thread stacks, no activation state. This is the memory-free lever.

### Reading the two together

The measured contrast is exactly the pre-implementation analysis:

| | throughput ceiling | memory cost | helps a single sentence? |
|---|---|---|---|
| **B** (`threads`) | high — 6.6× @ 18 cores | +~19 MiB/worker | no (decode is sequential) |
| **A** (`gemm-threads`) | ~1.7×, plateaus @ 4–6 | ~zero (flat RSS) | **yes** — 1.68× decode latency |

So: **Option B for throughput** (a server / document translator with many sentences
in flight) — highest ceiling, memory-lean via shared weights. **Option A for
single-sentence latency** (interactive, one short input) — the only lever when there
is no batch width, at no memory cost. They compose but must not nest (each sentence
on its own core *or* one sentence split across cores, not both); the recommended
policy is B across available sentences, A only when a batch can't fill the cores.

### Cross-architecture confirmation (GitHub CI)

The tables above are ARM (Apple Silicon). CI extends the check to the other SIMD
arch. Two levels:

- **Permanent (no model):** the `test` matrix (`ubuntu-24.04-arm` i8mm +
  `ubuntu-latest` x86 AVX2) runs the vocab-scale `gemm_parity` shape through the
  Option A pool with `FXT_GEMM_THREADS=4` and asserts it stays bit-identical to the
  scalar kernel — proving the intra-op pool is correct on **AVX2**, not just i8mm.
  `intra-op GEMM pool: threads=4 backend=avx2 … max_diff=0.00e0`.
- **One-shot end-to-end (temporary committed model, since force-removed):** a real
  en→fr model was committed on a throwaway commit so CI could run the engine
  end-to-end on both runners (4 vCPU each), then force-pushed away. Output was
  **bit-identical** between 1 and 4 threads on both arches. Throughput on
  `corpora/nllb-en-fr.blocks.txt` (307 blocks / 1000 sentences / 15,856 words):

  | | ARM i8mm (4 vCPU) | x86 AVX2 (4 vCPU) |
  |---|---|---|
  | **Option B** 1 → 2 → 4 threads | 950 → 1829 → 3590 w/s (**3.78×**) | 656 → 1292 → 1716 w/s (**2.62×**) |
  | **Option A** pool 1 → 2 → 4 | 953 → 1152 → 1442 w/s (1.51×) | 657 → 789 → 793 w/s (1.21×) |

  Lower absolute scaling than the 18-core table because these runners have 4 vCPUs
  (Option B tracks cores; Option A hits its Amdahl plateau even sooner with so few).
  The point of the run was cross-arch **correctness + that it scales on x86 AVX2**,
  which the author can't test locally — both hold. To reproduce, re-commit a small
  model under `data/models/<pair>/` and add a temporary block-harness CI step (the
  model isn't kept in-tree).

### How to reproduce

```
# Option B (data-parallel blocks): 1..N workers, shared weights
cargo build --release -p fxtranslate-oracle --features fast,threads
target/release/fxtranslate-oracle translate <model.bin> <vocab.spm> \
    --blocks corpora/frankenstein-en.blocks.txt --threads 8 --timing   # -> [run] words/s

# Option A (intra-op GEMM): batch serial, pool size via env
cargo build --release -p fxtranslate-oracle --features fast,threads,gemm-threads
FXT_GEMM_THREADS=6 target/release/fxtranslate-oracle translate <model.bin> <vocab.spm> \
    --blocks corpora/frankenstein-en.blocks.txt --threads 1 --timing
```

Raw sweeps: `artifacts/thread-scaling-B.txt`, `artifacts/thread-scaling-A.txt`
(gitignored). Correctness at every thread count verified by `diff` against the
`--threads 1` / `FXT_GEMM_THREADS=1` output, plus the `threads` batch-invariance test
(`tests/batched_decode.rs`) and `gemm_parity` with the pool active.

### Follow-ups not taken here

- CI now runs the Option A pool (bit-identical to scalar) on both SIMD arches via a
  raw-cargo step in the `test` job, but `task rs:check` / `scripts/check.py` (the
  local default) still only builds the single-thread config. Folding a
  `threads,gemm-threads` lane into `check.py` (kept in sync with the workflow via
  `lint-ci`) would guard the parallel paths locally too.
- End-to-end Option B parity + throughput on CI needs a model, which isn't kept
  in-tree; it was confirmed once via a throwaway committed model (see above) but
  isn't part of the standing CI (which skips model-dependent tests).
- The automatic B-vs-A coordination policy is documented but not wired: today the
  caller picks (`--threads` for B, `FXT_GEMM_THREADS` for A). A single knob that
  spends cores on B first and falls back to A for thin batches is the natural next step.
- `Engine::with_threads` (library Option B for a single big `translate_batch` call) is
  shipped and unit-tested but not the harness's measurement path; the harness measures
  block-level parallelism, which is the realistic document workload and bounds memory.
