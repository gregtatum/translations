# Porting the translation models to ONNX / ONNX Runtime

## Why this note exists

Companion to `notes/14-ggml-llamacpp-port.md`. Same underlying question — *can we
retire the bespoke `inference-rs` engine by running the translation models on a
runtime Gecko already ships?* — but evaluated for **ONNX + ONNX Runtime (ORT)**
instead of llama.cpp.

Firefox already ships ORT inside **`toolkit/components/ml`** (the "Firefox AI
Runtime": transformers.js + ONNX Runtime, used for PDF.js alt-text and smart tab
grouping). So ORT is a second runtime-already-in-Gecko candidate alongside
llama.cpp. Read this next to note 14; the two share the model facts, the
where-to-get-the-models section, the "requantize from the float `.npz`" plan, the
quality-equivalence-not-bit-exactness stance, and the tokenizer-parity gate. This
note only covers what's *different* about the ONNX route.

Same as note 14: **not** retraining; requantizing from the pre-quant float student
`.npz` is on the table; GPU is not the point (per-sentence, small batch → CPU-first).

For the model architecture (encoder-decoder, SSRU decoder, ~32–50k SPM vocab) see
notes 05 and 14. For where the float models live on GCS see note 14's
"Where to get the models" (the same `student-finetuned/…best-*.npz`).

## Headline — ONNX export is feasible, but it's a project, not a one-liner

**Feasibility verdict: exporting these models to ONNX is feasible.** The model is a
standard transformer encoder + an SSRU decoder whose every op maps to stock ONNX
ops, and the autoregressive recurrence is host-driven, so no ONNX `Loop`/`Scan`
control flow is needed. The open question is *which tool emits the graph*, not
*whether the graph is expressible.*

**But "just run the exporter" is wrong — that framing (an earlier draft of this
note) was too glib, and a past hand attempt at exporting failed.** Two things are
true at once and must not be conflated:

1. *The SSRU decoder is exportable in principle.* Marian's own `USE_ONNX` exporter
   was built with it in mind — verified in `~/dev/marian-dev`:
   - `src/onnx/expression_graph_onnx_exporter.cpp` emits `encode_source()`,
     `decode_first()`, `decode_next()`; `decode_next(…, decoder_state_0, …) →
     logits, decoder_state_0, …` threads the SSRU cell state as **explicit graph
     inputs/outputs** (`decoderState->getStates()` → `d.output`,`d.cell`, lines
     90–96). Host drives the token loop; no control flow in the graph.
   - The SSRU gate op is explicitly handled:
     `expression_graph_onnx_serialization.cpp:317` expands `highway` → `sigmoid` →
     Mul/Add/Sub, with the tell-tale comment (line 208) *"The only sigmoid in the
     system comes from highway."* `highway` **is** the SSRU gate — so the exporter
     targets these models, not just standard transformers. (`ReLU→Relu`,
     `affine→dot/MatMul`, `sin→Sin` are likewise mapped.)
   - The one genuine RNN limitation (`exporter.cpp:28`, "inner recurrences, e.g. an
     RNN **encoder**, are not supported… cannot export control flow") is about a
     recurrent *encoder* (an in-graph time loop). Our encoder is a standard
     transformer, so it does not apply.

2. *The exporter is a fragile, aged research artifact* — which is why a real attempt
   can fail. From the same serialization file:
   `@BUGBUG: Gemm always crashes with ONNX runtime … we will NEVER get here` (348,
   888) so it avoids `Gemm`; faked batched gather with
   `ABORT_IF("ONNX does not support batched select()")` (403); a hard
   `ABORT("ONNX export of operation {} is presently not supported")` (662) for any
   unmapped op; the sentinel input-length dim **`97`** hack (`exporter.cpp:66`) that
   silently breaks on collision; and `transformer.h:95`, where under `USE_ONNX`
   sinusoidal positions switch from a constant to a computed `Sin` with a `TODO`
   noting *"'Sin' op and constant sine generate different result"* — a correctness
   landmine. Plus the `USE_ONNX` build (protobuf) and ORT-version drift.

So the export is a real project with a fragile reference implementation, **not** a
one-liner.

## Build routes (how to emit the graph)

- **A. Revive / patch the ancient C++ `USE_ONNX` exporter.** Build marian-dev with
  `USE_ONNX=ON`, run `marian-conv --export-as onnx-{encode,decoder-step,…}` on the
  float `.npz`, then patch whatever aborts/miscomputes (sentinel-dim, Sin
  divergence, any unmapped op). *Pro:* highest fidelity to Marian's real graph;
  SSRU already wired. *Con:* fighting decade-old research C++ with pervasive
  `@BUGBUG`s and a protobuf build.
- **B. Clean-room builder in Python (`onnx.helper`).** Emit the same
  `encode` / `decode_step` graph shape from scratch: `make_node`/`make_graph`,
  weights as initializers via `numpy_helper.from_array`. An SSRU step is ~2 matmuls
  + sigmoid gate + elementwise recurrence — pure feed-forward. *Pro:* mature,
  well-documented ONNX tooling; full control; no C++ build. *Con:* we re-encode the
  op sequence + weight-name mapping ourselves (the ancient exporter is the reference
  for the exact graph, esp. the Sin/posrange handling and the decode-state I/O).
- **C. Clean-room exporter in Rust, in `inference-rs`.** `inference-rs` already has
  the exact model (`model.rs` parser, `ops.rs` math, `engine.rs` encoder + decode
  step). A Rust exporter would read the float `.npz` and emit ONNX protobuf
  (generate the `onnx.proto` types via `prost-build`, or a thin hand-written
  serializer). *Pro:* single source of truth for the op sequence, in-repo, and we
  can validate the emitted graph against the Rust engine's own intermediate tensors
  in-process (trivial numeric diffing). Fits the existing Rust stack; "something
  novel" but low-risk since the op set is tiny. *Con:* ONNX-writing tooling in Rust
  is less mature than Python (mostly boilerplate, not hard). Needs a float `.npz`
  reader (numpy zip).
- **Rejected: PyTorch + `torch.onnx.export`.** No off-the-shelf SSRU `nn.Module`;
  tracing unrolls Python loops so you'd export a single step and host-loop anyway —
  adds a reimplementation surface, buys nothing over A/B/C.

**Recommended de-risking order:** start with a *minimal* clean-room build (route B or
C) of just the **encoder**, load in ORT, and numeric-diff against `inference-rs`.
That proves the whole toolchain (build graph → quantize → ORT → matches) without
fighting the C++ exporter, and it's a prerequisite for the decode-step graph either
way. Keep the ancient C++ exporter as the graph-shape reference (and reconsider
route A only if clean-room fidelity to Marian's exact numerics matters).

No prior art models an SSRU in ONNX: HF optimum's Marian export and community
Marian→ONNX converters all wrap HF `MarianMTModel`, which uses a **standard**
self-attention + KV-cache decoder — not our SSRU. We'd be first; validate against
Marian/`inference-rs`.

## Graph shape (a solved pattern)

- **Single-step decoder graph; SSRU state as graph inputs/outputs; generation loop
  in host code. No ONNX Loop/Scan.** This is exactly the established ORT seq2seq
  pattern (HF optimum ships `encoder_model.onnx` + `decoder_model.onnx` +
  `decoder_with_past_model.onnx`, threading `past_key_values` as graph I/O). Our
  "past" is the SSRU cell vector per layer instead of a KV cache — simpler. Marian's
  `decode_next` already produces exactly this shape. Loop/Scan would only be needed
  if the whole generation loop had to live inside one graph (no host driver) — not
  our case.
- **All ops are standard main-domain ONNX ops** (Sigmoid, Mul, Sub, Add, Relu,
  MatMul, Gather, Softmax, LayerNormalization). `LayerNormalization` became standard
  at **opset 17** — that's the hard floor. Practical target opset 17–20.
- **No custom ops** ⇒ the graph runs on stock ORT everywhere with no partitioning
  surprises. *This is the key maintenance property* (see comparison below).

## Quantization

Requantize from the float `.npz`. ORT explicitly recommends **dynamic int8
quantization for transformer/RNN models** (`quantize_dynamic`, no calibration
data), **QDQ format** (default; avoid S8S8+QOperator, documented slow on x86),
**per-tensor** first, switching weights to per-channel only if accuracy drops. ORT
gives an accuracy knob intgemm doesn't (optional per-channel weight scales), so it
can match/beat our current per-tensor intgemm at some throughput cost. As in note
14, this means quality-equivalence validation (chrF/BLEU + token-overlap), not
bit-exactness with the marian oracle.

Perf caveats worth an early measurement: int8 wins need **VNNI on x86** (on
AVX2/AVX-512 without VNNI, int8 can be *slower* than fp32) and **I8MM on ARM** (M2+
have it, M1 has DOTPROD only). ORT's MLAS has I8MM QGEMM kernels by default since
≥1.17. Benchmark Apple Silicon ourselves — no published ORT M-series int8 number
found, and int8 is not guaranteed to beat our tuned `gemmology` path.

## Runtime / execution providers

- **CPU EP is the universal fallback** — ORT guarantees all ops on the default EP.
  Our custom-op-free graph runs with zero coverage risk. For tiny models this is
  likely the sweet spot.
- **onnxruntime-web (wasm)** runs the whole graph (WASM backend supports all ops;
  WebGL/WebGPU/WebNN only a subset). Levers: threads need `crossOriginIsolated`
  (COOP/COEP headers) or you're single-threaded; SIMD on by default. Decode-loop
  gotcha: **don't allocate a fresh tensor/OrtValue every step** — preallocate/reuse
  buffers (IO-binding), or per-step overhead dominates for a tiny model.
- **CoreML / WebGPU are benchmark-it experiments, not defaults.** Accelerator EPs
  fall back to CPU per-op, but an unsupported op mid-decoder fragments the graph and
  the CPU↔accelerator copies can make it net-slower than CPU-only — a real risk for
  a small, fragmentable decoder.

## Firefox angle (verified against `~/dev/firefox`)

Two distinct engines in Gecko; don't conflate them:

- **`toolkit/components/translations`** — today's production Translations feature,
  running the **Bergamot WASM engine** (Marian fork). No ONNX. This is where our
  model ships now. (`ml/docs/architecture.md:7-9` explicitly notes Translations
  "has its own separate architecture.")
- **`toolkit/components/ml`** — the Firefox AI Runtime. `content/backends/Pipeline.mjs`
  dispatches by backend: `onnx` (wasm) and **`onnx-native`** (native C++ ORT) both
  go through **`ONNXPipeline.mjs`, which is transformers.js**; `llama.cpp` goes
  through `LlamaCppPipeline.mjs` → native `LlamaRunner`; plus an OpenAI backend.
  **Default backend is `onnx-native`** (`Pipeline.mjs:37`).

Two corrections to an earlier draft of this note:

- **`onnx-native` does *not* give us a clean non-transformers.js path.** Native ORT
  only swaps the *session execution* for wasm inside `ONNXPipeline.mjs`; the seq2seq
  **generation loop is still transformers.js** either way. And transformers.js
  expects the HF-optimum encoder-decoder contract (KV-cache `past_key_values`,
  `decoder_model_merged*.onnx` — see `ml/docs/models.md:29`; `translation` defaults
  to `Xenova/t5-small`). Our SSRU exports `decoder_state_*`, which transformers.js's
  generation code doesn't know how to drive. So the custom glue — patch
  transformers.js, or bypass it with our own JS loop over the ORT sessions — is
  needed for **both** `onnx` and `onnx-native`, and it lives in **JS**.
- **`llama.cpp` is already a shipping backend here**, not "being integrated" (as
  note 14's phrasing implied). It runs via a native `LlamaRunner` with its **own C++
  generation loop** (text-generation / OpenAI-compatible), independent of
  transformers.js — see note 14.

**The real asymmetry:** ONNX's custom generation glue for the SSRU decoder lives in
**JS (transformers.js)** and is identical for wasm and native; llama.cpp's would
live in **C++** (the new arch + teaching `LlamaRunner`/llama.cpp encoder-decoder).
Also note the ml engine gates internal features behind a Remote Settings allowlist
(`ml-inference-options`); WebExtensions can supply models more freely.

## Tokenizer

Same gate as note 14 (SPM parity), same conclusion: keep tokenization **out of
graph**. Native path reuses our existing Rust SentencePiece (`spm.rs`). A web/
transformers.js path would use `@huggingface/tokenizers` unigram + a `Precompiled`
normalizer for the SPM `precompiled_charsmap`, fed from a converted `tokenizer.json`.
In-graph SPM (onnxruntime-extensions) is possible but not worth it and poorly
documented for the browser.

## Feasibility gates (do these before committing)

1. **A valid graph exists end-to-end.** Emit an `encode`/`decode_step` graph (route
   B or C, encoder first), load in ORT, numeric-diff against `inference-rs`. Proves
   the toolchain (build → quantize → ORT → matches) and resolves the real unknowns
   (Sin/posrange handling, unmapped ops, sentinel-dim). If reviving the C++ exporter
   (route A), this is where its `@BUGBUG`s surface.
2. **CPU perf, full-vocab.** Same as note 14 gate #1 — full-vocab both sides
   (shortlist is off in production; see note 14). Benchmark ORT CPU (dynamic int8)
   vs `inference-rs`; benchmark Apple Silicon int8 specifically.
3. **Tokenizer parity** — identical to note 14 (SPM normalization vs the host
   tokenizer used in the ORT pipeline).
4. **Integration target = the transformers.js contract.** The ml component drives
   seq2seq through transformers.js for *both* `onnx` and `onnx-native`, and it
   assumes a KV-cache decoder. Decide how to drive an SSRU decoder (`decoder_state_*`
   I/O): patch transformers.js, or run our own JS generation loop over the ORT
   sessions. This — not the graph — is where the SSRU costs us in Firefox.

## ONNX vs llama.cpp (since we've now evaluated both)

| Axis | ONNX + ORT | llama.cpp |
|---|---|---|
| Conversion | Emit graph (revive C++ exporter, or clean-room Python/Rust builder) | Write a new arch + GGUF converter from scratch |
| Runtime changes | **None** — stock ORT runs standard ops; no fork, no upstream | New `LLM_ARCH_MARIAN` inside the runtime (upstream or patch) |
| SSRU handling | State as graph I/O, host loop | Elementwise over recurrent memory, in-graph |
| Already in Gecko | `onnx` + `onnx-native` backends in `toolkit/components/ml` (default `onnx-native`) | `llama.cpp` backend in the same component (native `LlamaRunner`) |
| Generation loop | **transformers.js (JS)** for both wasm + native | `LlamaRunner` native C++ loop |
| Where the SSRU "custom glue" lives | **JS** — transformers.js can't drive `decoder_state_*` → patch it or bypass it | **C++** — the new arch + `LlamaRunner`/llama.cpp encoder-decoder support |
| Maintenance endgame | Converter + JS glue; **no runtime fork** | Upstreamed arch → ~0; else a rebased patch |

**The core tradeoff:** ONNX requires **no modifications to the runtime** — the model
is a graph of standard ops that stock ORT executes, so there's nothing to upstream
or fork. The custom code is the converter/exporter + the SSRU generation glue, which
in Firefox lives in **JS** (transformers.js, same for wasm and native). llama.cpp
requires adding and maintaining a new architecture *inside* the runtime (C++), but
its Gecko generation loop (`LlamaRunner`) is already native and transformers.js-free.
Neither is "drop-in"; they differ in **where** the SSRU-specific work lands (JS vs
C++) and whether you touch the runtime at all.

## Recommended next step

Cheapest decisive spike, and it doesn't require the C++ exporter: **emit a
clean-room ONNX graph of just the encoder** (route B in Python or route C in Rust)
from a float `.npz`, load it in ORT, and numeric-diff the encoder output against
`inference-rs`. That proves build→quantize→ORT→matches end-to-end, is the
prerequisite for the decode-step graph and the perf comparison, and sidesteps the
brittle `USE_ONNX` build. Revive the ancient C++ exporter (route A) only if we later
decide bit-fidelity to Marian's exact graph is worth it.

## Related

- `notes/14-ggml-llamacpp-port.md` — the llama.cpp evaluation + shared facts
  (model, GCS models, quantization stance, shortlist).
- `notes/05-fx-model-architecture.md` — full model architecture.
- `~/dev/marian-dev/src/onnx/` — the exporter (`USE_ONNX`).
