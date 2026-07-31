# Clean-room ONNX exporter (en-fr student)

A from-scratch Python exporter that rebuilds the `inference-rs` en-fr student
translation model as ONNX graphs, verifies them against a numpy float golden and
the production `inference-rs` int8 engine, and quantizes them to int8 for
shipping. The `inference-rs` engine is ground truth; no external exporter is
trusted. See `SPEC.md` for the full architecture (post-norm transformer encoder,
SSRU decoder, tied embeddings, sinusoidal PE baked as a constant).

## Pipeline

```
float .npz  →  build ONNX (float32)  →  quantize_dynamic  →  int8 ONNX  →  ship
              └─ validate: onnx-float vs numpy-float (~1e-5) ─┘
              └─ cross-check: numpy-float vs inference-rs int8 (argmax agree) ─┘
                                                          └ quality: chrF / overlap
```

The float32 graphs are a dev scaffold validated first so quantization is not a
confounding variable; the shipped artifact is the int8 graph.

## Files

- `model_npz.py`, `tokenizer.py` — validated npz/config + SentencePiece support.
- `numpy_ref.py` — float32 forward pass (the bit-close golden).
- `export_encoder.py`, `export_decoder.py` — ONNX graph builders → `models/*.onnx`.
- `engine.py` — the ONNX translation engine: both ORT sessions + the greedy decode
  loop (the loop runs in the driver, not in-graph). `translate(text, int8=False)`;
  the CLI translates a positional argument or stdin, with `--int8` for the quantized graphs.
- `quantize.py` — `onnxruntime.quantization.quantize_dynamic` → `models/*.int8.onnx`.
- `quality.py` — engine-vs-engine quality + rough perf.
- `validate_numpy.py` (Gate 1), `validate_onnx.py` (Gate 2).
- `requirements.txt` — the eval's Python deps (installed into the project `.venv`).

## How to run

Driven through the Taskfile (namespaced `rs:onnx-*`, like the other inference-rs tasks);
run from anywhere in the repo. The first run provisions the deps into the project-local
`.venv` (`rs:onnx-venv`) and fetches the float student model from GCS into
`data/models/en-fr/student-finetuned/` (`rs:onnx-download-model`) — both are dependencies
of the tasks below, so you don't call them directly, but the model download is also
available on its own:

```sh
task rs:onnx-download-model         # fetch the float student .npz + vocabs (auto-run below)
task rs:onnx-export                 # build the float32 graphs from the student .npz
task rs:onnx-translate -- "Hello."  # float translate (positional text or stdin)
task rs:onnx-dump                   # write inference-rs reference tensors → testdata/
task rs:onnx-validate               # Gate 2 (vs numpy) + Gate 1 (vs inference-rs)
task rs:onnx-quantize               # → models/*.int8.onnx
task rs:onnx-translate -- --int8 "Hello."   # int8 translate
task rs:onnx-quality                # chrF / overlap + rough perf table
```

`rs:onnx-quality` and `rs:onnx-dump` use the inference-rs engine as ground truth; they
build the release oracle (`fxtranslate-oracle`) on demand. To run a script directly for
debugging, use the same interpreter the tasks do: `../.venv/bin/python3 engine.py`.

## Validation results

**Gate 1 — numpy-float vs inference-rs int8** (argmax must agree; ~1% tensor diff
expected from int8): first-step argmax `16060 == 16060` **PASS**. Encoder abs
diff max 0.156 / mean 0.029; logits max 2.47 / mean 0.45 (consistent with int8).

**Gate 2 — onnx-float vs numpy-float** (tol 1e-4): encoder context max ~3e-6;
decode logits/states max ~3–4e-5; end-to-end string match on both fixtures
**PASS**. Fixed sentence float ONNX:
`Bonjour, monde. Ceci est un test du moteur de traduction.`

**int8 translation** matches float exactly on the fixed sentence:
`Bonjour, monde. Ceci est un test du moteur de traduction.`

### Quantization (`quantize.py`)

Dynamic quantization, `weight_type=QInt8`, per-tensor, `op_types_to_quantize=["MatMul"]`.

| graph | float MB | int8 MB | ratio |
|---|---|---|---|
| encoder | 92.4 | 58.7 | 1.57x |
| decode_step | 121.0 | 65.6 | 1.84x |
| **total** | **213.4** | **124.3** | **1.72x (−41.8%)** |

The ratio is capped by the tied embedding table `Wemb` (46 MB), which feeds the
`Gather` embedding lookup and stays float32 under dynamic quantization (per ORT
guidance embeddings are not quantized). The output-projection copy `Wemb_T` and
all FFN/attention MatMul weights *are* quantized (e.g. `Wemb_T` 46 → 11 MB). A
future win is deduplicating the tied table or int8-packing the Gather.

### Quality — engine vs engine (50 varied English sentences, `corpora/nllb-en-fr.txt`)

No parallel en→fr reference set exists in the repo, so this is engine-vs-engine
with float ONNX as the high-quality anchor (not absolute BLEU).

| pair | chrF | exact | token overlap |
|---|---|---|---|
| int8-ONNX vs float-ONNX | 98.39 | 82% | 96% |
| inference-rs vs float-ONNX | 97.40 | 74% | 95% |
| int8-ONNX vs inference-rs | 97.55 | 74% | 95% |

int8-ONNX tracks the float reference *more* closely (chrF 98.39) than the
production intgemm engine does (97.40): ORT dynamic int8 is quality-competitive.

### Perf (indicative only — ORT defaults, single-threaded, mean ms/sentence)

| engine | ms/sent |
|---|---|
| float ONNX | 16.6 |
| int8 ONNX | 9.8 |
| inference-rs | 119.7 (one process for the whole batch, includes model load) |

Not a rigorous benchmark: ORT uses default session settings and the inference-rs
figure is a single cold-loaded process amortized over the batch. int8 ONNX is
~1.7x faster than float ONNX here.
