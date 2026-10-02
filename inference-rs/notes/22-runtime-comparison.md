## Runtime comparison

How the runtime-consolidation candidates actually compare, all in one harness run: the
same base en→ru model, the same Frankenstein block corpus (103 blocks / 9,513 words),
single thread, model-load excluded, on Apple Silicon — the note 09/16/19 methodology.

| engine | words/s | tokens/s | translate s | init ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| ONNX ORT (int8, .venv) | 2398 | 3363 | 3.97 | 206 | 445 |
| marian `block-bench` (native C++) | 1330 | 1864 | 7.15 | 57 | 298 |
| inference-rs (fast) | 1280 | 1795 | 7.43 | 67 | 150 |
| llama.cpp (q8_0, `LLM_ARCH_MARIAN`), 1 thread | 832 | 1167 | 11.43 | 86 | 195 |
| llama.cpp (q8_0, `LLM_ARCH_MARIAN`), 5 threads | 1852 | 2598¹ | 5.14 | — | — |
| ggml G1 (q8_0) | 758 | 1063 | 12.55 | 52 | 150 |
| Firefox Wasm (Full-Page) | 419 | 567 | 22.85 | 135 | 355 |

¹ 5-thread `translate s` is the sweep's encode+decode split (0.54 + 4.60); `tokens/s` is
derived from that run's words/s and the corpus's fixed token/word ratio. The per-thread
sweep did not record init or peak RSS, so those are left blank.

- **llama.cpp is threaded work, not single-thread work.** Its 832 wps at one thread already
  edges ggml G1 (758) with a materially faster encoder, and it peaks at **1852 wps @ 5t**
  (note 19's thread sweep) before regressing past 6t. Even at peak its memory (195 MiB at 1t)
  stays a third of ONNX and half of Firefox.
- **Methodology note.** This set was measured *without* the concurrent 20 ms RSS sampler
  (the truer compute figure), so it reports peak RSS only — no settled-RSS column — and its
  wps run ~2% above the sampled `notes/16` figures. It is one internally consistent run, not
  a splice of separate runs.

See `notes/19` (llama.cpp), `notes/18` (ggml G1), and `notes/16` (ONNX) for the full sweeps
and the encode/decode split.
