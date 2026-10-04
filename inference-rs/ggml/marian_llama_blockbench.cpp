// Perf driver: blockbench for the LLM_ARCH_MARIAN engine in llama.cpp.
//
// This mirrors the bare-libggml engine's `blockbench` mode (ggml/marian_ggml.cpp) EXACTLY
// so scripts/final_comparison.py's parse_blocks consumes its output unchanged:
//
//   - load the Q8_0 GGUF once (model load EXCLUDED from per-block timing; init measured
//     separately as wall - sum(compute) by the harness),
//   - read the SAME pretokenized source-id block file that engine uses (ggml/pretokenize.py:
//     blank-line-separated blocks, one sentence per line, space-separated ids incl. EOS),
//   - translate each block sentence-by-sentence via the two-phase greedy path
//     (llama_encode -> greedy llama_decode*), timing encode vs decode separately,
//   - emit one `[block] {json}` span per block on stderr in the identical format.
//
// Threading: --threads N sets llama_context_params.n_threads / n_threads_batch (also
// FXT_LLAMA_THREADS env for harness convenience). Correctness is settled by the decoder
// gate (byte-identical to bare-libggml); this binary changes nothing about the graph math, only
// measures it.
//
// Note on batching: like bare-libggml's encoder (which still runs once per sentence), the
// llama.cpp driver decodes one sentence at a time (single-sequence). That engine's headline
// finding is that
// THREADS, not batching, are the lever; this driver's thread sweep tests the same lever on
// llama.cpp's graph scheduler.
//
// M3b tried block-batched multi-sequence decode here (one packed llama_encode + lockstep
// llama_decode over B sentences, with the new per-sequence enc/cross masks in the arch) and
// found it a LARGE regression (peak ~1044 wps vs this driver's ~1852), because llama.cpp's
// enc-dec stash is one flat [n_embd, sum-of-source-lengths] tensor: every batched decoder
// token then attends over the WHOLE block's encoder tokens (B x the cross-attn FLOPs, all
// masked away). Fixing that needs per-sequence-scoped cross K/V in shared llama.cpp enc-dec
// infra, not a driver change -- see notes/19 "M3b". The arch masking (correct, and bit-
// identical for single-sequence) was kept; the batched driver was not. Stay single-sequence.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static llama_context * make_ctx(llama_model * model, int n_threads) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    // The encoder phase must retain its per-token output so it can be stashed for cross-attention.
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    return llama_init_from_model(model, cparams);
}

// Encode the source, then greedy-decode from decoder_start. Adds encode/decode wall time to
// the running counters. Returns the greedy output ids (matches the bare-libggml path).
static std::vector<llama_token> translate_one(llama_context * ctx, const llama_model * model,
                                              const std::vector<llama_token> & src, int max_new,
                                              double * encode_ms, double * decode_ms) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // fresh recurrent (SSRU) state + KV for this sentence.
    llama_memory_clear(llama_get_memory(ctx), true);

    // --- encoder phase ---
    double t0 = now_ms();
    llama_batch enc = llama_batch_init((int) src.size(), 0, 1);
    enc.n_tokens = (int) src.size();
    for (int i = 0; i < (int) src.size(); ++i) {
        enc.token[i]     = src[i];
        enc.pos[i]       = i;
        enc.n_seq_id[i]  = 1;
        enc.seq_id[i][0] = 0;
        enc.logits[i]    = 1; // per-token encoder output (cross-attn stash)
    }
    if (llama_encode(ctx, enc) != 0) { fprintf(stderr, "llama_encode failed\n"); exit(1); }
    llama_batch_free(enc);
    *encode_ms += now_ms() - t0;

    // --- decoder phase (greedy) ---
    llama_token start = llama_model_decoder_start_token(model);
    if (start == LLAMA_TOKEN_NULL) start = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);

    t0 = now_ms();
    std::vector<llama_token> out;
    llama_token cur = start;
    for (int step = 0; step < max_new; ++step) {
        llama_batch dec = llama_batch_init(1, 0, 1);
        dec.n_tokens = 1;
        dec.token[0]     = cur;
        dec.pos[0]       = step;
        dec.n_seq_id[0]  = 1;
        dec.seq_id[0][0] = 0;
        dec.logits[0]    = 1;
        if (llama_decode(ctx, dec) != 0) { fprintf(stderr, "llama_decode failed\n"); exit(1); }
        llama_batch_free(dec);

        const float * logits = llama_get_logits_ith(ctx, -1);
        int argmax = 0;
        float best = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > best) { best = logits[i]; argmax = i; }
        if (argmax == eos) break;
        out.push_back(argmax);
        cur = argmax;
    }
    *decode_ms += now_ms() - t0;
    return out;
}

static std::vector<llama_token> parse_ids(const std::string & line) {
    std::vector<llama_token> ids;
    std::istringstream ss(line);
    int x;
    while (ss >> x) ids.push_back((llama_token) x);
    return ids;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--blocks FILE] [--threads N]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * blocks_path = nullptr;
    int n_threads = 1;
    if (const char * t = getenv("FXT_LLAMA_THREADS")) n_threads = atoi(t);
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--blocks" && i + 1 < argc) blocks_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc) n_threads = atoi(argv[++i]);
    }
    if (!blocks_path) { fprintf(stderr, "blockbench needs --blocks FILE\n"); return 1; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    llama_context * ctx = make_ctx(model, n_threads);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    std::ifstream f(blocks_path);
    if (!f) { fprintf(stderr, "cannot open %s\n", blocks_path); return 1; }

    std::vector<std::vector<llama_token>> block;
    int idx = 0;
    auto flush = [&]() {
        if (block.empty()) return;
        double encode_ms = 0.0, decode_ms = 0.0;
        int src_tokens = 0, out_tokens = 0;
        for (auto & s : block) {
            const int max_new = (int) std::min<size_t>(256, (size_t) std::ceil(2.0 * s.size()) + 4);
            auto out = translate_one(ctx, model, s, max_new, &encode_ms, &decode_ms);
            src_tokens += (int) s.size();
            out_tokens += (int) out.size();
        }
        fprintf(stderr,
                "[block] {\"block\": %d, \"sentences\": %zu, \"src_tokens\": %d, "
                "\"tokens\": %d, \"encode_ms\": %.3f, \"decode_ms\": %.3f}\n",
                idx, block.size(), src_tokens, out_tokens, encode_ms, decode_ms);
        ++idx;
        block.clear();
    };
    std::string line;
    while (std::getline(f, line)) {
        bool blank = line.find_first_not_of(" \t\r\n") == std::string::npos;
        if (blank) { flush(); continue; }
        block.push_back(parse_ids(line));
    }
    flush();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
