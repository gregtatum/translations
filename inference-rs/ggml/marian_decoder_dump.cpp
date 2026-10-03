// End-to-end decoder driver: feed EXACT source ids to the llama.cpp Marian encoder-decoder and
// run the two-phase greedy loop (llama_encode -> llama_decode*), producing output ids and
// (on the first step) the full [vocab] logits vector for numeric parity against G1.
//
// This is the decoder analog of marian_encoder_dump.cpp. Tokenization stays in Python (the shared
// SPM), so this binary consumes/produces ids — identical inputs across every engine.
//
// modes:
//   dump   <model.gguf> <logits_out.bin> <id0> <id1> ...   encode + ONE decode step from the
//                                                           decoder_start token; write [vocab]
//                                                           logits, print argmax + output id.
//   decode <model.gguf> --                <id0> <id1> ...   encode + full greedy loop; print the
//                                                           space-separated output ids to stdout.
//
// The recurrent SSRU cell state is owned by llama.cpp's recurrent memory; we clear it per
// sentence with llama_memory_clear so each decode starts from a zero cell (matching G1).

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 512;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 512;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    // The encoder phase must retain its per-token output so it can be stashed for cross-attention
    // (the enc-dec stash memcpy's from the embeddings buffer), so enable embeddings with no pooling.
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    return llama_init_from_model(model, cparams);
}

// Encode the source, then run the decoder starting from decoder_start. Returns the greedy
// output ids; if first_logits != nullptr, fills it with the first decode step's [vocab] logits.
static std::vector<llama_token> translate(llama_context * ctx, const llama_model * model,
                                          const std::vector<llama_token> & src, int max_new,
                                          std::vector<float> * first_logits) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // fresh recurrent state + KV for this sentence.
    llama_memory_clear(llama_get_memory(ctx), true);

    // --- encoder phase ---
    llama_batch enc = llama_batch_init((int) src.size(), 0, 1);
    enc.n_tokens = (int) src.size();
    for (int i = 0; i < (int) src.size(); ++i) {
        enc.token[i]     = src[i];
        enc.pos[i]       = i;
        enc.n_seq_id[i]  = 1;
        enc.seq_id[i][0] = 0;
        enc.logits[i]    = 1; // request per-token encoder output (needed for the cross-attn stash)
    }
    if (llama_encode(ctx, enc) != 0) { fprintf(stderr, "llama_encode failed\n"); exit(1); }
    llama_batch_free(enc);

    // --- decoder phase (greedy) ---
    llama_token start = llama_model_decoder_start_token(model);
    if (start == LLAMA_TOKEN_NULL) start = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);

    std::vector<llama_token> out;
    llama_token cur = start;
    for (int step = 0; step < max_new; ++step) {
        llama_batch dec = llama_batch_init(1, 0, 1);
        dec.n_tokens = 1;
        dec.token[0]     = cur;
        dec.pos[0]       = step;      // absolute decoder position -> sinusoidal PE row
        dec.n_seq_id[0]  = 1;
        dec.seq_id[0][0] = 0;
        dec.logits[0]    = 1;
        if (llama_decode(ctx, dec) != 0) { fprintf(stderr, "llama_decode failed\n"); exit(1); }
        llama_batch_free(dec);

        const float * logits = llama_get_logits_ith(ctx, -1);
        if (step == 0 && first_logits) {
            first_logits->assign(logits, logits + n_vocab);
        }
        int argmax = 0;
        float best = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > best) { best = logits[i]; argmax = i; }

        if (getenv("MARIAN_DBG")) fprintf(stderr, "[step %d] cur=%d argmax=%d\n", step, (int) cur, argmax);
        if (argmax == eos) break;
        out.push_back(argmax);
        cur = argmax;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dump|decode> <model.gguf> <out.bin|--> <id0> <id1> ...\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    const char * model_path = argv[2];
    const char * out_path   = argv[3];

    std::vector<llama_token> ids;
    for (int i = 4; i < argc; ++i) ids.push_back((llama_token) atoi(argv[i]));

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    llama_context * ctx = make_ctx(model);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    if (mode == "dump") {
        std::vector<float> logits;
        auto out = translate(ctx, model, ids, /*max_new=*/1, &logits);
        std::ofstream o(out_path, std::ios::binary);
        o.write((const char *) logits.data(), logits.size() * sizeof(float));
        o.close();
        int argmax = 0; float best = logits[0];
        for (int i = 1; i < (int) logits.size(); ++i) if (logits[i] > best) { best = logits[i]; argmax = i; }
        fprintf(stderr, "[marian-dec-dump] vocab=%zu argmax=%d out=%s\n",
                logits.size(), argmax, out.empty() ? "(eos)" : std::to_string(out[0]).c_str());
    } else if (mode == "decode") {
        const int max_new = (int) std::min<size_t>(256, (size_t) std::ceil(2.0 * ids.size()) + 4);
        auto out = translate(ctx, model, ids, max_new, nullptr);
        std::string s;
        for (size_t i = 0; i < out.size(); ++i) { if (i) s += " "; s += std::to_string(out[i]); }
        printf("%s\n", s.c_str());
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
