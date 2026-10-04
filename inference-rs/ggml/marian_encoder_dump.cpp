// Encoder parity driver: feed EXACT source ids to the llama.cpp Marian encoder and dump the
// full per-token encoder context [n_embd, seq] to a binary file for numpy comparison
// against the bare-libggml / numpy_ref golden.
//
// This is Option (a)+(c): a tiny libllama driver that enables embeddings with
// pooling_type = NONE, so llama_encode writes the whole encoder output, retrieved via
// llama_get_embeddings(). No debug hook in the arch code — the encoder output is exposed
// through the standard embeddings API (same path BERT / T5ENCODER use). Nothing here is
// temporary; this stays on as the encoder regression harness.
//
// usage: marian_encoder_dump <model.gguf> <out.bin> <id0> <id1> ...

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <id0> <id1> ...\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    std::vector<llama_token> ids;
    for (int i = 3; i < argc; ++i) ids.push_back((llama_token) atoi(argv[i]));
    const int n_tokens = (int) ids.size();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;
    cparams.n_batch      = 512;
    cparams.n_ubatch     = 512;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE; // full per-token encoder output
    cparams.n_threads    = 1;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // Build a single-sequence batch with our exact ids; request output for every token.
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // output embeddings for all tokens
    }

    if (llama_encode(ctx, batch) != 0) {
        fprintf(stderr, "llama_encode failed\n");
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);
    const float * embd = llama_get_embeddings(ctx);
    if (!embd) { fprintf(stderr, "no embeddings returned\n"); return 1; }

    // llama.cpp lays out embeddings as [n_tokens, n_embd] row-major (token-major), matching
    // bare-libggml's ggml_encoder.bin ([seq, dim], dim contiguous). Write it straight through.
    std::ofstream o(out_path, std::ios::binary);
    o.write((const char *) embd, (size_t) n_tokens * n_embd * sizeof(float));
    o.close();

    fprintf(stderr, "[marian-dump] seq=%d n_embd=%d -> %s\n", n_tokens, n_embd, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
