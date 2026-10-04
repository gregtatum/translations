// Bare-libggml translation engine for the Marian/Bergamot student model (G1 of the
// ggml-port evaluation; see notes/18-ggml-port-design.md). This is the ggml analog of
// onnx/engine.py: it loads the GGUF written by convert_marian_gguf.py, builds a
// post-norm transformer encoder graph and an SSRU decoder-step graph directly on
// libggml, and runs the greedy decode loop in this driver (no in-graph loop), threading
// the SSRU cell state between steps exactly as numpy_ref / inference-rs do.
//
// The whole point is a fair apples-to-apples number against inference-rs and ONNX: a
// compiled binary whose RSS is just ggml arenas + weights (not a Python+runtime process),
// single-threaded by default to match the native baselines.
//
// Modes (argv: <model.gguf> <mode> ...):
//   decode                : stdin lines of space-separated source ids -> stdout output ids
//   blockbench --blocks F  : emit one `[block] {json}` span per block (perf harness peer)
//   dump ID ID ...         : write encoder context + first-step logits to testdata/ (gates)
//
// Tokenization lives in Python (the shared SPM), so this binary consumes/produces ids —
// keeping tokenization identical across every engine and deferring llama.cpp SPM parity
// to G2.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct HParams {
    int dim, heads, head_dim, enc_depth, dec_depth, ffn_dim, vocab, max_seq, eos_id;
    float eps, embed_scale, attn_scale;
};

struct Model {
    HParams hp{};
    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;                 // holds weight tensors
    std::map<std::string, ggml_tensor *> w;        // name -> weight
    std::vector<float> positional_encoding;        // [max_seq*dim] host table
    int n_threads = 1;

    ggml_tensor * get(const std::string & n) const {
        auto it = w.find(n);
        if (it == w.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); }
        return it->second;
    }
};

uint32_t kv_u32(gguf_context * g, const char * k) {
    int64_t i = gguf_find_key(g, k);
    if (i < 0) { fprintf(stderr, "missing key %s\n", k); exit(1); }
    return gguf_get_val_u32(g, i);
}
float kv_f32(gguf_context * g, const char * k) {
    int64_t i = gguf_find_key(g, k);
    if (i < 0) { fprintf(stderr, "missing key %s\n", k); exit(1); }
    return gguf_get_val_f32(g, i);
}

void load_model(Model & m, const char * path) {
    ggml_context * meta = nullptr;
    gguf_init_params gp{ /*.no_alloc=*/true, /*.ctx=*/&meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "failed to open %s\n", path); exit(1); }

    m.hp.dim         = kv_u32(g, "marian.dim");
    m.hp.heads       = kv_u32(g, "marian.heads");
    m.hp.head_dim    = kv_u32(g, "marian.head_dim");
    m.hp.enc_depth   = kv_u32(g, "marian.enc_depth");
    m.hp.dec_depth   = kv_u32(g, "marian.dec_depth");
    m.hp.ffn_dim     = kv_u32(g, "marian.ffn_dim");
    m.hp.vocab       = kv_u32(g, "marian.vocab");
    m.hp.max_seq     = kv_u32(g, "marian.max_seq");
    m.hp.eos_id      = kv_u32(g, "marian.eos_id");
    m.hp.eps         = kv_f32(g, "marian.eps");
    m.hp.embed_scale = kv_f32(g, "marian.embed_scale");
    m.hp.attn_scale  = kv_f32(g, "marian.attn_scale");

    m.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m.backend, m.n_threads);

    m.wctx = meta;  // reuse the gguf-created metadata context to hold the weights
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(m.wctx, m.backend);
    if (!buf) { fprintf(stderr, "failed to alloc weight buffer\n"); exit(1); }

    // Read each tensor's bytes from the file into its (now-allocated) backend storage.
    std::ifstream f(path, std::ios::binary);
    const size_t base = gguf_get_data_offset(g);
    std::vector<char> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(m.wctx, name);
        const size_t off = base + gguf_get_tensor_offset(g, i);
        const size_t sz = ggml_nbytes(t);
        tmp.resize(sz);
        f.seekg(off);
        f.read(tmp.data(), sz);
        ggml_backend_tensor_set(t, tmp.data(), 0, sz);
        m.w[name] = t;
    }
    f.close();

    // Cache the positional-encoding table on the host so the decode step can feed the row for its position.
    ggml_tensor * positional_encoding = m.get("pos_enc");
    m.positional_encoding.resize((size_t) m.hp.max_seq * m.hp.dim);
    ggml_backend_tensor_get(positional_encoding, m.positional_encoding.data(), 0,
                            ggml_nbytes(positional_encoding));

    gguf_free(g);
}

// --- graph helpers -----------------------------------------------------------

ggml_tensor * linear(ggml_context * c, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, W, x);          // [out, T]
    if (b) y = ggml_add(c, y, b);                     // bias broadcast over T
    return y;
}

ggml_tensor * layernorm(ggml_context * c, ggml_tensor * x, ggml_tensor * scale,
                        ggml_tensor * bias, float eps) {
    ggml_tensor * n = ggml_norm(c, x, eps);           // biased var, eps inside sqrt
    n = ggml_mul(c, n, scale);
    n = ggml_add(c, n, bias);
    return n;
}

// Scaled dot-product multi-head attention. q:[dim,Tq], k/v:[dim,Tk] -> [dim,Tq].
ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        int n_head, int head_dim, float scale) {
    const int64_t Tq = q->ne[1], Tk = k->ne[1];
    ggml_tensor * Q = ggml_permute(c, ggml_reshape_3d(c, q, head_dim, n_head, Tq), 0, 2, 1, 3); // [hd,Tq,nh]
    ggml_tensor * K = ggml_permute(c, ggml_reshape_3d(c, k, head_dim, n_head, Tk), 0, 2, 1, 3); // [hd,Tk,nh]
    ggml_tensor * KQ = ggml_mul_mat(c, K, Q);                                                   // [Tk,Tq,nh]
    KQ = ggml_soft_max_ext(c, KQ, nullptr, scale, 0.0f);                                        // softmax over Tk
    ggml_tensor * V = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, head_dim, n_head, Tk), 1, 2, 0, 3)); // [Tk,hd,nh]
    ggml_tensor * KQV = ggml_mul_mat(c, V, KQ);                                                 // [hd,Tq,nh]
    KQV = ggml_cont(c, ggml_permute(c, KQV, 0, 2, 1, 3));                                        // [hd,nh,Tq]
    return ggml_reshape_2d(c, KQV, head_dim * n_head, Tq);                                       // [dim,Tq]
}

ggml_context * graph_ctx() {
    size_t sz = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
    ggml_init_params p{ sz, nullptr, /*no_alloc=*/true };
    return ggml_init(p);
}

// --- encoder graph: src ids -> context + per-layer cross K/V -----------------

struct EncGraph {
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_tensor * ids;      // input I32 [seq]
    ggml_tensor * positional_encoding; // input F32 [dim, seq]
    ggml_tensor * context;  // output [dim, seq]
    std::vector<ggml_tensor *> cross_k, cross_v; // per layer [dim, seq]
};

EncGraph build_encoder(Model & m, int seq) {
    EncGraph e;
    e.ctx = graph_ctx();
    ggml_context * c = e.ctx;
    e.gf = ggml_new_graph(c);
    const HParams & hp = m.hp;

    e.ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, seq); ggml_set_input(e.ids);
    e.positional_encoding = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, seq);
    ggml_set_input(e.positional_encoding);

    ggml_tensor * x = ggml_get_rows(c, m.get("token_embd.weight"), e.ids); // [dim,seq]
    x = ggml_scale(c, x, hp.embed_scale);
    x = ggml_add(c, x, e.positional_encoding);

    for (int l = 0; l < hp.enc_depth; ++l) {
        std::string p = "enc." + std::to_string(l) + ".";
        auto W = [&](const std::string & n){ return m.get(p + n); };
        ggml_tensor * q = linear(c, W("self.wq"), W("self.bq"), x);
        ggml_tensor * k = linear(c, W("self.wk"), W("self.bk"), x);
        ggml_tensor * v = linear(c, W("self.wv"), W("self.bv"), x);
        ggml_tensor * a = attention(c, q, k, v, hp.heads, hp.head_dim, hp.attn_scale);
        a = linear(c, W("self.wo"), W("self.bo"), a);
        x = layernorm(c, ggml_add(c, a, x), W("self.ln.scale"), W("self.ln.bias"), hp.eps);

        ggml_tensor * h = ggml_relu(c, linear(c, W("ffn.w1"), W("ffn.b1"), x));
        ggml_tensor * fo = linear(c, W("ffn.w2"), W("ffn.b2"), h);
        x = layernorm(c, ggml_add(c, fo, x), W("ffn.ln.scale"), W("ffn.ln.bias"), hp.eps);
    }
    e.context = x; ggml_set_output(e.context);

    // Fold cross-attention K/V once per decoder layer from the encoder context.
    for (int l = 0; l < hp.dec_depth; ++l) {
        std::string p = "dec." + std::to_string(l) + ".";
        auto W = [&](const std::string & n){ return m.get(p + n); };
        ggml_tensor * ck = linear(c, W("cross.wk"), W("cross.bk"), x);
        ggml_tensor * cv = linear(c, W("cross.wv"), W("cross.bv"), x);
        ggml_set_output(ck); ggml_set_output(cv);
        e.cross_k.push_back(ck); e.cross_v.push_back(cv);
    }
    ggml_build_forward_expand(e.gf, e.context);
    for (auto t : e.cross_k) ggml_build_forward_expand(e.gf, t);
    for (auto t : e.cross_v) ggml_build_forward_expand(e.gf, t);
    return e;
}

// --- batched decoder step graph ----------------------------------------------
// B sentences decode in lockstep. At m=1 a single-sentence step is bandwidth-bound
// streaming the 512x32000 Q8_0 projection to make ONE token; batching B rows reuses that
// same weight stream for B tokens (near-linear until compute-bound). Rows are independent
// (per-row SSRU state, per-row masked cross-attention), so a batched decode is
// token-identical to decoding each sentence alone — gated by batch-invariance.

struct DecGraph {
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_tensor * tok;                           // input I32 [B]
    ggml_tensor * positional_encoding;           // input F32 [dim,1] (shared lockstep pos)
    ggml_tensor * cross_bias;                    // input F32 [Smax,1,1,B] additive src mask
    std::vector<ggml_tensor *> state_in;         // input F32 [dim,B] per layer
    std::vector<ggml_tensor *> cross_k, cross_v; // input F32 [dim,Smax,B] per layer
    std::vector<ggml_tensor *> state_out;        // output F32 [dim,B] per layer
    ggml_tensor * logits;                        // output F32 [vocab,B]
    int B, Smax;
};

// Batched single-query cross-attention. q:[dim,B], ck/cv:[dim,Smax,B], bias:[Smax,1,1,B].
// Batches over (head, sentence) = (ne2, ne3); the additive bias masks each row's source
// padding. Returns [dim,B].
ggml_tensor * cross_attention_batched(ggml_context * c, ggml_tensor * q, ggml_tensor * ck,
                                     ggml_tensor * cv, ggml_tensor * bias, int n_head,
                                     int head_dim, int Smax, int B, float scale) {
    ggml_tensor * Q = ggml_cont(c, ggml_permute(c,
        ggml_reshape_4d(c, q, head_dim, n_head, 1, B), 0, 2, 1, 3));      // [hd,1,nh,B]
    ggml_tensor * K = ggml_cont(c, ggml_permute(c,
        ggml_reshape_4d(c, ck, head_dim, n_head, Smax, B), 0, 2, 1, 3));  // [hd,Smax,nh,B]
    ggml_tensor * KQ = ggml_mul_mat(c, K, Q);                             // [Smax,1,nh,B]
    KQ = ggml_scale(c, KQ, scale);
    KQ = ggml_add(c, KQ, bias);                                          // -inf on padding
    KQ = ggml_soft_max(c, KQ);                                           // over ne0=Smax
    ggml_tensor * V = ggml_cont(c, ggml_permute(c,
        ggml_reshape_4d(c, cv, head_dim, n_head, Smax, B), 1, 2, 0, 3));  // [Smax,hd,nh,B]
    ggml_tensor * KQV = ggml_mul_mat(c, V, KQ);                           // [hd,1,nh,B]
    KQV = ggml_cont(c, ggml_permute(c, KQV, 0, 2, 1, 3));                 // [hd,nh,1,B]
    return ggml_reshape_2d(c, KQV, head_dim * n_head, B);                 // [dim,B]
}

// `pos` is the absolute decoder position. The graph is rebuilt per position (see
// the greedy loop), so the position-dependent part of the embedding folds in here
// rather than needing a runtime gate input.
DecGraph build_decoder(Model & m, int B, int Smax, int pos) {
    DecGraph d; d.B = B; d.Smax = Smax;
    d.ctx = graph_ctx();
    ggml_context * c = d.ctx;
    d.gf = ggml_new_graph(c);
    const HParams & hp = m.hp;

    d.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, B); ggml_set_input(d.tok);
    d.positional_encoding = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, 1);
    ggml_set_input(d.positional_encoding);
    d.cross_bias = ggml_new_tensor_4d(c, GGML_TYPE_F32, Smax, 1, 1, B); ggml_set_input(d.cross_bias);

    ggml_tensor * u = ggml_get_rows(c, m.get("token_embd.weight"), d.tok); // [dim,B]
    // Decoder position 0 takes no embedding at all — only the positional encoding.
    // marian builds the decoder input by shifting the target embeddings right and
    // zero-padding the vacated first slot (`shift(embeddings, {0,1,0})` in
    // `DecoderTransformer::step`), so the first step has no previous token to embed.
    // Scaling by 0 here is exactly that zero-pad. See notes/23-float-model-support.md.
    u = ggml_scale(c, u, pos == 0 ? 0.0f : hp.embed_scale);
    u = ggml_add(c, u, d.positional_encoding);   // broadcast over B

    for (int l = 0; l < hp.dec_depth; ++l) {
        std::string p = "dec." + std::to_string(l) + ".";
        auto W = [&](const std::string & n){ return m.get(p + n); };

        ggml_tensor * s_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, B); ggml_set_input(s_in);
        ggml_tensor * ck = ggml_new_tensor_3d(c, GGML_TYPE_F32, hp.dim, Smax, B); ggml_set_input(ck);
        ggml_tensor * cv = ggml_new_tensor_3d(c, GGML_TYPE_F32, hp.dim, Smax, B); ggml_set_input(cv);
        d.state_in.push_back(s_in); d.cross_k.push_back(ck); d.cross_v.push_back(cv);

        // SSRU highway: c_t = g*c_prev + (1-g)*cand = cand + g*(c_prev - cand). Per-column
        // (per-row) over ne1=B; LayerNorm normalizes ne0=dim independently per row.
        ggml_tensor * cand = ggml_mul_mat(c, W("rnn.w"), u);                 // no bias
        ggml_tensor * g = ggml_sigmoid(c, linear(c, W("rnn.wf"), W("rnn.bf"), u));
        ggml_tensor * c_t = ggml_add(c, cand, ggml_mul(c, g, ggml_sub(c, s_in, cand)));
        ggml_set_output(c_t); d.state_out.push_back(c_t);
        ggml_tensor * hcell = ggml_relu(c, c_t);
        ggml_tensor * x_self = layernorm(c, ggml_add(c, hcell, u), W("rnn.ln.scale"), W("rnn.ln.bias"), hp.eps);

        ggml_tensor * q = linear(c, W("cross.wq"), W("cross.bq"), x_self);   // [dim,B]
        ggml_tensor * a = cross_attention_batched(c, q, ck, cv, d.cross_bias,
                                                  hp.heads, hp.head_dim, Smax, B, hp.attn_scale);
        a = linear(c, W("cross.wo"), W("cross.bo"), a);
        ggml_tensor * x_ctx = layernorm(c, ggml_add(c, a, x_self), W("cross.ln.scale"), W("cross.ln.bias"), hp.eps);

        ggml_tensor * h = ggml_relu(c, linear(c, W("ffn.w1"), W("ffn.b1"), x_ctx));
        ggml_tensor * fo = linear(c, W("ffn.w2"), W("ffn.b2"), h);
        u = layernorm(c, ggml_add(c, fo, x_ctx), W("ffn.ln.scale"), W("ffn.ln.bias"), hp.eps);
    }

    d.logits = linear(c, m.get("output.weight"), m.get("output.bias"), u); // tied projection [vocab,B]
    ggml_set_output(d.logits);
    ggml_build_forward_expand(d.gf, d.logits);
    for (auto t : d.state_out) ggml_build_forward_expand(d.gf, t);
    return d;
}

// --- engine ------------------------------------------------------------------

struct Timing { double encode_ms = 0, decode_ms = 0; };

// Greedy-decode a batch of sentences in lockstep (each src includes the trailing EOS).
// Returns per-sentence output ids. The encoder still runs once per sentence (its cross K/V
// are padded to the batch's longest source and masked); only the decoder is batched — a
// block-batched decode, the production shape (matching inference-rs / ONNX).
std::vector<std::vector<int>> translate_batch(Model & m, ggml_gallocr_t alloc,
                                              const std::vector<std::vector<int32_t>> & batch,
                                              Timing * tm) {
    const HParams & hp = m.hp;
    const int B = (int) batch.size();
    std::vector<int> seq(B);
    for (int b = 0; b < B; ++b) seq[b] = (int) batch[b].size();

    // Encoder per sentence -> per-row (unpadded) cross K/V; the active batch is packed each
    // decode step so padding shrinks with the live rows.
    double t0 = now_ms();
    std::vector<std::vector<std::vector<float>>> ckr(hp.dec_depth), cvr(hp.dec_depth);
    for (int l = 0; l < hp.dec_depth; ++l) { ckr[l].resize(B); cvr[l].resize(B); }
    for (int b = 0; b < B; ++b) {
        EncGraph e = build_encoder(m, seq[b]);
        ggml_gallocr_alloc_graph(alloc, e.gf);
        ggml_backend_tensor_set(e.ids, batch[b].data(), 0, seq[b] * sizeof(int32_t));
        ggml_backend_tensor_set(e.positional_encoding, m.positional_encoding.data(), 0,
                                (size_t) seq[b] * hp.dim * sizeof(float));
        ggml_backend_graph_compute(m.backend, e.gf);
        for (int l = 0; l < hp.dec_depth; ++l) {
            ckr[l][b].resize((size_t) seq[b] * hp.dim); cvr[l][b].resize((size_t) seq[b] * hp.dim);
            ggml_backend_tensor_get(e.cross_k[l], ckr[l][b].data(), 0, ckr[l][b].size() * sizeof(float));
            ggml_backend_tensor_get(e.cross_v[l], cvr[l][b].data(), 0, cvr[l][b].size() * sizeof(float));
        }
        ggml_free(e.ctx);
    }
    if (tm) tm->encode_ms += now_ms() - t0;

    // Lockstep greedy over the ACTIVE rows only. As rows hit eos/length they retire and the
    // batch compacts, so total row-steps == the sum of per-sentence lengths (no wasted
    // compute on finished rows — the difference between this being a win and a pessimization
    // on ragged blocks), while live rows still share amortized weight streams.
    t0 = now_ms();
    std::vector<std::vector<float>> state(hp.dec_depth,        // per-row [dim], indexed by row r
                                          std::vector<float>((size_t) hp.dim * B, 0.0f));
    std::vector<int> max_len(B);
    for (int b = 0; b < B; ++b) max_len[b] = std::min((int) std::ceil(2.0 * seq[b]) + 4, 256);
    std::vector<std::vector<int>> out(B);
    std::vector<int32_t> prev(B, hp.eos_id);
    std::vector<bool> fin(B, false);

    for (int pos = 0; ; ++pos) {
        for (int b = 0; b < B; ++b) if (!fin[b] && pos >= max_len[b]) fin[b] = true;
        std::vector<int> act;
        for (int b = 0; b < B; ++b) if (!fin[b]) act.push_back(b);
        if (act.empty()) break;
        const int Ba = (int) act.size();
        int Smax = 0;
        for (int r : act) Smax = std::max(Smax, seq[r]);

        DecGraph d = build_decoder(m, Ba, Smax, pos);
        ggml_gallocr_alloc_graph(alloc, d.gf);

        std::vector<int32_t> toks(Ba);
        std::vector<float> bias((size_t) Smax * Ba, 0.0f);
        for (int i = 0; i < Ba; ++i) {
            int r = act[i]; toks[i] = prev[r];
            for (int s = seq[r]; s < Smax; ++s) bias[(size_t) i * Smax + s] = -1e30f;
        }
        ggml_backend_tensor_set(d.tok, toks.data(), 0, Ba * sizeof(int32_t));
        ggml_backend_tensor_set(d.positional_encoding, &m.positional_encoding[(size_t) pos * hp.dim],
                                0, hp.dim * sizeof(float));
        ggml_backend_tensor_set(d.cross_bias, bias.data(), 0, bias.size() * sizeof(float));

        std::vector<float> sa((size_t) hp.dim * Ba);
        std::vector<float> ka((size_t) hp.dim * Smax * Ba), va((size_t) hp.dim * Smax * Ba);
        for (int l = 0; l < hp.dec_depth; ++l) {
            std::fill(ka.begin(), ka.end(), 0.0f); std::fill(va.begin(), va.end(), 0.0f);
            for (int i = 0; i < Ba; ++i) {
                int r = act[i];
                std::copy_n(&state[l][(size_t) r * hp.dim], hp.dim, &sa[(size_t) i * hp.dim]);
                std::copy(ckr[l][r].begin(), ckr[l][r].end(), ka.begin() + (size_t) i * Smax * hp.dim);
                std::copy(cvr[l][r].begin(), cvr[l][r].end(), va.begin() + (size_t) i * Smax * hp.dim);
            }
            ggml_backend_tensor_set(d.state_in[l], sa.data(), 0, sa.size() * sizeof(float));
            ggml_backend_tensor_set(d.cross_k[l], ka.data(), 0, ka.size() * sizeof(float));
            ggml_backend_tensor_set(d.cross_v[l], va.data(), 0, va.size() * sizeof(float));
        }

        ggml_backend_graph_compute(m.backend, d.gf);

        std::vector<float> so((size_t) hp.dim * Ba);
        for (int l = 0; l < hp.dec_depth; ++l) {
            ggml_backend_tensor_get(d.state_out[l], so.data(), 0, so.size() * sizeof(float));
            for (int i = 0; i < Ba; ++i)
                std::copy_n(&so[(size_t) i * hp.dim], hp.dim, &state[l][(size_t) act[i] * hp.dim]);
        }
        std::vector<float> logits((size_t) hp.vocab * Ba);
        ggml_backend_tensor_get(d.logits, logits.data(), 0, logits.size() * sizeof(float));
        ggml_free(d.ctx);

        for (int i = 0; i < Ba; ++i) {
            int r = act[i];
            float * lb = &logits[(size_t) i * hp.vocab];              // logits [vocab,Ba], row i
            int tok = (int) (std::max_element(lb, lb + hp.vocab) - lb);
            if (tok == hp.eos_id) { fin[r] = true; continue; }
            out[r].push_back(tok); prev[r] = tok;
        }
    }
    if (tm) tm->decode_ms += now_ms() - t0;
    return out;
}

// One-sentence convenience (B=1) — the batch-invariance reference.
std::vector<int> translate_one(Model & m, ggml_gallocr_t alloc, const std::vector<int32_t> & src) {
    return translate_batch(m, alloc, {src}, nullptr)[0];
}

std::vector<int32_t> parse_ids(const std::string & line) {
    std::vector<int32_t> ids;
    std::istringstream ss(line);
    int x;
    while (ss >> x) ids.push_back(x);
    return ids;
}

// --- modes -------------------------------------------------------------------

int mode_decode(Model & m, ggml_gallocr_t alloc, bool solo) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(std::cin, line)) lines.push_back(line);
    std::vector<std::vector<int32_t>> batch;
    std::vector<int> row_of;  // batch row -> line index
    for (size_t i = 0; i < lines.size(); ++i)
        if (lines[i].find_first_not_of(" \t\r\n") != std::string::npos) {
            batch.push_back(parse_ids(lines[i])); row_of.push_back((int) i);
        }
    std::vector<std::string> outstr(lines.size());
    // `solo` decodes each line as its own B=1 batch — the batch-invariance reference.
    std::vector<std::vector<int>> outs;
    if (solo) {
        for (auto & s : batch) outs.push_back(translate_one(m, alloc, s));
    } else if (!batch.empty()) {
        outs = translate_batch(m, alloc, batch, nullptr);
    }
    for (size_t r = 0; r < outs.size(); ++r) {
        std::string s;
        for (size_t j = 0; j < outs[r].size(); ++j) { if (j) s += " "; s += std::to_string(outs[r][j]); }
        outstr[row_of[r]] = s;
    }
    for (auto & s : outstr) std::cout << s << "\n";
    return 0;
}

int mode_blockbench(Model & m, ggml_gallocr_t alloc, const char * blocks_path) {
    std::ifstream f(blocks_path);
    if (!f) { fprintf(stderr, "cannot open %s\n", blocks_path); return 1; }
    std::vector<std::vector<int32_t>> block;
    int idx = 0;
    auto flush = [&]() {
        if (block.empty()) return;
        Timing tm;
        auto outs = translate_batch(m, alloc, block, &tm);  // whole block as one lockstep batch
        int src_tokens = 0, out_tokens = 0;
        for (auto & s : block) src_tokens += (int) s.size();
        for (auto & o : outs) out_tokens += (int) o.size();
        fprintf(stderr,
                "[block] {\"block\": %d, \"sentences\": %zu, \"src_tokens\": %d, "
                "\"tokens\": %d, \"encode_ms\": %.3f, \"decode_ms\": %.3f}\n",
                idx, block.size(), src_tokens, out_tokens, tm.encode_ms, tm.decode_ms);
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
    return 0;
}

int mode_dump(Model & m, ggml_gallocr_t alloc, const std::vector<int32_t> & src) {
    const HParams & hp = m.hp;
    const int seq = (int) src.size();

    EncGraph e = build_encoder(m, seq);
    ggml_gallocr_alloc_graph(alloc, e.gf);
    ggml_backend_tensor_set(e.ids, src.data(), 0, seq * sizeof(int32_t));
    ggml_backend_tensor_set(e.positional_encoding, m.positional_encoding.data(), 0, (size_t) seq * hp.dim * sizeof(float));
    ggml_backend_graph_compute(m.backend, e.gf);
    std::vector<float> ctx((size_t) seq * hp.dim);
    ggml_backend_tensor_get(e.context, ctx.data(), 0, ctx.size() * sizeof(float));
    std::vector<std::vector<float>> ck(hp.dec_depth), cv(hp.dec_depth);
    for (int l = 0; l < hp.dec_depth; ++l) {
        ck[l].resize((size_t) seq * hp.dim); cv[l].resize((size_t) seq * hp.dim);
        ggml_backend_tensor_get(e.cross_k[l], ck[l].data(), 0, ck[l].size() * sizeof(float));
        ggml_backend_tensor_get(e.cross_v[l], cv[l].data(), 0, cv[l].size() * sizeof(float));
    }
    ggml_free(e.ctx);

    // First decode step (B=1, pos 0, prev = eos, state zero, no source padding).
    // At pos 0 the seed token's embedding is zero-padded away (see build_decoder),
    // so `prev` below is inert — it is set for symmetry with the greedy loop.
    DecGraph d = build_decoder(m, 1, seq, 0);
    ggml_gallocr_alloc_graph(alloc, d.gf);
    int32_t prev = hp.eos_id;
    std::vector<float> zero((size_t) hp.dim, 0.0f);
    std::vector<float> nobias((size_t) seq, 0.0f);
    ggml_backend_tensor_set(d.tok, &prev, 0, sizeof(int32_t));
    ggml_backend_tensor_set(d.positional_encoding, &m.positional_encoding[0], 0, hp.dim * sizeof(float));
    ggml_backend_tensor_set(d.cross_bias, nobias.data(), 0, nobias.size() * sizeof(float));
    for (int l = 0; l < hp.dec_depth; ++l) {
        ggml_backend_tensor_set(d.state_in[l], zero.data(), 0, hp.dim * sizeof(float));
        ggml_backend_tensor_set(d.cross_k[l], ck[l].data(), 0, ck[l].size() * sizeof(float));
        ggml_backend_tensor_set(d.cross_v[l], cv[l].data(), 0, cv[l].size() * sizeof(float));
    }
    ggml_backend_graph_compute(m.backend, d.gf);
    std::vector<float> logits(hp.vocab);
    ggml_backend_tensor_get(d.logits, logits.data(), 0, hp.vocab * sizeof(float));
    ggml_free(d.ctx);

    system("mkdir -p ggml/testdata");
    { std::ofstream o("ggml/testdata/ggml_encoder.bin", std::ios::binary);
      o.write((char *) ctx.data(), ctx.size() * sizeof(float)); }
    { std::ofstream o("ggml/testdata/ggml_logits.bin", std::ios::binary);
      o.write((char *) logits.data(), logits.size() * sizeof(float)); }
    int argmax = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    fprintf(stderr, "[dump] seq=%d context[%d,%d] logits[%d] argmax=%d\n",
            seq, seq, hp.dim, hp.vocab, argmax);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <decode|blockbench|dump> [args]\n", argv[0]);
        return 1;
    }
    Model m;
    if (const char * t = getenv("FXT_GGML_THREADS")) m.n_threads = atoi(t);
    load_model(m, argv[1]);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));

    std::string mode = argv[2];
    int rc = 1;
    if (mode == "decode") {
        bool solo = false;  // `decode solo` = per-sentence B=1 (batch-invariance reference)
        for (int i = 3; i < argc; ++i) if (std::string(argv[i]) == "solo") solo = true;
        rc = mode_decode(m, alloc, solo);
    } else if (mode == "blockbench") {
        const char * blocks = nullptr;
        for (int i = 3; i < argc; ++i)
            if (std::string(argv[i]) == "--blocks" && i + 1 < argc) blocks = argv[++i];
        if (!blocks) { fprintf(stderr, "blockbench needs --blocks FILE\n"); return 1; }
        rc = mode_blockbench(m, alloc, blocks);
    } else if (mode == "dump") {
        std::vector<int32_t> src;
        for (int i = 3; i < argc; ++i) src.push_back(atoi(argv[i]));
        rc = mode_dump(m, alloc, src);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
    }

    ggml_gallocr_free(alloc);
    ggml_free(m.wctx);
    ggml_backend_free(m.backend);
    return rc;
}
