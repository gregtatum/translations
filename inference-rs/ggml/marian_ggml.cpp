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
    std::vector<float> pe;                          // [max_seq*dim] host PE table
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

    // Cache the PE table on the host so the decode step can feed the row for its position.
    ggml_tensor * pe = m.get("pos_enc");
    m.pe.resize((size_t) m.hp.max_seq * m.hp.dim);
    ggml_backend_tensor_get(pe, m.pe.data(), 0, ggml_nbytes(pe));

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
    ggml_tensor * pe;       // input F32 [dim, seq]
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
    e.pe  = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, seq); ggml_set_input(e.pe);

    ggml_tensor * x = ggml_get_rows(c, m.get("token_embd.weight"), e.ids); // [dim,seq]
    x = ggml_scale(c, x, hp.embed_scale);
    x = ggml_add(c, x, e.pe);

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

// --- decoder step graph: reused across all steps of one sentence -------------

struct DecGraph {
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_tensor * tok;                        // input I32 [1]
    ggml_tensor * pe;                         // input F32 [dim,1]
    std::vector<ggml_tensor *> state_in;      // input F32 [dim] per layer
    std::vector<ggml_tensor *> cross_k, cross_v; // input F32 [dim,seq] per layer
    std::vector<ggml_tensor *> state_out;     // output F32 [dim] per layer
    ggml_tensor * logits;                     // output F32 [vocab]
};

DecGraph build_decoder(Model & m, int seq) {
    DecGraph d;
    d.ctx = graph_ctx();
    ggml_context * c = d.ctx;
    d.gf = ggml_new_graph(c);
    const HParams & hp = m.hp;

    d.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(d.tok);
    d.pe  = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, 1); ggml_set_input(d.pe);

    ggml_tensor * u = ggml_get_rows(c, m.get("token_embd.weight"), d.tok); // [dim,1]
    u = ggml_scale(c, u, hp.embed_scale);
    u = ggml_add(c, u, d.pe);

    for (int l = 0; l < hp.dec_depth; ++l) {
        std::string p = "dec." + std::to_string(l) + ".";
        auto W = [&](const std::string & n){ return m.get(p + n); };

        ggml_tensor * s_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, hp.dim); ggml_set_input(s_in);
        ggml_tensor * ck = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, seq); ggml_set_input(ck);
        ggml_tensor * cv = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.dim, seq); ggml_set_input(cv);
        d.state_in.push_back(s_in); d.cross_k.push_back(ck); d.cross_v.push_back(cv);

        // SSRU highway: c_t = g*c_prev + (1-g)*cand = cand + g*(c_prev - cand).
        ggml_tensor * cand = ggml_mul_mat(c, W("rnn.w"), u);                 // no bias
        ggml_tensor * g = ggml_sigmoid(c, linear(c, W("rnn.wf"), W("rnn.bf"), u));
        ggml_tensor * c_t = ggml_add(c, cand, ggml_mul(c, g, ggml_sub(c, s_in, cand)));
        ggml_set_output(c_t); d.state_out.push_back(c_t);
        ggml_tensor * hcell = ggml_relu(c, c_t);
        ggml_tensor * x_self = layernorm(c, ggml_add(c, hcell, u), W("rnn.ln.scale"), W("rnn.ln.bias"), hp.eps);

        // cross-attention: Q from decoder, K/V precomputed from encoder context.
        ggml_tensor * q = linear(c, W("cross.wq"), W("cross.bq"), x_self);   // [dim,1]
        ggml_tensor * a = attention(c, q, ck, cv, hp.heads, hp.head_dim, hp.attn_scale);
        a = linear(c, W("cross.wo"), W("cross.bo"), a);
        ggml_tensor * x_ctx = layernorm(c, ggml_add(c, a, x_self), W("cross.ln.scale"), W("cross.ln.bias"), hp.eps);

        ggml_tensor * h = ggml_relu(c, linear(c, W("ffn.w1"), W("ffn.b1"), x_ctx));
        ggml_tensor * fo = linear(c, W("ffn.w2"), W("ffn.b2"), h);
        u = layernorm(c, ggml_add(c, fo, x_ctx), W("ffn.ln.scale"), W("ffn.ln.bias"), hp.eps);
    }

    d.logits = linear(c, m.get("output.weight"), m.get("output.bias"), u); // tied projection [vocab,1]
    ggml_set_output(d.logits);
    ggml_build_forward_expand(d.gf, d.logits);
    for (auto t : d.state_out) ggml_build_forward_expand(d.gf, t);
    return d;
}

// --- engine ------------------------------------------------------------------

struct Timing { double encode_ms = 0, decode_ms = 0; };

// Greedy-decode one sentence (source ids include the trailing EOS). Returns output ids.
std::vector<int> translate_one(Model & m, ggml_gallocr_t alloc,
                              const std::vector<int32_t> & src, Timing * tm) {
    const HParams & hp = m.hp;
    const int seq = (int) src.size();

    // Encoder.
    double t0 = now_ms();
    EncGraph e = build_encoder(m, seq);
    ggml_gallocr_alloc_graph(alloc, e.gf);
    ggml_backend_tensor_set(e.ids, src.data(), 0, seq * sizeof(int32_t));
    ggml_backend_tensor_set(e.pe, m.pe.data(), 0, (size_t) seq * hp.dim * sizeof(float));
    ggml_backend_graph_compute(m.backend, e.gf);

    // Pull cross K/V to host so the reused decode graph can reference them per sentence.
    std::vector<std::vector<float>> ck(hp.dec_depth), cv(hp.dec_depth);
    for (int l = 0; l < hp.dec_depth; ++l) {
        ck[l].resize((size_t) seq * hp.dim); cv[l].resize((size_t) seq * hp.dim);
        ggml_backend_tensor_get(e.cross_k[l], ck[l].data(), 0, ck[l].size() * sizeof(float));
        ggml_backend_tensor_get(e.cross_v[l], cv[l].data(), 0, cv[l].size() * sizeof(float));
    }
    ggml_free(e.ctx);
    if (tm) tm->encode_ms += now_ms() - t0;

    // Decoder greedy loop. The step graph is rebuilt per token: a single-token decode is
    // tiny (dec_depth * ~30 nodes) next to the matmuls, and rebuilding sidesteps any
    // gallocr input/scratch aliasing across reused computes. (A reuse optimization is a
    // possible follow-up once correctness is locked.)
    t0 = now_ms();
    std::vector<std::vector<float>> state(hp.dec_depth,
                                          std::vector<float>((size_t) hp.dim, 0.0f));
    const int max_len = std::min((int) std::ceil(2.0 * seq) + 4, 256);
    std::vector<int> out;
    std::vector<float> logits(hp.vocab);
    int32_t prev = hp.eos_id;

    for (int pos = 0; pos < max_len; ++pos) {
        DecGraph d = build_decoder(m, seq);
        ggml_gallocr_alloc_graph(alloc, d.gf);
        ggml_backend_tensor_set(d.tok, &prev, 0, sizeof(int32_t));
        ggml_backend_tensor_set(d.pe, &m.pe[(size_t) pos * hp.dim], 0, hp.dim * sizeof(float));
        for (int l = 0; l < hp.dec_depth; ++l) {
            ggml_backend_tensor_set(d.state_in[l], state[l].data(), 0, hp.dim * sizeof(float));
            ggml_backend_tensor_set(d.cross_k[l], ck[l].data(), 0, ck[l].size() * sizeof(float));
            ggml_backend_tensor_set(d.cross_v[l], cv[l].data(), 0, cv[l].size() * sizeof(float));
        }

        ggml_backend_graph_compute(m.backend, d.gf);

        for (int l = 0; l < hp.dec_depth; ++l)
            ggml_backend_tensor_get(d.state_out[l], state[l].data(), 0, hp.dim * sizeof(float));
        ggml_backend_tensor_get(d.logits, logits.data(), 0, hp.vocab * sizeof(float));
        ggml_free(d.ctx);

        int tok = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (tok == hp.eos_id) break;
        out.push_back(tok);
        prev = tok;
    }
    if (tm) tm->decode_ms += now_ms() - t0;
    return out;
}

std::vector<int32_t> parse_ids(const std::string & line) {
    std::vector<int32_t> ids;
    std::istringstream ss(line);
    int x;
    while (ss >> x) ids.push_back(x);
    return ids;
}

// --- modes -------------------------------------------------------------------

int mode_decode(Model & m, ggml_gallocr_t alloc) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) { std::cout << "\n"; continue; }
        auto src = parse_ids(line);
        auto out = translate_one(m, alloc, src, nullptr);
        for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? " " : "") << out[i];
        std::cout << "\n";
    }
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
        int src_tokens = 0, out_tokens = 0;
        for (auto & s : block) {
            auto o = translate_one(m, alloc, s, &tm);
            src_tokens += (int) s.size();
            out_tokens += (int) o.size();
        }
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
    ggml_backend_tensor_set(e.pe, m.pe.data(), 0, (size_t) seq * hp.dim * sizeof(float));
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

    // First decode step (pos 0, prev = eos), states zero.
    DecGraph d = build_decoder(m, seq);
    ggml_gallocr_alloc_graph(alloc, d.gf);
    int32_t prev = hp.eos_id;
    std::vector<float> zero((size_t) hp.dim, 0.0f);
    ggml_backend_tensor_set(d.tok, &prev, 0, sizeof(int32_t));
    ggml_backend_tensor_set(d.pe, &m.pe[0], 0, hp.dim * sizeof(float));
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
        rc = mode_decode(m, alloc);
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
