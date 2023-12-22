//
// Created by 咸的鱼 on 2023/12/21.
//

#ifndef MODELING_LLAMA_HPP
#define MODELING_LLAMA_HPP
#include "helper.hpp"

inline NetTensor *Attention_LLAMA(Context *ctx, NetTensor *x, int embedding_size, int hidden_size, int head_size, string name) {
    auto *q = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wq");
    auto *k = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wk");
    auto *v = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wv");
    q = _View(ctx, {q}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".q_view");
    k = _View(ctx, {k}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".k_view");
    v = _View(ctx, {v}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".v_view");
    q = _RoPE(ctx, {q}, 2, name + ".q_rope");
    k = _RoPE(ctx, {k}, 2, name + ".k_rope");
    k = _KVCache(ctx, {k}, true, name + ".k_cache");
    v = _KVCache(ctx, {v}, true, name + ".v_cache");
    auto *qk = _Matmul(ctx, {q, k}, false, true, name + ".qk");
    qk = _Scale(ctx, {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    qk = _Causalmask(ctx, {qk}, name + ".mask");
    qk = _Softmax(ctx, {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, false, name + ".qkv");
    o = _View(ctx, {o}, {-1, -1, -1, -1}, {BATCH, -1, SEQUENCE, HEAD + DIMENSION}, name + ".qkv_view");
    o = _Linear(ctx, {o}, hidden_size * head_size, embedding_size, false, name + ".wo");
    return o;
}

inline NetTensor *FFN_LLAMA(Context *ctx, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name + ".w1");
    x = _SiLU(ctx, {x});
    auto *y = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name + ".w3");
    x = _Mul(ctx, {x, y});
    x = _Linear(ctx, {x}, ffn_hidden_dim, hidden_dim, false, name + ".w2");
    return x;
}

inline void llama2(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32) {
    auto *i = _Input(c);
    i = _Embedding(c, {i}, vocab_size, hidden_dim, (string)"tok_embeddings");
    // loop
    for (int layer = 0; layer < 32; ++layer) {
        auto *x = _RMSNorm(c, {i}, (string)"layers." + std::to_string(layer) + ".attention_norm");
        //x = _Attention(c, {x}, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".attention");
        x = Attention_LLAMA(c, x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers." + std::to_string(layer) + ".attention");
        i = _Add(c, {x, i});
        x = _RMSNorm(c, {i}, (string)"layers." + std::to_string(layer) + ".ffn_norm");
        x = FFN_LLAMA(c, x, hidden_dim, ffn_hidden_dim, (string)"layers." + std::to_string(layer) + ".feed_forward");
        i = _Add(c, {x, i});
        _SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm(c, {i}, (string)"norm");
    i = _Linear(c, {i}, hidden_dim, vocab_size, false, "output");
}


inline unsigned int postProcessing_llama(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) {
    CHECK_EQ(result->batch(), 1);
    CHECK_EQ(result->head(), 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

#endif //MODELING_LLAMA_HPP
