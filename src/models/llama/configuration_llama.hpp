//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_LLAMA_HPP
#define CONFIG_LLAMA_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class LLaMANameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init(RoPEType type = LLAMAROPE) {
        switch (type) {
        case LLAMAROPE: {
            blk_name = "layers.";
            _attn_base_name = "attention.";
            _ffn_base_name = "feed_forward.";
            _q_proj_name = "wq";
            _k_proj_name = "wk";
            _v_proj_name = "wv";
            _o_proj_name = "wo";
            _gate_proj_name = "w1";
            _up_proj_name = "w3";
            _down_proj_name = "w2";
            _attn_norm_name = "attention_norm";
            _ffn_norm_name = "ffn_norm";
            token_embd_name = "tok_embeddings";
            post_norm_name = "norm";
            lm_head_name = "output";
            break;
        }
        case HFHUBROPE: {
            blk_name = "model.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "o_proj";
            _gate_proj_name = "gate_proj";
            _up_proj_name = "up_proj";
            _down_proj_name = "down_proj";
            _attn_norm_name = "input_layernorm";
            _ffn_norm_name = "post_attention_layernorm";
            token_embd_name = "model.embed_tokens";
            post_norm_name = "model.norm";
            lm_head_name = "lm_head";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported llama type");
        }
        }
    }
};

class LLaMAConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    LLaMANameConfig names_config;

    explicit LLaMAConfig(int token_limit, string billions = "7B", RoPEType type = LLAMAROPE, int vocab = 32000) {
        names_config.init(type);
        vocab_size = vocab;
        if (billions == "7B" || billions == "7b") {
            hidden_dim = 4096; /*For example, in this context, hidden_dim = 4096 means that each token in the input is represented as a 4096-dimensional vector, and all intermediate hidden layers also produce vectors of this size.*/
            head_size = 32;
            ffn_hidden = 11008;
            block_num = 32;
            /*
            hidden_dim (4096): Size of the vectors used throughout the model.
            head_size (32): Size of each attention head, determining the number of attention heads (128 in this case).
            ffn_hidden (11008): Size of the intermediate layer in the feed-forward network, which affects the model's capacity.
            block_num (32): Number of layers (transformer blocks) stacked in the model, affecting its depth and representational power.
            */
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }
};

#endif // CONFIG_LLAMA_HPP


/*
Further explaination:

1. hidden_dim
Definition: The hidden dimension size of the model layers.
Explanation: This is the size of the embeddings and the hidden states in the model. It determines the dimensionality of the vectors that flow through the network. For example, in this context, hidden_dim = 4096 means that each token in the input is represented as a 4096-dimensional vector, and all intermediate hidden layers also produce vectors of this size.
2. head_size
Definition: The size of each attention head.
Explanation: In a multi-head attention mechanism, the model splits the hidden dimension into multiple heads, each of which can focus on different parts of the input sequence. The head_size refers to the dimensionality of each of these individual heads. In this case, head_size = 32 indicates that each attention head has 32 dimensions. The number of heads is typically calculated as hidden_dim / head_size, so here it would be 4096 / 32 = 128 heads.
3. ffn_hidden
Definition: The size of the hidden layer in the feed-forward network (FFN) within each transformer block.
Explanation: Each transformer block typically includes a multi-head self-attention mechanism followed by a feed-forward neural network. The FFN has an intermediate layer whose size is larger than the hidden_dim. In this context, ffn_hidden = 11008 means that the feed-forward network's intermediate layer has 11008 dimensions. This layer increases the model's capacity to learn complex patterns by expanding and then reducing the dimensionality of the data.
4. block_num
Definition: The number of transformer blocks in the model.
Explanation: A transformer model is composed of several identical blocks stacked together. Each block includes layers for self-attention and feed-forward operations. The block_num specifies how many of these blocks are stacked in the model. Here, block_num = 32 indicates that there are 32 transformer blocks in the model. More blocks generally allow the model to learn more complex representations but also increase the computational cost.

*/


/*
Explaination of LLaMANameConfig's init() method members:
They are the component names to be used in upcoming steps.

blk_name

Use: This is the prefix used for naming the transformer blocks in the model.
Example: "layers.0", "layers.1", etc., where each block is part of the transformer.

token_embd_name

Use: Name for the token embedding layer that converts input tokens into their corresponding embeddings.
Example: "tok_embeddings" would be the layer that transforms input text tokens into 4096-dimensional vectors.

lm_head_name

Use: Name for the language modeling head, which generates the final output (e.g., predicted next token probabilities).
Example: "output" might be the layer that maps the final hidden states to vocabulary logits.


_gate_proj_name

Use: Name for the gate projection in the feed-forward network (part of the transformer block).
Example: "w1" represents one of the weight matrices used in the intermediate gating mechanism of the feed-forward layer.


#######Initialization Logic (init Method)

blk_name = "layers."

Use: Prefix for transformer blocks.
Example: Identifies each block with names like "layers.0", "layers.1", etc.

#####################Attention Components

_attn_base_name = "attention."

Use: Prefix for attention mechanism components.
Example: Names parts of the attention mechanism like "attention.q", "attention.k", etc.

_q_proj_name = "wq"

Use: Name for the query projection weight matrix.
Example: "wq" used for transforming input embeddings into query vectors.

_k_proj_name = "wk"

Use: Name for the key projection weight matrix.
Example: "wk" used for transforming input embeddings into key vectors.

_v_proj_name = "wv"

Use: Name for the value projection weight matrix.
Example: "wv" used for transforming input embeddings into value vectors.

_o_proj_name = "wo"

Use: Name for the output projection weight matrix.
Example: "wo" used for transforming the output of the attention mechanism back into the hidden dimension space.

####################Feed-Forward Network Components

_gate_proj_name = "w1"

Use: Name for the gate projection weight matrix.
Example: "w1" used in the intermediate step of the feed-forward network.

_up_proj_name = "w3"

Use: Name for the up projection weight matrix.
Example: "w3" used for expanding the dimensionality in the feed-forward network.

_down_proj_name = "w2"

Use: Name for the down projection weight matrix.
Example: "w2" used for reducing the dimensionality back to the hidden dimension.


####################Normalization Layers

_attn_norm_name = "attention_norm"

Use: Name for the normalization layer applied after the attention mechanism.
Example: "attention_norm" normalizes the output of the attention mechanism.

_ffn_norm_name = "ffn_norm"

Use: Name for the normalization layer applied after the feed-forward network.
Example: "ffn_norm" normalizes the output of the feed-forward network.



#####################Other components

token_embd_name = "tok_embeddings"

Use: Name for the token embedding layer.
Example: Converts tokens into embeddings with the name "tok_embeddings".

post_norm_name = "norm"

Use: Name for the normalization layer applied after all transformer blocks.
Example: "norm" used for final normalization.

lm_head_name = "output"

Use: Name for the language modeling head.
Example: Generates the final model output with the name "output".


*/
