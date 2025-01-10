#pragma once

#include <torch/extension.h>

torch::Tensor pa_fwd(torch::Tensor &Q,            //   [num_seqs, num_heads, head_size]
                     torch::Tensor &K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor &V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor &block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor &context_lens, //   [num_seqs]
                     std::optional<torch::Tensor> K_QScale = std::nullopt,
                     std::optional<torch::Tensor> V_QScale = std::nullopt);