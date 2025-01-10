#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"

#include "moe_smoothquant.hpp"


void moe_smoothquant_fwd(torch::Tensor &out,      // [topk * tokens, hidden_size]
                         torch::Tensor &input,    // [tokens, hidden_size]
                         torch::Tensor &x_scale,  // [experts, hidden_size]
                         torch::Tensor &topk_ids, // [tokens, topk]
                         torch::Tensor &y_scale)  // [topk * tokens,  1]
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck smoothquant only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int experts = x_scale.size(0);
    int topk = topk_ids.size(1);
    int stride = n;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_smoothquant({
                        dtype_str // input  dtype
                    },
                    {input.data_ptr(),    // [tokens, hidden_size], input, fp16/bf16
                     x_scale.data_ptr(),  // [experts, hidden_size], input, columnwise scale, fp32
                     topk_ids.data_ptr(), // [tokens, topk]

                     y_scale.data_ptr(), // [topk * tokens,  1], output, rowwise quant scale
                     out.data_ptr(),     // [topk * tokens, hidden_size], output
                     m, n, experts, topk, stride, stride},
                    {stream});
}
