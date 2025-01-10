
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include "hip_compat.h"
#include "dispatch_utils.h"
#include "fused_moe.hpp"

void moe_fused_experts_ck(torch::Tensor &hidden_states, torch::Tensor &w1, torch::Tensor &w2,
                          torch::Tensor &topk_weights, torch::Tensor &topk_ids,
                          at::optional<torch::Tensor> w1_scale, 
                          at::optional<torch::Tensor> w2_scale,
                          at::optional<torch::Tensor> a1_scale,
                          at::optional<torch::Tensor> a2_scale,
                          torch::Tensor &sorted_ids, torch::Tensor &sorted_weights,
                          torch::Tensor &sorted_expert_ids, torch::Tensor &num_tokens_post_pad,
                          torch::Tensor &out, int block_m, int fused_quant, int gate_only) {

    auto prec_i = torchDTypeToStr(hidden_states.dtype());
    auto prec_w = torchDTypeToStr(w1.dtype());
    auto prec_o = torchDTypeToStr(out.dtype());
    auto prec_kw = torchDTypeToStr(topk_weights.dtype());

    std::string prec_st = !a1_scale ? "fp32" : torchDTypeToStr(a1_scale->dtype());
    std::string prec_sw = !w1_scale ? "fp32" : torchDTypeToStr(w1_scale->dtype());
    std::string prec_sq = !a2_scale ? "fp32" : torchDTypeToStr(a2_scale->dtype());

    int hidden_size = w1.size(2);
    int shared_intermediate_size_0 = w1.size(1);

    int tokens = hidden_states.size(0);
    int experts = w1.size(0);

    int topk = topk_ids.size(1);

    int stride = hidden_size;

    fused_moe_traits traits{prec_i,
                                prec_w,
                                prec_o,
                                prec_st,
                                prec_sw,
                                prec_sq,
                                prec_kw,
                                block_m,
                                gate_only,
                                fused_quant};

    fused_moe_args args{hidden_states.data_ptr(),
	                    !a1_scale ? nullptr : a1_scale->data_ptr(),
                            w1.data_ptr(), 
                            w2.data_ptr(),
	                    !w1_scale ? nullptr : w1_scale->data_ptr(),
	                    !w2_scale ? nullptr : w2_scale->data_ptr(),
	                    !a2_scale ? nullptr : a2_scale->data_ptr(),
                            out.data_ptr(),
                            topk_ids.data_ptr(),
                            topk_weights.data_ptr(),
                            sorted_ids.data_ptr(),
                            sorted_weights.data_ptr(),
                            sorted_expert_ids.data_ptr(),
                            num_tokens_post_pad.data_ptr(),
                            block_m,
                            hidden_size,
                            shared_intermediate_size_0,
                            tokens,
                            experts,
                            topk,
                            stride};

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_moe(traits, args, {stream}); 

//    std::cout << "[moe_fused_experts_ck] prec_i:" << prec_i
//	      << " prec_w:" << prec_w
//	      << " prec_o:" << prec_o
//	      << " prec_st:" << prec_st
//	      << " prec_sw:" << prec_sw
//	      << " prec_sq:" << prec_sq
//	      << " prec_kw:" << prec_kw
//	      << " hidden_size:" << hidden_size
//	      << " shared_intermediate_size_0:" << shared_intermediate_size_0
//	      << " toekens:" << tokens
//	      << " experts:" << experts
//	      << " topk:" << topk
//	      << std::endl;
}

