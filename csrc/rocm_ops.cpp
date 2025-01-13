#include "activation.h"
#include "attention.h"
#include "attention_ck.h"
#include "attention_asm.h"
#include "cache.h"
#include "moe_op.h"
#include "moe_sorting.h"
#include "pos_encoding.h"
#include "smoothquant.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("topk_softmax", &topk_softmax,
            "Apply topk softmax to the gating outputs.");
      m.def("moe_align_block_size", &moe_align_block_size,
            "Aligning the number of tokens to be processed by each expert such "
            "that it is divisible by the block size.");
      m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
      m.def("moe_sum", &moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");
      m.def("paged_attention_rocm", &paged_attention,
            "paged_attention_rocm(Tensor! out, Tensor exp_sums,"
            "                Tensor max_logits, Tensor tmp_out,"
            "                Tensor query, Tensor key_cache,"
            "                Tensor value_cache, int num_kv_heads,"
            "                float scale, Tensor block_tables,"
            "                Tensor context_lens, int block_size,"
            "                int max_context_len,"
            "                Tensor? alibi_slopes,"
            "                str kv_cache_dtype,"
            "                float k_scale, float v_scale) -> ()");
      m.def("swap_blocks", &swap_blocks,
            "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
      m.def("copy_blocks", &copy_blocks,
            "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
            "Tensor block_mapping) -> ()");

      m.def("reshape_and_cache", &reshape_and_cache,
            "reshape_and_cache(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_flash", &reshape_and_cache_flash,
            "reshape_and_cache_flash(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype,"
            "                        float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor! k_dequant_scales, Tensor! v_dequant_scales,"
            "                  Tensor slot_mapping) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");

#if defined(FIND_CK)
      // ck staff start
      m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);
      m.def("moe_sorting_fwd", &moe_sorting_fwd);
      m.def("moe_fused_experts_ck", &moe_fused_experts_ck, "MOE implementation by ck");
      m.def("pa_fwd_naive", &pa_fwd_naive, "pa_fwd_naive",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("k_dequant_scales"),
            py::arg("v_dequant_scales"),
            py::arg("max_seq_len"),
            py::arg("num_kv_heads"),
            py::arg("scale_s"),
            py::arg("scale_k"),
            py::arg("scale_v"),
            py::arg("block_size"),
            py::arg("quant_algo"),
            py::arg("out_") = std::nullopt);
      // ck staff end
#endif

      m.def("fmoe", &fmoe);
      m.def("fmoe_int8_g1u0", &fmoe_int8_g1u0);
      m.def("fmoe_int8_g1u0_a16", &fmoe_int8_g1u0_a16);
      m.def("pa_fwd_asm", &pa_fwd, "pa_fwd",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("max_num_blocks"),
            py::arg("K_QScale") = std::nullopt,
            py::arg("V_QScale") = std::nullopt,
            py::arg("out_") = std::nullopt);

      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor! k_dequant_scales,"
            "                        Tensor! v_dequant_scales,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");
}
