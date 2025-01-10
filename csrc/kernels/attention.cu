/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include "hip_compat.h"

#include <algorithm>
#include "dtype_fp8.cuh"
#include "quant_utils.cuh"

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

  #define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
  #define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8 {
  _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
  _B16x4 xy[2];
} _B16x8;

using _B8x8 = uint2;
using _B8x4 = int32_t; //used in builtins
using bit8_t = uint8_t;

typedef struct _B8x16 {
  _B8x8 xy[2];
} _B8x16;

////// Non temporal load stores ///////

template <typename T>
__device__ __forceinline__ T load(T* addr) {
  return addr[0];
}

template <typename T>
__device__ __forceinline__ void store(T value, T* addr) {
  addr[0] = value;
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma_instr(const _B16x4& inpA,
                                                  const _B16x4& inpB,
                                                  const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                  const _B16x4& inpB,
                                                  const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float_b16(const bit16_t& inp) {
  union tmpcvt {
    bit16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  t16.u = inp;
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)t16.f;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(t16.b);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  _B16x4 ret;
#if 0
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t16.f = (_Float16)inp[i];
      ret[i] = t16.u;
    }
    return ret;
#else
  if constexpr (std::is_same<T, _Float16>::value) {
#if 0
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t16.f = (_Float16)inp[i];
      ret[i] = t16.u;
    }
    return ret;
#else
    union h2cvt {
        __half2 h2[2];
        _B16x4 b16x4;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0],inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2],inp[3]));
    return u.b16x4;
#endif
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      union fcvt {
          uint32_t u32;
          float f32;
      } u;
      u.f32 = inp[i];
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1); //RNE with no nan/inf check
      ret[i] = uint16_t(u.u32 >> 16);
      //t16.b = __float2bfloat16(inp[i]);
      //ret[i] = t16.u;
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
#endif
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1,
                                        const _B16x4& inp2) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t1, t2, res;
  _B16x4 ret;
#if 0
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t1.u = inp1[i];
      t2.u = inp2[i];
      res.f = t1.f + t2.f;
      ret[i] = res.u;
    }
    return ret;
#else
  if constexpr (std::is_same<T, _Float16>::value) {
#if 0
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t1.u = inp1[i];
      t2.u = inp2[i];
      res.f = t1.f + t2.f;
      ret[i] = res.u;
    }
    return ret;
#else
    union h2cvt {
        _B16x4 b16x4;
        __half2 h2[2];
    } u1,u2,s;
    u1.b16x4 = inp1; 
    u2.b16x4 = inp2; 
    s.h2[0] = u1.h2[0] + u2.h2[0];
    s.h2[1] = u1.h2[1] + u2.h2[1];
    return s.b16x4;
#endif
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      union fcvt {
          float f32;
          uint32_t i32;
      } u1,u2,s;
      u1.i32 = uint32_t(inp1[i])<<16;
      u2.i32 = uint32_t(inp2[i])<<16;
      s.f32 = u1.f32 + u2.f32;
      ret[i] = uint16_t(s.i32>>16);
      //t1.u = inp1[i];
      //t2.u = inp2[i];
      //res.b = t1.b + t2.b;
      //ret[i] = res.u;
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
#endif
}

template <typename T, vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ _B16x8 scaled_convert_b8x8(const _B8x8 input,
                                                      const float scale) {
  union alignas(16) {
    uint4 u4;
    _B16x8 u16x8;
    vllm::bf16_8_t b16x8;
  } tmp;
  if constexpr (std::is_same<T, _Float16>::value) {
    tmp.u4 = vllm::fp8::scaled_convert<uint4, _B8x8, KV_DTYPE>(input, scale);
    return tmp.u16x8;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    tmp.b16x8 = vllm::fp8::scaled_convert<vllm::bf16_8_t, _B8x8, KV_DTYPE>(
        input, scale);
    return tmp.u16x8;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 scaled_convert_b8x8_custom(const _B8x8 input,
                                                      const float scale) {
  union {
    floatx4 f32x4[2];
    vllm::Float8_ f32x8;
  } tmpf8;
  tmpf8.f32x8 = vllm::fp8::vec_conversion<vllm::Float8_,uint2>(*reinterpret_cast<const uint2*>(&input));
  
  tmpf8.f32x4[0] *= scale;
  tmpf8.f32x4[1] *= scale;
  
  _B16x8 ret;
  ret.xy[0] = from_floatx4<T>(tmpf8.f32x4[0]);
  ret.xy[1] = from_floatx4<T>(tmpf8.f32x4[1]);
  return ret;
}

__device__ __forceinline__ floatx4 to_float_fp8x4(const _B8x4& inp) {
    const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
    const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
    floatx4 ret;
    ret[0] = f0[0];
    ret[1] = f0[1];
    ret[2] = f1[0];
    ret[3] = f1[1];
    return ret;
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
        _Half2 h2[2];
        _B16x4 b16x4;
    } u;
    u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0],inp[1]);
    u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2],inp[3]);
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
          uint32_t i32;
          float f32;
      } u;
      u.f32 = inp[i];
      ret[i] = uint16_t(u.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 convert_b8x8_custom(const _B8x8 input) {
#if 0
  union {
    floatx4 f32x4[2];
    vllm::Float8_ f32x8;
    _B8x8 b8x8[2];
  } tmpf8;
  tmpf8.f32x8 = vllm::fp8::vec_conversion<vllm::Float8_,uint2>(*reinterpret_cast<const uint2*>(&input));
  //tmpf8.b8x8[0] = input;
  //tmpf8.b8x8[1] = input;
#endif
  union {
      _B8x8 b8x8;
      _B8x4 b8x4[2];
  } tmp;
  tmp.b8x8 = input;
  _B16x8 ret;
  for (int i=0; i<2; i++) {
      ret.xy[i] = from_floatx4_rtz<T>( to_float_fp8x4(tmp.b8x4[i]) );
  }
  //ret.xy[0] = from_floatx4<T>(tmpf8.f32x4[0]);
  //ret.xy[1] = from_floatx4<T>(tmpf8.f32x4[1]);
  return ret;
}
///////////////////////////////////////
// grid (num_seqs, num_partitions,num_heads/gqa_ratio)
// block (partition size)
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS,5) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale,
    const float* __restrict__ fp8_out_scale_ptr) {
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;
  const int lane16id = laneid % 16;
  const int rowid = laneid / 16;

  const int seq_idx = blockIdx.x;
  const int partition_idx = blockIdx.y;
  
  constexpr int T_PAR_SIZE = 256; //partition size set to 256 TODO move to template param
  //const int partition_size = 256; //blockDim.x; //TODO this could be head_size or partition_size

  const int max_num_partitions = gridDim.y;

  const int context_len = context_lens[seq_idx];
  
  const int partition_start_token_idx = partition_idx * T_PAR_SIZE; //partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= context_len) {
    return;
  }

  constexpr int GQA_RATIO4 = DIVIDE_ROUND_UP(GQA_RATIO,4);

  __shared__ float shared_qk_max[NWARPS][16 + 1];
  __shared__ float shared_exp_sum[NWARPS][16 + 1];
  //shared_logits is used for multiple purposes
  //__shared__ _B16x4 shared_logits[NWARPS][4][16][4 + 1];
  __shared__ _B16x4 shared_logits[NWARPS][4][16][4];
    
  //for QK mfma16x16, layout is QHead/Tokenx16 across every 16 lanes, 16 Bytes HeadElements in each lane, 4x16B HeadElements across 4 rows of warp
  constexpr int ROWS_PER_WARP = WARP_SIZE / 16; //rows refers to 16 lanes; refer dpp terminology
  constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD = 16 / sizeof(cache_t); //8 for 16 bit cache type, 16 for 8 bit types
  constexpr int QKHE_PER_FETCH = CONTIGUOUS_KV_ELEMS_16B_LOAD * ROWS_PER_WARP; //each fetch across a warp fetches these many elements
  constexpr int QK_SIZE_RATIO = sizeof(scalar_t) / sizeof(cache_t); //1 for 16bit types, 2 for 8bit types
  constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH; //4xQKHE_16B across warp

  _B16x8 Qlocal[QKHELOOP][QK_SIZE_RATIO]; //note that 16 contiguous elements of Q should be fetched per lane for 8 bit cache types : QK_SIZE_RATIO changes for this

  constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);
  //constexpr int x = CONTIGUOUS_SCALAR_ELEMS_16B; //x is defined by vLLM as 16Bytes

  //constexpr int TLOOP1 = CONTIGUOUS_KV_ELEMS_16B_LOAD / 4; //mfma16x16x16 outputs 4 elements per lane: will be moved to match layout for V dwordx4 loads  
  //constexpr int TOKENS_PER_WARP1 = 16 * TLOOP1; //16 tokens across lanes * TLOOP factor
  //constexpr int T_PAR_LOOP = T_PAR_SIZE / TOKENS_PER_WARP1 / NWARPS; 
  constexpr int TOKENS_PER_WARP = T_PAR_SIZE / NWARPS; //sub partition of tokens per warp for qk calculation
  constexpr int TLOOP = TOKENS_PER_WARP / 16; //each mfma16x16x16 instruction processes 16 tokens 

  _B16x8 Klocal[TLOOP][QKHELOOP]; //this could be B8x16 too

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;
  const int total_num_heads = gridDim.z * GQA_RATIO;

  //for QK mfma, tokens in multiples of TOKENS_PER_WARP are spread across warps
  //each mfma takes QH16xT16x16HE across warp
  //repeat mfmas across QKHELOOP dimension
  //output layout from QKmfma : QH16xT4x4 16 qheads across 16 lanes, 16 tokens across 4 rowsx4 tokens per lane

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block = num_context_blocks - 1;

    const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;
    
    int kphysical_block_number[TLOOP];

    //fetch k physical block numbers
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int klocal_token_idx = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
      const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
      const int kblock_idx = (kglobal_token_idx < context_len)
                              ? kglobal_token_idx / BLOCK_SIZE
                              : last_ctx_block;
      kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
    }

#if 0 //fetch Q into registers

    const int local_qhead_idx = lane16id % GQA_RATIO;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const int64_t seq_idx64 = static_cast<int64_t>(seq_idx);
    const scalar_t* q_ptr = q + seq_idx64 * q_stride + global_qhead_idx * HEAD_SIZE + rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;

    if (lane16id < GQA_RATIO) {
        for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
            const scalar_t* q_ptr2 = q_ptr + qkhe_depth * QKHE_PER_FETCH; 
            for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
                const scalar_t* q_fetch_ptr = q_ptr2 + qkratio * CONTIGUOUS_SCALAR_ELEMS_16B;
                const _B16x8* q_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(q_fetch_ptr);
                Qlocal[qkhe_depth][qkratio] = *q_fetch_ptr_16B; 
            }
        }
    } else {
        for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
            for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
                Qlocal[qkhe_depth][qkratio].xy[0] = {0}; 
                Qlocal[qkhe_depth][qkratio].xy[1] = {0}; 
            }
        }
    }
#else //fetch Q in shared
    const int local_qhead_idx = 4 * warpid + rowid;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const int64_t seq_idx64 = static_cast<int64_t>(seq_idx);
    const scalar_t* q_ptr = q + seq_idx64 * q_stride + global_qhead_idx * HEAD_SIZE; //+ rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;

    if (local_qhead_idx < GQA_RATIO) {
        const scalar_t* q_fetch_ptr = q_ptr + lane16id * CONTIGUOUS_SCALAR_ELEMS_16B; //this works for head size 128 : 16 lanes x 8 elems = 128 elems
        const _B16x8* q_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(q_fetch_ptr);
        _B16x8 tmp = *q_fetch_ptr_16B; 
        if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
            const int offset1 = lane16id/4; //16 contiguous chunks of head elems are spread across 4x4lanes
            shared_logits[offset1][lane4id][local_qhead_idx][0] = tmp.xy[0];
            shared_logits[offset1][lane4id][local_qhead_idx][1] = tmp.xy[1];
        } else {
            for (int i=0; i<2; i++) {
                const int head_elem = lane16id * 2 + i; //element id in _B16x4 terms
                const int offset3 = head_elem % 4;
                const int offset2 = (head_elem / 4) % 4;
                const int offset1 = head_elem /4/4;
                shared_logits[offset1][offset2][local_qhead_idx][offset3] = tmp.xy[i];
            }
        }
    } 
    __syncthreads();
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
            for (int i=0; i<2; i++) {
                Qlocal[qkhe_depth][qkratio].xy[i] = shared_logits[qkhe_depth][rowid][lane16id % GQA_RATIO][2*qkratio + i];
            }
        }
    }
#endif

    constexpr int KX = 16 / sizeof(cache_t);
    const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

    const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int64_t kblock_number = static_cast<int64_t>(kphysical_block_number[token_depth]);
      const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
      const int klocal_token_idx = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
      const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
      const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE; 
      const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * KX;

      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
        const int offset1 = head_elem / KX;
        const int offset2 = head_elem % KX;
        const cache_t* k_fetch_ptr = k_ptr3 + offset1 * BLOCK_SIZE * KX + offset2;
        const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
        Klocal[token_depth][qkhe_depth] = *k_fetch_ptr_16B;
      }
    }

    constexpr int VTOKENS_PER_LANE = TOKENS_PER_WARP / ROWS_PER_WARP;//    16 * T_PAR_SIZE / 256;
    constexpr int VBLOCKS_PER_LANE = DIVIDE_ROUND_UP(VTOKENS_PER_LANE,BLOCK_SIZE);
    constexpr int VTLOOP = NWARPS; //was * TOKENS_PER_WARP / ROWS_PER_WARP / VTOKENS_PER_LANE; 
    constexpr int VTLANELOOP = DIVIDE_ROUND_UP(VTOKENS_PER_LANE , CONTIGUOUS_KV_ELEMS_16B_LOAD); //optimized for 16B fetches; assumes minimum block size is 16
    constexpr int VHELOOP = HEAD_SIZE / 16 / NWARPS;
    
    int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

    //fetch v physical block numbers
    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
     for (int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE; vblock_depth++) {
      const int vlocal_token_idx = vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP + rowid * VTOKENS_PER_LANE + vblock_depth * BLOCK_SIZE;
      const int vglobal_token_idx = partition_start_token_idx + vlocal_token_idx;
      const int vblock_idx = (vglobal_token_idx < context_len)
                              ? vglobal_token_idx / BLOCK_SIZE
                              : last_ctx_block;
      vphysical_block_number[vtoken_depth][vblock_depth] =
        block_table_seq[vblock_idx];
     }
    }

    _B16x8 Vlocal[VTLOOP][VHELOOP][VTLANELOOP]; //this could be B8x16 too
    
    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;

    //v fetches are 16head elems across lanes x 16 tokens per lane
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
      const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
      const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;

      for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
          for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          const int vblock_depth = vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD / BLOCK_SIZE; 
          //const int token_depth = vtoken_depth * VBLOCKS_PER_LANE + vblock_depth; 
          const int64_t vblock_number = static_cast<int64_t>(vphysical_block_number[vtoken_depth][vblock_depth]);
          const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride);

              const cache_t* v_fetch_ptr = v_ptr3 + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
              const _B16x8* v_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(v_fetch_ptr);
              Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B;
          }
      }
    }

    //__syncthreads(); //if using shared Q
    float scale2 = scale;
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
        scale2 *= k_scale;
    }

    floatx4 dout[TLOOP];
#if 1 //Q stored in registers
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      dout[token_depth] = {0};
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
            for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
              for (int i=0; i<2; i++) {
                dout[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Klocal[token_depth][qkhe_depth].xy[i],
                        Qlocal[qkhe_depth][qkratio].xy[i], dout[token_depth]);
              }
            }
        } else { //kv cache dtype fp8
            auto Ktmp = Klocal[token_depth][qkhe_depth];
            _B8x16 Ktmp8x16 = *reinterpret_cast<_B8x16*>(&Ktmp);
            for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
              _B8x8 Ktmp8x8 = Ktmp8x16.xy[qkratio];
              _B16x8 Klocaltmp = convert_b8x8_custom<scalar_t>(Ktmp8x8);
              for (int i=0; i<2; i++) {
                dout[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Klocaltmp.xy[i],
                        Qlocal[qkhe_depth][qkratio].xy[i], dout[token_depth]);
              }
            }
        }
      }
      dout[token_depth] *= scale2;
    }

#else //Q in shared
    _B16x4 tmpQ[QKHELOOP][2];
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        tmpQ[qkhe_depth][0] = shared_logits[qkhe_depth][rowid][lane16id][0];
        tmpQ[qkhe_depth][1] = shared_logits[qkhe_depth][rowid][lane16id][1];
    }

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      dout[token_depth] = {0};
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        //for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
          for (int i=0; i<2; i++) {
            dout[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Klocal[token_depth][qkhe_depth].xy[i],
                    tmpQ[qkhe_depth][i], //shared_logits[qkhe_depth][rowid][lane16id][i],
                    dout[token_depth]);
          }
        //}
      }
      dout[token_depth] *= scale;
    }
#endif

#if 0 //DEBUG ONLY qk * scale
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        auto qkout_ptr2 = qkout_ptr + warpid * TLOOP * 16 + token_depth * 16 + rowid * 4; 
        auto qkout_write_ptr = reinterpret_cast<_B16x4 *>(qkout_ptr2);
        auto tmp = from_floatx4<scalar_t>(dout[token_depth]);
        *qkout_write_ptr = tmp;
    }
#endif

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;

    const int qkout_token_idx = partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 4;

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        const int local_token_idx = qkout_token_idx + token_depth * 16;
        for (int i=0; i<4; i++) {
            const float tmp = (local_token_idx + i < context_len) ? dout[token_depth][i] : -FLT_MAX;
            qk_max = fmaxf(qk_max, tmp);
        }
    }

    for (int mask = WARP_SIZE/2; mask >= 16; mask/=2) {
        qk_max = fmaxf(qk_max, __shfl_xor(qk_max,mask));
    }


    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        const int local_token_idx = qkout_token_idx + token_depth * 16;
        for (int i=0; i<4; i++) {
            const float tmp = (local_token_idx + i < context_len) ? __expf(dout[token_depth][i] - qk_max) : 0.0f;
            dout[token_depth][i] = tmp;
            exp_sum += tmp;
        }
    }

    for (int mask = WARP_SIZE/2; mask >= 16; mask/=2) {
        exp_sum += __shfl_xor(exp_sum,mask);
    }

    __syncthreads(); //sync before writing to shared mem

    float* shared_mem = reinterpret_cast<float*>(shared_logits); 
    if (laneid < 16) {
        //shared_qk_max[warpid][lane16id] = qk_max;
        //shared_exp_sum[warpid][lane16id] = exp_sum;
        const int qk_max_offset = warpid*16 + lane16id;
        shared_mem[qk_max_offset] = qk_max;
        const int exp_sum_offset = NWARPS*16 + qk_max_offset;
        shared_mem[exp_sum_offset] = exp_sum;
    }

#if 0 //DEBUG ONLY
    //scalar_t* qkout_ptr = out +
    //                      seq_idx * total_num_heads * T_PAR_SIZE + lane16id * T_PAR_SIZE;
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        //auto qkout_ptr2 = qkout_ptr + warpid * TLOOP * 16 + token_depth * 16 + rowid * 4; 
        //auto qkout_write_ptr = reinterpret_cast<_B16x4 *>(qkout_ptr2);
        auto tmp = from_floatx4<scalar_t>(dout[token_depth]);
        shared_tokens[warpid][token_depth][lane16id][rowid] = tmp;
        //*qkout_write_ptr = tmp;
    }
#endif
    __syncthreads();

    float partition_qk_max = -FLT_MAX;
    float warp_qk_max_exp[NWARPS];
    float partition_exp_sum = 0.0f;

    for (int w=0; w<NWARPS; w++) {
        //warp_qk_max_exp[w] = shared_qk_max[w][lane16id];
        warp_qk_max_exp[w] = shared_mem[w*16+lane16id];
        partition_qk_max = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
    }

    for (int w=0; w<NWARPS; w++) {
        warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
        //partition_exp_sum += shared_exp_sum[w][lane16id] * warp_qk_max_exp[w];
        partition_exp_sum += shared_mem[NWARPS*16 + w*16 + lane16id] * warp_qk_max_exp[w];
    }

    const float inv_sum_scale = __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

    __syncthreads(); //new

    //__shared__ _B16x4 shared_logits[NWARPS][TLOOP][16][VTOKENS_PER_LANE/4 + 1];
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        dout[token_depth] *= inv_sum_scale;
        shared_logits[warpid][token_depth][lane16id][rowid] = from_floatx4<scalar_t>(dout[token_depth]);
    }

    if (threadIdx.x < GQA_RATIO) {
        const int qhead_idx = lane16id;
        const int offset = seq_idx * total_num_heads * max_num_partitions + (wg_start_head_idx + qhead_idx) * max_num_partitions + partition_idx;
        max_logits[offset] = partition_qk_max;
        exp_sums[offset] = partition_exp_sum;
    }
    
    __syncthreads();

#if 0 //DEBUG ONLY
    scalar_t* qkout_ptr = out +
                          seq_idx * total_num_heads * T_PAR_SIZE + lane16id * T_PAR_SIZE;
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        auto qkout_ptr2 = qkout_ptr + warpid * TLOOP * 16 + token_depth * 16 + rowid * 4; 
        auto qkout_write_ptr = reinterpret_cast<_B16x4 *>(qkout_ptr2);
        //dout[token_depth] *= inv_sum_scale[warpid];
        //auto tmp = from_floatx4<scalar_t>(dout[token_depth]);
        auto tmp = shared_tokens[warpid][token_depth][lane16id][rowid];
        *qkout_write_ptr = tmp;
    }
#endif
#if 0
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
        for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
         for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
          for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
              _B16x8 Vtmp = Vlocal[vtoken_depth][vhe_depth][vfetch_depth];
              _B8x16 Vtmp8x16 = *reinterpret_cast<_B8x16*>(&Vtmp);
              for (int j=0; j<2; j++) {
               _B8x8 Vtmp8x8 = Vtmp8x16.xy[j]; 
               _B16x8 Vlocaltmp = convert_b8x8_custom<scalar_t>(Vtmp8x8);
               for (int i=0; i<2; i++) {
                const int offset = 4*rowid + 2*j + i; 
                const int offset1 = offset % 4;
                const int offset2 = offset / 4;
                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Vlocaltmp.xy[i],
                        shared_logits[vtoken_depth][offset2][lane16id][offset1],
                        tmp_out);
               }
              }
          }
        }
#endif
    _B16x4 outelems[VHELOOP];
    _B16x4 S_local[VTLOOP][2][2];
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
        for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
          //for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
              for (int j=0; j<2; j++) {
               for (int i=0; i<2; i++) {
                const int offset = 4*rowid + 2*j + i; 
                const int offset1 = offset % 4;
                const int offset2 = offset / 4;
                S_local[vtoken_depth][j][i] = shared_logits[vtoken_depth][offset2][lane16id][offset1];
               }
              }
          //}
        }
    }
    //v layout: 16he across lanes x 16 tokens per lane

    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
        floatx4 tmp_out = {0};

        for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {

        if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
          for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
              for (int i=0; i<2; i++) {
                //TODO generalize this for 8 bit dtypes: each lane needs 2*vfetch_depth + 2 _B16x4 K/token dimension elems; each row is multiplied by a factor of 4
                //layout: lane in depth dimension | row across ->
                //0 4 8  12
                //1 5 9  13
                //2 6 10 14
                //3 7 11 15
                const int offset = rowid * VTLANELOOP * 2 + 2*vfetch_depth + i; 
                const int offset1 = offset % 4; //4 corresponds to ROWS_PER_WARP
                const int offset2 = offset / 4;
#if 0
                //if output format is 16 head elems across 16 lanes, 16 qheads spread across 4 rows
                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(shared_logits[vtoken_depth][offset2][lane16id][offset1],
                        Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i], tmp_out);
#else
                //if output format is 16 qheads across 16 lanes, 16 head elems spread across 4 rows
                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i],
                        shared_logits[vtoken_depth][offset2][lane16id][offset1],
                        tmp_out);
#endif
              }
          }
        } else {
          for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
              _B16x8 Vtmp = Vlocal[vtoken_depth][vhe_depth][vfetch_depth];
              _B8x16 Vtmp8x16 = *reinterpret_cast<_B8x16*>(&Vtmp);
              for (int j=0; j<2; j++) {
               _B8x8 Vtmp8x8 = Vtmp8x16.xy[j]; 
               _B16x8 Vlocaltmp = convert_b8x8_custom<scalar_t>(Vtmp8x8);
               for (int i=0; i<2; i++) {
                const int offset = 4*rowid + 2*j + i; 
                const int offset1 = offset % 4;
                const int offset2 = offset / 4;
                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(Vlocaltmp.xy[i],
                          S_local[vtoken_depth][j][i], 
                        tmp_out);
                        //shared_logits[vtoken_depth][offset2][lane16id][offset1],
                        //tmp_out);
               }
              }
          }
            
        }
        }
        if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
            tmp_out *= v_scale;
        }
        outelems[vhe_depth] = from_floatx4<scalar_t>(tmp_out);
    }

#if 1
    __syncthreads();
    
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
        shared_logits[warpid][vhe_depth][lane16id][rowid] = outelems[vhe_depth]; //lane16 id head dimension; rowid head element dimension
    }

    __syncthreads();

    if (warpid == 0) {
        _B16x8 vout[GQA_RATIO4];
        for (int h = 0; h < GQA_RATIO4; h++) {
            const int local_head_idx = 4 * h + rowid;
            const int head_elem_idx = lane16id * 8;
            const int offset1 = (head_elem_idx / 16)%4;
            const int offset2 = head_elem_idx / 16 / NWARPS;
            const int offset3 = (head_elem_idx / 4)%4;
            for (int i=0; i<2; i++) {
                vout[h].xy[i] = shared_logits[offset1][offset2][local_head_idx][offset3+i];
            }
        }

        const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions; 
        scalar_t* out_ptr = out +
                          seq_idx * total_num_heads * hsz_maxp_mult + partition_idx * HEAD_SIZE;
        for (int h = 0; h < GQA_RATIO4; h++) {
            const int local_head_idx = 4 * h + rowid;
            if (local_head_idx < GQA_RATIO) {
                const int out_head_idx = wg_start_head_idx + local_head_idx;
                scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
                const int head_elem_idx = lane16id * 8;
                scalar_t* out_ptr3 = out_ptr2 + head_elem_idx;
                _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
                *out_ptr_B16x8 = vout[h];
            }
        }

    }
#endif

#if 0
    //if output format is 16 he across 16 lanes, 16 qheads spread across 4 rows
    const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions; 
    scalar_t* out_ptr = out +
                          seq_idx * total_num_heads * hsz_maxp_mult + partition_idx * HEAD_SIZE;

    const int vhe_offset = warpid * 16 + lane16id;

    #pragma unroll
    for (int i=0; i<4; i++) {
        const int local_head_idx = 4*rowid + i;
        if (local_head_idx < GQA_RATIO) { 
            const int out_head_idx = wg_start_head_idx + local_head_idx;
            scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
            for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
                const int vhead_elem = vhe_depth * NWARPS * 16 + vhe_offset;
                scalar_t* out_ptr3 = out_ptr2 + vhead_elem;
                bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr3);
                *out_ptr_b16 = outelems[vhe_depth][i];
            }
        }
    }
#endif
#if 0
    //if output format is 16 qheads across 16 lanes, 16 he spread across 4 rows
    if (lane16id < GQA_RATIO) {
        const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions; 
        scalar_t* out_ptr = out +
                          seq_idx * total_num_heads * hsz_maxp_mult + partition_idx * HEAD_SIZE;
        const int local_head_idx = lane16id;
        const int out_head_idx = wg_start_head_idx + local_head_idx;
        scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
        const int vhe_offset = warpid * 16 + rowid * 4;
        for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
            const int vhead_elem = vhe_depth * NWARPS * 16 + vhe_offset;
            scalar_t* out_ptr3 = out_ptr2 + vhead_elem;
            _B16x4* out_ptr_B16x4 = reinterpret_cast<_B16x4*>(out_ptr3);
            *out_ptr_B16x4 = outelems[vhe_depth];
        }
    }
#endif
#if 0 //DEBUG ONLY 
    floatx4 partition_out[VHELOOP];
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
        partition_out[vhe_depth] = {0};
        for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
            partition_out[vhe_depth] += inv_sum_scale[vtoken_depth] * vout[vhe_depth][vtoken_depth];
        }
    }
#endif
#if 0 //DEBUG ONLY
    if (laneid < GQA_RATIO) {
        auto* exp_sums_ptr = exp_sums + seq_idx * 8 * max_num_partitions +  partition_idx;
        floatx4 tmp = {0};
        //for (int t=0; t<TLOOP; t++) {
        //    tmp += dout[t];
        //}
        for (int h=0; h<VHELOOP; h++) {
            tmp += partition_out[h];
        }
        tmp *= shared_qk_max[warpid][lane16id];
        tmp *= shared_exp_sum[warpid][lane16id];
        auto tmp16 = addx4<scalar_t>(from_floatx4<scalar_t>(tmp), shared_tokens[warpid][lane4id][lane16id][rowid]);
        
        float2 tmpf = *reinterpret_cast<float2*>(&tmp16);
        *exp_sums_ptr = laneid%2 == 0 ? tmpf.x : tmpf.y;
    }
#endif
}
/////////////////////////////////////////////////////////////
// grid (num_seqs, num_partitions,num_heads/gqa_ratio)
// block (partition size)
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale,
    const float* __restrict__ fp8_out_scale_ptr) {
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;

  const int seq_idx = blockIdx.x;
  const int partition_idx = blockIdx.y;
  const int partition_size = blockDim.x;
  const int max_num_partitions = gridDim.y;

  const int context_len = context_lens[seq_idx];
  const int partition_start_token_idx = partition_idx * partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= context_len) {
    return;
  }
  constexpr int QHLOOP =
      DIVIDE_ROUND_UP(GQA_RATIO, 4);  // each 4 lanes fetch 4 different qheads,
                                      // total qheads =8, so qhloop is 2
  constexpr int GQA_RATIO4 = 4 * QHLOOP;
  __shared__ float shared_qk_max[NWARPS][GQA_RATIO4 + 1];
  __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4 + 1];
  _B16x8 Qlocal[QHLOOP];
  constexpr int x = 16 / sizeof(scalar_t);
  constexpr int KHELOOP = HEAD_SIZE / x;
  _B16x8 Klocal[KHELOOP];
  _B8x8 Klocalb8[KHELOOP];
  constexpr int VHELOOP =
      HEAD_SIZE /
      WARP_SIZE;  // v head_size dimension is distributed across lanes
  constexpr int VTLOOP = 8;  // 16 separate 4xtokens across warp -> 16/2
                             // 8xtokens
  constexpr int VBLOCKS = 8 * VTLOOP / BLOCK_SIZE;
  int vphysical_blocks[VBLOCKS];
  _B16x8 Vlocal[VHELOOP][VTLOOP];
  _B8x8 Vlocalb8[VHELOOP][VTLOOP];
  floatx4 dout[QHLOOP];
  float qk_max[QHLOOP];
  __shared__ _B16x4 vout_shared[QHLOOP][VHELOOP][WARP_SIZE][NWARPS + 1];
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    dout[h] = {0};
    qk_max[h] = -FLT_MAX;
  }

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;

  const int warp_start_token_idx =
      partition_start_token_idx + warpid * WARP_SIZE;

  if (warp_start_token_idx >= context_len) {  // warp out of context
  #pragma unroll
    for (int h = 0; h < GQA_RATIO4; h++) {
      shared_qk_max[warpid][h] = -FLT_MAX;
      shared_exp_sum[warpid][h] = 0.0f;
    }
  } else {  // warp within context

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block = num_context_blocks - 1;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    const int local_token_idx = threadIdx.x;
    const int global_token_idx = partition_start_token_idx + local_token_idx;

    const int block_idx = (global_token_idx < context_len)
                              ? global_token_idx / BLOCK_SIZE
                              : last_ctx_block;
    // fetch block number for q and k
    // int32 physical_block_number leads to overflow when multiplied with
    // kv_block_stride
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // fetch vphysical block numbers up front

    const int warp_start_block_idx = warp_start_token_idx / BLOCK_SIZE;
    if constexpr (GQA_RATIO < 12) {
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        const int vblock_idx = warp_start_block_idx + b;
        const int vblock_idx_ctx =
            (vblock_idx <= last_ctx_block) ? vblock_idx : last_ctx_block;
        vphysical_blocks[b] = block_table[vblock_idx_ctx];
      }
    }

    // each 4 lanes fetch 8 helems, so warp fetches 8*16 = 128 helems
    const scalar_t* q_ptr =
        q + seq_idx * q_stride + wg_start_head_idx * HEAD_SIZE;
    const _B16x8* q_ptrh8 = reinterpret_cast<const _B16x8*>(q_ptr);
    const int qhead_elemh8 = laneid / 4;
  #pragma unroll
    for (int h = 0; h < QHLOOP - 1; h++) {
      const int qhead_idx = h * 4 + lane4id;
      Qlocal[h] = q_ptrh8[qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    }
    const int final_qhead_idx = 4 * (QHLOOP - 1) + lane4id;
    if (final_qhead_idx < GQA_RATIO) {
      Qlocal[QHLOOP - 1] =
          q_ptrh8[final_qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    } else {
      Qlocal[QHLOOP - 1].xy[0] = {0};
      Qlocal[QHLOOP - 1].xy[1] = {0};
    }

    const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                           wg_start_kv_head_idx * kv_head_stride;

    const int physical_block_offset =
        local_token_idx % BLOCK_SIZE;  // since x=half8, physical_block_offset
                                       // is already cast as _H8
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] = k_ptrh8[d * BLOCK_SIZE + physical_block_offset];
      }
    } else {
      constexpr int X = 16 / sizeof(cache_t);
      const cache_t* k_ptr2 = k_ptr + physical_block_offset * X;
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        const int head_elem = d * 8;
        const int offset1 = head_elem / X;
        const int offset2 = head_elem % X;
        const cache_t* k_ptr3 = k_ptr2 + offset1 * BLOCK_SIZE * X + offset2;
        Klocalb8[d] = *reinterpret_cast<const _B8x8*>(k_ptr3);
      }
    }

#if 1
    float alibi_slope[QHLOOP];
    if (alibi_slopes != nullptr) {
  #pragma unroll
      for (int h = 0; h < QHLOOP; h++) {
        const int qhead_idx = h * 4 + lane4id;
        alibi_slope[h] = (qhead_idx < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + qhead_idx]
                             : 0.f;
      }
    }
#endif
#if 0
    float alibi_slope;
    const int lane16id = laneid % 16;
    if (alibi_slopes != nullptr) {
            alibi_slope = (lane16id < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + lane16id]
                             : 0.f;
  //#pragma unroll
   //   for (int h = 0; h < QHLOOP; h++) {
     //     for (int i=0; i<4; i++) {
      //      const int qhead_idx = h * 4 + i;
      //      alibi_slope[qhead_idx] = (qhead_idx < GQA_RATIO)
      //                       ? alibi_slopes[wg_start_head_idx + qhead_idx]
      //                       : 0.f;
      //    }
       //}
      //}
    }
#endif

    // fetch vphysical block numbers up front
    if constexpr (GQA_RATIO >= 12) {
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        const int vblock_idx = warp_start_block_idx + b;
        const int vblock_idx_ctx =
            (vblock_idx <= last_ctx_block) ? vblock_idx : last_ctx_block;
        vphysical_blocks[b] = block_table[vblock_idx_ctx];
      }
    }

#if 1 //fetch vcache in normal case
    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B16x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B16x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
          }
        }
      }
    } //if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
#endif
#if 1 //fetch vcache in fp8 case
    else { // if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)
      const _B8x8* v_ptrh8 = reinterpret_cast<const _B8x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B8x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B8x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocalb8[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
            //const _B8x8 Vlocalb8 = v_ptrh8be[d];
            //Vlocal[h][b * BLOCK_SIZE / 8 + d] =
            //    scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8, v_scale);
          }
        }
      }
    }
#endif
#if 0 //cvt kf8 to kf/bf16 up front
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] =
            //scaled_convert_b8x8<scalar_t, KV_DTYPE>(Klocalb8[d], k_scale);
            convert_b8x8_custom<scalar_t>(Klocalb8[d]);
      }
    }
#endif

      /*Klocal[x] = scaled_convert_b8x8<scalar_t, KV_DTYPE>(Klocalb8[x], k_scale); \*/
      /*Klocal[x] = scaled_convert_b8x8_custom<scalar_t>(Klocalb8[x], k_scale); \*/
#define QK_mfma(x) \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) { \
      Klocal[x] = convert_b8x8_custom<scalar_t>(Klocalb8[x]); \
    } \
    for (int h = 0; h < QHLOOP; h++) { \
      dout[h] = gcn_mfma_instr<scalar_t, 4, x, 0>(Qlocal[h].xy[0], \
                                                  Klocal[x].xy[0], dout[h]);\
      dout[h] = gcn_mfma_instr<scalar_t, 4, x, 0>(Qlocal[h].xy[1], \
                                                  Klocal[x].xy[1], dout[h]);\
    }

  //#pragma unroll
    //for (int h = 0; h < QHLOOP; h++) {
      QK_mfma(0);
      QK_mfma(1);
      QK_mfma(2);
      QK_mfma(3);
      QK_mfma(4);
      QK_mfma(5);
      QK_mfma(6);
      QK_mfma(7);
      if constexpr (KHELOOP > 8) {
        QK_mfma(8);
        QK_mfma(9);
        QK_mfma(10);
        QK_mfma(11);
        QK_mfma(12);
        QK_mfma(13);
        QK_mfma(14);
        QK_mfma(15);
      }
    //}
#undef QK_mfma
    float scale2 = scale;
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
        scale2 *= k_scale;
    }
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      dout[h] *= scale2;
      //if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
      //  dout[h] *= k_scale;
      //}
    }
#if 0
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      dout[h] = gcn_mfma_instr<scalar_t, 4, 0, 0>(Qlocal[h].xy[0],
                                                  Klocal[0].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 0, 0>(Qlocal[h].xy[1],
                                                  Klocal[0].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 1, 0>(Qlocal[h].xy[0],
                                                  Klocal[1].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 1, 0>(Qlocal[h].xy[1],
                                                  Klocal[1].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 2, 0>(Qlocal[h].xy[0],
                                                  Klocal[2].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 2, 0>(Qlocal[h].xy[1],
                                                  Klocal[2].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 3, 0>(Qlocal[h].xy[0],
                                                  Klocal[3].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 3, 0>(Qlocal[h].xy[1],
                                                  Klocal[3].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 4, 0>(Qlocal[h].xy[0],
                                                  Klocal[4].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 4, 0>(Qlocal[h].xy[1],
                                                  Klocal[4].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 5, 0>(Qlocal[h].xy[0],
                                                  Klocal[5].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 5, 0>(Qlocal[h].xy[1],
                                                  Klocal[5].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 6, 0>(Qlocal[h].xy[0],
                                                  Klocal[6].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 6, 0>(Qlocal[h].xy[1],
                                                  Klocal[6].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 7, 0>(Qlocal[h].xy[0],
                                                  Klocal[7].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 7, 0>(Qlocal[h].xy[1],
                                                  Klocal[7].xy[1], dout[h]);
      if constexpr (KHELOOP > 8) {
        dout[h] = gcn_mfma_instr<scalar_t, 4, 8, 0>(Qlocal[h].xy[0],
                                                    Klocal[8].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 8, 0>(Qlocal[h].xy[1],
                                                    Klocal[8].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 9, 0>(Qlocal[h].xy[0],
                                                    Klocal[9].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 9, 0>(Qlocal[h].xy[1],
                                                    Klocal[9].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 10, 0>(Qlocal[h].xy[0],
                                                     Klocal[10].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 10, 0>(Qlocal[h].xy[1],
                                                     Klocal[10].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 11, 0>(Qlocal[h].xy[0],
                                                     Klocal[11].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 11, 0>(Qlocal[h].xy[1],
                                                     Klocal[11].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 12, 0>(Qlocal[h].xy[0],
                                                     Klocal[12].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 12, 0>(Qlocal[h].xy[1],
                                                     Klocal[12].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 13, 0>(Qlocal[h].xy[0],
                                                     Klocal[13].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 13, 0>(Qlocal[h].xy[1],
                                                     Klocal[13].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 14, 0>(Qlocal[h].xy[0],
                                                     Klocal[14].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 14, 0>(Qlocal[h].xy[1],
                                                     Klocal[14].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 15, 0>(Qlocal[h].xy[0],
                                                     Klocal[15].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 15, 0>(Qlocal[h].xy[1],
                                                     Klocal[15].xy[1], dout[h]);
      }  // KHELOOP>8
      dout[h] *= scale;
    }
#endif

#if 0
    if (alibi_slopes != nullptr) {
      float alibi_slope_local[GQA_RATIO];
#define DPP_BCAST_ASM(id) asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:id " : "=v"(alibi_slope_local[id]) : "v"(alibi_slope));
      //for (int head=0; head < 16; head++) {
        //DPP_BCAST_ASM(0);
        if constexpr(GQA_RATIO>0) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:0 " : "=v"(alibi_slope_local[0]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>1) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:1 " : "=v"(alibi_slope_local[1]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>2) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:2 " : "=v"(alibi_slope_local[2]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>3) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:3 " : "=v"(alibi_slope_local[3]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>4) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:4 " : "=v"(alibi_slope_local[4]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>5) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:5 " : "=v"(alibi_slope_local[5]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>6) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:6 " : "=v"(alibi_slope_local[6]) : "v"(alibi_slope));}
        if constexpr(GQA_RATIO>7) { asm("s_nop 0\n\tv_mov_b32_dpp %0, %1 row_newbcast:7 " : "=v"(alibi_slope_local[7]) : "v"(alibi_slope));}
      //}

      const int alibi_offset = global_token_idx - context_len + 1;
  #pragma unroll
      for (int h = 0; h < QHLOOP; h++) {
  #pragma unroll
        for (int i = 0; i < 4; i++) {
          dout[h][i] += alibi_slope_local[4*h+i] * alibi_offset;
        }
      }
    }
#endif
  // transpose dout so that 4 token ids are in each lane, and 4 heads are across
  // 4 lanes
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
#if 1
      floatx4 tmp = {0};
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        const float B = (lane4id == i) ? 1.0f : 0.0f;
        // const float A = (global_token_idx < context_len) ? dout[h][i] : 0.0f;
        tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(dout[h][i], B, tmp, 0, 0, 0);
        // tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(A, B, tmp, 0, 0, 0);
      }
      dout[h] = tmp;
#endif
#if 0
      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] " : "=v"(dout[h][1]) : "v"(dout[h][1]) );
      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] " : "=v"(dout[h][2]) : "v"(dout[h][2]) );
      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[3,2,1,0] " : "=v"(dout[h][3]) : "v"(dout[h][3]) );

      bool mask = (lane4id % 2) == 1;
      float tmp = dout[h][1];
      dout[h][1] = mask ? dout[h][0] : dout[h][1];
      dout[h][0] = mask ? tmp : dout[h][0];
      tmp = dout[h][3];
      dout[h][3] = mask ? dout[h][2] : dout[h][3];
      dout[h][2] = mask ? tmp : dout[h][2];

      mask = (lane4id>>1) == 1;
      tmp = dout[h][2];
      dout[h][2] = mask ? dout[h][0] : dout[h][2];
      dout[h][0] = mask ? tmp : dout[h][0];
      tmp = dout[h][3];
      dout[h][3] = mask ? dout[h][1] : dout[h][3];
      dout[h][1] = mask ? tmp : dout[h][1];


      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] " : "=v"(dout[h][1]) : "v"(dout[h][1]) );
      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] " : "=v"(dout[h][2]) : "v"(dout[h][2]) );
      asm("s_nop 0\n\t v_mov_b32_dpp %0, %1 quad_perm:[3,2,1,0] " : "=v"(dout[h][3]) : "v"(dout[h][3]) );

#endif
    }

    const int lane4_token_idx = 4 * (global_token_idx >> 2);
#if 1 //alibi after transpose
    const int alibi_offset = lane4_token_idx - context_len + 1;
    if (alibi_slopes != nullptr) {
  #pragma unroll
      for (int h = 0; h < QHLOOP; h++) {
  #pragma unroll
        for (int i = 0; i < 4; i++) {
          dout[h][i] += alibi_slope[h] * (alibi_offset + i);
        }
      }
    }
#endif

    const int bpermute_mask = 4*(16*((laneid>>2)%4) + lane4id);

  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      qk_max[h] = -FLT_MAX;
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        qk_max[h] = (lane4_token_idx + i < context_len)
                        ? fmaxf(qk_max[h], dout[h][i])
                        : qk_max[h];
      }
  #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 64; mask /= 2) {
        qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h], mask));
      }
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4" : "=v"(qk_max[h]) : "v"(qk_max[h]), "v"(qk_max[h]) );
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8" : "=v"(qk_max[h]) : "v"(qk_max[h]), "v"(qk_max[h]) );

      //asm("v_nop\n v_nop\n ds_bpermute_b32 %0, %1, %2 \n s_waitcnt lgkmcnt(0)" : "=v"(qk_max[h]) : "v"(bpermute_mask), "v"(qk_max[h]) );
      
      //qk_max[h] = __builtin_amdgcn_ds_bpermute(bpermute_mask, qk_max[h]);
      auto tmp = __builtin_amdgcn_ds_bpermute(bpermute_mask, *reinterpret_cast<int*>(&qk_max[h]));
      qk_max[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4" : "=v"(qk_max[h]) : "v"(qk_max[h]), "v"(qk_max[h]) );
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8" : "=v"(qk_max[h]) : "v"(qk_max[h]), "v"(qk_max[h]) );
    }


    float exp_sum[QHLOOP];
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      exp_sum[h] = 0.0f;
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        dout[h][i] = (lane4_token_idx + i < context_len)
                         ? __expf(dout[h][i] - qk_max[h])
                         : 0.0f;
        exp_sum[h] += dout[h][i];
      }
  #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 64; mask /= 2) {
        exp_sum[h] += __shfl_xor(exp_sum[h], mask);
      }
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4" : "=v"(exp_sum[h]) : "v"(exp_sum[h]), "v"(exp_sum[h]) );
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8" : "=v"(exp_sum[h]) : "v"(exp_sum[h]), "v"(exp_sum[h]) );

      //asm("v_nop\n v_nop\n ds_bpermute_b32 %0, %1, %2 \n s_waitcnt lgkmcnt(0)" : "=v"(exp_sum[h]) : "v"(bpermute_mask), "v"(exp_sum[h]) );
      //exp_sum[h] = __builtin_amdgcn_ds_bpermute(bpermute_mask, exp_sum[h]);
      auto tmp = __builtin_amdgcn_ds_bpermute(bpermute_mask, *reinterpret_cast<int*>(&exp_sum[h]));
      exp_sum[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4" : "=v"(exp_sum[h]) : "v"(exp_sum[h]), "v"(exp_sum[h]) );
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8" : "=v"(exp_sum[h]) : "v"(exp_sum[h]), "v"(exp_sum[h]) );
    }

    if (laneid<4) {
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      const int head_idx = 4 * h + lane4id;
      shared_qk_max[warpid][head_idx] = qk_max[h];
      shared_exp_sum[warpid][head_idx] = exp_sum[h];
    }
    }
  }  // warp within context

  __syncthreads();

  const int num_heads = gridDim.z * GQA_RATIO;
  float* max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + partition_idx;
  float* exp_sums_ptr =
      exp_sums + seq_idx * num_heads * max_num_partitions + partition_idx;
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    float global_qk_max = -FLT_MAX;
    float warp_qk_max[NWARPS];
    const int head_idx = 4 * h + lane4id;
  #pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      warp_qk_max[w] = shared_qk_max[w][head_idx];
      global_qk_max = fmaxf(global_qk_max, warp_qk_max[w]);
    }
    float global_exp_sum = 0.0f;
  #pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      global_exp_sum +=
          shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
    }
    if (head_idx < GQA_RATIO) {
      max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_qk_max;
      exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_exp_sum;
    }
    const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) *
                                       __expf(qk_max[h] - global_qk_max);
    dout[h] *= global_inv_sum_scale;
  }
  // logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there
  // are 4x16 tokens across warp
  _B16x4 logits[QHLOOP];
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    logits[h] = from_floatx4<scalar_t>(dout[h]);
  }


  if (warp_start_token_idx >= context_len) {  // warp out of context
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = {0};
      }
    }
  } else {  // warp in context
#if 0 //fetch v cache
    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B16x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B16x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
          }
        }
      }
    } //if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)

    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
      const _B8x8* v_ptrh8 = reinterpret_cast<const _B8x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B8x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B8x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocalb8[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
            //const _B8x8 Vlocalb8 = v_ptrh8be[d];
            //Vlocal[h][b * BLOCK_SIZE / 8 + d] =
            //    scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8, v_scale);
          }
        }
      }
    }
#endif
#if 0 //cvt vf8 ->f16/bf16 up front
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        for (int b=0; b < VTLOOP; b++) {
          //Vlocal[vh][b] = scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8[vh][b], v_scale);
          Vlocal[vh][b] = convert_b8x8_custom<scalar_t>(Vlocalb8[vh][b]);
        }
      }
    }
#endif

        /*Vlocal[vh][x] = scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8[vh][x], v_scale);\*/
        /*Vlocal[vh][x] = scaled_convert_b8x8_custom<scalar_t>(Vlocalb8[vh][x], v_scale);\*/
  #define SV_mfma(x) \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {\
        Vlocal[vh][x] = convert_b8x8_custom<scalar_t>(Vlocalb8[vh][x]);\
    }\
    for (int qh = 0; qh < QHLOOP; qh++) { \
        acc[qh] = gcn_mfma_instr<scalar_t, 4, 2*x, 0>(logits[qh], Vlocal[vh][x].xy[0], \
                                                acc[qh]); \
        acc[qh] = gcn_mfma_instr<scalar_t, 4, 2*x+1, 0>(logits[qh], Vlocal[vh][x].xy[1], \
                                                acc[qh]); \
    }
#if 0
    floatx4 acc[QHLOOP][VHELOOP];
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        acc[qh][vh] = {0};
      }
    }
#endif
  //#pragma unroll
    // for (int qh = 0; qh < QHLOOP; qh++) {
  // iterate over each v head elem (within head_size)
  //#pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        floatx4 acc[QHLOOP];
        for (int qh = 0; qh < QHLOOP; qh++) {
                acc[qh] = {0};
        }
        // iterate over tokens
        SV_mfma(0);
        SV_mfma(1);
        SV_mfma(2);
        SV_mfma(3);
        SV_mfma(4);
        SV_mfma(5);
        SV_mfma(6);
        SV_mfma(7);
#if 0
        SV_mfma(8);
        SV_mfma(9);
        SV_mfma(10);
        SV_mfma(11);
        SV_mfma(12);
        SV_mfma(13);
        SV_mfma(14);
        SV_mfma(15);
#endif
        for (int qh = 0; qh < QHLOOP; qh++) {
            if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
                acc[qh] *= v_scale;
            }
            vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc[qh]);
        }
      }
    //}

#if 0
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc[qh][vh]);
      }
    }
#endif

#undef SV_mfma
#if 0
  // iterate across heads
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  // iterate over each v head elem (within head_size)
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        floatx4 acc = {0};
        // iterate over tokens
        acc = gcn_mfma_instr<scalar_t, 4, 0, 0>(logits[qh], Vlocal[vh][0].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 1, 0>(logits[qh], Vlocal[vh][0].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 2, 0>(logits[qh], Vlocal[vh][1].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 3, 0>(logits[qh], Vlocal[vh][1].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 4, 0>(logits[qh], Vlocal[vh][2].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 5, 0>(logits[qh], Vlocal[vh][2].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 6, 0>(logits[qh], Vlocal[vh][3].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 7, 0>(logits[qh], Vlocal[vh][3].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 8, 0>(logits[qh], Vlocal[vh][4].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 9, 0>(logits[qh], Vlocal[vh][4].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 10, 0>(logits[qh],
                                                 Vlocal[vh][5].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 11, 0>(logits[qh],
                                                 Vlocal[vh][5].xy[1], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 12, 0>(logits[qh],
                                                 Vlocal[vh][6].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 13, 0>(logits[qh],
                                                 Vlocal[vh][6].xy[1], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 14, 0>(logits[qh],
                                                 Vlocal[vh][7].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 15, 0>(logits[qh],
                                                 Vlocal[vh][7].xy[1], acc);
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc);
      }
    }
#endif
  }  // warp in context

  __syncthreads();

  if (warpid == 0) {
    // const float out_scale = (fp8_out_scale_ptr != nullptr) ?
    // __fdividef(1.0f,(*fp8_out_scale_ptr)) : 1.0f;
    const float out_scale =
        (fp8_out_scale_ptr != nullptr) ? 1.0f / (*fp8_out_scale_ptr) : 1.0f;
    _B16x4 vout[QHLOOP][VHELOOP];
    // iterate across heads
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  // iterate over each v head elem (within head_size)
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout[qh][vh] = {0};
  #pragma unroll
        for (int w = 0; w < NWARPS; w++) {
          vout[qh][vh] =
              addx4<scalar_t>(vout[qh][vh], vout_shared[qh][vh][laneid][w]);
        }
      }
    }

    if (context_len > partition_size) {
      scalar_t* out_ptr = out +
                          seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                          partition_idx * HEAD_SIZE;
      const int out_num_partitions = max_num_partitions;
      bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
  #pragma unroll
      for (int qh = 0; qh < QHLOOP; qh++) {
  #pragma unroll
        for (int vh = 0; vh < VHELOOP; vh++) {
          const int head_size_elem = vh * WARP_SIZE + laneid;
  #pragma unroll
          for (int i = 0; i < 4; i++) {
            const int head_idx = 4 * qh + i;
            if (head_idx < GQA_RATIO) {
              out_ptr_b16[(wg_start_head_idx + head_idx) * out_num_partitions *
                              HEAD_SIZE +
                          head_size_elem] = vout[qh][vh][i];
            }
          }
        }
      }
    }  // context_len > partition_size
    else {
      bit8_t* final_out_ptr_b8;
      bit16_t* final_out_ptr_b16;
      if constexpr (std::is_same<OUTT, bit8_t>::value) {
        final_out_ptr_b8 = final_out + seq_idx * num_heads * HEAD_SIZE;
      } else {
        OUTT* out_ptr = final_out + seq_idx * num_heads * HEAD_SIZE;
        final_out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
      }
  #pragma unroll
      for (int qh = 0; qh < QHLOOP; qh++) {
  #pragma unroll
        for (int vh = 0; vh < VHELOOP; vh++) {
          const int head_size_elem = vh * WARP_SIZE + laneid;
  #pragma unroll
          for (int i = 0; i < 4; i++) {
            const int head_idx = 4 * qh + i;
            if (head_idx < GQA_RATIO) {
              if constexpr (std::is_same<OUTT, bit8_t>::value) {
                const float tmpf =
                    out_scale * to_float_b16<scalar_t>(vout[qh][vh][i]);
                const OUTT tmp = hip_fp8(tmpf).data;
                final_out_ptr_b8[(wg_start_head_idx + head_idx) * HEAD_SIZE +
                                 head_size_elem] = tmp;
              } else {
                final_out_ptr_b16[(wg_start_head_idx + head_idx) * HEAD_SIZE +
                                  head_size_elem] = vout[qh][vh][i];
              }
            }
          }
        }
      }
    }
  }  // warpid == 0
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
#if 0 //disable this as mfma16 kernel does not support this optimization yet
  if (num_partitions == 1) {
    // if num_partitions==1, main kernel will write to out directly, no work in
    // reduction kernel
    return;
  }
#endif
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 64;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  // const float out_scale = (fp8_out_scale_ptr != nullptr) ?
  // __fdividef(1.0f,(*fp8_out_scale_ptr)) : 1.0f;
  const float out_scale =
      (fp8_out_scale_ptr != nullptr) ? 1.0f / (*fp8_out_scale_ptr) : 1.0f;
  acc *= inv_global_exp_sum;
  acc *= out_scale;
  OUTT* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  if constexpr (std::is_same<OUTT, bit8_t>::value) {
    out_ptr[threadIdx.x] = hip_fp8(acc).data;
  } else {
    out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale,
    const float* __restrict__ fp8_out_scale_ptr) {
  UNREACHABLE_CODE
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale,
    const float* __restrict__ fp8_out_scale_ptr) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)                                    \
  paged_attention_ll4mi_QKV_mfma16_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,        \
                                   HEAD_SIZE, NTHR, GQA_RATIO>                \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale, v_scale, fp8_out_scale_ptr);

#define LAUNCH_CUSTOM_ATTENTION(GQA_RATIO)                                    \
  paged_attention_ll4mi_QKV_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,        \
                                   HEAD_SIZE, NTHR, GQA_RATIO>                \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale, v_scale, fp8_out_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                          \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE, \
                                      PARTITION_SIZE, NPAR_LOOPS>    \
      <<<reduce_grid, reduce_block, 0, stream>>>(                    \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,        \
          context_lens_ptr, max_num_partitions, fp8_out_scale_ptr);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int max_context_len, const c10::optional<torch::Tensor>& alibi_slopes,
    float k_scale, float v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  // NOTE: fp8_out_scale is optional.
  const float* fp8_out_scale_ptr =
      fp8_out_scale
          ? reinterpret_cast<const float*>(fp8_out_scale.value().data_ptr())
          : nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);
  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions =
      DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256; //PARTITION_SIZE;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (gqa_ratio) {
    case 1:
      //LAUNCH_CUSTOM_ATTENTION(1);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(1);
      break;
    case 2:
      //LAUNCH_CUSTOM_ATTENTION(2);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(2);
      break;
    case 3:
      //LAUNCH_CUSTOM_ATTENTION(3);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(3);
      break;
    case 4:
      //LAUNCH_CUSTOM_ATTENTION(4);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(4);
      break;
    case 5:
      //LAUNCH_CUSTOM_ATTENTION(5);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      //LAUNCH_CUSTOM_ATTENTION(6);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      //LAUNCH_CUSTOM_ATTENTION(7);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      //LAUNCH_CUSTOM_ATTENTION(8);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      //LAUNCH_CUSTOM_ATTENTION(9);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      //LAUNCH_CUSTOM_ATTENTION(10);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      //LAUNCH_CUSTOM_ATTENTION(11);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      //LAUNCH_CUSTOM_ATTENTION(12);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      //LAUNCH_CUSTOM_ATTENTION(13);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      //LAUNCH_CUSTOM_ATTENTION(14);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      //LAUNCH_CUSTOM_ATTENTION(15);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      //LAUNCH_CUSTOM_ATTENTION(16);
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  // reduction kernel is only required if max_context_len > partition size,
  // otherwise main kernel writes directly to final output
  //  note there are cases with graphing where max_context_len is the max
  //  supported by graphing, not the actual max among all the sequences: in that
  //  case reduction kernel will still run but return immediately

  //above optimization is not yet implemented in mfma16 kernel
  //if (max_context_len > PARTITION_SIZE) {
    dim3 reduce_grid(num_heads, num_seqs);
    dim3 reduce_block(head_size);
    const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
    // support upto 8*64*256=128K context length
#if 1
    switch (npar_loops) {
      case 1:
        LAUNCH_CUSTOM_REDUCTION(1);
        break;
      case 2:
        LAUNCH_CUSTOM_REDUCTION(2);
        break;
      case 3:
        LAUNCH_CUSTOM_REDUCTION(3);
        break;
      case 4:
        LAUNCH_CUSTOM_REDUCTION(4);
        break;
      case 5:
        LAUNCH_CUSTOM_REDUCTION(5);
        break;
      case 6:
        LAUNCH_CUSTOM_REDUCTION(6);
        break;
      case 7:
        LAUNCH_CUSTOM_REDUCTION(7);
        break;
      case 8:
        LAUNCH_CUSTOM_REDUCTION(8);
        break;
      default:
        TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
        break;
    }
#endif
  //} //if max_context_len > partition_size
}

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT,      \
                             PSIZE)                                            \
  paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, \
                                  PSIZE>(                                      \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, max_context_len,        \
      alibi_slopes, k_scale, v_scale, fp8_out_scale);

#define CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,     \
                                   OUTT)                                      \
  switch (partition_size) {                                                   \
    case 256:                                                                 \
      CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 256); \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false, "Unsupported partition size: ", partition_size);     \
      break;                                                                  \
  }
/*
*/
#if defined(__HIPCC__) && defined(__gfx90a__)
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)   \
    if (fp8_out_scale) {                                                    \
      TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");           \
    } else {                                                                \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
#else
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)   \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T);
/*
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)   \
    if (fp8_out_scale) {                                                    \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,     \
                                 uint8_t);                                  \
    } else {                                                                \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
    */
#endif
#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)     \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE);  \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }
/*
    case 32:                                                      \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE);  \
      break;                                                      \
*/
#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)         \
  switch (head_size) {                                          \
    case 128:                                                   \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128);          \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "Unsupported head size: ", head_size); \
      break;                                                    \
  }
/*
    case 64:                                                    \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64);           \
      break;                                                    \
*/
void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens,  // [num_seqs]
    int64_t block_size, int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale, int64_t partition_size) {
  const int head_size = query.size(2);
  if (kv_cache_dtype == "auto") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, __hip_bfloat16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP