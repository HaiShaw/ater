import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.topk import select_experts

from sglang.srt.utils import permute_weight
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
# from sglang.srt.layers.moe.fused_moe_ck.fused_moe import fused_experts_ck
import ater
from ater.fused_moe import fused_experts_ck

padding_size = 128 if bool(int(os.getenv("MOE_PADDING", "0"))) else 0


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=''):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        print(f'{msg}[checkAllclose {atol=} {rtol=} passed~]')
    else:
        percent = (a[mask]).numel()/a.numel()
        delta = (a-b)[mask]
        if percent > 0.01:
            print(f'''{msg}[checkAllclose {atol=} {rtol=} failed!]
        a:  {a.shape}
            {a[mask]}
        b:  {b.shape}
            {b[mask]}
    dtlta:
            {delta}''')
        else:
            print(
                f'''{msg}[checkAllclose {atol=} {rtol=} waring!] a and b results are not all close''')
        print(
            f'-->max delta:{delta.max()}, delta details: {percent:.1%} ({(a[mask]).numel()} of {a.numel()}) elements')

def shuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK*2
    K = 16//x.element_size()
    BN = IN
    assert (x.shape[-2] %
            BN == 0), f'{x.shape[-2]} % {BN} == {x.shape[-2] % BN }'
    assert (x.shape[-1] %
            BK == 0), f'{x.shape[-1]} % {BK} == {x.shape[-1] % BK }'

    x_ = x
    x_ = x_.view(-1,
                 x.shape[-2]//BN, BN,
                 x.shape[-1]//BK, BK//K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_


def main(model, tp_size, dtype: str, batches):
    for bs in batches:
        run_test(int(bs), model=model, tp_size=tp_size, dtype_str=dtype)


def run_test(bs, model, tp_size, dtype_str: str):

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)

    top_k = config.num_experts_per_tok
    d_model = config.hidden_size
    model_intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    hidden_states_dtype = config.torch_dtype

    if config.num_experts_per_tok:
        if config.architectures[0] == "Grok1ModelForCausalLM":
            num_total_experts = config.num_experts
        else:
            num_total_experts = config.num_local_experts
    else:
        raise ValueError(f"Unsupported Mixtral model {model}")


    if dtype_str == "bfloat16":
        dtype = torch.bfloat16


    shard_intermediate_size = model_intermediate_size // tp_size

    x = torch.randn(
        (bs, d_model),
        device="cuda:0",
        dtype=dtype,
    )

    w1 = torch.randn(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=x.device,
        dtype=x.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=x.device,
        dtype=x.dtype,
    )

    #w1b = permute_weight(w1)
    #w2b = permute_weight(w2)

    w1b = shuffle_weight(w1)
    w2b = shuffle_weight(w2)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    #if dtype_str == "float8":
    #    w1 = w1.to(torch.float8_e4m3fnuz)
    #    w2 = w2.to(torch.float8_e4m3fnuz)
    #    w1_scale = torch.ones(
    #        num_total_experts, device=hidden_states.device, dtype=torch.float32
    #    )
    #    w2_scale = torch.ones(
    #        num_total_experts, device=hidden_states.device, dtype=torch.float32
    #    )
    #    a1_scale = torch.ones(1, device=hidden_states.device, dtype=torch.float32)
    #    a2_scale = torch.ones(1, device=hidden_states.device, dtype=torch.float32)


    router_logits=torch.randn(
        (bs, num_total_experts),
        device=x.device,
        dtype=torch.float32
    )

    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=False,
        top_k=top_k,
        renormalize=False,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
    )

    out = fused_experts_ck(
                hidden_states=x,
                w1=w1b,
                w2=w2b,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
          )

    ref_out = fused_experts(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
            )

    print(f"out: {out} ref_out:{ref_out}", flush=True)

    checkAllclose(out, ref_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe",
        description="Benchmark and tune the fused_moe kernel",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float8", "float16", "bfloat16"],
        help="Data type used for fused_moe kernel computations",
    )
    parser.add_argument("--model", type=str, default="hpcai-tech/grok-1")

    parser.add_argument("--tp-size", type=int, default=2, help="Tensor paralleli size")
    parser.add_argument("-b", "--batches", type=str)

    args = parser.parse_args()

    batches = args.batches.split(",")

    sys.exit(main(args.model, args.tp_size, args.dtype, batches))
