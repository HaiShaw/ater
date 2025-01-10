# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: shuffle.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-11 13:38:14
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-11 15:11:04
# @Description: This is description.

import torch


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
