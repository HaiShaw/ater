from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_activation"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/pybind/activation_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/activation_kernels.cu",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def silu_and_mul(out: Tensor, input: Tensor): ...


@compile_ops(**compile_ops_)
def gelu_and_mul(out: Tensor, input: Tensor): ...


@compile_ops(**compile_ops_)
def gelu_tanh_and_mul(out: Tensor, input: Tensor): ...
