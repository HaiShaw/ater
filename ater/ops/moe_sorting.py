import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_moe_sorting"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/py_itfs_ck/moe_sorting_kernels.cu",
        f"{ATER_CSRC_DIR}/pybind/moe_sorting_pybind.cu",
        f"{CK_DIR}/example/ck_tile/13_moe_sorting/",
    ],
    "extra_include": [
        f"{CK_DIR}/example/ck_tile/13_moe_sorting/",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def moe_sorting_fwd(input: Tensor, out: Tensor,
                    x_scale: Tensor, y_scale: Tensor): ...