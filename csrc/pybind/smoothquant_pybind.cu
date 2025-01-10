#include "smoothquant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);
}
