#pragma once
#include <torch/extension.h>

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input);