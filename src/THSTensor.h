#pragma once

#include <aten/core/Tensor.h>

struct tensor_wrapper
{
    at::Tensor tensor;

    tensor_wrapper(at::Tensor t) : tensor(t) {}
};
