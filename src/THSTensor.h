#pragma once

#include <aten/core/Tensor.h>

struct TensorWrapper
{
    at::Tensor tensor;

    TensorWrapper(at::Tensor t) : tensor(t) {}
};

struct TensorPointerWrapper
{
    TensorWrapper* ptr;
};
