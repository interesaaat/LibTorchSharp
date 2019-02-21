#pragma once

#include <torch/torch.h>

struct TensorMNISTDataWrapper
{
    torch::data::datasets::MNIST data;

    TensorMNISTDataWrapper(torch::data::datasets::MNIST m) : data(m) {}
};
