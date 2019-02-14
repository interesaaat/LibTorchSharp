#pragma once

#include "torch/torch.h"

struct NNModuleWrapper
{
    std::shared_ptr<torch::nn::Module> module;

    NNModuleWrapper(std::shared_ptr<torch::nn::Module> m) : module(m) {}
};
