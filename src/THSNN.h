#pragma once

#include "torch/torch.h"

struct NNModuleWrapper
{
    std::shared_ptr<torch::nn::Module> module;

    NNModuleWrapper(std::shared_ptr<torch::nn::Module> m) : module(m) {}
};

struct NNOptimizerWrapper
{
    std::shared_ptr<torch::optim::Optimizer> optimizer;

    NNOptimizerWrapper(std::shared_ptr<torch::optim::Optimizer> o) : optimizer(o) {}
};
