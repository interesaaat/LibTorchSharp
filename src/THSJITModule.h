#pragma once

#include "torch/script.h"

// Struct used to share TorchScript modules
struct jit_module_wrapper
{
    std::shared_ptr<torch::jit::script::Module> module;

    jit_module_wrapper(std::shared_ptr<torch::jit::script::Module> m) : module(m) {}
};