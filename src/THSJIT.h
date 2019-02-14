#pragma once

#include "torch/script.h"

// Struct used to share TorchScript modules
struct JITModuleWrapper
{
    std::shared_ptr<torch::jit::script::Module> module;

    JITModuleWrapper(std::shared_ptr<torch::jit::script::Module> m) : module(m) {}
};