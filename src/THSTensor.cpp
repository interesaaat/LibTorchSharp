#pragma once

#include "THSTensor.h"

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/torch.h"
#include "torch/script.h"

#include <fstream>
#include <iostream>
#include <exception>

struct module_wrapper
{
    std::shared_ptr<torch::jit::script::Module> module;

    module_wrapper(std::shared_ptr<torch::jit::script::Module> m) : module(m) {}
};

struct tensor_wrapper
{
    at::Tensor tensor;

    tensor_wrapper(at::Tensor t) : tensor(t) {}
};

// Create a variable containing a tensor composed of ones.
EXPORT_API(tensor_wrapper *) Ones(const int lenght)
{
    const auto options = torch::autograd::TensorOptions().dtype(at::ScalarType::Float);

    int64_t data[] = { 1, 3, 224, 224 };

    auto tensor = torch::ones(at::IntList(data, lenght), options);
    return new tensor_wrapper(tensor);
}

EXPORT_API(module_wrapper *) Load(const char* filename, void* result)
{
    auto module = torch::jit::load(filename);
    return new module_wrapper(module);
}

EXPORT_API(int) GetNumberOfModules(const module_wrapper * module_wrappers)
{
    return module_wrappers->module->get_modules().size();
}

EXPORT_API(const char*) GetModule(const module_wrapper * module_wrappers, int index)
{
    auto modules = module_wrappers->module->get_modules();
    auto keys = modules.keys();
    auto key = keys[index].c_str();

    size_t size = sizeof(key);
    char* result = new char[size];
    strncpy(result, key, size);
    result[size - 1] = '\0';
    return result;
}

EXPORT_API(THTensor *) Forward(const module_wrapper * module_wrapper, const tensor_wrapper * tensor_wrapper)
{
    std::ofstream out;
    out.open("out");
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(tensor_wrapper->tensor);
    out << "Tensor loaded.\n";
    at::Tensor result;
    try {
        result = module_wrapper->module->forward(inputs).toTensor();
    }
    catch (std::exception e)
    {
        out << "Exception.";
        out << e.what();
        out.close();
    }
    out << "Forward.\n";
    out << result.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    out.close();
    return result.unsafeReleaseTensorImpl();
}