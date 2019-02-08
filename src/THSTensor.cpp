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

struct jit_module_wrapper
{
    std::shared_ptr<torch::jit::script::Module> module;

    jit_module_wrapper(std::shared_ptr<torch::jit::script::Module> m) : module(m) {}
};

struct nn_module_wrapper
{
    std::shared_ptr<torch::nn::Module> module;

    nn_module_wrapper(std::shared_ptr<torch::nn::Module> m) : module(m) {}
};

struct tensor_wrapper
{
    at::Tensor tensor;

    tensor_wrapper(at::Tensor t) : tensor(t) {}
};

// Return the internal tensor implementation
EXPORT_API(THTensor *) GetTHTensor(const tensor_wrapper * tensor)
{
    return tensor->tensor.unsafeGetTensorImpl();
}

// Return the internal tensor implementation
EXPORT_API(void *) Tensor_data(const tensor_wrapper * tensor)
{
    return tensor->tensor.data_ptr();;
}

// Create a variable containing a tensor composed of ones.
EXPORT_API(tensor_wrapper *) Tensor_ones(const int64_t * sizes, const int lenght, const int8_t scalar_type, const char * device, const bool requires_grad)
{
    auto options = torch::autograd::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);
    at::Tensor tensor = torch::ones(at::IntList(sizes, lenght), options);

    return new tensor_wrapper(tensor);
}

EXPORT_API(tensor_wrapper *) Tensor_randn(const int64_t * sizes, const int lenght, const int8_t scalar_type, const char * device, const bool requires_grad)
{
    auto options = torch::autograd::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);
    at::Tensor tensor = torch::randn(at::IntList(sizes, lenght), options);

    return new tensor_wrapper(tensor);
}

EXPORT_API(const char*) Tensor_device(const tensor_wrapper * tensor)
{
    auto device = tensor->tensor.device();
    auto device_type = DeviceTypeName(device.type());
    auto device_index = std::to_string(device.index());
    string str_device = device_type + ":" + device_index;

    return makeResultString(str_device);
}


EXPORT_API(nn_module_wrapper *) Module_relu()
{
    auto relu = torch::nn::Functional(torch::relu);

    return new nn_module_wrapper(relu.ptr());
}

EXPORT_API(nn_module_wrapper *) Module_linear(const int input, const int output)
{
    auto linear = torch::nn::Linear(input, output);

    return new nn_module_wrapper(linear.ptr());
}

EXPORT_API(jit_module_wrapper *) Module_load(const char* filename)
{
    auto module = torch::jit::load(filename);

    return new jit_module_wrapper(module);
}

EXPORT_API(long) Get_number_of_modules(const jit_module_wrapper * module_wrappers)
{
    return module_wrappers->module->get_modules().size();
}

EXPORT_API(long) Get_number_of_children(const nn_module_wrapper * module_wrappers)
{
    return module_wrappers->module->children().size();
}

EXPORT_API(const char *) Module_name(const nn_module_wrapper * module_wrappers)
{
    //std::ofstream out;
    //out.open("c:/name");
    auto name = makeResultString(module_wrappers->module->name());
    /*out << name;
    out.close();*/
    return name;
}

EXPORT_API(const char*) Module_jit_get(const jit_module_wrapper * module_wrappers, int index)
{
    auto modules = module_wrappers->module->get_modules();
    auto keys = modules.keys();

    return makeResultString(keys[index]);
}

EXPORT_API(const char*) Module_nn_get(const nn_module_wrapper * module_wrappers, int index)
{
    auto modules = module_wrappers->module->named_children();
    auto keys = modules.keys();

    return makeResultString(keys[index]); 
}

EXPORT_API(tensor_wrapper *) Forward_jit(const jit_module_wrapper * mwrapper, const tensor_wrapper * twrapper)
{
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(twrapper->tensor);

    at::Tensor tensor = mwrapper->module->forward(inputs).toTensor();
   
    return new tensor_wrapper(tensor);
}

EXPORT_API(tensor_wrapper *) Forward_functional(const nn_module_wrapper * mwrapper, const tensor_wrapper * twrapper)
{
    at::Tensor tensor = mwrapper->module->as<torch::nn::Functional>()->forward(twrapper->tensor);

    return new tensor_wrapper(tensor);
}

EXPORT_API(tensor_wrapper *) Forward_linear(const nn_module_wrapper * mwrapper, const tensor_wrapper * twrapper)
{
    at::Tensor tensor = mwrapper->module->as<torch::nn::Linear>()->forward(twrapper->tensor);

    return new tensor_wrapper(tensor);
}

EXPORT_API(tensor_wrapper *) MSELoss(tensor_wrapper * src, tensor_wrapper trg, int64_t reduction)
{
    return new tensor_wrapper(torch::mse_loss(src->tensor, trg.tensor, reduction));
}

const char * makeResultString(string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}