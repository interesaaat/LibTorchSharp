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

//struct no_grad_wrapper
//{
//    std::shared_ptr<torch::NoGradGuard> no_grad;
//
//    no_grad_wrapper(torch::NoGradGuard n) : no_grad(std::make_shared<torch::NoGradGuard>(n)) {}
//};

struct tensor_wrapper2
{
    tensor_wrapper* ptr;
};

// Return the internal tensor implementation
EXPORT_API(THTensor *) GetTHTensor(const tensor_wrapper * tensor)
{
    return tensor->tensor.unsafeGetTensorImpl();
}

// Return the internal tensor implementation
EXPORT_API(void *) Tensor_data(const tensor_wrapper * tensor)
{
    return tensor->tensor.data_ptr();
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
    return makeResultString(module_wrappers->module->name());
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

EXPORT_API(void) Zero_grad_functional(const nn_module_wrapper * mwrapper)
{
    mwrapper->module->as<torch::nn::Functional>()->zero_grad();
}

EXPORT_API(void) Param_functional(const nn_module_wrapper * mwrapper, tensor_wrapper2* (*allocator)(size_t length))
{
    auto parameters = mwrapper->module->as<torch::nn::Functional>()->parameters();
    tensor_wrapper2 *result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i].ptr = new tensor_wrapper(parameters[i]);
    }
}

EXPORT_API(void) Param_linear(const nn_module_wrapper * mwrapper, tensor_wrapper2* (*allocator)(size_t length))
{
    auto parameters = mwrapper->module->as<torch::nn::Linear>()->parameters();
    tensor_wrapper2 *result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i].ptr = new tensor_wrapper(parameters[i]);
    }
}

EXPORT_API(tensor_wrapper *) Grad(const tensor_wrapper * wrapper)
{
    return new tensor_wrapper(wrapper->tensor.grad());
}

EXPORT_API(tensor_wrapper *) Sub_(const tensor_wrapper * swrapper, const tensor_wrapper * twrapper)
{
    torch::NoGradGuard no_grad;
   /* std::ofstream out;
    out.open("log");
    try
    {*/
        at::Tensor src = swrapper->tensor;
        auto result = new tensor_wrapper(src.sub_(twrapper->tensor));
        //out.close();
        return result;
   /* }
    catch (std::exception& e)
    {
        out << e.what();
        out.close();
    }*/
}

EXPORT_API(tensor_wrapper *) Mul(const tensor_wrapper * src, const float scalar)
{
    torch::NoGradGuard no_grad;
    return new tensor_wrapper(src->tensor.mul(scalar));
}

// Not working
//EXPORT_API(no_grad_wrapper *) No_grad()
//{
//    torch::NoGradGuard no_grad;
//
//    return new no_grad_wrapper(no_grad);
//}

EXPORT_API(tensor_wrapper *) Forward_linear(const nn_module_wrapper * mwrapper, const tensor_wrapper * twrapper)
{
    at::Tensor tensor = mwrapper->module->as<torch::nn::Linear>()->forward(twrapper->tensor);

    return new tensor_wrapper(tensor);
}

EXPORT_API(void) Zero_grad_linear(const nn_module_wrapper * mwrapper)
{
    mwrapper->module->as<torch::nn::Linear>()->zero_grad();
}

EXPORT_API(tensor_wrapper *) Loss_mse(tensor_wrapper * src, tensor_wrapper * trg, int64_t reduction)
{
    return new tensor_wrapper(torch::mse_loss(src->tensor, trg->tensor, reduction));
}

EXPORT_API(void) Backward(tensor_wrapper * t)
{
    t->tensor.backward();
}

const char * makeResultString(string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}