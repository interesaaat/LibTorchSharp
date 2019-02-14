#include "THSNN.h"

#include "THSTensor.h"
#include "stdafx.h"
#include "utils.h"

// Return a ReLu layer.
EXPORT_API(NNModuleWrapper *) NN_reluModule()
{
    auto relu = torch::nn::Functional(torch::relu);

    return new NNModuleWrapper(relu.ptr());
}

// Return a linear layer.
EXPORT_API(NNModuleWrapper *) NN_linearModule(const int inputSize, const int outputSize)
{
    auto linear = torch::nn::Linear(inputSize, outputSize);

    return new NNModuleWrapper(linear.ptr());
}

// Get the number of children modules.
EXPORT_API(long) NN_GetNumberOfChildren(const NNModuleWrapper * mwrapper)
{
    return mwrapper->module->children().size();
}

// Return the module name.
EXPORT_API(const char *) NN_GetModuleName(const NNModuleWrapper * mwrapper)
{
    return makeSharableString(mwrapper->module->name());
}

// Functional
// Trigger a forward pass over an input functional module (e.g., activation functions) using the input tensor. 
EXPORT_API(TensorWrapper *) NN_functionalModule_Forward(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Functional>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

// Zero-ing the grad parameters for the input functional module.
EXPORT_API(void) NN_functionalModule_ZeroGrad(const NNModuleWrapper * mwrapper)
{
    mwrapper->module->as<torch::nn::Functional>()->zero_grad();
}

// Get the parameters of the input functional module.
EXPORT_API(void) NN_functionalModule_GetParameters(
    const NNModuleWrapper * mwrapper, 
    TensorPointerWrapper* (*allocator)(size_t length))
{
    auto parameters = mwrapper->module->as<torch::nn::Functional>()->parameters();
    TensorPointerWrapper *result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i].ptr = new TensorWrapper(parameters[i]);
    }
}

// Linear
// Trigger a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
EXPORT_API(TensorWrapper *) NN_linearModule_Forward(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Linear>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

// Zero-ing the grad parameters for the input functional module.
EXPORT_API(void) NN_linearModule_ZeroGrad(const NNModuleWrapper * mwrapper)
{
    mwrapper->module->as<torch::nn::Linear>()->zero_grad();
}

// Get the parameters of the input functional module.
EXPORT_API(void) NN_linearModule_GetParameters(
    const NNModuleWrapper * mwrapper, 
    TensorPointerWrapper* (*allocator)(size_t length))
{
    auto parameters = mwrapper->module->as<torch::nn::Linear>()->parameters();
    TensorPointerWrapper *result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i].ptr = new TensorWrapper(parameters[i]);
    }
}

// Compute the MSE loss between the input and target tensors, using a spceificed reduction type.
EXPORT_API(TensorWrapper *) NN_LossMSE(TensorWrapper * srcwrapper, TensorWrapper * trgwrapper, int64_t reduction)
{
    return new TensorWrapper(torch::mse_loss(srcwrapper->tensor, trgwrapper->tensor, reduction));
}