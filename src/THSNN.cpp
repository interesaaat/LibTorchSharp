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

// Return a Conv2d layer.
EXPORT_API(NNModuleWrapper *) NN_conv2dModule(int64_t inputChannel, int64_t outputChannel, size_t kernelSize)
{
    auto options = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize);
    auto conv = torch::nn::Conv2d(options);

    return new NNModuleWrapper(conv.ptr());
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

// Apply a ReLu activation function on the input tensor. 
EXPORT_API(TensorWrapper *) NN_ReluModule_Forward(const TensorWrapper * tensor)
{
    at::Tensor result = torch::relu(tensor->tensor);

    return new TensorWrapper(result);
}

// Apply a maxpool 2d on the input tensor. 
EXPORT_API(TensorWrapper *) NN_MaxPool2DModule_Forward(const TensorWrapper * tensor, const int64_t kernelSize)
{
    at::Tensor result = torch::max_pool2d(tensor->tensor, kernelSize);

    return new TensorWrapper(result);
}

// Apply a log soft max on the input tensor. 
EXPORT_API(TensorWrapper *) NN_LogSoftMaxModule_Forward(const TensorWrapper * tensor, const int64_t dimension)
{
    at::Tensor result = torch::log_softmax(tensor->tensor, dimension);

    return new TensorWrapper(result);
}
}

// Apply a log soft max on the input tensor. 
EXPORT_API(TensorWrapper *) NN_FeatureDropout_Forward(const TensorWrapper * tensor)
{
    at::Tensor result = torch::nn::FeatureDropout()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

// Apply drop out on the input tensor. 
EXPORT_API(TensorWrapper *) NN_DropoutModule_Forward(const TensorWrapper * tensor, double probability, bool isTraining)
{
    at::Tensor result = torch::dropout(tensor->tensor, probability, isTraining);

    return new TensorWrapper(result);
}

// Trigger a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
EXPORT_API(TensorWrapper *) NN_linearModule_Forward(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Linear>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

// Trigger a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
EXPORT_API(TensorWrapper *) NN_conv2DModule_Forward(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Conv2d>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

// Zero-ing the grad parameters for the input functional module.
EXPORT_API(void) NN_Module_ZeroGrad(const NNModuleWrapper * mwrapper)
{
    mwrapper->module->zero_grad();
}

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) NN_Optimizer_ZeroGrad(const NNOptimizerWrapper * owrapper)
{
    owrapper->optimizer->zero_grad();
}

// Get the parameters of the module.
EXPORT_API(void) NN_GetParameters(
    const NNModuleWrapper * mwrapper, 
    TensorWrapper** (*allocator)(size_t length))
{
    auto parameters = mwrapper->module->parameters();
    TensorWrapper **result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i] = new TensorWrapper(parameters[i]);
    }
}

// Compute the MSE loss between the input and target tensors, using a specified reduction type.
EXPORT_API(TensorWrapper *) NN_LossMSE(TensorWrapper * srcwrapper, TensorWrapper * trgwrapper, int64_t reduction)
{
    return new TensorWrapper(torch::mse_loss(srcwrapper->tensor, trgwrapper->tensor, reduction));
}

// Compute the NLL loss between the input and target tensors, using a specified reduction type.
EXPORT_API(TensorWrapper *) NN_LossNLL(TensorWrapper * srcwrapper, TensorWrapper * trgwrapper)
{
    return new TensorWrapper(torch::nll_loss(srcwrapper->tensor, trgwrapper->tensor));
}

// Set up the Adam optimizer
EXPORT_API(NNOptimizerWrapper *) NN_OptimizerAdam(TensorWrapper** parameters, int len, double learnig_rate)
{
    std::vector<at::Tensor> params;

    for (int i = 0; i < len; i++)
    {
        params.push_back(parameters[i]->tensor);
    }

    return new NNOptimizerWrapper(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, learnig_rate)));
}

// Set up the SGD optimizer
EXPORT_API(NNOptimizerWrapper *) NN_OptimizerSGD(TensorWrapper** parameters, int len, double learnig_rate, double momentum)
{
    std::vector<at::Tensor> params;
    auto options = torch::optim::SGDOptions(learnig_rate)
        .momentum(momentum);

    for (int i = 0; i < len; i++)
    {
        params.push_back(parameters[i]->tensor);
    }

    return new NNOptimizerWrapper(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, options)));
}

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) NN_Optimizer_Step(const NNOptimizerWrapper * owrapper)
{
    owrapper->optimizer->step();
}

