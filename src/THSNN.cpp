#include "THSNN.h"

#include <math.h>

#include <torch/nn/init.h>

NNModule THSNN_reluModule()
{
    return new std::shared_ptr<torch::nn::Module>(torch::nn::Functional(torch::relu).ptr());
}

NNModule THSNN_linearModule(const int inputSize, const int outputSize)
{
    return new std::shared_ptr<torch::nn::Module>(torch::nn::Linear(inputSize, outputSize).ptr());
}

NNModule THSNN_conv2dModule(
    const int64_t inputChannel, 
    const int64_t outputChannel, 
    const size_t kernelSize)
{
    auto options = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize);
    auto conv = torch::nn::Conv2d(options);

    return new std::shared_ptr<torch::nn::Module>(conv.ptr());
}

long THSNN_getNumberOfChildren(const NNModule module)
{
    return (*module)->children().size();
}

const char * THSNN_getChildModuleName(const NNModule module, const int index)
{
    return makeSharableString((*module)->children()[index]->name());
}

const char * THSNN_getModuleName(const NNModule module)
{
    return makeSharableString((*module)->name());
}

Tensor THSNN_reluApply(const Tensor tensor)
{
    return new torch::Tensor(torch::relu(*tensor));
}

Tensor THSNN_maxPool2DApply(const Tensor tensor, const int64_t kernelSize)
{
    return new torch::Tensor(torch::max_pool2d(*tensor, kernelSize));
}

Tensor THSNN_logSoftMaxApply(const Tensor tensor, const int64_t dimension)
{
    return new torch::Tensor(torch::log_softmax(*tensor, dimension));
}

Tensor THSNN_featureDropoutApply(const Tensor tensor)
{
    return new torch::Tensor(torch::nn::FeatureDropout()->forward(*tensor));
}

Tensor THSNN_dropoutModuleApply(
    const Tensor tensor, 
    const double probability, 
    const bool isTraining)
{
    return new torch::Tensor(torch::dropout(*tensor, probability, isTraining));
}

Tensor THSNN_linearModuleApply(
    const NNModule module,
    const Tensor tensor)
{
    at::Tensor result = (*module)->as<torch::nn::Linear>()->forward(*tensor);

    return new torch::Tensor(result);
}

Tensor THSNN_conv2DModuleApply(
    const NNModule module,
    const Tensor tensor)
{
    at::Tensor result = (*module)->as<torch::nn::Conv2d>()->forward(*tensor);

    return new torch::Tensor(result);
}

void THSNN_moduleZeroGrad(const NNModule module)
{
    (*module)->zero_grad();
}

void THSNN_optimizerZeroGrad(const Optimizer optimizer)
{
    (*optimizer)->zero_grad();
}

void THSNN_getParameters(
    const NNModule module, 
    Tensor* (*allocator)(size_t length))
{

    auto parameters = (*module)->parameters();
    Tensor * result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i] = new torch::Tensor(parameters[i]);
    }
}

Tensor THSNN_lossBCE(
    const Tensor input, 
    const Tensor target, 
    const Tensor weight, 
    const int64_t reduction)
{
    return weight == NULL ?
        new torch::Tensor(torch::binary_cross_entropy(*input, *target, {}, reduction)) :
        new torch::Tensor(torch::binary_cross_entropy(*input, *target, *weight, reduction));
}

Tensor THSNN_lossMSE(const Tensor input, const Tensor target, const int64_t reduction)
{
    return new torch::Tensor(torch::mse_loss(*input, *target, reduction));
}

Tensor THSNN_lossNLL(
    const Tensor input, 
    const Tensor target, 
    const Tensor weight, 
    const int64_t reduction)
{
    return weight == NULL ?
        new torch::Tensor(torch::nll_loss(*input, *target, {}, reduction)) :
        new torch::Tensor(torch::nll_loss(*input, *target, *weight, reduction));
}

Tensor THSNN_lossPoissonNLL(
    const Tensor input,
    const Tensor target,
    const bool logInput,
    const bool full,
    const double eps,
    const int64_t reduction)
{
    torch::Tensor loss;

    if (logInput)
    {
        loss = torch::exp(*input) - (*target) * (*input);
    }
    else
    {
        loss = (*input) - (*target) * torch::log(*input + eps);
    }
    
    if (full)
    {
        auto mask = (*target) > 1;
        loss[mask] += ((*target) * torch::log(*target) - (*target) + 0.5 * torch::log(2 * M_PI * (*target)))[mask];
    }

    if (reduction == Reduction::None)
    {
        return new torch::Tensor(loss);
    }
    else if (reduction == Reduction::Mean)
    {
        return new torch::Tensor(torch::mean(loss));
    }
    else if (reduction == Reduction::Sum)
    {
        return new torch::Tensor(torch::sum(loss));
    }
    
    return NULL;
}

Optimizer THSNN_optimizerAdam(const Tensor* parameters, const int lenght, const double learnig_rate)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, lenght);

    auto optimizer = torch::optim::Adam(params, learnig_rate);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, learnig_rate)));
}

Optimizer THSNN_optimizerSGD(const Tensor* parameters, const int lenght, const double learnig_rate, const double momentum)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, lenght);
    auto options = torch::optim::SGDOptions(learnig_rate)
        .momentum(momentum);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, options)));
}

void THSNN_optimizerStep(const Optimizer optimizer)
{
    (*optimizer)->step();
}

void THSNN_optimizerDispose(const Optimizer optimizer)
{
    delete optimizer;
}

void THSNN_moduleDispose(const NNModule module)
{
    delete module;
}

