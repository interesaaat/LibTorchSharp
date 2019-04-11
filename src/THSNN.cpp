#include "THSNN.h"

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
    const Tensor src, 
    const Tensor trg, 
    const Tensor wgt, 
    const int64_t reduction)
{
    return wgt == NULL ?
        new torch::Tensor(torch::binary_cross_entropy(*src, *trg, {}, reduction)) :
        new torch::Tensor(torch::binary_cross_entropy(*src, *trg, *wgt, reduction));
}

Tensor THSNN_lossMSE(const Tensor src, const Tensor trg, const int64_t reduction)
{
    return new torch::Tensor(torch::mse_loss(*src, *trg, reduction));
}

Tensor THSNN_lossNLL(
    const Tensor src, 
    const Tensor trg, 
    const Tensor wgt, 
    const int64_t reduction)
{
    return wgt == NULL ?
        new torch::Tensor(torch::nll_loss(*src, *trg, {}, reduction)) :
        new torch::Tensor(torch::nll_loss(*src, *trg, *wgt, reduction));
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

