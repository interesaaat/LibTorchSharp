#include "THSNN.h"

void THSNN_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

NNModuleWrapper * THSNN_reluModule()
{
    auto relu = torch::nn::Functional(torch::relu);
    return new NNModuleWrapper(relu.ptr());
}

NNModuleWrapper * THSNN_linearModule(const int inputSize, const int outputSize)
{
    auto linear = torch::nn::Linear(inputSize, outputSize);

    return new NNModuleWrapper(linear.ptr());
}

NNModuleWrapper * THSNN_conv2dModule(
    const int64_t inputChannel, 
    const int64_t outputChannel, 
    const size_t kernelSize)
{
    auto options = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize);
    auto conv = torch::nn::Conv2d(options);

    return new NNModuleWrapper(conv.ptr());
}

long THSNN_getNumberOfChildren(const NNModuleWrapper * mwrapper)
{
    return mwrapper->module->children().size();
}

const char * THSNN_getChildModuleName(const NNModuleWrapper * mwrapper, const int index)
{
    return makeSharableString(mwrapper->module->children()[index]->name());
}

const char * THSNN_getModuleName(const NNModuleWrapper * mwrapper)
{
    return makeSharableString(mwrapper->module->name());
}

TensorWrapper * THSNN_reluApply(const TensorWrapper * tensor)
{
    at::Tensor result = torch::relu(tensor->tensor);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_maxPool2DApply(const TensorWrapper * tensor, const int64_t kernelSize)
{
    at::Tensor result = torch::max_pool2d(tensor->tensor, kernelSize);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_logSoftMaxApply(const TensorWrapper * tensor, const int64_t dimension)
{
    at::Tensor result = torch::log_softmax(tensor->tensor, dimension);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_featureDropoutApply(const TensorWrapper * tensor)
{
    at::Tensor result = torch::nn::FeatureDropout()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_dropoutModuleApply(
    const TensorWrapper * tensor, 
    const double probability, 
    const bool isTraining)
{
    at::Tensor result = torch::dropout(tensor->tensor, probability, isTraining);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_linearModuleApply(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Linear>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

TensorWrapper * THSNN_conv2DModuleApply(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor)
{
    at::Tensor result = mwrapper->module->as<torch::nn::Conv2d>()->forward(tensor->tensor);

    return new TensorWrapper(result);
}

void THSNN_moduleZeroGrad(const NNModuleWrapper * mwrapper)
{
    mwrapper->module->zero_grad();
}

void THSNN_optimizerZeroGrad(const NNOptimizerWrapper * owrapper)
{
    owrapper->optimizer->zero_grad();
}

void THSNN_getParameters(
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

TensorWrapper * THSNN_lossMSE(const TensorWrapper * srcwrapper, const TensorWrapper * trgwrapper, const int64_t reduction)
{
    return new TensorWrapper(torch::mse_loss(srcwrapper->tensor, trgwrapper->tensor, reduction));
}

TensorWrapper * THSNN_lossNLL(const TensorWrapper * srcwrapper, const TensorWrapper * trgwrapper, const int64_t reduction)
{
    return new TensorWrapper(torch::nll_loss(srcwrapper->tensor, trgwrapper->tensor, {} /*weights */, reduction));
}

NNOptimizerWrapper * THSNN_optimizerAdam(const TensorWrapper** parameters, const int len, const double learnig_rate)
{
    std::vector<at::Tensor> params;

    for (int i = 0; i < len; i++)
    {
        params.push_back(parameters[i]->tensor);
    }

    return new NNOptimizerWrapper(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, learnig_rate)));
}

NNOptimizerWrapper * THSNN_optimizerSGD(const TensorWrapper** parameters, const int len, const double learnig_rate, const double momentum)
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

void THSNN_optimizerStep(const NNOptimizerWrapper * owrapper)
{
    owrapper->optimizer->step();
}

void THSNN_optimizerDispose(const NNOptimizerWrapper * owrapper)
{
    delete owrapper;
}

void THSNN_moduleDispose(const NNModuleWrapper * mwrapper)
{
    delete mwrapper;
}

