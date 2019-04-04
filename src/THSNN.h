#pragma once

#include "torch/torch.h"

#include "THSTensor.h"

// Inter-op structs.

// Wrapper struct used to share NN modules.
struct NNModuleWrapper
{
    std::shared_ptr<torch::nn::Module> module;

    NNModuleWrapper(std::shared_ptr<torch::nn::Module> m) : module(m) {}
};

// Wrapper struct used to share optimizers.
struct NNOptimizerWrapper
{
    std::shared_ptr<torch::optim::Optimizer> optimizer;

    NNOptimizerWrapper(std::shared_ptr<torch::optim::Optimizer> o) : optimizer(o) {}
};

// API.

// Returns a ReLu layer.
THS_API NNModuleWrapper * THSNN_reluModule();

// Returns a linear layer.
THS_API NNModuleWrapper * THSNN_linearModule(const int inputSize, const int outputSize);

// Returns a Conv2d layer.
THS_API NNModuleWrapper * THSNN_conv2dModule(
    const int64_t inputChannel, 
    const int64_t outputChannel, 
    const size_t kernelSize);

// Gets the number of children modules.
THS_API long THSNN_getNumberOfChildren(const NNModuleWrapper * mwrapper);

// Returns the module name of the child submodule.
THS_API const char * THSNN_getChildModuleName(const NNModuleWrapper * mwrapper, const int index);

// Returns the module name.
THS_API const char * THSNN_getModuleName(const NNModuleWrapper * mwrapper);

// Applies a ReLu activation function on the input tensor. 
THS_API TensorWrapper * THSNN_reluApply(const TensorWrapper * tensor);

// Applies a maxpool 2d on the input tensor. 
THS_API TensorWrapper * THSNN_maxPool2DApply(const TensorWrapper * tensor, const int64_t kernelSize);

// Applies a log soft max on the input tensor. 
THS_API TensorWrapper * THSNN_logSoftMaxApply(const TensorWrapper * tensor, const int64_t dimension);

// Applies a log soft max on the input tensor. 
THS_API TensorWrapper * THSNN_featureDropoutApply(const TensorWrapper * tensor);

// Applies drop out on the input tensor. 
THS_API TensorWrapper * THSNN_dropoutModuleApply(
    const TensorWrapper * tensor, 
    const double probability, 
    const bool isTraining);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
THS_API TensorWrapper * THSNN_linearModuleApply(const NNModuleWrapper * mwrapper, const TensorWrapper * tensor);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
THS_API TensorWrapper * THSNN_conv2DModuleApply(
    const NNModuleWrapper * mwrapper,
    const TensorWrapper * tensor);

// Zero-ing the grad parameters for the input functional module.
THS_API void THSNN_moduleZeroGrad(const NNModuleWrapper * mwrapper);

// Zero-ing the grad parameters for the input optimizer.
THS_API void THSNN_optimizerZeroGrad(const NNOptimizerWrapper * owrapper);

// Gets the parameters of the module.
THS_API void THSNN_getParameters(
    const NNModuleWrapper * mwrapper,
    TensorWrapper** (*allocator)(size_t length));

// Computes the Binary Cross Entropy (BCE) loss between input and target tensors, using a specified reduction type
// and weights if classes are unbalanced.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss for further details.
THS_API TensorWrapper * THSNN_lossBCE(
    const TensorWrapper * srcwrapper,
    const TensorWrapper * trgwrapper,
    const TensorWrapper * wgtwrapper,
    const int64_t reduction);

// Computes the Mean squared Error (MSE, squared L2 norm) loss between the input and target tensors, using a specified reduction type.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss for further details.
THS_API TensorWrapper * THSNN_lossMSE(const TensorWrapper * srcwrapper, const TensorWrapper * trgwrapper, const int64_t reduction);

// Computes the Negative Log Likelihood (NLL) loss between the input and target tensors, using a specified reduction type
// and weights if classes are unbalanced. It is useful to train a classification problem with C classes.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss for further details.
THS_API TensorWrapper * THSNN_lossNLL(
    const TensorWrapper * srcwrapper, 
    const TensorWrapper * trgwrapper, 
    const TensorWrapper * wgtwrapper, 
    const int64_t reduction);

// Sets up the Adam optimizer
THS_API NNOptimizerWrapper * THSNN_optimizerAdam(const TensorWrapper** parameters, const int len, const double learnig_rate);

// Sets up the SGD optimizer
THS_API NNOptimizerWrapper * THSNN_optimizerSGD(const TensorWrapper** parameters, const int len, const double learnig_rate, const double momentum);

// Zero-ing the grad parameters for the input optimizer.
THS_API void THSNN_optimizerStep(const NNOptimizerWrapper * owrapper);

// Disposes the optimizer.
THS_API void THSNN_optimizerDispose(const NNOptimizerWrapper * owrapper);

// Disposes the module.
THS_API void THSNN_moduleDispose(const NNModuleWrapper * mwrapper);
