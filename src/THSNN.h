#pragma once

#include "torch/torch.h"

#include "THSTensor.h"

// Types for inter-op.

typedef std::shared_ptr<torch::nn::Module> * NNModule;
typedef std::shared_ptr<torch::optim::Optimizer> * Optimizer;

// API.

// Returns a ReLu layer.
THS_API NNModule THSNN_reluModule();

// Returns a linear layer.
THS_API NNModule THSNN_linearModule(const int inputSize, const int outputSize);

// Returns a Conv2d layer.
THS_API NNModule THSNN_conv2dModule(
    const int64_t inputChannel, 
    const int64_t outputChannel, 
    const size_t kernelSize);

// Gets the number of children modules.
THS_API long THSNN_getNumberOfChildren(const NNModule module);

// Returns the module name of the child submodule.
THS_API const char * THSNN_getChildModuleName(const NNModule module, const int index);

// Returns the module name.
THS_API const char * THSNN_getModuleName(const NNModule module);

// Applies a ReLu activation function on the input tensor. 
THS_API Tensor THSNN_reluApply(const Tensor tensor);

// Applies a maxpool 2d on the input tensor. 
THS_API Tensor THSNN_maxPool2DApply(const Tensor tensor, const int64_t kernelSize);

// Applies a log soft max on the input tensor. 
THS_API Tensor THSNN_logSoftMaxApply(const Tensor tensor, const int64_t dimension);

// Applies a log soft max on the input tensor. 
THS_API Tensor THSNN_featureDropoutApply(const Tensor tensor);

// Applies drop out on the input tensor. 
THS_API Tensor THSNN_dropoutModuleApply(
    const Tensor tensor, 
    const double probability, 
    const bool isTraining);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
THS_API Tensor THSNN_linearModuleApply(const NNModule module, const Tensor tensor);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
THS_API Tensor THSNN_conv2DModuleApply(
    const NNModule module,
    const Tensor tensor);

// Zero-ing the grad parameters for the input functional module.
THS_API void THSNN_moduleZeroGrad(const NNModule module);

// Zero-ing the grad parameters for the input optimizer.
THS_API void THSNN_optimizerZeroGrad(const Optimizer optimizer);

// Gets the parameters of the module.
THS_API void THSNN_getParameters(
    const NNModule module,
    Tensor* (*allocator)(size_t length));

// Computes the Binary Cross Entropy (BCE) loss between input and target tensors, using a specified reduction type
// and weights if classes are unbalanced.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss for further details.
THS_API Tensor THSNN_lossBCE(
    const Tensor inputwrapper,
    const Tensor targetwrapper,
    const Tensor weightwrapper,
    const int64_t reduction);

// Computes the Mean squared Error (MSE, squared L2 norm) loss between the input and target tensors, using a specified reduction type.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss for further details.
THS_API Tensor THSNN_lossMSE(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);

// Computes the Negative Log Likelihood (NLL) loss between the input and target tensors, using a specified reduction type
// and weights if classes are unbalanced. It is useful to train a classification problem with C classes.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss for further details.
THS_API Tensor THSNN_lossNLL(
    const Tensor inputwrapper, 
    const Tensor targetwrapper, 
    const Tensor weightwrapper, 
    const int64_t reduction);

// Negative log likelihood loss with Poisson distribution of target.
// See https://pytorch.org/docs/stable/nn.html#poisson-nll-loss for further details.
THS_API Tensor THSNN_lossPoissonNLL(
    const Tensor input,
    const Tensor target,
    const bool logInput,
    const bool full,
    const double eps,
    const int64_t reduction);

// Sets up the Adam optimizer
THS_API Optimizer THSNN_optimizerAdam(const Tensor* parameters, const int len, const double learnig_rate);

// Sets up the SGD optimizer
THS_API Optimizer THSNN_optimizerSGD(const Tensor* parameters, const int len, const double learnig_rate, const double momentum);

// Zero-ing the grad parameters for the input optimizer.
THS_API void THSNN_optimizerStep(const Optimizer optimizer);

// Disposes the optimizer.
THS_API void THSNN_optimizerDispose(const Optimizer optimizer);

// Disposes the module.
THS_API void THSNN_moduleDispose(const NNModule module);
