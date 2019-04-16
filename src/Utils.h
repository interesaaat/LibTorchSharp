#pragma once

#include <string>

#include "TH/THGeneral.h"
#include "torch/torch.h"

extern thread_local char *torch_last_err;

typedef torch::Tensor *Tensor;
typedef torch::Scalar *Scalar;
typedef std::shared_ptr<torch::nn::Module> * NNModule;
typedef std::shared_ptr<torch::optim::Optimizer> * Optimizer;
typedef std::shared_ptr<torch::jit::script::Module> * JITModule;
typedef std::shared_ptr<c10::Type> * JITType;
typedef std::shared_ptr<torch::jit::DynamicType> * JITDynamicType;
typedef std::shared_ptr<torch::jit::TensorType> * JITTensorType;

#define THS_API TH_API

#define CATCH(x) \
  try { \
    x \
  } catch (const exception& e) { \
      torch_last_err = strdup(e.what()); \
  }

// Utility method used to built sharable strings.
const char * make_sharable_string(const std::string str);

// Returns the latest error. This is thread-local.
THS_API const char * get_and_reset_last_err(); 

// Method concerting arrays of tensor poninters into arrays of tensors.
template<class T>
std::vector<T> toTensors(torch::Tensor ** tensorPtrs, const int length)
{
    std::vector<T> tensors;

    for (int i = 0; i < length; i++)
    {
        tensors.push_back(*tensorPtrs[i]);
    }

    return tensors;
}
