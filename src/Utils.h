#pragma once

#include <string>

#include "TH/THGeneral.h"
#include "torch/torch.h"

#define THS_API TH_API

// Utility method used to built sharable strings.
const char * makeSharableString(const std::string str);

// Utility method used for debugging through logging.
std::ofstream GetLog(const std::string filename);

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