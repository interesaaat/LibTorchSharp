#include "utils.h"

#include <cstring>
#include <fstream>

const char * makeSharableString(const std::string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}

std::ofstream GetLog(const std::string filename)
{
    std::ofstream logger;
    logger.open(filename);
    return logger;
}

//std::vector<torch::Tensor> toTensors(const Tensor * tensorPtrs, const int length)
//{
//    std::vector<torch::Tensor> tensors;
//
//    for (int i = 0; i < length; i++)
//    {
//        tensors.push_back(*tensorPtrs[i]);
//    }
//
//    return tensors;
//}