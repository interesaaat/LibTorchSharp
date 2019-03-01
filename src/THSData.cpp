#include "THSData.h"

#include "stdafx.h"
#include "utils.h"

#include "THSTensor.h"
#include <torch/torch.h>
#include <Windows.h>
#include <exception>

// Load an MNIST dataset from a file
EXPORT_API(DatasetIteratorWrapper *) Data_LoaderMNIST(
    const char* filename,
    int64_t batchSize,
    bool isTrain)
{
    torch::data::datasets::MNIST::Mode mode = torch::data::datasets::MNIST::Mode::kTrain;

    if (!isTrain)
    {
        mode = torch::data::datasets::MNIST::Mode::kTest;
    }

    auto dataset = torch::data::datasets::MNIST(filename, mode)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    size_t size = dataset.size().value();

    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset), batchSize).release();

    auto iter = new DatasetIterator(loader->begin(), size, loader->end(), (void*)loader);

    return new DatasetIteratorWrapper(iter);
}

// Gets the size in byte of some dataset wrapped as iterator
EXPORT_API(size_t) Data_Size(DatasetIteratorWrapper * wrapper)
{
    return wrapper->iter->size;
}

// Advance the pointer of the target iterator
EXPORT_API(bool) Data_MoveNext(DatasetIteratorWrapper * wrapper)
{
    ++(wrapper->iter->currentIter);

    return (wrapper->iter->currentIter != wrapper->iter->endIter);
}

// Get the curret data and target tensors pointed by the iterator
EXPORT_API(void) Data_Current(DatasetIteratorWrapper * wrapper, TensorWrapper** data, TensorWrapper** target)
{
    data[0] = new TensorWrapper((wrapper->iter->currentIter)->data);
    target[0] = new TensorWrapper((wrapper->iter->currentIter)->target);
}