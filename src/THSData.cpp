#include "THSData.h"

#include "stdafx.h"
#include "utils.h"

#include "THSTensor.h"
#include <torch/torch.h>
#include <Windows.h>
#include <exception>

typedef torch::data::DataLoader<
    std::remove_reference_t<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>&>, torch::data::samplers::SequentialSampler> MNIST_t;

// Load an MNIST dataset from a file
EXPORT_API(DatasetIteratorBase *) Data_LoaderMNIST(
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
        std::move(dataset), batchSize);

    std::shared_ptr<MNIST_t> shared = std::move(loader);

    return new DatasetIterator<MNIST_t>(shared->begin(), size, shared);
}

//// Gets the size in byte of some dataset wrapped as iterator
EXPORT_API(size_t) Data_Size(DatasetIteratorBase * iterator)
{
    return iterator->getSize();
}

// Advance the pointer of the target iterator
EXPORT_API(bool) Data_MoveNext(DatasetIteratorBase * iterator)
{
    return iterator->moveNext();
}

// Get the curret data and target tensors pointed by the iterator
EXPORT_API(void) Data_Current(DatasetIteratorBase * iterator, TensorWrapper** data, TensorWrapper** target)
{
    iterator->current(data, target);
}

// Reset the iterator.
EXPORT_API(void) Data_Reset(DatasetIteratorBase * iterator)
{
    iterator->reset();
}

// Dispose the iterator.
EXPORT_API(void) Data_Dispose(DatasetIteratorBase * iterator)
{
    delete iterator;
}