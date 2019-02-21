#include "THSData.h"

#include "stdafx.h"
#include "THSTensor.h"
#include <torch/torch.h>

// Load an MNIST dataset from a file
EXPORT_API(void) Data_LoaderMNIST(
    const char* filename, 
    int64_t batchSize, 
    bool isTrain,
    const TensorWrapper** (*dataAllocator)(size_t length),
    const TensorWrapper** (*targetAllocator)(size_t length))
{
    torch::data::datasets::MNIST::Mode mode = torch::data::datasets::MNIST::Mode::kTrain;

    if (!isTrain)
    {
        mode = torch::data::datasets::MNIST::Mode::kTest;
    }

    auto dataset = torch::data::datasets::MNIST(filename, mode)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    const size_t dataset_size = dataset.size().value();

    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset), batchSize);

    const TensorWrapper **data = dataAllocator(dataset_size);
    const TensorWrapper **target = targetAllocator(dataset_size);
    int i = 0;

    for (auto& batch : *loader) 
    {
        data[i] = new TensorWrapper(batch.data);
        target[i] = new TensorWrapper(batch.target);
    }
}