#pragma once

#include "stdafx.h"
#include "THSTensor.h"

#include <torch/torch.h>

// Base non-generic interator class. Used to communicate with C#.
class DatasetIteratorBase
{
public:
    explicit
    DatasetIteratorBase() {}
    virtual size_t getSize() = 0;
    virtual bool moveNext() = 0;
    virtual void current(TensorWrapper** data, TensorWrapper** target) = 0;
    virtual void reset() = 0;
};

// Generic version of the iterator class.
template<typename Dataset>
class DatasetIterator : public DatasetIteratorBase
{
public:
    DatasetIterator(
        torch::data::Iterator<torch::data::Example<>> i,
        size_t s,
        std::shared_ptr<Dataset> l) : 
        DatasetIteratorBase(), 
        currentIter(torch::data::Iterator<torch::data::Example<>>(i)),
        size(s),
        loaderPointer(l) {}

        size_t getSize();
        bool moveNext();
        void current(TensorWrapper** data, TensorWrapper** target);
        void reset();

private:
        std::shared_ptr<Dataset> loaderPointer;
        torch::data::Iterator<torch::data::Example<>> currentIter;
        size_t size;
};

// Get the total size in bytes of the input dataset.
template<typename Dataset>
inline size_t DatasetIterator<Dataset>::getSize()
{
    return size;
}

// Advance the iterator.
template<typename Dataset>
inline bool DatasetIterator<Dataset>::moveNext()
{
    ++currentIter;

    return currentIter != loaderPointer.get()->end();
}

// Get the current object pointed by the iterator.
template<typename Dataset>
inline void DatasetIterator<Dataset>::current(TensorWrapper** data, TensorWrapper** target)
{
    data[0] = new TensorWrapper(std::move(currentIter->data));
    target[0] = new TensorWrapper(std::move(currentIter->target));
}

// Reset the iterator to start from the beginning.
template<typename Dataset>
inline void DatasetIterator<Dataset>::reset()
{
    currentIter = loaderPointer.get()->begin();
}