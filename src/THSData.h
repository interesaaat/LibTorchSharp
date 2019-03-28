#pragma once

#include "THSTensor.h"

// Inter-op classes.

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

// Class-related methods.

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

// API.

// Load a MNIST dataset from a file.
THS_API DatasetIteratorBase * THSData_loaderMNIST(
    const char* filename,
    int64_t batchSize,
    bool isTrain);

// Gets the size in byte of some dataset wrapped as iterator.
THS_API size_t THSData_size(DatasetIteratorBase * iterator);

// Advances the pointer of the target iterator.
THS_API bool THSData_moveNext(DatasetIteratorBase * iterator);

// Gets the curret data and target tensors pointed by the iterator.
THS_API void THSData_current(DatasetIteratorBase * iterator, TensorWrapper** data, TensorWrapper** target);

// Resets the iterator.
THS_API void THSData_reset(DatasetIteratorBase * iterator);

// Disposes the iterator.
THS_API void THSData_dispose(DatasetIteratorBase * iterator);