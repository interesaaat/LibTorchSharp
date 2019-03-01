#pragma once

#include <torch/torch.h>

class DatasetIterator
{
    public:
       torch::data::Iterator<torch::data::Example<>> currentIter;
       void* tmp;
      torch::data::Iterator<torch::data::Example<>> endIter;
    size_t size;

   DatasetIterator(torch::data::Iterator<torch::data::Example<>> i, size_t size, torch::data::Iterator<torch::data::Example<>> e, void * t) :
        currentIter(torch::data::Iterator<torch::data::Example<>>(i)),
        endIter(torch::data::Iterator<torch::data::Example<>>(std::move(e))),
            size(size),
   tmp(t) {}

   ~DatasetIterator()
   {
       delete(&currentIter);
       delete(&endIter);
       delete(tmp);
   }
};

struct DatasetIteratorWrapper
{
    DatasetIterator * iter;

    DatasetIteratorWrapper(DatasetIterator * i) : iter(i) {}
};