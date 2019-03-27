#pragma once

#include "TH/THGeneral.h"
#include "TH/THTensor.h"
#include "torch/torch.h"

// Inter-op structs.

// // Wrapper struct used to share ATen tensors.
struct TensorWrapper
{
    at::Tensor tensor;

    TensorWrapper(at::Tensor t) : tensor(t) {}
};

// API.

//  Creates  a variable tensor containing a tensor composed of ones.
TH_API TensorWrapper * THSTensor_ones(
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor out of the input data, dimensions and strides.
TH_API TensorWrapper * THSTensor_new(
    void * data,
    const int64_t * sizes,
    const int szlenght,
    const int64_t * strides,
    const int stlenght,
    int8_t scalar_type);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newByteScalar(char data);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newShortScalar(short data);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newIntScalar(int data);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newLongScalar(long data);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newDoubleScalar(double data);

//  Creates  a variable tensor wrapping the input scalar.
TH_API TensorWrapper * THSTensor_newFloatScalar(float data);

// Returns a variable tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
TH_API TensorWrapper * THSTensor_randn(
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// Returns the internal tensor implementation.
TH_API THTensor * THSTensor_unsafeGetTensorImpl(const TensorWrapper * twrapper);

// Disposes the tensor.
TH_API void THSTensor_dispose(const TensorWrapper * twrapper);

// Returns the tensor data
// Note that calling GetTHTensorUnsafe and get data from there won't work 
// (see the note [Tensor versus Variable in C++] in Aten\core\Tensor.h)
TH_API void * THSTensor_data(const TensorWrapper * twrapper);

// Returns the inner type of the tensor.
TH_API int8_t THSTensor_type(const TensorWrapper * twrapper);

// Returns a printable version of the device type storing the tensor.
TH_API const char* THSTensor_deviceType(const TensorWrapper * twrapper);

//  Creates  a copy of this tensor (if necessary) on a CPU device.
// If this tensor is already on the CPU device, it does not  Creates  a copy.
TH_API TensorWrapper * THSTensor_cpu(const TensorWrapper * twrapper);

//  Creates  a copy of this tensor (if necessary) on a CUDA device.
// If this tensor is already on the CUDA device, it does not  Creates  a copy.
TH_API TensorWrapper * THSTensor_cuda(const TensorWrapper * twrapper);

// Gets the gradients for the input tensor.
// If grandients are not defined returns NULL;
TH_API TensorWrapper * THSTensor_grad(const TensorWrapper * twrapper);

// Backard pass starting from the input tensor.
TH_API void THSTensor_backward(TensorWrapper * twrapper);

// Returns a new tensor with the same data as the tensor in twrapper but of a different shape.
// The returned tensor shares the same data and must have the same number of elements, 
// but may have a different size. For a tensor to be viewed, the new view size must be compatible 
// with its original size and stride. If -1 is the size of one dimension, 
// that size is inferred from other dimensions.
TH_API TensorWrapper * THSTensor_view(const TensorWrapper * twrapper, const int64_t * shape, const int length);

// Returns the sum of all elements in the input tensor.
TH_API TensorWrapper * THSTensor_sum(const TensorWrapper * twrapper);

// Computes element-wise equality.
TH_API TensorWrapper * THSTensor_eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Inplace subtraction of rwrapper to lwrapper. 
// The shape of rwrapper must be broadcastable with the shape of the left tensor.
TH_API TensorWrapper * THSTensor_sub_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Multiplies each element of the target tensor with the scalar value and returns a new resulting tensor.
TH_API TensorWrapper * THSTensor_mul(const TensorWrapper * twrapper, const float scalar);

// Returns the indices of the maximum values of a tensor across a dimension.
TH_API TensorWrapper * THSTensor_argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim);
