#pragma once

#include "TH/THTensor.h"
#include "torch/torch.h"

#include "Utils.h"

// Inter-op structs.

// // Wrapper struct used to share ATen tensors.
struct TensorWrapper
{
    at::Tensor tensor;

    TensorWrapper(at::Tensor t) : tensor(t) {}
};

// API.

//  Creates  a variable tensor containing a tensor composed of zeros.
THS_API TensorWrapper * THSTensor_zeros(
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor containing a tensor composed of ones.
THS_API TensorWrapper * THSTensor_ones(
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor out of the input data, dimensions and strides.
THS_API TensorWrapper * THSTensor_new(
    void * data,
    const int64_t * sizes,
    const int szlenght,
    const int64_t * strides,
    const int stlenght,
    int8_t scalar_type);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newByteScalar(char data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newShortScalar(short data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newIntScalar(int data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newLongScalar(long data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newDoubleScalar(double data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API TensorWrapper * THSTensor_newFloatScalar(float data);

// Returns a variable tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
THS_API TensorWrapper * THSTensor_randn(
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// A sparse tensor is represented as a pair of dense tensors: a tensor of values and a 2D tensor of indices. 
// A sparse tensor can be constructed by providing these two tensors, as well as the size of the sparse tensor. 
THS_API TensorWrapper * THSTensor_sparse(
    TensorWrapper * indices,
    TensorWrapper * values,
    const int64_t * sizes,
    const int lenght,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// Returns the internal tensor implementation.
THS_API THTensor * THSTensor_unsafeGetTensorImpl(const TensorWrapper * twrapper);

// Disposes the tensor.
THS_API void THSTensor_dispose(const TensorWrapper * twrapper);

// Returns the tensor data
// Note that calling GetTHTensorUnsafe and get data from there won't work 
// (see the note [Tensor versus Variable in C++] in Aten\core\Tensor.h)
THS_API void * THSTensor_data(const TensorWrapper * twrapper);

// Returns the inner type of the tensor.
THS_API int8_t THSTensor_type(const TensorWrapper * twrapper);

// Returns a printable version of the device type storing the tensor.
THS_API const char* THSTensor_deviceType(const TensorWrapper * twrapper);

//  Creates  a copy of this tensor (if necessary) on a CPU device.
// If this tensor is already on the CPU device, it does not  Creates  a copy.
THS_API TensorWrapper * THSTensor_cpu(const TensorWrapper * twrapper);

//  Creates  a copy of this tensor (if necessary) on a CUDA device.
// If this tensor is already on the CUDA device, it does not  Creates  a copy.
THS_API TensorWrapper * THSTensor_cuda(const TensorWrapper * twrapper);

// Gets the gradients for the input tensor.
// If grandients are not defined returns NULL;
THS_API TensorWrapper * THSTensor_grad(const TensorWrapper * twrapper);

// Backard pass starting from the input tensor.
THS_API void THSTensor_backward(TensorWrapper * twrapper);

// Concatenates the given sequence of seq tensors in the given dimension. 
// All tensors must either have the same shape (except in the concatenating dimension) or be empty.
// See https://pytorch.org/docs/stable/torch.html#torch.cat for examples.
THS_API TensorWrapper * THSTensor_cat(const TensorWrapper ** twrapper, const int length, const int64_t dim);

// Returns a tensor with the same data and number of elements as input, but with the specified shape.
// When possible, the returned tensor will be a view of input.Otherwise, it will be a copy.
// Contiguous inputs and inputs with compatible strides can be reshaped without copying, 
// but you should not depend on the copying vs.viewing behavior.
THS_API TensorWrapper * THSTensor_reshape(const TensorWrapper * twrapper, const int64_t * shape, const int length);

// Concatenates sequence of tensors along a new dimension.
// All tensors need to be of the same size.
THS_API TensorWrapper * THSTensor_stack(const TensorWrapper ** twrapper, const int length, const int64_t dim);

// Returns a tensor that is a transposed version of input.The given dimensions dim0 and dim1 are swapped.
THS_API TensorWrapper * THSTensor_transpose(const TensorWrapper * twrapper, const int64_t dim1, const int64_t dim2);

// Returns a tensor that is a transposed version of input.The given dimensions dim0 and dim1 are swapped.
// This operation is in place.
THS_API void THSTensor_transpose_(const TensorWrapper * twrapper, const int64_t dim1, const int64_t dim2);

// Returns a new tensor with the same data as the tensor in twrapper but of a different shape.
// The returned tensor shares the same data and must have the same number of elements, 
// but may have a different size. For a tensor to be viewed, the new view size must be compatible 
// with its original size and stride. If -1 is the size of one dimension, 
// that size is inferred from other dimensions.
THS_API TensorWrapper * THSTensor_view(const TensorWrapper * twrapper, const int64_t * shape, const int length);

// Each element of the tensor other is multiplied by the scalar value 
// and added to each element of the tensor input. The resulting tensor is returned.
THS_API void THSTensor_add_(const TensorWrapper * lwrapper, const int value, const TensorWrapper * rwrapper);

// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced add step 
// (all matrix multiplications get accumulated along the first dimension). mat is added to the final result.
// Check https://pytorch.org/docs/stable/torch.html#torch.addbmm for details.
THS_API TensorWrapper * THSTensor_addbmm(
    const TensorWrapper * matWrapper,
    const TensorWrapper * batch1Wrapper,
    const TensorWrapper * batch2Wrapper,
    const float beta,
    const float alpha);

// Performs a matrix multiplication of the matrices mat1 and mat2. 
// The matrix mat is added to the final result.
THS_API TensorWrapper * THSTensor_addmm(
    const TensorWrapper * matWrapper,
    const TensorWrapper * mat1Wrapper,
    const TensorWrapper * mat2Wrapper,
    const float beta,
    const float alpha);

// Returns the indices of the maximum values of a tensor across a dimension.
THS_API TensorWrapper * THSTensor_argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim);

// Performs a batch matrix - matrix product of matrices in batch1 and batch2.mat is added to the final result.
// Batch1 and batch2 must be 3 - D tensors each containing the same number of matrices.
// Check https://pytorch.org/docs/stable/torch.html#torch.baddbmm for details.
THS_API TensorWrapper * THSTensor_baddbmm(
    const TensorWrapper * batch1Wrapper,
    const TensorWrapper * batch2Wrapper,
    const TensorWrapper * matWrapper,
    const float beta,
    const float alpha);

// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
THS_API TensorWrapper * THSTensor_bmm(const TensorWrapper * b1wrapper, const TensorWrapper * b2wrapper);

// Computes element-wise equality.
THS_API TensorWrapper * THSTensor_eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Returns a new tensor with the exponential of the elements of the input tensor input.
THS_API TensorWrapper * THSTensor_exp(const TensorWrapper * twrapper);

// Matrix product of two tensors.
// The behavior depends on the dimensionality of the tensors.
// Check https://pytorch.org/docs/stable/torch.html#torch.matmul for details.
THS_API TensorWrapper * THSTensor_matmul(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Performs a matrix multiplication of the matrices mat1 and mat2.
// This operation does not broadcast. For broadcasting use matmul.
THS_API TensorWrapper * THSTensor_mm(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Each element of the left tensor is multiplied by each element of the rigth Tensor. 
// The resulting tensor is returned.
THS_API TensorWrapper * THSTensor_mul(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Each element of the left tensor is multiplied by each element of the rigth Tensor. 
// This operation is in place.
THS_API void THSTensor_mul_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Multiplies each element of the target tensor with the scalar value and returns a new resulting tensor.
THS_API TensorWrapper * THSTensor_mulS(const TensorWrapper * twrapper, const float scalar);

// Takes the power of each element in input with exponent and returns a tensor with the result.
THS_API TensorWrapper * THSTensor_pow(const TensorWrapper * twrapper, const float scalar);

// Returns a new tensor with the sigmoid of the elements of input.
THS_API TensorWrapper * THSTensor_sigmoid(const TensorWrapper * twrapper);

// Subtraction of rwrapper to lwrapper. 
// The shape of rwrapper must be broadcastable with the shape of the left tensor.
THS_API TensorWrapper * THSTensor_sub(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Inplace subtraction of rwrapper to lwrapper. 
// The shape of rwrapper must be broadcastable with the shape of the left tensor.
THS_API void THSTensor_sub_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper);

// Returns the sum of all elements in the input tensor.
THS_API TensorWrapper * THSTensor_sum(const TensorWrapper * twrapper);

/// Fills the given 2-dimensional input tensor with values drawn from a uniform
/// distribution parameterized by `low` and `high`.
/// No gradient will be recorded for this operation.
THS_API void THSTensor_initUniform(TensorWrapper * twrapper, double low, double high);


