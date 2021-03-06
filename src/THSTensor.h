#pragma once

#include "TH/THTensor.h"
#include "torch/torch.h"

#include "Utils.h"

// API.

//  Creates  a variable tensor containing a tensor composed of zeros.
THS_API Tensor THSTensor_zeros(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor containing a tensor composed of ones.
THS_API Tensor THSTensor_ones(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor containing a an empty tensor.
THS_API Tensor THSTensor_empty(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

//  Creates  a variable tensor out of the input data, dimensions and strides.
THS_API Tensor THSTensor_new(
    void * data,
    const int64_t * sizes,
    const int szlength,
    const int64_t * strides,
    const int stlength,
    int8_t scalar_type);

THS_API Tensor THSTensor_newLong(
    int64_t * data,
    const int64_t * sizes,
    const int szlength,
    const int64_t * strides,
    const int stlength,
    int8_t scalar_type);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newByteScalar(char data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newShortScalar(short data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newIntScalar(int data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newLongScalar(int64_t data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newDoubleScalar(double data);

//  Creates  a variable tensor wrapping the input scalar.
THS_API Tensor THSTensor_newFloatScalar(float data);

// Returns a variable tensor filled with random numbers from a uniform distribution within [0, 1).
THS_API Tensor THSTensor_rand(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// Returns a variable tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
THS_API Tensor THSTensor_randn(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// A sparse tensor is represented as a pair of dense tensors: a tensor of values and a 2D tensor of indices. 
// A sparse tensor can be constructed by providing these two tensors, as well as the size of the sparse tensor. 
THS_API Tensor THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad);

// Returns the number of dimensions of the input tensor.
THS_API int64_t THSTensor_ndimension(const Tensor tensor);

// Returns the size of the target dimension of the input tensor.
THS_API int64_t THSTensor_size(const Tensor tensor, const int64_t dimension);

// Returns the stride of the target dimension of the input tensor.
THS_API int64_t THSTensor_stride(const Tensor tensor, const int64_t dimension);

// Disposes the tensor.
THS_API void THSTensor_dispose(const Tensor twrapper);

// Returns the tensor data
// Note that calling GetTHTensorUnsafe and get data from there won't work 
// (see the note [Tensor versus Variable in C++] in Aten\core\Tensor.h)
THS_API void * THSTensor_data(const Tensor twrapper);

// Returns the value of this tensor as a Scalar. 
// This only works for tensors with one element.
THS_API Scalar THSTensor_item(const Tensor tensor);

// Returns the sub-tensor identified by the index.
THS_API Tensor THSTensor_get1(const Tensor tensor, int64_t index);

// Returns the sub-tensor identified by the indexes.
THS_API Tensor THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2);

// Returns the sub-tensor identified by the indexes.
THS_API Tensor THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3);

// Set the sub-tensor identified by the index to value.
THS_API void THSTensor_set1(const Tensor tensor, int64_t index, Scalar value);

// Set the sub-tensor identified by the indexes to value.
THS_API void THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, Scalar value);

// Set the sub-tensor identified by the indexes to value.
THS_API void THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, Scalar value);

// Returns the inner type of the tensor.
THS_API int8_t THSTensor_type(const Tensor twrapper);

// Returns a printable version of the device type storing the tensor.
THS_API const char* THSTensor_deviceType(const Tensor twrapper);

// Wheter the tensor requires grad or not.
THS_API bool THSTensor_requires_grad(const Tensor tensor);

// Set the value for requires grad.
THS_API Tensor THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad);

// Returns whether the input tensor is sparse or not.
THS_API int THSTensor_isSparse(const Tensor twrapper);

// Returns whether the input tensor is a variable or not.
THS_API int THSTensor_isVariable(const Tensor twrapper);

// Returns the indices  of the sparse tensor.
THS_API Tensor THSTensor_indices(Tensor tensor);

// Returns the values of the sparse tensor.
THS_API Tensor THSTensor_values(Tensor tensor);

//  Creates  a copy of this tensor (if necessary) on a CPU device.
// If this tensor is already on the CPU device, it does not  Creates  a copy.
THS_API Tensor THSTensor_cpu(const Tensor twrapper);

//  Creates  a copy of this tensor (if necessary) on a CUDA device.
// If this tensor is already on the CUDA device, it does not  Creates  a copy.
THS_API Tensor THSTensor_cuda(const Tensor twrapper);

// Gets the gradients for the input tensor.
// If grandients are not defined returns NULL;
THS_API Tensor THSTensor_grad(const Tensor twrapper);

// Backard pass starting from the input tensor.
THS_API void THSTensor_backward(Tensor twrapper);

// Turns a sparse tensor into a dense representation.
THS_API Tensor THSTensor_to_dense(Tensor tensor);

// Concatenates the given sequence of seq tensors in the given dimension. 
// All tensors must either have the same shape (except in the concatenating dimension) or be empty.
// See https://pytorch.org/docs/stable/torch.html#torch.cat for examples.
THS_API Tensor THSTensor_cat(const Tensor* twrapper, const int length, const int64_t dim);

// Returns a contiguous tensor containing the same data as the input tensor. 
// If the input tensor is contiguous, this function returns input.
THS_API Tensor THSTensor_contiguous(const Tensor input);

// Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.
THS_API Tensor THSTensor_index_select(Tensor tensor, int64_t dimension, Tensor index);

// Returns a tensor with the same data and number of elements as input, but with the specified shape.
// When possible, the returned tensor will be a view of input.Otherwise, it will be a copy.
// Contiguous inputs and inputs with compatible strides can be reshaped without copying, 
// but you should not depend on the copying vs.viewing behavior.
THS_API Tensor THSTensor_reshape(const Tensor twrapper, const int64_t * shape, const int length);

// Returns a tensor with all the dimensions of input of size 1 removed.
THS_API Tensor THSTensor_squeeze(Tensor tensor, int64_t dimension);

// Concatenates sequence of tensors along a new dimension.
// All tensors need to be of the same size.
THS_API Tensor THSTensor_stack(const Tensor* twrapper, const int length, const int64_t dim);

// Returns a tensor that is a transposed version of input. 
THS_API Tensor THSTensor_t(const Tensor tensor);

// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
THS_API Tensor THSTensor_transpose(const Tensor twrapper, const int64_t dim1, const int64_t dim2);

// Returns a tensor that is a transposed version of input.The given dimensions dim0 and dim1 are swapped.
// This operation is in place.
THS_API void THSTensor_transpose_(const Tensor twrapper, const int64_t dim1, const int64_t dim2);

// Returns a new tensor with the same data as the tensor in twrapper but of a different shape.
// The returned tensor shares the same data and must have the same number of elements, 
// but may have a different size. For a tensor to be viewed, the new view size must be compatible 
// with its original size and stride. If -1 is the size of one dimension, 
// that size is inferred from other dimensions.
THS_API Tensor THSTensor_view(const Tensor twrapper, const int64_t * shape, const int length);

// Each element of the tensor other is multiplied by the scalar value 
// and added to each element of the tensor input. The resulting tensor is returned.
THS_API Tensor THSTensor_add(const Tensor left, const int value, const Tensor right);

// Each element of the tensor other is increased by the scalar value.
THS_API Tensor THSTensor_addS(const Tensor left, const Scalar right);

// Each element of the tensor other is multiplied by the scalar value 
// and added to each element of the tensor input. The resulting tensor is returned.
// This operation is in place.
THS_API void THSTensor_add_(const Tensor left, const int value, const Tensor right);

// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced add step 
// (all matrix multiplications get accumulated along the first dimension). mat is added to the final result.
// Check https://pytorch.org/docs/stable/torch.html#torch.addbmm for details.
THS_API Tensor THSTensor_addbmm(
    const Tensor matWrapper,
    const Tensor batch1Wrapper,
    const Tensor batch2Wrapper,
    const float beta,
    const float alpha);

// Performs a matrix multiplication of the matrices mat1 and mat2. 
// The matrix mat is added to the final result.
THS_API Tensor THSTensor_addmm(
    const Tensor matWrapper,
    const Tensor mat1Wrapper,
    const Tensor mat2Wrapper,
    const float beta,
    const float alpha);

// Returns the indices of the maximum values of a tensor across a dimension.
THS_API Tensor THSTensor_argmax(const Tensor twrapper, const int64_t dimension, bool keepDim);

// Performs a batch matrix - matrix product of matrices in batch1 and batch2.mat is added to the final result.
// Batch1 and batch2 must be 3 - D tensors each containing the same number of matrices.
// Check https://pytorch.org/docs/stable/torch.html#torch.baddbmm for details.
THS_API Tensor THSTensor_baddbmm(
    const Tensor batch1Wrapper,
    const Tensor batch2Wrapper,
    const Tensor matWrapper,
    const float beta,
    const float alpha);

// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
THS_API Tensor THSTensor_bmm(const Tensor b1wrapper, const Tensor b2wrapper);

// Clamp all elements in input into the range [ min, max ] and return a resulting tensor.
THS_API Tensor THSTensor_clamp(const Tensor input, const Scalar min, const Scalar max);

// Returns left / right. 
THS_API Tensor THSTensor_div(const Tensor left, const Tensor right);

// Returns left / right in place.
THS_API void THSTensor_div_(const Tensor left, const Tensor right);

// Returns left / right. 
THS_API Tensor THSTensor_divS(const Tensor left, const int right);

// Computes input > other element-wise.
THS_API Tensor THSTensor_gtS(const Tensor input, const Scalar other);

// Computes element-wise equality.
THS_API Tensor THSTensor_eq(const Tensor left, const Tensor right);

// Computes element-wise equality.
THS_API Tensor THSTensor_eqS(const Tensor left, const Scalar right);

// True if two tensors have the same size and elements, False otherwise.
THS_API int THSTensor_equal(const Tensor left, const Tensor right);

// Returns a new tensor with the exponential of the elements of the input tensor input.
THS_API Tensor THSTensor_exp(const Tensor twrapper);

// Matrix product of two tensors.
// The behavior depends on the dimensionality of the tensors.
// Check https://pytorch.org/docs/stable/torch.html#torch.matmul for details.
THS_API Tensor THSTensor_matmul(const Tensor left, const Tensor right);

// Returns the mean of all elements in the input tensor.
THS_API Tensor THSTensor_mean(const Tensor tensor);

// Performs a matrix multiplication of the matrices mat1 and mat2.
// This operation does not broadcast. For broadcasting use matmul.
THS_API Tensor THSTensor_mm(const Tensor left, const Tensor right);

// Each element of the left tensor is multiplied by each element of the rigth Tensor. 
// The resulting tensor is returned.
THS_API Tensor THSTensor_mul(const Tensor left, const Tensor right);

// Each element of the left tensor is multiplied by each element of the rigth Tensor. 
// This operation is in place.
THS_API void THSTensor_mul_(const Tensor left, const Tensor right);

// Returns the matrix norm or vector norm of a given tensor.
THS_API Tensor THSTensor_norm(const Tensor tensor, const int64_t dimension, const bool keep_dimension);

// Multiplies each element of the target tensor with the scalar value and returns a new resulting tensor.
THS_API Tensor THSTensor_mulS(const Tensor twrapper, const Scalar scalar);

// Takes the power of each element in input with exponent and returns a tensor with the result.
THS_API Tensor THSTensor_pow(const Tensor twrapper, const Scalar scalar);

// Returns a new tensor with the sigmoid of the elements of input.
THS_API Tensor THSTensor_sigmoid(const Tensor twrapper);

// Subtraction of right to left. 
// The shape of right must be broadcastable with the shape of the left tensor.
THS_API Tensor THSTensor_sub(const Tensor left, const Tensor right);

// Inplace subtraction of right to left. 
// The shape of right must be broadcastable with the shape of the left tensor.
THS_API void THSTensor_sub_(const Tensor left, const Tensor right);

// Returns the sum of all elements in the input tensor.
THS_API Tensor THSTensor_sum(const Tensor twrapper);

// Returns the sum of all elements over the input dimensions in the input tensor.
THS_API Tensor THSTensor_sum1(const Tensor tensor, const int64_t * dimensions, int length, bool keep_dimension);
