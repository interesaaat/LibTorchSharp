#include "THSTensor.h"

#include "stdafx.h"
#include "utils.h"

#include "TH/THTensor.h"
#include "torch/torch.h"

EXPORT_API(bool) THS_gradmode_is_enabled()
{
	return torch::autograd::GradMode::is_enabled();
}

EXPORT_API(void) THS_gradmode_set_enabled(bool enabled)
{
	torch::autograd::GradMode::set_enabled(enabled);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_ones(
    const int64_t * sizes, 
    const int lenght, 
    const int8_t scalar_type, 
    const char * device, 
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    at::Tensor tensor = torch::ones(at::IntList(sizes, lenght), options);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new(
    void * data, 
    const int64_t * sizes, 
    const int szlenght, 
    const int64_t * strides, 
    const int stlenght, 
    int8_t scalar_type)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type));

    at::Tensor tensor = torch::from_blob(data, at::IntList(sizes, szlenght), at::IntList(strides, stlenght), options);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_byteScalar(char data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_shortScalar(short data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_intScalar(int data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_longScalar(long data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_doubleScalar(double data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Create a variable tensor containing a tensor composed of ones.
EXPORT_API(TensorWrapper *) THS_new_floatScalar(float data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

// Returns a variable tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
EXPORT_API(TensorWrapper *) THS_randn(
    const int64_t * sizes, 
    const int lenght, 
    const int8_t scalar_type, 
    const char * device, 
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    at::Tensor tensor = torch::randn(at::IntList(sizes, lenght), options);

    return new TensorWrapper(tensor);
}

// Return the internal tensor implementation
EXPORT_API(THTensor *) THS_getTHTensorUnsafe(const TensorWrapper * twrapper)
{
    return twrapper->tensor.unsafeGetTensorImpl();
}

// Return the internal tensor implementation
EXPORT_API(void) THS_Dispose(const TensorWrapper * twrapper)
{
    delete twrapper;
}

// Return the tensor data
// Note that calling GetTHTensorUnsafe and get data from there won't work 
// (see the note [Tensor versus Variable in C++] in Aten\core\Tensor.h)
EXPORT_API(void *) THS_data(const TensorWrapper * twrapper)
{
    return twrapper->tensor.data_ptr();
}

// Return the inner type of the tensor.
EXPORT_API(int8_t) THS_Type(const TensorWrapper * twrapper)
{
    return (int8_t)twrapper->tensor.scalar_type();
}

// Return a printable version of the device type storing the tensor.
EXPORT_API(const char*) THS_deviceType(const TensorWrapper * twrapper)
{
    auto device = twrapper->tensor.device();
    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return makeSharableString(device_type);
}

// Create a copy of this tensor (if necessary) on a CPU device.
// If this tensor is already on the CPU device, it does not create a copy.
EXPORT_API(TensorWrapper *) THS_cpu(const TensorWrapper * twrapper)
{
	return new TensorWrapper(twrapper->tensor.cpu());
}

// Create a copy of this tensor (if necessary) on a CUDA device.
// If this tensor is already on the CUDA device, it does not create a copy.
EXPORT_API(TensorWrapper *) THS_cuda(const TensorWrapper * twrapper)
{
	return new TensorWrapper(twrapper->tensor.cuda());
}

// Get the gradients for the input tensor.
EXPORT_API(TensorWrapper *) THS_Grad(const TensorWrapper * twrapper)
{
	at::Tensor grad = twrapper->tensor.grad();
	return grad.defined() ? new TensorWrapper(grad) : NULL;
}

// Inplace subtraction with no grad
EXPORT_API(TensorWrapper *) THS_View(const TensorWrapper * lwrapper, const int64_t * shape, const int length)
{
    at::Tensor result = lwrapper->tensor.view(at::IntList(shape, length));
    return new TensorWrapper(result);
}

// Returns the sum of all elements in the input tensor.
EXPORT_API(TensorWrapper *) THS_Sum(const TensorWrapper * lwrapper)
{
    return new TensorWrapper(lwrapper->tensor.sum());
}

// Computes element-wise equality.
EXPORT_API(TensorWrapper *) THS_Eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    torch::NoGradGuard no_grad;

    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.eq(rwrapper->tensor));
}

// Inplace subtraction with no grad.
EXPORT_API(TensorWrapper *) THS_Sub_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    torch::NoGradGuard no_grad;

    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.sub_(rwrapper->tensor));
}

// Multiply the input tensor by the scalar. With no grad.
EXPORT_API(TensorWrapper *) THS_Mul(const TensorWrapper * twrapper, const float scalar)
{
    torch::NoGradGuard no_grad;
    return new TensorWrapper(twrapper->tensor.mul(scalar));
}

// Returns the indices of the maximum values of a tensor across a dimension.
EXPORT_API(TensorWrapper *) THS_Argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim)
{
    torch::NoGradGuard no_grad;
    return new TensorWrapper(twrapper->tensor.argmax(dimension, keepDim));
}

// Backard pass starting from the input tensor.
EXPORT_API(void) THS_Backward(TensorWrapper * twrapper)
{
    twrapper->tensor.backward();
}
