#include "THSTensor.h"

#include "stdafx.h"
#include "utils.h"

#include "TH/THTensor.h""
#include "torch/torch.h"

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
EXPORT_API(void) THS_Delete(const TensorWrapper * twrapper)
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

// Return a printable version of the device type storing the tensor.
EXPORT_API(const char*) THS_deviceType(const TensorWrapper * twrapper)
{
    auto device = twrapper->tensor.device();
    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return makeSharableString(device_type);
}

// Get the gradients for the input tensor.
EXPORT_API(TensorWrapper *) THS_Grad(const TensorWrapper * twrapper)
{
    return new TensorWrapper(twrapper->tensor.grad());
}

// Inplace subtraction with no grad
EXPORT_API(TensorWrapper *) THS_View(const TensorWrapper * lwrapper, const int64_t * shape, const int length)
{
    at::Tensor result = lwrapper->tensor.view(at::IntList(shape, length));
    return new TensorWrapper(result);
}

// Sum up values
EXPORT_API(TensorWrapper *) THS_Sum(const TensorWrapper * lwrapper)
{
    return new TensorWrapper(lwrapper->tensor.sum());
}

// Inplace subtraction with no grad
EXPORT_API(TensorWrapper *) THS_Eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.eq(rwrapper->tensor));
}

// Inplace subtraction with no grad
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

// Multiply the input tensor by the scalar. With no grad.
EXPORT_API(TensorWrapper *) THS_Argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim)
{
    return new TensorWrapper(twrapper->tensor.argmax(dimension, keepDim));
}

// Backard pass starting from the input tensor.
EXPORT_API(void) THS_Backward(TensorWrapper * twrapper)
{
    twrapper->tensor.backward();
}
