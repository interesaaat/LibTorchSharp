#include "THSTensor.h"

#include "Utils.h"

#include "TH/THTensor.h"
#include "torch/torch.h"

TensorWrapper * THSTensor_ones(
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

TensorWrapper * THSTensor_new(
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

TensorWrapper * THSTensor_newByteScalar(char data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_newShortScalar(short data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_newIntScalar(int data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_newLongScalar(long data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_newDoubleScalar(double data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_newFloatScalar(float data)
{
    at::Tensor tensor = torch::tensor(data);

    return new TensorWrapper(tensor);
}

TensorWrapper * THSTensor_randn(
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

THTensor * THSTensor_unsafeGetTensorImpl(const TensorWrapper * twrapper)
{
    return twrapper->tensor.unsafeGetTensorImpl();
}

void THSTensor_dispose(const TensorWrapper * twrapper)
{
    delete twrapper;
}

void * THSTensor_data(const TensorWrapper * twrapper)
{
    return twrapper->tensor.data_ptr();
}

int8_t THSTensor_type(const TensorWrapper * twrapper)
{
    return (int8_t)twrapper->tensor.scalar_type();
}

const char* THSTensor_deviceType(const TensorWrapper * twrapper)
{
    auto device = twrapper->tensor.device();
    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return makeSharableString(device_type);
}

TensorWrapper * THSTensor_cpu(const TensorWrapper * twrapper)
{
	return new TensorWrapper(twrapper->tensor.cpu());
}

TensorWrapper * THSTensor_cuda(const TensorWrapper * twrapper)
{
	return new TensorWrapper(twrapper->tensor.cuda());
}

TensorWrapper * THSTensor_grad(const TensorWrapper * twrapper)
{
    at::Tensor grad = twrapper->tensor.grad();
    return grad.defined() ? new TensorWrapper(grad) : NULL;
}

void THSTensor_backward(TensorWrapper * twrapper)
{
    twrapper->tensor.backward();
}

TensorWrapper * THSTensor_view(const TensorWrapper * lwrapper, const int64_t * shape, const int length)
{
    at::Tensor result = lwrapper->tensor.view(at::IntList(shape, length));
    return new TensorWrapper(result);
}

TensorWrapper * THSTensor_sum(const TensorWrapper * lwrapper)
{
    torch::NoGradGuard no_grad;

    return new TensorWrapper(lwrapper->tensor.sum());
}

TensorWrapper * THSTensor_eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.eq(rwrapper->tensor));
}

TensorWrapper * THSTensor_sub_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.sub_(rwrapper->tensor));
}

TensorWrapper * THSTensor_mul(const TensorWrapper * twrapper, const float scalar)
{
    return new TensorWrapper(twrapper->tensor.mul(scalar));
}

TensorWrapper * THSTensor_argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim)
{
    return new TensorWrapper(twrapper->tensor.argmax(dimension, keepDim));
}
