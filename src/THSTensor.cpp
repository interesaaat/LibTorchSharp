#include "THSTensor.h"

TensorWrapper * THSTensor_zeros(
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

    at::Tensor tensor = torch::zeros(at::IntList(sizes, lenght), options);

    return new TensorWrapper(tensor);
}

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

TensorWrapper * THSTensor_reshape(const TensorWrapper * twrapper, const int64_t * shape, const int length)
{
    at::Tensor result = twrapper->tensor.reshape(at::IntList(shape, length));
    return new TensorWrapper(result);
}

TensorWrapper * THSTensor_view(const TensorWrapper * twrapper, const int64_t * shape, const int length)
{
    at::Tensor result = twrapper->tensor.view(at::IntList(shape, length));
    return new TensorWrapper(result);
}

TensorWrapper * THSTensor_add(const TensorWrapper * lwrapper, const int value, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.add(rwrapper->tensor, value));
}

void THSTensor_add_(const TensorWrapper * lwrapper, const int value, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    left.add_(rwrapper->tensor, value);
}

TensorWrapper * THSTensor_addbmm(
    const TensorWrapper * matWrapper,
    const TensorWrapper * batch1Wrapper,
    const TensorWrapper * batch2Wrapper,
    const float beta,
    const float alpha)
{
    at::Tensor mat = matWrapper->tensor;
    return new TensorWrapper(mat.addbmm(batch1Wrapper->tensor, batch2Wrapper->tensor, beta, alpha));
}

TensorWrapper * THSTensor_argmax(const TensorWrapper * twrapper, const int64_t dimension, bool keepDim)
{
    return new TensorWrapper(twrapper->tensor.argmax(dimension, keepDim));
}

TensorWrapper * THSTensor_baddbmm(
    const TensorWrapper * batch1Wrapper,
    const TensorWrapper * batch2Wrapper,
    const TensorWrapper * matWrapper,
    const float beta,
    const float alpha)
{
    at::Tensor batch1 = batch1Wrapper->tensor;
    return new TensorWrapper(batch1.baddbmm(batch2Wrapper->tensor, matWrapper->tensor, beta, alpha));
}

TensorWrapper * THSTensor_eq(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.eq(rwrapper->tensor));
}

TensorWrapper * THSTensor_exp(const TensorWrapper * twrapper)
{
    return new TensorWrapper(twrapper->tensor.exp());
}

TensorWrapper * THSTensor_matMul(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.matmul(rwrapper->tensor));
}

TensorWrapper * THSTensor_mul(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.mul(rwrapper->tensor));
}

void THSTensor_mul_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    left.mul_(rwrapper->tensor);
}

TensorWrapper * THSTensor_mulS(const TensorWrapper * twrapper, const float scalar)
{
    return new TensorWrapper(twrapper->tensor.mul(scalar));
}

TensorWrapper * THSTensor_pow(const TensorWrapper * twrapper, const float scalar)
{
    return new TensorWrapper(twrapper->tensor.pow(scalar));
}

TensorWrapper * THSTensor_sigmoid(const TensorWrapper * twrapper)
{
    return new TensorWrapper(twrapper->tensor.sigmoid());
}

TensorWrapper * THSTensor_sub(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    return new TensorWrapper(left.sub(rwrapper->tensor));
}

void THSTensor_sub_(const TensorWrapper * lwrapper, const TensorWrapper * rwrapper)
{
    at::Tensor left = lwrapper->tensor;
    left.sub_(rwrapper->tensor);
}

TensorWrapper * THSTensor_sum(const TensorWrapper * lwrapper)
{
    return new TensorWrapper(lwrapper->tensor.sum());
}

void THSTensor_initUniform(TensorWrapper * twrapper, double low, double high)
{
    torch::nn::init::uniform_(twrapper->tensor, low, high);
}


