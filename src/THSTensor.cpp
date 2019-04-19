#include "THSTensor.h"

Tensor THSTensor_zeros(
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

    return new torch::Tensor(torch::zeros(at::IntList(sizes, lenght), options));
}

Tensor THSTensor_ones(
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

    return new torch::Tensor(torch::ones(at::IntList(sizes, lenght), options));
}

Tensor THSTensor_empty(
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

    return new torch::Tensor(torch::empty(at::IntList(sizes, lenght), options));
}

Tensor THSTensor_new(
    void * data, 
    const int64_t * sizes, 
    const int szlenght, 
    const int64_t * strides, 
    const int stlenght, 
    int8_t scalar_type)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type));

    return new torch::Tensor(torch::from_blob(data, at::IntList(sizes, szlenght), at::IntList(strides, stlenght), options));
}

Tensor THSTensor_newLong(
    int64_t * data,
    const int64_t * sizes,
    const int szlenght,
    const int64_t * strides,
    const int stlenght,
    int8_t scalar_type)
{
    return new torch::Tensor(torch::from_blob(data, at::IntList(sizes, szlenght), at::IntList(strides, stlenght), at::kLong));
}

Tensor THSTensor_newByteScalar(char data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_newShortScalar(short data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_newIntScalar(int data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_newLongScalar(int64_t data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_newDoubleScalar(double data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_newFloatScalar(float data)
{
    return new torch::Tensor(torch::tensor(data));
}

Tensor THSTensor_randn(
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

    return new torch::Tensor(torch::randn(at::IntList(sizes, lenght), options));
}

Tensor THSTensor_sparse(
    Tensor indices,
    Tensor values,
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

    auto i = torch::autograd::as_variable_ref(*indices).data();
    auto v = torch::autograd::as_variable_ref(*values).data();

    return new torch::Tensor(torch::sparse_coo_tensor(i, v, at::IntList(sizes, lenght), options));
}

int64_t THSTensor_ndimension(const Tensor tensor)
{
    return tensor->ndimension();
}

int64_t THSTensor_stride(const Tensor tensor, const int64_t dimension)
{
    return tensor->stride(dimension);
}

int64_t THSTensor_size(const Tensor tensor, const int64_t dimension)
{
    return tensor->size(dimension);
}

void THSTensor_dispose(const Tensor tensor)
{
    delete tensor;
}

void * THSTensor_data(const Tensor tensor)
{
    return tensor->data_ptr();
}

int8_t THSTensor_type(const Tensor tensor)
{
    return (int8_t)tensor->scalar_type();
}

Tensor THSTensor_get1(const Tensor tensor, int64_t index)
{
    return new torch::Tensor((*tensor)[index]);
}

Tensor THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2)
{
    return new torch::Tensor((*tensor)[index1, index2]);
}

Tensor THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3)
{
    return new torch::Tensor((*tensor)[index1, index2, index3]);
}

const char* THSTensor_deviceType(const Tensor tensor)
{
    auto device = tensor->device();
    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return make_sharable_string(device_type);
}

bool THSTensor_isSparse(const Tensor tensor)
{
    return tensor->is_sparse();
}

bool THSTensor_isVariable(const Tensor tensor)
{
    return tensor->is_variable();
}

Tensor THSTensor_cpu(const Tensor tensor)
{
	return new torch::Tensor(tensor->cpu());
}

Tensor THSTensor_cuda(const Tensor tensor)
{
	return new torch::Tensor(tensor->cuda());
}

Tensor THSTensor_grad(const Tensor tensor)
{
    torch::Tensor grad = tensor->grad();
    return grad.defined() ? new torch::Tensor(grad) : NULL;
}

void THSTensor_backward(Tensor tensor)
{
    tensor->backward();
}

Tensor THSTensor_cat(const Tensor* tensors, const int length, const int64_t dim)
{
    return new torch::Tensor(torch::cat(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_reshape(const Tensor tensor, const int64_t * shape, const int length)
{
    return new torch::Tensor(tensor->reshape(at::IntList(shape, length)));
}

Tensor THSTensor_stack(const Tensor* tensors, const int length, const int64_t dim)
{
    return new torch::Tensor(torch::stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_t(const Tensor tensor)
{
    return new torch::Tensor(tensor->t());
}

Tensor THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    return new torch::Tensor(tensor->transpose(dim1, dim2));
}

void THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    tensor->transpose_(dim1, dim2);
}

Tensor THSTensor_view(const Tensor tensor, const int64_t * shape, const int length)
{
    return new torch::Tensor(tensor->view(at::IntList(shape, length)));
}

Tensor THSTensor_add(const Tensor left, const int value, const Tensor right)
{
    return new torch::Tensor(left->add(*right, value));
}

void THSTensor_add_(const Tensor left, const int value, const Tensor right)
{
    left->add_(*right, value);
}

Tensor THSTensor_addbmm(
    const Tensor mat,
    const Tensor batch1,
    const Tensor batch2,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(mat->addbmm(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_addmm(
    const Tensor mat,
    const Tensor mat1,
    const Tensor mat2,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(mat->addmm(*mat1, *mat2, beta, alpha));
}

Tensor THSTensor_argmax(const Tensor tensor, const int64_t dimension, bool keepDim)
{
    return new torch::Tensor(tensor->argmax(dimension, keepDim));
}

Tensor THSTensor_baddbmm(
    const Tensor batch1,
    const Tensor batch2,
    const Tensor mat,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(batch1->baddbmm(*batch2, *mat, beta, alpha));
}

Tensor THSTensor_bmm(const Tensor batch1, const Tensor batch2)
{
    return new torch::Tensor(batch1->bmm(*batch2));
}

Tensor THSTensor_eq(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->eq(*right));
}

bool THSTensor_equal(const Tensor left, const Tensor right)
{
    return left->equal(*right);
}

Tensor THSTensor_exp(const Tensor tensor)
{
    return new torch::Tensor(tensor->exp());
}

Tensor THSTensor_matmul(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->matmul(*right));
}

Tensor THSTensor_mean(const Tensor tensor)
{
    return new torch::Tensor(tensor->mean());
}

Tensor THSTensor_mm(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->mm(*right));
}

Tensor THSTensor_mul(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->mul(*right));
}

void THSTensor_mul_(const Tensor left, const Tensor right)
{
    left->mul_(*right);
}

Tensor THSTensor_mulS(const Tensor tensor, const float scalar)
{
    return new torch::Tensor(tensor->mul(scalar));
}

Tensor THSTensor_pow(const Tensor tensor, const float scalar)
{
    return new torch::Tensor(tensor->pow(scalar));
}

Tensor THSTensor_sigmoid(const Tensor tensor)
{
    return new torch::Tensor(tensor->sigmoid());
}

Tensor THSTensor_sub(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->sub(*right));
}

void THSTensor_sub_(const Tensor left, const Tensor right)
{
    left->sub_(*right);
}

Tensor THSTensor_sum(const Tensor tensor)
{
    return new torch::Tensor(tensor->sum());
}
