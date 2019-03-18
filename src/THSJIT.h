#pragma once

#include "torch/script.h"

// Struct used to share TorchScript modules
struct JITModuleWrapper
{
    std::shared_ptr<torch::jit::script::Module> module;

    JITModuleWrapper(std::shared_ptr<torch::jit::script::Module> m) : module(m) {}
};

enum TypeKind : int8_t {
#define DEFINE_TYPE(T) T,
    C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

// Struct used to share jit base types
struct JITTypeWrapper
{
    int8_t enumType;
};

struct JITDynamicTypeWrapper : JITTypeWrapper
{
    std::shared_ptr<torch::jit::DynamicType> type;

    JITDynamicTypeWrapper(std::shared_ptr<torch::jit::DynamicType> t) : type(t)
    {
        enumType = (int8_t)TypeKind::DynamicType;
    }
};

struct JITTensorTypeWrapper : JITTypeWrapper
{
    std::shared_ptr<torch::jit::TensorType> type;

    JITTensorTypeWrapper(std::shared_ptr<torch::jit::TensorType> t) : type(t) 
    {
        enumType = (int8_t)TypeKind::TensorType;
    }
};

void * JIT_getType(const c10::TypePtr type);

