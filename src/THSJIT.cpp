#include "THSJIT.h"

#include "stdafx.h"
#include "utils.h"
#include "THSTensor.h"

// Load a TorchScript module from a file
EXPORT_API(JITModuleWrapper *) JIT_module_load(const char* filename)
{
    auto module = torch::jit::load(filename);

    return new JITModuleWrapper(module);
}

// Get the number of submodules contained into the source module 
EXPORT_API(long) JIT_getNumModules(const JITModuleWrapper * mwrapper)
{
    return mwrapper->module->get_modules().size();
}

// Get the name of the module containg into source at the given index
EXPORT_API(const char*) JIT_getModuleName(const JITModuleWrapper * mwrapper, int index)
{
    auto keys = mwrapper->module->get_modules().keys();

    return makeSharableString(keys[index]);
}

// Get the name of the module containg into source at the given index
EXPORT_API(JITModuleWrapper *) JIT_getModuleFromIndex(const JITModuleWrapper * mwrapper, int index)
{
    auto values = mwrapper->module->get_modules().values();

    return new JITModuleWrapper(values[index].module);
}

// Get the name of the module containg into source at the given index
EXPORT_API(JITModuleWrapper *) JIT_getModuleFromName(const JITModuleWrapper * mwrapper, const char* name)
{
    auto module = mwrapper->module->get_module(name);

    return new JITModuleWrapper(module);
}

EXPORT_API(int) JIT_getNumberOfInputs(const JITModuleWrapper * mwrapper)
{
    auto method = mwrapper->module->find_method("forward");
    auto args = method->getSchema().arguments();
    return args.size();
}

EXPORT_API(int) JIT_getNumberOfOutputs(const JITModuleWrapper * mwrapper)
{
    auto method = mwrapper->module->find_method("forward");
    auto outputs = method->getSchema().returns();
    return outputs.size();
}

EXPORT_API(void *) JIT_getInputType(const JITModuleWrapper * mwrapper, int input)
{
    auto method = mwrapper->module->find_method("forward");
    auto args = method->getSchema().arguments();
    auto type = args[input].type();

    return JIT_getType(type);
}

EXPORT_API(void *) JIT_getOutputType(const JITModuleWrapper * mwrapper, int index)
{
    auto method = mwrapper->module->find_method("forward");
    auto outputs = method->getSchema().returns();
    auto type = outputs[index].type();

    return JIT_getType(type);
}

void * JIT_getType(const c10::TypePtr type)
{
    switch (type->kind())
    {
    case c10::TypeKind::DynamicType:
        return new JITDynamicTypeWrapper(type->cast<c10::DynamicType>());
    case c10::TypeKind::TensorType:
        return new JITTensorTypeWrapper(type->cast<c10::TensorType>());
    default:
        return NULL;
    }
}

EXPORT_API(bool) JIT_TypeKind(const JITTypeWrapper * twrapper)
{
    return twrapper->enumType;
}

EXPORT_API(int8_t) JIT_TensorType_getScalar(const JITTensorTypeWrapper * ttwrapper)
{
    return (int8_t)ttwrapper->type->scalarType();
}

EXPORT_API(int) JIT_TensorType_getDimensions(const JITTensorTypeWrapper * ttwrapper)
{
    return ttwrapper->type->dim();
}

// Get the name of the module containg into source at the given index
EXPORT_API(const char *) JIT_TensorType_getDevice(const JITTensorTypeWrapper * ttwrapper)
{
    auto device = ttwrapper->type->device();

    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return makeSharableString(device_type);
}

// Forward pass over the input module using the input tensor
EXPORT_API(TensorWrapper *) JIT_forward(const JITModuleWrapper * mwrapper, const TensorWrapper ** twrapper, const int length)
{
    std::vector<c10::IValue> tensors;

    for (int i = 0; i < length; i++)
    {
        tensors.push_back(twrapper[i]->tensor);
    }

    auto result = mwrapper->module->forward(tensors);

    return new TensorWrapper(result.toTensor());
}

// Dispose the Module.
EXPORT_API(void) JIT_Module_Dispose(const JITModuleWrapper * mwrapper)
{
    delete mwrapper;
}

// Dispose the Type.
EXPORT_API(void) JIT_Type_Dispose(const JITTypeWrapper * twrapper)
{
    delete twrapper;
}