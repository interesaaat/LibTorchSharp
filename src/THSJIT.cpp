#include "THSJIT.h"

#include "Utils.h"
#include "THSTensor.h"

JITModuleWrapper * THSJIT_loadModule(const char* filename)
{
    auto module = torch::jit::load(filename);

    return new JITModuleWrapper(module);
}

long THSJIT_getNumModules(const JITModuleWrapper * mwrapper)
{
    return mwrapper->module->get_modules().size();
}

const char* THSJIT_getModuleName(const JITModuleWrapper * mwrapper, const int index)
{
    auto keys = mwrapper->module->get_modules().keys();

    return makeSharableString(keys[index]);
}

JITModuleWrapper * THSJIT_getModuleFromIndex(const JITModuleWrapper * mwrapper, const int index)
{
    auto values = mwrapper->module->get_modules().values();

    return new JITModuleWrapper(values[index].module);
}

JITModuleWrapper * THSJIT_getModuleFromName(const JITModuleWrapper * mwrapper, const char* name)
{
    auto module = mwrapper->module->get_module(name);

    return new JITModuleWrapper(module);
}

int THSJIT_getNumberOfInputs(const JITModuleWrapper * mwrapper)
{
    auto method = mwrapper->module->find_method("forward");
    auto args = method->getSchema().arguments();
    return args.size();
}

int THSJIT_getNumberOfOutputs(const JITModuleWrapper * mwrapper)
{
    auto method = mwrapper->module->find_method("forward");
    auto outputs = method->getSchema().returns();
    return outputs.size();
}

void * THSJIT_getInputType(const JITModuleWrapper * mwrapper, const int n)
{
    auto method = mwrapper->module->find_method("forward");
    auto args = method->getSchema().arguments();
    auto type = args[n].type();

    return THSJIT_getType(type);
}

void * THSJIT_getOutputType(const JITModuleWrapper * mwrapper, const int n)
{
    auto method = mwrapper->module->find_method("forward");
    auto outputs = method->getSchema().returns();
    auto type = outputs[n].type();

    return THSJIT_getType(type);
}

void * THSJIT_getType(const c10::TypePtr type)
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

int8_t THSJIT_typeKind(const JITTypeWrapper * twrapper)
{
    return twrapper->enumType;
}

int8_t THSJIT_getScalarFromTensorType(const JITTensorTypeWrapper * ttwrapper)
{
    return (int8_t)ttwrapper->type->scalarType();
}

int THSJIT_getTensorTypeDimensions(const JITTensorTypeWrapper * ttwrapper)
{
    return ttwrapper->type->dim();
}

const char * THSJIT_getTensorDevice(const JITTensorTypeWrapper * ttwrapper)
{
    auto device = ttwrapper->type->device();

    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return makeSharableString(device_type);
}

TensorWrapper * THSJIT_forward(const JITModuleWrapper * mwrapper, const TensorWrapper ** twrapper, const int length)
{
    std::vector<c10::IValue> tensors;

    for (int i = 0; i < length; i++)
    {
        tensors.push_back(twrapper[i]->tensor);
    }

    auto result = mwrapper->module->forward(tensors);

    return new TensorWrapper(result.toTensor());
}

void THSJIT_moduleDispose(const JITModuleWrapper * mwrapper)
{
    delete mwrapper;
}

void THSJIT_typeDispose(const JITTypeWrapper * twrapper)
{
    delete twrapper;
}