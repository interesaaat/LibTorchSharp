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

// Forward pass over the input module using the input tensor
EXPORT_API(TensorWrapper *) JIT_forward(const JITModuleWrapper * mwrapper, const TensorWrapper * twrapper)
{
    auto result = mwrapper->module->forward({ twrapper->tensor });

    return new TensorWrapper(result.toTensor());
}