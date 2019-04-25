
|**Linux**|[![Build Status](https://dev.azure.com/mainterl/LibTorchSharp/_apis/build/status/interesaaat.LibTorchSharp?branchName=master)](https://dev.azure.com/mainterl/LibTorchSharp/_build/latest?definitionId=1&branchName=master)|
|:---|----------------:|

# libTorchSharp

## Building instruction

### Requirements
* cmake
* libtorch download [here](https://pytorch.org/get-started/locally/)

### How to generate project files
 * Windows: `cmake -DTorch_DIR="YOUR_libtorch_DIR\share\cmake\Torch" . -G "Visual Studio 15 2017 Win64"`
 * Linux: `cmake -DTorch_DIR="YOUR_libtorch_DIR/share/cmake/Torch" .`
 * For CUDA support add `-DCMAKE_BUILD_TYPE="Release"` to the commands above.

### How to build in Windows
* Open the generated solution file, click on Properties on the LibTorch project and select Dynamic Library instead of Static in the Configuration type tab (remeber to also change the target extension).
* Build the solution
* (In case VS complains about some Python lib not in the path, go into Properties, Linked, Include, and manually include those libs)

### How to build in Linux
* Just type `make`. The `libTorchSharp.so` file will be generated in the project location.

### Tested on
* Windows 10 machine (with Linux subsystem)
* Ubuntu 18.04
* libtorch 1.0 and 1.0.1
* Visual Studio 2017 (v141, Windows SDK 10.0.17134.0)
