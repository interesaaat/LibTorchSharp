# libTorchSharp

## Building instruction

### Requirements
* cmake => 3.13
* libtorch download [here](https://pytorch.org/get-started/locally/) (v 1.0)
* Python source code (tried with version 3.6.8)
* Python installation (dlls) in PATH

### How to generate project files
 `cmake -DTorch_DIR="libtorch_DIR\share\cmake\Torch" -DPYTHON_INCLUDE_DIRS="Python_source_DIR\include" . -G "Visual Studio 15 2017 Win64"`

### How to build
Open the generated solution file and build from there

### Tested with
* Python 3.6.8
* libtorch 1.0
* Release build
