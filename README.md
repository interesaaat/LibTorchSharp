# libTorchSharp

## Building instruction

### Requirements
* cmake => 3.13
* libtorch download [here](https://pytorch.org/get-started/locally/)
* Python source code win32 compiled

### How to generate project files
 `cmake -DTorch_DIR="YOUR_libtorch_DIR" -DPython_DIR="YOUR_Python_source_DIR" . -G "Visual Studio 15 2017 Win64"`

### How to build
* Open the generated solution file, click on Properties on the LibTorch project and select Dynamic Library instead of Static in the Configuration type tab (remeber to also change the target extension).
* Build the solution
* (In case VS complains about some Python lib not in the path, go into Properties, Linked, Include, and manually include those libs)

### Tested on
* Windows machine
* Python 3.6.8
* libtorch 1.0
* Visual Studio 2017 (v141, Windows SDK 10.0.17134.0)
