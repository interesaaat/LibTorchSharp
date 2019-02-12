cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(LibTorchSharp)

find_package(Torch REQUIRED)

# Include files into project
add_library(LibTorchSharp src/THSTensor.cpp)
add_subdirectory(src)

# Add Python bindings
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(LibTorchSharp "${TORCH_LIBRARIES}")
set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 11)
