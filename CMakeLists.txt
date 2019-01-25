cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(LibTorchSharp)

find_package(Torch REQUIRED)

add_library(LibTorchSharp "")
add_subdirectory(src)

target_link_libraries(LibTorchSharp "${TORCH_LIBRARIES}")
set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 11)
