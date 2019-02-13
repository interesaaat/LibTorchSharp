cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(LibTorchSharp)

find_package(Torch REQUIRED)

# Include files into project
add_library(LibTorchSharp SHARED "")
add_subdirectory(src)

# Add Python includes and libs
include_directories("${PYTHON}/Include")
include_directories("${PYTHON}/PC")
FILE(GLOB PYTHON_LIBRARIES "${PYTHON}/PCbuild/win32/*.lib")

# Add LibTorch libs
FILE(GLOB LIBTORCH_LIBRARIES "${LIBTORCH}/lib/*.lib")

target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES} ${LIBTORCH_LIBRARIES} ${PYTHON_LIBRARIES})
set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 11)
