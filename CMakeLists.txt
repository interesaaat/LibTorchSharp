cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(LibTorchSharp)

find_package(Torch REQUIRED)

# Include files into project
add_library(LibTorchSharp SHARED "")
add_subdirectory(src)

# Add Python bindings
include_directories("${Python_DIR}/Include")
include_directories("${Python_DIR}/PC")
FILE(GLOB PYTHON_LIBRARIES "${Python_DIR}/PCbuild/win32/*.lib")

# Add LibTorch bindings
include_directories(${TORCH_INCLUDE_DIRS})

target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 11)
