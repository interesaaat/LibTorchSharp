if(WIN32)
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
endif()
if(UNIX)
    cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
endif()

project(libTorchSharp)

find_package(Torch REQUIRED)

# Include files into project
add_library(libTorchSharp SHARED "")
add_subdirectory(src)

# Add libTorch bindings
include_directories(${TORCH_INCLUDE_DIRS})

target_link_libraries(libTorchSharp ${TORCH_LIBRARIES})
set_property(TARGET libTorchSharp PROPERTY CXX_STANDARD 14)
set_target_properties(libTorchSharp PROPERTIES PREFIX "")
