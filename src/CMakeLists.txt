project(libTorchSharp)

target_sources(libTorchSharp PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR}//THSAutograd.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSData.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSJIT.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSNN.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSTensor.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSTorch.h
				${CMAKE_CURRENT_SOURCE_DIR}//Utils.h
				${CMAKE_CURRENT_SOURCE_DIR}//THSAutograd.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//THSData.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//THSJIT.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//THSNN.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//THSTensor.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//THSTorch.cpp
				${CMAKE_CURRENT_SOURCE_DIR}//Utils.cpp)
