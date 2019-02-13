project(LibTorchSharp)

target_sources(LibTorchSharp PUBLIC 
				THSJITModule.h
				THSTensor.h
				Utils.h
				THSJITModule.cpp
				THSTensor.cpp
				Utils.cpp)
