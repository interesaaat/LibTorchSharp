project(LibTorchSharp)

target_sources(LibTorchSharp PUBLIC 
				THSData.h
				THSJIT.h
				THSNN.h
				THSTensor.h
				Utils.h
				THSData.cpp
				THSJIT.cpp
				THSNN.cpp
				THSTensor.cpp
				Utils.cpp)
