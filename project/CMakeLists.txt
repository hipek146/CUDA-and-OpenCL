cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(vec_mat_alg VERSION 0.01 LANGUAGES CXX CUDA)

file(GLOB SOURCE_FILES
	"include/*.h"
	"src/*.cu"
)

set(INCLUDE_LS include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

include_directories(${INCLUDE_LS})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_LS})
