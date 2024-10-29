cmake_minimum_required(VERSION 3.28.0)
project(CUDASepFB LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_SOURCE_DIR}/cpu
        ${CMAKE_SOURCE_DIR}/gpu
        ${CMAKE_SOURCE_DIR}/utils
)

add_library(CUDASepFB_lib_cpu STATIC cpu/choquet.cpp cpu/computation.cpp)
target_include_directories(CUDASepFB_lib_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
add_subdirectory(cpu)

add_executable(CUDASepFB main.cu)

set_target_properties(CUDASepFB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CUDASepFB cuda cudart ${OpenCV_LIBS} CUDASepFB_lib_cpu)