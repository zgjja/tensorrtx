cmake_minimum_required(VERSION 3.14)

project(
  tensorrtx
  VERSION 0.1
  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE
    "Debug"
    CACHE STRING "build type" FORCE)

option(CUDA_USE_STATIC_CUDA_RUNTIME "use static cuda runtime lib" OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      60
      70
      72
      75
      80
      86
      89)
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(cmake/FindTensorRT.cmake)

set(TensorRT_7_TARGETS mlp lenet)

set(TensorRT_8_TARGETS)

set(TensorRT_10_TARGETS)

set(ALL_TARGETS ${TensorRT_7_TARGETS} ${TensorRT_8_TARGETS}
                ${TensorRT_10_TARGETS})

foreach(sub_dir ${ALL_TARGETS})
  message(STATUS "Add subdirectory: ${sub_dir}")
  add_subdirectory(${sub_dir})
endforeach()
