cmake_minimum_required(VERSION 3.17.0)

set(TRT_VERSION
    $ENV{TRT_VERSION}
    CACHE STRING
          "TensorRT version, e.g. \"8.6.1.6\" or \"8.6.1.6+cuda12.0.1.011\"")

# process for different TensorRT version
if(DEFINED TRT_VERSION AND NOT TRT_VERSION STREQUAL "")
  string(REGEX MATCH "([0-9]+)" _match ${TRT_VERSION})
  set(TRT_MAJOR_VERSION "${_match}")
  set(_modules nvinfer nvinfer_plugin)

  if(TRT_MAJOR_VERSION GREATER_EQUAL 8)
    list(APPEND _modules nvinfer_vc_plugin nvinfer_dispatch nvinfer_lean)
  endif()
else()
  message(FATAL_ERROR "Please set a environment variable \"TRT_VERSION\"")
endif()

if(NOT TensorRT_INCLUDE_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_trt_inc_dir "/usr/local/cuda/targets/aarch64-linux/include")
  else()
    set(_trt_inc_dir "/usr/include/x86_64-linux-gnu")
  endif()
  set(TensorRT_INCLUDE_DIR
      ${_trt_inc_dir}
      CACHE PATH "TensorRT_INCLUDE_DIR")
  message(STATUS "TensorRT: ${TensorRT_INCLUDE_DIR}")
endif()

if(NOT TensorRT_LIBRARY_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_trt_lib_dir "/usr/local/cuda/targets/aarch64-linux/lib")
  else()
    set(_trt_lib_dir "/usr/include/x86_64-linux-gnu")
  endif()
  set(TensorRT_LIBRARY_DIR
      ${_trt_lib_dir}
      CACHE PATH "TensorRT_LIBRARY_DIR")
  message(STATUS "TensorRT: ${TensorRT_LIBRARY_DIR}")
endif()

set(TensorRT_LIBRARIES)

foreach(lib IN LISTS _modules)
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    HINTS ${TensorRT_LIBRARY_DIR})

  list(APPEND TensorRT_LIBRARIES ${TensorRT_${lib}_LIBRARY})
endforeach()

message(STATUS "Found TensorRT lib: ${TensorRT_LIBRARIES}")

if(NOT TARGET TensorRT::TensorRT)
  add_library(TensorRT IMPORTED INTERFACE)
  add_library(TensorRT::TensorRT ALIAS TensorRT)
endif()

target_link_libraries(TensorRT INTERFACE ${TensorRT_LIBRARIES})

set_target_properties(
  TensorRT
  PROPERTIES C_STANDARD 17
             CXX_STANDARD 17
             POSITION_INDEPENDENT_CODE ON
             SKIP_BUILD_RPATH TRUE
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$\{ORIGIN\}"
             INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")

unset(TRT_MAJOR_VERSION)
unset(_modules)
