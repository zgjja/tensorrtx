cmake_minimum_required(VERSION 3.17.0)

project(
  lenet
  VERSION 0.1
  LANGUAGES C CXX CUDA)

add_executable(${PROJECT_NAME} lenet.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads CUDA::cudart
                                       TensorRT::TensorRT )
