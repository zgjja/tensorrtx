cmake_minimum_required(VERSION 3.14)

project(
  tensorrtx
  VERSION 0.1
  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 86 89 100 120)

set(TensorRT_7_8_10_TARGETS
    alexnet
    # csrnet
    googlenet
    lenet
    # lprnet
    mlp
    # mnasnet
    
    # # # mobilenet
    # resnet
    # # # sam3
    # shufflenetv2
    # squeezenet
    
    # # # superpoint
    # vgg
    # # inception
    # # retinaface
    # efficientnet
    # repvgg
    # unet
)
 # vit inception/inceptionv3 unet densenet resnet
set(TensorRT_8_10_TARGETS)

set(TensorRT_10_TARGETS)

set(ALL_TARGETS ${TensorRT_7_8_10_TARGETS} ${TensorRT_8_10_TARGETS}
                ${TensorRT_10_TARGETS})

foreach(sub_dir ${ALL_TARGETS})
  message(STATUS "Add subdirectory: ${sub_dir}")
  add_subdirectory(${sub_dir})
endforeach()
