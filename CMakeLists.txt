cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 89)
project(cuda_practice
        CUDA CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

file(GLOB_RECURSE SOURCES
        "src/kernels/*.cu"
        "src/kernels/*.cuh"
        "src/kernels/*.cpp"
        "src/core/*.cu"
        "src/core/*.cuh"
        "src/core/*.cpp"
        "src/main.cu"
)

add_executable(cuda_forge
        ${SOURCES}
)

target_include_directories(cuda_forge PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_forge
        PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        # CUDA::cublas
        # CUDA::cufft
        # CUDA::curand
)

set_target_properties(cuda_forge PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

#target_compile_options(cuda_forge
#    PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /bigobj>
#)