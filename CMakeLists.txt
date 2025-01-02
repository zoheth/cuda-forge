cmake_minimum_required(VERSION 3.18)
project(cuda_practice 
    CUDA CXX
)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_forge
    "src/main.cu"
)

target_include_directories(cuda_forge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

target_compile_options(cuda_forge
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /bigobj>
)