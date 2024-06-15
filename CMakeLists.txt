cmake_minimum_required(VERSION 3.28)
project(cuda_particles CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(cuda_particles main.cu)

set_target_properties(cuda_particles PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
