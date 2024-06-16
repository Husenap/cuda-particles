cmake_minimum_required(VERSION 3.28)
project(cuda_particles LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 26)

include("cmake/tetris.cmake")
include("cmake/glfw.cmake")
add_subdirectory("cmake/glad")

add_executable(cuda_particles main.cu)

target_link_libraries(cuda_particles
        glfw
        glad
        tetris)

set_target_properties(cuda_particles PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
