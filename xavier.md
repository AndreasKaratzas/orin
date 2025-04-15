If you are running the `libtorch` test script on NVIDIA Jetson AGX Xavier, then overwrite the `CMakeLists.txt` file with:

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchsc LANGUAGES CXX)

# Explicitly set CUDA paths
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
# Prevent CMake from finding the other CUDA install
set(CUDAToolkit_ROOT /usr/local/cuda-11.4)

# Set CUDA architecture for Jetson AGX Xavier
set(CMAKE_CUDA_ARCHITECTURES 72)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(torchsc torchsc.cpp)
target_link_libraries(torchsc "${TORCH_LIBRARIES}")
set_property(TARGET torchsc PROPERTY CXX_STANDARD 17)

# Windows-specific DLL handling
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torchsc
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
```
