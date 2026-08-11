cmake_minimum_required(VERSION 3.24)

project(tinylab
    VERSION 0.0.0
    DESCRIPTION "Hand-written inference runtime for very small models"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# No dependencies, by design. Adding one needs an ADR.
add_library(tinylab_runtime
    src/runtime.cpp
)

target_include_directories(tinylab_runtime
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# -O3 plus -march=native auto-vectorizes the naive convolution inner loop.
target_compile_options(tinylab_runtime PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TINYLAB_PORTABLE}>>>:-march=native>
    -Wall -Wextra -Wpedantic
)

add_executable(tinylab_cli src/main.cpp)
target_link_libraries(tinylab_cli PRIVATE tinylab_runtime)
