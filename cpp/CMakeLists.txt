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

# The whole point of this target is that it has no dependencies.
# Anything added here needs a line in docs/decisions/ explaining why.
add_library(tinylab_runtime
    src/runtime.cpp
)

target_include_directories(tinylab_runtime
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# -O3 plus a native target is what auto-vectorizes the contiguous inner loop of a
# naive convolution. Measure before reaching for intrinsics.
target_compile_options(tinylab_runtime PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TINYLAB_PORTABLE}>>>:-march=native>
    -Wall -Wextra -Wpedantic
)

add_executable(tinylab_cli src/main.cpp)
target_link_libraries(tinylab_cli PRIVATE tinylab_runtime)

# Tests are added once there is something to test.
# enable_testing()
# add_subdirectory(tests)
