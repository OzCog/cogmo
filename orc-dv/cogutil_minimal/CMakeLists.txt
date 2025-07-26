#
# Minimal Foundation Layer: Cogutil Implementation
# Tensor Architecture: [512, 128, 8] = 524,288 DOF
# Cognitive Function: utility-primitives
#
cmake_minimum_required(VERSION 3.16)
project(cogutil_minimal VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Cogutil minimal build type: ${CMAKE_BUILD_TYPE}")

# Foundation Layer tensor configuration
set(COGUTIL_TENSOR_SHAPE_X 512 CACHE STRING "Tensor X dimension")
set(COGUTIL_TENSOR_SHAPE_Y 128 CACHE STRING "Tensor Y dimension") 
set(COGUTIL_TENSOR_SHAPE_Z 8 CACHE STRING "Tensor Z dimension")
math(EXPR COGUTIL_DEGREES_OF_FREEDOM "${COGUTIL_TENSOR_SHAPE_X} * ${COGUTIL_TENSOR_SHAPE_Y} * ${COGUTIL_TENSOR_SHAPE_Z}")

message(STATUS "Foundation Layer Cogutil Configuration:")
message(STATUS "  Cognitive Function: utility-primitives")
message(STATUS "  Tensor Shape: [${COGUTIL_TENSOR_SHAPE_X}, ${COGUTIL_TENSOR_SHAPE_Y}, ${COGUTIL_TENSOR_SHAPE_Z}]")
message(STATUS "  Degrees of Freedom: ${COGUTIL_DEGREES_OF_FREEDOM}")

# Validate tensor requirements as per issue specification
if(NOT COGUTIL_DEGREES_OF_FREEDOM EQUAL 524288)
    message(FATAL_ERROR "Foundation Layer validation failed: Tensor DOF must be 524,288, got ${COGUTIL_DEGREES_OF_FREEDOM}")
endif()

# Required packages
find_package(Threads REQUIRED)

# Add preprocessor definitions
add_definitions(
    -DCOGUTIL_VERSION_MAJOR=1
    -DCOGUTIL_VERSION_MINOR=0
    -DCOGUTIL_VERSION_PATCH=0
    -DCOGUTIL_TENSOR_SHAPE_X=${COGUTIL_TENSOR_SHAPE_X}
    -DCOGUTIL_TENSOR_SHAPE_Y=${COGUTIL_TENSOR_SHAPE_Y}
    -DCOGUTIL_TENSOR_SHAPE_Z=${COGUTIL_TENSOR_SHAPE_Z}
    -DCOGUTIL_DEGREES_OF_FREEDOM=${COGUTIL_DEGREES_OF_FREEDOM}
)

# Source files
set(COGUTIL_SOURCES
    src/cogutil_minimal.cc
    src/tensor_utils_minimal.cc
)

set(COGUTIL_HEADERS
    include/cogutil_minimal.h
    include/tensor_utils_minimal.h
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the core library
add_library(cogutil_minimal SHARED ${COGUTIL_SOURCES})

# Link with threads
target_link_libraries(cogutil_minimal ${CMAKE_THREAD_LIBS_INIT})

# Install library and headers
install(TARGETS cogutil_minimal
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${COGUTIL_HEADERS}
    DESTINATION include/opencog/util
)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_cogutil_minimal tests/test_main.cc)
target_link_libraries(test_cogutil_minimal cogutil_minimal)

# Add the test
add_test(NAME cogutil_tensor_test COMMAND test_cogutil_minimal)

# Generate summary
message(STATUS "")
message(STATUS "=========================================")
message(STATUS "Cogutil Minimal Foundation Layer:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Tensor Shape: [${COGUTIL_TENSOR_SHAPE_X}, ${COGUTIL_TENSOR_SHAPE_Y}, ${COGUTIL_TENSOR_SHAPE_Z}]")
message(STATUS "  Degrees of Freedom: ${COGUTIL_DEGREES_OF_FREEDOM}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "=========================================")
message(STATUS "")