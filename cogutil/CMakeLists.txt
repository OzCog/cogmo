#
# Top-level CogUtil Interface
# Foundation Layer - Core utilities - CogUtil
# 
# This serves as the main interface to the Foundation Layer CogUtil implementation
# located in orc-dv/cogutil_minimal/
#
cmake_minimum_required(VERSION 3.16)
project(cogutil VERSION 1.0.0 LANGUAGES CXX)

message(STATUS "=========================================")
message(STATUS "Foundation Layer - Core utilities - CogUtil")
message(STATUS "=========================================")

# Include and build the actual implementation
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../orc-dv/cogutil_minimal cogutil_minimal_build)

# Create aliases for easier access
add_library(cogutil ALIAS cogutil_minimal)

# Expose the headers at a more standard location
file(GLOB COGUTIL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../orc-dv/cogutil_minimal/include/*.h")

install(FILES ${COGUTIL_HEADERS}
    DESTINATION include/opencog/util
)

# Create cogutil-config.cmake for find_package support
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cogutil-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cogutil-config.cmake"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cogutil-config.cmake"
    DESTINATION lib/cmake/cogutil
)

# Enable testing
enable_testing()

# Add the test as a top-level test
add_test(NAME foundation_cogutil_test 
         COMMAND test_cogutil_minimal
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cogutil_minimal_build)

message(STATUS "Foundation Layer CogUtil interface configured")
message(STATUS "  Implementation: orc-dv/cogutil_minimal")
message(STATUS "  Tensor Shape: [512, 128, 8] = 524,288 DOF")
message(STATUS "  Cognitive Function: utility-primitives")
message(STATUS "=========================================")