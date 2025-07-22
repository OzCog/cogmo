cmake_minimum_required(VERSION 3.16)
project(cogml LANGUAGES C CXX)

# 1. Assert no git submodules (optional, but highly recommended)
execute_process(
    COMMAND git submodule status
    OUTPUT_VARIABLE submodules_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT "${submodules_out}" STREQUAL "")
    message(FATAL_ERROR "Submodules detected! This project must be a pure monorepo.")
endif()

# Meta-validation: Explicitly fail if .gitmodules or submodules exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules")
    message(FATAL_ERROR "Monorepo policy violation: .gitmodules file detected.")
endif()

# 2. Dynamically discover all local modules (directories with CMakeLists.txt)
file(GLOB_RECURSE cmake_files "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
set(COGML_MODULES)

foreach(cmake_file ${cmake_files})
    get_filename_component(module_dir ${cmake_file} DIRECTORY)
    get_filename_component(module_name ${module_dir} NAME)
    
    # Only include top-level directories (not nested ones)
    get_filename_component(parent_dir ${module_dir} DIRECTORY)
    if(parent_dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        list(APPEND COGML_MODULES ${module_name})
    endif()
endforeach()

# Remove duplicates and sort
list(REMOVE_DUPLICATES COGML_MODULES)
list(SORT COGML_MODULES)

message(STATUS "Discovered modules with CMakeLists.txt: ${COGML_MODULES}")

# Exclude modules with known dependency issues for now
set(EXCLUDED_MODULES
    cogserver           # missing CogUtil
    gnucashcog-v3      # missing GnuCash CMake modules and glib-2.0
    orc-as             # missing CogUtil dependency 
    orc-oc             # missing OpenCog dependencies
    robotics-embodied  # missing CogUtil and AtomSpace
)

# Filter out excluded modules
foreach(excluded ${EXCLUDED_MODULES})
    list(REMOVE_ITEM COGML_MODULES ${excluded})
endforeach()

message(STATUS "Building modules: ${COGML_MODULES}")
message(STATUS "Excluded modules (missing dependencies): ${EXCLUDED_MODULES}")

foreach(module ${COGML_MODULES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${module}/CMakeLists.txt")
        message(STATUS "Adding subdirectory: ${module}")
        add_subdirectory(${module})
    endif()
endforeach()

# 3. Install global includes/resources (only if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY include/ DESTINATION include)
endif()

# 4. Enable testing if present
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()