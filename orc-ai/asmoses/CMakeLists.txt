#
# Master MOSES CMake file.
#
# General organization:
# -- check for different compilers, OS'es
# -- search for various required & optional libraries/tools
# -- decide what to build based on above results.
# -- configure various config files.
# -- print pretty summary
#
# cogutils already requires 2.8.12.2, so may as well ask for that.
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)
IF (COMMAND CMAKE_POLICY)
	CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND CMAKE_POLICY)

IF(CMAKE_VERSION VERSION_GREATER 3.0.2)
   CMAKE_POLICY(SET CMP0037 OLD)
ENDIF(CMAKE_VERSION VERSION_GREATER 3.0.2)

PROJECT(asmoses)

# ----------------------------------------------------------
# User-modifiable options. Feel free to change these!
#
# uncomment to be in Release mode [default]
# SET(CMAKE_BUILD_TYPE Release)

# uncomment to build in debug mode
# SET(CMAKE_BUILD_TYPE Debug)

# uncomment to be in coverage testing mode
# SET(CMAKE_BUILD_TYPE Coverage)

# uncomment to build in profile mode
# SET(CMAKE_BUILD_TYPE Profile)

# uncomment to build in release mode with debug information
# SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

ADD_DEFINITIONS(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                -DPROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}")

# ===============================================================
# MOSES' version, must be updated manually
SET (ASMOSES_VERSION_MAJOR 3)
SET (ASMOSES_VERSION_MINOR 7)
SET (ASMOSES_VERSION_PATCH 0)

ADD_DEFINITIONS(-DASMOSES_VERSION_MAJOR="${ASMOSES_VERSION_MAJOR}")
ADD_DEFINITIONS(-DASMOSES_VERSION_MINOR="${ASMOSES_VERSION_MINOR}")
ADD_DEFINITIONS(-DASMOSES_VERSION_PATCH="${ASMOSES_VERSION_PATCH}")

SET (ASMOSES_VERSION "${ASMOSES_VERSION_MAJOR}.${ASMOSES_VERSION_MINOR}.${ASMOSES_VERSION_PATCH}")

# ===============================================================
# Check for existance of various required, optional packages.
# Listed in alphabetical, more or less.
# CogUtil must come first, because it supplies various FindXXX macros.

# Add the 'lib' dir to cmake's module search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/lib/")

# CogUtil
FIND_PACKAGE(CogUtil CONFIG REQUIRED)
IF (COGUTIL_FOUND)
	MESSAGE(STATUS "CogUtil found.")
	ADD_DEFINITIONS(-DHAVE_COGUTIL)
	SET(HAVE_COGUTIL 1)
ELSE (COGUTIL_FOUND)
	MESSAGE(FATAL_ERROR "CogUtil missing: it is needed!")
ENDIF (COGUTIL_FOUND)

# add the 'cmake' directory from cogutil to search path
list(APPEND CMAKE_MODULE_PATH  ${COGUTIL_DATA_DIR}/cmake)
include(OpenCogGccOptions)
include(OpenCogLibOptions)
include(OpenCogInstallOptions)
include(Summary)

# ===================================================================
# Check for existance of various required, optional packages.

# Check for boost. We need dynamic-linked, threaded libs by default.
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)

# Required boost packages
FIND_PACKAGE(Boost 1.60 COMPONENTS date_time filesystem program_options regex serialization system thread REQUIRED)

IF(Boost_FOUND)
	SET(Boost_FOUND_SAVE 1)
ELSE(Boost_FOUND)
	MESSAGE(FATAL_ERROR "Boost 1.60 or newer is needed to build Moses!")
ENDIF(Boost_FOUND)

# Optional boost packages; can build without these.
FIND_PACKAGE(Boost 1.60 COMPONENTS math_c99 QUIET)

# Arghhh. Except cmake is treating above as required, not optional. #$%**&
IF(Boost_FOUND_SAVE)
	SET(Boost_FOUND 1)
ENDIF(Boost_FOUND_SAVE)

IF(Boost_PROGRAM_OPTIONS_FOUND)
	MESSAGE(STATUS "Found Boost::program_options")
ELSE(Boost_PROGRAM_OPTIONS_FOUND)
	MESSAGE(STATUS "Boost program_options missing: needed for command line.")
ENDIF(Boost_PROGRAM_OPTIONS_FOUND)

IF(Boost_MATH_C99_FOUND)
	MESSAGE(STATUS "Found Boost::math")
ELSE(Boost_MATH_C99_FOUND)
	MESSAGE(STATUS "Boost math missing: needed to run combo learning tests.")
ENDIF(Boost_MATH_C99_FOUND)

# ----------------------------------------------------------
# Needed for unit tests

FIND_PACKAGE(Cxxtest)
IF (CXXTEST_FOUND)
	MESSAGE(STATUS "Found CxxTest.")
ELSE (CXXTEST_FOUND)
	MESSAGE(STATUS "CxxTest missing: needed for unit tests.")
ENDIF (CXXTEST_FOUND)

# ----------------------------------------------------------
# This is required for Guile, Python
include(OpenCogFindGuile)
include(OpenCogFindPython)

# ----------------------------------------------------------
# AtomSpace
FIND_PACKAGE(AtomSpace 5.0.3 CONFIG REQUIRED)
IF (ATOMSPACE_FOUND)
	MESSAGE(STATUS "AtomSpace found.")
	ADD_DEFINITIONS(-DHAVE_ATOMSPACE)
	SET(HAVE_ATOMSPACE 1)
ELSE (ATOMSPACE_FOUND)
	MESSAGE(FATAL_ERROR "AtomSpace missing: it is needed to build this package!")
ENDIF (ATOMSPACE_FOUND)

# Macros that define how atom types get declared.
#IF (NOT DEFINED ATOMSPACE_DATA_DIR)
#	SET (ATOMSPACE_DATA_DIR "${COGUTIL_DATA_DIR}")
#ENDIF (NOT DEFINED ATOMSPACE_DATA_DIR)

INCLUDE("${ATOMSPACE_DATA_DIR}/cmake/OpenCogMacros.cmake")
INCLUDE("${ATOMSPACE_DATA_DIR}/cmake/OpenCogFunctions.cmake")

# ----------------------------------------------------------
# MPI is needed for MOSES on compute clusters.
# FIND_PACKAGE(MPI)

IF (MPI_FOUND)
	ADD_DEFINITIONS(-DHAVE_MPI)
	SET(HAVE_MPI 1)
	MESSAGE(STATUS "MPI was found.")
	INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
ELSE (MPI_FOUND)
	MESSAGE(STATUS "MPI missing: needed for MPI-MOSES.")
ENDIF (MPI_FOUND)

# ----------------------------------------------------------
# URE, for URE-Reduct
FIND_PACKAGE(URE 1.0.0 CONFIG REQUIRED)
IF (URE_FOUND)
	MESSAGE(STATUS "URE found.")
	ADD_DEFINITIONS(-DHAVE_URE)
	SET(HAVE_URE 1)
ELSE (URE_FOUND)
	MESSAGE(STATUS "URE missing: it is needed!")
ENDIF (URE_FOUND)

# ----------------------------------------------------------
# Optional, currently needed only to hush up DRD in util/Logger.cc
FIND_PACKAGE(VALGRIND)
IF (VALGRIND_FOUND)
	MESSAGE(STATUS "VALGRIND was found.")
	IF (VALGRIND_INCLUDE_DIR)
		MESSAGE(STATUS "VALGRIND devel headers found.")
		ADD_DEFINITIONS(-DHAVE_VALGRIND)
	ELSE (VALGRIND_INCLUDE_DIR)
		MESSAGE(STATUS "VALGRIND devel headers NOT FOUND: needed for thread debugging.")
	ENDIF (VALGRIND_INCLUDE_DIR)
ELSE (VALGRIND_FOUND)
	MESSAGE(STATUS "VALGRIND missing: needed for thread debugging.")
ENDIF (VALGRIND_FOUND)

# ===================================================================
# Global includes

# -------------------------------------------------
# Include configuration.

# Set default include paths
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS}
	${COGUTIL_INCLUDE_DIR})

# Macros
INCLUDE("${ATOMSPACE_DATA_DIR}/cmake/OpenCogMacros.cmake")
INCLUDE("${ATOMSPACE_DATA_DIR}/cmake/OpenCogFunctions.cmake")

# -------------------------------------------------
# Library configuration

# Small hack to handle unixes that use "/usr/lib64" instead of
# "/usr/lib" as the # default lib path on 64 bit archs
IF (NOT DEFINED LIB_DIR_SUFFIX)
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs OUTPUT_VARIABLE PRINT_SEARCH_DIRS_OUTPUT)
	STRING(REGEX MATCH "\r?\nlibraries:.*\r?\n" COMPILER_LIB_SEARCH_DIRS ${PRINT_SEARCH_DIRS_OUTPUT})
	IF (NOT ${COMPILER_LIB_SEARCH_DIRS} STREQUAL "")
		STRING(REGEX MATCH "/lib64/:|/lib64:|/lib64\n" HAS_LIB64 ${COMPILER_LIB_SEARCH_DIRS})
		IF (NOT ${HAS_LIB64} STREQUAL "")
			SET(LIB_DIR_SUFFIX "64")
		ENDIF (NOT ${HAS_LIB64} STREQUAL "")
	ENDIF (NOT ${COMPILER_LIB_SEARCH_DIRS} STREQUAL "")
ENDIF (NOT DEFINED LIB_DIR_SUFFIX)

# rpath handling
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/moses")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# -------------------------------------------------
# Install configuration

# Only list install files that have actually changed.
SET(CMAKE_INSTALL_MESSAGE "LAZY")

# Set confdir and datadir
IF (NOT DEFINED CONFDIR)
	SET (CONFDIR "${CMAKE_INSTALL_PREFIX}/etc")
ENDIF (NOT DEFINED CONFDIR)

# DATADIR was "${CMAKE_INSTALL_PREFIX}/share/moses" but since
# OpenCogMacros.cmake in atomspace expects DATADIR to conatain
# "OpenCogAtomTypes.cmake" I am changing it to "../opencog" at
# least for now. XXX FIXME ... what is DATADIR used for, anyway??
IF (NOT DEFINED DATADIR)
#	SET (DATADIR "${CMAKE_INSTALL_PREFIX}/share/moses")
	SET (DATADIR "${CMAKE_INSTALL_PREFIX}/share/opencog")
ENDIF (NOT DEFINED DATADIR)

ADD_DEFINITIONS(-DCONFDIR="${CONFDIR}")
ADD_DEFINITIONS(-DDATADIR="${DATADIR}")

# (re?)define MAN_INSTALL_DIR
SET (MAN_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/man")

# ==========================================================
# Decide what to build, based on the packages found.

IF(Boost_FOUND AND COGUTIL_FOUND AND URE_FOUND)
	SET(HAVE_COMBOREDUCT 1)
	SET(HAVE_MOSES 1)
	SET(HAVE_FEATURE_SELECTION 1)
ENDIF(Boost_FOUND AND COGUTIL_FOUND AND URE_FOUND)

ADD_SUBDIRECTORY(opencog)

IF (CXXTEST_FOUND AND COGUTIL_FOUND)
	enable_testing()
	ADD_CUSTOM_TARGET(tests)
	ADD_SUBDIRECTORY(tests EXCLUDE_FROM_ALL)
	IF (CMAKE_BUILD_TYPE STREQUAL "Coverage")
		# doing coverage stuff while running tests if this is the Coverage build
		ADD_CUSTOM_TARGET(covtest
			# TODO lcov should be found by cmake first
			# TODO set it up so that we can pick to run coverage per test, or
			# combined across all tests (the latter is MUCH faster). Use a define?
			# There is coverage specific stuff in AddCxxTest.cmake now...
			# -
			WORKING_DIRECTORY tests
			COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process $(ARGS)
			COMMENT "Running tests with coverage..."
		)
		ADD_CUSTOM_TARGET(check
			DEPENDS covtest
			WORKING_DIRECTORY ./
			# This script combines the coverage analysis of each test,
			# then creates html in tests/lcov
			# COMMAND genhtml -o ../lcov -t "All AtomSpace unit tests" *.info
			COMMAND ${PROJECT_SOURCE_DIR}/scripts/combine_lcov.sh
			COMMENT "Generating lcov report..."
		)
	ELSE (CMAKE_BUILD_TYPE STREQUAL "Coverage")
		# Coverage is disabled; test normally
		ADD_CUSTOM_TARGET(check
			DEPENDS tests
			WORKING_DIRECTORY tests
			# COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process $(ARGS)
			COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure $(ARGS)
			COMMENT "Running tests..."
		)
	ENDIF (CMAKE_BUILD_TYPE STREQUAL "Coverage")

	# When CMP00037 is finally turned off, just remove these two lines.
	ADD_CUSTOM_TARGET(test)
	ADD_DEPENDENCIES(test check)

	ADD_CUSTOM_TARGET(test_python
		WORKING_DIRECTORY tests/cython
		COMMAND ${CMAKE_CTEST_COMMAND} $(ARGS)
		COMMENT "Running Python and Cython tests..."
	)
ENDIF (CXXTEST_FOUND AND COGUTIL_FOUND)

ADD_SUBDIRECTORY(examples EXCLUDE_FROM_ALL)

IF (NOT WIN32)
	ADD_CUSTOM_TARGET (asmoses_examples
		# using CMAKE_BUILD_TOOL results in teh cryptic error message
		# warning: jobserver unavailable: using -j1.  Add `+' to parent make rule.
		# This is because make doesn't know how to pass jobserver args to
		# the submake.  So, instead, just use $(MAKE) (with round parens)
		# -- that will do the right thing.
		# COMMAND ${CMAKE_BUILD_TOOL}
		COMMAND $(MAKE)
		WORKING_DIRECTORY examples
		COMMENT "Building examples"
	)
ENDIF (NOT WIN32)

ADD_CUSTOM_TARGET(asmoses_cscope
	COMMAND find moses tests -name '*.cc' -o -name '*.h' -o -name '*.cxxtest' > ${CMAKE_SOURCE_DIR}/cscope.files
	COMMAND cscope -b
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Generating CScope database"
)

# ===================================================================
# Packaging
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Meta-Optimizing Semantic Evolutionary Search")
SET(CPACK_PACKAGE_NAME "moses")
SET(CPACK_PACKAGE_VENDOR "opencog.org")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "4")
SET(CPACK_BINARY_STGZ "OFF")
SET(CPACK_BINARY_TBZ2 "OFF")
SET(CPACK_BINARY_TGZ "OFF")
SET(CPACK_BINARY_TZ "OFF")
SET(CPACK_SOURCE_STGZ "OFF")
SET(CPACK_SOURCE_TBZ2 "ON")
SET(CPACK_SOURCE_TGZ "OFF")
SET(CPACK_SOURCE_TZ "OFF")
SET(CPACK_GENERATOR "TBZ2")
SET(CPACK_SOURCE_GENERATOR "TBZ2")
SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}")
SET(CPACK_SOURCE_IGNORE_FILES
	"~$"
	".a$"
	".so$"
	".log$"
	".tar.z$"
	".tar.gz$"
	".tar.bz2$"
	"/Makefile$"
	"/CMakeFiles/"
	"/CMakeCache.txt$"
	"/DartTestfile.txt$"
	"/cmake_install.cmake$"
	"/Testfile.cmake$"
	"/CTestTestfile.cmake$"
	"/CTestTestfile.cmake$"
	"UTest$"
	"UTest.cpp$"
	"/_CPack_Packages/"
	"/CPackConfig.cmake$"
	"/CPackSourceConfig.cmake$"
	"/sniff$"
	"/install_manifest.txt$"
	"${CPACK_PACKAGE_FILE_NAME}"
	"^${PROJECT_SOURCE_DIR}/debian/"
	"^${PROJECT_SOURCE_DIR}/.*spec$"
	"^${PROJECT_SOURCE_DIR}/vendor/"
	"^${PROJECT_SOURCE_DIR}/.bzr/"
	"^${PROJECT_SOURCE_DIR}/.bzrignore$"
	"^${PROJECT_SOURCE_DIR}/.git/"
)
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
IF (WIN32)
	# Its "not unix" is cygwin is being used.
	IF (NOT UNIX)
		# There is a bug in NSI that does not handle full unix paths
		# properly. Make sure there is at least one set of four (4)
		# backslashes.
		SET(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}\\\\lib\\\\opencog.ico")
		SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} The Open Cognition Framework")
		SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.opencog.org")
		SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.opencog.org")
		SET(CPACK_NSIS_CONTACT "opencog@googlegroups.com")
		SET(CPACK_NSIS_MODIFY_PATH ON)
	ELSE(NOT UNIX)
		SET(CPACK_STRIP_FILES "bin/moses")
		SET(CPACK_SOURCE_STRIP_FILES "")
	ENDIF(NOT UNIX)
ENDIF(WIN32)

SET(CPACK_PACKAGE_EXECUTABLES "asmoses" "The Moses solver")
INCLUDE(CPack)

# ===================================================================
# documentation
FIND_PACKAGE(Doxygen)
ADD_SUBDIRECTORY(doc EXCLUDE_FROM_ALL)
IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib")
	ADD_SUBDIRECTORY(lib)
ENDIF()

# ===================================================================
# Show a summary of what we found, what we will do.

SUMMARY_ADD("ComboReduct" "Library for reduction of combo program trees" HAVE_COMBOREDUCT)
SUMMARY_ADD("Doxygen" "Code documentation" DOXYGEN_FOUND)
SUMMARY_ADD("Feature selection" "MOSES based feature selection tool" HAVE_FEATURE_SELECTION)
SUMMARY_ADD("MOSES" "Meta-Optimizing Semantic Evolutionary Search" HAVE_MOSES)
SUMMARY_ADD("MPI-MOSES" "Distributed MOSES via MPI" HAVE_MPI)
SUMMARY_ADD("Python bindings" "Python (cython) bindings" HAVE_CYTHON)
SUMMARY_ADD("Python tests" "Python bindings nose tests" HAVE_NOSETESTS)
SUMMARY_ADD("Unit tests" "Unit tests" CXXTEST_FOUND AND COGUTIL_FOUND)
SUMMARY_SHOW()
