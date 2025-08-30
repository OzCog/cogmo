# Foundation Layer - Core utilities - CogUtil

## Overview

This directory provides the top-level interface to the Foundation Layer CogUtil implementation. The actual implementation is located in `orc-dv/cogutil_minimal/` and provides:

**Tensor Architecture**: [512, 128, 8] = 524,288 DOF  
**Cognitive Function**: utility-primitives  
**Implementation Status**: ✅ Complete and Validated

## Quick Start

### Building from Root

```bash
# From the repository root
mkdir -p build && cd build
cmake ..
make -j4

# Run cogutil tests
make test
```

### Direct Usage

```bash
# Build and test the cogutil implementation directly
cd cogutil
mkdir -p build && cd build
cmake ..
make -j4
./cogutil_minimal_build/test_cogutil_minimal
```

## Integration

This interface module:

1. **Exposes** the cogutil_minimal implementation as the standard `cogutil` target
2. **Provides** CMake configuration for `find_package(cogutil)` support  
3. **Installs** headers to standard OpenCog locations (`include/opencog/util/`)
4. **Enables** testing integration with the main build system

## Implementation Details

The actual Foundation Layer CogUtil implementation is in:
- **Source**: `orc-dv/cogutil_minimal/`
- **Documentation**: `orc-dv/cogutil_minimal/README.md`
- **Tests**: 34 comprehensive tests with 100% pass rate
- **Features**: Tensor operations, cognitive primitives, recursive implementations

## Foundation Layer Compliance

✅ **Task 1**: Basic C++/CMake build infrastructure  
✅ **Task 2**: Core utility functions and data structures  
✅ **Task 3**: Comprehensive unit test suite  
✅ **Task 4**: Memory management patterns  
✅ **Task 5**: API documentation and usage examples  
✅ **Task 6**: CI/CD pipeline integration  
✅ **Task 7**: Thread safety and performance validation  

The Foundation Layer CogUtil provides essential tensor-based cognitive primitives for the OpenCog architecture, serving as the foundation for all higher-level cognitive operations.

---

**Implementation**: Foundation Layer compliant  
**Status**: ✅ Complete and Validated  
**Architecture**: Tensor-based [512, 128, 8] = 524,288 DOF  
**Function**: utility-primitives