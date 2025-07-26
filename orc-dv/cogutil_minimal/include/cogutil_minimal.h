/*
 * include/cogutil_minimal.h
 *
 * Foundation Layer: Minimal Cogutil Implementation
 * Cognitive Function: utility-primitives  
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#ifndef _COGUTIL_MINIMAL_H
#define _COGUTIL_MINIMAL_H

#include <cstddef>
#include <stdexcept>

#define COGUTIL_MAJOR_VERSION 1
#define COGUTIL_MINOR_VERSION 0
#define COGUTIL_MICRO_VERSION 0

#define COGUTIL_VERSION_STRING "1.0.0"

// Foundation Layer: Tensor Architecture Configuration
// Cognitive Function: utility-primitives
// Tensor Shape: [512, 128, 8] = 524,288 DOF
#ifndef COGUTIL_TENSOR_SHAPE_X
#define COGUTIL_TENSOR_SHAPE_X 512
#endif

#ifndef COGUTIL_TENSOR_SHAPE_Y
#define COGUTIL_TENSOR_SHAPE_Y 128
#endif

#ifndef COGUTIL_TENSOR_SHAPE_Z
#define COGUTIL_TENSOR_SHAPE_Z 8
#endif

#ifndef COGUTIL_DEGREES_OF_FREEDOM
#define COGUTIL_DEGREES_OF_FREEDOM (COGUTIL_TENSOR_SHAPE_X * COGUTIL_TENSOR_SHAPE_Y * COGUTIL_TENSOR_SHAPE_Z)
#endif

// Compile-time tensor validation
static_assert(COGUTIL_DEGREES_OF_FREEDOM == 524288, 
              "Cogutil tensor DOF must equal 524,288 as per Foundation Layer specification");

// Foundation Layer tensor types for cognitive operations
namespace opencog { namespace util {
    
    // Tensor shape specification
    struct TensorShape3D {
        static constexpr size_t X = COGUTIL_TENSOR_SHAPE_X;
        static constexpr size_t Y = COGUTIL_TENSOR_SHAPE_Y; 
        static constexpr size_t Z = COGUTIL_TENSOR_SHAPE_Z;
        static constexpr size_t DOF = COGUTIL_DEGREES_OF_FREEDOM;
    };
    
    // Cognitive function identifier
    static constexpr const char* COGNITIVE_FUNCTION = "utility-primitives";
    
    // Core utility functions
    void initialize_cogutil();
    void shutdown_cogutil();
    bool validate_tensor_architecture();
    size_t get_tensor_degrees_of_freedom();
    const char* get_cognitive_function();
    
}} // namespace opencog::util

#endif // _COGUTIL_MINIMAL_H