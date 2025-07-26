/*
 * src/cogutil_minimal.cc
 *
 * Foundation Layer: Minimal Cogutil Implementation
 * Cognitive Function: utility-primitives
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#include "cogutil_minimal.h"
#include <iostream>

namespace opencog { namespace util {

static bool cogutil_initialized = false;

void initialize_cogutil() {
    if (cogutil_initialized) {
        return;
    }
    
    std::cout << "Initializing Foundation Layer Cogutil..." << std::endl;
    std::cout << "  Cognitive Function: " << COGNITIVE_FUNCTION << std::endl;
    std::cout << "  Tensor Shape: [" << TensorShape3D::X << ", " 
              << TensorShape3D::Y << ", " << TensorShape3D::Z << "]" << std::endl;
    std::cout << "  Degrees of Freedom: " << TensorShape3D::DOF << std::endl;
    
    // Validate tensor architecture
    if (!validate_tensor_architecture()) {
        throw std::runtime_error("Foundation Layer tensor architecture validation failed");
    }
    
    cogutil_initialized = true;
    std::cout << "Foundation Layer Cogutil initialized successfully!" << std::endl;
}

void shutdown_cogutil() {
    if (!cogutil_initialized) {
        return;
    }
    
    std::cout << "Shutting down Foundation Layer Cogutil..." << std::endl;
    cogutil_initialized = false;
}

bool validate_tensor_architecture() {
    // Validate tensor shape specification
    bool valid = true;
    
    if (TensorShape3D::X != 512) {
        std::cerr << "Tensor X dimension validation failed: expected 512, got " 
                  << TensorShape3D::X << std::endl;
        valid = false;
    }
    
    if (TensorShape3D::Y != 128) {
        std::cerr << "Tensor Y dimension validation failed: expected 128, got " 
                  << TensorShape3D::Y << std::endl;
        valid = false;
    }
    
    if (TensorShape3D::Z != 8) {
        std::cerr << "Tensor Z dimension validation failed: expected 8, got " 
                  << TensorShape3D::Z << std::endl;
        valid = false;
    }
    
    if (TensorShape3D::DOF != 524288) {
        std::cerr << "Tensor DOF validation failed: expected 524,288, got " 
                  << TensorShape3D::DOF << std::endl;
        valid = false;
    }
    
    return valid;
}

size_t get_tensor_degrees_of_freedom() {
    return TensorShape3D::DOF;
}

const char* get_cognitive_function() {
    return COGNITIVE_FUNCTION;
}

}} // namespace opencog::util