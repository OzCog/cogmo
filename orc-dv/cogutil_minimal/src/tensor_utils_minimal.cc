/*
 * src/tensor_utils_minimal.cc
 *
 * Foundation Layer: Minimal Tensor Utilities Implementation
 * Cognitive Function: utility-primitives
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#include "tensor_utils_minimal.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace opencog { namespace util {

// Static member initialization
size_t TensorUtils::tensor_count_ = 0;

// ===================================================================
// TensorUtils Implementation
// ===================================================================

void TensorUtils::initializeTensor(TensorPtr tensor, float init_value) {
    if (!validateTensorShape(tensor)) {
        throw std::invalid_argument("Invalid tensor shape for Foundation Layer");
    }
    
    // Recursive initialization with tensor-aware patterns
    recursiveInitialize(*tensor, 0, tensor->size(), init_value, 0);
}

void TensorUtils::recursiveInitialize(TensorData& tensor, size_t start_idx, 
                                     size_t end_idx, float value, size_t depth) {
    const size_t threshold = 1024; // Recursive threshold
    
    if (end_idx - start_idx <= threshold || depth >= 8) {
        // Base case: direct initialization
        std::fill(tensor.begin() + start_idx, tensor.begin() + end_idx, value);
        return;
    }
    
    // Recursive case: divide and conquer
    size_t mid = start_idx + (end_idx - start_idx) / 2;
    
    // Create threads for parallel recursive initialization
    std::thread left_thread([&]() {
        recursiveInitialize(tensor, start_idx, mid, value, depth + 1);
    });
    
    std::thread right_thread([&]() {
        recursiveInitialize(tensor, mid, end_idx, value, depth + 1);
    });
    
    left_thread.join();
    right_thread.join();
}

void TensorUtils::tensorAdd(const TensorPtr& a, const TensorPtr& b, TensorPtr& result) {
    if (!validateTensorShape(a) || !validateTensorShape(b) || !validateTensorShape(result)) {
        throw std::invalid_argument("Invalid tensor shapes for addition");
    }
    
    const size_t size = TensorShape3D::DOF;
    for (size_t i = 0; i < size; ++i) {
        (*result)[i] = (*a)[i] + (*b)[i];
    }
}

void TensorUtils::tensorMultiply(const TensorPtr& a, const TensorPtr& b, TensorPtr& result) {
    if (!validateTensorShape(a) || !validateTensorShape(b) || !validateTensorShape(result)) {
        throw std::invalid_argument("Invalid tensor shapes for multiplication");
    }
    
    const size_t size = TensorShape3D::DOF;
    for (size_t i = 0; i < size; ++i) {
        (*result)[i] = (*a)[i] * (*b)[i];
    }
}

float TensorUtils::tensorDotProduct(const TensorPtr& a, const TensorPtr& b) {
    if (!validateTensorShape(a) || !validateTensorShape(b)) {
        throw std::invalid_argument("Invalid tensor shapes for dot product");
    }
    
    return std::inner_product(a->begin(), a->end(), b->begin(), 0.0f);
}

void TensorUtils::tensorNormalize(TensorPtr& tensor) {
    if (!validateTensorShape(tensor)) {
        throw std::invalid_argument("Invalid tensor shape for normalization");
    }
    
    // Calculate L2 norm
    float norm = 0.0f;
    for (const auto& value : *tensor) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-8f) { // Avoid division by zero
        float inv_norm = 1.0f / norm;
        for (auto& value : *tensor) {
            value *= inv_norm;
        }
    }
}

std::vector<TensorUtils::TensorPtr> TensorUtils::decomposeTensor(const TensorPtr& tensor, size_t depth) {
    if (!validateTensorShape(tensor) || depth == 0) {
        return {tensor};
    }
    
    std::vector<TensorPtr> components;
    const size_t component_size = TensorShape3D::DOF / (1 << depth); // 2^depth components
    const size_t num_components = 1 << depth;
    
    for (size_t i = 0; i < num_components; ++i) {
        auto component = createTensor();
        size_t start_idx = i * component_size;
        size_t end_idx = std::min(start_idx + component_size, TensorShape3D::DOF);
        
        // Copy data to component (recursive decomposition)
        std::copy(tensor->begin() + start_idx, tensor->begin() + end_idx, 
                  component->begin());
        
        components.push_back(component);
    }
    
    return components;
}

TensorUtils::TensorPtr TensorUtils::composeTensors(const std::vector<TensorPtr>& tensors) {
    if (tensors.empty()) {
        return createTensor();
    }
    
    auto result = createTensor();
    size_t current_idx = 0;
    
    for (const auto& tensor : tensors) {
        if (!validateTensorShape(tensor)) continue;
        
        size_t copy_size = std::min(tensor->size(), TensorShape3D::DOF - current_idx);
        std::copy(tensor->begin(), tensor->begin() + copy_size,
                  result->begin() + current_idx);
        
        current_idx += copy_size;
        if (current_idx >= TensorShape3D::DOF) break;
    }
    
    return result;
}

void TensorUtils::parallelTensorOperation(TensorPtr& tensor, 
                                         std::function<void(float&, size_t)> operation) {
    if (!validateTensorShape(tensor)) {
        throw std::invalid_argument("Invalid tensor shape for parallel operation");
    }
    
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = TensorShape3D::DOF / num_threads;
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? TensorShape3D::DOF : start + chunk_size;
        
        threads.emplace_back([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                operation((*tensor)[i], i);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

bool TensorUtils::validateTensorPerformance(const TensorPtr& tensor) {
    if (!validateTensorShape(tensor)) return false;
    
    // Basic performance validation: all elements should be accessible
    volatile float sum = 0.0f;
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        sum += (*tensor)[i];
    }
    
    return true; // If we reach here, tensor is accessible
}

// ===================================================================
// CognitivePrimitives Implementation
// ===================================================================

void CognitivePrimitives::spatialTransform(CognitivePrimitives::TensorPtr& tensor, float x, float y, float z) {
    if (!TensorUtils::validateTensorShape(tensor)) {
        throw std::invalid_argument("Invalid tensor for spatial transform");
    }
    
    // Apply 3D spatial transformation to tensor
    const size_t stride = TensorShape3D::Y * TensorShape3D::Z;
    
    for (size_t i = 0; i < TensorShape3D::X; ++i) {
        for (size_t j = 0; j < TensorShape3D::Y; ++j) {
            for (size_t k = 0; k < TensorShape3D::Z; ++k) {
                size_t idx = i * stride + j * TensorShape3D::Z + k;
                
                // Apply spatial transformation
                float spatial_factor = std::cos(x * i / TensorShape3D::X) * 
                                      std::sin(y * j / TensorShape3D::Y) * 
                                      std::exp(-z * k / TensorShape3D::Z);
                
                (*tensor)[idx] *= spatial_factor;
            }
        }
    }
}

std::array<float, 3> CognitivePrimitives::extractSpatialCoordinates(const CognitivePrimitives::TensorPtr& tensor, size_t index) {
    if (!TensorUtils::validateTensorShape(tensor) || index >= TensorShape3D::DOF) {
        return {0.0f, 0.0f, 0.0f};
    }
    
    // Convert linear index to 3D coordinates
    const size_t stride_y = TensorShape3D::Z;
    const size_t stride_x = TensorShape3D::Y * TensorShape3D::Z;
    
    size_t x = index / stride_x;
    size_t y = (index % stride_x) / stride_y;
    size_t z = index % stride_y;
    
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

void CognitivePrimitives::temporalShift(CognitivePrimitives::TensorPtr& tensor, int time_delta) {
    if (!TensorUtils::validateTensorShape(tensor)) {
        throw std::invalid_argument("Invalid tensor for temporal shift");
    }
    
    // Simple temporal shift by rotating elements
    if (time_delta == 0) return;
    
    auto temp_tensor = TensorUtils::createTensor();
    *temp_tensor = *tensor;
    
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        size_t shifted_idx = (i + time_delta + TensorShape3D::DOF) % TensorShape3D::DOF;
        (*tensor)[shifted_idx] = (*temp_tensor)[i];
    }
}

float CognitivePrimitives::semanticSimilarity(const CognitivePrimitives::TensorPtr& a, const CognitivePrimitives::TensorPtr& b) {
    if (!TensorUtils::validateTensorShape(a) || !TensorUtils::validateTensorShape(b)) {
        return 0.0f;
    }
    
    // Compute cosine similarity for semantic comparison
    float dot_product = TensorUtils::tensorDotProduct(a, b);
    
    // Compute norms
    float norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        norm_a += (*a)[i] * (*a)[i];
        norm_b += (*b)[i] * (*b)[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

bool CognitivePrimitives::validateLogicalConsistency(const CognitivePrimitives::TensorPtr& tensor) {
    if (!TensorUtils::validateTensorShape(tensor)) {
        return false;
    }
    
    // Basic logical consistency: check for NaN or infinite values
    for (const auto& value : *tensor) {
        if (std::isnan(value) || std::isinf(value)) {
            return false;
        }
    }
    
    return true;
}

void CognitivePrimitives::recursiveCognitiveFold(CognitivePrimitives::TensorPtr& tensor, size_t recursion_depth) {
    if (!TensorUtils::validateTensorShape(tensor) || recursion_depth == 0) {
        return;
    }
    
    // Recursive folding operation for cognitive processing
    for (size_t depth = 0; depth < recursion_depth; ++depth) {
        const size_t current_fold_size = TensorShape3D::DOF / (1 << (depth + 1));
        
        for (size_t i = 0; i < current_fold_size; ++i) {
            // Combine pairs of elements recursively
            (*tensor)[i] = ((*tensor)[2 * i] + (*tensor)[2 * i + 1]) * 0.5f;
        }
    }
}

CognitivePrimitives::TensorPtr CognitivePrimitives::recursiveCognitiveExpand(const CognitivePrimitives::TensorPtr& seed, size_t expansion_factor) {
    if (!TensorUtils::validateTensorShape(seed) || expansion_factor == 0) {
        return seed;
    }
    
    auto expanded = TensorUtils::createTensor();
    TensorUtils::initializeTensor(expanded, 0.0f);
    
    // Recursive expansion with cognitive patterns
    const size_t base_size = TensorShape3D::DOF / expansion_factor;
    
    for (size_t i = 0; i < base_size && i < TensorShape3D::DOF; ++i) {
        float base_value = (*seed)[i];
        
        // Expand with cognitive variation
        for (size_t j = 0; j < expansion_factor && i * expansion_factor + j < TensorShape3D::DOF; ++j) {
            float variation = std::sin(2.0f * M_PI * j / expansion_factor);
            (*expanded)[i * expansion_factor + j] = base_value * (1.0f + 0.1f * variation);
        }
    }
    
    return expanded;
}

}} // namespace opencog::util