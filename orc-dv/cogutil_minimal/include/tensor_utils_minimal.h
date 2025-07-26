/*
 * include/tensor_utils_minimal.h
 *
 * Foundation Layer: Minimal Tensor Utilities
 * Cognitive Function: utility-primitives  
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#ifndef _TENSOR_UTILS_MINIMAL_H
#define _TENSOR_UTILS_MINIMAL_H

#include "cogutil_minimal.h"
#include <array>
#include <vector>
#include <memory>
#include <functional>

namespace opencog { namespace util {

/**
 * Minimal Tensor Operations for Foundation Layer
 * 
 * Implements essential tensor-based cognitive operations with the specified
 * tensor shape [512, 128, 8] = 524,288 degrees of freedom.
 * These are true recursive implementations, not mock implementations.
 */
class TensorUtils {
public:
    // Foundation Layer tensor type definitions
    using TensorData = std::array<float, TensorShape3D::DOF>;
    using TensorPtr = std::shared_ptr<TensorData>;
    
    // Core tensor operations
    static TensorPtr createTensor();
    static void initializeTensor(TensorPtr tensor, float init_value = 0.0f);
    static bool validateTensorShape(const TensorPtr& tensor);
    
    // Basic tensor arithmetic
    static void tensorAdd(const TensorPtr& a, const TensorPtr& b, TensorPtr& result);
    static void tensorMultiply(const TensorPtr& a, const TensorPtr& b, TensorPtr& result);
    static float tensorDotProduct(const TensorPtr& a, const TensorPtr& b);
    static void tensorNormalize(TensorPtr& tensor);
    
    // Recursive tensor operations
    static std::vector<TensorPtr> decomposeTensor(const TensorPtr& tensor, size_t depth = 1);
    static TensorPtr composeTensors(const std::vector<TensorPtr>& tensors);
    
    // Memory management
    static size_t getTensorMemoryFootprint();
    static bool validateTensorPerformance(const TensorPtr& tensor);
    
    // Parallel operations
    static void parallelTensorOperation(TensorPtr& tensor, 
                                       std::function<void(float&, size_t)> operation);
    
private:
    static void recursiveInitialize(TensorData& tensor, size_t start_idx, 
                                   size_t end_idx, float value, size_t depth);
    static size_t tensor_count_;
};

/**
 * Cognitive Primitives for Foundation Layer
 */
class CognitivePrimitives {
public:
    // Use the same tensor type as TensorUtils
    using TensorPtr = TensorUtils::TensorPtr;
    
    // Spatial operations (3D cognitive space)
    static void spatialTransform(TensorPtr& tensor, float x, float y, float z);
    static std::array<float, 3> extractSpatialCoordinates(const TensorPtr& tensor, size_t index);
    
    // Temporal operations (cognitive time series)
    static void temporalShift(TensorPtr& tensor, int time_delta);
    
    // Semantic operations (concept embeddings)
    static float semanticSimilarity(const TensorPtr& a, const TensorPtr& b);
    
    // Logical operations (inference states)
    static bool validateLogicalConsistency(const TensorPtr& tensor);
    
    // Recursive cognitive operations
    static void recursiveCognitiveFold(TensorPtr& tensor, size_t recursion_depth = 3);
    static TensorPtr recursiveCognitiveExpand(const TensorPtr& seed, size_t expansion_factor = 2);
};

// Inline implementations for performance
inline TensorUtils::TensorPtr TensorUtils::createTensor() {
    tensor_count_++;
    return std::make_shared<TensorData>();
}

inline bool TensorUtils::validateTensorShape(const TensorPtr& tensor) {
    return tensor && tensor->size() == TensorShape3D::DOF;
}

inline size_t TensorUtils::getTensorMemoryFootprint() {
    return TensorShape3D::DOF * sizeof(float);
}

}} // namespace opencog::util

#endif // _TENSOR_UTILS_MINIMAL_H