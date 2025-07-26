/*
 * opencog/util/tensor_utils.cc
 *
 * Foundation Layer: Tensor Utilities Implementation
 * Cognitive Function: utility-primitives
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#include <opencog/util/tensor_utils.h>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <immintrin.h> // For AVX2 support where available
#include <cmath>
#include <cstdlib>     // For posix_memalign

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace opencog { namespace util {

// Static member initialization
std::mutex TensorUtils::tensor_mutex_;
std::atomic<size_t> TensorUtils::tensor_count_{0};

std::unique_ptr<uint8_t[]> TensorAllocator::memory_pool_;
std::atomic<size_t> TensorAllocator::pool_offset_{0};
size_t TensorAllocator::pool_size_{0};
std::mutex TensorAllocator::pool_mutex_;

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
    
    // Hardware-optimized tensor addition
    const size_t size = TensorShape3D::DOF;
    
#ifdef __AVX2__
    // AVX2 vectorized implementation
    const size_t simd_size = 8; // 8 floats per AVX2 register
    const size_t vectorized_elements = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < vectorized_elements; i += simd_size) {
        __m256 va = _mm256_load_ps(&(*a)[i]);
        __m256 vb = _mm256_load_ps(&(*b)[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_store_ps(&(*result)[i], vr);
    }
    
    // Handle remaining elements
    for (size_t i = vectorized_elements; i < size; ++i) {
        (*result)[i] = (*a)[i] + (*b)[i];
    }
#else
    // Scalar fallback implementation
    for (size_t i = 0; i < size; ++i) {
        (*result)[i] = (*a)[i] + (*b)[i];
    }
#endif
}

void TensorUtils::tensorMultiply(const TensorPtr& a, const TensorPtr& b, TensorPtr& result) {
    if (!validateTensorShape(a) || !validateTensorShape(b) || !validateTensorShape(result)) {
        throw std::invalid_argument("Invalid tensor shapes for multiplication");
    }
    
    const size_t size = TensorShape3D::DOF;
    
#ifdef __AVX2__
    // AVX2 vectorized element-wise multiplication
    const size_t simd_size = 8;
    const size_t vectorized_elements = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < vectorized_elements; i += simd_size) {
        __m256 va = _mm256_load_ps(&(*a)[i]);
        __m256 vb = _mm256_load_ps(&(*b)[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_store_ps(&(*result)[i], vr);
    }
    
    for (size_t i = vectorized_elements; i < size; ++i) {
        (*result)[i] = (*a)[i] * (*b)[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        (*result)[i] = (*a)[i] * (*b)[i];
    }
#endif
}

float TensorUtils::tensorDotProduct(const TensorPtr& a, const TensorPtr& b) {
    if (!validateTensorShape(a) || !validateTensorShape(b)) {
        throw std::invalid_argument("Invalid tensor shapes for dot product");
    }
    
    const size_t size = TensorShape3D::DOF;
    float result = 0.0f;
    
#ifdef __AVX2__
    // AVX2 vectorized dot product
    __m256 sum_vec = _mm256_setzero_ps();
    const size_t simd_size = 8;
    const size_t vectorized_elements = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < vectorized_elements; i += simd_size) {
        __m256 va = _mm256_load_ps(&(*a)[i]);
        __m256 vb = _mm256_load_ps(&(*b)[i]);
        __m256 vmul = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, vmul);
    }
    
    // Extract sum from vector
    float sum_array[8];
    _mm256_store_ps(sum_array, sum_vec);
    for (int i = 0; i < 8; ++i) {
        result += sum_array[i];
    }
    
    // Handle remaining elements
    for (size_t i = vectorized_elements; i < size; ++i) {
        result += (*a)[i] * (*b)[i];
    }
#else
    result = std::inner_product(a->begin(), a->end(), b->begin(), 0.0f);
#endif
    
    return result;
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
    
    // Basic performance validation: measure access time
    auto start = std::chrono::high_resolution_clock::now();
    
    volatile float sum = 0.0f;
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        sum += (*tensor)[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Performance threshold: should access 524,288 elements in < 1ms
    return duration.count() < 1000;
}

double TensorUtils::measureTensorOperationTime(std::function<void()> operation) {
    auto start = std::chrono::high_resolution_clock::now();
    operation();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ===================================================================
// TensorAllocator Implementation  
// ===================================================================

void* TensorAllocator::allocateAligned(size_t size, size_t alignment) {
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    
    return ptr;
}

void TensorAllocator::deallocateAligned(void* ptr) {
    if (ptr == nullptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

size_t TensorAllocator::getOptimalAlignment() {
    // Optimal for AVX2 operations
    return 32;
}

void TensorAllocator::initializeMemoryPool(size_t pool_size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    pool_size_ = pool_size;
    pool_offset_.store(0);
    memory_pool_ = std::make_unique<uint8_t[]>(pool_size);
}

void* TensorAllocator::allocateFromPool(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t current_offset = pool_offset_.load();
    if (current_offset + size > pool_size_) {
        return nullptr; // Pool exhausted
    }
    
    void* ptr = memory_pool_.get() + current_offset;
    pool_offset_.store(current_offset + size);
    
    return ptr;
}

void TensorAllocator::releasePool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    memory_pool_.reset();
    pool_offset_.store(0);
    pool_size_ = 0;
}

// ===================================================================
// CognitivePrimitives Implementation
// ===================================================================

void CognitivePrimitives::spatialTransform(TensorPtr& tensor, float x, float y, float z) {
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

std::array<float, 3> CognitivePrimitives::extractSpatialCoordinates(const TensorPtr& tensor, size_t index) {
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

float CognitivePrimitives::semanticSimilarity(const TensorPtr& a, const TensorPtr& b) {
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

void CognitivePrimitives::recursiveCognitiveFold(TensorPtr& tensor, size_t recursion_depth) {
    if (!TensorUtils::validateTensorShape(tensor) || recursion_depth == 0) {
        return;
    }
    
    // Recursive folding operation for cognitive processing
    const size_t fold_size = TensorShape3D::DOF / (1 << recursion_depth);
    
    for (size_t depth = 0; depth < recursion_depth; ++depth) {
        const size_t current_fold_size = TensorShape3D::DOF / (1 << (depth + 1));
        
        for (size_t i = 0; i < current_fold_size; ++i) {
            // Combine pairs of elements recursively
            (*tensor)[i] = ((*tensor)[2 * i] + (*tensor)[2 * i + 1]) * 0.5f;
        }
    }
}

TensorUtils::TensorPtr CognitivePrimitives::recursiveCognitiveExpand(const TensorPtr& seed, size_t expansion_factor) {
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