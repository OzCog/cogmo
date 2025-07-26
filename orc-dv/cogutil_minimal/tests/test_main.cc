/*
 * tests/test_main.cc
 *
 * Foundation Layer: Comprehensive Tensor Validation Tests
 * Cognitive Function: utility-primitives
 * Tensor Shape: [512, 128, 8] = 524,288 DOF
 *
 * Copyright (C) 2024 by OpenCog Foundation
 * All Rights Reserved
 */

#include "cogutil_minimal.h"
#include "tensor_utils_minimal.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

using namespace opencog::util;

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            std::cout << "✓ PASS: " << message << std::endl; \
            tests_passed++; \
        } else { \
            std::cout << "✗ FAIL: " << message << std::endl; \
            tests_failed++; \
        } \
    } while(0)

void test_foundation_layer_specification() {
    std::cout << "\n=== Foundation Layer Specification Tests ===" << std::endl;
    
    // Test tensor shape requirements as per issue
    TEST_ASSERT(TensorShape3D::X == 512, "Tensor X dimension must be 512");
    TEST_ASSERT(TensorShape3D::Y == 128, "Tensor Y dimension must be 128");
    TEST_ASSERT(TensorShape3D::Z == 8, "Tensor Z dimension must be 8");
    TEST_ASSERT(TensorShape3D::DOF == 524288, "Tensor DOF must be 524,288");
    
    // Test cognitive function specification
    TEST_ASSERT(std::string(COGNITIVE_FUNCTION) == "utility-primitives", 
                "Cognitive function must be 'utility-primitives'");
    
    // Test degrees of freedom calculation
    TEST_ASSERT(get_tensor_degrees_of_freedom() == 524288, 
                "DOF function must return 524,288");
    
    // Test tensor architecture validation
    TEST_ASSERT(validate_tensor_architecture(), 
                "Tensor architecture validation must pass");
}

void test_tensor_creation_and_validation() {
    std::cout << "\n=== Tensor Creation and Validation Tests ===" << std::endl;
    
    // Test tensor creation
    auto tensor = TensorUtils::createTensor();
    TEST_ASSERT(tensor != nullptr, "Tensor creation must succeed");
    TEST_ASSERT(TensorUtils::validateTensorShape(tensor), "Created tensor must have valid shape");
    TEST_ASSERT(tensor->size() == TensorShape3D::DOF, "Tensor size must match DOF");
    
    // Test tensor initialization
    TensorUtils::initializeTensor(tensor, 1.5f);
    bool all_initialized = true;
    for (size_t i = 0; i < 100; ++i) { // Sample check
        if (std::abs((*tensor)[i] - 1.5f) > 1e-6f) {
            all_initialized = false;
            break;
        }
    }
    TEST_ASSERT(all_initialized, "Tensor initialization must set all values");
    
    // Test memory footprint
    size_t expected_footprint = TensorShape3D::DOF * sizeof(float);
    TEST_ASSERT(TensorUtils::getTensorMemoryFootprint() == expected_footprint, 
                "Memory footprint calculation must be correct");
    
    // Test performance validation
    TEST_ASSERT(TensorUtils::validateTensorPerformance(tensor), 
                "Tensor performance validation must pass");
}

void test_tensor_arithmetic_operations() {
    std::cout << "\n=== Tensor Arithmetic Operations Tests ===" << std::endl;
    
    auto tensor_a = TensorUtils::createTensor();
    auto tensor_b = TensorUtils::createTensor();
    auto result = TensorUtils::createTensor();
    
    // Initialize test tensors
    TensorUtils::initializeTensor(tensor_a, 3.0f);
    TensorUtils::initializeTensor(tensor_b, 2.0f);
    TensorUtils::initializeTensor(result, 0.0f);
    
    // Test tensor addition
    TensorUtils::tensorAdd(tensor_a, tensor_b, result);
    bool addition_correct = true;
    for (size_t i = 0; i < 100; ++i) { // Sample check
        if (std::abs((*result)[i] - 5.0f) > 1e-6f) {
            addition_correct = false;
            break;
        }
    }
    TEST_ASSERT(addition_correct, "Tensor addition must be correct");
    
    // Test tensor multiplication
    TensorUtils::tensorMultiply(tensor_a, tensor_b, result);
    bool multiplication_correct = true;
    for (size_t i = 0; i < 100; ++i) { // Sample check
        if (std::abs((*result)[i] - 6.0f) > 1e-6f) {
            multiplication_correct = false;
            break;
        }
    }
    TEST_ASSERT(multiplication_correct, "Tensor multiplication must be correct");
    
    // Test dot product
    float dot_product = TensorUtils::tensorDotProduct(tensor_a, tensor_b);
    float expected_dot = 3.0f * 2.0f * TensorShape3D::DOF;
    TEST_ASSERT(std::abs(dot_product - expected_dot) < 1e-3f, 
                "Tensor dot product must be correct");
    
    // Test normalization
    TensorUtils::tensorNormalize(tensor_a);
    float norm_squared = 0.0f;
    for (const auto& value : *tensor_a) {
        norm_squared += value * value;
    }
    TEST_ASSERT(std::abs(std::sqrt(norm_squared) - 1.0f) < 1e-5f, 
                "Tensor normalization must produce unit vector");
}

void test_recursive_tensor_operations() {
    std::cout << "\n=== Recursive Tensor Operations Tests ===" << std::endl;
    
    auto tensor = TensorUtils::createTensor();
    TensorUtils::initializeTensor(tensor, 1.0f);
    
    // Test recursive decomposition
    auto components = TensorUtils::decomposeTensor(tensor, 2);
    TEST_ASSERT(components.size() == 4, "Decomposition depth 2 must create 4 components");
    
    bool all_components_valid = true;
    for (const auto& component : components) {
        if (!TensorUtils::validateTensorShape(component)) {
            all_components_valid = false;
            break;
        }
    }
    TEST_ASSERT(all_components_valid, "All decomposed components must be valid");
    
    // Test composition
    auto recomposed = TensorUtils::composeTensors(components);
    TEST_ASSERT(TensorUtils::validateTensorShape(recomposed), 
                "Recomposed tensor must be valid");
    
    // Test parallel operations
    auto parallel_tensor = TensorUtils::createTensor();
    TensorUtils::initializeTensor(parallel_tensor, 0.0f);
    
    TensorUtils::parallelTensorOperation(parallel_tensor, 
        [](float& value, size_t index) {
            value = static_cast<float>(index % 1000);
        });
    
    bool parallel_correct = true;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs((*parallel_tensor)[i] - static_cast<float>(i % 1000)) > 1e-6f) {
            parallel_correct = false;
            break;
        }
    }
    TEST_ASSERT(parallel_correct, "Parallel tensor operation must be correct");
}

void test_cognitive_primitives() {
    std::cout << "\n=== Cognitive Primitives Tests ===" << std::endl;
    
    auto tensor_a = TensorUtils::createTensor();
    auto tensor_b = TensorUtils::createTensor();
    
    // Initialize with patterns
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        (*tensor_a)[i] = std::sin(static_cast<float>(i) * 0.001f);
        (*tensor_b)[i] = std::cos(static_cast<float>(i) * 0.001f);
    }
    
    // Test spatial transformation
    auto spatial_tensor = TensorUtils::createTensor();
    TensorUtils::initializeTensor(spatial_tensor, 1.0f);
    
    CognitivePrimitives::spatialTransform(spatial_tensor, 0.1f, 0.2f, 0.3f);
    bool spatial_changed = false;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs((*spatial_tensor)[i] - 1.0f) > 1e-6f) {
            spatial_changed = true;
            break;
        }
    }
    TEST_ASSERT(spatial_changed, "Spatial transformation must modify tensor values");
    
    // Test spatial coordinate extraction
    auto coords = CognitivePrimitives::extractSpatialCoordinates(tensor_a, 10000);
    TEST_ASSERT(coords[0] >= 0.0f && coords[0] < TensorShape3D::X, 
                "Extracted X coordinate must be in valid range");
    TEST_ASSERT(coords[1] >= 0.0f && coords[1] < TensorShape3D::Y, 
                "Extracted Y coordinate must be in valid range");
    TEST_ASSERT(coords[2] >= 0.0f && coords[2] < TensorShape3D::Z, 
                "Extracted Z coordinate must be in valid range");
    
    // Test temporal shift
    auto temporal_tensor = TensorUtils::createTensor();
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        (*temporal_tensor)[i] = static_cast<float>(i);
    }
    
    float original_value = (*temporal_tensor)[100];
    CognitivePrimitives::temporalShift(temporal_tensor, 50);
    float shifted_value = (*temporal_tensor)[150];
    TEST_ASSERT(std::abs(original_value - shifted_value) < 1e-6f, 
                "Temporal shift must preserve values at new positions");
    
    // Test semantic similarity
    float similarity = CognitivePrimitives::semanticSimilarity(tensor_a, tensor_b);
    TEST_ASSERT(similarity >= -1.0f && similarity <= 1.0f, 
                "Semantic similarity must be in [-1, 1] range");
    
    float self_similarity = CognitivePrimitives::semanticSimilarity(tensor_a, tensor_a);
    TEST_ASSERT(std::abs(self_similarity - 1.0f) < 1e-5f, 
                "Self-similarity must be approximately 1.0");
    
    // Test logical consistency
    TEST_ASSERT(CognitivePrimitives::validateLogicalConsistency(tensor_a), 
                "Tensor with valid values must pass logical consistency");
    
    // Test recursive cognitive fold
    auto fold_tensor = TensorUtils::createTensor();
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        (*fold_tensor)[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    float original_first = (*fold_tensor)[0];
    CognitivePrimitives::recursiveCognitiveFold(fold_tensor, 2);
    bool fold_changed = (std::abs((*fold_tensor)[0] - original_first) > 1e-6f);
    TEST_ASSERT(fold_changed, "Recursive cognitive fold must modify tensor");
    
    // Test recursive cognitive expand
    auto expand_seed = TensorUtils::createTensor();
    TensorUtils::initializeTensor(expand_seed, 1.0f);
    
    auto expanded = CognitivePrimitives::recursiveCognitiveExpand(expand_seed, 2);
    TEST_ASSERT(TensorUtils::validateTensorShape(expanded), 
                "Recursive cognitive expand must produce valid tensor");
}

void test_performance_benchmarks() {
    std::cout << "\n=== Performance Benchmark Tests ===" << std::endl;
    
    auto tensor_a = TensorUtils::createTensor();
    auto tensor_b = TensorUtils::createTensor();
    auto result = TensorUtils::createTensor();
    
    TensorUtils::initializeTensor(tensor_a, 1.0f);
    TensorUtils::initializeTensor(tensor_b, 2.0f);
    
    // Benchmark tensor addition
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        TensorUtils::tensorAdd(tensor_a, tensor_b, result);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time_us = duration.count() / 100.0f;
    
    std::cout << "  Average tensor addition time: " << avg_time_us << " microseconds" << std::endl;
    TEST_ASSERT(avg_time_us < 10000.0f, "Tensor addition must complete in reasonable time");
    
    // Benchmark dot product
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        volatile float dot = TensorUtils::tensorDotProduct(tensor_a, tensor_b);
        (void)dot; // Prevent optimization
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_time_us = duration.count() / 100.0f;
    
    std::cout << "  Average dot product time: " << avg_time_us << " microseconds" << std::endl;
    TEST_ASSERT(avg_time_us < 5000.0f, "Dot product must complete in reasonable time");
    
    // Memory access benchmark
    start = std::chrono::high_resolution_clock::now();
    volatile float sum = 0.0f;
    for (size_t i = 0; i < TensorShape3D::DOF; ++i) {
        sum += (*tensor_a)[i];
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float access_time_us = duration.count();
    
    std::cout << "  Full tensor access time: " << access_time_us << " microseconds" << std::endl;
    TEST_ASSERT(access_time_us < 2000.0f, "Full tensor access must be fast");
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "Foundation Layer: Cogutil Validation Tests" << std::endl;
    std::cout << "Tensor Shape: [512, 128, 8] = 524,288 DOF" << std::endl;
    std::cout << "Cognitive Function: utility-primitives" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Initialize cogutil
        initialize_cogutil();
        
        // Run all test suites
        test_foundation_layer_specification();
        test_tensor_creation_and_validation();
        test_tensor_arithmetic_operations();
        test_recursive_tensor_operations();
        test_cognitive_primitives();
        test_performance_benchmarks();
        
        // Shutdown cogutil
        shutdown_cogutil();
        
    } catch (const std::exception& e) {
        std::cerr << "Test execution failed: " << e.what() << std::endl;
        tests_failed++;
    }
    
    // Print summary
    std::cout << "\n=========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Tests Passed: " << tests_passed << std::endl;
    std::cout << "  Tests Failed: " << tests_failed << std::endl;
    std::cout << "  Success Rate: " << (100.0f * tests_passed / (tests_passed + tests_failed)) << "%" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "🎉 ALL TESTS PASSED! Foundation Layer cogutil implementation validated." << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed. Foundation Layer implementation needs review." << std::endl;
        return 1;
    }
}