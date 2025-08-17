#!/usr/bin/env python3
"""
Test Suite for Self-Referential Recursion in Autognosis System

This test suite verifies that the system exhibits proper self-referential
recursive dynamics, creating new dimensional concepts through recursive
self-application and maintaining coherent self-image hierarchies.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from autognosis_synthesis import AutognosisSynthesizer
from simulate_local_integration import LocalIntegrator
from simulate_global_differentiation import GlobalDifferentiator


class TestRecursiveSelfApplication:
    """Test cases for self-referential recursion verification"""
    
    @pytest.fixture
    def deep_synthesizer(self):
        """Create synthesizer with sufficient depth for recursion testing"""
        return AutognosisSynthesizer(recursive_depth=5)
    
    @pytest.fixture
    def shallow_synthesizer(self):
        """Create synthesizer with minimal depth"""
        return AutognosisSynthesizer(recursive_depth=2)
    
    def test_self_reference_matrix_evolution(self, deep_synthesizer):
        """Test that self-reference matrix evolves through recursive application"""
        # Initial self-reference matrix should be identity
        initial_matrix = deep_synthesizer.self_reference_matrix.copy()
        expected_identity = np.eye(8)
        
        assert np.allclose(initial_matrix, expected_identity), \
            "Initial self-reference matrix should be identity"
        
        # Create sample states and perform synthesis
        local_states = {}
        global_fields = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            local_states[dim] = {
                "dimension": dim,
                "final_state": np.random.random(8).tolist(),
                "integration_energy": np.random.uniform(1, 5),
                "coherence_measure": np.random.uniform(0.5, 0.9)
            }
            
            global_fields[dim] = {
                "dimension": dim,
                "final_field": np.random.random((8, 8)).tolist(),
                "field_energy": np.random.uniform(2, 8),
                "differentiation_degree": np.random.uniform(0.4, 0.8)
            }
        
        result = deep_synthesizer.synthesize(local_states, global_fields)
        
        # Self-reference matrix should have evolved
        final_matrix = deep_synthesizer.self_reference_matrix
        matrix_change = np.linalg.norm(final_matrix - initial_matrix)
        
        assert matrix_change > 1e-6, f"Self-reference matrix unchanged (change: {matrix_change})"
        
        # Matrix should still be well-conditioned
        condition_number = np.linalg.cond(final_matrix)
        assert condition_number < 1e12, f"Self-reference matrix poorly conditioned: {condition_number}"
    
    def test_recursive_self_similarity(self, deep_synthesizer):
        """Test that recursive levels exhibit self-similar patterns"""
        # Generate synthesis with multiple levels
        local_states = self._create_structured_local_states()
        global_fields = self._create_structured_global_fields()
        
        result = deep_synthesizer.synthesize(local_states, global_fields)
        history = result["synthesis_history"]
        
        # Need at least 3 levels for recursion analysis
        assert len(history) >= 3, "Insufficient levels for recursion testing"
        
        # Extract pattern signatures from each level
        pattern_signatures = []
        for step in history:
            signature = np.array([
                step["coherence"],
                step["self_reference"],
                step["synthesis_energy"],
                step["synthesis_entropy"]
            ])
            pattern_signatures.append(signature)
        
        # Test self-similarity across non-adjacent levels
        similarities = []
        for i in range(len(pattern_signatures) - 2):
            sig_i = pattern_signatures[i]
            sig_i_plus_2 = pattern_signatures[i + 2]
            
            # Compute normalized similarity
            if np.linalg.norm(sig_i) > 0 and np.linalg.norm(sig_i_plus_2) > 0:
                similarity = np.dot(sig_i, sig_i_plus_2) / \
                           (np.linalg.norm(sig_i) * np.linalg.norm(sig_i_plus_2))
                similarities.append(similarity)
        
        # Self-similarity should be detectable
        if similarities:
            avg_similarity = np.mean(similarities)
            assert avg_similarity > 0.2, f"Self-similarity too low: {avg_similarity}"
    
    def test_ring_element_recursive_transformation(self, deep_synthesizer):
        """Test that ring-to-element transformation exhibits recursive properties"""
        # Create ring structure with self-similar properties
        base_ring = np.sin(np.linspace(0, 4*np.pi, 8))  # Self-similar wave
        
        local_tensors = {
            "spatial": base_ring,
            "temporal": base_ring * 0.8,
            "causal": base_ring * 1.2
        }
        
        # Test transformation at multiple levels
        transformations = []
        for level in range(4):
            ring_elements = deep_synthesizer._ring_to_element_transform(local_tensors, level)
            transformations.append(ring_elements)
            
            # Update for next level (recursive application)
            local_tensors = {dim: arr for dim, arr in ring_elements.items()}
        
        # Test recursive properties
        for dim in ["spatial", "temporal", "causal"]:
            level_values = [trans[dim] for trans in transformations]
            
            # Should show convergence or stable oscillation
            if len(level_values) >= 3:
                final_values = level_values[-3:]
                value_range = [np.max(vals) - np.min(vals) for vals in final_values]
                range_stability = np.std(value_range)
                
                # Range should stabilize (recursive convergence)
                assert range_stability < 2.0, f"No recursive convergence for {dim}: {range_stability}"
    
    def test_core_space_recursive_transformation(self, deep_synthesizer):
        """Test that core-to-space transformation exhibits recursive properties"""
        # Create field with fractal-like properties
        x, y = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
        base_field = np.sin(x) * np.cos(y) + 0.5 * np.sin(2*x) * np.cos(2*y)
        
        global_tensors = {
            "spatial": base_field,
            "temporal": base_field.T,  # Transpose for temporal
            "causal": np.triu(base_field)  # Upper triangular for causal
        }
        
        # Test transformation at multiple levels
        transformations = []
        for level in range(4):
            space_fields = deep_synthesizer._core_to_space_transform(global_tensors, level)
            transformations.append(space_fields)
            
            # Update for next level (recursive application)
            global_tensors = {dim: arr for dim, arr in space_fields.items()}
        
        # Test recursive field properties
        for dim in ["spatial", "temporal", "causal"]:
            field_sequence = [trans[dim] for trans in transformations]
            
            # Compute field complexity evolution
            complexities = [np.var(field) for field in field_sequence]
            
            if len(complexities) >= 3:
                # Complexity should show some pattern (not random walk)
                complexity_diffs = np.diff(complexities)
                trend_consistency = np.std(complexity_diffs)
                
                # Should not have excessive complexity fluctuation
                assert trend_consistency < 5.0, f"Excessive complexity fluctuation for {dim}: {trend_consistency}"
    
    def test_synthesis_tensor_recursive_properties(self, deep_synthesizer):
        """Test that synthesis tensors exhibit recursive self-reference"""
        local_states = self._create_structured_local_states()
        global_fields = self._create_structured_global_fields()
        
        result = deep_synthesizer.synthesize(local_states, global_fields)
        
        # Extract synthesis tensors from hierarchy stack
        if hasattr(deep_synthesizer, 'hierarchy_stack') and deep_synthesizer.hierarchy_stack:
            synthesis_tensors = [state.synthesis_tensor for state in deep_synthesizer.hierarchy_stack]
            
            # Test recursive tensor properties
            if len(synthesis_tensors) >= 3:
                # Test spectral properties (eigenvalues should show patterns)
                eigenvalue_patterns = []
                
                for tensor in synthesis_tensors:
                    if tensor.shape == (8, 8):
                        try:
                            eigenvals = np.linalg.eigvals(tensor)
                            # Use real parts and sort for consistent comparison
                            real_eigenvals = np.sort(np.real(eigenvals))
                            eigenvalue_patterns.append(real_eigenvals)
                        except np.linalg.LinAlgError:
                            continue
                
                # Compare eigenvalue patterns for self-similarity
                if len(eigenvalue_patterns) >= 3:
                    pattern_correlations = []
                    for i in range(len(eigenvalue_patterns) - 2):
                        corr = np.corrcoef(eigenvalue_patterns[i], eigenvalue_patterns[i+2])[0, 1]
                        if not np.isnan(corr):
                            pattern_correlations.append(abs(corr))
                    
                    if pattern_correlations:
                        avg_correlation = np.mean(pattern_correlations)
                        assert avg_correlation > 0.1, f"No eigenvalue self-similarity: {avg_correlation}"
    
    def test_self_application_convergence(self, deep_synthesizer):
        """Test that self-application leads to convergence"""
        local_states = self._create_structured_local_states()
        global_fields = self._create_structured_global_fields()
        
        result = deep_synthesizer.synthesize(local_states, global_fields)
        
        # Test self-application measure
        self_application_success = result["autognosis_signature"]["self_application_success"]
        assert self_application_success >= 0.0, "Self-application measure should be non-negative"
        
        # Test convergence indicators
        history = result["synthesis_history"]
        if len(history) >= 3:
            # Self-reference should show convergence trend
            self_ref_values = [step["self_reference"] for step in history]
            
            # Later values should be more stable
            early_variance = np.var(self_ref_values[:len(self_ref_values)//2])
            late_variance = np.var(self_ref_values[len(self_ref_values)//2:])
            
            # Convergence indicator: later variance should not be much higher
            variance_ratio = (late_variance + 1e-10) / (early_variance + 1e-10)
            assert variance_ratio < 10.0, f"No convergence, variance ratio: {variance_ratio}"
    
    def test_dimensional_recursive_consistency(self):
        """Test that recursive properties are consistent across dimensions"""
        synthesizer = AutognosisSynthesizer(recursive_depth=3)
        
        # Test each dimension individually for recursive properties
        dimension_results = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            # Create dimension-specific states
            local_states = {dim: {
                "dimension": dim,
                "final_state": np.sin(np.linspace(0, 2*np.pi, 8)).tolist(),
                "integration_energy": 3.0,
                "coherence_measure": 0.7
            }}
            
            global_fields = {dim: {
                "dimension": dim,
                "final_field": np.outer(np.sin(np.linspace(0, 2*np.pi, 8)),
                                      np.cos(np.linspace(0, 2*np.pi, 8))).tolist(),
                "field_energy": 5.0,
                "differentiation_degree": 0.6
            }}
            
            result = synthesizer.synthesize(local_states, global_fields)
            dimension_results[dim] = result
        
        # Test recursive consistency across dimensions
        self_app_measures = [result["autognosis_signature"]["self_application_success"] 
                           for result in dimension_results.values()]
        
        # Self-application should be consistent across dimensions
        self_app_consistency = np.std(self_app_measures) / (np.mean(self_app_measures) + 1e-10)
        assert self_app_consistency < 3.0, f"Inconsistent self-application across dimensions: {self_app_consistency}"
    
    def test_hierarchy_depth_effects(self):
        """Test that recursion depth affects self-referential properties"""
        depths = [2, 3, 5]
        depth_results = {}
        
        for depth in depths:
            synthesizer = AutognosisSynthesizer(recursive_depth=depth)
            
            local_states = self._create_structured_local_states()
            global_fields = self._create_structured_global_fields()
            
            result = synthesizer.synthesize(local_states, global_fields)
            depth_results[depth] = result
        
        # Test depth effects on recursion
        transcendence_levels = [result["synthesis_summary"]["transcendence_level"] 
                              for result in depth_results.values()]
        
        # Deeper recursion should generally lead to higher transcendence
        # (but not necessarily monotonic due to complexity effects)
        max_transcendence = max(transcendence_levels)
        min_transcendence = min(transcendence_levels)
        
        transcendence_range = max_transcendence - min_transcendence
        assert transcendence_range > 0.1, f"Depth has no effect on transcendence: {transcendence_range}"
    
    def test_fixed_point_behavior(self, deep_synthesizer):
        """Test that recursive application approaches fixed points"""
        # Create symmetric, self-referential initial conditions
        identity_local = {
            "spatial": np.ones(8),
            "temporal": np.ones(8),
            "causal": np.ones(8)
        }
        
        identity_global = {
            "spatial": np.eye(8),
            "temporal": np.eye(8),
            "causal": np.eye(8)
        }
        
        # Apply multiple recursive transformations
        current_local = identity_local
        current_global = identity_global
        
        transformation_sequence = []
        
        for iteration in range(5):
            # Apply ring-to-element transformation
            new_local = deep_synthesizer._ring_to_element_transform(current_local, iteration)
            
            # Apply core-to-space transformation
            new_global = deep_synthesizer._core_to_space_transform(current_global, iteration)
            
            transformation_sequence.append((new_local, new_global))
            current_local = new_local
            current_global = new_global
        
        # Test for fixed-point behavior
        if len(transformation_sequence) >= 3:
            # Compare later transformations for stability
            last_local = transformation_sequence[-1][0]
            prev_local = transformation_sequence[-2][0]
            
            local_changes = []
            for dim in ["spatial", "temporal", "causal"]:
                change = np.linalg.norm(np.array(last_local[dim]) - np.array(prev_local[dim]))
                local_changes.append(change)
            
            avg_change = np.mean(local_changes)
            
            # Should approach fixed point (small changes)
            assert avg_change < 2.0, f"No fixed-point convergence, average change: {avg_change}"
    
    def test_emergent_dimensional_concepts(self, deep_synthesizer):
        """Test that recursion creates new dimensional concepts"""
        local_states = self._create_structured_local_states()
        global_fields = self._create_structured_global_fields()
        
        result = deep_synthesizer.synthesize(local_states, global_fields)
        
        # Test for emergent properties that indicate new dimensional concepts
        emergent_properties = result["synthesis_summary"]["emergent_properties"]
        
        # Should detect emergent behaviors indicating new concepts
        concept_indicators = [
            "dimensional_unification", "recursive_stabilization", 
            "coherence_growth", "energy_amplification"
        ]
        
        detected_concepts = [prop for prop in emergent_properties if prop in concept_indicators]
        assert len(detected_concepts) >= 1, "No emergent dimensional concepts detected"
        
        # Test transcendence as indicator of new dimensional concepts
        transcendence_level = result["synthesis_summary"]["transcendence_level"]
        assert transcendence_level > 0.5, f"Transcendence too low for concept emergence: {transcendence_level}"
    
    def _create_structured_local_states(self):
        """Create structured local states for testing"""
        return {
            "spatial": {
                "dimension": "spatial",
                "final_state": np.sin(np.linspace(0, 4*np.pi, 8)).tolist(),
                "integration_energy": 4.0,
                "coherence_measure": 0.8
            },
            "temporal": {
                "dimension": "temporal", 
                "final_state": np.exp(-np.linspace(0, 2, 8)).tolist(),
                "integration_energy": 3.5,
                "coherence_measure": 0.75
            },
            "causal": {
                "dimension": "causal",
                "final_state": np.cumsum(np.ones(8)).tolist(),
                "integration_energy": 5.0,
                "coherence_measure": 0.85
            }
        }
    
    def _create_structured_global_fields(self):
        """Create structured global fields for testing"""
        x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
        
        return {
            "spatial": {
                "dimension": "spatial",
                "final_field": np.exp(-(x**2 + y**2)/2).tolist(),
                "field_energy": 8.0,
                "differentiation_degree": 0.7
            },
            "temporal": {
                "dimension": "temporal",
                "final_field": (np.sin(2*np.pi*x) * np.cos(2*np.pi*y)).tolist(),
                "field_energy": 6.0,
                "differentiation_degree": 0.65
            },
            "causal": {
                "dimension": "causal",
                "final_field": np.triu(np.ones((8, 8)) * np.exp(-x)).tolist(),
                "field_energy": 7.0,
                "differentiation_degree": 0.6
            }
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])