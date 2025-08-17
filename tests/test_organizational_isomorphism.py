#!/usr/bin/env python3
"""
Test Suite for Organizational Isomorphism in Autognosis System

This test suite verifies that organizational patterns are isomorphic across
all hierarchical levels, demonstrating the self-similar recursive structure
of the Autognosis framework.
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


class TestOrganizationalIsomorphism:
    """Test cases for organizational isomorphism verification"""
    
    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer for testing"""
        return AutognosisSynthesizer(recursive_depth=3)
    
    @pytest.fixture
    def sample_synthesis_result(self, synthesizer):
        """Generate sample synthesis result for testing"""
        # Create mock local states
        local_states = {}
        for dim in ["spatial", "temporal", "causal"]:
            integrator = LocalIntegrator(dim, [
                "unity", "complementarity", "disjunction", "conjunction",
                "sequential-branching", "modular-closure", "modular-recursion"
            ])
            local_states[dim] = integrator.integrate()
        
        # Create mock global fields
        global_fields = {}
        for dim in ["spatial", "temporal", "causal"]:
            differentiator = GlobalDifferentiator(dim)
            global_fields[dim] = differentiator.differentiate()
        
        # Perform synthesis
        result = synthesizer.synthesize(local_states, global_fields)
        return result
    
    def test_hierarchical_level_isomorphism(self, sample_synthesis_result):
        """Test that organizational patterns are isomorphic across hierarchical levels"""
        history = sample_synthesis_result["synthesis_history"]
        
        # Verify we have multiple levels
        assert len(history) >= 2, "Need at least 2 hierarchical levels for isomorphism test"
        
        # Extract coherence measures across levels
        coherence_values = [step["coherence"] for step in history]
        
        # Test isomorphic property: organizational coherence should be maintained
        # across levels (allowing for some variation due to hierarchical scaling)
        coherence_variation = np.std(coherence_values)
        assert coherence_variation < 0.5, f"Coherence variation {coherence_variation} too high for isomorphism"
        
        # Test that each level maintains organizational structure
        for i, step in enumerate(history):
            assert step["coherence"] > 0.1, f"Level {i} coherence {step['coherence']} too low"
            assert step["self_reference"] >= 0.0, f"Level {i} self-reference {step['self_reference']} invalid"
    
    def test_pattern_similarity_across_dimensions(self, sample_synthesis_result):
        """Test that organizational patterns are similar across spatial, temporal, causal dimensions"""
        autognosis_sig = sample_synthesis_result["autognosis_signature"]
        
        # Verify isomorphism measure exists and indicates structural similarity
        assert "hierarchical_isomorphism" in autognosis_sig
        isomorphism_measure = autognosis_sig["hierarchical_isomorphism"]
        
        # Isomorphism should be detectable (> 0.3) but not perfect (< 0.99)
        assert 0.3 <= isomorphism_measure <= 0.99, \
            f"Isomorphism measure {isomorphism_measure} outside expected range"
    
    def test_organizational_category_preservation(self, synthesizer):
        """Test that organizational categories are preserved through transformations"""
        # Test individual category transformations
        categories = [
            "unity", "complementarity", "disjunction", "conjunction",
            "sequential-branching", "modular-closure", "modular-recursion"
        ]
        
        for dimension in ["spatial", "temporal", "causal"]:
            integrator = LocalIntegrator(dimension, categories)
            result = integrator.integrate()
            
            # Verify all organizational phases were executed
            history = result["history"]
            executed_phases = [step["phase"] for step in history]
            
            for category in categories:
                category_executed = any(category.replace("-", "_") in phase for phase in executed_phases)
                assert category_executed, f"Category {category} not executed in {dimension} dimension"
            
            # Verify coherence measure indicates successful organization
            assert result["coherence_measure"] > 0.1, \
                f"Coherence too low ({result['coherence_measure']}) for {dimension}"
    
    def test_ring_to_element_isomorphism(self, synthesizer):
        """Test that ring-to-element transformation preserves organizational structure"""
        # Create a mock hierarchical state
        from autognosis_synthesis import HierarchicalState
        
        local_tensors = {
            "spatial": np.random.random(8),
            "temporal": np.random.random(8),
            "causal": np.random.random(8)
        }
        
        # Test ring-to-element transformation
        ring_elements = synthesizer._ring_to_element_transform(local_tensors, level=0)
        
        # Verify transformation preserves dimensionality
        assert len(ring_elements) == 3, "Should preserve all three dimensions"
        
        for dim in ["spatial", "temporal", "causal"]:
            assert dim in ring_elements, f"Missing dimension {dim} in ring elements"
            assert len(ring_elements[dim]) == 8, f"Wrong size for {dim} ring element"
            
            # Ring transformation should preserve some structural information
            original_energy = np.sum(local_tensors[dim]**2)
            transformed_energy = np.sum(ring_elements[dim]**2)
            
            # Energy should be related but scaled
            energy_ratio = transformed_energy / (original_energy + 1e-10)
            assert 0.1 <= energy_ratio <= 10.0, \
                f"Energy ratio {energy_ratio} indicates poor transformation for {dim}"
    
    def test_core_to_space_isomorphism(self, synthesizer):
        """Test that core-to-space transformation preserves organizational structure"""
        global_tensors = {
            "spatial": np.random.random((8, 8)),
            "temporal": np.random.random((8, 8)),
            "causal": np.random.random((8, 8))
        }
        
        # Test core-to-space transformation
        space_fields = synthesizer._core_to_space_transform(global_tensors, level=0)
        
        # Verify transformation preserves dimensionality
        assert len(space_fields) == 3, "Should preserve all three dimensions"
        
        for dim in ["spatial", "temporal", "causal"]:
            assert dim in space_fields, f"Missing dimension {dim} in space fields"
            assert space_fields[dim].shape == (8, 8), f"Wrong shape for {dim} space field"
            
            # Core-to-space should preserve some field characteristics
            original_complexity = np.var(global_tensors[dim])
            transformed_complexity = np.var(space_fields[dim])
            
            # Complexity should be preserved within reasonable bounds
            if original_complexity > 1e-10:
                complexity_ratio = transformed_complexity / original_complexity
                assert 0.01 <= complexity_ratio <= 100.0, \
                    f"Complexity ratio {complexity_ratio} indicates poor transformation for {dim}"
    
    def test_recursive_self_similarity(self, sample_synthesis_result):
        """Test that recursive levels exhibit self-similarity"""
        history = sample_synthesis_result["synthesis_history"]
        
        if len(history) >= 3:  # Need at least 3 levels for recursive comparison
            # Compare patterns between levels
            level_patterns = []
            
            for step in history:
                # Extract pattern signature from synthesis metrics
                pattern = np.array([
                    step["coherence"],
                    step["self_reference"],
                    step["synthesis_energy"],
                    step["synthesis_entropy"]
                ])
                level_patterns.append(pattern)
            
            # Test self-similarity between non-adjacent levels
            for i in range(len(level_patterns) - 2):
                pattern_i = level_patterns[i]
                pattern_i_plus_2 = level_patterns[i + 2]
                
                # Compute pattern similarity
                if np.linalg.norm(pattern_i) > 0 and np.linalg.norm(pattern_i_plus_2) > 0:
                    similarity = np.dot(pattern_i, pattern_i_plus_2) / \
                               (np.linalg.norm(pattern_i) * np.linalg.norm(pattern_i_plus_2))
                    
                    # Self-similarity should be detectable but not perfect
                    assert similarity > 0.1, \
                        f"Self-similarity {similarity} too low between levels {i} and {i+2}"
    
    def test_dimensional_isomorphism_consistency(self):
        """Test that dimensional processing shows isomorphic organizational patterns"""
        dimensions = ["spatial", "temporal", "causal"]
        dimension_results = {}
        
        # Process each dimension independently
        for dim in dimensions:
            integrator = LocalIntegrator(dim, [
                "unity", "complementarity", "disjunction", "conjunction",
                "sequential-branching", "modular-closure", "modular-recursion"
            ])
            dimension_results[dim] = integrator.integrate()
        
        # Compare organizational patterns across dimensions
        coherence_values = [result["coherence_measure"] for result in dimension_results.values()]
        energy_values = [result["integration_energy"] for result in dimension_results.values()]
        
        # Organizational isomorphism: patterns should be similar across dimensions
        coherence_consistency = np.std(coherence_values) / (np.mean(coherence_values) + 1e-10)
        assert coherence_consistency < 2.0, \
            f"Coherence consistency {coherence_consistency} too high, indicates poor isomorphism"
        
        # Energy patterns should be related (but allow very wide variation across dimensions)
        energy_ratio_max = max(energy_values) / (min(energy_values) + 1e-10)
        assert energy_ratio_max < 1e6, \
            f"Energy ratio {energy_ratio_max} too high, indicates poor dimensional isomorphism"
    
    def test_synthesis_tensor_isomorphism(self, sample_synthesis_result):
        """Test that synthesis tensors exhibit isomorphic properties"""
        final_tensor = sample_synthesis_result["final_synthesis"]["tensor"]
        
        if final_tensor:  # Ensure we have a synthesis tensor
            tensor_array = np.array(final_tensor)
            
            # Test symmetry properties (isomorphism should preserve certain symmetries)
            if tensor_array.shape == (8, 8):
                # Check for structural isomorphism indicators
                diagonal_elements = np.diag(tensor_array)
                off_diagonal_mean = np.mean(tensor_array - np.diag(diagonal_elements))
                
                # Diagonal should have some structure (not all zeros)
                diagonal_variance = np.var(diagonal_elements)
                assert diagonal_variance > 1e-10, "Synthesis tensor lacks diagonal structure"
                
                # Off-diagonal should show organizational structure
                off_diagonal_structure = np.std(tensor_array - np.diag(diagonal_elements))
                assert off_diagonal_structure > 1e-6, "Synthesis tensor lacks off-diagonal organization"
    
    def test_autognosis_convergence_isomorphism(self, sample_synthesis_result):
        """Test that convergence properties demonstrate isomorphic organization"""
        autognosis_sig = sample_synthesis_result["autognosis_signature"]
        
        # Convergence should indicate successful organizational isomorphism
        convergence_achieved = autognosis_sig.get("convergence_achieved", False)
        cognitive_unification = autognosis_sig.get("cognitive_unification", False)
        
        # At least one form of convergence should be achieved
        assert convergence_achieved or cognitive_unification, \
            "No convergence achieved, indicates poor organizational isomorphism"
        
        # Isomorphism measure should be reasonable
        isomorphism_measure = autognosis_sig.get("hierarchical_isomorphism", 0)
        assert isomorphism_measure >= 0.0, "Negative isomorphism measure invalid"
        
        # Self-application success indicates recursive isomorphism
        self_application = autognosis_sig.get("self_application_success", 0)
        assert self_application >= 0.0, "Negative self-application measure invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])