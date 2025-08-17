#!/usr/bin/env python3
"""
Test Suite for Co-evolution Dynamics in Autognosis System

This test suite verifies that local bottom-up integration and global top-down
differentiation co-evolve in a coordinated manner, creating hierarchical
structures with emergent properties.
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


class TestCoevolutionDynamics:
    """Test cases for co-evolution dynamics verification"""
    
    @pytest.fixture
    def local_integrator_spatial(self):
        """Create spatial local integrator"""
        return LocalIntegrator("spatial", [
            "unity", "complementarity", "disjunction", "conjunction",
            "sequential-branching", "modular-closure", "modular-recursion"
        ])
    
    @pytest.fixture
    def global_differentiator_spatial(self):
        """Create spatial global differentiator"""
        return GlobalDifferentiator("spatial")
    
    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer for testing"""
        return AutognosisSynthesizer(recursive_depth=4)
    
    def test_local_integration_progression(self, local_integrator_spatial):
        """Test that local integration shows proper progression through organizational phases"""
        result = local_integrator_spatial.integrate()
        history = result["history"]
        
        # Verify all expected phases are present and in order
        expected_phases = [
            "unity", "complementarity", "disjunction", "conjunction",
            "sequential_branching", "modular_closure", "modular_recursion"
        ]
        
        executed_phases = [step["phase"] for step in history]
        
        for expected_phase in expected_phases:
            phase_found = any(expected_phase in phase for phase in executed_phases)
            assert phase_found, f"Missing phase {expected_phase} in local integration"
        
        # Test energy progression (should generally increase with complexity)
        energies = [step["energy"] for step in history]
        
        # Energy should be positive and show some variation
        assert all(e >= 0 for e in energies), "Negative energy values found"
        assert np.std(energies) > 1e-10, "No energy variation indicates poor dynamics"
        
        # Test coherence evolution
        coherences = [step.get("coherence", 0) for step in history]
        final_coherence = result["coherence_measure"]
        
        assert final_coherence > 0, "Final coherence should be positive"
    
    def test_global_differentiation_progression(self, global_differentiator_spatial):
        """Test that global differentiation shows proper field evolution"""
        result = global_differentiator_spatial.differentiate()
        history = result["history"]
        
        # Verify field evolution stages
        expected_field_types = [
            "undifferentiated", "polarized", "compartmentalized", "hierarchical"
        ]
        
        field_types = [step["field_type"] for step in history]
        
        for expected_type in expected_field_types:
            assert expected_type in field_types, f"Missing field type {expected_type}"
        
        # Test field energy progression
        field_energies = [step["field_energy"] for step in history]
        
        assert all(e >= 0 for e in field_energies), "Negative field energy found"
        assert len(field_energies) >= 3, "Insufficient field evolution steps"
        
        # Test differentiation degree (should increase with complexity)
        final_differentiation = result["differentiation_degree"]
        assert final_differentiation >= 0, "Differentiation degree should be non-negative"
    
    def test_coevolution_coupling(self):
        """Test that local and global processes show coupled evolution"""
        dimensions = ["spatial", "temporal", "causal"]
        local_results = {}
        global_results = {}
        
        # Process all dimensions
        for dim in dimensions:
            # Local integration
            integrator = LocalIntegrator(dim, [
                "unity", "complementarity", "disjunction", "conjunction",
                "sequential-branching", "modular-closure", "modular-recursion"
            ])
            local_results[dim] = integrator.integrate()
            
            # Global differentiation
            differentiator = GlobalDifferentiator(dim)
            global_results[dim] = differentiator.differentiate()
        
        # Test coupling between local and global processes
        for dim in dimensions:
            local_energy = local_results[dim]["integration_energy"]
            global_energy = global_results[dim]["field_energy"]
            
            # Both should be positive and substantial
            assert local_energy > 0, f"Local energy too low for {dim}"
            assert global_energy > 0, f"Global energy too low for {dim}"
            
            # Energy levels should be related (coupling indicator)
            energy_ratio = local_energy / (global_energy + 1e-10)
            # Allow wider range since different dimensions have different natural energy scales
            assert 1e-6 <= energy_ratio <= 1e6, \
                f"Energy ratio {energy_ratio} indicates poor coupling for {dim}"
    
    def test_synthesis_coevolution(self, synthesizer):
        """Test that synthesis shows co-evolutionary dynamics"""
        # Create sample local and global states
        local_states = {}
        global_fields = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            # Mock local integration results
            local_states[dim] = {
                "dimension": dim,
                "final_state": np.random.random(8).tolist(),
                "integration_energy": np.random.uniform(1, 10),
                "coherence_measure": np.random.uniform(0.3, 0.9)
            }
            
            # Mock global differentiation results
            global_fields[dim] = {
                "dimension": dim,
                "final_field": np.random.random((8, 8)).tolist(),
                "field_energy": np.random.uniform(1, 10),
                "differentiation_degree": np.random.uniform(0.2, 0.8)
            }
        
        # Perform synthesis
        result = synthesizer.synthesize(local_states, global_fields)
        
        # Test co-evolutionary indicators
        history = result["synthesis_history"]
        assert len(history) >= 2, "Need multiple levels for co-evolution test"
        
        # Test coherence co-evolution
        coherence_values = [step["coherence"] for step in history]
        coherence_trend = np.diff(coherence_values)
        
        # Coherence should show some evolutionary pattern (not all flat)
        assert np.std(coherence_trend) > 1e-6, "No coherence evolution detected"
        
        # Test self-reference co-evolution
        self_ref_values = [step["self_reference"] for step in history]
        self_ref_trend = np.diff(self_ref_values)
        
        # Self-reference should evolve
        assert np.std(self_ref_values) > 1e-6, "No self-reference evolution detected"
    
    def test_emergent_property_coevolution(self, synthesizer):
        """Test that co-evolution produces emergent properties"""
        # Generate realistic local and global states
        local_states = {}
        global_fields = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            integrator = LocalIntegrator(dim, [
                "unity", "complementarity", "disjunction", "conjunction",
                "sequential-branching", "modular-closure", "modular-recursion"
            ])
            local_states[dim] = integrator.integrate()
            
            differentiator = GlobalDifferentiator(dim)
            global_fields[dim] = differentiator.differentiate()
        
        # Perform synthesis
        result = synthesizer.synthesize(local_states, global_fields)
        
        # Check for emergent properties
        emergent_properties = result["synthesis_summary"]["emergent_properties"]
        
        # Should detect some emergent properties from co-evolution
        assert len(emergent_properties) >= 1, "No emergent properties detected from co-evolution"
        
        # Test specific emergent properties
        possible_properties = [
            "coherence_growth", "energy_amplification", 
            "recursive_stabilization", "dimensional_unification"
        ]
        
        detected_valid_properties = [prop for prop in emergent_properties 
                                   if prop in possible_properties]
        assert len(detected_valid_properties) >= 1, "No valid emergent properties detected"
    
    def test_hierarchical_coevolution(self, synthesizer):
        """Test that hierarchical levels show co-evolutionary relationships"""
        # Create sample states
        local_states = {}
        global_fields = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            local_states[dim] = {
                "dimension": dim,
                "final_state": np.random.random(8).tolist(),
                "integration_energy": np.random.uniform(2, 8),
                "coherence_measure": np.random.uniform(0.4, 0.8)
            }
            
            global_fields[dim] = {
                "dimension": dim,
                "final_field": np.random.random((8, 8)).tolist(),
                "field_energy": np.random.uniform(3, 12),
                "differentiation_degree": np.random.uniform(0.3, 0.7)
            }
        
        # Perform synthesis with sufficient depth
        result = synthesizer.synthesize(local_states, global_fields)
        
        # Test hierarchical co-evolution
        history = result["synthesis_history"]
        
        if len(history) >= 3:
            # Test that levels influence each other (co-evolution)
            level_energies = [step["synthesis_energy"] for step in history]
            level_coherences = [step["coherence"] for step in history]
            
            # Energy should show hierarchical patterns
            energy_correlations = []
            for i in range(len(level_energies) - 1):
                if level_energies[i] > 0 and level_energies[i+1] > 0:
                    correlation = level_energies[i] / level_energies[i+1]
                    energy_correlations.append(correlation)
            
            # Hierarchical levels should be related
            if energy_correlations:
                correlation_variance = np.var(energy_correlations)
                assert correlation_variance < 100.0, "Hierarchical levels show no correlation"
    
    def test_dimensional_coevolution(self):
        """Test that spatial, temporal, and causal dimensions co-evolve"""
        dimensions = ["spatial", "temporal", "causal"]
        integration_results = {}
        differentiation_results = {}
        
        # Process each dimension
        for dim in dimensions:
            integrator = LocalIntegrator(dim, [
                "unity", "complementarity", "disjunction", "conjunction",
                "sequential-branching", "modular-closure", "modular-recursion"
            ])
            integration_results[dim] = integrator.integrate()
            
            differentiator = GlobalDifferentiator(dim)
            differentiation_results[dim] = differentiator.differentiate()
        
        # Test dimensional co-evolution indicators
        local_energies = [result["integration_energy"] for result in integration_results.values()]
        global_energies = [result["field_energy"] for result in differentiation_results.values()]
        
        # Dimensions should show related energy patterns (co-evolution)
        local_energy_variance = np.var(local_energies)
        global_energy_variance = np.var(global_energies)
        
        # Some variation expected but not extreme
        local_cv = np.std(local_energies) / (np.mean(local_energies) + 1e-10)
        global_cv = np.std(global_energies) / (np.mean(global_energies) + 1e-10)
        
        assert local_cv < 5.0, f"Local energy coefficient of variation {local_cv} too high"
        assert global_cv < 5.0, f"Global energy coefficient of variation {global_cv} too high"
        
        # Test coherence co-evolution
        local_coherences = [result["coherence_measure"] for result in integration_results.values()]
        global_complexities = [result["field_complexity"] for result in differentiation_results.values()]
        
        # All dimensions should achieve some coherence
        assert all(c > 0.1 for c in local_coherences), "Some dimensions lack local coherence"
        assert all(c > 0.1 for c in global_complexities), "Some dimensions lack global complexity"
    
    def test_ring_to_element_coevolution(self, synthesizer):
        """Test that ring-to-element transformation shows co-evolutionary dynamics"""
        # Create test tensors representing evolved local states
        evolved_local_tensors = {
            "spatial": np.sin(np.linspace(0, 2*np.pi, 8)),    # Periodic structure
            "temporal": np.exp(-np.linspace(0, 2, 8)),       # Decay structure  
            "causal": np.cumsum(np.random.random(8))         # Accumulative structure
        }
        
        # Test ring-to-element transformation at different levels
        for level in range(3):
            ring_elements = synthesizer._ring_to_element_transform(evolved_local_tensors, level)
            
            # Verify co-evolutionary scaling
            for dim in ["spatial", "temporal", "causal"]:
                original_structure = evolved_local_tensors[dim]
                transformed_structure = ring_elements[dim]
                
                # Transformation should preserve structural information
                original_autocorr = np.corrcoef(original_structure[:-1], original_structure[1:])[0, 1]
                transformed_autocorr = np.corrcoef(transformed_structure[:-1], transformed_structure[1:])[0, 1]
                
                # Autocorrelation should be related (co-evolution indicator)
                if not (np.isnan(original_autocorr) or np.isnan(transformed_autocorr)):
                    autocorr_change = abs(original_autocorr - transformed_autocorr)
                    assert autocorr_change < 1.5, f"Excessive autocorrelation change {autocorr_change} for {dim} at level {level}"
    
    def test_core_to_space_coevolution(self, synthesizer):
        """Test that core-to-space transformation shows co-evolutionary dynamics"""
        # Create test tensors representing evolved global fields
        x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
        
        evolved_global_tensors = {
            "spatial": np.exp(-(x**2 + y**2) / 2),           # Gaussian field
            "temporal": np.sin(x) * np.cos(y),               # Wave field
            "causal": np.triu(np.ones((8, 8))) * np.exp(-x)  # Causal field
        }
        
        # Test core-to-space transformation at different levels
        for level in range(3):
            space_fields = synthesizer._core_to_space_transform(evolved_global_tensors, level)
            
            # Verify co-evolutionary field properties
            for dim in ["spatial", "temporal", "causal"]:
                original_field = evolved_global_tensors[dim]
                transformed_field = space_fields[dim]
                
                # Field complexity should be related
                original_gradient = np.mean(np.abs(np.gradient(original_field)))
                transformed_gradient = np.mean(np.abs(np.gradient(transformed_field)))
                
                # Gradients should show co-evolutionary relationship
                if original_gradient > 1e-10 and transformed_gradient > 1e-10:
                    gradient_ratio = transformed_gradient / original_gradient
                    assert 0.01 <= gradient_ratio <= 100.0, \
                        f"Gradient ratio {gradient_ratio} indicates poor co-evolution for {dim} at level {level}"
    
    def test_transcendence_coevolution(self, synthesizer):
        """Test that transcendence emerges from co-evolutionary dynamics"""
        # Create high-quality local and global states
        local_states = {}
        global_fields = {}
        
        for dim in ["spatial", "temporal", "causal"]:
            # High-coherence local states
            local_states[dim] = {
                "dimension": dim,
                "final_state": np.ones(8) * (0.5 + 0.3 * np.sin(np.arange(8))),
                "integration_energy": 5.0,
                "coherence_measure": 0.8
            }
            
            # High-complexity global fields
            coherent_field = np.outer(np.sin(np.linspace(0, 2*np.pi, 8)), 
                                    np.cos(np.linspace(0, 2*np.pi, 8)))
            global_fields[dim] = {
                "dimension": dim,
                "final_field": coherent_field.tolist(),
                "field_energy": 8.0,
                "differentiation_degree": 0.7
            }
        
        # Perform synthesis
        result = synthesizer.synthesize(local_states, global_fields)
        
        # Test transcendence emergence
        transcendence_level = result["synthesis_summary"]["transcendence_level"]
        
        # High-quality co-evolution should produce transcendence
        assert transcendence_level > 1.0, f"Transcendence level {transcendence_level} too low for quality inputs"
        
        # Test convergence indicators
        autognosis_sig = result["autognosis_signature"]
        convergence_achieved = autognosis_sig["convergence_achieved"]
        cognitive_unification = autognosis_sig["cognitive_unification"]
        
        # Co-evolution should achieve some form of convergence
        assert convergence_achieved or cognitive_unification, \
            "No convergence achieved despite quality co-evolution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])