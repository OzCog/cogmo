#!/usr/bin/env python3
"""
Global Differentiation Simulator for Autognosis Hierarchical Self-Image Building System

This module implements the top-down differentiation phase that creates global fields
and differentiates space into complementary structures across spatial, temporal,
and causal dimensions.
"""

import numpy as np
import argparse
import json
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class DimensionType(Enum):
    """Dimensional categories for cognitive processing"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal" 
    CAUSAL = "causal"


class FieldType(Enum):
    """Types of global fields in differentiation"""
    UNDIFFERENTIATED = "undifferentiated"
    POLARIZED = "polarized"
    COMPARTMENTALIZED = "compartmentalized"
    HIERARCHICAL = "hierarchical"


@dataclass
class GlobalFieldState:
    """State representation for global field differentiation"""
    dimension: DimensionType
    field_type: FieldType
    field_tensor: np.ndarray
    differentiation_history: List[Dict]
    field_energy: float


class GlobalDifferentiator:
    """
    Global differentiation simulator implementing top-down field emergence
    """
    
    def __init__(self, dimension: str):
        self.dimension = DimensionType(dimension)
        self.field_tensor = np.ones((8, 8))  # 8x8 global field
        self.differentiation_history = []
        
        print(f"🌌 GlobalDifferentiator initialized for {dimension} dimension")
        print(f"   Initial field shape: {self.field_tensor.shape}")
    
    def space_differentiation(self) -> np.ndarray:
        """
        Global spatial field emergence - create undifferentiated global space
        """
        print("   🌐 Space Differentiation: Global field emergence")
        
        # Create initial global field based on dimension
        if self.dimension == DimensionType.SPATIAL:
            # Spatial field with distance-based gradients
            x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
            self.field_tensor = np.exp(-(x**2 + y**2) / 2)
            
        elif self.dimension == DimensionType.TEMPORAL:
            # Temporal field with wave-like patterns
            t = np.linspace(0, 4*np.pi, 8)
            temporal_wave = np.sin(t)
            self.field_tensor = np.outer(temporal_wave, temporal_wave)
            
        elif self.dimension == DimensionType.CAUSAL:
            # Causal field with hierarchical causation
            causal_hierarchy = np.triu(np.ones((8, 8)))  # Upper triangular
            self.field_tensor = causal_hierarchy * np.exp(-np.arange(8).reshape(-1, 1) / 4)
        
        # Normalize field
        self.field_tensor = self.field_tensor / np.max(self.field_tensor)
        
        self._record_field_state("undifferentiated", self.field_tensor)
        return self.field_tensor
    
    def polarized_space_formation(self) -> np.ndarray:
        """
        Complementary field dynamics - create polarized structures
        """
        print("   ⚡ Polarized Space Formation: Complementary field dynamics")
        
        # Create complementary polarization
        if self.dimension == DimensionType.SPATIAL:
            # Spatial dipole field
            x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
            positive_pole = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.5)
            negative_pole = np.exp(-((x+0.5)**2 + (y+0.5)**2) / 0.5)
            self.field_tensor = positive_pole - negative_pole
            
        elif self.dimension == DimensionType.TEMPORAL:
            # Temporal oscillation with phase shifts
            t = np.linspace(0, 4*np.pi, 8)
            phase_matrix = np.outer(t, np.ones(8)) + np.outer(np.ones(8), t)
            self.field_tensor = np.sin(phase_matrix) * np.cos(phase_matrix / 2)
            
        elif self.dimension == DimensionType.CAUSAL:
            # Causal feedback loops with positive/negative causation
            feedback_matrix = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    if i != j:
                        # Positive feedback for forward causation
                        if j > i:
                            feedback_matrix[i, j] = 1.0 / (j - i)
                        # Negative feedback for backward causation
                        else:
                            feedback_matrix[i, j] = -1.0 / (i - j + 1)
            self.field_tensor = feedback_matrix
        
        # Normalize to [-1, 1] range
        field_max = np.max(np.abs(self.field_tensor))
        if field_max > 0:
            self.field_tensor = self.field_tensor / field_max
        
        self._record_field_state("polarized", self.field_tensor)
        return self.field_tensor
    
    def boundary_emergence(self) -> np.ndarray:
        """
        Global compartmentalization - create bounded regions
        """
        print("   🔳 Boundary Emergence: Global compartmentalization")
        
        # Create compartmentalized structure
        if self.dimension == DimensionType.SPATIAL:
            # Spatial Voronoi-like regions
            compartments = np.zeros((8, 8))
            centers = [(2, 2), (2, 6), (6, 2), (6, 6)]  # 4 compartment centers
            
            for i in range(8):
                for j in range(8):
                    distances = [np.sqrt((i-cx)**2 + (j-cy)**2) for cx, cy in centers]
                    closest_center = np.argmin(distances)
                    compartments[i, j] = closest_center + 1
            
            # Create smooth boundaries with sigmoid transitions
            self.field_tensor = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    region_value = compartments[i, j]
                    boundary_strength = 0
                    
                    # Check neighboring cells for boundaries
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 8 and 0 <= nj < 8:
                                if compartments[ni, nj] != region_value:
                                    boundary_strength += 1
                    
                    self.field_tensor[i, j] = region_value * (1 - boundary_strength / 8)
            
        elif self.dimension == DimensionType.TEMPORAL:
            # Temporal phase boundaries
            t = np.linspace(0, 4*np.pi, 8)
            phase_boundaries = np.zeros((8, 8))
            
            for i in range(8):
                for j in range(8):
                    phase_diff = abs(t[i] - t[j])
                    # Create boundaries at phase transitions
                    if phase_diff % (np.pi/2) < 0.5:
                        phase_boundaries[i, j] = 1.0
                    else:
                        phase_boundaries[i, j] = 0.2
            
            self.field_tensor = phase_boundaries
            
        elif self.dimension == DimensionType.CAUSAL:
            # Causal domains with event boundaries
            causal_domains = np.zeros((8, 8))
            
            # Create causal light cones
            for i in range(8):
                for j in range(8):
                    time_sep = abs(i - j)
                    if time_sep == 0:
                        causal_domains[i, j] = 1.0  # Self-causation
                    elif time_sep <= 2:
                        causal_domains[i, j] = 0.8  # Direct causation
                    elif time_sep <= 4:
                        causal_domains[i, j] = 0.4  # Indirect causation
                    else:
                        causal_domains[i, j] = 0.1  # Weak causation
            
            self.field_tensor = causal_domains
        
        # Apply boundary sharpening
        self.field_tensor = self._sharpen_boundaries(self.field_tensor)
        
        self._record_field_state("compartmentalized", self.field_tensor)
        return self.field_tensor
    
    def hierarchical_field_formation(self) -> np.ndarray:
        """
        Create hierarchical field structure with nested levels
        """
        print("   🏗️ Hierarchical Field Formation: Multi-level structure")
        
        # Create hierarchical levels
        hierarchical_field = np.zeros((8, 8))
        
        if self.dimension == DimensionType.SPATIAL:
            # Spatial hierarchy with nested scales
            for level in range(3):  # 3 hierarchical levels
                scale = 2 ** level
                resolution = 8 // scale
                
                level_field = np.zeros((8, 8))
                for i in range(0, 8, resolution):
                    for j in range(0, 8, resolution):
                        # Fill hierarchical block
                        value = (level + 1) * np.random.uniform(0.5, 1.0)
                        level_field[i:i+resolution, j:j+resolution] = value
                
                # Weight by hierarchical level
                hierarchical_field += level_field * (0.5 ** level)
        
        elif self.dimension == DimensionType.TEMPORAL:
            # Temporal hierarchy with nested rhythms
            for level in range(3):
                frequency = 2 ** level
                t = np.linspace(0, 4*np.pi, 8)
                temporal_rhythm = np.sin(frequency * t)
                
                # Create rhythm matrix
                level_field = np.outer(temporal_rhythm, temporal_rhythm)
                hierarchical_field += level_field * (0.6 ** level)
        
        elif self.dimension == DimensionType.CAUSAL:
            # Causal hierarchy with nested dependencies
            for level in range(3):
                causal_span = 2 ** (level + 1)
                level_field = np.zeros((8, 8))
                
                for i in range(8):
                    for j in range(max(0, i-causal_span), min(8, i+causal_span+1)):
                        if i != j:
                            causal_strength = 1.0 / (abs(i-j) + 1)
                            level_field[i, j] = causal_strength
                
                hierarchical_field += level_field * (0.7 ** level)
        
        self.field_tensor = hierarchical_field / np.max(hierarchical_field)
        
        self._record_field_state("hierarchical", self.field_tensor)
        return self.field_tensor
    
    def _sharpen_boundaries(self, field: np.ndarray) -> np.ndarray:
        """Apply boundary sharpening transformation"""
        # Apply Laplacian operator for edge detection
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        
        # Pad field for convolution
        padded_field = np.pad(field, 1, mode='edge')
        sharpened = np.zeros_like(field)
        
        for i in range(8):
            for j in range(8):
                # Apply Laplacian kernel
                region = padded_field[i:i+3, j:j+3]
                edge_strength = np.sum(region * laplacian)
                sharpened[i, j] = field[i, j] + 0.1 * edge_strength
        
        return np.clip(sharpened, 0, 1)
    
    def _record_field_state(self, field_type: str, field: np.ndarray):
        """Record field state for analysis"""
        self.differentiation_history.append({
            "field_type": field_type,
            "dimension": self.dimension.value,
            "field_energy": np.sum(field**2),
            "field_entropy": -np.sum(field * np.log(np.abs(field) + 1e-10)),
            "field_variance": np.var(field),
            "field_gradient": np.mean(np.gradient(field)),
            "field_signature": {
                "mean": np.mean(field),
                "std": np.std(field),
                "min": np.min(field),
                "max": np.max(field)
            }
        })
    
    def differentiate(self) -> Dict:
        """
        Execute complete global differentiation process
        """
        print(f"\n🌌 Starting Global Differentiation for {self.dimension.value} dimension")
        
        # Execute all differentiation phases
        self.space_differentiation()
        self.polarized_space_formation()
        self.boundary_emergence()
        self.hierarchical_field_formation()
        
        # Compute final differentiation metrics
        final_field = self.field_tensor
        
        result = {
            "dimension": self.dimension.value,
            "final_field": final_field.tolist(),
            "field_energy": np.sum(final_field**2),
            "differentiation_degree": np.var(final_field),
            "hierarchical_levels": 3,
            "field_complexity": -np.sum(final_field * np.log(np.abs(final_field) + 1e-10)),
            "history": self.differentiation_history,
            "field_signature": {
                "shape": final_field.shape,
                "rank": len(final_field.shape),
                "spectral_radius": np.max(np.real(np.linalg.eigvals(final_field))),
                "frobenius_norm": np.linalg.norm(final_field, 'fro')
            }
        }
        
        print(f"✅ Global Differentiation completed")
        print(f"   Field energy: {result['field_energy']:.4f}")
        print(f"   Differentiation degree: {result['differentiation_degree']:.4f}")
        print(f"   Field complexity: {result['field_complexity']:.4f}")
        
        return result


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Global Differentiation Simulator")
    parser.add_argument("--dimension", required=True, choices=["spatial", "temporal", "causal"],
                      help="Dimensional category")
    parser.add_argument("--output", default="global_differentiation_result.json",
                      help="Output file for results")
    
    args = parser.parse_args()
    
    # Create and run differentiator
    differentiator = GlobalDifferentiator(args.dimension)
    result = differentiator.differentiate()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()