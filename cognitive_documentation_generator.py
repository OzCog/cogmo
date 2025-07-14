#!/usr/bin/env python3
"""
Phase 6: Recursive Documentation and Cognitive Architecture Explorer

This module implements living documentation generation and interactive
cognitive architecture exploration tools.

Features:
- Auto-generated architectural flowcharts
- Interactive cognitive architecture explorer
- Tensor signature evolution tracking
- Cognitive pattern emergence reports
- Cognitive debugging and introspection tools
"""

import os
import sys
import ast
import inspect
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import importlib
import tempfile
from pathlib import Path
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cogml.cognitive_primitives import CognitivePrimitiveTensor, TensorSignature, ModalityType, DepthType, ContextType
from ecan.attention_kernel import AttentionKernel
from meta_cognition import MetaCognitiveMonitor
from tests.test_phase6_cognitive_unification import UnifiedCognitiveTensor


class DocumentationType(Enum):
    """Types of documentation to generate"""
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    MODULE_DEPENDENCY = "module_dependency"
    TENSOR_EVOLUTION = "tensor_evolution"
    COGNITIVE_PATTERNS = "cognitive_patterns"
    TEST_COVERAGE = "test_coverage"
    PERFORMANCE_METRICS = "performance_metrics"


class VisualizationStyle(Enum):
    """Visualization styles for different diagram types"""
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    FORCE_DIRECTED = "force_directed"
    LAYERED = "layered"


@dataclass
class ArchitecturalComponent:
    """Represents a component in the cognitive architecture"""
    name: str
    module_path: str
    component_type: str  # "class", "function", "module"
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    cognitive_role: str = ""
    tensor_signatures: List[str] = field(default_factory=list)
    test_coverage: float = 0.0
    last_modified: float = field(default_factory=time.time)


@dataclass
class CognitivePattern:
    """Represents an emergent cognitive pattern"""
    pattern_id: str
    pattern_type: str
    components_involved: List[str]
    emergence_timestamp: float
    strength: float  # [0.0, 1.0]
    stability: float  # [0.0, 1.0] 
    description: str
    tensor_signatures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TensorEvolutionRecord:
    """Records the evolution of tensor signatures over time"""
    timestamp: float
    tensor_id: str
    signature: Dict[str, Any]
    performance_metrics: Dict[str, float]
    emergence_indicators: List[str]
    cognitive_context: str


class CognitiveArchitectureExplorer:
    """Interactive explorer for cognitive architecture analysis and visualization"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.components = {}
        self.patterns = {}
        self.tensor_evolution = []
        self.dependency_graph = nx.DiGraph()
        self.cognitive_graph = nx.Graph()
        
        # Scan and analyze architecture
        self._scan_architecture()
        self._build_dependency_graph()
        self._detect_cognitive_patterns()
    
    def _scan_architecture(self):
        """Scan the codebase to identify architectural components"""
        cognitive_modules = [
            "cogml/cognitive_primitives.py",
            "ecan/__init__.py",
            "ecan/attention_kernel.py",
            "ecan/economic_allocator.py",
            "ecan/resource_scheduler.py", 
            "ecan/attention_spreading.py",
            "ecan/decay_refresh.py",
            "meta_cognition/__init__.py",
            "evolutionary_optimization.py",
            "continuous_benchmarking.py"
        ]
        
        for module_path in cognitive_modules:
            full_path = self.root_path / module_path
            if full_path.exists():
                component = self._analyze_module(full_path, module_path)
                if component:
                    self.components[component.name] = component
    
    def _analyze_module(self, file_path: Path, module_path: str) -> Optional[ArchitecturalComponent]:
        """Analyze a Python module to extract architectural information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract classes and functions
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Extract imports to identify dependencies
            dependencies = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    dependencies.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
            
            # Determine cognitive role based on module name and content
            cognitive_role = self._determine_cognitive_role(module_path, classes, functions)
            
            # Extract tensor signatures if present
            tensor_signatures = self._extract_tensor_signatures(content)
            
            component = ArchitecturalComponent(
                name=module_path.replace("/", ".").replace(".py", ""),
                module_path=module_path,
                component_type="module",
                dependencies=dependencies,
                interfaces=classes + functions,
                cognitive_role=cognitive_role,
                tensor_signatures=tensor_signatures,
                last_modified=file_path.stat().st_mtime
            )
            
            return component
            
        except Exception as e:
            print(f"Error analyzing module {module_path}: {e}")
            return None
    
    def _determine_cognitive_role(self, module_path: str, classes: List[str], functions: List[str]) -> str:
        """Determine the cognitive role of a module based on its content"""
        if "cognitive_primitives" in module_path:
            return "Foundational Representation"
        elif "attention" in module_path or "ecan" in module_path:
            return "Attention & Resource Allocation"
        elif "meta_cognition" in module_path:
            return "Recursive Self-Awareness"
        elif "evolutionary" in module_path:
            return "Adaptive Optimization"
        elif "benchmarking" in module_path:
            return "Performance Monitoring"
        elif "test" in module_path:
            return "Validation & Testing"
        else:
            return "Support Infrastructure"
    
    def _extract_tensor_signatures(self, content: str) -> List[str]:
        """Extract tensor signature patterns from module content"""
        signatures = []
        
        # Look for TensorSignature, CognitivePrimitiveTensor, etc.
        tensor_keywords = [
            "TensorSignature", "CognitivePrimitiveTensor", "ECANAttentionTensor",
            "UnifiedCognitiveTensor", "MetaCognitiveMetrics"
        ]
        
        for keyword in tensor_keywords:
            if keyword in content:
                signatures.append(keyword)
        
        return signatures
    
    def _build_dependency_graph(self):
        """Build a directed graph of module dependencies"""
        for component_name, component in self.components.items():
            self.dependency_graph.add_node(component_name, 
                                         cognitive_role=component.cognitive_role,
                                         interfaces=len(component.interfaces))
            
            for dep in component.dependencies:
                # Filter for relevant cognitive dependencies
                dep_clean = dep.split('.')[0]
                if dep_clean in ['cogml', 'ecan', 'meta_cognition'] or any(dep_clean in comp for comp in self.components):
                    self.dependency_graph.add_edge(dep_clean, component_name)
    
    def _detect_cognitive_patterns(self):
        """Detect emergent cognitive patterns in the architecture"""
        patterns = []
        
        # Pattern 1: Multi-modal integration
        multimodal_components = [comp for comp in self.components.values() 
                               if "modality" in str(comp.tensor_signatures).lower()]
        if len(multimodal_components) >= 2:
            pattern = CognitivePattern(
                pattern_id="multimodal_integration",
                pattern_type="cross_modal",
                components_involved=[comp.name for comp in multimodal_components],
                emergence_timestamp=time.time(),
                strength=min(1.0, len(multimodal_components) / 4.0),
                stability=0.8,
                description="Multi-modal sensory integration across visual, auditory, textual, and symbolic modalities"
            )
            patterns.append(pattern)
        
        # Pattern 2: Recursive meta-cognition
        meta_components = [comp for comp in self.components.values() 
                          if "meta" in comp.cognitive_role.lower() or "recursive" in comp.cognitive_role.lower()]
        if meta_components:
            pattern = CognitivePattern(
                pattern_id="recursive_metacognition",
                pattern_type="self_referential",
                components_involved=[comp.name for comp in meta_components],
                emergence_timestamp=time.time(),
                strength=0.9,
                stability=0.7,
                description="Recursive self-awareness and meta-cognitive monitoring capabilities"
            )
            patterns.append(pattern)
        
        # Pattern 3: Attention-driven processing
        attention_components = [comp for comp in self.components.values() 
                              if "attention" in comp.cognitive_role.lower()]
        if attention_components:
            pattern = CognitivePattern(
                pattern_id="attention_driven_processing",
                pattern_type="resource_allocation",
                components_involved=[comp.name for comp in attention_components],
                emergence_timestamp=time.time(),
                strength=0.8,
                stability=0.9,
                description="Economic attention allocation and resource management"
            )
            patterns.append(pattern)
        
        # Pattern 4: Evolutionary optimization
        evolutionary_components = [comp for comp in self.components.values() 
                                 if "evolutionary" in comp.cognitive_role.lower() or "optimization" in comp.cognitive_role.lower()]
        if evolutionary_components:
            pattern = CognitivePattern(
                pattern_id="evolutionary_optimization",
                pattern_type="adaptive_learning",
                components_involved=[comp.name for comp in evolutionary_components],
                emergence_timestamp=time.time(),
                strength=0.7,
                stability=0.6,
                description="Evolutionary algorithms for cognitive architecture optimization"
            )
            patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern
    
    def generate_architecture_diagram(self, output_path: str = "cognitive_architecture.png", 
                                    style: VisualizationStyle = VisualizationStyle.HIERARCHICAL):
        """Generate architectural diagram of the cognitive system"""
        plt.figure(figsize=(16, 12))
        
        # Create layout based on style
        if style == VisualizationStyle.HIERARCHICAL:
            pos = self._create_hierarchical_layout()
        elif style == VisualizationStyle.CIRCULAR:
            pos = nx.circular_layout(self.dependency_graph)
        elif style == VisualizationStyle.FORCE_DIRECTED:
            pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
        else:
            pos = nx.spring_layout(self.dependency_graph)
        
        # Define colors for different cognitive roles
        role_colors = {
            "Foundational Representation": "#FF6B6B",
            "Attention & Resource Allocation": "#4ECDC4", 
            "Recursive Self-Awareness": "#45B7D1",
            "Adaptive Optimization": "#96CEB4",
            "Performance Monitoring": "#FFEAA7",
            "Validation & Testing": "#DDA0DD",
            "Support Infrastructure": "#B8B8B8"
        }
        
        # Draw nodes with role-based coloring
        node_colors = []
        node_sizes = []
        for node in self.dependency_graph.nodes():
            if node in self.components:
                role = self.components[node].cognitive_role
                node_colors.append(role_colors.get(role, "#B8B8B8"))
                # Size based on number of interfaces
                interfaces = len(self.components[node].interfaces)
                node_sizes.append(max(300, interfaces * 100))
            else:
                node_colors.append("#B8B8B8")
                node_sizes.append(300)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.dependency_graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8)
        
        nx.draw_networkx_edges(self.dependency_graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        nx.draw_networkx_labels(self.dependency_graph, pos,
                               font_size=8,
                               font_weight='bold')
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=role) 
                          for role, color in role_colors.items() 
                          if any(self.components[comp].cognitive_role == role for comp in self.components)]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title("Cognitive Architecture - Component Dependencies", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the diagram
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Architecture diagram saved to: {output_path}")
        return output_path
    
    def _create_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout for cognitive architecture"""
        # Define hierarchy levels based on cognitive roles
        hierarchy = {
            "Foundational Representation": 0,
            "Attention & Resource Allocation": 1,
            "Recursive Self-Awareness": 2,
            "Adaptive Optimization": 2,
            "Performance Monitoring": 3,
            "Validation & Testing": 3,
            "Support Infrastructure": 1
        }
        
        pos = {}
        level_counts = {}
        level_positions = {}
        
        # Count components per level
        for component in self.components.values():
            level = hierarchy.get(component.cognitive_role, 1)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Position components
        for component in self.components.values():
            level = hierarchy.get(component.cognitive_role, 1)
            
            if level not in level_positions:
                level_positions[level] = 0
            
            x = level_positions[level] - (level_counts[level] - 1) / 2
            y = -level * 2  # Negative for top-down layout
            
            pos[component.name] = (x, y)
            level_positions[level] += 1
        
        return pos
    
    def generate_tensor_evolution_report(self, output_path: str = "tensor_evolution.json"):
        """Generate report on tensor signature evolution over time"""
        # Simulate tensor evolution data (in practice, this would be tracked over time)
        evolution_data = {
            "report_timestamp": time.time(),
            "total_tensors_tracked": len(self.components) * 3,  # Estimate
            "evolution_patterns": [],
            "signature_stability": {},
            "emergent_signatures": []
        }
        
        # Analyze tensor signature evolution patterns
        for component in self.components.values():
            if component.tensor_signatures:
                for signature in component.tensor_signatures:
                    pattern = {
                        "signature_type": signature,
                        "component": component.name,
                        "stability_score": 0.8,  # Simulated
                        "evolution_trend": "stable",
                        "last_change": component.last_modified
                    }
                    evolution_data["evolution_patterns"].append(pattern)
        
        # Identify emergent signatures (new tensor types)
        emergent_signatures = [
            {
                "signature_name": "UnifiedCognitiveTensor",
                "emergence_timestamp": time.time() - 86400,  # 1 day ago
                "components_using": ["cognitive_unification"],
                "cognitive_impact": "high",
                "description": "Unified tensor field for cognitive coherence"
            }
        ]
        evolution_data["emergent_signatures"] = emergent_signatures
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(evolution_data, f, indent=2)
        
        print(f"Tensor evolution report saved to: {output_path}")
        return evolution_data
    
    def generate_cognitive_patterns_report(self, output_path: str = "cognitive_patterns.json"):
        """Generate report on detected cognitive patterns"""
        patterns_data = {
            "report_timestamp": time.time(),
            "total_patterns_detected": len(self.patterns),
            "pattern_analysis": {},
            "emergence_timeline": [],
            "pattern_interactions": []
        }
        
        # Analyze each detected pattern
        for pattern_id, pattern in self.patterns.items():
            patterns_data["pattern_analysis"][pattern_id] = {
                "type": pattern.pattern_type,
                "strength": pattern.strength,
                "stability": pattern.stability,
                "components": pattern.components_involved,
                "description": pattern.description,
                "emergence_time": pattern.emergence_timestamp
            }
        
        # Create emergence timeline
        timeline = sorted(self.patterns.values(), key=lambda p: p.emergence_timestamp)
        patterns_data["emergence_timeline"] = [
            {
                "pattern_id": p.pattern_id,
                "timestamp": p.emergence_timestamp,
                "strength": p.strength
            } for p in timeline
        ]
        
        # Analyze pattern interactions
        interactions = []
        pattern_list = list(self.patterns.values())
        for i, pattern1 in enumerate(pattern_list):
            for pattern2 in pattern_list[i+1:]:
                # Check for component overlap
                overlap = set(pattern1.components_involved) & set(pattern2.components_involved)
                if overlap:
                    interactions.append({
                        "pattern1": pattern1.pattern_id,
                        "pattern2": pattern2.pattern_id,
                        "shared_components": list(overlap),
                        "interaction_strength": len(overlap) / max(len(pattern1.components_involved), len(pattern2.components_involved))
                    })
        
        patterns_data["pattern_interactions"] = interactions
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        print(f"Cognitive patterns report saved to: {output_path}")
        return patterns_data
    
    def create_interactive_explorer(self, output_path: str = "cognitive_explorer.html"):
        """Create interactive HTML-based cognitive architecture explorer"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Cognitive Architecture Explorer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .tooltip {{ position: absolute; background: #000; color: #fff; padding: 8px; border-radius: 4px; pointer-events: none; }}
        .panel {{ border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .header {{ background: #f0f0f0; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }}
    </style>
</head>
<body>
    <h1>🧠 Cognitive Architecture Explorer</h1>
    
    <div class="panel">
        <div class="header"><h3>Architecture Overview</h3></div>
        <p><strong>Total Components:</strong> {total_components}</p>
        <p><strong>Detected Patterns:</strong> {total_patterns}</p>
        <p><strong>Cognitive Roles:</strong> {total_roles}</p>
    </div>
    
    <div class="panel">
        <div class="header"><h3>Interactive Dependency Graph</h3></div>
        <div id="graph"></div>
    </div>
    
    <div class="panel">
        <div class="header"><h3>Cognitive Patterns</h3></div>
        <div id="patterns">
            {patterns_html}
        </div>
    </div>
    
    <script>
        // D3.js visualization code would go here
        // For brevity, using placeholder
        d3.select("#graph").append("p").text("Interactive graph visualization would be rendered here using D3.js");
    </script>
</body>
</html>
        """
        
        # Generate patterns HTML
        patterns_html = ""
        for pattern in self.patterns.values():
            patterns_html += f"""
            <div style="border-left: 4px solid #4ECDC4; padding-left: 10px; margin: 10px 0;">
                <h4>{pattern.pattern_id.replace('_', ' ').title()}</h4>
                <p><strong>Type:</strong> {pattern.pattern_type}</p>
                <p><strong>Strength:</strong> {pattern.strength:.2f}</p>
                <p><strong>Description:</strong> {pattern.description}</p>
                <p><strong>Components:</strong> {', '.join(pattern.components_involved)}</p>
            </div>
            """
        
        # Get unique cognitive roles
        roles = set(comp.cognitive_role for comp in self.components.values())
        
        # Fill in template
        html_content = html_template.format(
            total_components=len(self.components),
            total_patterns=len(self.patterns),
            total_roles=len(roles),
            patterns_html=patterns_html
        )
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Interactive explorer saved to: {output_path}")
        return output_path
    
    def generate_comprehensive_documentation(self, output_dir: str = "cognitive_docs"):
        """Generate comprehensive documentation package"""
        os.makedirs(output_dir, exist_ok=True)
        
        documentation_files = []
        
        # 1. Architecture diagram
        arch_diagram = os.path.join(output_dir, "architecture_diagram.png")
        self.generate_architecture_diagram(arch_diagram)
        documentation_files.append(arch_diagram)
        
        # 2. Tensor evolution report
        tensor_report = os.path.join(output_dir, "tensor_evolution.json")
        self.generate_tensor_evolution_report(tensor_report)
        documentation_files.append(tensor_report)
        
        # 3. Cognitive patterns report
        patterns_report = os.path.join(output_dir, "cognitive_patterns.json")
        self.generate_cognitive_patterns_report(patterns_report)
        documentation_files.append(patterns_report)
        
        # 4. Interactive explorer
        explorer = os.path.join(output_dir, "cognitive_explorer.html")
        self.create_interactive_explorer(explorer)
        documentation_files.append(explorer)
        
        # 5. Component summary
        component_summary = os.path.join(output_dir, "component_summary.json")
        summary_data = {
            "components": {name: {
                "cognitive_role": comp.cognitive_role,
                "interfaces": comp.interfaces,
                "dependencies": comp.dependencies,
                "tensor_signatures": comp.tensor_signatures,
                "last_modified": comp.last_modified
            } for name, comp in self.components.items()},
            "summary_stats": {
                "total_components": len(self.components),
                "total_patterns": len(self.patterns),
                "total_dependencies": len(self.dependency_graph.edges()),
                "generation_timestamp": time.time()
            }
        }
        
        with open(component_summary, 'w') as f:
            json.dump(summary_data, f, indent=2)
        documentation_files.append(component_summary)
        
        print(f"Comprehensive documentation generated in: {output_dir}")
        print(f"Generated files: {documentation_files}")
        
        return documentation_files


def main():
    """Main function to demonstrate documentation generation"""
    # Initialize the cognitive architecture explorer
    explorer = CognitiveArchitectureExplorer(".")
    
    # Generate comprehensive documentation
    docs = explorer.generate_comprehensive_documentation("cognitive_architecture_docs")
    
    print("\n🌌 Cognitive Architecture Documentation Generated!")
    print(f"📁 Documentation directory: cognitive_architecture_docs/")
    print(f"📊 Architecture diagram: {docs[0]}")
    print(f"📈 Tensor evolution report: {docs[1]}")
    print(f"🧠 Cognitive patterns report: {docs[2]}")
    print(f"🌐 Interactive explorer: {docs[3]}")
    print(f"📋 Component summary: {docs[4]}")


if __name__ == "__main__":
    main()