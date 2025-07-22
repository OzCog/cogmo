#!/usr/bin/env python3
"""
Workflow Orchestration Validation Script

This script validates that the CircleCI workflow orchestration 
matches the requirements specified in issue #5.
"""

import yaml
import sys
from typing import Dict, List, Set


def load_circleci_config(config_path: str = '.circleci/config.yml') -> Dict:
    """Load and parse the CircleCI configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading CircleCI config: {e}")
        sys.exit(1)


def extract_workflow_jobs(config: Dict) -> Dict[str, List[str]]:
    """Extract job dependencies from the workflow configuration."""
    workflows = config.get('workflows', {})
    build_workflow = workflows.get('opencog-ecosystem-build', {})
    jobs = build_workflow.get('jobs', [])
    
    job_dependencies = {}
    
    for job in jobs:
        if isinstance(job, str):
            # Job with no dependencies
            job_dependencies[job] = []
        elif isinstance(job, dict):
            # Job with dependencies
            for job_name, job_config in job.items():
                requires = job_config.get('requires', [])
                job_dependencies[job_name] = requires
    
    return job_dependencies


def validate_layer_structure(job_dependencies: Dict[str, List[str]]) -> bool:
    """Validate that the workflow structure matches the expected layer architecture."""
    
    # Expected structure from issue #5
    expected_structure = {
        # Foundation Layer
        'cogutil': [],
        
        # Core Layer  
        'atomspace': ['cogutil'],
        'atomspace-rocks': ['atomspace'],
        'atomspace-restful': ['atomspace'],
        
        # Logic Layer
        'unify': ['atomspace'],
        'ure': ['atomspace', 'unify'],
        
        # Cognitive Systems Layer
        'cogserver': ['atomspace'],
        'attention': ['atomspace', 'cogserver'],
        'spacetime': ['atomspace'],
        
        # Advanced Systems Layer
        'pln': ['atomspace', 'ure', 'spacetime'],
        'miner': ['atomspace', 'ure'],
        
        # Learning Systems Layer
        'moses': ['cogutil'],
        'asmoses': ['atomspace', 'ure'],
        
        # Language Processing Layer
        'lg-atomese': ['atomspace'],
        'learn': ['atomspace', 'cogserver'],
        'language-learning': ['cogutil'],
        
        # Integration Layer
        'opencog': ['atomspace', 'cogserver', 'attention', 'ure', 'lg-atomese'],
        
        # Packaging
        'package': ['opencog']
    }
    
    validation_passed = True
    
    print("=== Workflow Orchestration Validation ===\n")
    
    # Check each expected job
    for job_name, expected_deps in expected_structure.items():
        if job_name not in job_dependencies:
            print(f"❌ MISSING: Job '{job_name}' not found in workflow")
            validation_passed = False
            continue
            
        actual_deps = set(job_dependencies[job_name])
        expected_deps_set = set(expected_deps)
        
        if actual_deps == expected_deps_set:
            print(f"✅ {job_name}: Dependencies match ({sorted(expected_deps)})")
        else:
            print(f"❌ {job_name}: Dependencies mismatch")
            print(f"   Expected: {sorted(expected_deps)}")
            print(f"   Actual: {sorted(actual_deps)}")
            validation_passed = False
    
    # Check for unexpected jobs
    expected_jobs = set(expected_structure.keys())
    actual_jobs = set(job_dependencies.keys())
    unexpected_jobs = actual_jobs - expected_jobs
    
    if unexpected_jobs:
        print(f"\n⚠️  Additional jobs found: {sorted(unexpected_jobs)}")
    
    return validation_passed


def validate_dependency_cycles(job_dependencies: Dict[str, List[str]]) -> bool:
    """Check for circular dependencies in the workflow."""
    
    def has_cycle(job: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        visited.add(job)
        rec_stack.add(job)
        
        for dep in job_dependencies.get(job, []):
            if dep not in visited:
                if has_cycle(dep, visited, rec_stack):
                    return True
            elif dep in rec_stack:
                return True
                
        rec_stack.remove(job)
        return False
    
    visited = set()
    for job in job_dependencies:
        if job not in visited:
            if has_cycle(job, visited, set()):
                print(f"❌ Circular dependency detected involving job: {job}")
                return False
    
    print("✅ No circular dependencies detected")
    return True


def main():
    """Main validation function."""
    print("CircleCI Workflow Orchestration Validation")
    print("=" * 50)
    
    # Load configuration
    config = load_circleci_config()
    
    # Extract job dependencies
    job_dependencies = extract_workflow_jobs(config)
    
    print(f"Found {len(job_dependencies)} jobs in workflow\n")
    
    # Validate structure
    structure_valid = validate_layer_structure(job_dependencies)
    
    print("\n=== Dependency Cycle Check ===")
    cycle_free = validate_dependency_cycles(job_dependencies)
    
    # Final result
    print("\n" + "=" * 50)
    if structure_valid and cycle_free:
        print("🎉 VALIDATION PASSED: Workflow orchestration is correctly implemented!")
        print("✅ All components and dependencies match the requirements")
        print("✅ No circular dependencies found")
        return 0
    else:
        print("❌ VALIDATION FAILED: Issues found in workflow orchestration")
        return 1


if __name__ == '__main__':
    sys.exit(main())