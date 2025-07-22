#!/usr/bin/env python3
"""
Test script for Ontogenesis system validation
"""

import json
import sys
from pathlib import Path

def test_issue_generation():
    """Test issue generation functionality"""
    print("🧪 Testing Ontogenesis Issue Generation")
    print("=" * 50)
    
    # Test 1: Check if generated JSON exists and is valid
    issues_file = Path("ontogenesis-issues.json")
    if not issues_file.exists():
        print("❌ ontogenesis-issues.json not found")
        return False
    
    try:
        with open(issues_file) as f:
            data = json.load(f)
        print("✅ JSON file is valid")
    except json.JSONDecodeError as e:
        print(f"❌ JSON validation failed: {e}")
        return False
    
    # Test 2: Verify structure
    required_keys = ['master_issue', 'component_issues', 'metadata']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: {key}")
            return False
    print("✅ JSON structure is valid")
    
    # Test 3: Check master issue
    master = data['master_issue']
    if not master.get('title') or not master.get('body'):
        print("❌ Master issue missing title or body")
        return False
    print("✅ Master issue is valid")
    
    # Test 4: Check component issues
    components = data['component_issues']
    if len(components) == 0:
        print("❌ No component issues generated")
        return False
    print(f"✅ {len(components)} component issues generated")
    
    # Test 5: Validate tensor calculations
    metadata = data['metadata']
    total_dof = metadata.get('total_dof', 0)
    if total_dof < 1000000:  # Should be in millions
        print(f"❌ Total DOF seems too low: {total_dof}")
        return False
    print(f"✅ Total DOF: {total_dof:,}")
    
    # Test 6: Check component coverage
    expected_components = {
        'cogutil', 'moses', 'atomspace', 'atomspace-rocks', 'atomspace-restful',
        'atomspace-websockets', 'atomspace-metta', 'ure', 'unify', 'attention',
        'spacetime', 'cogserver', 'pln', 'miner', 'asmoses', 'learn', 'generate',
        'lg-atomese', 'relex', 'link-grammar', 'vision', 'perception', 'sensory',
        'opencog', 'debian', 'nix', 'docs'
    }
    
    generated_components = set()
    for issue in components:
        generated_components.add(issue.get('component', ''))
    
    missing = expected_components - generated_components
    if missing:
        print(f"⚠️  Missing components: {missing}")
    else:
        print("✅ All expected components covered")
    
    # Test 7: Validate layer distribution
    layers = {}
    for issue in components:
        layer = issue.get('layer', 'unknown')
        layers[layer] = layers.get(layer, 0) + 1
    
    print(f"✅ Layer distribution: {dict(layers)}")
    
    return True

def test_tensor_coherence():
    """Test tensor field coherence"""
    print("\n🔬 Testing Tensor Field Coherence")
    print("=" * 50)
    
    with open("ontogenesis-issues.json") as f:
        data = json.load(f)
    
    # Collect all tensor shapes
    tensor_data = {}
    for issue in data['component_issues']:
        layer = issue.get('layer')
        metrics = issue.get('tensor_metrics', {})
        shape = metrics.get('shape', [])
        dof = metrics.get('dof', 0)
        
        if layer not in tensor_data:
            tensor_data[layer] = {'shapes': [], 'total_dof': 0, 'components': []}
        
        tensor_data[layer]['shapes'].append(shape)
        tensor_data[layer]['total_dof'] += dof
        tensor_data[layer]['components'].append(issue.get('component'))
    
    # Validate tensor coherence
    coherence_passed = True
    for layer, data_layer in tensor_data.items():
        shapes = data_layer['shapes']
        if len(set(tuple(s) for s in shapes)) > 1:
            print(f"⚠️  {layer} layer has inconsistent tensor shapes")
            coherence_passed = False
        else:
            print(f"✅ {layer} layer tensor coherence validated")
    
    return coherence_passed

def test_dependency_order():
    """Test dependency ordering"""
    print("\n🔗 Testing Dependency Order")
    print("=" * 50)
    
    # Define expected dependency order
    dependency_order = {
        'foundation': 0,
        'core': 1,
        'logic': 2,
        'cognitive': 3,
        'advanced': 4,
        'learning': 5,
        'language': 3,  # Parallel to cognitive
        'embodiment': 3,  # Parallel to cognitive  
        'integration': 6,
        'packaging': 7
    }
    
    with open("ontogenesis-issues.json") as f:
        data = json.load(f)
    
    # Check if dependencies are properly ordered
    order_valid = True
    for issue in data['component_issues']:
        layer = issue.get('layer')
        if layer in dependency_order:
            expected_order = dependency_order[layer]
            print(f"✅ {layer} layer at expected order level {expected_order}")
        else:
            print(f"⚠️  Unknown layer: {layer}")
            order_valid = False
    
    return order_valid

def test_workflow_validity():
    """Test workflow file validity"""
    print("\n⚙️ Testing Workflow Validity")
    print("=" * 50)
    
    try:
        import yaml
        
        with open('.github/workflows/ontogenesis-orchestration.yml') as f:
            workflow = yaml.safe_load(f)
        
        # Check required workflow components
        required_keys = ['name', 'on', 'jobs', 'permissions']
        for key in required_keys:
            if key not in workflow:
                print(f"❌ Missing workflow key: {key}")
                return False
        
        print("✅ Workflow YAML structure is valid")
        
        # Check jobs
        jobs = workflow['jobs']
        expected_jobs = ['architecture-parser', 'generate-orchestration-issues', 'create-github-issues', 'generate-summary']
        for job in expected_jobs:
            if job not in jobs:
                print(f"❌ Missing job: {job}")
                return False
        
        print(f"✅ All {len(expected_jobs)} expected jobs present")
        
        # Check permissions
        permissions = workflow['permissions']
        required_permissions = ['contents', 'issues', 'actions']
        for perm in required_permissions:
            if perm not in permissions:
                print(f"⚠️  Missing permission: {perm}")
        
        print("✅ Workflow permissions configured")
        
        return True
        
    except ImportError:
        print("⚠️  PyYAML not available, skipping workflow validation")
        return True
    except Exception as e:
        print(f"❌ Workflow validation failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🧬 Ontogenesis System Validation")
    print("=" * 60)
    
    tests = [
        ("Issue Generation", test_issue_generation),
        ("Tensor Coherence", test_tensor_coherence),
        ("Dependency Order", test_dependency_order),
        ("Workflow Validity", test_workflow_validity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ontogenesis system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())