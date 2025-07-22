#!/usr/bin/env python3
"""
Test script for the workflow validation functionality
"""

import unittest
import sys
import os

# Add current directory to path to import validate_workflows
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_workflows import load_circleci_config, extract_workflow_jobs, validate_layer_structure, validate_dependency_cycles


class TestWorkflowOrchestration(unittest.TestCase):
    """Test cases for workflow orchestration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = load_circleci_config()
        self.job_dependencies = extract_workflow_jobs(self.config)
    
    def test_config_loads_successfully(self):
        """Test that the CircleCI config loads without errors"""
        self.assertIsInstance(self.config, dict)
        self.assertIn('workflows', self.config)
        
    def test_job_extraction(self):
        """Test that job dependencies are extracted correctly"""
        self.assertIsInstance(self.job_dependencies, dict)
        self.assertGreater(len(self.job_dependencies), 0)
        
    def test_foundation_layer(self):
        """Test foundation layer components"""
        self.assertIn('cogutil', self.job_dependencies)
        self.assertEqual(self.job_dependencies['cogutil'], [])
        
    def test_core_layer(self):
        """Test core layer components and dependencies"""
        # AtomSpace should depend on cogutil
        self.assertIn('atomspace', self.job_dependencies)
        self.assertEqual(self.job_dependencies['atomspace'], ['cogutil'])
        
        # AtomSpace extensions should depend on atomspace
        self.assertIn('atomspace-rocks', self.job_dependencies)
        self.assertEqual(self.job_dependencies['atomspace-rocks'], ['atomspace'])
        
        self.assertIn('atomspace-restful', self.job_dependencies)
        self.assertEqual(self.job_dependencies['atomspace-restful'], ['atomspace'])
        
    def test_logic_layer(self):
        """Test logic layer components"""
        self.assertIn('unify', self.job_dependencies)
        self.assertEqual(self.job_dependencies['unify'], ['atomspace'])
        
        self.assertIn('ure', self.job_dependencies)
        self.assertEqual(set(self.job_dependencies['ure']), {'atomspace', 'unify'})
        
    def test_integration_layer(self):
        """Test that opencog has all required dependencies"""
        self.assertIn('opencog', self.job_dependencies)
        opencog_deps = set(self.job_dependencies['opencog'])
        required_deps = {'atomspace', 'cogserver', 'attention', 'ure', 'lg-atomese'}
        self.assertEqual(opencog_deps, required_deps)
        
    def test_packaging_layer(self):
        """Test that packaging depends on opencog"""
        self.assertIn('package', self.job_dependencies)
        self.assertEqual(self.job_dependencies['package'], ['opencog'])
        
    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies"""
        self.assertTrue(validate_dependency_cycles(self.job_dependencies))
        
    def test_layer_structure_validation(self):
        """Test that the layer structure validation passes"""
        self.assertTrue(validate_layer_structure(self.job_dependencies))
        
    def test_all_expected_jobs_present(self):
        """Test that all expected jobs are present"""
        expected_jobs = {
            'cogutil', 'atomspace', 'atomspace-rocks', 'atomspace-restful',
            'unify', 'ure', 'cogserver', 'attention', 'spacetime',
            'pln', 'miner', 'moses', 'asmoses', 'lg-atomese',
            'learn', 'language-learning', 'opencog', 'package'
        }
        actual_jobs = set(self.job_dependencies.keys())
        self.assertEqual(actual_jobs, expected_jobs)


def run_tests():
    """Run all test cases"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()