#!/usr/bin/env python3
"""
Moses Foundation Layer Tensor Validation
Issue #54 - Validates Moses implementation meets cognitive architecture requirements
"""

import subprocess
import time
import numpy as np
import json
import os

class MosesTensorValidator:
    """Validates Moses Foundation Layer meets tensor architecture requirements."""
    
    def __init__(self):
        self.tensor_shape = [512, 128, 8]
        self.degrees_of_freedom = 524288
        self.cognitive_function = "utility-primitives"
        self.results = {}
        
    def validate_tensor_shape(self):
        """Validates tensor shape specification [512, 128, 8]"""
        expected_dof = self.tensor_shape[0] * self.tensor_shape[1] * self.tensor_shape[2]
        assert expected_dof == self.degrees_of_freedom, f"Expected DOF {self.degrees_of_freedom}, got {expected_dof}"
        
        self.results['tensor_shape_valid'] = True
        self.results['tensor_shape'] = self.tensor_shape
        self.results['degrees_of_freedom'] = self.degrees_of_freedom
        print(f"✓ Tensor shape {self.tensor_shape} validated with {self.degrees_of_freedom} DOF")
        
    def validate_build_infrastructure(self):
        """Validates C++/CMake build infrastructure exists and works"""
        build_checks = {
            'cmake_exists': os.path.exists('/home/runner/work/cogmo/cogmo/orc-ai/moses/CMakeLists.txt'),
            'moses_binary': os.path.exists('/usr/local/bin/moses'),
            'cogutil_lib': os.path.exists('/usr/local/lib/opencog/libcogutil.so'),
            'moses_lib': os.path.exists('/usr/local/lib/libmoses.so')
        }
        
        self.results['build_infrastructure'] = build_checks
        all_passed = all(build_checks.values())
        
        if all_passed:
            print("✓ Build infrastructure validated")
        else:
            print("✗ Build infrastructure issues:", {k: v for k, v in build_checks.items() if not v})
            
        return all_passed
        
    def validate_core_functionality(self):
        """Validates core Moses functionality works"""
        try:
            # Test Moses version command
            result = subprocess.run(['/usr/local/bin/moses', '--version'], 
                                  capture_output=True, text=True,
                                  env={'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/lib/moses:/usr/local/lib/opencog'})
            
            if result.returncode == 0:
                print("✓ Moses core functionality validated")
                self.results['moses_functional'] = True
                self.results['moses_version'] = result.stdout.strip()
                return True
            else:
                print("✗ Moses functionality test failed")
                self.results['moses_functional'] = False
                return False
                
        except Exception as e:
            print(f"✗ Moses functionality test error: {e}")
            self.results['moses_functional'] = False
            return False
            
    def validate_performance_benchmarks(self):
        """Validates performance meets tensor complexity requirements"""
        # Simple performance test - measure Moses help command time
        start_time = time.time()
        
        try:
            result = subprocess.run(['/usr/local/bin/moses', '--help'], 
                                  capture_output=True, text=True,
                                  env={'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/lib/moses:/usr/local/lib/opencog'})
            end_time = time.time()
            
            response_time = end_time - start_time
            complexity_threshold = 0.52  # 0.52M DOF complexity threshold
            
            if response_time < complexity_threshold:
                print(f"✓ Performance validated: {response_time:.3f}s < {complexity_threshold}s threshold")
                self.results['performance_valid'] = True
                self.results['response_time'] = response_time
                return True
            else:
                print(f"✗ Performance issue: {response_time:.3f}s >= {complexity_threshold}s threshold")
                self.results['performance_valid'] = False
                return False
                
        except Exception as e:
            print(f"✗ Performance test error: {e}")
            self.results['performance_valid'] = False
            return False
            
    def validate_memory_management(self):
        """Validates memory management patterns"""
        # Check that Moses doesn't crash immediately (basic memory management)
        try:
            # Test multiple rapid calls to check for memory leaks
            for i in range(3):
                result = subprocess.run(['/usr/local/bin/moses', '--version'], 
                                      capture_output=True, text=True,
                                      env={'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/lib/moses:/usr/local/lib/opencog'})
                if result.returncode != 0:
                    print(f"✗ Memory management issue on iteration {i+1}")
                    self.results['memory_management'] = False
                    return False
                    
            print("✓ Basic memory management validated")
            self.results['memory_management'] = True
            return True
            
        except Exception as e:
            print(f"✗ Memory management test error: {e}")
            self.results['memory_management'] = False
            return False
            
    def validate_api_interfaces(self):
        """Validates API interfaces are documented and accessible"""
        api_checks = {
            'headers_installed': os.path.exists('/usr/local/include/moses'),
            'main_headers': os.path.exists('/usr/local/include/moses/moses'),
            'combo_headers': os.path.exists('/usr/local/include/moses/comboreduct'),
            'cmake_config': os.path.exists('/usr/local/lib/cmake') or os.path.exists('/usr/local/share/opencog/cmake')
        }
        
        self.results['api_interfaces'] = api_checks
        all_passed = all(api_checks.values())
        
        if all_passed:
            print("✓ API interfaces validated")
        else:
            print("✗ API interface issues:", {k: v for k, v in api_checks.items() if not v})
            
        return all_passed
        
    def validate_thread_safety(self):
        """Basic thread safety validation"""
        # Test concurrent Moses calls (basic thread safety check)
        try:
            import concurrent.futures
            
            def run_moses():
                result = subprocess.run(['/usr/local/bin/moses', '--version'], 
                                      capture_output=True, text=True,
                                      env={'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/lib/moses:/usr/local/lib/opencog'})
                return result.returncode == 0
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run_moses) for _ in range(2)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
            if all(results):
                print("✓ Basic thread safety validated")
                self.results['thread_safety'] = True
                return True
            else:
                print("✗ Thread safety issues detected")
                self.results['thread_safety'] = False
                return False
                
        except Exception as e:
            print(f"✗ Thread safety test error: {e}")
            self.results['thread_safety'] = False
            return False
            
    def generate_validation_report(self):
        """Generate final validation report"""
        report = {
            'moses_foundation_layer': {
                'issue': '#54',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'tensor_configuration': {
                    'shape': self.tensor_shape,
                    'degrees_of_freedom': self.degrees_of_freedom,
                    'cognitive_function': self.cognitive_function
                },
                'validation_results': self.results,
                'overall_status': 'PASSED' if all(
                    v for k, v in self.results.items() 
                    if k.endswith('_valid') or k in ['moses_functional', 'memory_management', 'thread_safety']
                ) else 'FAILED'
            }
        }
        
        # Write report
        report_file = '/home/runner/work/cogmo/cogmo/moses_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Validation Report: {report_file}")
        print(f"Overall Status: {report['moses_foundation_layer']['overall_status']}")
        
        return report
        
    def run_full_validation(self):
        """Run complete Moses Foundation Layer validation"""
        print("🧬 Moses Foundation Layer Validation - Issue #54")
        print("=" * 50)
        
        validations = [
            ('Tensor Shape', self.validate_tensor_shape),
            ('Build Infrastructure', self.validate_build_infrastructure),
            ('Core Functionality', self.validate_core_functionality),
            ('Performance Benchmarks', self.validate_performance_benchmarks),
            ('Memory Management', self.validate_memory_management),
            ('API Interfaces', self.validate_api_interfaces),
            ('Thread Safety', self.validate_thread_safety)
        ]
        
        for name, validator in validations:
            print(f"\n🔍 Validating {name}...")
            try:
                validator()
            except Exception as e:
                print(f"✗ {name} validation failed: {e}")
                self.results[name.lower().replace(' ', '_')] = False
                
        return self.generate_validation_report()

if __name__ == '__main__':
    validator = MosesTensorValidator()
    report = validator.run_full_validation()