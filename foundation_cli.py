#!/usr/bin/env python3
"""
Foundation Layer: Cognitive Kernel Genesis CLI Tool

A command-line interface for interacting with the Foundation Layer
cognitive kernel and running various demonstrations and benchmarks.

Usage:
    python3 foundation_cli.py demo          # Run complete demonstration
    python3 foundation_cli.py test          # Run cognitive kernel tests
    python3 foundation_cli.py benchmark     # Run performance benchmarks
    python3 foundation_cli.py status        # Check foundation layer status
    python3 foundation_cli.py interactive   # Interactive cognitive processing
"""

import asyncio
import sys
import json
from pathlib import Path
import subprocess

# Import our cognitive kernel
try:
    from cognitive_kernel_genesis import CognitiveKernel, CognitiveTensor
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure numpy is installed: pip3 install numpy")
    sys.exit(1)


async def run_demo():
    """Run the complete foundation layer demonstration."""
    print("🧬 Running Foundation Layer: Cognitive Kernel Genesis Demo")
    print("=" * 60)
    
    # Run the comprehensive demo script
    try:
        result = subprocess.run(
            ["./foundation_genesis_demo.sh"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
            
        print("✅ Demo completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Demo script not found. Please ensure foundation_genesis_demo.sh exists and is executable.")
        return False


async def run_tests():
    """Run cognitive kernel tests."""
    print("🧠 Running Cognitive Kernel Tests")
    print("=" * 40)
    
    kernel = CognitiveKernel()
    
    print("Initializing cognitive kernel...")
    if not await kernel.initialize():
        print("❌ Kernel initialization failed")
        return False
    
    print("✅ Kernel initialized successfully")
    
    # Run basic tests
    test_results = []
    
    print("\nRunning basic tensor processing tests...")
    for i in range(5):
        test_tensor = kernel._create_test_tensor()
        try:
            result = await kernel.process_cognitive_tensor(test_tensor)
            test_results.append({
                'test': i+1,
                'status': 'PASS',
                'confidence_gain': result.confidence - test_tensor.confidence
            })
            print(f"  Test {i+1}: ✅ PASS (confidence gain: {result.confidence - test_tensor.confidence:.3f})")
        except Exception as e:
            test_results.append({
                'test': i+1,
                'status': 'FAIL',
                'error': str(e)
            })
            print(f"  Test {i+1}: ❌ FAIL ({e})")
    
    # Summary
    passed = len([t for t in test_results if t['status'] == 'PASS'])
    total = len(test_results)
    
    print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


async def run_benchmark():
    """Run performance benchmarks."""
    print("🏃 Running Performance Benchmarks")
    print("=" * 40)
    
    kernel = CognitiveKernel()
    
    print("Initializing cognitive kernel...")
    if not await kernel.initialize():
        print("❌ Kernel initialization failed")
        return False
    
    print("✅ Kernel initialized")
    print("\nRunning comprehensive benchmarks...")
    
    try:
        results = await kernel.run_comprehensive_benchmark()
        
        print("\n📊 Benchmark Results:")
        print("=" * 30)
        
        for benchmark_name, metrics in results['benchmarks'].items():
            print(f"\n{benchmark_name.replace('_', ' ').title()}:")
            
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 'time' in metric_name.lower():
                            print(f"  {metric_name}: {value:.4f}s")
                        elif 'memory' in metric_name.lower():
                            print(f"  {metric_name}: {value:.2f} MB")
                        else:
                            print(f"  {metric_name}: {value}")
                    else:
                        print(f"  {metric_name}: {value}")
            else:
                print(f"  Result: {metrics}")
        
        print(f"\n✅ Benchmarks completed successfully!")
        print(f"📁 Detailed results saved to: cognitive_kernel_reports/")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


async def check_status():
    """Check foundation layer status."""
    print("🔍 Foundation Layer Status Check")
    print("=" * 40)
    
    # Check if cogutil_minimal is built
    cogutil_test = Path("orc-dv/cogutil_minimal/build/test_cogutil_minimal")
    if cogutil_test.exists():
        print("✅ cogutil_minimal: Built and available")
        
        # Test cogutil_minimal
        try:
            result = subprocess.run(
                [str(cogutil_test)],
                cwd="orc-dv/cogutil_minimal/build",
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✅ cogutil_minimal: Tests passing")
            else:
                print("⚠️  cogutil_minimal: Tests failing")
        except Exception as e:
            print(f"⚠️  cogutil_minimal: Test error ({e})")
    else:
        print("❌ cogutil_minimal: Not built")
    
    # Check cognitive kernel
    try:
        kernel = CognitiveKernel()
        if await kernel.initialize():
            print("✅ Cognitive Kernel: Operational")
            
            # Test basic processing
            test_tensor = kernel._create_test_tensor()
            result = await kernel.process_cognitive_tensor(test_tensor)
            print(f"✅ Tensor Processing: Operational (confidence: {result.confidence:.3f})")
            
        else:
            print("❌ Cognitive Kernel: Initialization failed")
            
    except Exception as e:
        print(f"❌ Cognitive Kernel: Error ({e})")
    
    # Check reports
    reports_dir = Path("cognitive_kernel_reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.json"))
        print(f"📁 Reports: {len(report_files)} files available")
    else:
        print("📁 Reports: No reports generated yet")
    
    print("\n🧬 Foundation Layer Status: Ready for cognitive operations")


async def interactive_mode():
    """Interactive cognitive processing mode."""
    print("🧠 Interactive Cognitive Processing Mode")
    print("=" * 45)
    print("Type 'help' for commands, 'quit' to exit")
    
    kernel = CognitiveKernel()
    
    print("\nInitializing cognitive kernel...")
    if not await kernel.initialize():
        print("❌ Kernel initialization failed")
        return
    
    print("✅ Cognitive kernel ready for interactive processing")
    
    while True:
        try:
            command = input("\ncognitive> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive mode")
                break
                
            elif command == 'help':
                print("\nAvailable commands:")
                print("  process   - Process a random cognitive tensor")
                print("  status    - Show kernel status")
                print("  metrics   - Show performance metrics")
                print("  history   - Show processing history")
                print("  benchmark - Run quick benchmark")
                print("  quit      - Exit interactive mode")
                
            elif command == 'process':
                print("Processing cognitive tensor...")
                test_tensor = kernel._create_test_tensor()
                result = await kernel.process_cognitive_tensor(test_tensor)
                print(f"✅ Processed: confidence {test_tensor.confidence:.3f} → {result.confidence:.3f}")
                
            elif command == 'status':
                print(f"Kernel State: {kernel.kernel_state}")
                print(f"Tensors Processed: {len(kernel.processing_history)}")
                if kernel.performance_metrics:
                    print(f"Last Processing Time: {kernel.performance_metrics.get('last_processing_time', 'N/A'):.4f}s")
                
            elif command == 'metrics':
                if kernel.processing_history:
                    times = [h['processing_time'] for h in kernel.processing_history]
                    gains = [h['confidence_gain'] for h in kernel.processing_history]
                    print(f"Average Processing Time: {np.mean(times):.4f}s")
                    print(f"Average Confidence Gain: {np.mean(gains):.3f}")
                    print(f"Total Tensors Processed: {len(kernel.processing_history)}")
                else:
                    print("No processing history available")
                    
            elif command == 'history':
                if kernel.processing_history:
                    print(f"Recent processing history ({len(kernel.processing_history)} entries):")
                    for i, entry in enumerate(kernel.processing_history[-5:], 1):
                        print(f"  {i}: {entry['processing_time']:.4f}s, gain: {entry['confidence_gain']:.3f}")
                else:
                    print("No processing history available")
                    
            elif command == 'benchmark':
                print("Running quick benchmark...")
                times = []
                for _ in range(10):
                    test_tensor = kernel._create_test_tensor()
                    start = asyncio.get_event_loop().time()
                    await kernel.process_cognitive_tensor(test_tensor)
                    times.append(asyncio.get_event_loop().time() - start)
                
                print(f"✅ Quick benchmark: {np.mean(times):.4f}s avg, {np.min(times):.4f}s min, {np.max(times):.4f}s max")
                
            elif command:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_help():
    """Show help information."""
    print("🧬 Foundation Layer: Cognitive Kernel Genesis CLI")
    print("=" * 55)
    print("Usage:")
    print("  python3 foundation_cli.py <command>")
    print("")
    print("Commands:")
    print("  demo         Run complete foundation layer demonstration")
    print("  test         Run cognitive kernel tests")
    print("  benchmark    Run performance benchmarks")
    print("  status       Check foundation layer status")
    print("  interactive  Interactive cognitive processing mode")
    print("  help         Show this help message")
    print("")
    print("Examples:")
    print("  python3 foundation_cli.py demo")
    print("  python3 foundation_cli.py interactive")
    print("")


async def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        show_help()
        return 1
    
    command = sys.argv[1].lower()
    
    if command == 'demo':
        success = await run_demo()
        return 0 if success else 1
        
    elif command == 'test':
        success = await run_tests()
        return 0 if success else 1
        
    elif command == 'benchmark':
        success = await run_benchmark()
        return 0 if success else 1
        
    elif command == 'status':
        await check_status()
        return 0
        
    elif command == 'interactive':
        await interactive_mode()
        return 0
        
    elif command in ['help', '-h', '--help']:
        show_help()
        return 0
        
    else:
        print(f"❌ Unknown command: {command}")
        show_help()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(130)