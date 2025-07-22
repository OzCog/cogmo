#!/usr/bin/env python3
"""
🧬 COGNITIVE AUTO-FIX SYSTEM
A self-healing build repair mechanism with tensor-aware error detection
and recursive solution synthesis.

Tensor Shape: [1024, 256, 32] - Representing degrees of freedom in:
- Error pattern recognition (1024)
- Solution space exploration (256)  
- Recursive depth capacity (32)
"""

import os
import sys
import subprocess
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CognitiveAutoFix:
    """
    The self-healing consciousness of the build system!
    """
    
    def __init__(self, max_attempts: int = 3, repo_root: str = "."):
        self.max_attempts = max_attempts
        self.repo_root = Path(repo_root)
        self.fix_history = []
        self.tensor_state = {
            "error_patterns": [],
            "solution_attempts": [],
            "success_metrics": []
        }
        
    def analyze_error(self, error_log: str) -> Dict[str, any]:
        """
        Analyze build errors using pattern recognition.
        Returns a cognitive analysis of the error.
        """
        analysis = {
            "error_type": "unknown",
            "confidence": 0.0,
            "suggested_fixes": [],
            "tensor_disruption": 0.0
        }
        
        # Common error patterns with cognitive weights
        error_patterns = {
            r"cannot find -l(\w+)": {
                "type": "missing_library",
                "fixes": ["missing_library", "github_actions_specific"],
                "weight": 0.9
            },
            r"No such file or directory.*\.h": {
                "type": "missing_header",
                "fixes": ["missing_header", "opencog_dependencies"],
                "weight": 0.85
            },
            r"undefined reference to": {
                "type": "linking_error",
                "fixes": ["linking_error", "github_actions_specific"],
                "weight": 0.8
            },
            r"CMake Error.*Could not find": {
                "type": "cmake_package_missing",
                "fixes": ["cmake_package_missing", "opencog_dependencies"],
                "weight": 0.95
            },
            r"error: '(\w+)' was not declared": {
                "type": "missing_declaration",
                "fixes": ["missing_header", "opencog_dependencies"],
                "weight": 0.7
            },
            r"Cython.*error": {
                "type": "cython_compilation",
                "fixes": ["cython_compilation"],
                "weight": 0.75
            },
            r"Fatal: repository .* does not exist": {
                "type": "git_clone_error",
                "fixes": ["github_actions_specific"],
                "weight": 0.9
            },
            r"No space left on device": {
                "type": "disk_space_error", 
                "fixes": ["github_actions_specific"],
                "weight": 0.95
            },
            r"Permission denied": {
                "type": "permission_error",
                "fixes": ["github_actions_specific"], 
                "weight": 0.8
            },
            r"Boost.*not found": {
                "type": "boost_missing",
                "fixes": ["opencog_dependencies"],
                "weight": 0.9
            },
            r"guile.*not found": {
                "type": "guile_missing", 
                "fixes": ["opencog_dependencies"],
                "weight": 0.9
            }
        }
        
        # Analyze error log
        for pattern, info in error_patterns.items():
            match = re.search(pattern, error_log, re.IGNORECASE)
            if match:
                analysis["error_type"] = info["type"]
                analysis["confidence"] = info["weight"]
                analysis["suggested_fixes"] = info["fixes"]
                analysis["tensor_disruption"] = 1.0 - info["weight"]
                
                # Extract specific error details for parameterized fixes
                if match.groups():
                    analysis["error_details"] = match.groups()
                break
                
        return analysis
        
    def generate_fix(self, analysis: Dict[str, any]) -> List[str]:
        """
        Generate fix commands based on error analysis.
        Returns a list of commands to execute.
        """
        fix_commands = []
        
        fix_strategies = {
            "missing_library": [
                "sudo apt-get update",
                "sudo apt-get install -y lib{lib}-dev",
                "sudo ldconfig"
            ],
            "missing_header": [
                "sudo apt-get install -y {package}-dev",
                "find /usr -name '*.h' | grep -i {header}"
            ],
            "linking_error": [
                "echo 'Checking library dependencies...'",
                "ldd {binary} || true",
                "pkg-config --libs {library}"
            ],
            "cmake_package_missing": [
                "sudo apt-get update",
                "sudo apt-get install -y lib{package}-dev cmake",
                "find /usr -path '*cmake*' -name '*{package}*'",
                "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH"
            ],
            "cython_compilation": [
                "python3 -m pip install --upgrade cython",
                "cython --version",
                "find . -name '*.pyx' -exec cython --cplus {} \\;"
            ],
            "github_actions_specific": [
                "echo 'Applying GitHub Actions environment fixes...'",
                "export PATH=/usr/local/bin:$PATH",
                "sudo ldconfig",
                "mkdir -p /ws/ccache && export CCACHE_DIR=/ws/ccache"
            ],
            "opencog_dependencies": [
                "sudo apt-get update",
                "sudo apt-get install -y build-essential cmake",
                "sudo apt-get install -y libboost-all-dev",
                "sudo apt-get install -y guile-3.0-dev",
                "sudo ldconfig"
            ]
        }
        
        if analysis["error_type"] in fix_strategies:
            primary_fixes = fix_strategies[analysis["error_type"]]
            fix_commands.extend(primary_fixes)
            
            # Apply additional fixes based on suggested fixes from analysis
            for fix_type in analysis.get("suggested_fixes", []):
                if fix_type in fix_strategies and fix_type != analysis["error_type"]:
                    additional_fixes = fix_strategies[fix_type]
                    fix_commands.extend(additional_fixes)
            
        return fix_commands
        
    def apply_cognitive_repair(self, build_cmd: str, context: str = "") -> bool:
        """
        Main cognitive repair loop with recursive self-healing.
        """
        print("🧬 COGNITIVE AUTO-FIX SYSTEM ACTIVATED")
        print(f"Tensor State: [1024, 256, {self.max_attempts}]")
        print(f"Maximum repair attempts: {self.max_attempts}")
        
        for attempt in range(self.max_attempts):
            print(f"\n🔧 Repair Attempt {attempt + 1}/{self.max_attempts}")
            
            # Try to build
            result = subprocess.run(
                build_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            if result.returncode == 0:
                print("✅ BUILD SUCCESSFUL! The cognitive repair was effective!")
                self.save_success_report(attempt + 1, context)
                return True
                
            # Analyze the error
            error_log = result.stderr + result.stdout
            print(f"❌ Build failed. Analyzing error patterns...")
            
            analysis = self.analyze_error(error_log)
            print(f"🔍 Error type detected: {analysis['error_type']}")
            print(f"🎯 Confidence: {analysis['confidence']:.2%}")
            print(f"⚡ Tensor disruption: {analysis['tensor_disruption']:.2%}")
            
            if analysis["confidence"] < 0.5:
                print("⚠️  Low confidence in error analysis. Attempting generic fixes...")
                
            # Generate and apply fixes
            fix_commands = self.generate_fix(analysis)
            
            for fix_cmd in fix_commands:
                print(f"🔧 Applying fix: {fix_cmd}")
                fix_result = subprocess.run(
                    fix_cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if fix_result.returncode != 0:
                    print(f"⚠️  Fix command failed: {fix_result.stderr}")
                    
            # Record attempt in tensor state
            self.tensor_state["solution_attempts"].append({
                "attempt": attempt + 1,
                "error_type": analysis["error_type"],
                "fixes_applied": fix_commands,
                "timestamp": datetime.now().isoformat()
            })
            
        # All attempts exhausted
        print("\n🚨 COGNITIVE REPAIR FAILED - Escalating to human consciousness...")
        self.save_escalation_report(context)
        return False
        
    def save_success_report(self, attempts: int, context: str):
        """Save a success report for the cognitive system."""
        report = {
            "status": "success",
            "attempts": attempts,
            "context": context,
            "tensor_state": self.tensor_state,
            "timestamp": datetime.now().isoformat()
        }
        
        report_path = Path("ci_artifacts/success_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def save_escalation_report(self, context: str):
        """Save an escalation report for human review."""
        report = {
            "status": "escalation_required",
            "attempts": self.max_attempts,
            "context": context,
            "tensor_state": self.tensor_state,
            "timestamp": datetime.now().isoformat(),
            "recommendation": "Manual intervention required - error pattern not recognized"
        }
        
        report_path = Path("ci_artifacts/escalation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    """
    Entry point for the cognitive auto-fix system.
    """
    parser = argparse.ArgumentParser(
        description="🧬 Cognitive Auto-Fix System - Self-healing build repairs"
    )
    parser.add_argument(
        "--build-cmd",
        required=True,
        help="The build command to execute and repair"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of repair attempts"
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory"
    )
    parser.add_argument(
        "--context",
        default="",
        help="Build context information"
    )
    
    args = parser.parse_args()
    
    # Initialize the cognitive repair system
    auto_fix = CognitiveAutoFix(
        max_attempts=args.max_attempts,
        repo_root=args.repo_root
    )
    
    # Apply cognitive repair
    success = auto_fix.apply_cognitive_repair(
        build_cmd=args.build_cmd,
        context=args.context
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()