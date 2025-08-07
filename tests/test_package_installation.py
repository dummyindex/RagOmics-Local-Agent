#!/usr/bin/env python3
"""Test installation of individual pseudotime packages to identify issues."""

import subprocess
import sys
from pathlib import Path

def test_package_installation(package_name):
    """Try to install a package and report the result."""
    print(f"\n{'='*60}")
    print(f"Testing installation of: {package_name}")
    print('='*60)
    
    try:
        # Try to install the package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", package_name],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✓ {package_name} installed successfully (without dependencies)")
            
            # Now try with dependencies
            result2 = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result2.returncode == 0:
                print(f"✓ {package_name} installed successfully WITH dependencies")
            else:
                print(f"✗ {package_name} failed to install dependencies:")
                print(result2.stderr[-1000:])  # Last 1000 chars
        else:
            print(f"✗ {package_name} failed to install:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
    except subprocess.TimeoutExpired:
        print(f"✗ {package_name} installation timed out")
    except Exception as e:
        print(f"✗ Error testing {package_name}: {e}")


def check_package_info():
    """Check package information from PyPI."""
    packages = {
        "scanpy": "Standard single-cell analysis",
        "palantir": "Pseudotime analysis (Setty et al.)",
        "scFates": "Trajectory inference (scFates)",
        "cellrank": "Cell fate mapping (CellRank)",
        "scikit-misc": "Dependency of scFates"
    }
    
    print("\nPackage Information:")
    print("="*80)
    
    for pkg, desc in packages.items():
        print(f"\n{pkg}: {desc}")
        
        # Check if it's available on PyPI
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", pkg],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"  Latest version: {lines[1].split()[0] if len(lines[1].split()) > 0 else 'unknown'}")
        else:
            print(f"  Not found on PyPI")


def analyze_dependencies():
    """Analyze dependencies of each package."""
    print("\n\nDependency Analysis:")
    print("="*80)
    
    packages = ["palantir", "scFates", "cellrank"]
    
    for pkg in packages:
        print(f"\n{pkg} dependencies:")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    deps = line.replace('Requires:', '').strip()
                    if deps:
                        for dep in deps.split(','):
                            print(f"  - {dep.strip()}")
        else:
            # Try to get info without installing
            result2 = subprocess.run(
                ["curl", "-s", f"https://pypi.org/pypi/{pkg}/json"],
                capture_output=True,
                text=True
            )
            
            if result2.returncode == 0:
                import json
                try:
                    data = json.loads(result2.stdout)
                    requires = data.get('info', {}).get('requires_dist', [])
                    if requires:
                        print("  Dependencies from PyPI:")
                        for req in requires[:10]:  # First 10
                            print(f"    - {req}")
                except:
                    pass


def main():
    """Run all tests."""
    print("PSEUDOTIME PACKAGE INSTALLATION ANALYSIS")
    print("="*80)
    
    # Check package info
    check_package_info()
    
    # Test individual installations
    print("\n\nIndividual Package Installation Tests:")
    packages_to_test = ["palantir", "scFates", "cellrank", "scikit-misc"]
    
    for pkg in packages_to_test:
        test_package_installation(pkg)
    
    # Analyze dependencies
    analyze_dependencies()
    
    print("\n\nSUMMARY OF ISSUES:")
    print("="*80)
    print("""
1. scikit-misc (dependency of scFates):
   - Requires 'meson' build system for compilation
   - Not included in standard Docker image
   - Blocks scFates installation

2. Package compatibility:
   - Different packages may require conflicting versions of dependencies
   - Some packages are not regularly maintained

3. Installation complexity:
   - Some packages require compilation of C/C++ extensions
   - Build dependencies not always documented

4. Why it's hard to run in one function block:
   - Installation failures cascade and block execution
   - Missing build tools in container
   - Conflicting dependencies between methods
   - Each method has its own ecosystem of dependencies
""")


if __name__ == "__main__":
    main()