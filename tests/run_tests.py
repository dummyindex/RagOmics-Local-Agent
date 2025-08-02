#!/usr/bin/env python3
"""
Test runner script for ragomics_agent_local.
Cleans up test outputs and runs all tests systematically.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import json

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print("="*60)

def print_success(text):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def clean_test_outputs():
    """Clean up all test output directories."""
    print_header("Cleaning Test Outputs")
    
    # Main test output directory
    main_test_dir = "test_outputs"
    
    # Legacy test directories to clean (for backwards compatibility)
    legacy_dirs = [
        "test_CLI_main_agent", 
        "test_tree_structure_output",
        "test_agent_mock_output",
        "outputs",
        "test_enhanced_outputs",
        "test_scvelo_outputs",
        "test_scfates_outputs"
    ]
    
    # Also look for any directory matching test patterns
    patterns = [
        "test_*_output*",
        "*_test_output*",
        "temp_test_*"
    ]
    
    project_root = Path(__file__).parent.parent
    cleaned_count = 0
    
    # Clean main test output directory
    if (project_root / main_test_dir).exists():
        try:
            shutil.rmtree(project_root / main_test_dir)
            print_success(f"Removed {main_test_dir}/")
            cleaned_count += 1
        except Exception as e:
            print_error(f"Failed to remove {main_test_dir}/: {e}")
    
    # Clean legacy directories
    for dir_name in legacy_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print_success(f"Removed {dir_name}/")
                cleaned_count += 1
            except Exception as e:
                print_error(f"Failed to remove {dir_name}/: {e}")
    
    # Clean directories matching patterns
    for pattern in patterns:
        for dir_path in project_root.glob(pattern):
            if dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    print_success(f"Removed {dir_path.name}/")
                    cleaned_count += 1
                except Exception as e:
                    print_error(f"Failed to remove {dir_path.name}/: {e}")
    
    # Clean __pycache__ directories
    pycache_count = 0
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            pycache_count += 1
        except:
            pass
    
    if pycache_count > 0:
        print_success(f"Removed {pycache_count} __pycache__ directories")
    
    if cleaned_count == 0:
        print_info("No test output directories found to clean")
    else:
        print_success(f"Cleaned {cleaned_count} test output directories")
    
    return cleaned_count

def check_environment():
    """Check if the test environment is properly set up."""
    print_header("Checking Environment")
    
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_warning(f"Python {python_version.major}.{python_version.minor} (recommended: 3.8+)")
    
    # Check for required modules
    required_modules = [
        "scanpy",
        "pandas", 
        "numpy",
        "anndata",
        "docker"
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print_success(f"{module} installed")
        except ImportError:
            print_error(f"{module} not installed")
            issues.append(f"Missing module: {module}")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print_success("OPENAI_API_KEY set")
    else:
        print_warning("OPENAI_API_KEY not set (some tests will be skipped)")
    
    # Check for test data
    test_data_path = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if test_data_path.exists():
        print_success(f"Test data found: {test_data_path.name}")
    else:
        print_error("Test data not found")
        issues.append("Missing test data file")
    
    # Check Docker
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("Docker is running")
        else:
            print_warning("Docker not running (some tests may fail)")
    except FileNotFoundError:
        print_warning("Docker not installed (some tests may fail)")
    
    if issues:
        print_warning(f"\nFound {len(issues)} environment issues")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def get_test_files(test_dir, category=None):
    """Get list of test files to run."""
    test_files = []
    
    if category:
        # Run tests in specific category
        category_dirs = {
            "unit": ["test_*.py"],
            "agents": ["agents/test_*.py"],
            "clustering": ["clustering/test_*.py"],
            "cli": ["cli/test_*.py"],
            "integration": ["test_enhanced_*.py", "test_agent_*.py"],
            "manual": ["*manual*.py", "test_scfates_*.py", "test_scvelo_*.py"]
        }
        
        patterns = category_dirs.get(category, [f"*{category}*.py"])
        for pattern in patterns:
            test_files.extend(test_dir.glob(pattern))
    else:
        # Get all test files
        test_files = list(test_dir.glob("test_*.py"))
        test_files.extend(test_dir.glob("*/test_*.py"))
    
    # Filter out this script and __pycache__
    test_files = [
        f for f in test_files 
        if f.name != "run_tests.py" 
        and "__pycache__" not in str(f)
        and f.is_file()
    ]
    
    return sorted(test_files)

def run_test(test_file, verbose=False):
    """Run a single test file."""
    test_name = test_file.name
    
    # Skip certain tests that require special setup
    skip_patterns = [
        "test_openai",  # Requires API key
        "test_agent_system",  # Long running
        "test_enhanced_framework",  # Long running
        "manual_test",  # Manual tests
    ]
    
    if not os.getenv("OPENAI_API_KEY"):
        skip_patterns.extend([
            "test_clustering_agent",
            "test_main_agent",
            "test_llm_schema"
        ])
    
    for pattern in skip_patterns:
        if pattern in test_name:
            print_warning(f"Skipping {test_name} (requires special setup)")
            return "skipped"
    
    try:
        # Run the test
        cmd = [sys.executable, str(test_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print_success(f"PASSED: {test_name}")
            if verbose and result.stdout:
                print(f"  Output: {result.stdout[:200]}...")
            return "passed"
        else:
            print_error(f"FAILED: {test_name}")
            if verbose:
                print(f"  Error: {result.stderr[:500]}")
            return "failed"
            
    except subprocess.TimeoutExpired:
        print_error(f"TIMEOUT: {test_name} (exceeded 60s)")
        return "timeout"
    except Exception as e:
        print_error(f"ERROR: {test_name} - {e}")
        return "error"

def run_test_suite(category=None, verbose=False, clean=True):
    """Run the complete test suite."""
    start_time = datetime.now()
    
    # Clean if requested
    if clean:
        clean_test_outputs()
    
    # Ensure test_outputs directory exists
    test_outputs_dir = Path(__file__).parent.parent / "test_outputs"
    test_outputs_dir.mkdir(exist_ok=True)
    print_info(f"Test outputs directory: {test_outputs_dir}")
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print_warning("\nEnvironment check found issues, some tests may fail")
    
    # Get test files
    test_dir = Path(__file__).parent
    test_files = get_test_files(test_dir, category)
    
    if not test_files:
        print_error(f"No test files found for category: {category}")
        return 1
    
    # Run tests
    print_header(f"Running {len(test_files)} Tests")
    if category:
        print_info(f"Category: {category}")
    
    results = {
        "passed": [],
        "failed": [],
        "skipped": [],
        "timeout": [],
        "error": []
    }
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] Running {test_file.relative_to(test_dir)}...")
        result = run_test(test_file, verbose)
        results[result].append(test_file.name)
    
    # Print summary
    duration = (datetime.now() - start_time).total_seconds()
    print_header("Test Summary")
    
    total = len(test_files)
    passed = len(results["passed"])
    failed = len(results["failed"])
    skipped = len(results["skipped"])
    timeout = len(results["timeout"])
    error = len(results["error"])
    
    print(f"Total tests: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    print(f"  {Colors.YELLOW}Skipped: {skipped}{Colors.END}")
    print(f"  {Colors.RED}Timeout: {timeout}{Colors.END}")
    print(f"  {Colors.RED}Error: {error}{Colors.END}")
    print(f"\nDuration: {duration:.2f} seconds")
    
    # Show failed tests
    if results["failed"]:
        print(f"\n{Colors.RED}Failed tests:{Colors.END}")
        for test in results["failed"]:
            print(f"  - {test}")
    
    # Save results to file
    results_file = test_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "total": total,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Return exit code
    if failed or timeout or error:
        return 1
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for ragomics_agent_local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  unit         - Basic unit tests
  agents       - Agent-specific tests
  clustering   - Clustering workflow tests
  cli          - CLI interface tests
  integration  - Integration tests
  manual       - Manual/interactive tests

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --category agents  # Run agent tests only
  python run_tests.py --no-clean        # Don't clean outputs
  python run_tests.py --verbose         # Show detailed output
  python run_tests.py --clean-only      # Only clean outputs
        """
    )
    
    parser.add_argument(
        "--category", "-c",
        help="Test category to run",
        choices=["unit", "agents", "clustering", "cli", "integration", "manual"]
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed test output"
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean test outputs before running"
    )
    
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean test outputs, don't run tests"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available tests without running"
    )
    
    args = parser.parse_args()
    
    # Clean only mode
    if args.clean_only:
        clean_test_outputs()
        return 0
    
    # List mode
    if args.list:
        test_dir = Path(__file__).parent
        test_files = get_test_files(test_dir, args.category)
        
        print_header(f"Available Tests ({len(test_files)})")
        if args.category:
            print_info(f"Category: {args.category}")
        
        for test_file in test_files:
            rel_path = test_file.relative_to(test_dir)
            print(f"  - {rel_path}")
        return 0
    
    # Run tests
    clean = not args.no_clean
    return run_test_suite(
        category=args.category,
        verbose=args.verbose,
        clean=clean
    )

if __name__ == "__main__":
    sys.exit(main())