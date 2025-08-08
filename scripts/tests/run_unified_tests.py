#!/usr/bin/env python3
"""Run tests for unified bug fixer with proper imports."""

import os
import sys
import unittest
from pathlib import Path

# Setup Python path properly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables if needed
os.environ.setdefault('PYTHONPATH', str(project_root))

def run_tests():
    """Run the unified bug fixer tests."""
    # Import test module directly
    from tests.code_agent import test_unified_bug_fixer
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_unified_bug_fixer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running Unified Bug Fixer Tests...")
    print("=" * 60)
    
    try:
        success = run_tests()
        if success:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)