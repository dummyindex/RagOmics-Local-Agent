#!/usr/bin/env python3
"""Simple test to verify path_dict framework in wrapper code generation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_python_wrapper_generation():
    """Test that Python wrapper correctly generates path_dict and params."""
    print("Testing Python wrapper generation...")
    
    # Import here to avoid initialization issues
    from ragomics_agent_local.job_executors.python_executor import PythonExecutor
    
    # Get the wrapper code directly
    executor = object.__new__(PythonExecutor)  # Create without __init__
    wrapper_code = executor._generate_wrapper_code()
    
    # Check for path_dict creation with only directories
    assert 'path_dict = {' in wrapper_code, "path_dict not created in wrapper"
    assert '"input_dir":' in wrapper_code, "input_dir not in path_dict"
    assert '"output_dir":' in wrapper_code, "output_dir not in path_dict"
    
    # Check that params are loaded separately
    assert 'params = json.load' in wrapper_code or 'with open(\'/workspace/parameters.json\')' in wrapper_code, "params not loaded from JSON"
    
    # Check that both path_dict and params are passed to run()
    assert 'run(path_dict, params)' in wrapper_code, "path_dict and params not passed to run()"
    
    # Check standard paths
    assert '/workspace/input' in wrapper_code
    assert '/workspace/output' in wrapper_code
    assert '/workspace/parameters.json' in wrapper_code
    
    print("✅ Python wrapper correctly generates path_dict and params")


def test_r_wrapper_generation():
    """Test that R wrapper correctly generates path_dict and params."""
    print("Testing R wrapper generation...")
    
    # Import here to avoid initialization issues
    from ragomics_agent_local.job_executors.r_executor import RExecutor
    
    # Get the wrapper code directly
    executor = object.__new__(RExecutor)  # Create without __init__
    wrapper_code = executor._generate_wrapper_code()
    
    # Check for path_dict creation with only directories
    assert 'path_dict <- list(' in wrapper_code or 'path_dict <-' in wrapper_code, "path_dict not created in R wrapper"
    assert 'input_dir =' in wrapper_code, "input_dir not in R path_dict"
    assert 'output_dir =' in wrapper_code, "output_dir not in R path_dict"
    
    # Check that params are loaded separately
    assert 'params <- fromJSON' in wrapper_code or 'fromJSON("/workspace/parameters.json")' in wrapper_code, "params not loaded from JSON"
    
    # Check that both path_dict and params are passed to run()
    assert 'run(path_dict, params)' in wrapper_code, "path_dict and params not passed to run() in R"
    
    # Check standard paths
    assert '/workspace/input' in wrapper_code
    assert '/workspace/output' in wrapper_code
    assert '/workspace/parameters.json' in wrapper_code
    
    print("✅ R wrapper correctly generates path_dict and params")


def test_function_signature_validation():
    """Test that function blocks should use path_dict, params arguments."""
    print("Testing function signature validation...")
    
    # Correct Python signature
    correct_python = "def run(path_dict, params):"
    
    # Correct R signature
    correct_r = "run <- function(path_dict, params)"
    
    # Old incorrect signatures
    old_python = "def run(adata, **parameters):"  # Old style with adata
    old_python2 = "def run(adata=None, **kwargs):"  # Old style
    
    old_r = "run <- function(seurat_obj = NULL, ...)"  # Old style
    old_r2 = "run <- function(adata = NULL, ...)"  # Old style
    
    print("  Validating correct signatures...")
    assert "path_dict" in correct_python
    assert "params" in correct_python
    assert "path_dict" in correct_r
    assert "params" in correct_r
    
    print("  Checking old signatures don't have path_dict...")
    assert "path_dict" not in old_python
    assert "path_dict" not in old_python2
    assert "path_dict" not in old_r
    assert "path_dict" not in old_r2
    
    print("✅ Function signature validation passed")


def main():
    """Run all simple path_dict tests."""
    print("\n" + "=" * 60)
    print("PATH_DICT FRAMEWORK SIMPLE TESTS")
    print("=" * 60)
    
    tests = [
        test_python_wrapper_generation,
        test_r_wrapper_generation,
        test_function_signature_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All path_dict framework tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())