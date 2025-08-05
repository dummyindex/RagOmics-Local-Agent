#!/usr/bin/env python3
"""Test path_dict framework implementation."""

import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.job_executors import PythonExecutor, RExecutor
from ragomics_agent_local.models import NewFunctionBlock, FunctionBlockType, StaticConfig


def test_python_executor_path_dict():
    """Test that Python executor passes path_dict correctly."""
    print("Testing Python executor with path_dict...")
    
    # Create a test function block
    function_block = NewFunctionBlock(
        name="test_path_dict",
        type=FunctionBlockType.PYTHON,
        description="Test path_dict framework",
        static_config=StaticConfig(args=[], description="Test", tag="test"),
        code="""
def run(path_dict):
    '''Test function that uses path_dict.'''
    import json
    import os
    
    # Verify path_dict contains expected keys
    assert "input_dir" in path_dict
    assert "output_dir" in path_dict
    assert "params_file" in path_dict
    assert "input_file" in path_dict
    assert "output_file" in path_dict
    
    # Load parameters
    with open(path_dict["params_file"]) as f:
        params = json.load(f)
    
    # Create a test output
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    
    # Write output using path_dict
    test_data = {"test": "success", "params": params}
    with open(path_dict["output_file"], 'w') as f:
        json.dump(test_data, f)
    
    print(f"Successfully used path_dict to write to {path_dict['output_file']}")
""",
        requirements="",
        parameters={"test_param": "test_value"}
    )
    
    # Create execution directory
    with tempfile.TemporaryDirectory() as temp_dir:
        exec_dir = Path(temp_dir) / "execution"
        exec_dir.mkdir()
        
        # Create mock input
        input_dir = exec_dir / "mock_input"
        input_dir.mkdir()
        
        # Create mock input file
        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        input_file = input_dir / "_node_anndata.h5ad"
        adata.write(input_file)
        
        # Prepare execution
        executor = PythonExecutor()
        executor.prepare_execution_dir(
            execution_dir=exec_dir,
            function_block=function_block,
            input_data_path=input_dir,
            parameters={"test_param": "test_value"}
        )
        
        # Check wrapper code
        wrapper_file = exec_dir / "run.py"
        assert wrapper_file.exists(), "Wrapper not created"
        
        with open(wrapper_file) as f:
            wrapper_code = f.read()
        
        # Verify wrapper creates and passes path_dict
        assert "path_dict = {" in wrapper_code, "path_dict not created in wrapper"
        assert "run(path_dict)" in wrapper_code, "path_dict not passed to run()"
        assert '"input_dir"' in wrapper_code, "input_dir not in path_dict"
        assert '"output_dir"' in wrapper_code, "output_dir not in path_dict"
        
        print("✅ Python executor correctly generates path_dict wrapper")
    
    return True


def test_r_executor_path_dict():
    """Test that R executor passes path_dict correctly."""
    print("Testing R executor with path_dict...")
    
    # Create a test R function block
    function_block = NewFunctionBlock(
        name="test_r_path_dict",
        type=FunctionBlockType.R,
        description="Test R path_dict framework",
        static_config=StaticConfig(args=[], description="Test", tag="test"),
        code="""
run <- function(path_dict) {
    # Test function that uses path_dict
    library(jsonlite)
    
    # Verify path_dict contains expected elements
    stopifnot("input_dir" %in% names(path_dict))
    stopifnot("output_dir" %in% names(path_dict))
    stopifnot("params_file" %in% names(path_dict))
    
    # Load parameters
    params <- fromJSON(path_dict$params_file)
    
    # Create output directory
    dir.create(path_dict$output_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Write test output
    test_data <- list(test = "success", params = params)
    
    output_file <- file.path(path_dict$output_dir, "test_output.json")
    write(toJSON(test_data, auto_unbox = TRUE), output_file)
    
    cat(sprintf("Successfully used path_dict to write to %s\\n", output_file))
}
""",
        requirements="jsonlite",
        parameters={"test_param": "test_value"}
    )
    
    # Create execution directory
    with tempfile.TemporaryDirectory() as temp_dir:
        exec_dir = Path(temp_dir) / "execution"
        exec_dir.mkdir()
        
        # Create mock input
        input_dir = exec_dir / "mock_input"
        input_dir.mkdir()
        
        # Create mock input file
        with open(input_dir / "_node_anndata.h5ad", 'w') as f:
            json.dump({"mock": "data"}, f)
        
        # Prepare execution
        executor = RExecutor()
        executor.prepare_execution_dir(
            execution_dir=exec_dir,
            function_block=function_block,
            input_data_path=input_dir,
            parameters={"test_param": "test_value"}
        )
        
        # Check wrapper code
        wrapper_file = exec_dir / "run.R"
        assert wrapper_file.exists(), "R wrapper not created"
        
        with open(wrapper_file) as f:
            wrapper_code = f.read()
        
        # Verify wrapper creates and passes path_dict
        assert "path_dict <- list(" in wrapper_code, "path_dict not created in R wrapper"
        assert "run(path_dict)" in wrapper_code, "path_dict not passed to run() in R"
        assert "input_dir =" in wrapper_code, "input_dir not in R path_dict"
        assert "output_dir =" in wrapper_code, "output_dir not in R path_dict"
        
        print("✅ R executor correctly generates path_dict wrapper")
    
    return True


def test_path_dict_contents():
    """Test that path_dict contains correct paths."""
    print("Testing path_dict contents...")
    
    function_block = NewFunctionBlock(
        name="test_paths",
        type=FunctionBlockType.PYTHON,
        description="Test path contents",
        static_config=StaticConfig(args=[], description="Test", tag="test"),
        code="""
def run(path_dict):
    import json
    # Save path_dict for inspection
    with open('/tmp/path_dict_test.json', 'w') as f:
        json.dump(path_dict, f)
""",
        requirements="",
        parameters={}
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        exec_dir = Path(temp_dir) / "execution"
        exec_dir.mkdir()
        
        input_dir = exec_dir / "inputs"
        input_dir.mkdir()
        
        # Create input file
        input_file = input_dir / "_node_anndata.h5ad"
        with open(input_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        executor = PythonExecutor()
        executor.prepare_execution_dir(
            execution_dir=exec_dir,
            function_block=function_block,
            input_data_path=input_dir,
            parameters={}
        )
        
        # Read wrapper to check path_dict values
        wrapper_file = exec_dir / "run.py"
        with open(wrapper_file) as f:
            wrapper = f.read()
        
        # Check standard paths
        assert '"/workspace/input"' in wrapper
        assert '"/workspace/output"' in wrapper
        assert '"/workspace/parameters.json"' in wrapper
        assert '"/workspace/input/_node_anndata.h5ad"' in wrapper
        assert '"/workspace/output/_node_anndata.h5ad"' in wrapper
        
        print("✅ path_dict contains correct standard paths")
    
    return True


def main():
    """Run all path_dict tests."""
    print("\n" + "=" * 60)
    print("PATH_DICT FRAMEWORK TESTS")
    print("=" * 60)
    
    tests = [
        test_python_executor_path_dict,
        test_r_executor_path_dict,
        test_path_dict_contents
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All path_dict framework tests passed!")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())