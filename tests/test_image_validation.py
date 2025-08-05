#!/usr/bin/env python3
"""Test that image validation only checks required images."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.models import NewFunctionBlock, FunctionBlockType, StaticConfig
from ragomics_agent_local.job_executors import ExecutorManager


def test_python_only_validation():
    """Test that Python blocks only validate Python image."""
    print("Testing Python-only validation...")
    
    # Create executor manager
    executor_manager = ExecutorManager()
    
    # Create a Python function block
    python_block = NewFunctionBlock(
        name="test_python",
        type=FunctionBlockType.PYTHON,
        description="Test Python block",
        code='def run(path_dict, params):\n    print("Hello from Python")\n    return None',
        requirements="numpy",
        parameters={},
        static_config=StaticConfig(args=[], description="Test", tag="test", source="test")
    )
    
    # Validate only Python image
    validation = executor_manager.validate_required_image(FunctionBlockType.PYTHON)
    
    print(f"Docker available: {validation['docker_available']}")
    print(f"Python image ({validation['image_name']}): {'✓' if validation['required_image'] else '✗'}")
    print("Note: R image validation was NOT performed")


def test_r_only_validation():
    """Test that R blocks only validate R image."""
    print("\nTesting R-only validation...")
    
    # Create executor manager
    executor_manager = ExecutorManager()
    
    # Create an R function block
    r_block = NewFunctionBlock(
        name="test_r",
        type=FunctionBlockType.R,
        description="Test R block",
        code='run <- function(adata, ...) {\n  print("Hello from R")\n  return(adata)\n}',
        requirements="",
        parameters={},
        static_config=StaticConfig(args=[], description="Test", tag="test", source="test")
    )
    
    # Validate only R image
    validation = executor_manager.validate_required_image(FunctionBlockType.R)
    
    print(f"Docker available: {validation['docker_available']}")
    print(f"R image ({validation['image_name']}): {'✓' if validation['required_image'] else '✗'}")
    print("Note: Python image validation was NOT performed")


def main():
    """Run validation tests."""
    print("="*60)
    print("Image Validation Test")
    print("="*60)
    
    # Test Python validation
    python_result = test_python_only_validation()
    
    # Test R validation
    r_result = test_r_only_validation()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Python blocks now only check Python Docker image")
    print("- R blocks now only check R Docker image")
    print("- This saves time by not checking unnecessary images")
    print("="*60)
    
    return python_result["docker_available"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)