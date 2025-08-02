#!/usr/bin/env python3
"""Simple test for enhanced framework components."""

import sys
import json
from pathlib import Path

# Simple test of function block loader
def test_function_block_loader():
    """Test loading function blocks from directory."""
    print("\n=== Testing Function Block Loader ===")
    
    # Add to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from utils.function_block_loader import FunctionBlockLoader
    except ImportError:
        # Try alternative import
        from ragomics_agent_local.utils.function_block_loader import FunctionBlockLoader
    
    loader = FunctionBlockLoader("test_function_blocks")
    
    # List available blocks
    blocks = loader.list_available_blocks()
    print(f"\nFound {len(blocks)} function blocks:")
    for block in blocks:
        print(f"  - {block['name']} ({block['type']}): {block['description']}")
        print(f"    Path: {block['path']}")
        print(f"    Tags: {', '.join(block['tags'])}")
    
    # Load a specific block
    print("\n--- Testing Block Loading ---")
    for block_info in blocks[:2]:  # Test first 2 blocks
        print(f"\nLoading: {block_info['name']}")
        block = loader.load_function_block(block_info['path'])
        if block:
            print(f"  ✓ Successfully loaded")
            print(f"  Type: {block.type}")
            print(f"  Args: {[arg.name for arg in block.static_config.args]}")
            
            # Check for new features
            if block.static_config.input_specification:
                print(f"  Input files: {len(block.static_config.input_specification.required_files)} required")
            if block.static_config.output_specification:
                print(f"  Output files: {len(block.static_config.output_specification.output_files)} expected")
        else:
            print(f"  ✗ Failed to load")
    
    return len(blocks) > 0


def test_execution_context():
    """Test execution context creation."""
    print("\n=== Testing Execution Context ===")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models import ExecutionContext, FileInfo, FileType
    
    # Create sample context
    context = ExecutionContext(
        node_id="test-node-123",
        tree_id="test-tree-456",
        input_files=[
            FileInfo(
                filename="anndata.h5ad",
                filepath="/test/input/anndata.h5ad",
                filetype=FileType.ANNDATA,
                description="Test data"
            ),
            FileInfo(
                filename="metadata.csv",
                filepath="/test/input/metadata.csv",
                filetype=FileType.CSV,
                description="Sample metadata"
            )
        ],
        available_files=[],
        input_dir="/workspace/input",
        output_dir="/workspace/output",
        figures_dir="/workspace/output/figures",
        tree_metadata={"test": True}
    )
    
    print(f"Context created for node: {context.node_id}")
    print(f"Input files: {len(context.input_files)}")
    for file_info in context.input_files:
        print(f"  - {file_info.filename} ({file_info.filetype})")
    
    # Convert to JSON (for passing to function blocks)
    context_json = json.dumps(context.model_dump(), indent=2, default=str)
    print(f"\nContext JSON preview:")
    print(context_json[:300] + "...")
    
    return True


def test_function_block_execution():
    """Test function block code structure."""
    print("\n=== Testing Function Block Code Structure ===")
    
    # Check that function blocks have proper structure
    blocks_dir = Path("test_function_blocks")
    
    required_dirs = ["preprocessing", "velocity_analysis", "trajectory_inference", "quality_control"]
    
    print("Checking directory structure:")
    for dir_name in required_dirs:
        dir_path = blocks_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
            # List subdirectories
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    # Check for required files
                    config = subdir / "config.json"
                    code = subdir / "code.py"
                    reqs = subdir / "requirements.txt"
                    
                    files_status = []
                    if config.exists():
                        files_status.append("config.json")
                    if code.exists():
                        files_status.append("code.py")
                    elif (subdir / "code.R").exists():
                        files_status.append("code.R")
                    if reqs.exists():
                        files_status.append("requirements.txt")
                    
                    print(f"    - {subdir.name}/: {', '.join(files_status)}")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
    
    # Test loading a code file
    print("\n--- Testing Code Loading ---")
    test_code_file = blocks_dir / "quality_control/basic_qc/code.py"
    if test_code_file.exists():
        with open(test_code_file, 'r') as f:
            code = f.read()
        
        # Check for required function signature
        if "def run(context, parameters, input_dir, output_dir):" in code:
            print("  ✓ Code has correct function signature")
        else:
            print("  ✗ Code missing required function signature")
        
        # Check for key features
        features = [
            ("context usage", "context.get(" in code or "context[" in code),
            ("parameters usage", "parameters.get(" in code or "parameters[" in code),
            ("input_dir usage", "input_dir" in code),
            ("output_dir usage", "output_dir" in code),
            ("metadata return", "return" in code and "metadata" in code)
        ]
        
        print("\n  Code features:")
        for feature, present in features:
            status = "✓" if present else "✗"
            print(f"    {status} {feature}")
    
    return True


def main():
    """Run simple tests."""
    print("="*60)
    print("Testing Enhanced Function Block Framework (Simple)")
    print("="*60)
    
    results = []
    
    # Test 1: Function block loader
    try:
        success = test_function_block_loader()
        results.append(("Function Block Loader", success))
    except Exception as e:
        print(f"\nError in function block loader test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Function Block Loader", False))
    
    # Test 2: Execution context
    try:
        success = test_execution_context()
        results.append(("Execution Context", success))
    except Exception as e:
        print(f"\nError in execution context test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Execution Context", False))
    
    # Test 3: Function block structure
    try:
        success = test_function_block_execution()
        results.append(("Function Block Structure", success))
    except Exception as e:
        print(f"\nError in function block structure test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Function Block Structure", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} passed")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)