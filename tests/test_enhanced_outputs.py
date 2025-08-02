#!/usr/bin/env python3
"""Test enhanced framework and generate actual outputs."""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

# Import the standalone loader directly
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from function_block_loader_standalone import FunctionBlockLoaderStandalone


def run_function_block_standalone(block_path, input_data_path, output_dir):
    """Run a function block in standalone mode (without Docker)."""
    # Load the block
    loader = FunctionBlockLoaderStandalone("test_function_blocks")
    block = loader.load_function_block(block_path)
    
    if not block:
        print(f"Failed to load block: {block_path}")
        return False
    
    print(f"\nRunning {block.name}...")
    
    # Create working directory
    work_dir = output_dir / block.name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create input/output directories
    input_dir = work_dir / "input"
    output_dir_block = work_dir / "output"
    figures_dir = output_dir_block / "figures"
    
    input_dir.mkdir(exist_ok=True)
    output_dir_block.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Copy input data
    if input_data_path.is_file():
        shutil.copy2(input_data_path, input_dir / "anndata.h5ad")
    elif input_data_path.is_dir():
        for file in input_data_path.glob("*.h5ad"):
            shutil.copy2(file, input_dir / file.name)
            break  # Just copy first h5ad file
    
    # Create execution context
    context = {
        "node_id": f"test-{block.name}",
        "tree_id": "test-tree",
        "input_files": [
            {
                "filename": "anndata.h5ad",
                "filepath": str(input_dir / "anndata.h5ad"),
                "filetype": "anndata"
            }
        ],
        "available_files": [],
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir_block),
            "figures_dir": str(figures_dir)
        }
    }
    
    # Save context
    context_file = work_dir / "context.json"
    with open(context_file, 'w') as f:
        json.dump(context, f, indent=2)
    
    # Create runner script
    runner_script = work_dir / "run_block.py"
    runner_content = f'''
import sys
import json
from pathlib import Path

# Load context
with open('context.json', 'r') as f:
    context = json.load(f)

# Load parameters
parameters = {json.dumps(block.parameters)}

# Import and run the block
import importlib.util
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
block_path = "{block_path}"
code_path = os.path.join(base_dir, "test_function_blocks", block_path, "code.py")
spec = importlib.util.spec_from_file_location("block_code", code_path)
block_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_module)
run = block_module.run

result = run(
    context=context,
    parameters=parameters,
    input_dir='{input_dir}',
    output_dir='{output_dir_block}'
)

print("Result:", result)
'''
    
    with open(runner_script, 'w') as f:
        f.write(runner_content)
    
    # Run the block
    try:
        result = subprocess.run(
            [sys.executable, "run_block.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"  ✓ {block.name} completed successfully")
            
            # Check outputs
            output_files = list(output_dir_block.glob("*"))
            print(f"  Output files: {len(output_files)}")
            for file in output_files[:5]:
                print(f"    - {file.name}")
            
            # Check figures
            figures = list(figures_dir.glob("*.png"))
            if figures:
                print(f"  Figures: {len(figures)}")
                for fig in figures:
                    print(f"    - {fig.name}")
            
            return True
        else:
            print(f"  ✗ {block.name} failed")
            print(f"  Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error running {block.name}: {e}")
        return False


def test_pipeline():
    """Test a complete pipeline with enhanced framework."""
    print("=== Testing Enhanced Framework Pipeline ===")
    
    # Create test data
    print("\n1. Creating test data...")
    test_data_dir = Path("test_outputs/enhanced_framework/test_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    import scanpy as sc
    import numpy as np
    
    n_cells = 500
    n_genes = 2000
    
    # Create expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    adata = sc.AnnData(X)
    
    # Add gene names
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    # Add some mitochondrial genes
    var_names = adata.var_names.tolist()
    mt_genes = np.random.choice(n_genes, size=50, replace=False)
    for idx in mt_genes:
        var_names[idx] = f"MT-{var_names[idx]}"
    adata.var_names = var_names
    
    # Add spliced/unspliced for velocity
    adata.layers['spliced'] = X * 0.7
    adata.layers['unspliced'] = X * 0.3
    
    # Save test data
    test_file = test_data_dir / "test_data.h5ad"
    adata.write(test_file)
    print(f"  Created test data: {test_file}")
    print(f"  Shape: {adata.shape}")
    
    # Define pipeline
    pipeline = [
        ("quality_control/basic_qc", {"min_genes": 100, "max_mt_percent": 25}),
        ("preprocessing/scvelo_preprocessing", {"n_top_genes": 1000}),
        ("velocity_analysis/velocity_steady_state", {"mode": "stochastic"}),
        ("trajectory_inference/elpigraph_trajectory", {"n_nodes": 30})
    ]
    
    # Run pipeline
    output_base = Path("test_outputs/enhanced_framework/pipeline_output")
    output_base.mkdir(parents=True, exist_ok=True)
    
    current_input = test_file
    results = []
    
    for block_path, params in pipeline:
        # Update parameters in block
        loader = FunctionBlockLoaderStandalone("test_function_blocks")
        block = loader.load_function_block(block_path)
        if block:
            block.parameters = params
        
        # Run block
        success = run_function_block_standalone(block_path, current_input, output_base)
        results.append((block_path, success))
        
        if success:
            # Update input for next step
            block_output_dir = output_base / block.name / "output"
            output_file = block_output_dir / "anndata.h5ad"
            if output_file.exists():
                current_input = output_file
            else:
                print(f"  Warning: No output anndata.h5ad found for {block.name}")
                break
    
    # Summary
    print("\n=== Pipeline Summary ===")
    for block_path, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {block_path}")
    
    return all(success for _, success in results)


def test_trajectory_bug_fixer():
    """Test the trajectory bug fixer workflow."""
    print("\n=== Testing Trajectory Bug Fixer Workflow ===")
    
    # Create test data
    test_data_dir = Path("test_outputs/enhanced_framework/bug_fixer_test")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simple test data
    import scanpy as sc
    import numpy as np
    
    X = np.random.randn(100, 50)
    adata = sc.AnnData(X)
    test_file = test_data_dir / "test_data.h5ad"
    adata.write(test_file)
    
    output_dir = Path("test_outputs/enhanced_framework/bug_fixer_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test failing block
    print("\n1. Testing failing trajectory block...")
    success_fail = run_function_block_standalone(
        "trajectory_inference/trajectory_failing", 
        test_file, 
        output_dir
    )
    print(f"  Expected to fail: {'✓' if not success_fail else '✗'}")
    
    # Test fixed block
    print("\n2. Testing fixed trajectory block...")
    success_fixed = run_function_block_standalone(
        "trajectory_inference/trajectory_fixed", 
        test_file, 
        output_dir
    )
    print(f"  Expected to succeed: {'✓' if success_fixed else '✗'}")
    
    return not success_fail and success_fixed


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Enhanced Framework with Output Generation")
    print("="*60)
    
    # Clean up old outputs
    output_dir = Path("test_outputs/enhanced_framework")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    results = []
    
    # Test 1: Complete pipeline
    try:
        success = test_pipeline()
        results.append(("Enhanced Pipeline", success))
    except Exception as e:
        print(f"\nError in pipeline test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Enhanced Pipeline", False))
    
    # Test 2: Bug fixer workflow
    try:
        success = test_trajectory_bug_fixer()
        results.append(("Bug Fixer Workflow", success))
    except Exception as e:
        print(f"\nError in bug fixer test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Bug Fixer Workflow", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} passed")
    
    # List generated outputs
    print("\n=== Generated Outputs ===")
    output_base = Path("test_outputs/enhanced_framework")
    if output_base.exists():
        for item in output_base.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(output_base)
                print(f"  {rel_path}")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)