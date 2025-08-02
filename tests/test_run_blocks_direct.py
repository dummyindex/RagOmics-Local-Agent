#!/usr/bin/env python3
"""Run function blocks directly to generate test outputs."""

import sys
import json
import shutil
from pathlib import Path

# Get the absolute path to test_function_blocks
project_root = Path(__file__).parent.parent
test_blocks_dir = project_root / "test_function_blocks"
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Test blocks dir: {test_blocks_dir}")
print(f"Test blocks exists: {test_blocks_dir.exists()}")


def run_single_block(block_name, block_path, input_h5ad, output_dir, parameters=None):
    """Run a single function block."""
    print(f"\n{'='*50}")
    print(f"Running: {block_name}")
    print(f"{'='*50}")
    
    # Import the block's code
    code_file = test_blocks_dir / block_path / "code.py"
    if not code_file.exists():
        print(f"✗ Code file not found: {code_file}")
        return False
    
    # Load the code module
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{block_name}_code", str(code_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Prepare directories
    block_output_dir = output_dir / block_name
    block_output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = block_output_dir / "input"
    output_dir_block = block_output_dir / "output"
    figures_dir = output_dir_block / "figures"
    
    input_dir.mkdir(exist_ok=True)
    output_dir_block.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Copy input data
    if input_h5ad and input_h5ad.exists():
        shutil.copy2(input_h5ad, input_dir / "anndata.h5ad")
    
    # Create context
    context = {
        "node_id": f"test-{block_name}",
        "tree_id": "test-direct",
        "input_files": [
            {
                "filename": "anndata.h5ad",
                "filepath": str(input_dir / "anndata.h5ad"),
                "filetype": "anndata"
            }
        ] if input_h5ad else [],
        "available_files": [],
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir_block),
            "figures_dir": str(figures_dir)
        }
    }
    
    # Run the block
    try:
        result = module.run(
            context=context,
            parameters=parameters or {},
            input_dir=str(input_dir),
            output_dir=str(output_dir_block)
        )
        
        print("✓ Block completed successfully")
        
        # Check outputs
        output_files = list(output_dir_block.glob("*"))
        print(f"\nOutput files ({len(output_files)}):")
        for f in output_files:
            if f.is_file():
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        # Check figures
        figures = list(figures_dir.glob("*.png"))
        if figures:
            print(f"\nFigures ({len(figures)}):")
            for f in figures:
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        if result and 'metadata' in result:
            print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
        
        return output_dir_block / "anndata.h5ad"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run test pipeline."""
    print("="*60)
    print("Direct Function Block Execution Test")
    print("="*60)
    
    # Create test data
    output_base = Path("test_outputs/direct_execution")
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Creating test data...")
    import scanpy as sc
    import numpy as np
    
    n_cells = 500
    n_genes = 2000
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    adata = sc.AnnData(X)
    
    # Gene names
    var_names = [f"Gene_{i}" for i in range(n_genes)]
    # Add mitochondrial genes
    mt_indices = np.random.choice(n_genes, 50, replace=False)
    for idx in mt_indices:
        var_names[idx] = f"MT-{var_names[idx]}"
    adata.var_names = var_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    # Add layers for velocity
    adata.layers['spliced'] = X * 0.7
    adata.layers['unspliced'] = X * 0.3
    
    test_data = output_base / "test_data.h5ad"
    adata.write(test_data)
    print(f"  Created: {test_data}")
    print(f"  Shape: {adata.shape}")
    
    # Run pipeline
    pipeline = [
        ("basic_qc", "quality_control/basic_qc", {"min_genes": 100, "max_mt_percent": 25}),
        ("scvelo_preprocessing", "preprocessing/scvelo_preprocessing", {"n_top_genes": 1000}),
        ("velocity_steady_state", "velocity_analysis/velocity_steady_state", {"mode": "stochastic"}),
        ("elpigraph_trajectory", "trajectory_inference/elpigraph_trajectory", {"n_nodes": 30})
    ]
    
    current_input = test_data
    results = []
    
    for block_name, block_path, params in pipeline:
        output_h5ad = run_single_block(
            block_name, 
            block_path, 
            current_input, 
            output_base,
            params
        )
        
        results.append((block_name, output_h5ad is not None))
        
        if output_h5ad and output_h5ad.exists():
            current_input = output_h5ad
        else:
            print(f"\nPipeline stopped at {block_name}")
            break
    
    # Test bug fixer blocks
    print("\n" + "="*60)
    print("Testing Bug Fixer Blocks")
    print("="*60)
    
    # Test failing block
    failing_output = run_single_block(
        "trajectory_failing",
        "trajectory_inference/trajectory_failing",
        test_data,
        output_base,
        {}
    )
    results.append(("trajectory_failing (should fail)", failing_output is None))
    
    # Test fixed block
    fixed_output = run_single_block(
        "trajectory_fixed",
        "trajectory_inference/trajectory_fixed",
        test_data,
        output_base,
        {"n_nodes": 20}
    )
    results.append(("trajectory_fixed (should succeed)", fixed_output is not None))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    print(f"\nAll outputs saved to: {output_base}")
    
    # List all generated files
    print("\nGenerated files:")
    for f in output_base.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(output_base)
            print(f"  {rel_path}")


if __name__ == "__main__":
    main()