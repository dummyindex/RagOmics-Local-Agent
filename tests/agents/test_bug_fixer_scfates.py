#!/usr/bin/env python3
"""Test bug fixer agent with scFates trajectory inference."""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
from ragomics_agent_local.agents import BugFixerAgent
from ragomics_agent_local.utils import setup_logger

logger = setup_logger("test_bug_fixer_scfates")


def create_buggy_scfates_block():
    """Create a scFates function block with intentional dependency issues."""
    config = StaticConfig(
        args=[
            Arg(name="n_waypoints", value_type="int", description="Number of waypoints",
                optional=True, default_value=150)
        ],
        description="Run scFates trajectory inference",
        tag="trajectory",
        source="test"
    )
    
    # This code has intentional issues:
    # 1. Missing import for scFates
    # 2. Wrong attribute access patterns
    # 3. Missing preprocessing steps
    code = '''
def run(path_dict, params):
    """Run scFates trajectory inference."""
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Running scFates trajectory inference...")
    
    # This will fail - scFates not imported
    scf.tl.curve(adata, Nodes=n_waypoints)
    
    # This will fail - wrong method name
    scf.tl.tree(adata, method='ppt')
    
    # This will fail - missing preprocessing
    scf.tl.test_association(adata)
    
    # Plotting will fail
    scf.pl.graph(adata, basis="umap")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="scfates_trajectory",
        type=FunctionBlockType.PYTHON,
        description="scFates trajectory inference",
        code=code,
        requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0\npandas>=2.0.0",  # Missing scFates!
        parameters={"n_waypoints": 150},
        static_config=config
    )


def create_fixed_scfates_block():
    """Create a properly working scFates function block."""
    config = StaticConfig(
        args=[
            Arg(name="n_waypoints", value_type="int", description="Number of waypoints",
                optional=True, default_value=150),
            Arg(name="n_jobs", value_type="int", description="Number of parallel jobs",
                optional=True, default_value=4)
        ],
        description="Run scFates trajectory inference",
        tag="trajectory",
        source="test"
    )
    
    code = '''
def run(path_dict, params):
    """Run scFates trajectory inference."""
    import scanpy as sc
    import scFates as scf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    warnings.filterwarnings('ignore')
    
    print("Running scFates trajectory inference...")
    print(f"Input shape: {adata.shape}")
    
    # Ensure we have required preprocessing
    if 'X_pca' not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, svd_solver='arpack')
    
    if 'neighbors' not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
    
    # Compute force-directed graph if not present
    if 'X_draw_graph_fa' not in adata.obsm:
        print("Computing force-directed layout...")
        sc.tl.draw_graph(adata, layout='fa')
    
    # Run elastic principal graph
    print(f"Learning trajectory with {n_waypoints} waypoints...")
    scf.tl.curve(adata, Nodes=n_waypoints, use_rep="X_pca", ndims_rep=20)
    
    # Convert to principal tree
    print("Converting to principal tree...")
    scf.tl.tree(adata, method='ppt', Nodes=n_waypoints//2, use_rep="X_pca")
    
    # Select root based on pseudotime if available
    if 'dpt_pseudotime' in adata.obs.columns:
        root_cell = adata.obs['dpt_pseudotime'].idxmin()
        root_milestone = adata.obs.loc[root_cell, 'milestones']
        print(f"Setting root to milestone {root_milestone}")
        scf.tl.root(adata, root_milestone=root_milestone)
    else:
        # Use first milestone as root
        scf.tl.root(adata, root_milestone=1)
    
    # Calculate pseudotime
    print("Calculating trajectory pseudotime...")
    scf.tl.pseudotime(adata, n_jobs=n_jobs)
    
    # Test associations with pseudotime
    print("Testing feature associations...")
    scf.tl.test_association(adata, n_jobs=n_jobs)
    
    # Fit GAM for gene expression trends
    if 'highly_variable' in adata.var.columns:
        # Get top variable genes
        top_genes = adata.var[adata.var.highly_variable].index[:50]
        print(f"Fitting GAM for {len(top_genes)} genes...")
        scf.tl.fit(adata, top_genes, n_jobs=n_jobs)
    
    # Generate plots
    print("Generating trajectory plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory on UMAP
    scf.pl.graph(adata, basis="umap", ax=axes[0, 0], show=False)
    axes[0, 0].set_title("Trajectory on UMAP")
    
    # Pseudotime
    sc.pl.umap(adata, color='t', cmap='viridis', ax=axes[0, 1], show=False)
    axes[0, 1].set_title("Pseudotime")
    
    # Milestones
    sc.pl.umap(adata, color='milestones', ax=axes[1, 0], show=False, 
               legend_loc='on data')
    axes[1, 0].set_title("Milestones")
    
    # Segments
    if 'seg' in adata.obs.columns:
        sc.pl.umap(adata, color='seg', ax=axes[1, 1], show=False,
                   legend_loc='on data')
        axes[1, 1].set_title("Segments")
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/trajectory_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Tree plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scf.pl.tree(adata, basis="umap", ax=ax, show=False)
    plt.savefig('/workspace/output/figures/trajectory_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save trajectory statistics
    stats = {
        'n_milestones': len(adata.uns['milestones_colors']),
        'n_segments': adata.obs['seg'].nunique() if 'seg' in adata.obs else 0,
        'pseudotime_range': [float(adata.obs['t'].min()), float(adata.obs['t'].max())],
        'root_milestone': int(adata.uns.get('root', 1))
    }
    
    with open('/workspace/output/trajectory_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("scFates analysis completed!")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="scfates_trajectory_fixed",
        type=FunctionBlockType.PYTHON,
        description="scFates trajectory inference (fixed)",
        code=code,
        requirements="scanpy>=1.9.0\nscFates>=1.0.0\nmatplotlib>=3.6.0\npandas>=2.0.0\nnumpy>=1.24.0\nscikit-learn>=1.0.0",
        parameters={"n_waypoints": 150, "n_jobs": 4},
        static_config=config
    )


def test_bug_fixer_common_patterns():
    """Test bug fixer with common error patterns."""
    print("\n=== Testing Bug Fixer Common Patterns ===")
    
    bug_fixer = BugFixerAgent()
    
    # Test 1: Missing module
    print("\n1. Testing missing module fix...")
    buggy_block = create_buggy_scfates_block()
    
    result = bug_fixer.process({
        'function_block': buggy_block,
        'error_message': "NameError: name 'scf' is not defined",
        'stdout': '',
        'stderr': "NameError: name 'scf' is not defined\n  File 'function_block.py', line 10"
    })
    
    print(f"   Success: {result['success']}")
    print(f"   Reasoning: {result['reasoning']}")
    
    if result['success'] and result['fixed_code']:
        print("   Fixed code snippet:")
        lines = result['fixed_code'].split('\n')
        for i, line in enumerate(lines[8:13]):  # Show lines around the fix
            print(f"     {i+8}: {line}")
        
    # Test 2: Missing dependency in requirements
    print("\n2. Testing missing dependency fix...")
    
    result = bug_fixer.process({
        'function_block': buggy_block,
        'error_message': "ModuleNotFoundError: No module named 'scFates'",
        'stdout': '',
        'stderr': "ModuleNotFoundError: No module named 'scFates'"
    })
    
    print(f"   Success: {result['success']}")
    print(f"   Reasoning: {result['reasoning']}")
    
    if result['success'] and result['fixed_requirements']:
        print("   Fixed requirements:")
        for req in result['fixed_requirements'].split('\n'):
            if req.strip():
                print(f"     - {req.strip()}")
    
    # Test 3: Attribute error
    print("\n3. Testing attribute error fix...")
    
    result = bug_fixer.process({
        'function_block': buggy_block,
        'error_message': "AttributeError: module 'scanpy' has no attribute 'velocity'",
        'stdout': '',
        'stderr': "AttributeError: module 'scanpy.pl' has no attribute 'velocity_embedding'"
    })
    
    print(f"   Success: {result['success']}")
    print(f"   Reasoning: {result['reasoning']}")


def test_bug_fixer_iterative():
    """Test iterative bug fixing process."""
    print("\n=== Testing Iterative Bug Fixing ===")
    
    bug_fixer = BugFixerAgent()
    buggy_block = create_buggy_scfates_block()
    
    # Simulate iterative fixing
    errors = [
        {
            'error': "NameError: name 'scf' is not defined",
            'stderr': "NameError: name 'scf' is not defined"
        },
        {
            'error': "ModuleNotFoundError: No module named 'scFates'",
            'stderr': "ModuleNotFoundError: No module named 'scFates'"
        },
        {
            'error': "AttributeError: 'AnnData' object has no attribute 'obsm'",
            'stderr': "KeyError: 'X_pca' not found in adata.obsm"
        }
    ]
    
    current_block = buggy_block
    previous_attempts = []
    
    for i, error_info in enumerate(errors):
        print(f"\nIteration {i+1}: {error_info['error']}")
        
        result = bug_fixer.process({
            'function_block': current_block,
            'error_message': error_info['error'],
            'stdout': '',
            'stderr': error_info['stderr'],
            'previous_attempts': previous_attempts
        })
        
        print(f"   Success: {result['success']}")
        print(f"   Reasoning: {result['reasoning']}")
        
        if result['success']:
            if result['fixed_code']:
                current_block.code = result['fixed_code']
            if result['fixed_requirements']:
                current_block.requirements = result['fixed_requirements']
            
            previous_attempts.append(current_block.code)
        else:
            print("   Unable to fix automatically")
            break
    
    print("\nFinal requirements:")
    for req in current_block.requirements.split('\n'):
        if req.strip():
            print(f"  - {req.strip()}")


def test_bug_fixer_comparison():
    """Compare buggy and fixed versions."""
    print("\n=== Comparing Buggy vs Fixed scFates ===")
    
    buggy = create_buggy_scfates_block()
    fixed = create_fixed_scfates_block()
    
    print("\nBuggy version issues:")
    print("  - Missing 'import scFates as scf'")
    print("  - Missing scFates in requirements")
    print("  - No preprocessing checks")
    print("  - No error handling")
    
    print("\nFixed version improvements:")
    print("  - Proper imports")
    print("  - Complete requirements")
    print("  - Preprocessing validation")
    print("  - Comprehensive plotting")
    print("  - Statistics export")
    
    print("\nRequirements diff:")
    buggy_reqs = set(r.strip() for r in buggy.requirements.split('\n') if r.strip())
    fixed_reqs = set(r.strip() for r in fixed.requirements.split('\n') if r.strip())
    
    print("  Added:")
    for req in sorted(fixed_reqs - buggy_reqs):
        print(f"    + {req}")


def main():
    """Run all bug fixer tests."""
    test_bug_fixer_common_patterns()
    test_bug_fixer_iterative()
    test_bug_fixer_comparison()
    
    print("\n=== Bug Fixer Tests Complete ===")


if __name__ == "__main__":
    main()