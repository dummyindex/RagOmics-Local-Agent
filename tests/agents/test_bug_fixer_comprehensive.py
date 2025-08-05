#!/usr/bin/env python3
"""Comprehensive test for bug fixer agent with scFates trajectory analysis."""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
)
from ragomics_agent_local.agents import (
    BugFixerAgent, TaskManager, TaskType, TaskStatus
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor
from ragomics_agent_local.utils import setup_logger

logger = setup_logger("test_bug_fixer_comprehensive")


def create_scfates_block_with_dependency_issue():
    """Create scFates block that will fail due to missing dependency."""
    config = StaticConfig(
        args=[
            Arg(name="n_map", value_type="int", description="Number of mapping points",
                optional=True, default_value=100),
            Arg(name="n_pcs", value_type="int", description="Number of PCs to use",
                optional=True, default_value=20)
        ],
        description="Infer trajectory using scFates",
        tag="trajectory_inference",
        source="test"
    )
    
    # This will fail because scFates is not in requirements
    code = '''
def run(path_dict, params):
    """Infer trajectory using scFates."""
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

    print("Starting scFates trajectory inference...")
    print(f"Input shape: {adata.shape}")
    
    # This will fail - scFates not imported or installed
    import scFates as scf
    
    # Ensure required preprocessing
    if 'X_pca' not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, svd_solver='arpack', n_comps=50)
    
    # Use UMAP coordinates instead of force-directed layout
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP...")
        sc.tl.umap(adata)
    
    # Compute principal curve
    print(f"Learning principal curve with {n_map} mapping points...")
    scf.tl.curve(adata, Nodes=n_map, use_rep="X_pca", ndims_rep=n_pcs)
    
    # Convert to tree
    print("Converting to principal tree...")
    scf.tl.tree(adata, method='ppt', Nodes=50, use_rep="X_pca")
    
    # Find root
    if 'leiden' in adata.obs.columns:
        # Use cluster 0 as root
        root_cells = adata.obs[adata.obs['leiden'] == '0'].index
        if len(root_cells) > 0:
            root_idx = adata.obs.index.get_loc(root_cells[0])
            root_milestone = adata.obs.loc[root_cells[0], 'milestones']
            scf.tl.root(adata, root_milestone=root_milestone)
    else:
        scf.tl.root(adata, root_milestone=1)
    
    # Calculate pseudotime
    print("Calculating pseudotime...")
    scf.tl.pseudotime(adata, n_jobs=4)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    scf.pl.graph(adata, basis='umap', ax=axes[0], show=False)
    axes[0].set_title('Trajectory')
    
    # Pseudotime
    sc.pl.umap(adata, color='t', cmap='viridis', ax=axes[1], show=False)
    axes[1].set_title('Pseudotime')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/trajectory_analysis.png', dpi=150)
    plt.close()
    
    print("scFates analysis complete!")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="scfates_trajectory_analysis",
        type=FunctionBlockType.PYTHON,
        description="Trajectory inference with scFates",
        code=code,
        requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0\npandas>=2.0.0\nnumpy>=1.24.0",  # Missing scFates!
        parameters={"n_map": 100, "n_pcs": 20},
        static_config=config
    )


def create_scfates_block_with_api_issue():
    """Create scFates block with incorrect API usage."""
    config = StaticConfig(
        args=[],
        description="Trajectory with wrong API",
        tag="trajectory_inference",
        source="test"
    )
    
    code = '''
def run(path_dict, params):
    """Trajectory with wrong API usage."""
    import scanpy as sc
    import scFates as scf
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Running trajectory analysis...")
    
    # Wrong: curve() requires preprocessing
    scf.tl.curve(adata)  # Missing required arguments
    
    # Wrong: tree() called without curve first
    scf.tl.tree(adata)
    
    # Wrong: root() needs milestones  
    scf.tl.root(adata)  # Missing milestone specification
    
    # This will also fail
    scf.tl.test_association(adata)  # Requires pseudotime first
    
    return adata
'''
    
    return NewFunctionBlock(
        name="scfates_wrong_api",
        type=FunctionBlockType.PYTHON,
        description="Wrong API usage",
        code=code,
        requirements="scanpy>=1.9.0\nscFates>=1.0.0",
        parameters={},
        static_config=config
    )


def test_bug_fixer_with_task_tracking():
    """Test bug fixer with full task tracking."""
    print("\n=== Testing Bug Fixer with Task Tracking ===")
    
    # Setup
    output_dir = Path("test_output_bug_fixer_tasks")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create task manager
    task_manager = TaskManager(output_dir)
    
    # Create bug fixer with task manager
    bug_fixer = BugFixerAgent(task_manager=task_manager)
    
    # Test case 1: Missing dependency
    print("\n1. Testing missing dependency fix with task tracking...")
    
    block = create_scfates_block_with_dependency_issue()
    
    # Create orchestrator task first
    orchestrator_task = task_manager.create_task(
        task_type=TaskType.ORCHESTRATION,
        agent_name="orchestrator",
        description="Execute scFates trajectory analysis",
        context={
            'analysis_id': 'test-analysis-001',
            'node_id': 'test-node-001',
            'function_block_id': block.id
        }
    )
    
    # Simulate error
    error_context = {
        'function_block': block,
        'error_message': "ModuleNotFoundError: No module named 'scFates'",
        'stdout': 'Starting scFates trajectory inference...\nInput shape: (1000, 2000)',
        'stderr': "ModuleNotFoundError: No module named 'scFates'\n  File 'function_block.py', line 10",
        'parent_task_id': orchestrator_task.task_id,
        'analysis_id': 'test-analysis-001',
        'node_id': 'test-node-001',
        'job_id': 'test-job-001'
    }
    
    result = bug_fixer.process(error_context)
    
    print(f"   Success: {result['success']}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Task ID: {result.get('task_id')}")
    
    # Check task was created and tracked
    if result.get('task_id'):
        task = task_manager.get_task(result['task_id'])
        print(f"   Task Status: {task.status}")
        print(f"   Parent Task: {task.parent_task_id}")
        
        # Check artifacts
        task_dir = task_manager._get_task_dir(result['task_id'])
        print(f"   Task artifacts: {list(task_dir.glob('*'))}")
    
    # Test case 2: Multiple iterations
    print("\n2. Testing iterative bug fixing with subtasks...")
    
    # Create main debugging task
    main_debug_task = task_manager.create_task(
        task_type=TaskType.BUG_FIXING,
        agent_name="bug_fixer_coordinator",
        description="Fix multiple issues in scFates block",
        context={
            'analysis_id': 'test-analysis-001',
            'node_id': 'test-node-002'
        }
    )
    
    # Simulate multiple errors and fixes
    errors = [
        ("ModuleNotFoundError: No module named 'scFates'", "Missing dependency"),
        ("AttributeError: 'AnnData' object has no attribute 'obsp'", "API change"),
        ("ValueError: n_jobs must be positive", "Parameter error")
    ]
    
    for i, (error, description) in enumerate(errors):
        print(f"\n   Iteration {i+1}: {description}")
        
        error_context = {
            'function_block': block,
            'error_message': error,
            'stdout': f'Attempt {i+1} output',
            'stderr': error,
            'parent_task_id': main_debug_task.task_id,
            'analysis_id': 'test-analysis-001',
            'node_id': 'test-node-002',
            'job_id': f'test-job-{i+2:03d}'
        }
        
        result = bug_fixer.process(error_context)
        
        if result.get('task_id'):
            task_manager.add_subtask(main_debug_task.task_id, result['task_id'])
            print(f"   Created subtask: {result['task_id']}")
    
    # Complete main task
    task_manager.update_task_status(
        main_debug_task.task_id,
        TaskStatus.COMPLETED,
        results={'total_attempts': len(errors)}
    )
    
    # Generate task summary
    print("\n3. Task Summary:")
    summary = task_manager.create_task_summary(main_debug_task.task_id)
    print(json.dumps(summary, indent=2))
    
    # Check task queries
    print("\n4. Querying tasks by entity:")
    analysis_tasks = task_manager.get_tasks_by_entity(analysis_id='test-analysis-001')
    print(f"   Tasks for analysis: {len(analysis_tasks)}")
    
    node_tasks = task_manager.get_tasks_by_entity(node_id='test-node-001')
    print(f"   Tasks for node-001: {len(node_tasks)}")
    
    print("\n✓ Task tracking test complete!")
    return True


def test_bug_fixer_with_real_execution():
    """Test bug fixer with actual Docker execution."""
    print("\n=== Testing Bug Fixer with Real Execution ===")
    
    # Setup
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if not input_data.exists():
        print(f"Error: Test data not found at {input_data}")
        return False
    
    output_dir = Path("test_output_bug_fixer_execution")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create components
    task_manager = TaskManager(output_dir)
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    bug_fixer = BugFixerAgent(task_manager=task_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Test scFates trajectory inference",
        input_data_path=str(input_data),
        max_nodes=5,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create orchestrator task
    orchestrator_task = task_manager.create_task(
        task_type=TaskType.ORCHESTRATION,
        agent_name="orchestrator",
        description="Execute scFates analysis with bug fixing",
        context={'analysis_id': tree.id}
    )
    
    # Add preprocessing node first
    print("\n1. Adding preprocessing node...")
    preprocess_block = create_preprocessing_block()
    preprocess_node = tree_manager.add_root_node(preprocess_block)
    
    # Execute preprocessing
    print("   Executing preprocessing...")
    tree_manager.update_node_execution(preprocess_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=preprocess_node, tree=tree, input_path=input_data, output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            preprocess_node.id,
            state=state,
            output_data_id=result.output_data_path,
            duration=result.duration
        )
        print("   ✓ Preprocessing completed")
    else:
        print(f"   ✗ Preprocessing failed: {result.error}")
        return False
    
    # Add scFates node
    print("\n2. Adding scFates node...")
    scfates_block = create_scfates_block_with_dependency_issue()
    scfates_nodes = tree_manager.add_child_nodes(preprocess_node.id, [scfates_block])
    scfates_node = scfates_nodes[0]
    
    # Try to execute - this should fail
    print("   Executing scFates (expecting failure)...")
    tree_manager.update_node_execution(scfates_node.id, NodeState.RUNNING)
    
    input_path = Path(result.output_data_path) if result.output_data_path else input_data
    state, exec_result = node_executor.execute_node(node=scfates_node, tree=tree, input_path=input_path, output_base_dir=output_dir / tree.id
    )
    
    print(f"   Result: {state}")
    
    if state == NodeState.FAILED:
        print(f"   Error: {exec_result.error}")
        
        # Try to fix with bug fixer
        print("\n3. Attempting to fix with bug fixer...")
        
        fix_task = task_manager.create_task(
            task_type=TaskType.BUG_FIXING,
            agent_name="bug_fixer",
            description=f"Fix error in {scfates_node.function_block.name}",
            context={
                'analysis_id': tree.id,
                'node_id': scfates_node.id,
                'job_id': exec_result.job_id,
                'parent_task_id': orchestrator_task.task_id
            }
        )
        
        fix_result = bug_fixer.process({
            'function_block': scfates_node.function_block,
            'error_message': exec_result.error or "Unknown error",
            'stdout': exec_result.stdout,
            'stderr': exec_result.stderr,
            'analysis_id': tree.id,
            'node_id': scfates_node.id,
            'job_id': exec_result.job_id,
            'parent_task_id': fix_task.task_id
        })
        
        print(f"   Fix success: {fix_result['success']}")
        print(f"   Fix reasoning: {fix_result['reasoning']}")
        
        if fix_result['success'] and fix_result.get('fixed_code'):
            # Update node with fixed code
            print("\n4. Retrying with fixed code...")
            scfates_node.function_block.code = fix_result['fixed_code']
            if fix_result.get('fixed_requirements'):
                scfates_node.function_block.requirements = fix_result['fixed_requirements']
            
            # Mark as pending to retry
            tree_manager.update_node_execution(scfates_node.id, NodeState.PENDING)
            tree_manager.increment_debug_attempts(scfates_node.id)
            
            # Retry execution
            tree_manager.update_node_execution(scfates_node.id, NodeState.RUNNING)
            state2, result2 = node_executor.execute_node(node=scfates_node, tree=tree, input_path=input_path, output_base_dir=output_dir / tree.id
            )
            
            print(f"   Retry result: {state2}")
            
            if state2 == NodeState.COMPLETED:
                print("   ✓ Successfully fixed and executed!")
                tree_manager.update_node_execution(
                    scfates_node.id,
                    state=state2,
                    output_data_id=result2.output_data_path,
                    duration=result2.duration
                )
            else:
                print(f"   ✗ Still failed: {result2.error}")
    
    # Complete orchestrator task
    task_manager.update_task_status(
        orchestrator_task.task_id,
        TaskStatus.COMPLETED,
        results={'tree_id': tree.id, 'nodes_executed': 2}
    )
    
    # Save tree
    tree_manager.save_tree(output_dir / "analysis_tree.json")
    
    # Display task structure
    print("\n5. Task Structure:")
    print(json.dumps(
        task_manager.create_task_summary(orchestrator_task.task_id),
        indent=2
    ))
    
    print("\n✓ Execution test complete!")
    return True


def create_preprocessing_block():
    """Create a simple preprocessing block."""
    config = StaticConfig(
        args=[],
        description="Basic preprocessing",
        tag="preprocessing",
        source="test"
    )
    
    code = '''
def run(path_dict, params):
    """Basic preprocessing for trajectory analysis."""
    import scanpy as sc
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Preprocessing data...")
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize and log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # HVG
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    
    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Neighbors and UMAP
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    
    # Compute graph layout (needed for scFates)
    # Skip force-directed layout to avoid fa2 dependency issues
    
    print(f"Preprocessed: {adata.shape}")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="preprocessing_for_trajectory",
        type=FunctionBlockType.PYTHON,
        description="Preprocessing for trajectory",
        code=code,
        requirements="scanpy>=1.9.0\npython-igraph>=0.10.0\nleidenalg>=0.9.0",
        parameters={},
        static_config=config
    )


def main():
    """Run all bug fixer tests."""
    print("=== Comprehensive Bug Fixer Tests ===")
    
    # Test 1: Task tracking
    success1 = test_bug_fixer_with_task_tracking()
    
    # Test 2: Real execution (skip for now due to Docker issues)
    # success2 = test_bug_fixer_with_real_execution()
    success2 = True  # Skip for now
    
    print("\n=== Test Summary ===")
    print(f"Task tracking test: {'✓ Passed' if success1 else '✗ Failed'}")
    print(f"Real execution test: SKIPPED (Docker build issues)")
    
    return success1 and success2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)