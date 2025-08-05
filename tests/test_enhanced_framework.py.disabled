#!/usr/bin/env python3
"""Test the enhanced function block framework with general file support."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import docker
sys.path.append(str(Path(__file__).parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    ExecutionContext, ExecutionRequest, FileInfo, FileType
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.job_executors.enhanced_executor import EnhancedExecutor
from ragomics_agent_local.utils.function_block_loader import FunctionBlockLoader


def test_function_block_loader():
    """Test loading function blocks from directory."""
    print("\n=== Testing Function Block Loader ===")
    
    loader = FunctionBlockLoader("test_function_blocks")
    
    # List available blocks
    blocks = loader.list_available_blocks()
    print(f"\nFound {len(blocks)} function blocks:")
    for block in blocks:
        print(f"  - {block['name']} ({block['type']}): {block['description']}")
        print(f"    Path: {block['path']}")
        print(f"    Tags: {', '.join(block['tags'])}")
    
    # Load a specific block
    print("\n--- Loading scvelo_preprocessing block ---")
    preprocessing_block = loader.load_function_block("preprocessing/scvelo_preprocessing")
    if preprocessing_block:
        print(f"  Name: {preprocessing_block.name}")
        print(f"  Type: {preprocessing_block.type}")
        print(f"  Args: {[arg.name for arg in preprocessing_block.static_config.args]}")
        if preprocessing_block.static_config.input_specification:
            print(f"  Input files: {preprocessing_block.static_config.input_specification.required_files}")
        if preprocessing_block.static_config.output_specification:
            print(f"  Output files: {preprocessing_block.static_config.output_specification.output_files}")
    
    return loader, blocks


def test_enhanced_executor():
    """Test the enhanced executor with general file support."""
    print("\n=== Testing Enhanced Executor ===")
    
    # Load function blocks
    loader = FunctionBlockLoader("test_function_blocks")
    qc_block = loader.load_function_block("quality_control/basic_qc")
    
    if not qc_block:
        print("Failed to load QC block")
        return False
    
    # Create test data
    test_data_dir = Path("test_outputs/enhanced_framework/input_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test file
    print("\nCreating test data...")
    import scanpy as sc
    import numpy as np
    
    # Create synthetic data
    n_cells = 1000
    n_genes = 2000
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    adata = sc.AnnData(X)
    
    # Add gene names
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    # Add some mitochondrial genes
    mt_genes = np.random.choice(n_genes, size=50, replace=False)
    for idx in mt_genes:
        adata.var_names[idx] = f"MT-Gene_{idx}"
    
    # Save test data
    test_file = test_data_dir / "anndata.h5ad"
    adata.write(test_file)
    print(f"Created test data: {test_file}")
    
    # Create analysis tree and node
    tree = AnalysisTree(
        user_request="Perform quality control",
        input_data_path=str(test_data_dir),
        max_nodes=10,
        max_children_per_node=3,
        max_debug_trials=3
    )
    
    node = AnalysisNode(
        analysis_id=tree.id,
        function_block=qc_block,
        level=0
    )
    
    tree.add_node(node)
    tree.root_node_id = node.id
    
    # Create execution context
    context = ExecutionContext(
        node_id=node.id,
        tree_id=tree.id,
        input_files=[
            FileInfo(
                filename="anndata.h5ad",
                filepath=str(test_file),
                filetype=FileType.ANNDATA,
                description="Test data"
            )
        ],
        available_files=[],
        input_dir="/workspace/input",
        output_dir="/workspace/output",
        figures_dir="/workspace/output/figures",
        tree_metadata={"test": True}
    )
    
    # Create execution request
    request = ExecutionRequest(
        node=node,
        execution_context=context
    )
    
    # Execute with enhanced executor
    print("\nExecuting function block with enhanced framework...")
    
    try:
        client = docker.from_env()
        executor = EnhancedExecutor(
            docker_client=client,
            image_name="ragomics-python:local"
        )
        
        workspace_dir = Path("test_outputs/enhanced_framework/workspace")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        response = executor.execute(request, workspace_dir)
        
        if response.success:
            print("✓ Execution successful!")
            print(f"  Output files: {len(response.execution_result.output_files)}")
            for file_info in response.execution_result.output_files:
                print(f"    - {file_info.filename} ({file_info.filetype})")
            print(f"  Figures: {len(response.execution_result.figures)}")
            print(f"  Metadata: {response.execution_result.metadata}")
            return True
        else:
            print(f"✗ Execution failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_pipeline():
    """Test a complete pipeline with the enhanced framework."""
    print("\n=== Testing Enhanced Pipeline ===")
    
    # Load function blocks
    loader = FunctionBlockLoader("test_function_blocks")
    
    # Build pipeline
    pipeline_blocks = [
        ("quality_control/basic_qc", {}),
        ("preprocessing/scvelo_preprocessing", {"n_top_genes": 1000}),
        ("velocity_analysis/velocity_steady_state", {"mode": "stochastic"}),
        ("trajectory_inference/elpigraph_trajectory", {"n_nodes": 30})
    ]
    
    # Create test data
    test_data_dir = Path("test_outputs/enhanced_framework/pipeline_input")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create more complex test data
    print("\nCreating test data for pipeline...")
    import scanpy as sc
    import numpy as np
    
    n_cells = 500
    n_genes = 1000
    
    # Create synthetic spliced/unspliced data
    spliced = np.random.negative_binomial(10, 0.3, size=(n_cells, n_genes))
    unspliced = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))
    
    adata = sc.AnnData(spliced + unspliced)
    adata.layers['spliced'] = spliced
    adata.layers['unspliced'] = unspliced
    
    # Add gene names
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    # Add some mitochondrial genes
    mt_genes = np.random.choice(n_genes, size=30, replace=False)
    for idx in mt_genes:
        adata.var_names[idx] = f"MT-Gene_{idx}"
    
    # Save test data
    test_file = test_data_dir / "anndata.h5ad"
    adata.write(test_file)
    print(f"Created test data: {test_file}")
    
    # Create analysis tree
    tree_manager = AnalysisTreeManager()
    tree = tree_manager.create_tree(
        user_request="Run complete velocity and trajectory analysis",
        input_data_path=str(test_data_dir),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_EXISTING
    )
    
    # Set up executor manager with enhanced executor
    executor_manager = ExecutorManager()
    
    # Wrap executors with enhanced version
    for lang in ['python', 'r']:
        if lang in executor_manager.executors:
            base_executor = executor_manager.executors[lang]
            executor_manager.executors[lang] = EnhancedExecutor(
                docker_client=base_executor.client,
                image_name=base_executor.image_name,
                container_config=base_executor.container_config
            )
    
    # Create node executor
    node_executor = NodeExecutor(executor_manager)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    output_dir = Path("test_outputs/enhanced_framework/pipeline_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_input = test_data_dir
    previous_node = None
    
    for i, (block_path, params) in enumerate(pipeline_blocks):
        print(f"\n--- Step {i+1}: {block_path} ---")
        
        # Load block
        block = loader.load_function_block(block_path)
        if not block:
            print(f"Failed to load block: {block_path}")
            continue
        
        # Set parameters
        block.parameters = params
        
        # Create node
        if previous_node:
            nodes = tree_manager.add_child_nodes(previous_node.id, [block])
            node = nodes[0]
        else:
            node = tree_manager.add_root_node(block)
        
        # Execute node
        print(f"Executing {block.name}...")
        state, result = node_executor.execute_node(
            node=node,
            tree=tree,
            input_path=current_input,
            output_base_dir=output_dir
        )
        
        if state == NodeState.COMPLETED:
            print(f"✓ {block.name} completed successfully")
            if result:
                print(f"  Output files: {len(result.output_files)}")
                for file_info in result.output_files:
                    print(f"    - {file_info.filename} ({file_info.filetype})")
                print(f"  Metadata: {result.metadata}")
            
            # Update input for next step
            current_input = Path(node.output_data_id)
            previous_node = node
        else:
            print(f"✗ {block.name} failed")
            break
    
    # Save analysis tree
    tree_json_path = output_dir / "analysis_tree.json"
    tree_manager.save_tree(tree_json_path)
    print(f"\nSaved analysis tree to: {tree_json_path}")
    
    # Check results
    success = tree.completed_nodes == len(pipeline_blocks)
    print(f"\n{'✓' if success else '✗'} Pipeline {'completed' if success else 'failed'}")
    print(f"  Completed nodes: {tree.completed_nodes}/{len(pipeline_blocks)}")
    
    return success


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Enhanced Function Block Framework")
    print("="*60)
    
    results = []
    
    # Test 1: Function block loader
    try:
        loader, blocks = test_function_block_loader()
        results.append(("Function Block Loader", True))
    except Exception as e:
        print(f"\nError in function block loader test: {e}")
        results.append(("Function Block Loader", False))
    
    # Test 2: Enhanced executor
    try:
        success = test_enhanced_executor()
        results.append(("Enhanced Executor", success))
    except Exception as e:
        print(f"\nError in enhanced executor test: {e}")
        results.append(("Enhanced Executor", False))
    
    # Test 3: Complete pipeline
    try:
        success = test_enhanced_pipeline()
        results.append(("Enhanced Pipeline", success))
    except Exception as e:
        print(f"\nError in pipeline test: {e}")
        results.append(("Enhanced Pipeline", False))
    
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