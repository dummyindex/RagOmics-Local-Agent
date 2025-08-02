#!/usr/bin/env python3
"""Updated scVelo manual tests using the enhanced framework with test function blocks."""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode
)
from analysis_tree_management import AnalysisTreeManager
from analysis_tree_management.node_executor import NodeExecutor
from job_executors import ExecutorManager
from job_executors.enhanced_executor import EnhancedExecutor
from utils.function_block_loader import FunctionBlockLoader


def test_scvelo_tree_1_steady_state(input_data_path: Path, output_dir: Path):
    """Test tree 1: Basic steady-state velocity analysis using enhanced framework."""
    print("\n=== Test Tree 1: Steady-State Velocity Analysis (Enhanced) ===")
    
    # Load function blocks from test directory
    loader = FunctionBlockLoader("test_function_blocks")
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    
    # Replace executors with enhanced versions
    for lang in ['python', 'r']:
        if lang in executor_manager.executors:
            base_executor = executor_manager.executors[lang]
            executor_manager.executors[lang] = EnhancedExecutor(
                docker_client=base_executor.client,
                image_name=base_executor.image_name,
                container_config=base_executor.container_config
            )
    
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Compute RNA velocity using steady-state model",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_EXISTING
    )
    
    # Load function blocks
    preprocessing_block = loader.load_function_block("preprocessing/scvelo_preprocessing")
    velocity_block = loader.load_function_block("velocity_analysis/velocity_steady_state")
    
    if not preprocessing_block or not velocity_block:
        print("Failed to load required function blocks")
        return False
    
    # Build tree structure
    preprocessing_node = tree_manager.add_root_node(preprocessing_block)
    
    # Execute preprocessing
    print("\n1. Executing preprocessing...")
    tree_manager.update_node_execution(preprocessing_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(
        node=preprocessing_node,
        tree=tree,
        input_path=input_data_path,
        output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            preprocessing_node.id,
            state=state,
            output_data_id=preprocessing_node.output_data_id,
            duration=preprocessing_node.duration
        )
        print("   ✓ Preprocessing completed")
        if result:
            print(f"   Output files: {[f.filename for f in result.output_files]}")
            print(f"   Metadata: {result.metadata}")
    else:
        print(f"   ✗ Preprocessing failed: {preprocessing_node.error}")
        return False
    
    # Add velocity computation
    velocity_nodes = tree_manager.add_child_nodes(preprocessing_node.id, [velocity_block])
    velocity_node = velocity_nodes[0]
    
    # Execute velocity computation
    print("\n2. Computing steady-state velocity...")
    tree_manager.update_node_execution(velocity_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(
        node=velocity_node,
        tree=tree,
        input_path=Path(preprocessing_node.output_data_id),
        output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            velocity_node.id,
            state=state,
            output_data_id=velocity_node.output_data_id,
            duration=velocity_node.duration
        )
        print("   ✓ Velocity computation completed")
        if result:
            print(f"   Output files: {[f.filename for f in result.output_files]}")
            print(f"   Metadata: {result.metadata}")
    else:
        print(f"   ✗ Velocity computation failed: {velocity_node.error}")
        return False
    
    # Save tree
    tree_manager.save_tree(output_dir / tree.id / "analysis_tree.json")
    
    # Display tree summary
    summary = tree_manager.get_summary()
    print(f"\nTree Summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Completed: {summary['completed_nodes']}")
    print(f"  Failed: {summary['failed_nodes']}")
    print(f"  Total duration: {summary['total_duration_seconds']:.1f}s")
    
    # Verify outputs
    success = state == NodeState.COMPLETED
    if success:
        # Check that output files exist
        tree_output_dir = output_dir / tree.id
        tree_json = tree_output_dir / "analysis_tree.json"
        
        if not tree_json.exists():
            print(f"   ✗ Missing analysis_tree.json")
            success = False
        
        # Check for figures in the final node
        figures_dir = tree_output_dir / tree.id / velocity_node.id / "output" / "figures"
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png"))
            print(f"\n   Generated {len(figures)} figures:")
            for fig in figures[:5]:  # Show first 5
                print(f"     - {fig.name}")
            if len(figures) > 5:
                print(f"     ... and {len(figures) - 5} more")
        else:
            print(f"   ✗ No figures directory found")
            success = False
    
    return success


def test_scvelo_tree_2_pipeline(input_data_path: Path, output_dir: Path):
    """Test tree 2: Complete pipeline with QC, preprocessing, and velocity."""
    print("\n=== Test Tree 2: Complete Pipeline (Enhanced) ===")
    
    # Load function blocks
    loader = FunctionBlockLoader("test_function_blocks")
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    
    # Replace executors with enhanced versions
    for lang in ['python', 'r']:
        if lang in executor_manager.executors:
            base_executor = executor_manager.executors[lang]
            executor_manager.executors[lang] = EnhancedExecutor(
                docker_client=base_executor.client,
                image_name=base_executor.image_name,
                container_config=base_executor.container_config
            )
    
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Complete pipeline: QC, preprocessing, velocity, trajectory",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_EXISTING
    )
    
    # Pipeline configuration
    pipeline = [
        ("quality_control/basic_qc", {"min_genes": 200, "max_mt_percent": 20}),
        ("preprocessing/scvelo_preprocessing", {"n_top_genes": 2000}),
        ("velocity_analysis/velocity_steady_state", {"mode": "stochastic"}),
        ("trajectory_inference/elpigraph_trajectory", {"n_nodes": 40})
    ]
    
    # Execute pipeline
    previous_node = None
    all_success = True
    
    for i, (block_path, params) in enumerate(pipeline):
        print(f"\n{i+1}. Loading {block_path}...")
        
        # Load block
        block = loader.load_function_block(block_path)
        if not block:
            print(f"   ✗ Failed to load block: {block_path}")
            all_success = False
            break
        
        # Set parameters
        block.parameters = params
        
        # Create node
        if previous_node:
            nodes = tree_manager.add_child_nodes(previous_node.id, [block])
            node = nodes[0]
            input_path = Path(previous_node.output_data_id)
        else:
            node = tree_manager.add_root_node(block)
            input_path = input_data_path
        
        # Execute node
        print(f"   Executing {block.name}...")
        tree_manager.update_node_execution(node.id, NodeState.RUNNING)
        
        state, result = node_executor.execute_node(
            node=node,
            tree=tree,
            input_path=input_path,
            output_base_dir=output_dir / tree.id
        )
        
        if state == NodeState.COMPLETED:
            tree_manager.update_node_execution(
                node.id,
                state=state,
                output_data_id=node.output_data_id,
                duration=node.duration
            )
            print(f"   ✓ {block.name} completed")
            if result:
                print(f"     Output files: {[f.filename for f in result.output_files]}")
                if result.metadata:
                    print(f"     Key metrics: {list(result.metadata.keys())}")
            previous_node = node
        else:
            print(f"   ✗ {block.name} failed: {node.error}")
            all_success = False
            break
    
    # Save tree
    tree_manager.save_tree(output_dir / tree.id / "analysis_tree.json")
    
    # Display tree summary
    summary = tree_manager.get_summary()
    print(f"\nTree Summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Completed: {summary['completed_nodes']}")
    print(f"  Failed: {summary['failed_nodes']}")
    print(f"  Total duration: {summary['total_duration_seconds']:.1f}s")
    
    # Verify outputs
    if all_success:
        tree_output_dir = output_dir / tree.id
        
        # Count total figures across all nodes
        total_figures = 0
        for node_id in tree.nodes:
            figures_dir = tree_output_dir / tree.id / node_id / "output" / "figures"
            if figures_dir.exists():
                figures = list(figures_dir.glob("*.png"))
                total_figures += len(figures)
        
        print(f"\n   Total figures generated: {total_figures}")
        
        # Check for output files
        for node_id in tree.nodes:
            output_dir_node = tree_output_dir / tree.id / node_id / "output"
            if output_dir_node.exists():
                files = list(output_dir_node.glob("*"))
                if files:
                    print(f"   Node {node_id[:8]} outputs: {len(files)} files")
    
    return all_success


def main():
    """Run all enhanced scVelo test trees."""
    # Check if test data exists
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if not input_data.exists():
        print(f"Error: Test data not found at {input_data}")
        print("Creating synthetic test data...")
        
        # Create synthetic data
        import scanpy as sc
        import numpy as np
        
        n_cells = 500
        n_genes = 2000
        
        # Create spliced/unspliced layers
        spliced = np.random.negative_binomial(10, 0.3, size=(n_cells, n_genes))
        unspliced = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))
        
        adata = sc.AnnData(spliced + unspliced)
        adata.layers['spliced'] = spliced
        adata.layers['unspliced'] = unspliced
        
        # Add gene names
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        
        # Add mitochondrial genes
        mt_genes = np.random.choice(n_genes, size=50, replace=False)
        for idx in mt_genes:
            adata.var_names[idx] = f"MT-{adata.var_names[idx]}"
        
        # Save
        input_data.parent.mkdir(parents=True, exist_ok=True)
        adata.write(input_data)
        print(f"Created synthetic data: {input_data}")
    
    output_base = Path("test_outputs/scvelo_trees_enhanced")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define tests
    tests = [
        ("Steady-State Model", test_scvelo_tree_1_steady_state),
        ("Complete Pipeline", test_scvelo_tree_2_pipeline)
    ]
    
    # Run tests
    results = []
    
    for name, test_func in tests:
        output_dir = output_base / name.lower().replace(" ", "_")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        try:
            success = test_func(input_data, output_dir)
            results.append((name, success))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} passed")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)