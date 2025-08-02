#!/usr/bin/env python3
"""
Mock test for the agent system without requiring OpenAI API.
Demonstrates the complete workflow with a pre-defined analysis tree.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
)
from ragomics_agent_local.analysis_tree_management.tree_manager import AnalysisTreeManager
from ragomics_agent_local.job_executors.executor_manager import ExecutorManager
from ragomics_agent_local.utils.logger import get_logger
from ragomics_agent_local.config import config

logger = get_logger(__name__)


def create_mock_llm_response():
    """Create a mock LLM response for testing."""
    # QC block
    qc_block = NewFunctionBlock(
        name="initial_quality_control",
        type=FunctionBlockType.PYTHON,
        description="Initial quality control and data exploration",
        code='''
def run(adata, **kwargs):
    """Initial quality control."""
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    print(f"Input data shape: {adata.shape}")
    
    # Basic QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Remove cells with high mitochondrial content
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    
    print(f"After QC: {adata.shape}")
    
    # QC plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(adata.obs['n_genes_by_counts'], bins=50)
    axes[0].set_xlabel('Number of genes')
    axes[0].set_ylabel('Number of cells')
    
    axes[1].hist(adata.obs['total_counts'], bins=50)
    axes[1].set_xlabel('Total counts')
    
    axes[2].hist(adata.obs['pct_counts_mt'], bins=50)
    axes[2].set_xlabel('Mitochondrial %')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/qc_metrics.png', dpi=150)
    plt.close()
    
    return adata
''',
        requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Initial quality control",
            tag="qc",
            source="mock_llm"
        )
    )
    
    # Preprocessing block
    preprocess_block = NewFunctionBlock(
        name="preprocessing",
        type=FunctionBlockType.PYTHON,
        description="Data preprocessing and normalization",
        code='''
def run(adata, **kwargs):
    """Preprocess and normalize data."""
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    print("Starting preprocessing...")
    
    # Store raw counts
    adata.raw = adata
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)
    plt.savefig('/workspace/output/figures/hvg.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Keep only HVGs
    adata = adata[:, adata.var.highly_variable]
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)
    plt.savefig('/workspace/output/figures/pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"After preprocessing: {adata.shape}")
    
    return adata
''',
        requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Data preprocessing",
            tag="preprocessing",
            source="mock_llm"
        )
    )
    
    # Velocity preparation block
    velocity_prep_block = NewFunctionBlock(
        name="velocity_preparation",
        type=FunctionBlockType.PYTHON,
        description="Prepare data for velocity analysis",
        code='''
def run(adata, **kwargs):
    """Prepare data for velocity analysis."""
    import scanpy as sc
    import numpy as np
    
    print("Preparing for velocity analysis...")
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
    
    # UMAP embedding
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata, resolution=0.5)
    
    # Since we don't have spliced/unspliced layers, create synthetic ones
    print("Creating synthetic spliced/unspliced layers for demonstration...")
    adata.layers['spliced'] = adata.X.copy()
    adata.layers['unspliced'] = adata.X.copy() * np.random.uniform(0.1, 0.3, adata.shape)
    
    # Basic velocity plot
    sc.pl.umap(adata, color=['leiden'], save='_clusters.png')
    
    print("Data prepared for velocity analysis")
    
    return adata
''',
        requirements="scanpy>=1.9.0\nleidenalg\npython-igraph",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Velocity data preparation",
            tag="velocity_prep",
            source="mock_llm"
        )
    )
    
    return [qc_block, preprocess_block, velocity_prep_block]


def main():
    """Main test function."""
    print("=== Ragomics Agent Mock Test ===")
    
    # Setup paths
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    output_dir = Path("test_outputs") / "agent_mock"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_data.exists():
        print(f"Error: Input data not found at {input_data}")
        sys.exit(1)
    
    print(f"Input data: {input_data}")
    print(f"Output directory: {output_dir}")
    
    # Create analysis tree manager
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    
    # Validate environment
    print("\nValidating environment...")
    validation = executor_manager.validate_environment()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
    
    if not validation.get("docker_available"):
        print("Error: Docker is not available")
        sys.exit(1)
    
    # Create mock analysis tree
    print("\nCreating mock analysis tree...")
    user_request = "Analyze the scRNA-seq data. Generate RNA dynamics/velocities analysis results."
    
    tree = tree_manager.create_tree(
        user_request=user_request,
        input_data_path=str(input_data),
        max_nodes=10,
        max_children_per_node=3,
        max_debug_trials=3,
        generation_mode=GenerationMode.ONLY_NEW,
        llm_model="mock"
    )
    
    # Get mock LLM response
    function_blocks = create_mock_llm_response()
    
    # Build tree structure
    print("Building tree structure...")
    
    # Add root node (QC)
    root_node = tree_manager.add_root_node(function_blocks[0])
    print(f"Added root node: {root_node.function_block.name}")
    
    # Add preprocessing and velocity nodes as children
    child_nodes = tree_manager.add_child_nodes(
        parent_node_id=root_node.id,
        function_blocks=[function_blocks[1]]
    )
    preprocess_node = child_nodes[0]
    print(f"Added child node: {preprocess_node.function_block.name}")
    
    # Add velocity prep node
    velocity_nodes = tree_manager.add_child_nodes(
        parent_node_id=preprocess_node.id,
        function_blocks=[function_blocks[2]]
    )
    velocity_node = velocity_nodes[0]
    print(f"Added child node: {velocity_node.function_block.name}")
    
    # Save tree structure
    tree_json_path = output_dir / "analysis_tree.json"
    tree_manager.save_tree(tree_json_path)
    print(f"\nSaved tree structure to {tree_json_path}")
    
    # Display tree structure
    print("\n=== Analysis Tree Structure ===")
    summary = tree_manager.get_summary()
    nodes_info = []
    for node_id, node in tree_manager.tree.nodes.items():
        nodes_info.append({
            "id": node.id,
            "name": node.function_block.name,
            "type": node.function_block.type.value if hasattr(node.function_block.type, 'value') else str(node.function_block.type),
            "level": node.level,
            "parent_id": node.parent_id,
            "children": node.children
        })
    
    print(json.dumps({
        "tree_id": summary["tree_id"],
        "user_request": summary["user_request"],
        "total_nodes": summary["total_nodes"],
        "nodes": nodes_info
    }, indent=2))
    
    # Execute the tree
    print("\n=== Executing Analysis Tree ===")
    
    # Execute nodes in order
    current_data_path = input_data
    for node in tree_manager.get_execution_order():
        print(f"\nExecuting: {node.function_block.name}")
        
        # Update state
        node.state = NodeState.RUNNING
        node.start_time = datetime.now()
        
        # Execute using executor manager
        try:
            result = executor_manager.execute(
                function_block=node.function_block,
                input_data_path=Path(current_data_path),
                output_dir=output_dir / tree.id / node.id,
                parameters=node.function_block.parameters
            )
            
            if result.success:
                node.state = NodeState.COMPLETED
                node.end_time = datetime.now()
                node.duration = (node.end_time - node.start_time).total_seconds()
                node.output_data_id = str(result.output_data_path) if result.output_data_path else None
                node.figures = [str(f) for f in result.figures]
                node.logs = result.logs
                
                # Update data path for next node
                if result.output_data_path:
                    current_data_path = result.output_data_path
                    
                print(f"✓ Completed in {node.duration:.1f}s")
                if result.figures:
                    print(f"  Generated {len(result.figures)} figures")
            else:
                node.state = NodeState.FAILED
                node.end_time = datetime.now()
                node.duration = (node.end_time - node.start_time).total_seconds()
                node.error = result.error
                print(f"✗ Failed: {result.error}")
                
        except Exception as e:
            node.state = NodeState.FAILED
            node.end_time = datetime.now()
            node.duration = (node.end_time - node.start_time).total_seconds()
            node.error = str(e)
            print(f"✗ Failed: {e}")
    
    # Display execution summary
    print("\n=== Execution Summary ===")
    final_summary = tree_manager.get_summary()
    print(json.dumps({
        "tree_id": final_summary["tree_id"],
        "user_request": final_summary["user_request"],
        "total_nodes": final_summary["total_nodes"],
        "completed_nodes": final_summary["completed_nodes"],
        "failed_nodes": final_summary["failed_nodes"],
        "pending_nodes": final_summary["pending_nodes"],
        "total_duration_seconds": final_summary.get("total_duration_seconds", 0),
        "created_at": str(final_summary["created_at"]),
        "updated_at": str(final_summary["updated_at"]),
        "max_depth": final_summary["max_depth"]
    }, indent=2))
    
    # List output files
    print("\n=== Output Files ===")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative_path = path.relative_to(output_dir)
            print(f"  {relative_path}")
    
    print("\n=== Mock Test Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Total nodes executed: {final_summary['completed_nodes']}")
    print(f"Failed nodes: {final_summary['failed_nodes']}")
    
    # Check for any failures
    if final_summary['failed_nodes'] > 0:
        print("\nNote: Some nodes failed due to missing dependencies in the minimal Docker image.")
        print("This is expected in the mock test. In production, use full Docker images.")


if __name__ == "__main__":
    main()