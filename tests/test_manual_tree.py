#!/usr/bin/env python3
"""Test script for manual analysis tree construction and execution."""

import json
from pathlib import Path
from datetime import datetime
import sys

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NewFunctionBlock, 
    FunctionBlockType, StaticConfig, Arg, NodeState, GenerationMode
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor
from ragomics_agent_local.utils import setup_logger

# Set up logging
logger = setup_logger("test_manual_tree")

def create_quality_control_block():
    """Create a quality control function block."""
    config = StaticConfig(
        args=[
            Arg(
                name="min_genes",
                value_type="int",
                description="Minimum number of genes per cell",
                optional=True,
                default_value=200
            ),
            Arg(
                name="max_genes",
                value_type="int", 
                description="Maximum number of genes per cell",
                optional=True,
                default_value=5000
            ),
            Arg(
                name="max_pct_mito",
                value_type="float",
                description="Maximum mitochondrial percentage",
                optional=True,
                default_value=20.0
            )
        ],
        description="Quality control filtering for single-cell data",
        tag="qc",
        source="manual"
    )
    
    code = '''
def run(adata, min_genes=200, max_genes=5000, max_pct_mito=20.0, **kwargs):
    """Perform quality control on single-cell data."""
    import scanpy as sc
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"Initial data shape: {adata.shape}")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Add QC columns
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    adata.obs['percent_mito'] = adata.obs['pct_counts_mt']
    
    # QC plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Violin plots
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, ax=axes[0, :])
    
    # Scatter plots
    axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.3)
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Number of genes')
    
    axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.3)
    axes[1, 1].set_xlabel('Total counts')
    axes[1, 1].set_ylabel('Mitochondrial %')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/qc_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Filter cells
    print(f"Filtering cells with {min_genes} < n_genes < {max_genes} and pct_mito < {max_pct_mito}%")
    adata = adata[adata.obs['n_genes_by_counts'] > min_genes, :]
    adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
    adata = adata[adata.obs['pct_counts_mt'] < max_pct_mito, :]
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"After filtering: {adata.shape}")
    
    # Summary statistics plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    summary_df = pd.DataFrame({
        'Before': [adata.raw.n_obs if hasattr(adata, 'raw') and adata.raw else adata.shape[0], 
                   adata.raw.n_vars if hasattr(adata, 'raw') and adata.raw else adata.shape[1]],
        'After': [adata.n_obs, adata.n_vars]
    }, index=['Cells', 'Genes'])
    
    summary_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Count')
    ax.set_title('QC Filtering Summary')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/qc_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return adata
'''
    
    return NewFunctionBlock(
        name="quality_control",
        type=FunctionBlockType.PYTHON,
        description="Quality control and filtering",
        code=code,
        requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0\nseaborn>=0.12.0",
        parameters={
            "min_genes": 200,
            "max_genes": 5000,
            "max_pct_mito": 20.0
        },
        static_config=config
    )


def create_normalization_block():
    """Create a normalization function block."""
    config = StaticConfig(
        args=[
            Arg(
                name="target_sum",
                value_type="float",
                description="Target sum for normalization",
                optional=True,
                default_value=10000.0
            )
        ],
        description="Normalize and log-transform the data",
        tag="normalization",
        source="manual"
    )
    
    code = '''
def run(adata, target_sum=10000.0, **kwargs):
    """Normalize and log-transform single-cell data."""
    import scanpy as sc
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(f"Normalizing data with target_sum={target_sum}")
    
    # Store raw counts
    adata.raw = adata
    
    # Normalize every cell to target_sum
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Logarithmize the data
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Plot highly variable genes
    sc.pl.highly_variable_genes(adata)
    plt.savefig('/workspace/output/figures/highly_variable_genes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Keep only highly variable genes for downstream analysis
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    print(f"Number of highly variable genes: {adata_hvg.n_vars}")
    print(f"Data normalized and log-transformed")
    
    # Store the full data in raw if not already there
    if not hasattr(adata, 'raw') or adata.raw is None:
        adata.raw = adata
    
    return adata
'''
    
    return NewFunctionBlock(
        name="normalization",
        type=FunctionBlockType.PYTHON,
        description="Data normalization and HVG selection",
        code=code,
        requirements="scanpy>=1.9.0\nnumpy>=1.24.0\nmatplotlib>=3.6.0",
        parameters={"target_sum": 10000.0},
        static_config=config
    )


def create_velocity_analysis_block():
    """Create RNA velocity analysis block."""
    config = StaticConfig(
        args=[
            Arg(
                name="n_pcs",
                value_type="int",
                description="Number of principal components",
                optional=True,
                default_value=30
            ),
            Arg(
                name="n_neighbors", 
                value_type="int",
                description="Number of neighbors for velocity graph",
                optional=True,
                default_value=30
            )
        ],
        description="Perform RNA velocity analysis using scVelo",
        tag="velocity",
        source="manual"
    )
    
    code = '''
def run(adata, n_pcs=30, n_neighbors=30, **kwargs):
    """Perform RNA velocity analysis."""
    import scanpy as sc
    import scvelo as scv
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    scv.settings.verbosity = 3
    scv.settings.presenter_view = True
    scv.settings.set_figure_params('scvelo')
    
    print(f"Starting RNA velocity analysis")
    print(f"Data shape: {adata.shape}")
    
    # Check if spliced/unspliced layers exist
    has_velocity_data = 'spliced' in adata.layers and 'unspliced' in adata.layers
    
    if not has_velocity_data:
        print("No spliced/unspliced layers found. Creating synthetic velocity data for demonstration...")
        # Create synthetic layers for demonstration
        # In real analysis, these would come from velocyto or other tools
        adata.layers['spliced'] = adata.X.copy()
        adata.layers['unspliced'] = adata.X.copy() * np.random.uniform(0.1, 0.3, adata.shape)
    
    # Basic preprocessing if not done
    if 'highly_variable' not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Keep only highly variable genes
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Normalize and log transform if needed
    if adata.X.max() > 100:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.pp.pca(adata, n_comps=n_pcs)
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # UMAP embedding
    sc.tl.umap(adata)
    
    # Clustering for visualization
    sc.tl.leiden(adata, resolution=0.5)
    
    # Basic UMAP plot
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data', 
               save='_clusters.png', show=False)
    plt.savefig('/workspace/output/figures/umap_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Velocity preprocessing
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    # Compute velocity
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    
    # Velocity embedding
    scv.pl.velocity_embedding_stream(adata, basis='umap', 
                                     save='_velocity_stream.png', show=False)
    plt.savefig('/workspace/output/figures/velocity_stream.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Velocity grid
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120,
                              save='_velocity_grid.png', show=False)
    plt.savefig('/workspace/output/figures/velocity_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute velocity confidence
    scv.tl.velocity_confidence(adata)
    
    # Plot velocity confidence
    scv.pl.scatter(adata, c='velocity_confidence', cmap='coolwarm', 
                   save='_velocity_confidence.png', show=False)
    plt.savefig('/workspace/output/figures/velocity_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute velocity pseudotime
    scv.tl.velocity_pseudotime(adata)
    
    # Plot pseudotime
    scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot', 
                   save='_pseudotime.png', show=False)
    plt.savefig('/workspace/output/figures/velocity_pseudotime.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    summary = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "has_velocity": True,
        "mean_velocity_confidence": float(np.mean(adata.obs['velocity_confidence'])),
        "velocity_genes": int(np.sum(adata.var['velocity_genes'])) if 'velocity_genes' in adata.var else 0
    }
    
    print(f"Velocity analysis completed")
    print(f"Mean velocity confidence: {summary['mean_velocity_confidence']:.3f}")
    
    return {"adata": adata, "summary": summary}
'''
    
    return NewFunctionBlock(
        name="rna_velocity_analysis",
        type=FunctionBlockType.PYTHON,
        description="RNA velocity analysis with scVelo",
        code=code,
        requirements="scanpy>=1.9.0\nscvelo>=0.2.5\nnumpy>=1.24.0\npandas>=2.0.0\nmatplotlib>=3.6.0",
        parameters={"n_pcs": 30, "n_neighbors": 30},
        static_config=config
    )


def main():
    """Test manual tree construction and execution."""
    
    # Paths
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    output_dir = Path("test_outputs/manual_tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    logger.info("Creating analysis tree...")
    tree = tree_manager.create_tree(
        user_request="Manual test: QC -> Normalization -> Velocity Analysis",
        input_data_path=str(input_data),
        max_nodes=10,
        max_children_per_node=3,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create function blocks
    qc_block = create_quality_control_block()
    norm_block = create_normalization_block()
    velocity_block = create_velocity_analysis_block()
    
    # Build tree structure
    logger.info("Building tree structure...")
    
    # Add QC as root
    qc_node = tree_manager.add_root_node(qc_block)
    logger.info(f"Added root node: {qc_node.function_block.name}")
    
    # Add normalization as child of QC
    norm_nodes = tree_manager.add_child_nodes(qc_node.id, [norm_block])
    norm_node = norm_nodes[0]
    logger.info(f"Added child node: {norm_node.function_block.name}")
    
    # Add velocity analysis as child of normalization
    velocity_nodes = tree_manager.add_child_nodes(norm_node.id, [velocity_block])
    velocity_node = velocity_nodes[0]
    logger.info(f"Added child node: {velocity_node.function_block.name}")
    
    # Save tree structure
    tree_json_path = output_dir / "analysis_tree.json"
    tree_manager.save_tree(tree_json_path)
    logger.info(f"Saved tree structure to {tree_json_path}")
    
    # Print tree structure
    print("\n=== Analysis Tree Structure ===")
    print(json.dumps({
        "tree_id": tree.id,
        "user_request": tree.user_request,
        "total_nodes": tree.total_nodes,
        "nodes": [
            {
                "id": node.id,
                "name": node.function_block.name,
                "type": node.function_block.type,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": node.children
            }
            for node in tree.nodes.values()
        ]
    }, indent=2))
    
    # Execute tree
    print("\n=== Executing Analysis Tree ===")
    current_data_path = input_data
    
    for node in tree_manager.get_execution_order():
        print(f"\nExecuting: {node.function_block.name}")
        
        # Update state
        tree_manager.update_node_execution(node.id, NodeState.RUNNING)
        
        # Execute node
        state, result = node_executor.execute_node(
            node=node,
            input_data_path=current_data_path,
            output_base_dir=output_dir
        )
        
        # Update tree with results
        if state == NodeState.COMPLETED:
            tree_manager.update_node_execution(
                node.id,
                state=state,
                output_data_id=result.output_data_path,
                figures=result.figures,
                logs=[result.logs] if result.logs else [],
                duration=result.duration
            )
            
            if result.output_data_path:
                current_data_path = Path(result.output_data_path)
                
            print(f"✓ Completed in {result.duration:.1f}s")
            if result.figures:
                print(f"  Generated {len(result.figures)} figures")
        else:
            tree_manager.update_node_execution(
                node.id,
                state=state,
                error=result.error,
                logs=[result.logs] if result.logs else []
            )
            print(f"✗ Failed: {result.error}")
            break
    
    # Save final tree state
    tree_manager.save_tree(tree_json_path)
    
    # Print summary
    summary = tree_manager.get_summary()
    print("\n=== Execution Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    
    # List output files
    print("\n=== Output Files ===")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()