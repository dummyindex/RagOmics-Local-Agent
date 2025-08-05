#!/usr/bin/env python3
"""Manual tests for scvelo RNA velocity analysis with different parameter configurations."""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor


def create_preprocessing_block():
    """Create preprocessing function block for scvelo."""
    config = StaticConfig(
        args=[
            Arg(name="min_shared_counts", value_type="int", 
                description="Minimum shared counts for gene filtering",
                optional=True, default_value=20),
            Arg(name="n_top_genes", value_type="int",
                description="Number of top variable genes to keep",
                optional=True, default_value=2000),
            Arg(name="n_pcs", value_type="int",
                description="Number of principal components for moments",
                optional=True, default_value=30),
            Arg(name="n_neighbors", value_type="int",
                description="Number of neighbors for moments",
                optional=True, default_value=30)
        ],
        description="Preprocess data for RNA velocity analysis",
        tag="preprocessing",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Preprocess data for RNA velocity analysis."""
    import scanpy as sc
    import scvelo as scv
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Starting preprocessing for RNA velocity...")
    print(f"Input shape: {adata.shape}")
    
    # Basic info about the data
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")
    
    # Make observation names unique
    adata.obs_names_make_unique()
    
    # Check if spliced/unspliced layers exist
    if 'spliced' not in adata.layers or 'unspliced' not in adata.layers:
        print("WARNING: No spliced/unspliced layers found!")
        print("Creating mock layers for testing...")
        # Create mock spliced/unspliced for testing
        # First store original X
        X_orig = adata.X.copy()
        if hasattr(X_orig, 'toarray'):
            X_orig = X_orig.toarray()
        
        # Create realistic mock layers
        adata.layers['spliced'] = X_orig * 0.7  # 70% spliced
        adata.layers['unspliced'] = X_orig * 0.3  # 30% unspliced
        
    # Store raw counts before normalization
    adata.raw = adata.copy()
        
    # Filter and normalize using scvelo's method
    print(f"Filtering genes (min_shared_counts={min_shared_counts}) and selecting top {n_top_genes} genes...")
    
    # Filter genes manually first
    scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    
    # Get current gene count
    n_genes_before = adata.n_vars
    
    # Normalize and find highly variable genes
    scv.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    
    # Log transform only X, not layers
    adata.X = np.log1p(adata.X)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, n_genes_before), 
                                flavor='seurat', batch_key=None, subset=False)
    
    # Subset to highly variable genes
    adata = adata[:, adata.var['highly_variable']]
    
    print(f"After filtering: {adata.shape}")
    
    # PCA if not already computed
    if 'X_pca' not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, svd_solver='arpack')
    
    # Compute neighbors if not already done
    if 'neighbors' not in adata.uns:
        print(f"Computing neighbors (n_neighbors={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)
    
    # Compute UMAP if not already done
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP embedding...")
        sc.tl.umap(adata)
    
    # Compute moments for velocity estimation
    print(f"Computing moments (n_pcs={n_pcs}, n_neighbors={n_neighbors})...")
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    # Basic clustering if not present
    if 'leiden' not in adata.obs.columns:
        print("Computing Leiden clustering...")
        sc.tl.leiden(adata)
    
    # Plot UMAP with clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata, color='leiden', legend_loc='on data', ax=ax, show=False)
    plt.title('Leiden Clustering')
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/preprocessing_umap_clusters.png', dpi=150)
    plt.close()
    
    print("Preprocessing completed!")
    print(f"Output shape: {adata.shape}")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="scvelo_preprocessing",
        type=FunctionBlockType.PYTHON,
        description="Preprocess data for RNA velocity",
        code=code,
        requirements="scanpy>=1.9.0\nscvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0\npython-igraph>=0.10.0\nleidenalg>=0.9.0",
        parameters={
            "min_shared_counts": 20,
            "n_top_genes": 2000,
            "n_pcs": 30,
            "n_neighbors": 30
        },
        static_config=config
    )


def create_velocity_steady_state_block():
    """Create velocity computation block using steady-state model."""
    config = StaticConfig(
        args=[
            Arg(name="vkey", value_type="str",
                description="Key to store velocity in adata",
                optional=True, default_value="velocity"),
            Arg(name="n_jobs", value_type="int",
                description="Number of parallel jobs",
                optional=True, default_value=4)
        ],
        description="Compute RNA velocity using steady-state model",
        tag="velocity_computation",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Compute RNA velocity using steady-state model."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Computing RNA velocity using steady-state model...")
    
    # Compute velocity
    scv.tl.velocity(adata, mode='steady_state', vkey=vkey, n_jobs=n_jobs)
    print(f"Velocity computed and stored in '{vkey}'")
    
    # Compute velocity graph
    print("Computing velocity graph...")
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
    
    # Compute velocity confidence
    print("Computing velocity confidence...")
    scv.tl.velocity_confidence(adata, vkey=vkey)
    
    # Visualize velocity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stream plot
    scv.pl.velocity_embedding_stream(adata, basis='umap', vkey=vkey, 
                                    ax=axes[0], show=False, 
                                    title='Velocity Stream (Steady-State)')
    
    # Arrow plot
    scv.pl.velocity_embedding(adata, basis='umap', vkey=vkey,
                            arrow_length=3, arrow_size=2,
                            ax=axes[1], show=False,
                            title='Velocity Arrows (Steady-State)')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_steady_state.png', dpi=150)
    plt.close()
    
    # Plot velocity confidence
    fig, ax = plt.subplots(figsize=(8, 6))
    scv.pl.scatter(adata, color='velocity_confidence', cmap='coolwarm',
                   ax=ax, show=False)
    plt.title('Velocity Confidence (Steady-State)')
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_confidence_steady_state.png', dpi=150)
    plt.close()
    
    # Summary statistics
    velocities = adata.layers[vkey]
    print(f"Velocity statistics:")
    print(f"  Mean velocity magnitude: {np.mean(np.abs(velocities)):.4f}")
    print(f"  Max velocity magnitude: {np.max(np.abs(velocities)):.4f}")
    print(f"  Velocity confidence mean: {adata.obs['velocity_confidence'].mean():.4f}")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="velocity_steady_state",
        type=FunctionBlockType.PYTHON,
        description="RNA velocity with steady-state model",
        code=code,
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0",
        parameters={"vkey": "velocity", "n_jobs": 4},
        static_config=config
    )


def create_velocity_stochastic_block():
    """Create velocity computation block using stochastic model."""
    config = StaticConfig(
        args=[
            Arg(name="vkey", value_type="str",
                description="Key to store velocity in adata",
                optional=True, default_value="velocity_stochastic"),
            Arg(name="n_jobs", value_type="int",
                description="Number of parallel jobs",
                optional=True, default_value=4),
            Arg(name="perc", value_type="int",
                description="Percentile for velocity estimation",
                optional=True, default_value=95)
        ],
        description="Compute RNA velocity using stochastic model",
        tag="velocity_computation",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Compute RNA velocity using stochastic model."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Computing RNA velocity using stochastic model...")
    print(f"Parameters: perc={perc}")
    
    # Compute velocity with stochastic model
    scv.tl.velocity(adata, mode='stochastic', vkey=vkey, n_jobs=n_jobs, perc=perc)
    print(f"Velocity computed and stored in '{vkey}'")
    
    # Compute velocity graph
    print("Computing velocity graph...")
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
    
    # Compute velocity confidence
    print("Computing velocity confidence...")
    scv.tl.velocity_confidence(adata, vkey=vkey)
    
    # Visualize velocity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stream plot
    scv.pl.velocity_embedding_stream(adata, basis='umap', vkey=vkey,
                                    ax=axes[0], show=False,
                                    title='Velocity Stream (Stochastic)')
    
    # Grid plot
    scv.pl.velocity_embedding_grid(adata, basis='umap', vkey=vkey,
                                  ax=axes[1], show=False,
                                  title='Velocity Grid (Stochastic)')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_stochastic.png', dpi=150)
    plt.close()
    
    # Compare with steady-state if available
    if 'velocity' in adata.layers:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate correlation between models
        steady_v = adata.layers['velocity'].flatten()
        stoch_v = adata.layers[vkey].flatten()
        corr = np.corrcoef(steady_v, stoch_v)[0, 1]
        
        ax.scatter(steady_v, stoch_v, alpha=0.1, s=1)
        ax.set_xlabel('Steady-state velocity')
        ax.set_ylabel('Stochastic velocity')
        ax.set_title(f'Model Comparison (correlation={corr:.3f})')
        
        # Add diagonal line
        lims = [ax.get_xlim(), ax.get_ylim()]
        lims = [min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1])]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('/workspace/output/figures/velocity_model_comparison.png', dpi=150)
        plt.close()
    
    # Summary statistics
    velocities = adata.layers[vkey]
    print(f"Velocity statistics (stochastic):")
    print(f"  Mean velocity magnitude: {np.mean(np.abs(velocities)):.4f}")
    print(f"  Max velocity magnitude: {np.max(np.abs(velocities)):.4f}")
    print(f"  Velocity confidence mean: {adata.obs['velocity_confidence'].mean():.4f}")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="velocity_stochastic",
        type=FunctionBlockType.PYTHON,
        description="RNA velocity with stochastic model",
        code=code,
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0",
        parameters={"vkey": "velocity_stochastic", "n_jobs": 4, "perc": 95},
        static_config=config
    )


def create_velocity_dynamical_block():
    """Create velocity computation block using dynamical model."""
    config = StaticConfig(
        args=[
            Arg(name="vkey", value_type="str",
                description="Key to store velocity in adata",
                optional=True, default_value="velocity_dynamical"),
            Arg(name="n_jobs", value_type="int",
                description="Number of parallel jobs",
                optional=True, default_value=4),
            Arg(name="n_top_genes", value_type="int",
                description="Number of top likelihood genes to use",
                optional=True, default_value=2000)
        ],
        description="Compute RNA velocity using dynamical model",
        tag="velocity_computation",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Compute RNA velocity using dynamical model."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Computing RNA velocity using dynamical model...")
    print("This may take a few minutes...")
    
    # Recover dynamics
    print("Recovering dynamics...")
    scv.tl.recover_dynamics(adata, n_jobs=n_jobs)
    
    # Filter genes by likelihood
    if 'fit_likelihood' in adata.var.columns:
        top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:n_top_genes]
        print(f"Using top {len(top_genes)} genes by likelihood")
    else:
        top_genes = adata.var_names
    
    # Compute velocity with dynamical model
    scv.tl.velocity(adata, mode='dynamical', vkey=vkey, n_jobs=n_jobs)
    print(f"Velocity computed and stored in '{vkey}'")
    
    # Compute velocity graph
    print("Computing velocity graph...")
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
    
    # Compute latent time
    print("Computing latent time...")
    scv.tl.latent_time(adata, vkey=vkey)
    
    # Visualize velocity and dynamics
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Stream plot
    scv.pl.velocity_embedding_stream(adata, basis='umap', vkey=vkey,
                                    ax=axes[0, 0], show=False,
                                    title='Velocity Stream (Dynamical)')
    
    # Latent time
    scv.pl.scatter(adata, color='latent_time', color_map='gnuplot',
                   ax=axes[0, 1], show=False,
                   title='Latent Time')
    
    # Phase portraits for top genes
    if 'fit_likelihood' in adata.var.columns:
        top_genes_plot = adata.var['fit_likelihood'].sort_values(ascending=False).index[:2]
        for i, gene in enumerate(top_genes_plot):
            if gene in adata.var_names:
                scv.pl.velocity(adata, var_names=[gene], basis='umap',
                              ax=axes[1, i], show=False)
                axes[1, i].set_title(f'Phase Portrait: {gene}')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_dynamical.png', dpi=150)
    plt.close()
    
    # Plot kinetic parameters
    if 'fit_alpha' in adata.var.columns and 'fit_beta' in adata.var.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Transcription rates
        axes[0].hist(adata.var['fit_alpha'], bins=50, alpha=0.7)
        axes[0].set_xlabel('Transcription rate (α)')
        axes[0].set_ylabel('Number of genes')
        axes[0].set_title('Distribution of Transcription Rates')
        
        # Splicing rates
        axes[1].hist(adata.var['fit_beta'], bins=50, alpha=0.7, color='orange')
        axes[1].set_xlabel('Splicing rate (β)')
        axes[1].set_ylabel('Number of genes')
        axes[1].set_title('Distribution of Splicing Rates')
        
        plt.tight_layout()
        plt.savefig('/workspace/output/figures/kinetic_parameters.png', dpi=150)
        plt.close()
    
    # Summary statistics
    velocities = adata.layers[vkey]
    print(f"Velocity statistics (dynamical):")
    print(f"  Mean velocity magnitude: {np.mean(np.abs(velocities)):.4f}")
    print(f"  Max velocity magnitude: {np.max(np.abs(velocities)):.4f}")
    if 'latent_time' in adata.obs.columns:
        print(f"  Latent time range: [{adata.obs['latent_time'].min():.3f}, {adata.obs['latent_time'].max():.3f}]")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="velocity_dynamical",
        type=FunctionBlockType.PYTHON,
        description="RNA velocity with dynamical model",
        code=code,
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0",
        parameters={"vkey": "velocity_dynamical", "n_jobs": 4, "n_top_genes": 2000},
        static_config=config
    )


def create_velocity_analysis_block():
    """Create comprehensive velocity analysis and visualization block."""
    config = StaticConfig(
        args=[
            Arg(name="n_genes", value_type="int",
                description="Number of top genes to analyze",
                optional=True, default_value=10),
            Arg(name="groups", value_type="str",
                description="Groups to analyze (e.g., clusters)",
                optional=True, default_value="leiden")
        ],
        description="Comprehensive velocity analysis and visualization",
        tag="velocity_analysis",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Perform comprehensive velocity analysis."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Performing comprehensive velocity analysis...")
    
    # Determine which velocity to use
    vkey = 'velocity_dynamical' if 'velocity_dynamical' in adata.layers else 'velocity'
    print(f"Using velocity key: {vkey}")
    
    # 1. Velocity pseudotime
    print("Computing velocity pseudotime...")
    scv.tl.velocity_pseudotime(adata, vkey=vkey)
    
    # 2. PAGA velocity graph
    if groups in adata.obs.columns:
        print(f"Computing PAGA velocity graph for {groups}...")
        try:
            scv.tl.paga(adata, groups=groups, vkey=vkey)
        except ModuleNotFoundError:
            print("PAGA requires igraph, skipping...")
    
    # 3. Rank velocity genes
    print("Ranking velocity genes...")
    scv.tl.rank_velocity_genes(adata, groupby=groups, n_genes=n_genes, vkey=vkey)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Velocity pseudotime
    ax1 = plt.subplot(3, 3, 1)
    scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot',
                   ax=ax1, show=False)
    ax1.set_title('Velocity Pseudotime')
    
    # 2. Velocity length
    ax2 = plt.subplot(3, 3, 2)
    scv.pl.scatter(adata, color='velocity_length', cmap='coolwarm',
                   ax=ax2, show=False)
    ax2.set_title('Velocity Length')
    
    # 3. Velocity confidence
    ax3 = plt.subplot(3, 3, 3)
    scv.pl.scatter(adata, color='velocity_confidence', cmap='RdYlBu_r',
                   ax=ax3, show=False)
    ax3.set_title('Velocity Confidence')
    
    # 4. PAGA graph if computed
    if 'paga' in adata.uns:
        ax4 = plt.subplot(3, 3, 4)
        try:
            scv.pl.paga(adata, ax=ax4, show=False)
            ax4.set_title(f'PAGA Graph ({groups})')
        except:
            ax4.text(0.5, 0.5, 'PAGA not available', ha='center', va='center')
            ax4.set_xticks([])
            ax4.set_yticks([])
    
    # 5. Velocity embedding with different parameters
    ax5 = plt.subplot(3, 3, 5)
    scv.pl.velocity_embedding_stream(adata, basis='umap', vkey=vkey,
                                    color=groups, legend_loc='right margin',
                                    ax=ax5, show=False)
    ax5.set_title('Velocity Stream by Groups')
    
    # 6. Grid velocity
    ax6 = plt.subplot(3, 3, 6)
    scv.pl.velocity_embedding_grid(adata, basis='umap', vkey=vkey,
                                  color=groups, ax=ax6, show=False)
    ax6.set_title('Velocity Grid')
    
    # 7. Top velocity genes heatmap
    if 'rank_velocity_genes' in adata.uns:
        ax7 = plt.subplot(3, 1, 3)
        
        # Get top genes per cluster
        top_genes = []
        for cluster in adata.obs[groups].unique():
            cluster_genes = adata.uns['rank_velocity_genes']['names'][cluster][:5]
            top_genes.extend(cluster_genes)
        top_genes = list(set(top_genes))[:20]  # Limit to 20 unique genes
        
        if len(top_genes) > 0 and 'latent_time' in adata.obs.columns:
            # Sort cells by latent time
            time_order = adata.obs.sort_values('latent_time').index
            adata_ordered = adata[time_order]
            
            # Create expression matrix
            expr_matrix = adata_ordered[:, top_genes].X
            if hasattr(expr_matrix, 'toarray'):
                expr_matrix = expr_matrix.toarray()
            
            im = ax7.imshow(expr_matrix.T, aspect='auto', cmap='RdBu_r',
                           vmin=-2, vmax=2)
            ax7.set_yticks(range(len(top_genes)))
            ax7.set_yticklabels(top_genes)
            ax7.set_xlabel('Cells (ordered by latent time)')
            ax7.set_title('Top Velocity Genes Expression')
            plt.colorbar(im, ax=ax7, label='Expression (z-score)')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_comprehensive_analysis.png', dpi=150)
    plt.close()
    
    # Save velocity statistics
    stats = {
        'velocity_key': vkey,
        'mean_velocity_length': float(adata.obs['velocity_length'].mean()),
        'mean_velocity_confidence': float(adata.obs['velocity_confidence'].mean()),
        'mean_velocity_pseudotime': float(adata.obs['velocity_pseudotime'].mean()),
    }
    
    if 'latent_time' in adata.obs.columns:
        stats['latent_time_range'] = [
            float(adata.obs['latent_time'].min()),
            float(adata.obs['latent_time'].max())
        ]
    
    # Save top velocity genes
    if 'rank_velocity_genes' in adata.uns:
        top_genes_dict = {}
        for cluster in adata.obs[groups].unique():
            genes = adata.uns['rank_velocity_genes']['names'][cluster][:5].tolist()
            scores = adata.uns['rank_velocity_genes']['scores'][cluster][:5].tolist()
            top_genes_dict[str(cluster)] = {
                'genes': genes,
                'scores': [float(s) for s in scores]
            }
        stats['top_velocity_genes'] = top_genes_dict
    
    import json
    with open('/workspace/output/velocity_analysis_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Comprehensive velocity analysis completed!")
    print(f"Results saved to figures and velocity_analysis_stats.json")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="velocity_comprehensive_analysis",
        type=FunctionBlockType.PYTHON,
        description="Comprehensive velocity analysis",
        code=code,
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0\npandas>=2.0.0",
        parameters={"n_genes": 10, "groups": "leiden"},
        static_config=config
    )


def create_driver_genes_block():
    """Create driver genes identification block."""
    config = StaticConfig(
        args=[
            Arg(name="n_top_genes", value_type="int",
                description="Number of top driver genes to identify",
                optional=True, default_value=20),
            Arg(name="min_likelihood", value_type="float",
                description="Minimum likelihood threshold for genes",
                optional=True, default_value=0.1)
        ],
        description="Identify driver genes and plot phase portraits",
        tag="driver_genes",
        source="scvelo_tutorial"
    )
    
    code = '''
def run(path_dict, params):
    """Identify and visualize driver genes."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Identifying driver genes...")
    
    # Determine velocity key
    vkey = 'velocity_dynamical' if 'velocity_dynamical' in adata.layers else 'velocity'
    
    # Identify driver genes
    if 'fit_likelihood' in adata.var.columns:
        # Use dynamical model likelihood
        top_genes = adata.var.query(f'fit_likelihood > {min_likelihood}')
        top_genes = top_genes.sort_values('fit_likelihood', ascending=False)
        driver_genes = top_genes.index[:n_top_genes].tolist()
        print(f"Found {len(driver_genes)} driver genes based on dynamical model")
    else:
        # Use velocity genes
        scv.tl.rank_velocity_genes(adata, groupby='leiden', n_genes=n_top_genes)
        # Collect unique top genes from all groups
        driver_genes = []
        for group in adata.obs['leiden'].unique():
            genes = adata.uns['rank_velocity_genes']['names'][group][:5]
            driver_genes.extend(genes)
        driver_genes = list(set(driver_genes))[:n_top_genes]
        print(f"Found {len(driver_genes)} driver genes based on velocity ranking")
    
    # Create phase portraits for top driver genes
    n_genes_plot = min(12, len(driver_genes))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, gene in enumerate(driver_genes[:n_genes_plot]):
        try:
            scv.pl.velocity(adata, var_names=[gene], 
                          colorbar=False, ax=axes[i], show=False)
            axes[i].set_title(f'{gene}', fontsize=10)
        except:
            axes[i].text(0.5, 0.5, f'{gene}\\n(Error)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_genes_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Phase Portraits of Top Driver Genes', fontsize=14)
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/driver_genes_phase_portraits.png', dpi=150)
    plt.close()
    
    # Create scatter plots for top driver genes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, gene in enumerate(driver_genes[:6]):
        if gene in adata.var_names:
            scv.pl.scatter(adata, var_names=[gene], color_map='RdBu_r',
                         ax=axes[i], show=False)
            axes[i].set_title(f'{gene} Expression')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/driver_genes_expression.png', dpi=150)
    plt.close()
    
    # Heatmap of driver genes over pseudotime
    if 'latent_time' in adata.obs.columns and len(driver_genes) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort cells by latent time
        time_order = adata.obs.sort_values('latent_time').index
        adata_ordered = adata[time_order]
        
        # Get expression matrix
        expr_matrix = adata_ordered[:, driver_genes].X
        if hasattr(expr_matrix, 'toarray'):
            expr_matrix = expr_matrix.toarray()
        
        # Smooth expression for visualization
        from scipy.ndimage import gaussian_filter1d
        expr_smooth = gaussian_filter1d(expr_matrix, sigma=10, axis=0)
        
        # Z-score normalize
        expr_norm = (expr_smooth - expr_smooth.mean(axis=0)) / (expr_smooth.std(axis=0) + 1e-8)
        
        im = ax.imshow(expr_norm.T, aspect='auto', cmap='RdBu_r',
                      vmin=-2, vmax=2, interpolation='bilinear')
        ax.set_yticks(range(len(driver_genes)))
        ax.set_yticklabels(driver_genes, fontsize=8)
        ax.set_xlabel('Cells (ordered by latent time)')
        ax.set_title('Driver Genes Expression over Latent Time')
        plt.colorbar(im, ax=ax, label='Expression (z-score)')
        
        plt.tight_layout()
        plt.savefig('/workspace/output/figures/driver_genes_heatmap.png', dpi=150)
        plt.close()
    
    # Save driver genes information
    driver_info = []
    for gene in driver_genes:
        info = {'gene': gene}
        if 'fit_likelihood' in adata.var.columns:
            info['likelihood'] = float(adata.var.loc[gene, 'fit_likelihood'])
        if 'fit_alpha' in adata.var.columns:
            info['alpha'] = float(adata.var.loc[gene, 'fit_alpha'])
        if 'fit_beta' in adata.var.columns:
            info['beta'] = float(adata.var.loc[gene, 'fit_beta'])
        if 'fit_gamma' in adata.var.columns:
            info['gamma'] = float(adata.var.loc[gene, 'fit_gamma'])
        driver_info.append(info)
    
    df_drivers = pd.DataFrame(driver_info)
    df_drivers.to_csv('/workspace/output/driver_genes.csv', index=False)
    
    print(f"Identified {len(driver_genes)} driver genes")
    print(f"Results saved to driver_genes.csv and figures")
    
    return adata
'''
    
    return NewFunctionBlock(
        name="driver_genes_identification",
        type=FunctionBlockType.PYTHON,
        description="Identify driver genes",
        code=code,
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0\npandas>=2.0.0\nscipy>=1.10.0",
        parameters={"n_top_genes": 20, "min_likelihood": 0.1},
        static_config=config
    )


def test_scvelo_tree_1_steady_state(input_data_path: Path, output_dir: Path):
    """Test tree 1: Basic steady-state velocity analysis."""
    print("\n=== Test Tree 1: Steady-State Velocity Analysis ===")
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Compute RNA velocity using steady-state model",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create nodes
    preprocessing_block = create_preprocessing_block()
    velocity_block = create_velocity_steady_state_block()
    analysis_block = create_velocity_analysis_block()
    
    # Build tree structure
    preprocessing_node = tree_manager.add_root_node(preprocessing_block)
    
    # Execute preprocessing
    print("\n1. Executing preprocessing...")
    tree_manager.update_node_execution(preprocessing_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=preprocessing_node, tree=tree, input_path=input_data_path, output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            preprocessing_node.id,
            state=state,
            output_data_id=result.output_data_path,
            duration=result.duration
        )
        print("   ✓ Preprocessing completed")
    else:
        print(f"   ✗ Preprocessing failed: {result.error}")
        return False
    
    # Add velocity computation
    velocity_nodes = tree_manager.add_child_nodes(preprocessing_node.id, [velocity_block])
    velocity_node = velocity_nodes[0]
    
    # Execute velocity computation
    print("\n2. Computing steady-state velocity...")
    tree_manager.update_node_execution(velocity_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=velocity_node, tree=tree, input_path=Path(preprocessing_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            velocity_node.id,
            state=state,
            output_data_id=result.output_data_path,
            duration=result.duration
        )
        print("   ✓ Velocity computation completed")
    else:
        print(f"   ✗ Velocity computation failed: {result.error}")
        return False
    
    # Add analysis
    analysis_nodes = tree_manager.add_child_nodes(velocity_node.id, [analysis_block])
    analysis_node = analysis_nodes[0]
    
    # Execute analysis
    print("\n3. Performing velocity analysis...")
    tree_manager.update_node_execution(analysis_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=analysis_node, tree=tree, input_path=Path(velocity_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            analysis_node.id,
            state=state,
            output_data_id=result.output_data_path,
            duration=result.duration
        )
        print("   ✓ Analysis completed")
    else:
        print(f"   ✗ Analysis failed: {result.error}")
    
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
        figures_dir = tree_output_dir / tree.id / analysis_node.id / "figures"
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


def test_scvelo_tree_2_model_comparison(input_data_path: Path, output_dir: Path):
    """Test tree 2: Compare all three velocity models."""
    print("\n=== Test Tree 2: Model Comparison (Steady-State vs Stochastic vs Dynamical) ===")
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Compare RNA velocity models: steady-state, stochastic, and dynamical",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create nodes
    preprocessing_block = create_preprocessing_block()
    preprocessing_block.parameters = {
        "min_shared_counts": 30,  # More stringent filtering
        "n_top_genes": 3000,      # More genes
        "n_pcs": 40,              # More PCs
        "n_neighbors": 40         # More neighbors
    }
    
    steady_block = create_velocity_steady_state_block()
    stochastic_block = create_velocity_stochastic_block()
    dynamical_block = create_velocity_dynamical_block()
    
    # Build tree structure
    preprocessing_node = tree_manager.add_root_node(preprocessing_block)
    
    # Execute preprocessing
    print("\n1. Executing preprocessing with custom parameters...")
    tree_manager.update_node_execution(preprocessing_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=preprocessing_node, tree=tree, input_path=input_data_path, output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"   ✗ Preprocessing failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        preprocessing_node.id,
        state=state,
        output_data_id=result.output_data_path,
        duration=result.duration
    )
    print("   ✓ Preprocessing completed")
    
    # Add all three velocity models as parallel branches
    velocity_blocks = [steady_block, stochastic_block, dynamical_block]
    velocity_nodes = tree_manager.add_child_nodes(preprocessing_node.id, velocity_blocks)
    
    # Execute each velocity model
    for i, (node, name) in enumerate(zip(velocity_nodes, ['steady-state', 'stochastic', 'dynamical'])):
        print(f"\n{i+2}. Computing {name} velocity...")
        tree_manager.update_node_execution(node.id, NodeState.RUNNING)
        state, result = node_executor.execute_node(node=node, tree=tree, input_path=Path(preprocessing_node.output_data_id), output_base_dir=output_dir / tree.id
        )
        
        if state == NodeState.COMPLETED:
            tree_manager.update_node_execution(
                node.id,
                state=state,
                output_data_id=result.output_data_path,
                duration=result.duration
            )
            print(f"   ✓ {name} velocity completed in {result.duration:.1f}s")
        else:
            print(f"   ✗ {name} velocity failed: {result.error}")
    
    # Add comprehensive analysis to dynamical model branch
    if velocity_nodes[2].state == NodeState.COMPLETED:
        analysis_block = create_velocity_analysis_block()
        analysis_nodes = tree_manager.add_child_nodes(velocity_nodes[2].id, [analysis_block])
        analysis_node = analysis_nodes[0]
        
        print("\n5. Performing comprehensive analysis on dynamical model...")
        tree_manager.update_node_execution(analysis_node.id, NodeState.RUNNING)
        state, result = node_executor.execute_node(node=analysis_node, tree=tree, input_path=Path(velocity_nodes[2].output_data_id), output_base_dir=output_dir / tree.id
        )
        
        if state == NodeState.COMPLETED:
            tree_manager.update_node_execution(
                analysis_node.id,
                state=state,
                output_data_id=result.output_data_path,
                duration=result.duration
            )
            print("   ✓ Analysis completed")
    
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
    success = summary['completed_nodes'] > 0
    if success:
        # Check that output files exist
        tree_output_dir = output_dir / tree.id
        tree_json = tree_output_dir / "analysis_tree.json"
        
        if not tree_json.exists():
            print(f"   ✗ Missing analysis_tree.json")
            success = False
        
        # Check for figures from multiple model nodes
        figures_found = 0
        for node_id in tree_manager.tree.nodes:
            figures_dir = tree_output_dir / tree.id / node_id / "figures"
            if figures_dir.exists():
                figures = list(figures_dir.glob("*.png"))
                figures_found += len(figures)
        
        print(f"\n   Total figures generated: {figures_found}")
        if figures_found == 0:
            print(f"   ✗ No figures found in any node")
            success = False
    
    return success


def test_scvelo_tree_3_driver_genes(input_data_path: Path, output_dir: Path):
    """Test tree 3: Dynamical model with driver gene analysis."""
    print("\n=== Test Tree 3: Dynamical Model with Driver Gene Analysis ===")
    
    # Create tree manager and executor
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create analysis tree
    tree = tree_manager.create_tree(
        user_request="Identify driver genes using dynamical velocity model",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create nodes with specific parameters
    preprocessing_block = create_preprocessing_block()
    preprocessing_block.parameters = {
        "min_shared_counts": 20,
        "n_top_genes": 2000,
        "n_pcs": 30,
        "n_neighbors": 30
    }
    
    dynamical_block = create_velocity_dynamical_block()
    dynamical_block.parameters = {
        "vkey": "velocity_dynamical",
        "n_jobs": 4,
        "n_top_genes": 1000  # Focus on top genes
    }
    
    driver_block = create_driver_genes_block()
    driver_block.parameters = {
        "n_top_genes": 30,    # More driver genes
        "min_likelihood": 0.05  # Lower threshold
    }
    
    # Build tree structure
    preprocessing_node = tree_manager.add_root_node(preprocessing_block)
    
    # Execute preprocessing
    print("\n1. Executing preprocessing...")
    tree_manager.update_node_execution(preprocessing_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=preprocessing_node, tree=tree, input_path=input_data_path, output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"   ✗ Preprocessing failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        preprocessing_node.id,
        state=state,
        output_data_id=result.output_data_path,
        duration=result.duration
    )
    print("   ✓ Preprocessing completed")
    
    # Add dynamical velocity
    velocity_nodes = tree_manager.add_child_nodes(preprocessing_node.id, [dynamical_block])
    velocity_node = velocity_nodes[0]
    
    # Execute velocity computation
    print("\n2. Computing dynamical velocity...")
    tree_manager.update_node_execution(velocity_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=velocity_node, tree=tree, input_path=Path(preprocessing_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"   ✗ Velocity computation failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        velocity_node.id,
        state=state,
        output_data_id=result.output_data_path,
        duration=result.duration
    )
    print(f"   ✓ Velocity computation completed in {result.duration:.1f}s")
    
    # Add driver gene analysis
    driver_nodes = tree_manager.add_child_nodes(velocity_node.id, [driver_block])
    driver_node = driver_nodes[0]
    
    # Execute driver gene analysis
    print("\n3. Identifying driver genes...")
    tree_manager.update_node_execution(driver_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=driver_node, tree=tree, input_path=Path(velocity_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            driver_node.id,
            state=state,
            output_data_id=result.output_data_path,
            duration=result.duration
        )
        print("   ✓ Driver gene analysis completed")
    else:
        print(f"   ✗ Driver gene analysis failed: {result.error}")
    
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
    success = summary['completed_nodes'] == summary['total_nodes']
    if success:
        # Check that output files exist
        tree_output_dir = output_dir / tree.id
        tree_json = tree_output_dir / "analysis_tree.json"
        
        if not tree_json.exists():
            print(f"   ✗ Missing analysis_tree.json")
            success = False
        
        # Check for driver genes specific outputs
        driver_node = driver_nodes[0]
        figures_dir = tree_output_dir / tree.id / driver_node.id / "figures"
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png"))
            print(f"\n   Driver genes analysis generated {len(figures)} figures:")
            for fig in figures:
                print(f"     - {fig.name}")
        else:
            print(f"   ✗ No driver genes figures found")
            success = False
    
    return success


def test_scvelo_tree_4_multi_branch(input_data_path: Path, output_dir: Path):
    """Test tree with multiple branches for different analysis paths."""
    print("\n=== Test Tree 4: Multi-Branch Analysis Tree ===")
    print("This tree demonstrates branching analysis paths from a single preprocessing step.")
    
    # Create analysis tree
    tree_manager = AnalysisTreeManager()
    tree = tree_manager.create_tree(
        user_request="Multi-branch scVelo analysis with different velocity models and downstream analyses",
        input_data_path=str(input_data_path),
        max_nodes=10,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create function blocks
    preprocessing_block = create_preprocessing_block()
    steady_state_block = create_velocity_steady_state_block()
    dynamical_block = create_velocity_dynamical_block()
    driver_genes_block = create_driver_genes_block()
    analysis_block = create_velocity_analysis_block()
    
    # Create custom parameter sweep block for steady-state
    parameter_sweep_block = NewFunctionBlock(
        name="velocity_parameter_sweep",
        type=FunctionBlockType.PYTHON,
        description="Parameter sweep for velocity computation",
        code='''
def run(path_dict, params):
    """Run velocity computation with different parameters."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Running parameter sweep for velocity computation...")
    
    # Different neighbor values
    n_neighbors_values = [10, 30, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, n_neighbors in enumerate(n_neighbors_values):
        print(f"Computing velocity with n_neighbors={n_neighbors}")
        
        # Recompute neighbors
        scv.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)
        
        # Compute velocity
        scv.tl.velocity(adata, mode='stochastic')
        scv.tl.velocity_graph(adata)
        
        # Plot
        scv.pl.velocity_embedding_stream(adata, basis='umap', 
                                        ax=axes[i], show=False,
                                        title=f'n_neighbors={n_neighbors}')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_parameter_sweep.png', dpi=150)
    plt.close()
    
    # Reset to default
    scv.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
    scv.tl.velocity(adata, mode='stochastic')
    scv.tl.velocity_graph(adata)
    
    return adata
''',
        requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0\nnumpy>=1.24.0",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Parameter sweep for velocity parameters",
            tag="velocity_parameter_sweep",
            source="custom"
        )
    )
    
    # Initialize executor manager and node executor
    from ragomics_agent_local.job_executors import ExecutorManager
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Add root node (preprocessing)
    root_node = tree_manager.add_root_node(preprocessing_block)
    
    # Execute preprocessing
    print("\n1. Executing preprocessing (root node)...")
    tree_manager.update_node_execution(root_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=root_node, tree=tree, input_path=input_data_path, output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"   ✗ Preprocessing failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        root_node.id, NodeState.COMPLETED,
        output_data_id=str(result.output_data_path),
        figures=result.figures,
        duration=result.duration
    )
    print("   ✓ Preprocessing completed")
    
    # Branch 1: Steady-state model → Parameter sweep
    print("\n2. Branch 1: Steady-state model → Parameter sweep")
    branch1_nodes = tree_manager.add_child_nodes(root_node.id, [steady_state_block])
    steady_state_node = branch1_nodes[0]
    
    print("   2.1. Computing steady-state velocity...")
    tree_manager.update_node_execution(steady_state_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=steady_state_node, tree=tree, input_path=Path(root_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"      ✗ Steady-state failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        steady_state_node.id, NodeState.COMPLETED,
        output_data_id=str(result.output_data_path),
        figures=result.figures,
        duration=result.duration
    )
    print("      ✓ Steady-state completed")
    
    # Add parameter sweep as child of steady-state
    sweep_nodes = tree_manager.add_child_nodes(steady_state_node.id, [parameter_sweep_block])
    sweep_node = sweep_nodes[0]
    
    print("   2.2. Running parameter sweep...")
    tree_manager.update_node_execution(sweep_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=sweep_node, tree=tree, input_path=Path(steady_state_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            sweep_node.id, NodeState.COMPLETED,
            output_data_id=str(result.output_data_path),
            figures=result.figures,
            duration=result.duration
        )
        print("      ✓ Parameter sweep completed")
    else:
        print(f"      ✗ Parameter sweep failed: {result.error}")
    
    # Branch 2: Dynamical model → Driver genes + Comprehensive analysis
    print("\n3. Branch 2: Dynamical model → Driver genes + Comprehensive analysis")
    branch2_nodes = tree_manager.add_child_nodes(root_node.id, [dynamical_block])
    dynamical_node = branch2_nodes[0]
    
    print("   3.1. Computing dynamical velocity...")
    tree_manager.update_node_execution(dynamical_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=dynamical_node, tree=tree, input_path=Path(root_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state != NodeState.COMPLETED:
        print(f"      ✗ Dynamical failed: {result.error}")
        return False
    
    tree_manager.update_node_execution(
        dynamical_node.id, NodeState.COMPLETED,
        output_data_id=str(result.output_data_path),
        figures=result.figures,
        duration=result.duration
    )
    print("      ✓ Dynamical completed")
    
    # Add two parallel children to dynamical: driver genes AND comprehensive analysis
    parallel_nodes = tree_manager.add_child_nodes(dynamical_node.id, [driver_genes_block, analysis_block])
    driver_node = parallel_nodes[0]
    analysis_node = parallel_nodes[1]
    
    print("   3.2. Running parallel analyses...")
    
    # Execute driver genes
    print("      3.2.1. Identifying driver genes...")
    tree_manager.update_node_execution(driver_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=driver_node, tree=tree, input_path=Path(dynamical_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            driver_node.id, NodeState.COMPLETED,
            output_data_id=str(result.output_data_path),
            figures=result.figures,
            duration=result.duration
        )
        print("         ✓ Driver genes completed")
    else:
        print(f"         ✗ Driver genes failed: {result.error}")
    
    # Execute comprehensive analysis
    print("      3.2.2. Running comprehensive analysis...")
    tree_manager.update_node_execution(analysis_node.id, NodeState.RUNNING)
    state, result = node_executor.execute_node(node=analysis_node, tree=tree, input_path=Path(dynamical_node.output_data_id), output_base_dir=output_dir / tree.id
    )
    
    if state == NodeState.COMPLETED:
        tree_manager.update_node_execution(
            analysis_node.id, NodeState.COMPLETED,
            output_data_id=str(result.output_data_path),
            figures=result.figures,
            duration=result.duration
        )
        print("         ✓ Comprehensive analysis completed")
    else:
        print(f"         ✗ Comprehensive analysis failed: {result.error}")
    
    # Print tree structure
    print("\n4. Analysis tree structure:")
    print("   Root: Preprocessing")
    print("   ├── Branch 1: Steady-state velocity")
    print("   │   └── Parameter sweep")
    print("   └── Branch 2: Dynamical velocity")
    print("       ├── Driver genes identification")
    print("       └── Comprehensive analysis")
    
    # Save tree
    tree_manager.save_tree(output_dir / tree.id / "analysis_tree.json")
    print(f"\n5. Analysis tree saved to: {output_dir / tree.id / 'analysis_tree.json'}")
    
    # Verify outputs
    success = True
    tree_output_dir = output_dir / tree.id
    tree_json = tree_output_dir / "analysis_tree.json"
    
    if not tree_json.exists():
        print(f"   ✗ Missing analysis_tree.json")
        success = False
    
    # Check for figures in both branches
    branch_figures = {
        "Branch 1 (Steady-state → Parameter sweep)": 0,
        "Branch 2 (Dynamical → Driver genes)": 0,
        "Branch 2 (Dynamical → Comprehensive)": 0
    }
    
    # Check steady-state branch
    if 'sweep_node' in locals():
        figures_dir = tree_output_dir / tree.id / sweep_node.id / "figures"
        if figures_dir.exists():
            branch_figures["Branch 1 (Steady-state → Parameter sweep)"] = len(list(figures_dir.glob("*.png")))
    
    # Check dynamical branch outputs
    if 'driver_node' in locals():
        figures_dir = tree_output_dir / tree.id / driver_node.id / "figures"
        if figures_dir.exists():
            branch_figures["Branch 2 (Dynamical → Driver genes)"] = len(list(figures_dir.glob("*.png")))
    
    if 'analysis_node' in locals():
        figures_dir = tree_output_dir / tree.id / analysis_node.id / "figures"
        if figures_dir.exists():
            branch_figures["Branch 2 (Dynamical → Comprehensive)"] = len(list(figures_dir.glob("*.png")))
    
    print("\n6. Verification:")
    for branch, count in branch_figures.items():
        print(f"   {branch}: {count} figures")
        if count == 0:
            print(f"      ✗ No figures found")
            success = False
    
    return success


def main():
    """Run all scvelo test trees."""
    print("=== scVelo Manual Test Trees ===")
    
    # Use zebrafish data as test input
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if not input_data.exists():
        print(f"Error: Test data not found at {input_data}")
        return False
    
    output_base = Path("test_outputs/scvelo_trees")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Run all test trees
    tests = [
        ("Steady-State Model", test_scvelo_tree_1_steady_state),
        ("Model Comparison", test_scvelo_tree_2_model_comparison),
        ("Driver Genes Analysis", test_scvelo_tree_3_driver_genes),
        ("Multi-Branch Analysis", test_scvelo_tree_4_multi_branch)
    ]
    
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