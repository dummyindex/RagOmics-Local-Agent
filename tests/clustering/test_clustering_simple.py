#!/usr/bin/env python3
"""Simple test for clustering benchmark without LLM."""

import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    NodeState, GenerationMode, FunctionBlockType, 
    NewFunctionBlock, StaticConfig, Arg
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def create_clustering_benchmark_block():
    """Create a comprehensive clustering benchmark function block."""
    code = '''
def run(adata, **parameters):
    """Benchmark different clustering methods on scRNA-seq data."""
    import scanpy as sc
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    import warnings
    warnings.filterwarnings('ignore')
    
    print("Starting clustering benchmark...")
    print(f"Input shape: {adata.shape}")
    
    # Get ground truth if available
    ground_truth_key = parameters.get('ground_truth_key', 'Cell_type')
    if ground_truth_key not in adata.obs.columns:
        print(f"Warning: Ground truth key '{ground_truth_key}' not found in obs")
        ground_truth_key = None
    else:
        print(f"Using ground truth: {ground_truth_key}")
    
    # Preprocessing if not already done
    if 'highly_variable' not in adata.var.columns:
        print("Performing preprocessing...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        
        # Scale
        sc.pp.scale(adata, max_value=10)
    
    # PCA if not already computed
    if 'X_pca' not in adata.obsm:
        print("Computing PCA...")
        sc.tl.pca(adata, svd_solver='arpack')
    
    # Compute neighbors if not already done
    if 'connectivities' not in adata.obsp:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # 1. UMAP with different parameters
    print("\\n1. Computing UMAP embeddings with different parameters...")
    umap_params = [
        {'n_neighbors': 15, 'min_dist': 0.1},
        {'n_neighbors': 30, 'min_dist': 0.3},
        {'n_neighbors': 50, 'min_dist': 0.5}
    ]
    
    for i, params in enumerate(umap_params):
        print(f"   UMAP {i+1}: n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
        sc.pp.neighbors(adata, n_neighbors=params['n_neighbors'], n_pcs=40)
        sc.tl.umap(adata, min_dist=params['min_dist'])
        adata.obsm[f'X_umap_{i+1}'] = adata.obsm['X_umap'].copy()
    
    # Reset to default neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    
    # 2. Run multiple clustering methods
    print("\\n2. Running clustering methods...")
    clustering_methods = []
    metrics_results = []
    
    # Method 1: Leiden clustering with different resolutions
    print("   - Leiden clustering...")
    for resolution in [0.5, 1.0, 1.5]:
        sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_res_{resolution}')
        clustering_methods.append(f'leiden_res_{resolution}')
    
    # Method 2: Louvain clustering
    print("   - Louvain clustering...")
    for resolution in [0.5, 1.0]:
        sc.tl.louvain(adata, resolution=resolution, key_added=f'louvain_res_{resolution}')
        clustering_methods.append(f'louvain_res_{resolution}')
    
    # Method 3: K-means clustering
    print("   - K-means clustering...")
    from sklearn.cluster import KMeans
    for n_clusters in [10, 15, 20]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        adata.obs[f'kmeans_{n_clusters}'] = kmeans.fit_predict(adata.obsm['X_pca'][:, :50]).astype(str)
        clustering_methods.append(f'kmeans_{n_clusters}')
    
    # Method 4: Hierarchical clustering
    print("   - Hierarchical clustering...")
    from sklearn.cluster import AgglomerativeClustering
    for n_clusters in [10, 15]:
        hclust = AgglomerativeClustering(n_clusters=n_clusters)
        adata.obs[f'hierarchical_{n_clusters}'] = hclust.fit_predict(adata.obsm['X_pca'][:, :50]).astype(str)
        clustering_methods.append(f'hierarchical_{n_clusters}')
    
    # Method 5: DBSCAN
    print("   - DBSCAN clustering...")
    from sklearn.cluster import DBSCAN
    for eps in [0.5, 1.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        adata.obs[f'dbscan_eps_{eps}'] = dbscan.fit_predict(adata.obsm['X_pca'][:, :50]).astype(str)
        clustering_methods.append(f'dbscan_eps_{eps}')
    
    # 3. Calculate metrics for each clustering method
    print("\\n3. Calculating clustering metrics...")
    
    for method in clustering_methods:
        print(f"   Evaluating {method}...")
        labels = adata.obs[method]
        
        # Skip if all points in one cluster or noise
        n_clusters = len(set(labels)) - (1 if '-1' in labels else 0)
        if n_clusters < 2:
            print(f"     Skipping {method} (only {n_clusters} clusters)")
            continue
        
        # Internal metrics (no ground truth needed)
        try:
            silhouette = metrics.silhouette_score(adata.obsm['X_pca'][:, :50], labels)
            calinski = metrics.calinski_harabasz_score(adata.obsm['X_pca'][:, :50], labels)
            davies = metrics.davies_bouldin_score(adata.obsm['X_pca'][:, :50], labels)
        except:
            silhouette = calinski = davies = np.nan
        
        # External metrics (if ground truth available)
        if ground_truth_key:
            try:
                ari = metrics.adjusted_rand_score(adata.obs[ground_truth_key], labels)
                ami = metrics.adjusted_mutual_info_score(adata.obs[ground_truth_key], labels)
                nmi = metrics.normalized_mutual_info_score(adata.obs[ground_truth_key], labels)
                completeness = metrics.completeness_score(adata.obs[ground_truth_key], labels)
                homogeneity = metrics.homogeneity_score(adata.obs[ground_truth_key], labels)
            except:
                ari = ami = nmi = completeness = homogeneity = np.nan
        else:
            ari = ami = nmi = completeness = homogeneity = np.nan
        
        # Store metrics
        metrics_dict = {
            'method': method,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'adjusted_rand_index': ari,
            'adjusted_mutual_info': ami,
            'normalized_mutual_info': nmi,
            'completeness_score': completeness,
            'homogeneity_score': homogeneity
        }
        metrics_results.append(metrics_dict)
        
        # Store in adata.uns
        adata.uns[f'metrics_{method}'] = metrics_dict
    
    # 4. Create summary DataFrame
    print("\\n4. Creating metrics summary...")
    metrics_df = pd.DataFrame(metrics_results)
    
    # Store in adata
    adata.uns['clustering_metrics'] = metrics_df.to_dict()
    adata.uns['clustering_methods'] = clustering_methods
    
    # Print summary
    print("\\nClustering Benchmark Results:")
    print("="*60)
    print(metrics_df.to_string())
    print("="*60)
    
    # Find best method by different criteria
    if not metrics_df.empty:
        if ground_truth_key:
            best_ari = metrics_df.loc[metrics_df['adjusted_rand_index'].idxmax()]
            print(f"\\nBest by ARI: {best_ari['method']} (ARI={best_ari['adjusted_rand_index']:.3f})")
        
        best_silhouette = metrics_df.loc[metrics_df['silhouette_score'].idxmax()]
        print(f"Best by Silhouette: {best_silhouette['method']} (score={best_silhouette['silhouette_score']:.3f})")
    
    print(f"\\nOutput shape: {adata.shape}")
    print(f"Clustering methods tested: {len(clustering_methods)}")
    print(f"Metrics calculated and saved to adata.uns['clustering_metrics']")
    
    return adata
'''
    
    static_config = StaticConfig(
        args=[
            Arg(
                name="ground_truth_key",
                value_type="str",
                description="Column name in adata.obs containing ground truth labels",
                optional=True,
                default_value="Cell_type"
            )
        ],
        description="Comprehensive clustering benchmark for scRNA-seq data",
        tag="clustering_benchmark"
    )
    
    return NewFunctionBlock(
        name="clustering_benchmark",
        type=FunctionBlockType.PYTHON,
        description="Benchmark multiple clustering methods and calculate metrics",
        static_config=static_config,
        code=code,
        requirements="scanpy\nnumpy\npandas\nscikit-learn",
        parameters={"ground_truth_key": "Cell_type"}
    )


def main():
    """Run simple clustering benchmark test."""
    
    # Input/output paths
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if not input_data.exists():
        print(f"Error: Input data not found at {input_data}")
        return 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_outputs") / "clustering" / f"simple_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Simple Clustering Benchmark Test")
    print("="*60)
    print(f"Input: {input_data}")
    print(f"Output: {output_dir}")
    
    # Create components
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create tree
    print("\n1. Creating analysis tree...")
    tree = tree_manager.create_tree(
        user_request="Benchmark clustering methods",
        input_data_path=str(input_data),
        max_nodes=1,
        generation_mode=GenerationMode.ONLY_NEW
    )
    
    # Create and add clustering benchmark block
    print("2. Adding clustering benchmark node...")
    block = create_clustering_benchmark_block()
    node = tree_manager.add_root_node(block)
    
    # Execute
    print("3. Executing clustering benchmark...")
    state, output_path = node_executor.execute_node(
        node=node,
        tree=tree,
        input_path=input_data,
        output_base_dir=output_dir
    )
    
    # Results
    print("\n" + "="*60)
    if state == NodeState.COMPLETED:
        print("✓ Clustering benchmark completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Check for output file
        output_file = Path(output_path) / "output_data.h5ad"
        if output_file.exists():
            print(f"Output file: {output_file}")
            print("\nTo inspect results:")
            print("  import scanpy as sc")
            print(f"  adata = sc.read_h5ad('{output_file}')")
            print("  print(adata.uns['clustering_metrics'])")
        return 0
    else:
        print("✗ Clustering benchmark failed!")
        print(f"Error: {node.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())