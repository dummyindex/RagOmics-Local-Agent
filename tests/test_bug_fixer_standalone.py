#!/usr/bin/env python
"""Standalone test for bug fixer with real clustering failure."""

import os
import json
from pathlib import Path

def test_bug_fixer_prompt():
    """Test what prompt would be sent to fix the scanpy scatter error."""
    
    failing_code = """def run(path_dict, params):
    import scanpy as sc
    import pandas as pd
    from sklearn import metrics
    from sklearn.cluster import KMeans, AgglomerativeClustering
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    def get_param(params, key, default):
        val = params.get(key, default)
        if isinstance(val, dict) and 'default_value' in val:
            return val.get('default_value', default)
        return val if val is not None else default

    # Construct file paths
    input_file = os.path.join(path_dict['input_dir'], '_node_anndata.h5ad')
    output_file = os.path.join(path_dict['output_dir'], '_node_anndata.h5ad')

    # Read input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file not found: {input_file}')
    adata = sc.read_h5ad(input_file)

    # Get parameters
    n_clusters = get_param(params, 'n_clusters', 5)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca'])

    # Apply Agglomerative clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    adata.obs['agglo'] = agglo.fit_predict(adata.obsm['X_pca'])

    # Calculate metrics
    if 'ground_truth' in adata.obs.columns:
        ari_kmeans = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['kmeans'])
        ari_agglo = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['agglo'])
        print(f'KMeans ARI: {ari_kmeans:.3f}')
        print(f'Agglomerative ARI: {ari_agglo:.3f}')

        # Save metrics to file
        metrics_df = pd.DataFrame({'metric': ['KMeans ARI', 'Agglomerative ARI'], 'value': [ari_kmeans, ari_agglo]})
        metrics_file = os.path.join(path_dict['output_dir'], 'clustering_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)

    # Create figures directory
    figures_dir = Path(path_dict['output_dir']) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot clustering results - THIS FAILS!
    plt.figure(figsize=(10, 5))
    sc.pl.scatter(adata, x=adata.obsm['X_pca'][:, 0], y=adata.obsm['X_pca'][:, 1], color='kmeans', title='KMeans Clustering', show=False)
    plt.savefig(figures_dir / 'kmeans_clustering.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    sc.pl.scatter(adata, x=adata.obsm['X_pca'][:, 0], y=adata.obsm['X_pca'][:, 1], color='agglo', title='Agglomerative Clustering', show=False)
    plt.savefig(figures_dir / 'agglo_clustering.png')
    plt.close()

    # Save output
    adata.write(output_file)
    print(f'Output saved to {output_file}')

    return adata"""
    
    error_message = """ValueError: `x`, `y`, and potential `color` inputs must all come from either `.obs` or `.var`"""
    
    parent_data_structure = {
        "shape": "4161 cells x 15496 genes",
        "obs_columns": [
            "split_id", "sample", "Size_Factor", "condition", 
            "Cluster", "Cell_type", "umap_1", "umap_2", 
            "batch", "n_genes"
        ],
        "var_columns": ["n_cells"],
        "obsm_keys": ["X_pca", "X_umap"],
        "varm_keys": ["PCs"],
        "uns_keys": ["log1p", "neighbors", "pca", "umap"],
        "layers": ["spliced", "unspliced"]
    }
    
    # Build the prompt that would be sent to GPT
    prompt = f"""Fix this Python code that's failing with an error.

## Original Code:
{failing_code}

## Error:
{error_message}

## Context from Parent Node:
The input AnnData has this structure:
{json.dumps(parent_data_structure, indent=2)}

## Problem Analysis:
The scanpy scatter plot function expects x and y to be column names from .obs or .var, but the code is passing numpy arrays from .obsm['X_pca']. 

## Requirements:
1. Fix the plotting code to work with scanpy's API
2. Keep all functionality intact
3. Use the available data structure (note 'Cell_type' column exists, not 'ground_truth')
4. Ensure plots are saved correctly

## Fixed Code:
Please provide the complete fixed code."""

    print("="*80)
    print("BUG FIXER PROMPT")
    print("="*80)
    print(prompt)
    print("\n" + "="*80)
    print("EXPECTED FIX")
    print("="*80)
    print("""
The bug fixer should:
1. NOT pass numpy arrays to sc.pl.scatter
2. Either:
   a) Use sc.pl.umap or sc.pl.pca for visualization 
   b) Add PCA coordinates to .obs columns first
   c) Use matplotlib directly with the numpy arrays
3. Fix 'ground_truth' to 'Cell_type'
""")

if __name__ == "__main__":
    test_bug_fixer_prompt()