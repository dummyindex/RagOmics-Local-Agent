#!/usr/bin/env python3
"""Check clustering output."""

import scanpy as sc
from pathlib import Path

# Load output data
output_path = Path("/Users/random/Ragomics-workspace-all/agent_cc/ragomics_agent_local/test_outputs/clustering_final_20250802_004825/7326eff9-c835-49ee-9e9f-3f16bba41357/f16356d2-e0f4-40be-ac3e-f5180e9e4f50/output/_node_anndata.h5ad")

if output_path.exists():
    adata = sc.read_h5ad(output_path)
    
    print(f"Output data shape: {adata.shape}")
    print(f"\nObservation columns: {list(adata.obs.columns)}")
    
    # Check for clustering results
    clustering_cols = [col for col in adata.obs.columns if 
                      'leiden' in col.lower() or 
                      'louvain' in col.lower() or 
                      'kmeans' in col.lower() or
                      'cluster' in col.lower()]
    
    print(f"\nClustering columns found: {clustering_cols}")
    
    # Check for metrics
    print(f"\nUns keys: {list(adata.uns.keys())}")
    
    if 'clustering_metrics' in adata.uns:
        print("\n✓ clustering_metrics found!")
        metrics = adata.uns['clustering_metrics']
        print(f"Metrics type: {type(metrics)}")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                print(f"  {key}: {value}")
    else:
        print("\n✗ No clustering_metrics in uns")
        
    # Check for embeddings
    print(f"\nObsm keys: {list(adata.obsm.keys())}")
    
    # Check for figures
    figures_dir = output_path.parent / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        print(f"\nFigures found ({len(figures)}):")
        for fig in figures:
            print(f"  - {fig.name}")
else:
    print(f"Output file not found: {output_path}")