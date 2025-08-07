#!/usr/bin/env python3
"""Check what's in the AnnData object."""

import scanpy as sc

# Read the output from DPT node
adata_path = "/Users/random/Ragomics-workspace-all/agent_cc/ragomics_agent_local/tests/test_outputs/pseudotime_python_benchmark/results/67a5d651-d916-42ff-98e5-24c9d0b5a0d6/nodes/node_a057087d-5fae-4f7c-88a1-9693c2c1e8b0/outputs/_node_anndata.h5ad"
adata = sc.read_h5ad(adata_path)

print("Shape:", adata.shape)
print("obsm keys:", list(adata.obsm.keys()))
print("obs keys:", list(adata.obs.keys()))
print("uns keys:", list(adata.uns.keys()))
print("var keys:", list(adata.var.keys()))

# Check if diffmap was computed
if 'X_diffmap' in adata.obsm:
    print(f"X_diffmap shape: {adata.obsm['X_diffmap'].shape}")

if 'iroot' in adata.uns:
    print(f"iroot: {adata.uns['iroot']}")