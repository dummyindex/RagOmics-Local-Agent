def run(adata=None, min_genes=200, **parameters):
    import scanpy as sc
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # FRAMEWORK CONVENTION: Load from standard input
    if adata is None:
        adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    
    print(f"QC: Input shape: {adata.shape}")
    
    # Simulate QC processing
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Add QC metrics
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    adata.obs['qc_pass'] = True
    
    # Create QC plot
    os.makedirs('/workspace/output/figures', exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(adata.obs['n_genes'], bins=50)
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Number of cells')
    ax.set_title('QC: Gene Distribution')
    plt.savefig('/workspace/output/figures/qc_genes_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"QC: Output shape: {adata.shape}")
    
    # FRAMEWORK CONVENTION: Save to standard output
    os.makedirs('/workspace/output', exist_ok=True)
    adata.write('/workspace/output/_node_anndata.h5ad')
    
    # Save QC report
    with open('/workspace/output/qc_report.txt', 'w') as f:
        f.write(f"QC Report
")
        f.write(f"Input cells: {adata.shape[0]}
")
        f.write(f"Input genes: {adata.shape[1]}
")
        f.write(f"Min genes filter: {min_genes}
")
    
    return adata