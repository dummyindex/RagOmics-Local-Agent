# Function Block Framework

## Overview

This document defines the standardized framework and conventions for function blocks in the Ragomics Agent system, with special emphasis on single-cell RNA-seq data workflows.

## Core Principles

### 1. Data Flow Convention

All function blocks follow a standardized input/output pattern:

```
Parent Node Outputs → Child Node Inputs → Processing → Child Node Outputs
```

**Key Rules:**
- ALL files in a parent node's `outputs/` folder automatically pass to child nodes' `input/` folder
- Child nodes receive ALL parent outputs in their `/workspace/input/` directory
- No explicit file path configuration needed between nodes
- Python blocks use `_node_anndata.h5ad` as the standard filename
- R blocks use `_node_seuratObject.rds` as the standard filename

### 2. Single-Cell RNA-seq Data Convention

For single-cell genomics workflows, function blocks **MUST** follow these conventions:

#### NEW FRAMEWORK CONVENTION (PATH_DICT + PARAMS)
```python
def run(path_dict, params):
    """Function block for single-cell analysis.
    
    Args:
        path_dict: Dictionary containing paths:
            - input_dir: Input directory path
            - output_dir: Output directory path
        params: Dictionary of parameters for this function block
    """
    import scanpy as sc
    import os
    
    # Construct file paths using directories
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Load input data from standard location
    adata = sc.read_h5ad(input_file)
    
    # Process data using params
    # params is already a dictionary, no need to load from file
    
    # Save to standard output location
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    adata.write(output_file)
    
    # No return value needed - output is written to file
```

#### File Naming Conventions
- **Python blocks**: Use `_node_anndata.h5ad` for both input and output
- **R blocks**: Use `_node_seuratObject.rds` for both input and output
- **Cross-language**: Python can read R's `.rds` via anndata conversion, R can read Python's `.h5ad`

### 3. Standard Directory Structure

Each node execution creates this structure:

```
node_NODE_ID/
├── function_block/     # Function block definition
│   ├── code.py
│   ├── config.json
│   └── requirements.txt
├── jobs/              # Execution jobs
│   └── job_TIMESTAMP/
│       ├── input/     # ALL files from parent's outputs/ folder
│       │   ├── _node_anndata.h5ad  # Standard Python data file
│       │   ├── _node_seuratObject.rds  # Standard R data file (if R parent)
│       │   └── ...  # ALL other files from parent
│       └── output/    # Output data
│           ├── _node_anndata.h5ad  # Standard output name
│           └── figures/
├── outputs/           # Final outputs (copied from job)
│   ├── _node_anndata.h5ad  # Standard data file
│   └── figures/
└── agent_tasks/       # LLM interactions only
```

## Function Block Template

### Python Function Block for Single-Cell Analysis (PATH_DICT + PARAMS)

```python
def run(path_dict, params):
    """
    Standard function block for single-cell RNA-seq analysis.
    
    Args:
        path_dict: Dictionary with paths:
            - input_dir: Input directory
            - output_dir: Output directory
        params: Dictionary of parameters
    """
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Construct standard file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Ensure output directories exist
    figures_dir = os.path.join(path_dict["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load input data from standard location
    print(f"Loading data from {input_file}")
    adata = sc.read_h5ad(input_file)
    print(f"Input data shape: {adata.shape}")
    
    # ========================================
    # MAIN PROCESSING LOGIC HERE
    # ========================================
    
    # Example: Quality control using params directly
    if params.get('perform_qc', False):
        sc.pp.filter_cells(adata, min_genes=params.get('min_genes', 200))
        sc.pp.filter_genes(adata, min_cells=params.get('min_cells', 3))
    
    # ========================================
    # SAVE OUTPUTS
    # ========================================
    
    # Save processed data with standard name
    print(f"Saving processed data to {output_file}")
    adata.write(output_file)
    
    # Save any figures
    if params.get('generate_plots', False):
        fig, ax = plt.subplots(figsize=(10, 6))
        # ... plotting code ...
        plt.savefig(os.path.join(figures_dir, 'analysis_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Output data shape: {adata.shape}")
    # No return value needed - output is written to file
```

### R Function Block for Single-Cell Analysis (PATH_DICT + PARAMS)

```r
run <- function(path_dict, params) {
    # Function block with path dictionary and parameters
    # path_dict contains:
    #   - input_dir: Input directory
    #   - output_dir: Output directory
    # params is a list of parameters
    
    # Load required libraries
    library(Seurat)
    
    # Construct standard file paths
    input_r <- file.path(path_dict$input_dir, "_node_seuratObject.rds")
    input_py <- file.path(path_dict$input_dir, "_node_anndata.h5ad")
    output_r <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    output_py <- file.path(path_dict$output_dir, "_node_anndata.h5ad")
    
    # Create output directories
    figures_dir <- file.path(path_dict$output_dir, "figures")
    dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Load input data from standard location
    # Try standard R object name first
    if (file.exists(input_r)) {
        cat(paste("Loading Seurat object from", input_r, "\n"))
        seurat_obj <- readRDS(input_r)
    } else if (file.exists(input_py)) {
        # Load from Python parent node
        library(anndata)
        cat(paste("Loading from h5ad:", input_py, "\n"))
        adata <- read_h5ad(input_py)
        # Convert to Seurat object
        seurat_obj <- CreateSeuratObject(counts = adata$X)
        # Transfer metadata if available
        if (!is.null(adata$obs)) {
            seurat_obj@meta.data <- cbind(seurat_obj@meta.data, adata$obs)
        }
    } else {
        stop(paste("No input data found in", path_dict$input_dir))
    }
    
    # ========================================
    # MAIN PROCESSING LOGIC HERE
    # ========================================
    
    # Example: Normalization using params directly
    if (!is.null(params$normalize) && params$normalize) {
        seurat_obj <- NormalizeData(seurat_obj)
    }
    
    # ========================================
    # SAVE OUTPUTS
    # ========================================
    
    # Save Seurat object with standard name
    cat(paste("Saving Seurat object to", output_r, "\n"))
    saveRDS(seurat_obj, output_r)
    
    # Optionally also save as h5ad for Python compatibility
    if (!is.null(params$save_h5ad) && params$save_h5ad) {
        library(anndata)
        adata <- Convert(from = seurat_obj, to = "anndata")
        write_h5ad(adata, output_py)
    }
    
    # No return value needed - output is written to file
}
```

## File Passing Rules

### Automatic File Passing

1. **Parent → Child Transfer**
   - ALL files in parent's `outputs/` folder are copied to child's `input/` folder
   - Python nodes: `_node_anndata.h5ad` is the standard data file
   - R nodes: `_node_seuratObject.rds` is the standard data file
   - ALL supplementary files preserved with original names

2. **Multi-Parent Handling**
   - When a node has multiple parents, inputs are merged or selected based on node logic
   - Convention: Use the most recent parent's output as primary input

3. **File Discovery Priority**
   ```python
   # Python: Priority order for finding input data
   search_order = [
       '/workspace/input/_node_anndata.h5ad',  # Standard Python data
       'any .h5ad file in /workspace/input/',  # Any h5ad from parent
   ]
   
   # R: Priority order for finding input data
   search_order = [
       '/workspace/input/_node_seuratObject.rds',  # Standard R data
       'any .rds file in /workspace/input/',       # Any RDS from parent
       'any .h5ad file (convert to Seurat)',       # Cross-language support
   ]
   ```

## Best Practices

### 1. Always Check Input Existence

```python
import os

if not os.path.exists('/workspace/input/_node_anndata.h5ad'):
    print("Warning: No input data found, using alternative source")
    # Handle missing input appropriately
```

### 2. Preserve Data Lineage

```python
# Add processing history to adata
adata.uns['processing_history'] = adata.uns.get('processing_history', [])
adata.uns['processing_history'].append({
    'step': 'current_analysis',
    'timestamp': str(datetime.now()),
    'parameters': parameters
})
```

### 3. Handle Large Files Efficiently

```python
# For large datasets, use backed mode when possible
adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad', backed='r')
# Process in chunks if needed
```

### 4. Validate Outputs

```python
# Always validate output before saving
assert adata.n_obs > 0, "Output has no cells"
assert adata.n_vars > 0, "Output has no genes"

# Save with compression
adata.write('/workspace/output/_node_anndata.h5ad', compression='gzip')
```

### 5. Document Parameters

```python
def run(path_dict, params):
    """
    Comprehensive docstring explaining:
    - What the function does
    - path_dict contains input/output directories
    - params contains all parameters
    - Expected input format: _node_anndata.h5ad in input_dir
    - Output format: _node_anndata.h5ad in output_dir
    
    Args:
        path_dict: Dictionary with keys:
            - input_dir: Input directory path
            - output_dir: Output directory path
        params: Dictionary with parameters:
            - min_genes: int, default 200 - minimum genes per cell
            - max_genes: int, default 5000 - maximum genes per cell  
            - max_mt_percent: float, default 20 - maximum mitochondrial percentage
    """
    # Use parameters directly with defaults
    min_genes = params.get('min_genes', 200)
    max_genes = params.get('max_genes', 5000)
    max_mt_percent = params.get('max_mt_percent', 20)
```

## Integration with Agents

### Function Creator Agent

When creating new function blocks, the agent MUST:
1. Follow the standard template above
2. Include data loading from `/workspace/input/_node_anndata.h5ad`
3. Save outputs to `/workspace/output/_node_anndata.h5ad`
4. Handle missing inputs gracefully

### Bug Fixer Agent

When fixing function blocks, the agent MUST:
1. Preserve the input/output conventions
2. Check for file existence before operations
3. Ensure output directories exist
4. Maintain data flow compatibility

### Orchestrator Agent

The orchestrator MUST:
1. Ensure parent outputs are available to child nodes
2. Validate data flow between nodes
3. Handle multi-parent scenarios appropriately

## Validation Checklist

Before deploying a function block, verify:

- [ ] Loads data from `/workspace/input/_node_anndata.h5ad` when not provided
- [ ] Saves primary output to `/workspace/output/_node_anndata.h5ad`
- [ ] Creates output directories if they don't exist
- [ ] Handles missing input files gracefully
- [ ] Preserves data shape information in logs
- [ ] Includes appropriate error handling
- [ ] Follows naming conventions
- [ ] Documents all parameters
- [ ] Validates output before saving

## Common Patterns

### Pattern 1: Quality Control
```python
def run(path_dict, params):
    # Load data → Filter → Save
    import scanpy as sc
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    adata = sc.read_h5ad(input_file)
    
    min_genes = params.get('min_genes', 200)
    max_genes = params.get('max_genes', 5000)
    
    # Process...
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    
    adata.write(output_file)
```

### Pattern 2: Normalization
```python
def run(path_dict, params):
    # Load data → Normalize → Log-transform → Save
    import scanpy as sc
    import os
    
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    adata = sc.read_h5ad(input_file)
    target_sum = params.get('target_sum', 1e4)
    
    # Normalize and save
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.write(output_file)
```

### Pattern 3: Dimensionality Reduction
```python
def run(path_dict, params):
    # Load data → PCA/UMAP → Save with embeddings
    import scanpy as sc
    import os
    
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    adata = sc.read_h5ad(input_file)
    n_components = params.get('n_components', 50)
    
    # PCA and save
    sc.tl.pca(adata, n_comps=n_components)
    adata.write(output_file)
```

### Pattern 4: Clustering
```python
def run(path_dict, params):
    # Load data → Cluster → Save with labels
    import scanpy as sc
    import os
    
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    adata = sc.read_h5ad(input_file)
    resolution = params.get('resolution', 0.5)
    
    # Cluster and save
    sc.tl.leiden(adata, resolution=resolution)
    adata.write(output_file)
```

### Pattern 5: Differential Expression
```python
def run(path_dict, params):
    # Load data → Compare groups → Save with DE results
    import scanpy as sc
    import os
    
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    adata = sc.read_h5ad(input_file)
    group_by = params.get('group_by', 'leiden')
    
    # DE analysis and save
    sc.tl.rank_genes_groups(adata, group_by)
    adata.write(output_file)
```

## Error Handling

### Standard Error Messages

```python
# File not found
raise FileNotFoundError(
    "Input data not found at /workspace/input/_node_anndata.h5ad. "
    "Ensure parent node completed successfully."
)

# Invalid input
raise ValueError(
    f"Expected AnnData object, got {type(adata)}. "
    "Check data loading logic."
)

# Output failure
raise IOError(
    f"Failed to save output to /workspace/output/_node_anndata.h5ad. "
    "Check disk space and permissions."
)
```

## Testing Requirements

All function blocks MUST be tested for:

1. **Input handling**: Missing files, corrupted data, wrong format
2. **Processing logic**: Expected transformations, edge cases
3. **Output generation**: File creation, correct format, accessibility
4. **Parameter validation**: Type checking, range validation
5. **Error scenarios**: Graceful failures, informative messages

## Version Compatibility

- **AnnData**: >= 0.8.0
- **Scanpy**: >= 1.9.0
- **Python**: >= 3.8
- **R**: >= 4.0 (for R function blocks)

## References

- [AnnData Documentation](https://anndata.readthedocs.io/)
- [Scanpy Best Practices](https://scanpy.readthedocs.io/en/stable/tutorials.html)
- [Single-cell Best Practices Book](https://www.sc-best-practices.org/)

---

*This framework ensures consistent, reliable data flow through the analysis pipeline while maintaining flexibility for diverse analytical approaches.*