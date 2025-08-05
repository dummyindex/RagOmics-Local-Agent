# Path Dictionary Framework Update

## Date: 2025-08-03

## Overview
Updated the Ragomics Agent system to use a `path_dict` argument in function blocks instead of hardcoded paths. This provides better abstraction and makes function blocks more portable.

## Key Changes

### 1. Function Block Signature Change

**Old Convention (hardcoded paths):**
```python
def run():
    # No arguments, hardcoded paths
    adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    # ...
    adata.write('/workspace/output/_node_anndata.h5ad')
```

**New Convention (path_dict):**
```python
def run(path_dict):
    # Takes path_dict with all necessary paths
    adata = sc.read_h5ad(path_dict["input_file"])
    # ...
    adata.write(path_dict["output_file"])
```

### 2. Path Dictionary Contents

The `path_dict` provided to function blocks contains:

#### Python Blocks:
```python
path_dict = {
    "input_dir": "/workspace/input",
    "output_dir": "/workspace/output",
    "params_file": "/workspace/parameters.json",
    "input_file": "/workspace/input/_node_anndata.h5ad",  # Standard input
    "output_file": "/workspace/output/_node_anndata.h5ad"  # Standard output
}
```

#### R Blocks:
```r
path_dict <- list(
    input_dir = "/workspace/input",
    output_dir = "/workspace/output",
    params_file = "/workspace/parameters.json",
    input_file_r = "/workspace/input/_node_seuratObject.rds",  # R input
    input_file_py = "/workspace/input/_node_anndata.h5ad",     # Python input
    output_file_r = "/workspace/output/_node_seuratObject.rds", # R output
    output_file_py = "/workspace/output/_node_anndata.h5ad"    # Python output (optional)
)
```

## Updated Components

### 1. Executors
- **python_executor.py**: Updated wrapper to create and pass `path_dict`
- **r_executor.py**: Updated wrapper to create and pass `path_dict`

### 2. Documentation
- **FUNCTION_BLOCK_FRAMEWORK.md**: Complete update with path_dict examples
- All patterns and templates updated

### 3. Agent Prompts
- **function_creator_agent.py**: Updated to generate blocks with `path_dict`
- All examples use the new convention

### 4. Tests
- **test_file_passing.py**: Updated all test blocks to use `path_dict`
- **test_path_dict_simple.py**: New test to verify framework

## Benefits

1. **No Hardcoded Paths**: Function blocks don't need to know about `/workspace/`
2. **Better Abstraction**: Paths can be changed without modifying function blocks
3. **Clearer Interface**: Explicit about what paths are available
4. **Easier Testing**: Can provide test paths without Docker
5. **Cross-Platform**: Paths can be adapted for different environments

## Migration Guide

### For Python Function Blocks

**Before:**
```python
def run():
    import scanpy as sc
    import json
    
    # Hardcoded paths
    with open('/workspace/parameters.json') as f:
        params = json.load(f)
    
    adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    # Process...
    adata.write('/workspace/output/_node_anndata.h5ad')
```

**After:**
```python
def run(path_dict):
    import scanpy as sc
    import json
    
    # Use path_dict
    with open(path_dict["params_file"]) as f:
        params = json.load(f)
    
    adata = sc.read_h5ad(path_dict["input_file"])
    # Process...
    adata.write(path_dict["output_file"])
```

### For R Function Blocks

**Before:**
```r
run <- function() {
    library(jsonlite)
    
    # Hardcoded paths
    params <- fromJSON("/workspace/parameters.json")
    
    if (file.exists("/workspace/input/_node_seuratObject.rds")) {
        seurat_obj <- readRDS("/workspace/input/_node_seuratObject.rds")
    }
    
    # Process...
    saveRDS(seurat_obj, "/workspace/output/_node_seuratObject.rds")
}
```

**After:**
```r
run <- function(path_dict) {
    library(jsonlite)
    
    # Use path_dict
    params <- fromJSON(path_dict$params_file)
    
    if (file.exists(path_dict$input_file_r)) {
        seurat_obj <- readRDS(path_dict$input_file_r)
    }
    
    # Process...
    saveRDS(seurat_obj, path_dict$output_file_r)
}
```

## Example Function Block

### Complete Python Example
```python
def run(path_dict):
    """
    Quality control for single-cell data.
    
    Args:
        path_dict: Dictionary with paths for input/output
    """
    import scanpy as sc
    import json
    import os
    
    # Load parameters
    with open(path_dict["params_file"]) as f:
        parameters = json.load(f)
    
    # Load input data
    print(f"Loading data from {path_dict['input_file']}")
    adata = sc.read_h5ad(path_dict["input_file"])
    print(f"Input shape: {adata.shape}")
    
    # Get parameters with defaults
    min_genes = parameters.get('min_genes', 200)
    min_cells = parameters.get('min_cells', 3)
    
    # Apply QC filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"After QC: {adata.shape}")
    
    # Save output
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    adata.write(path_dict["output_file"])
    print(f"Output saved to {path_dict['output_file']}")
    
    # Save figures if needed
    figures_dir = os.path.join(path_dict["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    # ... create and save plots ...
```

## Testing

All tests pass with the new framework:

```bash
# Test wrapper generation
python tests/test_path_dict_simple.py
# ✅ All path_dict framework tests passed!

# Test file passing with path_dict
python -m pytest tests/test_file_passing.py::TestFunctionBlockConventions -v
# ✅ 2 passed
```

## Status

✅ **Complete and Working**
- All executors updated
- Documentation updated
- Tests passing
- Ready for production use

## Important Notes

1. Function blocks MUST accept `path_dict` as their only argument
2. Parameters are loaded from `path_dict["params_file"]`
3. Standard file names remain the same (_node_anndata.h5ad, _node_seuratObject.rds)
4. The wrapper handles creating the path_dict - function blocks just use it
5. This change is backward-incompatible with old function blocks