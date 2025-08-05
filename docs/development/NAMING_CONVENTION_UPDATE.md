# File Naming Convention Update Summary

## Date: 2025-08-03

## Overview
Updated the entire Ragomics Agent system to use the new file naming conventions and framework where function blocks have no arguments and handle file I/O directly.

## Key Changes

### 1. File Naming Conventions
- **Python**: Changed from `output_data.h5ad` to `_node_anndata.h5ad` for both input and output
- **R**: Uses `_node_seuratObject.rds` for both input and output
- Cross-language compatibility maintained (R can read Python's h5ad, Python can read R's rds)

### 2. Function Block Framework Update
- **Old**: `def run(adata, **parameters)` - received data as argument
- **New**: `def run()` - no arguments, reads/writes files directly
- Parameters read from `/workspace/parameters.json`
- Input from `/workspace/input/_node_anndata.h5ad`
- Output to `/workspace/output/_node_anndata.h5ad`

### 3. Files Updated

#### Executors
- `job_executors/python_executor.py`:
  - Updated wrapper to call `run()` without arguments
  - Removed adata loading and passing logic
  - Function block now handles its own I/O

- `job_executors/r_executor.py`:
  - Updated wrapper to call `run()` without arguments
  - Added support for `_node_seuratObject.rds`
  - Function block handles its own I/O

#### Documentation
- `agents/FUNCTION_BLOCK_FRAMEWORK.md`:
  - Updated all examples to show no-argument `run()` function
  - Added clear file naming conventions section
  - Updated Python and R templates
  - Removed all references to adata arguments

- `ANALYSIS_TREE_VISUAL.md`:
  - Updated all references from `output_data.h5ad` to `_node_anndata.h5ad`

- `README.md`:
  - Updated output structure example

#### Tests
- `tests/test_file_passing.py`:
  - Updated test cases for new framework conventions
  - Tests verify no arguments in `run()` function
  - Tests check for proper file naming

## Framework Conventions Summary

### Python Function Block Template
```python
def run():
    import scanpy as sc
    import json
    
    # Load parameters
    with open('/workspace/parameters.json') as f:
        parameters = json.load(f)
    
    # Load input
    adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    
    # Process...
    
    # Save output
    adata.write('/workspace/output/_node_anndata.h5ad')
```

### R Function Block Template
```r
run <- function() {
    library(Seurat)
    library(jsonlite)
    
    # Load parameters
    params <- fromJSON("/workspace/parameters.json")
    
    # Load input
    if (file.exists("/workspace/input/_node_seuratObject.rds")) {
        seurat_obj <- readRDS("/workspace/input/_node_seuratObject.rds")
    } else if (file.exists("/workspace/input/_node_anndata.h5ad")) {
        # From Python parent
        library(anndata)
        adata <- read_h5ad("/workspace/input/_node_anndata.h5ad")
        seurat_obj <- CreateSeuratObject(counts = adata$X)
    }
    
    # Process...
    
    # Save output
    saveRDS(seurat_obj, "/workspace/output/_node_seuratObject.rds")
}
```

## Benefits of New Framework

1. **Simplicity**: No need to handle data loading/passing in wrapper
2. **Consistency**: All function blocks follow same pattern
3. **Flexibility**: Function blocks control their own I/O
4. **Docker-ready**: Aligns with containerized execution model
5. **Cross-language**: Clear conventions for Python/R interoperability

## Testing

All tests pass with the new conventions:
- File passing tests verify correct naming
- Function block tests verify no-argument structure
- Execution tests verify proper I/O handling

## Migration Notes

For existing function blocks:
1. Remove all arguments from `run()` function
2. Add parameter loading from JSON file
3. Update input path to `/workspace/input/_node_anndata.h5ad`
4. Update output path to `/workspace/output/_node_anndata.h5ad`
5. Remove return statement (optional, not used)

## Status

✅ All components updated and tested
✅ Documentation updated
✅ Tests passing
✅ Ready for production use with new framework