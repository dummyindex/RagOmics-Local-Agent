# R-Python Data Conversion Documentation

## Overview

This document describes the bidirectional conversion system between Python's AnnData format and R's Seurat objects, as implemented in the ragene-R-job and ragene-python-job repositories. The system uses an intermediate "disassembled" file structure to facilitate data exchange without requiring direct Python-R interoperability.

## Architecture

### Data Flow

```
Python AnnData → disassemble_anndata.py → Intermediate Files → adata_reader.R → R Seurat Object
     ↑                                                                              ↓
Python AnnData ← assemble_anndata.py ← Intermediate Files ← adata_writer.R ← R Seurat Object
```

### Key Repositories

1. **ragene-R-job**: Handles R-side conversion (reading from and writing to intermediate format)
2. **ragene-python-job**: Handles Python-side conversion and data management

## Decomposition Process

### Python → Intermediate Format (disassemble_anndata.py)

The disassembly process breaks down an AnnData object into its component parts:

```
disassembled_directory/
├── obs_names.txt              # Cell barcodes/names
├── var_names.txt              # Gene names
├── X.mtx/.csv                 # Main expression matrix
├── X.doc                      # Matrix metadata (JSON)
├── obs/                       # Cell metadata directory
│   ├── obs_key1.csv          # Individual metadata columns
│   ├── obs_key2.csv
│   └── obs.doc               # Metadata documentation
├── var/                       # Gene metadata directory
│   ├── var_key1.csv
│   └── var.doc
├── layers/                    # Additional expression layers
│   ├── spliced.mtx           # RNA velocity layers
│   ├── unspliced.mtx
│   └── layers.doc
├── obsm/                      # Cell embeddings
│   ├── X_pca.mtx             # PCA coordinates
│   ├── X_umap.mtx            # UMAP coordinates
│   └── obsm.doc
├── varm/                      # Gene embeddings
├── obsp/                      # Cell-cell pairwise matrices
├── varp/                      # Gene-gene pairwise matrices
├── uns.h5ad                   # Unstructured metadata
└── spatial/                   # Spatial transcriptomics data
    ├── spatial.doc
    └── [library_id]/
        └── image-*.json       # Spatial images as JSON arrays
```

### Intermediate Format → R (adata_reader.R)

The `AdataReader` R6 class reconstructs Seurat objects from the disassembled components:

- Implements lazy loading with caching
- Handles automatic matrix transposition
- Converts Python data types to R equivalents
- Maps AnnData structure to Seurat hierarchy

### R → Intermediate Format (adata_writer.R)

The `AdataWriter` R6 class (when implemented) disassembles Seurat objects:

- Extracts expression matrices from various assay versions
- Preserves metadata with proper type conversion
- Handles dimensionality reductions
- Currently lacks spatial data support

### Intermediate Format → Python (assemble_anndata.py)

Reconstructs AnnData objects from the disassembled components:

- Restores original Python data types
- Handles various scipy.sparse matrix formats
- Validates data structure integrity

## Key Data Structures

### AnnData (Python)

```python
adata = {
    'X': sparse/dense matrix,           # Primary expression data
    'obs': DataFrame,                   # Cell metadata
    'var': DataFrame,                   # Gene metadata
    'obsm': {                          # Multi-dimensional cell data
        'X_pca': array,
        'X_umap': array,
        'spatial': array               # Spatial coordinates
    },
    'varm': {},                        # Multi-dimensional gene data
    'layers': {                        # Additional expression matrices
        'spliced': matrix,
        'unspliced': matrix
    },
    'obsp': {},                        # Cell-cell graphs
    'varp': {},                        # Gene-gene graphs
    'uns': {                           # Unstructured data
        'spatial': {                   # Spatial images and metadata
            'library_id': {
                'images': {},
                'scalefactors': {}
            }
        }
    }
}
```

### Seurat (R)

```r
seurat_object = {
    @assays = {
        $RNA = {
            @counts: matrix,           # Raw counts
            @data: matrix,            # Normalized data
            @scale.data: matrix,      # Scaled data
            @meta.features: DataFrame # Gene metadata
        },
        $spliced: Assay,             # Additional assays
        $unspliced: Assay
    },
    @meta.data: DataFrame,           # Cell metadata
    @reductions = {                  # Dimensionality reductions
        $pca: DimReduc,
        $umap: DimReduc
    },
    @images = {                      # Spatial data
        $slice1: VisiumV1 = {
            @image: array,           # Tissue image
            @coordinates: DataFrame, # Spot coordinates
            @scale.factors: list    # Scaling information
        }
    }
}
```

### Documentation Files (.doc)

Each component directory contains a `.doc` JSON file that stores metadata:

```json
{
    "type": "scipy.sparse.csr_matrix",  // Data type information
    "suffix": "mtx",                    // File format
    "shape": [n_cells, n_genes],       // Matrix dimensions
    "key": "X_pca",                     // Component identifier
    "is_ref": false,                    // Reference flag
    "source_data_id": "sample_001"      // Data provenance
}
```

## File Roles

### Core Conversion Scripts

1. **disassemble_anndata.py**
   - Entry point for AnnData → Intermediate conversion
   - Handles all AnnData components
   - Writes matrices in MTX (sparse) or CSV (dense) format
   - Creates documentation files for each component

2. **adata_reader.R**
   - R6 class for reading disassembled data
   - Provides methods like `get.seurat.object()`
   - Implements caching for performance
   - Handles type conversions and data mapping

3. **adata_writer.R**
   - R6 class for writing Seurat data (partial implementation)
   - Handles matrix extraction from different Seurat versions
   - Creates intermediate format from R objects

4. **assemble_anndata.py** (referenced but not found in current repos)
   - Should reconstruct AnnData from intermediate format
   - Handles type restoration and validation

### Utility Functions

1. **split_matrix.py**
   - Handles large matrix splitting for parallel processing
   - Uses column-wise chunking (default: 100 columns)

2. **data.py** (in ragene-python-job)
   - Provides comparison utilities for incremental updates
   - Implements `write_uns_spatial()` for spatial data
   - Handles change detection between AnnData versions

## Format Mappings

| AnnData Component | Intermediate Format | Seurat Equivalent |
|-------------------|-------------------|-------------------|
| `X` | X.mtx/csv + X.doc | `@assays$RNA@counts` |
| `obs` | obs/*.csv + obs.doc | `@meta.data` |
| `var` | var/*.csv + var.doc | `@assays$RNA@meta.features` |
| `obsm['X_pca']` | obsm/X_pca.mtx | `@reductions$pca@cell.embeddings` |
| `obsm['spatial']` | obsm/spatial.mtx | `@images$slice@coordinates` |
| `layers['spliced']` | layers/spliced.mtx | `@assays$spliced@counts` |
| `uns` | uns.h5ad | Limited support |
| `uns['spatial']` | spatial/* | `@images` (partial) |

## Current Limitations

1. **Spatial Data**: One-way conversion only (Python → R)
2. **Layer Selection**: Limited to 'spliced' and 'unspliced' layers
3. **Complex uns Data**: Some unstructured data may not convert perfectly
4. **Missing Assembly Script**: The Python assembly script appears incomplete

## Spatial Data Conversion Improvements

### Current State

- **Python → R**: Spatial data is disassembled but not read by R
- **R → Python**: Spatial data is not written by `adata_writer.R`

### Proposed Enhancements

#### 1. Complete Python → R Spatial Support

Add to `adata_reader.R`:

```r
get_spatial_coordinates = function() {
    if ("spatial" %in% names(self$obsm_doc)) {
        coords <- self$get_reduced_dims_item("spatial")
        colnames(coords) <- c("x", "y")
        return(coords)
    }
    return(NULL)
}

get.seurat.spatial = function() {
    coords <- self$get_spatial_coordinates()
    spatial_data <- self$get_spatial_images()
    
    # Create VisiumV1 objects for each library_id
    # Map coordinates and images appropriately
}
```

#### 2. Implement R → Python Spatial Support

Add to `adata_writer.R`:

```r
write_spatial_coordinates <- function(seurat_obj, obsm_path) {
    # Extract coordinates from Seurat @images slot
    # Convert to AnnData format (x,y coordinates)
}

write_spatial_data <- function(seurat_obj, disassemble_path, data_id) {
    # Extract images and scale factors
    # Write to spatial/ directory structure
}
```

#### 3. Key Considerations for Spatial Data

1. **Coordinate System Mapping**:
   - AnnData: Uses (x, y) coordinates in `obsm['spatial']`
   - Seurat: Uses (imagerow, imagecol) in `@coordinates`

2. **Image Handling**:
   - Store as JSON arrays for portability
   - Support both high and low resolution images
   - Preserve scale factors for proper mapping

3. **Multiple Samples**:
   - Support multiple spatial samples via library_id keys
   - Maintain sample-specific metadata

4. **Data Validation**:
   - Ensure coordinate dimensions match cell counts
   - Validate image array dimensions
   - Check scale factor completeness

## Performance Optimizations

1. **Sparse Matrix Preservation**: Maintains sparsity throughout pipeline
2. **Lazy Loading**: AdataReader implements caching to avoid re-reading
3. **Incremental Updates**: Change detection for efficient updates
4. **Parallel Processing**: Matrix splitting for large datasets

## Usage Examples

### Python to R Conversion

```python
# Python side
import anndata
adata = anndata.read_h5ad("sample.h5ad")
disassemble_anndata(adata, "/path/to/output", "sample_001")
```

```r
# R side
reader <- AdataReader$new("sample_001")
seurat_obj <- reader$get.seurat.object()
```

### R to Python Conversion

```r
# R side (proposed)
writer <- AdataWriter$new(seurat_obj, "sample_002")
writer$write_to_disk("/path/to/output")
```

```python
# Python side (proposed)
adata = assemble_anndata("/path/to/output")
```

## Future Directions

1. **Complete Bidirectional Support**: Implement missing assembly components
2. **Spatial Data**: Full support for spatial transcriptomics conversion
3. **Performance**: Optimize for very large datasets
4. **Validation**: Add comprehensive data integrity checks
5. **Extended Format Support**: Handle additional assay types and modalities