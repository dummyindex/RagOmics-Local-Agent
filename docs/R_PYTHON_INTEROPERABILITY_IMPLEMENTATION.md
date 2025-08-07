# R-Python Interoperability Implementation

## Overview

This document summarizes the implementation of automatic R-Python interoperability in the RagOmics Local Agent system. The implementation allows seamless switching between Python (AnnData) and R (Seurat) function blocks through automatic conversion nodes.

## Implementation Summary

### Core Concept

The system automatically detects when a function block requires data in a different language format and inserts conversion nodes that create a shared `_node_sc_matrix` format readable by both languages. This eliminates the need for rpy2 or reticulate dependencies.

### Key Components Added/Modified

```
ragomics_agent_local/
├── src/ragomics_agent_local/function_blocks/builtin/
│   ├── convert_anndata_to_sc_matrix.py    [NEW]
│   ├── convert_seurat_to_sc_matrix.r      [NEW]
│   ├── convert_sc_matrix_to_anndata.py    [NEW]
│   └── convert_sc_matrix_to_seurat.r      [NEW]
├── agents/
│   ├── main_agent.py                       [MODIFIED]
│   ├── function_selector_agent.py          [MODIFIED]
│   └── function_creator_agent.py           [MODIFIED]
├── docker/
│   ├── Dockerfile.r.basic                  [NEW]
│   ├── Dockerfile.r.conversion             [NEW]
│   └── Dockerfile.r.seurat                 [NEW]
├── tests/
│   ├── test_sc_conversion_direct.py        [NEW]
│   ├── test_sc_matrix_conversion.py        [NEW]
│   ├── test_manual_r_python_conversion.py  [NEW]
│   ├── test_r_basic_execution.py           [NEW]
│   ├── test_r_conversion_docker.py         [NEW]
│   └── llm_required/
│       └── test_r_python_interop.py        [NEW]
└── docs/
    ├── PREV_REPO_R_python_conversion.md    [NEW]
    └── R_PYTHON_INTEROPERABILITY_IMPLEMENTATION.md [NEW]
```

## Detailed Changes

### 1. Conversion Function Blocks

#### `convert_anndata_to_sc_matrix.py`
- Converts AnnData objects to shared SC matrix format
- Creates structured output with:
  - `obs_names.txt`: Cell barcodes
  - `var_names.txt`: Gene names
  - `X.mtx`: Expression matrix (sparse) or `X.csv` (dense)
  - `obs/`: Cell metadata as CSV files
  - `metadata.json`: Data structure information
- Preserves original `_node_anndata.h5ad` in output

#### `convert_seurat_to_sc_matrix.r`
- Converts Seurat objects to shared SC matrix format
- Handles both Seurat v4 and v5 assay structures
- Extracts counts or data matrix as appropriate
- Transposes matrix to match AnnData format (cells × genes)
- Preserves original `_node_seuratObject.rds` in output

### 2. Main Agent Modifications

#### `main_agent.py`
Added method `_check_conversion_needed()`:
```python
def _check_conversion_needed(
    self,
    parent_node: AnalysisNode,
    child_block: Union[NewFunctionBlock, ExistingFunctionBlock],
    output_dir: Path
) -> Optional[Union[NewFunctionBlock, ExistingFunctionBlock]]
```

This method:
- Checks parent node output for data type
- Compares with child block requirements
- Returns appropriate conversion block if needed
- Returns None if no conversion needed or `_node_sc_matrix` already exists

Modified child node addition logic to insert conversion nodes:
```python
# Check if conversion is needed
conversion_block = self._check_conversion_needed(parent_node, block, output_dir)

if conversion_block:
    # First add conversion node
    conv_nodes = self.tree_manager.add_child_nodes(parent_id, [conversion_block])
    # Then add actual child to conversion node
```

### 3. Agent Prompt Updates

#### `function_selector_agent.py`
Updated `SYSTEM_PROMPT` to include:
- Awareness of R/Python function blocks
- Information about automatic conversion
- List of available conversion blocks
- Instruction to avoid rpy2/reticulate

#### `function_creator_agent.py`
- Added R function block documentation
- Added R function signature template
- Added language interoperability section
- Updated to support creating both Python and R blocks

### 4. Shared SC Matrix Format

The `_node_sc_matrix` directory structure:
```
_node_sc_matrix/
├── metadata.json         # Source type, shape, components
├── obs_names.txt        # Cell identifiers
├── var_names.txt        # Gene identifiers
├── X.mtx or X.csv      # Expression matrix
├── obs/                 # Cell metadata (optional)
│   ├── cell_type.csv
│   ├── batch.csv
│   └── ...
├── var/                 # Gene metadata (optional)
│   └── ...
└── layers/             # Additional matrices (optional)
    ├── spliced.mtx
    └── unspliced.mtx
```

### 5. Testing Infrastructure

#### Direct Conversion Tests
- `test_sc_conversion_direct.py`: Tests conversion functions directly
- Verifies both Python→SC and R→SC conversion
- Checks output structure and metadata

#### Integration Tests
- `test_manual_r_python_conversion.py`: Manual workflow test
- Creates Python → R → Python pipeline
- Verifies automatic conversion node insertion

#### LLM Integration Tests
- `test_r_python_interop.py`: Tests with full agent system
- Two test cases:
  1. Python → R workflow (scanpy → Seurat FindMarkers)
  2. R → Python workflow (Seurat → scVelo)

## Usage Examples

### Example 1: Mixed Analysis Request
```python
user_request = """
Analyze this dataset using quality control in Python, 
then use Seurat for clustering, and finally create 
visualizations with scanpy.
"""
```

The system will create:
1. Python QC node
2. [Auto] AnnData → SC matrix conversion
3. R Seurat clustering node
4. [Auto] Seurat → SC matrix conversion
5. Python visualization node

### Example 2: Leveraging R-specific Methods
```python
user_request = """
Use Seurat's SCTransform normalization method on this dataset,
then perform trajectory analysis with Python's scVelo.
"""
```

### 6. Docker Infrastructure

Three Dockerfiles have been created for different R environments:

#### `Dockerfile.r.basic`
- Minimal R environment with only Matrix and jsonlite packages
- Used for basic R execution and testing
- Successfully built and tested

#### `Dockerfile.r.conversion`
- Simplified environment for conversion testing
- Intended for quick builds but still includes Seurat dependencies

#### `Dockerfile.r.seurat`
- Full R environment with SeuratObject for complete conversion
- Currently building with essential dependencies

## Testing Progress

1. **Basic R Execution**: ✅ Successfully tested with Matrix/jsonlite
2. **SC Matrix Reading in R**: ✅ Verified R can read the shared format
3. **Python → SC Matrix**: ✅ Conversion function implemented and tested
4. **R → SC Matrix**: ✅ Conversion function implemented
5. **SC Matrix → Python**: ✅ Reverse conversion implemented
6. **SC Matrix → R**: ✅ Reverse conversion implemented
7. **Full Seurat Integration**: 🔄 In progress (building Docker image)

## Current Limitations

1. **Spatial Data**: Conversion for spatial transcriptomics data not fully implemented
2. **Complex Metadata**: Some uns (unstructured) data may not convert perfectly
3. **Layer Selection**: Currently limited to specific layers (spliced/unspliced)
4. **R Docker Image**: Full Seurat image build is time-consuming on ARM64

## Future Enhancements

1. Complete spatial data conversion support
2. Support for all layers and metadata
3. Optimization for very large datasets
4. Pre-built R Docker images for faster deployment
5. Additional format support (e.g., SingleCellExperiment, Loom)
6. Caching mechanism for conversion results

## Benefits

1. **No Bridge Dependencies**: No rpy2 or reticulate needed
2. **Language Flexibility**: Use best tools from both ecosystems
3. **Automatic Handling**: No manual conversion steps required
4. **Performance**: Preserves sparse matrix formats
5. **Debugging**: Clear intermediate format for inspection