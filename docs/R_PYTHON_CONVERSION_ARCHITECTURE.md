# R-Python Conversion Architecture

## Overview

The RagOmics Local Agent implements automatic language interoperability between Python and R without requiring rpy2 or reticulate. This is achieved through a shared intermediate format and automatic conversion node insertion.

## How It Works

### 1. Conversion Detection (No LLM Required)

The agent uses **deterministic logic** to detect when conversion is needed - it does NOT use the LLM for this decision. The detection happens in `main_agent.py`:

```python
def _check_conversion_needed(
    self,
    parent_node: AnalysisNode,
    child_block: Union[NewFunctionBlock, ExistingFunctionBlock],
    output_dir: Path
) -> Optional[Union[NewFunctionBlock, ExistingFunctionBlock]]:
```

This method:
1. Checks the parent node's output type (Python or R)
2. Checks the child block's input requirements (Python or R)
3. If they differ, returns the appropriate conversion block
4. If they match, returns None (no conversion needed)

### 2. Automatic Conversion Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Request Example:                         │
│         "Normalize data with scanpy, then cluster with Seurat"      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Orchestrator Agent                          │
│                    (Uses LLM to plan workflow)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  Function Selector  │         │  Function Creator   │
        │   (Chooses scanpy)  │         │  (Creates Seurat    │
        │                     │         │   clustering code)  │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            Main Agent                                │
│                   (Deterministic Conversion Logic)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
    ┌─────────────────────┐                ┌─────────────────────┐
    │ 1. Python Node      │                │ 3. R Node           │
    │    (scanpy)         │                │    (Seurat)         │
    │ Output: anndata.h5ad│                │ Input: needs R data │
    └─────────────────────┘                └─────────────────────┘
                │                                       ▲
                │         ┌─────────────────────┐       │
                └────────►│ 2. Conversion Node │───────┘
                          │ (Auto-inserted)     │
                          │ Python → R format   │
                          └─────────────────────┘
```

### 3. Shared SC Matrix Format

The intermediate format (`_node_sc_matrix`) is a directory structure readable by both languages:

```
_node_sc_matrix/
├── metadata.json         # Format metadata
├── obs_names.txt        # Cell identifiers  
├── var_names.txt        # Gene identifiers
├── X.mtx                # Expression matrix (Matrix Market format)
├── obs/                 # Cell metadata
│   ├── cell_type.csv
│   └── batch.csv
└── var/                 # Gene metadata
    └── highly_variable.csv
```

## Conversion Decision Process

### Step 1: Parent Node Execution
When a node completes, it produces output in its native format:
- Python nodes → `_node_anndata.h5ad`
- R nodes → `_node_seuratObject.rds`

### Step 2: Child Node Addition
When adding a child node, the agent:

```python
# In main_agent.py - add_child_nodes()

# Check if conversion needed
conversion_block = self._check_conversion_needed(parent_node, child_block, output_dir)

if conversion_block:
    # Insert conversion node first
    conv_nodes = self.tree_manager.add_child_nodes(parent_id, [conversion_block])
    self.execute_node(conv_nodes[0].id)
    
    # Then add actual child to conversion node
    child_nodes = self.tree_manager.add_child_nodes(conv_nodes[0].id, [child_block])
else:
    # No conversion needed, add directly
    child_nodes = self.tree_manager.add_child_nodes(parent_id, [child_block])
```

### Step 3: Conversion Detection Logic

```python
# Simplified logic in _check_conversion_needed()

parent_type = parent_node.function_block.type  # FunctionBlockType.PYTHON or .R
child_type = child_block.type                   # FunctionBlockType.PYTHON or .R

# Check output files
has_anndata = (output_dir / "_node_anndata.h5ad").exists()
has_seurat = (output_dir / "_node_seuratObject.rds").exists()
has_sc_matrix = (output_dir / "_node_sc_matrix").exists()

# If already have SC matrix, no conversion needed
if has_sc_matrix:
    return None

# Python → R: need to convert AnnData
if parent_type == PYTHON and child_type == R and has_anndata:
    return ExistingFunctionBlock(
        name="convert_anndata_to_sc_matrix",
        function_block_id="builtin_convert_anndata_to_sc_matrix"
    )

# R → Python: need to convert Seurat
if parent_type == R and child_type == PYTHON and has_seurat:
    return ExistingFunctionBlock(
        name="convert_seurat_to_sc_matrix", 
        function_block_id="builtin_convert_seurat_to_sc_matrix"
    )

# Same language, no conversion
return None
```

## LLM vs Deterministic Components

### Uses LLM:
- **Orchestrator Agent**: Plans the analysis workflow
- **Function Selector**: Chooses appropriate existing functions
- **Function Creator**: Generates new function code
- **Bug Fixer**: Debugs failed nodes

### Deterministic (No LLM):
- **Conversion Detection**: Pure logic based on file types
- **Node Execution**: Docker container management
- **File Management**: Path handling and data flow
- **Tree Structure**: Parent-child relationships

## Benefits of This Architecture

1. **No Bridge Dependencies**: No rpy2 or reticulate needed
2. **Language Flexibility**: Use best tools from each ecosystem
3. **Automatic Handling**: Users don't need to think about conversion
4. **Performance**: Preserves sparse matrix formats
5. **Debugging**: Clear intermediate format for inspection
6. **Extensibility**: Easy to add more language conversions

## Example Workflow

User request: "Load data, normalize with scanpy, find markers with Seurat, visualize with scanpy"

Resulting execution tree:
```
1. Load Data (Python)
   └── 2. Normalize (Python) 
       └── 3. [AUTO] Convert AnnData → SC Matrix
           └── 4. Find Markers (R)
               └── 5. [AUTO] Convert SC Matrix → AnnData  
                   └── 6. Visualize (Python)
```

The user only specified steps 1, 2, 4, and 6. Steps 3 and 5 were automatically inserted by the agent's deterministic logic.