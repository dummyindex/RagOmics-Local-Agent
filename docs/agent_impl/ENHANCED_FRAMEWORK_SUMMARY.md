# Enhanced Function Block Framework Implementation Summary

## Overview
Implemented a more general function block framework that supports multiple input/output files and provides execution context with JSON metadata.

## Key Changes

### 1. Enhanced Data Models (`models_v2.py`)
- **New Models Added:**
  - `FileType` enum: Supports various file types (anndata, csv, json, image, etc.)
  - `FileInfo`: Detailed information about input/output files
  - `ExecutionContext`: Comprehensive context provided to function blocks
  - `InputSpecification`/`OutputSpecification`: Define expected inputs/outputs
  - `NodeExecutionResult`: Enhanced result tracking with file metadata
  - `ExecutionRequest`/`ExecutionResponse`: Structured execution flow

### 2. Enhanced Executor (`job_executors/enhanced_executor.py`)
- Handles general file inputs/outputs (not limited to anndata)
- Creates execution environment with proper file structure:
  ```
  execution/<node_id>/
  ├── input/           # Input files copied here
  ├── output/          # Function block outputs
  │   └── figures/     # Generated figures
  └── execution_context.json  # Context metadata
  ```
- Automatically collects and categorizes output files
- Provides execution context JSON to function blocks

### 3. Enhanced Node Executor (`analysis_tree_management/enhanced_node_executor.py`)
- Builds execution context from analysis tree
- Tracks files from previous nodes
- Supports directory-based inputs
- Maintains compatibility with legacy executors

### 4. Function Block Loader (`utils/function_block_loader.py`)
- Loads function blocks from directory structure
- Parses configuration, code, and requirements
- Supports listing blocks by tags
- Directory structure:
  ```
  test_function_blocks/
  ├── preprocessing/
  │   └── scvelo_preprocessing/
  │       ├── config.json
  │       ├── code.py
  │       └── requirements.txt
  ├── velocity_analysis/
  ├── trajectory_inference/
  └── quality_control/
  ```

### 5. Test Function Blocks Created
- **Quality Control** (`basic_qc`): Basic QC with multiple outputs
- **Preprocessing** (`scvelo_preprocessing`): Enhanced with context support
- **Velocity Analysis** (`velocity_steady_state`): Uses new framework
- **Trajectory Inference** (`elpigraph_trajectory`): Demonstrates metadata
- **Test Blocks** (`trajectory_failing`, `trajectory_fixed`): For bug fixer tests

## Function Block Structure

### Configuration (`config.json`)
```json
{
  "name": "block_name",
  "type": "python",
  "description": "Block description",
  "static_config": {
    "args": [...],
    "input_specification": {
      "required_files": [
        {"name": "anndata.h5ad", "type": "anndata", "description": "..."}
      ]
    },
    "output_specification": {
      "output_files": [...],
      "figures": [...],
      "metadata_keys": [...]
    }
  }
}
```

### Code Structure (`code.py`)
```python
def run(context, parameters, input_dir, output_dir):
    """Enhanced function block with context support."""
    # Access context
    input_files = context.get('input_files', [])
    tree_metadata = context.get('tree_metadata', {})
    
    # Process inputs from input_dir
    # Write outputs to output_dir
    # Save figures to output_dir/figures/
    
    # Return metadata
    return {"metadata": {...}}
```

## Benefits

1. **Flexibility**: Not limited to anndata inputs
2. **Context Awareness**: Function blocks know about previous results
3. **Better Organization**: Clear input/output structure
4. **Metadata Tracking**: Comprehensive metadata throughout pipeline
5. **File Type Support**: Handles CSV, JSON, images, etc.
6. **Backward Compatibility**: Works with existing infrastructure

## Testing

Created comprehensive tests:
- `test_enhanced_simple.py`: Basic component testing
- `test_enhanced_framework.py`: Full pipeline testing
- `scvelo_manual_tests_v2.py`: Updated scVelo tests with new framework

## Usage Example

```python
# Load function blocks
loader = FunctionBlockLoader("test_function_blocks")
qc_block = loader.load_function_block("quality_control/basic_qc")

# Create execution context
context = ExecutionContext(
    node_id="node-123",
    tree_id="tree-456",
    input_files=[FileInfo(filename="data.h5ad", ...)],
    input_dir="/workspace/input",
    output_dir="/workspace/output"
)

# Execute with enhanced executor
executor = EnhancedExecutor(docker_client, image_name)
response = executor.execute(ExecutionRequest(node, context), workspace_dir)
```

## Next Steps

1. Migrate existing tests to use test function blocks
2. Update agents to leverage enhanced metadata
3. Implement file caching for large datasets
4. Add support for streaming/chunked processing
5. Create more specialized function blocks

The enhanced framework provides a solid foundation for more flexible and powerful single-cell analysis pipelines.