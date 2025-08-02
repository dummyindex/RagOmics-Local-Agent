# Analysis Tree Structure Specification

## Overview
The analysis tree represents a hierarchical workflow of computational tasks. Each tree has a unique structure that organizes all analysis components, node executions, and agent interactions.

## Directory Structure

```
output_dir/
├── main_TIMESTAMP_TREEID/           # Main agent task folder
│   ├── task_metadata.json          # Main task metadata
│   └── orchestrator_TIMESTAMP/     # Orchestrator subfolder
│       ├── analysis_tree.json      # Tree structure and metadata
│       ├── selector_TIMESTAMP/     # Selector agent task
│       │   ├── context.json
│       │   └── results.json
│       └── creator_*.json          # References to creator tasks
│
└── tree_TREEID/                    # Analysis tree folder
    ├── analysis_tree.json          # Complete tree structure
    ├── tree_metadata.json          # Tree-level metadata
    ├── nodes/                      # All nodes in flat structure
    │   ├── node_NODEID1/          # Individual node folder
    │   │   ├── node_info.json     # Node metadata
    │   │   ├── function_block/    # Function block definition
    │   │   │   ├── code.py        # Function code
    │   │   │   ├── config.json    # Static configuration
    │   │   │   └── requirements.txt
    │   │   ├── jobs/              # Execution history
    │   │   │   ├── job_TIMESTAMP_JOBID/
    │   │   │   │   ├── execution_summary.json
    │   │   │   │   ├── logs/
    │   │   │   │   │   ├── stdout.txt
    │   │   │   │   │   └── stderr.txt
    │   │   │   │   └── output/
    │   │   │   │       ├── output_data.h5ad
    │   │   │   │       └── figures/
    │   │   │   └── latest -> job_TIMESTAMP_JOBID/
    │   │   ├── outputs/           # Final outputs
    │   │   │   ├── output_data.h5ad
    │   │   │   └── figures/
    │   │   └── agent_tasks/      # Related agent tasks
    │   │       ├── creator_TIMESTAMP/
    │   │       │   ├── task_info.json
    │   │       │   ├── llm_input.json
    │   │       │   ├── llm_output.json
    │   │       │   └── generated_code.py
    │   │       ├── fixer_TIMESTAMP/
    │   │       │   ├── task_info.json
    │   │       │   ├── context.json
    │   │       │   ├── result.json
    │   │       │   └── fixed_code.py
    │   │       └── selector_TIMESTAMP/
    │   │           ├── task_info.json
    │   │           └── results.json
    │   └── node_NODEID2/
    │       └── ... (same structure)
    └── agent_tasks/               # Tree-level agent tasks
        └── orchestrator_TIMESTAMP/
            ├── task_info.json
            └── decisions.json

```

## Component Descriptions

### Main Agent Task Folder (`main_TIMESTAMP_TREEID/`)
- Created by MainAgent when initiating an analysis
- Contains orchestrator decisions and high-level task metadata
- TIMESTAMP: Task creation time (YYYYMMDD_HHMMSS)
- TREEID: First 8 characters of the analysis tree UUID

### Analysis Tree Folder (`tree_TREEID/`)
- Root folder for all tree-related data
- TREEID: Full UUID of the analysis tree
- Contains the complete analysis workflow

### Nodes Folder (`nodes/`)
- Flat structure containing all nodes (no recursion needed)
- Each node has its own folder with complete information
- Node relationships are stored in analysis_tree.json

### Node Folder (`node_NODEID/`)
Each node folder contains:

1. **node_info.json**: Node metadata including:
   - Node ID
   - Function block name and type
   - Parent/children relationships
   - Execution state
   - Creation timestamp

2. **function_block/**: Complete function block definition
   - `code.py`: Executable code
   - `config.json`: Static configuration and parameters
   - `requirements.txt`: Package dependencies

3. **jobs/**: Execution history
   - Each job in timestamped folder
   - Symbolic link `latest` points to most recent job
   - Contains logs, outputs, and execution metrics

4. **outputs/**: Final successful outputs
   - Latest successful output_data.h5ad
   - Generated figures
   - Linked from latest successful job

5. **agent_tasks/**: Node-specific agent activities
   - Creator tasks: Function block generation
   - Fixer tasks: Bug fixing attempts
   - Selector tasks: Function selection decisions

## File Formats

### analysis_tree.json
```json
{
  "id": "tree-uuid",
  "user_request": "Analysis description",
  "input_data_path": "/path/to/input",
  "created_at": "2025-08-02T10:00:00",
  "generation_mode": "mixed",
  "max_nodes": 20,
  "nodes": {
    "node-id": {
      "id": "node-id",
      "parent_id": null,
      "children_ids": ["child-id"],
      "function_block": {...},
      "state": "completed",
      "level": 0
    }
  }
}
```

### node_info.json
```json
{
  "id": "node-uuid",
  "name": "function_block_name",
  "type": "python",
  "parent_id": "parent-uuid or null",
  "children_ids": ["child-uuid"],
  "state": "completed|running|failed|pending",
  "created_at": "2025-08-02T10:00:00",
  "last_execution": "2025-08-02T10:05:00",
  "execution_count": 1,
  "debug_attempts": 0
}
```

### execution_summary.json
```json
{
  "job_id": "job-uuid",
  "node_id": "node-uuid",
  "start_time": "2025-08-02T10:00:00",
  "end_time": "2025-08-02T10:05:00",
  "duration_seconds": 300,
  "exit_code": 0,
  "state": "success|failed",
  "input_path": "/path/to/input",
  "output_path": "/path/to/output",
  "container_id": "docker-container-id",
  "error_message": null
}
```

## Key Principles

1. **Flat Node Structure**: All nodes are stored at the same level in the `nodes/` folder, avoiding recursive directory structures.

2. **Separation of Concerns**: 
   - Tree structure in JSON files
   - Node execution in jobs folders
   - Agent activities in agent_tasks folders

3. **Traceability**: Every action is timestamped and linked to its source (agent task, job execution, etc.)

4. **Idempotency**: Re-running a node creates a new job folder while preserving history.

5. **Self-Contained Nodes**: Each node folder contains everything needed to understand and reproduce its execution.

## Usage Examples

### Finding a Node's Output
```bash
# Latest output for a node
tree_TREEID/nodes/node_NODEID/outputs/output_data.h5ad

# Or via latest job link
tree_TREEID/nodes/node_NODEID/jobs/latest/output/output_data.h5ad
```

### Tracking Agent Activities
```bash
# Node-specific agent tasks
tree_TREEID/nodes/node_NODEID/agent_tasks/creator_*/

# Tree-level orchestration
tree_TREEID/agent_tasks/orchestrator_*/
```

### Debugging Failed Nodes
```bash
# Check error logs
tree_TREEID/nodes/node_NODEID/jobs/latest/logs/stderr.txt

# Review fixer attempts
tree_TREEID/nodes/node_NODEID/agent_tasks/fixer_*/
```

## Migration Notes

When migrating from the old structure:
1. Create the `nodes/` folder structure
2. Move node-specific data from scattered locations
3. Update symbolic links and references
4. Preserve job history in new format