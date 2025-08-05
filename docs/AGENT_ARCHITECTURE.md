# RagOmics Agent Architecture Documentation

## Overview

The RagOmics Local Agent system employs a sophisticated multi-agent architecture designed for automated single-cell RNA sequencing analysis. The system orchestrates complex bioinformatics workflows through a hierarchical analysis tree structure, where each node represents an independent analysis step that can be executed in isolation or in parallel with other nodes.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Agent Descriptions](#agent-descriptions)
3. [Input/Output Specifications](#inputoutput-specifications)
4. [Analysis Tree Integration](#analysis-tree-integration)
5. [Agent Communication](#agent-communication)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Directory Structure](#directory-structure)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                          │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Main Agent                            │
│                  (System Orchestrator)                       │
│  • Initializes all agents                                    │
│  • Manages analysis tree                                     │
│  • Coordinates execution                                     │
└─────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Orchestrator │    │Function Selector │    │  Bug Fixer   │
│    Agent     │◄───│     Agent        │    │    Agent     │
│              │    │                  │    │              │
│ • Planning   │    │ • Selection      │    │ • Debugging  │
│ • Parallel   │    │ • Coordination   │    │ • Fixing     │
│   execution  │    │                  │    │              │
└──────────────┘    └──────────────────┘    └──────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Function Creator │
                    │      Agent       │
                    │                  │
                    │ • Code generation│
                    │ • Requirements   │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Agent Output    │
                    │     Utils        │
                    │                  │
                    │ • Logging        │
                    │ • Tracking      │
                    └──────────────────┘
```

## Agent Descriptions

### 1. Main Agent (`agents/main_agent.py`)

**Purpose**: Primary system orchestrator that coordinates all other agents and manages the overall analysis workflow.

**Key Responsibilities**:
- Initialize and coordinate all specialized agents
- Create and manage the analysis tree structure
- Execute nodes through the NodeExecutor
- Handle node failures through bug fixing
- Implement iterative tree expansion with safeguards
- Manage output directory structure and logging

**Key Methods**:
- `analyze()`: Main entry point for analysis
- `_generate_and_execute_plan()`: Iterative tree expansion and execution
- `_execute_single_node()`: Execute individual nodes
- `_attempt_fix()`: Coordinate bug fixing for failed nodes
- `_log_function_creation_to_node()`: Log agent activities

### 2. Orchestrator Agent (`agents/orchestrator_agent.py`)

**Purpose**: Strategic planning and workflow orchestration with parallel execution capabilities.

**Key Responsibilities**:
- Plan analysis workflows based on user requests
- Manage parallel execution of independent nodes
- Make tree expansion decisions
- Coordinate between function selection and execution
- Handle reactive tree expansion based on completion

**Key Methods**:
- `plan_next_steps()`: Generate analysis plan
- `execute_tree()`: Execute tree with parallelism
- `_can_expand_from_node()`: Determine expansion eligibility
- `_execute_parallel_nodes()`: Parallel execution coordination

### 3. Function Creator Agent (`agents/function_creator_agent.py`)

**Purpose**: Generate new function blocks for specific analysis tasks.

**Key Responsibilities**:
- Generate Python/R code using LLM
- Implement standardized function block framework
- Create comprehensive error handling
- Generate requirements and configuration
- Specialize in single-cell analysis patterns

**Key Methods**:
- `create_function()`: Generate new function block
- `_build_creation_prompt()`: Construct LLM prompt
- `_parse_llm_response()`: Parse generated code
- `_validate_function_block()`: Validate generated code

**Function Block Signature**:
```python
def run(path_dict, params):
    """
    Standard function block interface.
    
    Args:
        path_dict: Dictionary with 'input_dir' and 'output_dir' paths
        params: Dictionary of analysis parameters
    
    Returns:
        Processed data object
    """
```

### 4. Function Selector Agent (`agents/function_selector_agent.py`)

**Purpose**: Select appropriate function blocks or coordinate creation of new ones.

**Key Responsibilities**:
- Analyze analysis progress
- Recommend appropriate function blocks
- Interface with Function Creator Agent
- Manage generation modes
- Make satisfaction decisions

**Key Methods**:
- `select_next_functions()`: Select/create function blocks
- `_build_selection_prompt()`: Create selection prompt
- `_process_selection()`: Process LLM selection
- `_create_new_function()`: Coordinate new function creation

### 5. Bug Fixer Agent (`agents/bug_fixer_agent.py`)

**Purpose**: Automatically debug and fix failed function blocks.

**Key Responsibilities**:
- Pattern-based common error fixes
- LLM-based complex debugging
- Handle import and attribute errors
- Manage multiple fix attempts
- Track fix versions

**Key Methods**:
- `fix_error()`: Main error fixing entry point
- `_apply_common_fixes()`: Pattern-based fixes
- `_fix_with_llm()`: LLM-based debugging
- `_extract_error_info()`: Parse error details

**Common Fix Patterns**:
- Missing module imports
- Incorrect function signatures
- Data type mismatches
- Plotting configuration issues
- File path errors

### 6. Agent Output Utils (`agents/agent_output_utils.py`)

**Purpose**: Centralized logging and output management for all agents.

**Key Responsibilities**:
- Log LLM interactions
- Track bug fix attempts
- Record function selections
- Create structured outputs
- Generate task summaries

**Key Functions**:
- `log_agent_task()`: Log agent activities
- `log_bug_fix_attempt()`: Track debugging attempts
- `log_function_selection()`: Record selection process
- `get_task_summary()`: Generate activity summaries

## Input/Output Specifications

### Main Agent

**Inputs**:
```python
{
    "user_request": str,           # Natural language analysis request
    "input_data_path": Path,        # Path to input data (H5AD/RDS)
    "output_dir": Path,             # Output directory path
    "llm_model": str,               # LLM model to use
    "generation_mode": str,         # "new_only", "existing_only", "mixed"
    "max_iterations": int,          # Maximum planning iterations
    "verbose": bool                 # Enable verbose logging
}
```

**Outputs**:
```python
{
    "tree_id": str,                 # Unique tree identifier
    "output_dir": str,              # Output directory path
    "tree_file": str,               # Analysis tree JSON path
    "total_nodes": int,             # Total nodes created
    "completed_nodes": int,         # Successfully executed nodes
    "failed_nodes": int,            # Failed nodes
    "results": Dict[str, Any]       # Node execution results
}
```

### Orchestrator Agent

**Inputs**:
```python
{
    "user_request": str,            # Analysis request
    "tree_state": Dict,             # Current tree state
    "iteration": int,               # Current iteration
    "parent_node": Dict,            # Parent node info (if expanding)
    "phase": str                    # "root" or "expansion"
}
```

**Outputs**:
```python
{
    "satisfied": bool,              # User request satisfied
    "recommendations": List[Dict],  # Recommended function blocks
    "reasoning": str,               # Decision reasoning
    "parallel_groups": List[List]   # Nodes for parallel execution
}
```

### Function Creator Agent

**Inputs**:
```python
{
    "task_description": str,        # Task to implement
    "user_request": str,            # Original request
    "parent_output": Dict,          # Parent node output info
    "data_summary": Dict,           # Current data state
    "language": str                 # "python" or "r"
}
```

**Outputs**:
```python
NewFunctionBlock(
    name: str,                      # Function block name
    type: FunctionBlockType,        # PYTHON or R
    description: str,               # Block description
    code: str,                      # Generated code
    requirements: str,              # Package requirements
    parameters: Dict[str, Any],     # Default parameters
    static_config: StaticConfig     # Configuration
)
```

### Function Selector Agent

**Inputs**:
```python
{
    "user_request": str,            # Analysis request
    "current_node": AnalysisNode,   # Current node (optional)
    "parent_nodes": List[Node],     # Parent nodes
    "data_summary": Dict,           # Data state
    "generation_mode": str,         # Generation mode
    "max_branches": int             # Max functions to select
}
```

**Outputs**:
```python
{
    "satisfied": bool,              # Request satisfied
    "function_blocks": List[Block], # Selected/created blocks
    "reasoning": str,               # Selection reasoning
    "is_terminal": bool            # Terminal node indicator
}
```

### Bug Fixer Agent

**Inputs**:
```python
{
    "node_id": str,                 # Failed node ID
    "error": str,                   # Error message
    "function_block": FunctionBlock,# Failed block
    "stdout": str,                  # Standard output
    "stderr": str,                  # Standard error
    "node_dir": Path               # Node directory path
}
```

**Outputs**:
```python
{
    "success": bool,                # Fix successful
    "fixed_code": str,             # Fixed code
    "fixed_requirements": str,      # Updated requirements
    "fix_description": str,         # What was fixed
    "attempt": int                 # Attempt number
}
```

## Analysis Tree Integration

### Tree Structure

```
AnalysisTree
├── id: UUID
├── user_request: str
├── nodes: Dict[str, AnalysisNode]
│   ├── Node 1 (root)
│   │   ├── id: UUID
│   │   ├── function_block: FunctionBlock
│   │   ├── state: NodeState
│   │   ├── parent_id: None
│   │   └── children: [Node 2 ID]
│   └── Node 2 (child)
│       ├── id: UUID
│       ├── function_block: FunctionBlock
│       ├── state: NodeState
│       ├── parent_id: Node 1 ID
│       └── children: []
├── max_nodes: int
├── max_branches: int
└── max_debug_trials: int
```

### Node States

- **PENDING**: Node created but not executed
- **RUNNING**: Currently executing
- **COMPLETED**: Successfully executed
- **FAILED**: Execution failed
- **DEBUGGING**: Being fixed by Bug Fixer

### Tree Expansion Strategy

1. **Root Creation**: Main Agent creates root node based on initial request
2. **Execution**: Nodes execute in dependency order
3. **Expansion Decision**: Orchestrator decides if expansion needed
4. **Child Creation**: Function Selector/Creator generate children
5. **Iterative Process**: Repeat until satisfied or limits reached

## Agent Communication

### Communication Patterns

1. **Direct Method Calls**: Agents call each other's methods directly
2. **Shared Data Structures**: Analysis tree shared between agents
3. **Event Callbacks**: Completion callbacks for reactive expansion
4. **Logging Framework**: Centralized activity tracking

### Data Exchange Format

```python
# Standard context dictionary
context = {
    "user_request": str,
    "tree": AnalysisTree,
    "current_node": AnalysisNode,
    "data_summary": Dict,
    "execution_state": Dict,
    "agent_logs": List[Dict]
}
```

## Error Handling and Recovery

### Error Detection

1. **Execution Monitoring**: Track node execution status
2. **Output Validation**: Verify expected outputs exist
3. **Log Analysis**: Parse stdout/stderr for errors
4. **Timeout Detection**: Handle long-running nodes

### Recovery Strategies

1. **Pattern-Based Fixes**: Common error patterns
   - Import errors → Add missing imports
   - Attribute errors → Fix method calls
   - Type errors → Correct data types
   - Path errors → Fix file paths

2. **LLM-Based Debugging**: Complex errors
   - Send error context to LLM
   - Generate fix based on understanding
   - Validate fix before applying

3. **Retry Logic**:
   - Maximum attempts per node (configurable)
   - Version tracking for each attempt
   - Fallback to simpler implementation

### Fallback Mechanism

When LLM services are unavailable, MainAgent provides a default pipeline:

1. **Automatic Detection**: If no LLM service is configured
2. **Default Pipeline Creation**: 
   - Basic preprocessing (filtering, normalization)
   - Standard analysis (PCA, clustering)
3. **Execution**: Uses the same execution framework
4. **Output**: Results saved in standard format

## Directory Structure

### Output Organization

```
output_dir/
├── analysis_tree.json              # Complete tree structure
├── DIRECTORY_TREE.md               # Visual tree representation
├── tree_UUID/                      # Tree-specific directory
│   ├── main_task/                  # Main agent logs
│   │   ├── agent_info.json
│   │   └── orchestrator_tasks/
│   └── nodes/                      # Flat node structure
│       ├── node_UUID1/             # Individual node
│       │   ├── node_info.json     # Node metadata
│       │   ├── function_block/    # Code and config
│       │   │   ├── code.py
│       │   │   ├── config.json
│       │   │   └── requirements.txt
│       │   ├── jobs/              # Execution attempts
│       │   │   ├── job_TIMESTAMP_{id}/
│       │   │   │   ├── execution_summary.json
│       │   │   │   ├── input/
│       │   │   │   ├── output/
│       │   │   │   │   ├── _node_anndata.h5ad
│       │   │   │   │   └── figures/
│       │   │   │   └── logs/
│       │   │   │       ├── stdout.txt
│       │   │   │       └── stderr.txt
│       │   │   └── latest -> job_TIMESTAMP_{id}
│       │   ├── outputs/           # Final outputs
│       │   └── agent_tasks/       # Agent-specific logs
│       │       ├── creator_task_1.json
│       │       ├── selector_task_1.json
│       │       └── fixer_task_1/
│       └── node_UUID2/
```

### Log Files

**Agent Activity Logs** (`agent_tasks/`):
- LLM prompts and responses
- Function selection reasoning
- Bug fix attempts and versions
- Task summaries and statistics

**Execution Logs** (`jobs/`):
- stdout.txt: Standard output
- stderr.txt: Standard error
- job_metrics.csv: Performance metrics
- job_info.json: Execution metadata

## Best Practices

### Agent Development

1. **Clear Interfaces**: Define explicit input/output contracts
2. **Error Handling**: Comprehensive try-catch blocks
3. **Logging**: Detailed activity logging for debugging
4. **Validation**: Validate inputs and outputs
5. **Documentation**: Clear docstrings and comments

### System Configuration

1. **Timeout Settings**: Configure appropriate timeouts
2. **Retry Limits**: Set reasonable retry attempts
3. **Resource Limits**: Control parallel execution
4. **Model Selection**: Choose appropriate LLM models
5. **Logging Levels**: Configure verbosity as needed

### Performance Optimization

1. **Parallel Execution**: Leverage job pools for independent nodes
2. **Caching**: Cache LLM responses when possible
3. **Resource Management**: Clean up temporary files
4. **Batch Operations**: Group related operations
5. **Early Termination**: Stop on satisfaction

## Future Enhancements

### Planned Features

1. **Agent Specialization**: Domain-specific agents
2. **Learning System**: Learn from successful patterns
3. **Distributed Execution**: Multi-machine support
4. **Interactive Mode**: User intervention points
5. **Visualization**: Real-time execution monitoring

### Extension Points

1. **Custom Agents**: Add specialized analysis agents
2. **New Languages**: Support additional languages
3. **External Tools**: Integrate third-party tools
4. **Cloud Services**: Add cloud execution backends
5. **API Integration**: REST API for remote access