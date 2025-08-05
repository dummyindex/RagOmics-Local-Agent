"""Agent responsible for creating new function blocks with proper implementation."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from .agent_output_utils import AgentOutputLogger
from ..models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg,
    InputSpecification, OutputSpecification, FileInfo, FileType
)
from ..llm_service import OpenAIService
from ..utils import setup_logger

logger = setup_logger(__name__)


class FunctionCreatorAgent(BaseAgent):
    """Agent that creates new function blocks based on analysis requirements."""
    
    # Detailed documentation for function block implementation
    FUNCTION_BLOCK_DOCUMENTATION = """
# Function Block Implementation Guide - SIMPLIFIED FRAMEWORK

## Required Structure
Every function block MUST have a function named 'run' that takes path_dict and params:

```python
def run(path_dict, params):
    '''
    Main entry point for the function block.
    
    Args:
        path_dict: Dictionary containing paths:
            - input_dir: Input directory path
            - output_dir: Output directory path
        params: Dictionary of parameters for this block
    
    Returns:
        None (outputs are written to files)
    '''
    # Your implementation here
```

## CRITICAL INPUT/OUTPUT CONVENTIONS

### Input Data:
- Function blocks receive path_dict and params arguments
- Construct input path using path_dict["input_dir"]
- For anndata workflows, use standard naming:
```python
import scanpy as sc
import os

# Construct and read input data - REQUIRED pattern
input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")

if os.path.exists(input_file):
    adata = sc.read_h5ad(input_file)
else:
    # Handle missing input appropriately
    raise FileNotFoundError(f"Input file not found: {input_file}")
```

### Output Data:
- Construct output path using path_dict["output_dir"]
- Use standard naming for data passing between nodes
```python
# Save output data - REQUIRED pattern
output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
adata.write(output_file)
print(f"Output saved to {output_file}")
```

### Additional Files:
- Figures: Save to `os.path.join(path_dict["output_dir"], "figures")`
- Metrics/Reports: Save to `path_dict["output_dir"]`

### Scanpy Figure Saving:
When using scanpy plotting functions, configure the figure directory properly:
```python
import scanpy as sc
from pathlib import Path

# Set scanpy figure directory - CRITICAL for saving plots
figures_dir = Path(path_dict["output_dir"]) / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = figures_dir

# Now plots will save to the correct location
sc.pl.umap(adata, color='leiden', save='_leiden.png', show=False)
# This saves to: figures_dir / 'umap_leiden.png'
```
- Logs: Print to stdout (will be captured)

## Directory Structure
- `path_dict["input_dir"]`: Input directory (read-only)
  - Contains `_node_anndata.h5ad`: Primary input from previous node
  - Other files if specified in input specification
- `path_dict["output_dir"]`: Output directory (write here)
  - Must create `_node_anndata.h5ad`: REQUIRED output for next node
  - `figures/`: Directory for plots
  - Other output files as needed

## Common Patterns

### 1. Quality Control Function Block
```python
def run(path_dict, params):
    import scanpy as sc
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Read input
    adata = sc.read_h5ad(input_file)
    
    # Get parameters with defaults
    min_genes = params.get('min_genes', 200)
    min_cells = params.get('min_cells', 3)
    
    # Apply filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Filtered to {adata.shape[0]} cells and {adata.shape[1]} genes")
    
    # Save output - REQUIRED
    adata.write(output_file)
    print(f"Output saved to {output_file}")
```

### 2. Clustering with Metrics
```python
def run(path_dict, params):
    import scanpy as sc
    import pandas as pd
    from sklearn import metrics
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Read input
    adata = sc.read_h5ad(input_file)
    
    # Get parameters
    resolution = params.get('resolution', 1.0)
    ground_truth_key = params.get('ground_truth_key', 'cell_type')
    
    # Run clustering - choose appropriate method and library
    sc.tl.leiden(adata, resolution=resolution)
    
    # Calculate metrics if ground truth exists
    if ground_truth_key in adata.obs.columns:
        ari = metrics.adjusted_rand_score(
            adata.obs[ground_truth_key], 
            adata.obs['leiden']
        )
        adata.uns['clustering_metrics'] = {'ARI': ari}
        print(f"Clustering ARI: {ari:.3f}")
        
        # Save metrics to file
        metrics_df = pd.DataFrame({'metric': ['ARI'], 'value': [ari]})
        metrics_file = os.path.join(path_dict["output_dir"], 'clustering_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
    
    # Save output
    adata.write(output_file)
    print(f"Output saved to {output_file}")
```

### 3. Common Scanpy Pitfalls to Avoid

**CRITICAL: Scanpy plotting API requirements:**
- `sc.pl.scatter`, `sc.pl.umap`, `sc.pl.pca` expect column names (strings), NOT numpy arrays or pandas Series
- The `color` parameter must be a string representing a column name in `.obs` or `.var`
- To save figures properly, always set `sc.settings.figdir` first

**Incorrect (will fail):**
```python
# WRONG - passing Series/array instead of column name
labels = adata.obs['kmeans']  
sc.pl.umap(adata, color=labels, ...)  # ERROR!

# WRONG - passing numpy array from obsm
sc.pl.scatter(adata, x=adata.obsm['X_pca'][:, 0], ...)  # ERROR!
```

**Correct:**
```python
# RIGHT - pass column name as string
sc.pl.umap(adata, color='kmeans', ...)  

# RIGHT - for PCA visualization
sc.pl.pca(adata, color='leiden', ...)
```

### 4. Saving Figures
```python
def run(path_dict, params):
    import scanpy as sc
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Read input
    adata = sc.read_h5ad(input_file)
    
    # Create figures directory
    figures_dir = Path(path_dict["output_dir"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save UMAP plot
    if 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='leiden' if 'leiden' in adata.obs.columns else None, 
                   show=False, save='_clusters.png')
        print(f"Saved UMAP figure")
    
    # Save output
    adata.write(output_file)
    print(f"Output saved to {output_file}")
```

## Important Requirements
1. ALWAYS name the main function 'run' with path_dict and params arguments
2. params is passed directly as a dictionary - no need to load from file
3. ALWAYS construct input path: os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
4. ALWAYS construct output path: os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
5. Include all necessary imports inside the function
6. Use parameters from params dict with appropriate defaults
7. Print informative messages about what was done
8. Handle errors gracefully with try/except blocks
9. ALWAYS create meaningful visualizations and save to os.path.join(path_dict["output_dir"], "figures"):
   - Quality control: histogram of gene counts, cell counts distribution
   - PCA: variance ratio plot, PCA scatter plots
   - UMAP/tSNE: embedding plots with different colors
   - Clustering: cluster visualization, silhouette scores, metrics
   - Use matplotlib.pyplot with Agg backend: plt.switch_backend('Agg')
   - Always close plots: plt.close() after saving
10. Check if input files exist before reading
"""

    SYSTEM_PROMPT = """You are an expert bioinformatics function block creator specializing in single-cell RNA sequencing analysis.

Your task is to create complete, working function blocks that follow the framework conventions for processing single-cell data.

CRITICAL REQUIREMENTS:
1. Function signature: def run(path_dict, params)
2. Input/output paths use path_dict["input_dir"] and path_dict["output_dir"]
3. Read input anndata from "_node_anndata.h5ad" file
4. Save output anndata to "_node_anndata.h5ad" file
5. Create figures in "figures" subdirectory
6. Handle parameters appropriately from params argument
7. Return the processed adata object

IMPORTANT CODING GUIDELINES:
- Use proper string formatting - avoid complex f-strings with nested quotes/brackets
- For complex paths, use variables: figures_dir = Path(path_dict["output_dir"]) / "figures"
- Check if required data exists before using it (e.g., check for 'X_pca' before using it)
- Always create the figures directory before saving figures to it
- Handle missing ground truth data gracefully with informative messages

You are experienced with bioinformatics libraries and will create appropriate implementations based on the task requirements.

Always ensure your code is production-ready with proper error handling, informative output, and meaningful visualizations."""

    def __init__(self, llm_service: Optional[OpenAIService] = None):
        super().__init__("function_creator")
        self.llm_service = llm_service
        self.logger = logger
    
    def process(self, context: Dict[str, Any]) -> Optional[NewFunctionBlock]:
        """Create a new function block based on requirements.
        
        Required context keys:
            - task_description: str - What the function should do
            - parent_output: Optional[Dict] - Output from parent node
            - user_request: str - Original user request
            - generation_mode: str - Generation mode
            
        Optional context keys:
            - input_specification: InputSpecification - Required input files
            - output_specification: OutputSpecification - Expected outputs
            - parameters: Dict - Default parameters
            - node_dir: Path - Node directory for logging (nodes/node_xxx)
            
        Returns:
            NewFunctionBlock or None if creation fails
        """
        self.validate_context(context, ['task_description', 'user_request'])
        
        if not self.llm_service:
            self.logger.error("No LLM service available for function creation")
            return None
        
        # Initialize agent logger if node_dir is provided
        agent_logger = None
        if 'node_dir' in context:
            # This is node-specific logging
            agent_logger = AgentOutputLogger(context['node_dir'], 'function_creator')
        
        try:
            # Build the prompt
            prompt = self._build_creation_prompt(context)
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.FUNCTION_BLOCK_DOCUMENTATION},
                {"role": "user", "content": prompt}
            ]
            
            # Define response schema for structured JSON output
            schema = {
                "name": "function_block_creation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "function_block": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Function block name (snake_case)"},
                                "description": {"type": "string", "description": "Brief description"},
                                "code": {"type": "string", "description": "Complete Python code with def run(path_dict, params)"},
                                "requirements": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of package requirements - ONLY external packages like scanpy, pandas, NOT built-in modules like os, sys, pathlib"
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Default parameter values as key-value pairs",
                                    "additionalProperties": True
                                },
                                "static_config": {
                                    "type": "object",
                                    "properties": {
                                        "args": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "value_type": {"type": "string", "enum": ["int", "float", "str", "bool", "list", "dict"]},
                                                    "description": {"type": "string"},
                                                    "optional": {"type": "boolean"},
                                                    "default_value": {}
                                                },
                                                "required": ["name", "value_type", "description", "optional"]
                                            }
                                        },
                                        "description": {"type": "string"},
                                        "tag": {"type": "string", "enum": ["quality_control", "normalization", "clustering", "visualization", "analysis", "integration"]}
                                    },
                                    "required": ["args", "description", "tag"]
                                }
                            },
                            "required": ["name", "description", "code", "requirements", "parameters", "static_config"]
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of the approach and implementation choices"
                        }
                    },
                    "required": ["function_block", "reasoning"]
                }
            }
            
            # Prepare LLM input for logging
            llm_input = {
                'messages': messages,
                'schema': schema,
                'temperature': 0.3,
                'max_tokens': 4000,
                'model': self.llm_service.model,
                'timestamp': datetime.now().isoformat()
            }
            
            # Call LLM
            self.logger.info(f"Creating function block with {self.llm_service.model}")
            try:
                result = self.llm_service.chat_completion_json(
                    messages=messages,
                    json_schema=schema,
                    temperature=0.3,
                    max_tokens=4000
                )
                self.logger.debug(f"LLM response type: {type(result)}")
                self.logger.debug(f"LLM response: {result}")
            except Exception as e:
                self.logger.error(f"Error calling LLM: {e}")
                # Return None instead of raising to allow graceful failure
                if agent_logger:
                    agent_logger.log_llm_interaction(
                        task_type='create_function',
                        llm_input=llm_input,
                        llm_output=None,
                        error=str(e),
                        metadata={'context': context}
                    )
                return None
            
            # Log LLM interaction if agent_logger available
            if agent_logger:
                agent_logger.log_llm_interaction(
                    task_type='create_function',
                    llm_input=llm_input,
                    llm_output=result,
                    metadata={
                        'task_description': context.get('task_description'),
                        'parent_output': context.get('parent_output')
                    }
                )
            
            # Extract function_block from nested structure
            fb_data = result.get('function_block', result)  # Handle both nested and flat formats
            
            # Convert to NewFunctionBlock
            try:
                function_block = self._create_function_block(fb_data, context)
            except Exception as e:
                self.logger.error(f"Error in _create_function_block: {e}")
                self.logger.error(f"fb_data: {fb_data}")
                raise
            
            # Save function block code version if agent_logger available
            if function_block and agent_logger:
                agent_logger.save_function_block_versions(
                    original_code=function_block.code,
                    fixed_code=None,
                    version=1
                )
            
            return function_block
            
        except Exception as e:
            self.logger.error(f"Error creating function block: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Log error if agent_logger available
            if agent_logger:
                agent_logger.log_llm_interaction(
                    task_type='create_function',
                    llm_input=llm_input if 'llm_input' in locals() else None,
                    llm_output=None,
                    error=str(e),
                    metadata={'context': context}
                )
            
            return None
    
    def _build_creation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for function block creation."""
        parts = []
        
        parts.append("## Task")
        parts.append(f"Create a function block for: {context['task_description']}")
        
        parts.append("\n## Context")
        parts.append(f"User Request: {context['user_request']}")
        
        # Add parent output structure if available
        if 'parent_output_structure' in context:
            parts.append("\n## Input Data Structure (from previous processing step)")
            parent_struct = context['parent_output_structure']
            parts.append(f"Data shape: {parent_struct.get('shape', 'Unknown')}")
            
            if parent_struct.get('obs_columns'):
                parts.append(f"Available observation columns: {', '.join(parent_struct['obs_columns'])}")
            
            if parent_struct.get('obsm_keys'):
                parts.append(f"Available embeddings: {', '.join(parent_struct['obsm_keys'])}")
            
            if parent_struct.get('uns_keys'):
                parts.append(f"Available unstructured data: {', '.join(parent_struct['uns_keys'])}")
        
        if 'parent_output' in context:
            parts.append(f"\nParent Node Output: {context['parent_output']}")
        
        if 'input_specification' in context:
            spec = context['input_specification']
            parts.append("\n## Required Input Files")
            for file_info in spec.required_files:
                parts.append(f"- {file_info['name']}: {file_info['description']}")
        
        if 'parameters' in context:
            parts.append("\n## Suggested Parameters")
            for key, value in context['parameters'].items():
                parts.append(f"- {key}: {value}")
        
        parts.append("\n## Function Structure")
        parts.append("Your function should:")
        parts.append("1. Define function with signature: def run(path_dict, params)")
        parts.append("2. Read input data from path_dict['input_dir']")
        parts.append("3. Process the data according to the task")
        parts.append("4. Save outputs to path_dict['output_dir']")
        parts.append("5. Create visualizations if appropriate")
        parts.append("6. Print informative messages about the processing")
        parts.append("7. Return the processed data")
        
        parts.append("\n## Parameter Handling")
        parts.append("The params argument may contain nested dictionaries with metadata.")
        parts.append("To extract parameter values safely, define and use this helper function:")
        parts.append("```python")
        parts.append("def get_param(params, key, default):")
        parts.append("    val = params.get(key, default)")
        parts.append("    if isinstance(val, dict) and 'default_value' in val:")
        parts.append("        return val.get('default_value', default)")
        parts.append("    return val if val is not None else default")
        parts.append("```")
        parts.append("Then use it like: min_genes = get_param(params, 'min_genes', 200)")
        
        parts.append("\n## Requirements")
        parts.append("List all necessary Python packages in requirements.txt")
        parts.append("Only include external packages that need to be installed via pip")
        parts.append("Do not include built-in Python modules")
        
        return "\n".join(parts)
    
    def _create_function_block(self, result: Dict[str, Any], context: Dict[str, Any]) -> NewFunctionBlock:
        """Convert LLM result to NewFunctionBlock."""
        
        # Parse static config - handle if it's missing
        static_config_data = result.get('static_config', {})
        
        # Ensure we have required fields
        if not static_config_data:
            static_config_data = {
                'args': [],
                'description': result.get('description', 'Generated function block'),
                'tag': 'analysis'
            }
        
        args = []
        for arg_data in static_config_data.get('args', []):
            try:
                args.append(Arg(
                    name=arg_data['name'],
                    value_type=arg_data['value_type'],
                    description=arg_data['description'],
                    optional=arg_data.get('optional', True),
                    default_value=arg_data.get('default_value')
                ))
            except Exception as e:
                self.logger.error(f"Error creating Arg: {e}")
                self.logger.error(f"arg_data: {arg_data}")
                raise
        
        static_config = StaticConfig(
            args=args,
            description=static_config_data.get('description', result.get('description', 'Generated function block')),
            tag=static_config_data.get('tag', 'analysis')
        )
        
        # Add input/output specifications if provided
        if 'input_specification' in context:
            static_config.input_specification = context['input_specification']
        if 'output_specification' in context:
            static_config.output_specification = context['output_specification']
        
        # Determine language from code
        code = result['code']
        fb_type = FunctionBlockType.R if 'library(' in code or '<-' in code else FunctionBlockType.PYTHON
        
        # Handle requirements as either array or string
        requirements = result.get('requirements', '')
        if isinstance(requirements, list):
            requirements = '\n'.join(requirements)
        
        return NewFunctionBlock(
            name=result['name'],
            type=fb_type,
            description=result['description'],
            code=code,
            requirements=requirements,
            parameters=result.get('parameters', {}),
            static_config=static_config
        )
    
    def create_function_block(self, specification: Dict[str, Any]) -> Optional[NewFunctionBlock]:
        """Create a function block from specification.
        
        Args:
            specification: Dictionary with name, description, task, etc.
            
        Returns:
            NewFunctionBlock if successful, None otherwise
        """
        # Convert specification to context format expected by process method
        context = {
            "task_description": specification.get("task", specification.get("description", "")),
            "user_request": specification.get("user_request", specification.get("task", specification.get("description", ""))),
            "function_name": specification.get("name", "unknown_function"),
            "description": specification.get("description", ""),
            "requirements": specification.get("requirements", ""),
            "parameters": specification.get("parameters", {}),
            "input_type": "adata",
            "output_type": "adata"
        }
        
        # Pass through parent output structure if available
        if "parent_output_structure" in specification:
            context["parent_output_structure"] = specification["parent_output_structure"]
        
        return self.process(context)