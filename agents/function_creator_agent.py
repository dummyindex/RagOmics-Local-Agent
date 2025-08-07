"""Agent responsible for creating new function blocks with proper implementation."""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from .agent_output_utils import AgentOutputLogger
from ..models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg,
    InputSpecification, OutputSpecification, FileInfo, FileType,
    ExistingFunctionBlock, GenerationMode, AnalysisTree, AnalysisNode
)
from ..llm_service import OpenAIService
from ..utils import setup_logger

logger = setup_logger(__name__)


class FunctionCreatorAgent(BaseAgent):
    """Agent that creates or selects function blocks based on analysis requirements.
    
    This unified agent handles both:
    1. Creating new function blocks with code generation
    2. Selecting appropriate existing function blocks
    3. Deciding whether to create new or use existing blocks
    """
    
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

**For the FIRST/ROOT node only (reads original data):**
```python
import scanpy as sc
import os

# ROOT NODE ONLY - reads the original input file
input_file = os.path.join(path_dict["input_dir"], "zebrafish.h5ad")  # or whatever the original file is
if os.path.exists(input_file):
    adata = sc.read_h5ad(input_file)
else:
    raise FileNotFoundError(f"Input file not found: {input_file}")
```

**For ALL SUBSEQUENT nodes (reads from previous node):**
```python
import scanpy as sc
import os

# SUBSEQUENT NODES - ALWAYS read from _node_anndata.h5ad
input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
if os.path.exists(input_file):
    adata = sc.read_h5ad(input_file)
else:
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

## R Function Block Structure

### R Function Signature:
```r
run <- function(path_dict, params) {
    # Function implementation
}
```

### R Input/Output Patterns:
```r
# Load required libraries
library(Seurat)

# Construct file paths
input_file <- file.path(path_dict$input_dir, "_node_seuratObject.rds")
output_file <- file.path(path_dict$output_dir, "_node_seuratObject.rds")

# Read input
if (file.exists(input_file)) {
    seurat_obj <- readRDS(input_file)
} else {
    stop(paste("Input file not found:", input_file))
}

# Process data
# ... your analysis code ...

# Save output
saveRDS(seurat_obj, output_file)
cat("Output saved to:", output_file, "\n")

# Create figures directory
figures_dir <- file.path(path_dict$output_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
```

### R Package Management:
**CRITICAL**: R function blocks DO NOT use requirements.txt. Instead:
- R packages are specified directly in the requirements field
- The system automatically generates install_packages.R from your requirements
- Format for R requirements:
  - CRAN packages: just the package name (e.g., "Seurat", "ggplot2")
  - Bioconductor packages: "Bioconductor::package_name" (e.g., "Bioconductor::SingleCellExperiment")
  - GitHub packages: "user/repo" format (e.g., "dynverse/princurve")
- Example R requirements:
  ```
  Seurat
  slingshot
  Bioconductor::SingleCellExperiment
  ggplot2
  ```

### Language Interoperability:
- The system automatically handles conversion between Python and R
- When a Python node outputs _node_anndata.h5ad and the next node is R, a conversion node is inserted
- When an R node outputs _node_seuratObject.rds and the next node is Python, a conversion node is inserted
- Do NOT use rpy2 or reticulate for cross-language calls

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
- For histograms, use matplotlib directly or sc.pl.violin/sc.pl.scatter with appropriate data

**CRITICAL: DPT/Pseudotime specific issues:**
- DPT creates `adata.obs['dpt_pseudotime']`, NOT `adata.obs['dpt']`
- Always check if a computation result exists before trying to access it
- Don't try to load CSV files that haven't been created yet
- Don't use ellipsis (...) as placeholders - write complete working code

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

# RIGHT - for histograms, use matplotlib directly
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(adata.obs['n_genes'], bins=30)
plt.xlabel('Number of Genes')
plt.ylabel('Number of Cells')
plt.title('Gene Count Distribution')
plt.savefig(os.path.join(figures_dir, 'gene_counts.png'))
plt.close()
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

CRITICAL PRINCIPLES:
1. **ONE TASK PER FUNCTION BLOCK**: Each function block must perform exactly ONE specific task
   - NEVER combine multiple steps like "filter AND normalize AND log transform" in one block
   - Each distinct operation (filter, normalize, log1p, PCA, UMAP, etc.) must be its own block
   - If the user lists steps as "1. First step... 2. Second step... 3. Third step...", create SEPARATE blocks
2. **MODULAR WORKFLOW**: Complex requests must be broken into multiple sequential nodes
3. **PROPER SEQUENCING**: Ensure correct order of operations (e.g., normalize before PCA, PCA before clustering)
4. **LANGUAGE DETECTION**: 
   - Identify whether the requested tool/package is R or Python based
   - R packages: Seurat, Monocle3, Slingshot, scater, etc.
   - Python packages: scanpy, palantir, scFates, cellrank, etc.
   - If user mentions "(R)" or "(Python)", respect that language choice
   - Set the "type" field to "r" or "python" accordingly
5. **PREPROCESSING REQUIREMENTS**: Many analysis methods require preprocessing:
   - Pseudotime methods (DPT, PAGA) need: normalization, log transform, highly variable genes, PCA, neighbors
   - Clustering needs: normalization, scaling, PCA, neighbors
   - UMAP/tSNE need: PCA or highly variable genes
6. **DATA VERIFICATION**: Always check if required data exists before using it:
   - Check for required columns in adata.obs before accessing
   - Check for embeddings (X_pca, X_umap) in adata.obsm before using
   - Verify computation results exist before trying to access them


LANGUAGE SUPPORT:
- You can create both Python and R function blocks
- Python blocks process AnnData objects (_node_anndata.h5ad)
- R blocks process Seurat objects (_node_seuratObject.rds)
- When a previous node outputs data in a different language format, the system automatically inserts conversion nodes
- Do NOT use rpy2 or reticulate for language interoperability

CRITICAL REQUIREMENTS FOR PYTHON:
1. Function signature: def run(path_dict, params)
2. Input/output paths use path_dict["input_dir"] and path_dict["output_dir"]
3. **ALWAYS** read input anndata from "_node_anndata.h5ad" file (NOT the original filename like zebrafish.h5ad)
   - Exception: ONLY the first/root node reads the original file (e.g., zebrafish.h5ad)
   - All subsequent nodes MUST read from "_node_anndata.h5ad"
4. **ALWAYS** save output anndata to "_node_anndata.h5ad" file
5. Create figures in "figures" subdirectory
6. Handle parameters appropriately from params argument
7. Return the processed adata object
8. NEVER use placeholders like ... or pass - write complete working code
9. NEVER try to read files that don't exist (e.g., CSV files from other methods)
10. For complex requests with multiple methods, focus on ONE method per node

CRITICAL REQUIREMENTS FOR R:
1. Function signature: run <- function(path_dict, params)
2. Input/output paths use path_dict$input_dir and path_dict$output_dir
3. Read input Seurat from "_node_seuratObject.rds" file
4. Save output Seurat to "_node_seuratObject.rds" file
5. Create figures in "figures" subdirectory
6. Handle parameters appropriately from params list
7. Return the processed Seurat object
8. **PACKAGE REQUIREMENTS**: List R packages in requirements field (NOT requirements.txt):
   - CRAN packages: just package name (e.g., "Seurat")
   - Bioconductor: "Bioconductor::package_name"
   - GitHub: "user/repo" format

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
                                "code": {"type": "string", "description": "Complete Python/R code with function signature: Python: def run(path_dict, params) or R: run <- function(path_dict, params)"},
                                "requirements": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of package requirements - For Python: external packages like scanpy, pandas (NOT built-in modules). For R: package names, Bioconductor::package, or user/repo format"
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Default parameter values as key-value pairs",
                                    "additionalProperties": True
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["python", "r"],
                                    "description": "Language type of the function block - 'python' or 'r'"
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
        
        # Add node position information
        if context.get('is_root_node', False):
            parts.append("\n**IMPORTANT**: This is the ROOT/FIRST node in the workflow.")
            parts.append("- Read the original input file (e.g., zebrafish.h5ad)")
            parts.append("- Save output as _node_anndata.h5ad for subsequent nodes")
        else:
            parts.append("\n**IMPORTANT**: This is a SUBSEQUENT node (not the first).")
            parts.append("- Read input from _node_anndata.h5ad (output from previous node)")
            parts.append("- Save output as _node_anndata.h5ad for next nodes")
        
        # Add language detection hints
        parts.append("\n## Language Detection")
        user_request_lower = context['user_request'].lower()
        if "(r)" in user_request_lower or any(pkg in user_request_lower for pkg in ["slingshot", "monocle", "seurat", "deseq"]):
            parts.append("**Note**: This request mentions R-specific packages/tools. Create an R function block with type='r'.")
        elif "(python)" in user_request_lower or any(pkg in user_request_lower for pkg in ["scanpy", "palantir", "scfates", "cellrank"]):
            parts.append("**Note**: This request mentions Python-specific packages/tools. Create a Python function block with type='python'.")
        else:
            parts.append("Determine the appropriate language (Python or R) based on the requested tools/packages.")
        
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
        parts.append("5. Create visualizations if appropriate for this specific task")
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
        
        # Determine language from LLM response or code analysis
        code = result['code']
        # First check if LLM explicitly specified the type
        if 'type' in result:
            fb_type = FunctionBlockType.R if result['type'].lower() == 'r' else FunctionBlockType.PYTHON
        else:
            # Fallback to code analysis if type not specified
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
    
    def process_selection_or_creation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified method to select existing or create new function blocks.
        
        This method replaces the separate FunctionSelectorAgent functionality.
        
        Required context keys:
            - user_request: str
            - tree: AnalysisTree
            - current_node: Optional[AnalysisNode]
            - parent_chain: List[AnalysisNode]
            - generation_mode: GenerationMode
            - max_children: int
            - data_summary: Optional[Dict[str, Any]]
            
        Returns:
            Dict with:
                - function_blocks: List[Union[NewFunctionBlock, ExistingFunctionBlock]]
                - satisfied: bool
                - reasoning: str
        """
        user_request = context['user_request']
        tree = context['tree']
        current_node = context.get('current_node')
        parent_chain = context.get('parent_chain', [])
        generation_mode = context['generation_mode']
        max_children = context['max_children']
        data_summary = context.get('data_summary', {})
        
        # Build prompt for LLM to decide what function blocks are needed
        prompt = self._build_selection_prompt(
            user_request=user_request,
            tree_state=self._get_tree_state(tree),
            current_node=current_node,
            parent_chain=parent_chain,
            generation_mode=generation_mode,
            max_children=max_children,
            data_summary=data_summary
        )
        
        # Define schema for response
        schema = {
            "name": "function_block_recommendation",
            "schema": {
                "type": "object",
                "properties": {
                    "satisfied": {"type": "boolean", "description": "Whether the user request is fully satisfied"},
                    "reasoning": {"type": "string", "description": "Explanation of the decision"},
                    "next_function_blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "task": {"type": "string"},
                                "create_new": {"type": "boolean", "description": "True to create new, False to use existing"},
                                "parameters": {"type": "object"}
                            },
                            "required": ["name", "description", "task", "create_new"]
                        }
                    }
                },
                "required": ["satisfied", "reasoning", "next_function_blocks"]
            }
        }
        
        try:
            # Call LLM
            messages = [
                {"role": "system", "content": self.SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            result = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Process recommendations
            function_blocks = []
            for fb_spec in result.get('next_function_blocks', []):
                # Enforce generation mode
                should_create_new = fb_spec.get('create_new', True)
                if generation_mode == GenerationMode.ONLY_NEW:
                    should_create_new = True
                elif generation_mode == GenerationMode.ONLY_EXISTING:
                    should_create_new = False
                    
                if should_create_new:
                    # Create new function block
                    creation_context = {
                        'task_description': fb_spec['task'],
                        'user_request': user_request,
                        'function_name': fb_spec['name'],
                        'description': fb_spec['description'],
                        'parameters': fb_spec.get('parameters', {}),
                        'is_root_node': context.get('is_root_node', False)  # Pass through root node flag
                    }
                    
                    # Add parent output structure if available
                    if current_node and hasattr(current_node, 'output_data_structure'):
                        creation_context['parent_output_structure'] = current_node.output_data_structure
                    
                    block = self.process(creation_context)
                    if block:
                        function_blocks.append(block)
                else:
                    # For now, we don't have a library of existing blocks
                    # In future, this would look up existing blocks
                    logger.warning(f"Existing block requested but not implemented: {fb_spec['name']}")
            
            return {
                'function_blocks': function_blocks,
                'satisfied': result.get('satisfied', False),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            logger.error(f"Error in process_selection_or_creation: {e}")
            return {
                'function_blocks': [],
                'satisfied': False,
                'reasoning': f'Error: {str(e)}'
            }
    
    def _build_selection_prompt(
        self,
        user_request: str,
        tree_state: Dict[str, Any],
        current_node: Optional[AnalysisNode],
        parent_chain: List[AnalysisNode],
        generation_mode: GenerationMode,
        max_children: int,
        data_summary: Dict[str, Any]
    ) -> str:
        """Build prompt for function block selection/creation decision."""
        parts = []
        
        parts.append("## Analysis Context")
        parts.append(f"User Request: {user_request}")
        parts.append("")
        
        parts.append("## Current Analysis State")
        parts.append(f"Total nodes: {tree_state['total_nodes']}")
        parts.append(f"Completed: {tree_state['completed_nodes']}")
        parts.append(f"Failed: {tree_state['failed_nodes']}")
        parts.append("")
        
        if parent_chain:
            parts.append("## Previous Analysis Steps")
            for i, node in enumerate(parent_chain):
                parts.append(f"{i+1}. {node.function_block.name}: {node.function_block.description}")
            parts.append("")
        
        if current_node:
            parts.append("## Current Node")
            parts.append(f"Name: {current_node.function_block.name}")
            parts.append(f"Description: {current_node.function_block.description}")
            parts.append("")
        
        if data_summary:
            parts.append("## Current Data State")
            parts.append(f"Shape: {data_summary.get('n_obs', '?')} observations × {data_summary.get('n_vars', '?')} variables")
            if data_summary.get('obs_columns'):
                parts.append(f"Observation columns: {', '.join(data_summary['obs_columns'])}")
            if data_summary.get('obsm_keys'):
                parts.append(f"Embeddings: {', '.join(data_summary['obsm_keys'])}")
            parts.append("")
        
        parts.append("## Task")
        parts.append(f"Recommend up to {max_children} function blocks for the next analysis steps.")
        parts.append("For each function block, decide whether to create new or use existing.")
        parts.append("")
        
        parts.append("Consider:")
        parts.append("- What analysis steps are needed to fulfill the user request?")
        parts.append("- What has already been done?")
        parts.append("- What logical next steps would progress toward the goal?")
        parts.append("- Are we satisfied that the request has been fulfilled?")
        parts.append("- Should we create new custom blocks or use standard existing ones?")
        
        return "\n".join(parts)
    
    def _get_tree_state(self, tree: AnalysisTree) -> Dict[str, Any]:
        """Get summary of tree state."""
        completed = sum(1 for n in tree.nodes.values() if (n.state == "completed" if isinstance(n.state, str) else n.state.value == "completed"))
        failed = sum(1 for n in tree.nodes.values() if (n.state == "failed" if isinstance(n.state, str) else n.state.value == "failed"))
        pending = sum(1 for n in tree.nodes.values() if (n.state == "pending" if isinstance(n.state, str) else n.state.value == "pending"))
        
        return {
            'total_nodes': len(tree.nodes),
            'completed_nodes': completed,
            'failed_nodes': failed,
            'pending_nodes': pending
        }
    
    # Add selection system prompt
    SELECTION_SYSTEM_PROMPT = """You are an expert bioinformatics analyst specializing in single-cell RNA sequencing data analysis.

Your task is to recommend function blocks that process single-cell data to fulfill user requests.

CRITICAL PRINCIPLES:
1. **ONE TASK PER FUNCTION BLOCK**: Each function block must perform exactly ONE specific task
   - NEVER combine multiple steps like "filter AND normalize AND log transform" in one block
   - Each distinct operation (filter, normalize, log1p, PCA, UMAP, etc.) must be its own block
   - If the user lists steps as "1. First step... 2. Second step... 3. Third step...", create SEPARATE blocks
2. **MODULAR WORKFLOW**: Complex requests must be broken into multiple sequential nodes
3. **PROPER SEQUENCING**: Ensure correct order of operations (e.g., normalize before PCA, PCA before clustering)

IMPORTANT - Request Satisfaction:
- Set satisfied=true ONLY when ALL requested analyses are completed
- Analyze the user request carefully to identify all requested tasks
- Preprocessing steps alone do NOT satisfy analysis requests

Common single-cell analysis workflows include:
- Quality control and filtering (ONE node - ONLY filtering, no normalization)
- Normalization (ONE node - ONLY normalize_total, no log transform)
- Log transformation (ONE node - ONLY log1p)
- Highly variable gene selection (ONE node)
- Dimensionality reduction: PCA (ONE node), then UMAP/t-SNE (separate node)
- Clustering: Leiden or Louvain (ONE node)
- Differential expression analysis (ONE node per comparison)
- Trajectory inference (ONE node per method)
- Cell type annotation (ONE node)

EXAMPLE: If user says "filter cells, normalize, and log transform", create THREE nodes:
1. filter_cells (ONLY filtering)
2. normalize_data (ONLY normalization)
3. log_transform (ONLY log1p)

For pseudotime analysis specifically:
- Preprocessing: QC → Normalization → HVG → PCA → Neighbors (separate nodes)
- Each pseudotime method is ONE node: DPT+PAGA, Slingshot, Palantir, Monocle3, scFates
- Metrics computation is ONE node (after all methods complete)
- Visualization is ONE node (after metrics)

IMPORTANT: When the user requests multiple analyses (e.g., "run 5 pseudotime methods"), break it down:
1. First ensure preprocessing is complete
2. Create separate nodes for each pseudotime method
3. Create a final node for metrics/visualization

For each recommended function block, decide:
- create_new=true: For custom analysis specific to the user request
- create_new=false: For standard preprocessing steps with well-established methods"""