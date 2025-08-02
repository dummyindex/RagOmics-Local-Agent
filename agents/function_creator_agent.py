"""Agent responsible for creating new function blocks with proper implementation."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
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
# Function Block Implementation Guide

## Required Structure
Every function block MUST have a function named 'run' with this exact signature:

```python
def run(adata, **parameters):
    '''
    Main entry point for the function block.
    
    Args:
        adata: AnnData object with single-cell data
        **parameters: Additional parameters from static config
    
    Returns:
        adata: Modified AnnData object (or dict with 'adata' key)
    '''
    # Your implementation here
    return adata
```

## Input/Output Handling

### For Single AnnData Input (default):
- Input: adata parameter is the loaded AnnData object
- Output: Return the modified adata object

### For Multiple Input Files:
When your function needs additional input files (CSVs, images, etc.):

1. Declare in static_config:
```python
static_config = {
    "input_specification": {
        "required_files": [
            {"name": "metadata.csv", "type": "csv", "description": "Cell metadata"},
            {"name": "markers.txt", "type": "text", "description": "Gene markers"}
        ]
    }
}
```

2. Access in function:
```python
def run(adata, **parameters):
    # Access additional input files from parameters
    input_dir = parameters.get('input_dir', '/workspace/inputs')
    
    # Read metadata CSV
    import pandas as pd
    metadata_path = Path(input_dir) / 'metadata.csv'
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
    
    # Read markers text file
    markers_path = Path(input_dir) / 'markers.txt'
    if markers_path.exists():
        with open(markers_path) as f:
            markers = f.read().splitlines()
    
    # Your processing logic here
    return adata
```

## Directory Structure Available
- `/workspace/inputs/`: Input files directory
- `/workspace/output/`: Output directory for results
- `/workspace/output/figures/`: Directory for saving figures

## Common Patterns

### 1. Quality Control Function Block
```python
def run(adata, **parameters):
    import scanpy as sc
    
    # Get parameters with defaults
    min_genes = parameters.get('min_genes', 200)
    min_cells = parameters.get('min_cells', 3)
    
    # Apply filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Filtered to {adata.shape[0]} cells and {adata.shape[1]} genes")
    return adata
```

### 2. Clustering with Metrics
```python
def run(adata, **parameters):
    import scanpy as sc
    import pandas as pd
    from sklearn import metrics
    
    # Parameters
    resolution = parameters.get('resolution', 1.0)
    ground_truth_key = parameters.get('ground_truth_key', 'cell_type')
    
    # Run clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    # Calculate metrics if ground truth exists
    if ground_truth_key in adata.obs.columns:
        ari = metrics.adjusted_rand_score(
            adata.obs[ground_truth_key], 
            adata.obs['leiden']
        )
        adata.uns['clustering_metrics'] = {'ARI': ari}
        print(f"Clustering ARI: {ari:.3f}")
    
    return adata
```

### 3. Saving Figures
```python
def run(adata, **parameters):
    import scanpy as sc
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Create figures directory
    figures_dir = Path('/workspace/output/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save UMAP plot
    sc.pl.umap(adata, color='leiden', show=False)
    plt.savefig(figures_dir / 'umap_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to {figures_dir / 'umap_clusters.png'}")
    return adata
```

## Important Requirements
1. ALWAYS name the main function 'run'
2. ALWAYS accept 'adata' as first parameter
3. ALWAYS return adata (or dict with 'adata' key)
4. Include all necessary imports inside the function
5. Use parameters.get() with defaults for optional parameters
6. Print informative messages about what was done
7. Handle errors gracefully with try/except blocks
8. Save figures to /workspace/output/figures/
9. For additional input files, check if they exist before reading
"""

    SYSTEM_PROMPT = """You are an expert bioinformatics function block creator specializing in single-cell RNA sequencing analysis.

Your task is to create complete, working function blocks that process AnnData objects according to user requirements.

CRITICAL: Every function block MUST follow the exact structure documented above. The main function MUST be named 'run' and follow the specified signature.

You have deep knowledge of:
- scanpy for single-cell analysis
- pandas for data manipulation  
- numpy for numerical operations
- matplotlib/seaborn for visualization
- scikit-learn for machine learning
- scipy for statistical analysis

Always ensure your code is production-ready with proper error handling and informative output."""

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
            - task_dir: Path - Directory to save LLM interactions
            
        Returns:
            NewFunctionBlock or None if creation fails
        """
        self.validate_context(context, ['task_description', 'user_request'])
        
        if not self.llm_service:
            self.logger.error("No LLM service available for function creation")
            return None
        
        try:
            # Build the prompt
            prompt = self._build_creation_prompt(context)
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.FUNCTION_BLOCK_DOCUMENTATION},
                {"role": "user", "content": prompt}
            ]
            
            # Define response schema
            schema = {
                "name": "function_block",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Function block name (snake_case)"},
                        "description": {"type": "string", "description": "Brief description"},
                        "code": {"type": "string", "description": "Complete Python code with def run(adata, **parameters)"},
                        "requirements": {"type": "string", "description": "Package requirements (one per line)"},
                        "static_config": {
                            "type": "object",
                            "properties": {
                                "args": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "value_type": {"type": "string"},
                                            "description": {"type": "string"},
                                            "optional": {"type": "boolean"},
                                            "default_value": {}
                                        }
                                    }
                                },
                                "description": {"type": "string"},
                                "tag": {"type": "string"}
                            },
                            "required": ["args", "description", "tag"]
                        },
                        "parameters": {"type": "object", "description": "Default parameter values"}
                    },
                    "required": ["name", "description", "code", "requirements", "static_config"]
                }
            }
            
            # Save LLM input if task_dir provided
            task_dir = context.get('task_dir')
            if task_dir:
                task_dir = Path(task_dir)
                with open(task_dir / 'llm_input.json', 'w') as f:
                    json.dump({
                        'messages': messages,
                        'schema': schema,
                        'temperature': 0.3,
                        'max_tokens': 4000,
                        'model': self.llm_service.model,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            
            # Call LLM
            self.logger.info(f"Creating function block with {self.llm_service.model}")
            result = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Save LLM output if task_dir provided
            if task_dir:
                with open(task_dir / 'llm_output.json', 'w') as f:
                    json.dump({
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            
            # Convert to NewFunctionBlock
            function_block = self._create_function_block(result, context)
            
            # Save final function block if created
            if function_block and task_dir:
                with open(task_dir / 'function_block.json', 'w') as f:
                    json.dump({
                        'name': function_block.name,
                        'type': function_block.type.value if hasattr(function_block.type, 'value') else str(function_block.type),
                        'description': function_block.description,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            
            return function_block
            
        except Exception as e:
            self.logger.error(f"Error creating function block: {e}")
            return None
    
    def _build_creation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for function block creation."""
        parts = []
        
        parts.append("## Task")
        parts.append(f"Create a function block for: {context['task_description']}")
        
        parts.append("\n## Context")
        parts.append(f"User Request: {context['user_request']}")
        
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
        
        parts.append("\n## Requirements")
        parts.append("1. The function MUST be named 'run' with signature: def run(adata, **parameters)")
        parts.append("2. Must handle the specific task described above")
        parts.append("3. Include all necessary error handling")
        parts.append("4. Print informative messages about processing steps")
        parts.append("5. Save any figures to /workspace/output/figures/")
        
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
            args.append(Arg(
                name=arg_data['name'],
                value_type=arg_data['value_type'],
                description=arg_data['description'],
                optional=arg_data.get('optional', True),
                default_value=arg_data.get('default_value')
            ))
        
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
        
        return NewFunctionBlock(
            name=result['name'],
            type=fb_type,
            description=result['description'],
            code=code,
            requirements=result['requirements'],
            parameters=result.get('parameters', {}),
            static_config=static_config
        )