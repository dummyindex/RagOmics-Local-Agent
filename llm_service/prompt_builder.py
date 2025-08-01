"""Prompt building utilities for LLM interactions."""

from typing import Dict, List, Optional, Any
from ..models import AnalysisNode, AnalysisTree, GenerationMode
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Builds prompts for LLM function block generation."""
    
    SYSTEM_PROMPT = """You are an expert bioinformatics analyst specializing in single-cell RNA sequencing data analysis. 
Your task is to generate or select function blocks that process AnnData objects to fulfill user requests.

Each function block should:
1. Have a clear, specific purpose
2. Take an AnnData object as input and return an AnnData object (or dict with 'adata' key)
3. Include all necessary imports and dependencies
4. Be self-contained and executable
5. Include appropriate error handling
6. Save any generated figures to /workspace/output/figures/

You understand common single-cell analysis workflows including:
- Quality control and filtering
- Normalization and scaling
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Clustering (Leiden, Louvain)
- Differential expression analysis
- Trajectory inference
- Cell type annotation
- Gene regulatory network analysis
"""

    @staticmethod
    def build_generation_prompt(
        user_request: str,
        current_node: Optional[AnalysisNode],
        parent_nodes: List[AnalysisNode],
        max_branches: int,
        generation_mode: GenerationMode,
        data_summary: Dict[str, Any],
        rest_task: Optional[str] = None
    ) -> str:
        """Build prompt for generating next function blocks."""
        
        prompt_parts = []
        
        # Add context about the analysis
        prompt_parts.append("## Analysis Context\n")
        prompt_parts.append(f"User Request: {user_request}\n")
        
        if rest_task:
            prompt_parts.append(f"Remaining Tasks: {rest_task}\n")
        
        # Add data summary
        prompt_parts.append("\n## Current Data State\n")
        prompt_parts.append(f"- Shape: {data_summary.get('n_obs', 'unknown')} observations × {data_summary.get('n_vars', 'unknown')} variables")
        prompt_parts.append(f"- Observations columns: {', '.join(data_summary.get('obs_columns', []))}")
        prompt_parts.append(f"- Layers: {', '.join(data_summary.get('layers', []))}")
        prompt_parts.append(f"- Embeddings: {', '.join(data_summary.get('obsm_keys', []))}")
        prompt_parts.append(f"- Annotations: {', '.join(data_summary.get('uns_keys', []))}")
        
        # Add parent node context
        if parent_nodes:
            prompt_parts.append("\n## Previous Analysis Steps\n")
            for i, node in enumerate(parent_nodes):
                prompt_parts.append(f"{i+1}. {node.function_block.name}: {node.function_block.description}")
        
        # Add current node if exists
        if current_node:
            prompt_parts.append(f"\nCurrent Step: {current_node.function_block.name}")
            prompt_parts.append(f"Description: {current_node.function_block.description}")
        
        # Add generation instructions
        prompt_parts.append(f"\n## Task\n")
        prompt_parts.append(f"Generate up to {max_branches} function blocks for the next analysis steps.")
        
        if generation_mode == GenerationMode.ONLY_NEW:
            prompt_parts.append("Generate NEW function blocks with complete implementation code.")
        elif generation_mode == GenerationMode.ONLY_EXISTING:
            prompt_parts.append("Select from EXISTING function blocks in the library.")
        else:
            prompt_parts.append("You may either generate NEW function blocks or select EXISTING ones.")
        
        prompt_parts.append("\nConsider:")
        prompt_parts.append("- What analysis steps are needed to fulfill the user request?")
        prompt_parts.append("- What has already been done in previous steps?")
        prompt_parts.append("- What logical next steps would progress toward the goal?")
        prompt_parts.append("- Are we satisfied that the request has been fulfilled?")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_function_block_template(language: str = "python") -> str:
        """Get function block code template."""
        
        if language == "python":
            return '''def run(adata, **kwargs):
    """
    Function block description here.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    anndata.AnnData or dict
        The processed AnnData object or dict with 'adata' key
    """
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up plotting
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Your analysis code here
    
    return adata
'''
        else:  # R
            return '''run <- function(adata, ...) {
    #' Function block description here
    #' 
    #' @param adata The input AnnData object
    #' @param ... Additional parameters
    #' @return The processed AnnData object or list with 'adata' element
    
    library(Seurat)
    library(ggplot2)
    library(dplyr)
    
    # Your analysis code here
    
    return(adata)
}
'''
    
    @staticmethod
    def build_debug_prompt(
        function_block_code: str,
        error_message: str,
        previous_attempts: List[str]
    ) -> str:
        """Build prompt for debugging failed function blocks."""
        
        prompt_parts = [
            "## Debugging Function Block\n",
            "The following function block failed to execute:\n",
            "```python",
            function_block_code,
            "```\n",
            f"Error: {error_message}\n"
        ]
        
        if previous_attempts:
            prompt_parts.append("\n## Previous Fix Attempts\n")
            for i, attempt in enumerate(previous_attempts):
                prompt_parts.append(f"Attempt {i+1}: {attempt}")
        
        prompt_parts.extend([
            "\n## Task",
            "Please fix the code to resolve the error.",
            "Ensure the fixed code:",
            "1. Addresses the specific error",
            "2. Maintains the original functionality",
            "3. Follows best practices",
            "4. Includes proper error handling"
        ])
        
        return "\n".join(prompt_parts)