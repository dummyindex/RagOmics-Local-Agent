"""Prompt building utilities for LLM interactions."""

from typing import Dict, List, Optional, Any
from ..models import AnalysisNode, AnalysisTree, GenerationMode
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Builds prompts for LLM function block generation."""
    
    SYSTEM_PROMPT = """You are an expert bioinformatics analyst specializing in single-cell RNA sequencing data analysis. 
Your task is to generate or select function blocks that process AnnData objects to fulfill user requests.

IMPORTANT: All function blocks MUST follow the Function Block Framework conventions:
- See: agents/FUNCTION_BLOCK_FRAMEWORK.md for complete specifications

Key Requirements:
1. Python functions MUST use signature: def run(path_dict, params)
2. Load data from: os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
3. Save data to: os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
4. R functions MUST use: run <- function(path_dict, params)
5. ALL files in parent's outputs/ folder automatically pass to child nodes' input/ folder
6. Follow the standard template:
   ```python
   def run(path_dict, params):
       import scanpy as sc
       import os
       
       # Load data from input directory
       input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
       adata = sc.read_h5ad(input_path)
       
       # Process data using params...
       
       # Save output with standard name
       output_path = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
       adata.write(output_path)
       
       return adata
   ```

Each function block should:
1. Have a clear, specific purpose
2. Follow input/output conventions strictly
3. Include all necessary imports and dependencies
4. Be self-contained and executable
5. Include appropriate error handling
6. Save any generated figures to path_dict["output_dir"]/figures/
7. Handle missing inputs gracefully

IMPORTANT CONSIDERATIONS:
- Always verify that API functions exist in the libraries you're using
- Check the official documentation when implementing algorithms
- Consider what libraries provide the functionality you need
- For any algorithm, think about which library would implement it
- Ensure proper import statements for all functions used
- When working with large datasets, consider computational complexity and memory usage
- Add progress messages using print statements to track long-running operations
- For computationally expensive algorithms, consider if there are more efficient alternatives
- Be aware that some clustering algorithms may scale poorly with dataset size

You understand common single-cell analysis workflows including:
- Quality control and filtering
- Normalization and scaling
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Clustering (various algorithms)
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
        """Get function block code template following framework conventions."""
        
        if language == "python":
            return '''def run(path_dict, params):
    """
    Function block description here.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing 'input_dir' and 'output_dir' paths
    params : dict
        Parameters for the analysis
        
    Returns
    -------
    anndata.AnnData
        The processed AnnData object
    """
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Ensure output directories exist
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    figures_dir = os.path.join(path_dict["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data (FRAMEWORK CONVENTION)
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        # Try any .h5ad file if standard name not found
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith('.h5ad')]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
        else:
            raise FileNotFoundError(f"No .h5ad files found in {path_dict['input_dir']}")
    
    print(f"Loading data from {input_path}")
    adata = sc.read_h5ad(input_path)
    
    print(f"Input data shape: {adata.shape}")
    
    # ========================================
    # YOUR ANALYSIS CODE HERE
    # ========================================
    
    # Example: Basic filtering
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    
    # ========================================
    # SAVE OUTPUTS (FRAMEWORK CONVENTION)
    # ========================================
    
    # Save processed data with standard name
    output_path = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    print(f"Saving processed data to {output_path}")
    adata.write(output_path)
    
    # Save any figures
    # plt.savefig(os.path.join(path_dict["output_dir"], "figures", "your_plot.png"), dpi=150, bbox_inches='tight')
    
    print(f"Output data shape: {adata.shape}")
    
    return adata
'''
        else:  # R
            return '''run <- function(path_dict, params) {
    #' Function block description here
    #' 
    #' @param path_dict List containing 'input_dir' and 'output_dir' paths
    #' @param params List of parameters for the analysis
    #' @return The processed Seurat object
    
    library(Seurat)
    library(ggplot2)
    library(dplyr)
    
    # Create output directories
    dir.create(path_dict$output_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(path_dict$output_dir, "figures"), recursive = TRUE, showWarnings = FALSE)
    
    # Load data (FRAMEWORK CONVENTION)
    # Try standard R object name first
    input_path <- file.path(path_dict$input_dir, "_node_seuratObject.rds")
    if (file.exists(input_path)) {
        cat(paste("Loading Seurat object from", input_path, "\\n"))
        seurat_obj <- readRDS(input_path)
    } else {
        # Try any .rds file in input
        rds_files <- list.files(path_dict$input_dir, pattern = "\\\\.rds$", full.names = TRUE)
        if (length(rds_files) > 0) {
            cat(paste("Loading Seurat object from", rds_files[1], "\\n"))
            seurat_obj <- readRDS(rds_files[1])
        } else {
            # Try to load from h5ad if coming from Python parent
            h5ad_files <- list.files(path_dict$input_dir, pattern = "\\\\.h5ad$", full.names = TRUE)
            if (length(h5ad_files) > 0) {
                library(anndata)
                cat(paste("Loading from h5ad:", h5ad_files[1], "\\n"))
                adata <- read_h5ad(h5ad_files[1])
                seurat_obj <- CreateSeuratObject(counts = adata$X)
            } else {
                stop("No input data found (.rds or .h5ad)")
            }
        }
    }
    
    cat(paste("Input data shape:", nrow(seurat_obj), "cells x", ncol(seurat_obj), "features\\n"))
    
    # ========================================
    # YOUR ANALYSIS CODE HERE
    # ========================================
    
    # Example: Basic Seurat processing
    # seurat_obj <- NormalizeData(seurat_obj)
    # seurat_obj <- FindVariableFeatures(seurat_obj)
    
    # ========================================
    # SAVE OUTPUTS (FRAMEWORK CONVENTION)
    # ========================================
    
    # Save Seurat object with standard name
    output_path <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    cat(paste("Saving Seurat object to", output_path, "\\n"))
    saveRDS(seurat_obj, output_path)
    
    # Save any figures
    # ggsave(file.path(path_dict$output_dir, "figures", "your_plot.png"), dpi = 150)
    
    cat(paste("Output data shape:", nrow(seurat_obj), "cells x", ncol(seurat_obj), "features\\n"))
    
    return(seurat_obj)
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