#!/usr/bin/env python3
"""Test script to run an R function block with Monocle analysis."""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import NewFunctionBlock, FunctionBlockType
from job_executors.executor_manager import ExecutorManager
from utils.docker_utils import DockerManager

def create_monocle_function_block():
    """Create a function block that uses Monocle for trajectory analysis."""
    
    code = '''
run <- function(path_dict, params) {
    # Load required libraries
    library(SeuratObject)
    library(SingleCellExperiment)
    
    # Try to install Monocle3 if not available
    if (!requireNamespace("monocle3", quietly = TRUE)) {
        message("Installing Monocle3 dependencies...")
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        
        # Install dependencies
        BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
                               'S4Vectors', 'SingleCellExperiment',
                               'SummarizedExperiment', 'batchelor'), ask=FALSE, update=FALSE)
        
        # Install additional dependencies
        install.packages(c('lme4', 'flexmix', 'plyr', 'reshape2'), repos='https://cloud.r-project.org')
        
        # Install monocle3
        if (!requireNamespace("remotes", quietly = TRUE))
            install.packages("remotes")
        remotes::install_github('cole-trapnell-lab/monocle3', quiet=TRUE)
    }
    
    library(monocle3)
    library(ggplot2)
    
    message("Starting Monocle3 analysis...")
    
    # Load input data
    input_file <- file.path(path_dict$input_dir, "pbmc3k_seurat_object.rds")
    message(paste("Loading data from:", input_file))
    
    seurat_obj <- readRDS(input_file)
    message(paste("Loaded Seurat object with", ncol(seurat_obj), "cells"))
    
    # Convert Seurat to Monocle3 CDS
    message("Converting to Monocle3 cell_data_set...")
    
    # Extract data
    expression_matrix <- seurat_obj@assays$RNA@counts
    cell_metadata <- seurat_obj@meta.data
    gene_metadata <- data.frame(
        gene_short_name = rownames(expression_matrix),
        row.names = rownames(expression_matrix)
    )
    
    # Create CDS object
    cds <- new_cell_data_set(
        expression_matrix,
        cell_metadata = cell_metadata,
        gene_metadata = gene_metadata
    )
    
    message(paste("Created CDS with", ncol(cds), "cells and", nrow(cds), "genes"))
    
    # Preprocess the data
    message("Preprocessing data...")
    cds <- preprocess_cds(cds, num_dim = 50)
    
    # Reduce dimensions
    message("Reducing dimensions with UMAP...")
    cds <- reduce_dimension(cds, reduction_method = "UMAP", preprocess_method = "PCA")
    
    # Cluster cells
    message("Clustering cells...")
    cds <- cluster_cells(cds, reduction_method = "UMAP")
    
    # Learn graph
    message("Learning trajectory graph...")
    cds <- learn_graph(cds)
    
    # Plot UMAP with clusters
    message("Creating visualizations...")
    p1 <- plot_cells(cds, 
                     color_cells_by = "cluster",
                     label_cell_groups = FALSE,
                     label_leaves = FALSE,
                     label_branch_points = FALSE,
                     graph_label_size = 1.5)
    
    ggsave(file.path(path_dict$output_dir, "figures", "monocle_umap_clusters.png"), 
           p1, width = 8, height = 6, dpi = 300)
    
    # Plot trajectory
    p2 <- plot_cells(cds,
                     color_cells_by = "cluster",
                     label_cell_groups = FALSE,
                     label_leaves = TRUE,
                     label_branch_points = TRUE,
                     graph_label_size = 1.5)
    
    ggsave(file.path(path_dict$output_dir, "figures", "monocle_trajectory.png"), 
           p2, width = 8, height = 6, dpi = 300)
    
    # If seurat_clusters exist, use them for coloring
    if ("seurat_clusters" %in% colnames(colData(cds))) {
        p3 <- plot_cells(cds,
                         color_cells_by = "seurat_clusters",
                         label_cell_groups = FALSE,
                         label_leaves = FALSE,
                         label_branch_points = TRUE,
                         graph_label_size = 1.5)
        
        ggsave(file.path(path_dict$output_dir, "figures", "monocle_seurat_clusters.png"), 
               p3, width = 8, height = 6, dpi = 300)
    }
    
    # Save the CDS object
    output_file <- file.path(path_dict$output_dir, "_node_monocle_cds.rds")
    saveRDS(cds, output_file)
    message(paste("Saved Monocle3 CDS object to:", output_file))
    
    # Also save as Seurat object for compatibility
    output_seurat <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    saveRDS(seurat_obj, output_seurat)
    
    # Create summary
    summary_info <- list(
        n_cells = ncol(cds),
        n_genes = nrow(cds),
        n_clusters = length(unique(clusters(cds))),
        analysis_type = "Monocle3 trajectory analysis",
        timestamp = Sys.time()
    )
    
    # Save summary
    summary_file <- file.path(path_dict$output_dir, "analysis_summary.json")
    jsonlite::write_json(summary_info, summary_file, pretty = TRUE)
    
    message("Monocle3 analysis completed successfully!")
}
'''
    
    requirements = '''
Matrix
jsonlite
SeuratObject
BiocManager
ggplot2
lme4
flexmix
plyr
reshape2
remotes
'''
    
    return NewFunctionBlock(
        name="monocle3_trajectory_analysis",
        type=FunctionBlockType.R,
        description="Perform trajectory analysis using Monocle3",
        code=code,
        requirements=requirements,
        parameters={}
    )

def main():
    """Main test function."""
    print("=== Testing R Function Block with Monocle3 ===")
    
    # Create output directory with node structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"monocle_test_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_test_monocle"
    node_outputs = node_dir / "outputs"
    node_outputs.mkdir(parents=True, exist_ok=True)
    
    # Create function block
    print("\n1. Creating Monocle3 function block...")
    function_block = create_monocle_function_block()
    
    # Initialize executor
    print("\n2. Initializing Docker executor...")
    docker_manager = DockerManager()
    executor_manager = ExecutorManager(docker_manager)
    
    # Check if image exists
    validation = executor_manager.validate_environment()
    if not validation.get("r_image"):
        print("ERROR: R Docker image not found. Please build it first.")
        return
    
    # Get test data path
    test_data_path = Path("test_data") / "pbmc3k_seurat_object.rds"
    if not test_data_path.exists():
        print(f"ERROR: Test data not found at {test_data_path}")
        return
    
    # Execute function block
    print(f"\n3. Executing function block with input: {test_data_path}")
    print("This may take a few minutes as Monocle3 needs to be installed...")
    
    result = executor_manager.execute(
        function_block=function_block,
        input_data_path=test_data_path,
        output_dir=node_outputs,
        parameters={}
    )
    
    # Check results
    print(f"\n4. Execution {'succeeded' if result.success else 'failed'}")
    if result.success:
        print(f"   Duration: {result.duration:.2f} seconds")
        print(f"   Output path: {node_outputs}")
        
        # List output files
        print("\n5. Output files:")
        for file in sorted(node_outputs.rglob("*")):
            if file.is_file():
                rel_path = file.relative_to(node_outputs)
                size = file.stat().st_size
                print(f"   - {rel_path} ({size:,} bytes)")
        
        # Show figures if any
        figures_dir = node_outputs / "figures"
        if figures_dir.exists():
            print(f"\n6. Generated figures in: {figures_dir}")
            for fig in sorted(figures_dir.glob("*.png")):
                print(f"   - {fig.name}")
                
    else:
        print(f"   Error: {result.error}")
        if result.logs:
            print("\n   Execution logs:")
            print("   " + "\n   ".join(result.logs.split("\n")[-20:]))  # Last 20 lines

if __name__ == "__main__":
    main()