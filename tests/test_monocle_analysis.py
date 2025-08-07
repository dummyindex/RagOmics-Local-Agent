#!/usr/bin/env python3
"""Test script to run an R function block with Monocle analysis."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from datetime import datetime
from models import NewFunctionBlock, FunctionBlockType
from job_executors.executor_manager import ExecutorManager
from utils.docker_utils import DockerManager

def create_monocle_function_block():
    """Create a simplified function block for Monocle analysis."""
    
    code = '''
run <- function(path_dict, params) {
    # Load required libraries
    library(SeuratObject)
    library(ggplot2)
    
    message("Starting simplified Monocle-style analysis...")
    
    # Load input data
    input_file <- file.path(path_dict$input_dir, "pbmc3k_seurat_object.rds")
    message(paste("Loading data from:", input_file))
    
    seurat_obj <- readRDS(input_file)
    message(paste("Loaded Seurat object with", ncol(seurat_obj), "cells"))
    
    # Perform basic dimensionality reduction if not already done
    if (!"pca" %in% names(seurat_obj@reductions)) {
        message("Running PCA...")
        # Note: This is simplified - real Seurat workflow would normalize first
        # For now we'll just create a basic plot from existing data
    }
    
    # Create UMAP plot if UMAP exists
    if ("umap" %in% names(seurat_obj@reductions)) {
        message("Creating UMAP visualization...")
        umap_coords <- as.data.frame(seurat_obj@reductions$umap@cell.embeddings)
        colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
        
        # Add metadata
        umap_coords$cluster <- seurat_obj@meta.data$seurat_clusters
        
        # Create plot
        p <- ggplot(umap_coords, aes(x = UMAP_1, y = UMAP_2, color = cluster)) +
            geom_point(size = 0.5, alpha = 0.7) +
            theme_minimal() +
            theme(legend.position = "right") +
            labs(title = "PBMC3k UMAP Plot",
                 subtitle = paste("Total cells:", ncol(seurat_obj))) +
            guides(color = guide_legend(override.aes = list(size = 3)))
        
        ggsave(file.path(path_dict$output_dir, "figures", "pbmc_umap_clusters.png"), 
               p, width = 8, height = 6, dpi = 300)
    }
    
    # Create basic statistics plot
    message("Creating cell statistics...")
    
    # Get cell counts per cluster
    cluster_counts <- table(seurat_obj@meta.data$seurat_clusters)
    cluster_df <- data.frame(
        cluster = names(cluster_counts),
        count = as.numeric(cluster_counts)
    )
    
    p2 <- ggplot(cluster_df, aes(x = cluster, y = count, fill = cluster)) +
        geom_bar(stat = "identity") +
        theme_minimal() +
        theme(legend.position = "none") +
        labs(title = "Cells per Cluster",
             x = "Cluster",
             y = "Number of Cells")
    
    ggsave(file.path(path_dict$output_dir, "figures", "cells_per_cluster.png"), 
           p2, width = 8, height = 6, dpi = 300)
    
    # Save the Seurat object
    output_seurat <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    saveRDS(seurat_obj, output_seurat)
    message(paste("Saved Seurat object to:", output_seurat))
    
    # Create summary
    summary_info <- list(
        n_cells = ncol(seurat_obj),
        n_genes = nrow(seurat_obj),
        n_clusters = length(unique(seurat_obj@meta.data$seurat_clusters)),
        analysis_type = "Basic trajectory-style analysis",
        timestamp = Sys.time()
    )
    
    # Save summary
    summary_file <- file.path(path_dict$output_dir, "analysis_summary.json")
    jsonlite::write_json(summary_info, summary_file, pretty = TRUE)
    
    message("Analysis completed successfully!")
}
'''
    
    requirements = '''
Matrix
jsonlite
SeuratObject
ggplot2
'''
    
    return NewFunctionBlock(
        name="basic_trajectory_analysis",
        type=FunctionBlockType.R,
        description="Perform basic trajectory-style analysis",
        code=code,
        requirements=requirements,
        parameters={}
    )

def main():
    """Main test function."""
    print("=== Testing R Function Block with Trajectory Analysis ===")
    
    # Create output directory with node structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"monocle_test_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_test_monocle"
    node_outputs = node_dir / "outputs"
    node_outputs.mkdir(parents=True, exist_ok=True)
    
    # Create function block
    print("\n1. Creating analysis function block...")
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
            
        if result.stderr:
            print("\n   Error output:")
            print("   " + "\n   ".join(result.stderr.split("\n")[-20:]))  # Last 20 lines

if __name__ == "__main__":
    main()