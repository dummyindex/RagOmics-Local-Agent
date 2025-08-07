#!/usr/bin/env python3
"""Test Monocle3 pseudotime analysis with pre-built Docker image."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

def create_monocle_script():
    """Create R script for Monocle3 pseudotime analysis."""
    return '''
# Monocle3 Pseudotime Analysis
message("Starting Monocle3 pseudotime analysis...")

# First, check what packages are available
installed_packages <- installed.packages()[,"Package"]
message("Checking for required packages...")
message(paste("SeuratObject:", "SeuratObject" %in% installed_packages))
message(paste("SingleCellExperiment:", "SingleCellExperiment" %in% installed_packages))

# Try to install Monocle3 if not available
if (!"monocle3" %in% installed_packages) {
    message("Monocle3 not found. Installing dependencies...")
    
    # Install system dependencies are already in Dockerfile
    # Install R dependencies
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager", repos='https://cloud.r-project.org')
    
    # Core Bioconductor packages
    BiocManager::install(c(
        'BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
        'S4Vectors', 'SingleCellExperiment', 'SummarizedExperiment',
        'batchelor', 'limma', 'BiocParallel'
    ), ask=FALSE, update=FALSE)
    
    # CRAN dependencies
    install.packages(c('lme4', 'flexmix', 'MASS', 'Matrix.utils', 
                      'viridis', 'ggrepel', 'plyr', 'reshape2'),
                    repos='https://cloud.r-project.org')
    
    # Install monocle3
    if (!requireNamespace("remotes", quietly = TRUE))
        install.packages("remotes", repos='https://cloud.r-project.org')
    
    remotes::install_github('cole-trapnell-lab/monocle3', quiet=TRUE)
}

# Load libraries
library(SeuratObject)
library(Matrix)
suppressPackageStartupMessages({
    library(monocle3)
    library(ggplot2)
})

# Paths
input_file <- "/workspace/input/pbmc3k_seurat_object.rds"
output_dir <- "/workspace/output"
figures_dir <- file.path(output_dir, "figures")

# Load data
message("Loading Seurat object...")
seurat_obj <- readRDS(input_file)
message(paste("Loaded", ncol(seurat_obj), "cells and", nrow(seurat_obj), "genes"))

# Convert to Monocle3 CDS
message("Converting to cell_data_set...")

# Extract data
counts <- seurat_obj@assays$RNA@counts
cell_metadata <- seurat_obj@meta.data
gene_metadata <- data.frame(
    gene_short_name = rownames(counts),
    row.names = rownames(counts)
)

# Create CDS
cds <- new_cell_data_set(
    counts,
    cell_metadata = cell_metadata,
    gene_metadata = gene_metadata
)

message("Preprocessing CDS...")
# Preprocess
cds <- preprocess_cds(cds, num_dim = 50, method = "PCA")

# Reduce dimensions
message("Reducing dimensions with UMAP...")
cds <- reduce_dimension(cds, reduction_method = "UMAP", preprocess_method = "PCA")

# Cluster cells
message("Clustering cells...")
cds <- cluster_cells(cds, reduction_method = "UMAP", k = 20)

# Learn trajectory graph
message("Learning trajectory graph for pseudotime...")
cds <- learn_graph(cds, use_partition = TRUE)

# Plot trajectories
message("Creating trajectory plots...")

# Basic trajectory plot
p1 <- plot_cells(cds, 
                 color_cells_by = "cluster",
                 label_cell_groups = FALSE,
                 label_leaves = TRUE,
                 label_branch_points = TRUE,
                 graph_label_size = 1.5) +
    ggtitle("Monocle3 Trajectory Analysis")

ggsave(file.path(figures_dir, "monocle_trajectory.png"), 
       p1, width = 10, height = 8, dpi = 300)

# Order cells in pseudotime
# Select root cells (you might want to customize this based on biology)
# For now, we'll use cells from cluster 0 as root
root_cells <- row.names(subset(cds@colData, clusters == "1"))
if (length(root_cells) > 10) {
    root_cells <- sample(root_cells, 10)
}

message("Ordering cells in pseudotime...")
cds <- order_cells(cds, root_cells = root_cells)

# Plot pseudotime
p2 <- plot_cells(cds,
                 color_cells_by = "pseudotime",
                 label_cell_groups = FALSE,
                 label_leaves = FALSE,
                 label_branch_points = FALSE,
                 graph_label_size = 1.5) +
    scale_color_viridis_c() +
    ggtitle("Cells Ordered by Pseudotime")

ggsave(file.path(figures_dir, "monocle_pseudotime.png"), 
       p2, width = 10, height = 8, dpi = 300)

# Plot by original Seurat clusters
if ("seurat_clusters" %in% colnames(colData(cds))) {
    p3 <- plot_cells(cds,
                     color_cells_by = "seurat_clusters",
                     label_cell_groups = TRUE,
                     label_leaves = FALSE,
                     label_branch_points = TRUE) +
        ggtitle("Trajectory colored by Seurat Clusters")
    
    ggsave(file.path(figures_dir, "monocle_seurat_clusters.png"), 
           p3, width = 10, height = 8, dpi = 300)
}

# Extract pseudotime values
pseudotime_df <- data.frame(
    cell_id = colnames(cds),
    pseudotime = pseudotime(cds),
    cluster = clusters(cds),
    partition = partitions(cds)
)

# Save pseudotime data
write.csv(pseudotime_df, 
          file.path(output_dir, "pseudotime_values.csv"), 
          row.names = FALSE)

# Save CDS object
message("Saving results...")
saveRDS(cds, file.path(output_dir, "_node_monocle_cds.rds"))

# Create summary
summary_info <- list(
    n_cells = ncol(cds),
    n_genes = nrow(cds),
    n_clusters = length(unique(clusters(cds))),
    n_partitions = length(unique(partitions(cds))),
    pseudotime_range = range(pseudotime(cds), na.rm = TRUE),
    analysis_type = "Monocle3 pseudotime analysis",
    root_cluster = "1",
    timestamp = Sys.time()
)

# Save summary
jsonlite::write_json(summary_info, 
                    file.path(output_dir, "analysis_summary.json"), 
                    pretty = TRUE)

message("Monocle3 pseudotime analysis completed!")
'''

def main():
    """Run Monocle3 pseudotime analysis."""
    print("=== Running Monocle3 Pseudotime Analysis ===")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"monocle_pseudotime_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_monocle_pseudotime"
    node_outputs = node_dir / "outputs"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        figures_dir = output_dir / "figures"
        
        input_dir.mkdir()
        output_dir.mkdir()
        figures_dir.mkdir()
        
        # Copy test data
        test_data = Path("test_data/pbmc3k_seurat_object.rds")
        if not test_data.exists():
            print(f"ERROR: Test data not found at {test_data}")
            return
            
        shutil.copy2(test_data, input_dir / "pbmc3k_seurat_object.rds")
        
        # Write R script
        r_script_path = temp_path / "monocle_analysis.R"
        with open(r_script_path, "w") as f:
            f.write(create_monocle_script())
        
        print(f"\n1. Created analysis script")
        print(f"2. Test data: {test_data}")
        
        # First, let's check if we need to use a different image
        # Try with the seurat image from r_image_variants
        images_to_try = [
            "ragomics-r:minimal",
            "ragomics-r:seurat",
            "rocker/tidyverse:4.3.2"  # Fallback with more packages
        ]
        
        for image in images_to_try:
            print(f"\n3. Trying with Docker image: {image}")
            
            # Check if image exists
            check_cmd = ["docker", "images", "-q", image]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if not result.stdout.strip() and image == "rocker/tidyverse:4.3.2":
                print(f"   Pulling {image}...")
                pull_cmd = ["docker", "pull", image]
                subprocess.run(pull_cmd)
            
            # Run Docker container
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{temp_path}:/workspace",
                image,
                "Rscript", "/workspace/monocle_analysis.R"
            ]
            
            try:
                print("\n4. Running analysis (this may take several minutes)...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                print(f"\n5. Execution completed with exit code: {result.returncode}")
                
                if result.stdout:
                    print("\nOutput (last 50 lines):")
                    print("\n".join(result.stdout.split("\n")[-50:]))
                    
                if result.stderr:
                    print("\nErrors/Warnings (last 30 lines):")
                    print("\n".join(result.stderr.split("\n")[-30:]))
                
                # Check if successful
                if result.returncode == 0 or (output_dir / "analysis_summary.json").exists():
                    # Copy outputs
                    node_outputs.mkdir(parents=True, exist_ok=True)
                    
                    for item in output_dir.rglob("*"):
                        if item.is_file():
                            rel_path = item.relative_to(output_dir)
                            dest_path = node_outputs / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_path)
                    
                    print(f"\n6. Results saved to: {node_outputs}")
                    
                    # List output files
                    print("\n7. Output files:")
                    for file in sorted(node_outputs.rglob("*")):
                        if file.is_file():
                            rel_path = file.relative_to(node_outputs)
                            size = file.stat().st_size
                            print(f"   - {rel_path} ({size:,} bytes)")
                    
                    # Read summary
                    summary_file = node_outputs / "analysis_summary.json"
                    if summary_file.exists():
                        with open(summary_file) as f:
                            summary = json.load(f)
                        print("\n8. Analysis Summary:")
                        for key, value in summary.items():
                            print(f"   - {key}: {value}")
                    
                    # Success - break the loop
                    break
                    
            except subprocess.TimeoutExpired:
                print(f"   Timeout with {image}")
                continue
            except Exception as e:
                print(f"   Error with {image}: {e}")
                continue

if __name__ == "__main__":
    main()