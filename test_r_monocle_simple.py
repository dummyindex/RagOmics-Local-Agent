#!/usr/bin/env python3
"""Simple test to run R analysis in Docker."""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

def create_r_script():
    """Create R script for analysis."""
    return '''
# Load required libraries
library(SeuratObject)
library(jsonlite)

# Set up paths
input_dir <- "/workspace/input"
output_dir <- "/workspace/output"
figures_dir <- file.path(output_dir, "figures")

message("Starting analysis...")

# Load input data
input_file <- file.path(input_dir, "pbmc3k_seurat_object.rds")
message(paste("Loading data from:", input_file))

seurat_obj <- readRDS(input_file)
message(paste("Loaded Seurat object with", ncol(seurat_obj), "cells"))

# Create UMAP plot if UMAP exists
if ("umap" %in% names(seurat_obj@reductions)) {
    message("Creating UMAP visualization...")
    umap_coords <- as.data.frame(seurat_obj@reductions$umap@cell.embeddings)
    colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
    
    # Add metadata
    umap_coords$cluster <- as.numeric(as.character(seurat_obj@meta.data$seurat_clusters))
    
    # Create plot using base R
    png(file.path(figures_dir, "pbmc_umap_clusters.png"), width = 800, height = 600)
    
    # Set up color palette
    n_clusters <- length(unique(umap_coords$cluster))
    colors <- rainbow(n_clusters)
    
    # Create plot
    plot(umap_coords$UMAP_1, umap_coords$UMAP_2, 
         col = colors[umap_coords$cluster + 1],
         pch = 16, cex = 0.5,
         main = paste("PBMC3k UMAP Plot (", ncol(seurat_obj), " cells)"),
         xlab = "UMAP_1", ylab = "UMAP_2")
    
    # Add legend
    legend("topright", legend = sort(unique(umap_coords$cluster)), 
           col = colors[sort(unique(umap_coords$cluster)) + 1], 
           pch = 16, title = "Cluster", cex = 0.8)
    
    dev.off()
    message("UMAP plot saved")
}

# Create basic statistics plot
message("Creating cell statistics...")

# Get cell counts per cluster
cluster_counts <- table(seurat_obj@meta.data$seurat_clusters)

# Create barplot using base R
png(file.path(figures_dir, "cells_per_cluster.png"), width = 800, height = 600)

barplot(cluster_counts, 
        main = "Cells per Cluster",
        xlab = "Cluster",
        ylab = "Number of Cells",
        col = rainbow(length(cluster_counts)),
        las = 1)

dev.off()
message("Cluster statistics plot saved")

# Save the Seurat object
output_seurat <- file.path(output_dir, "_node_seuratObject.rds")
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
summary_file <- file.path(output_dir, "analysis_summary.json")
write_json(summary_info, summary_file, pretty = TRUE)

message("Analysis completed successfully!")
'''

def main():
    """Run R analysis in Docker."""
    print("=== Running R Analysis with Docker ===")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"r_analysis_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_test_r"
    node_outputs = node_dir / "outputs"
    
    # Create temporary directory for Docker execution
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
        r_script_path = temp_path / "analysis.R"
        with open(r_script_path, "w") as f:
            f.write(create_r_script())
        
        print(f"\n1. Created temporary directory: {temp_path}")
        print(f"2. Copied test data to: {input_dir}")
        
        # Run Docker container
        print("\n3. Running Docker container...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{temp_path}:/workspace",
            "ragomics-r:minimal",
            "Rscript", "/workspace/analysis.R"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            print(f"\n4. Docker execution completed with exit code: {result.returncode}")
            
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
                
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
                
            # Copy outputs to final location
            if result.returncode == 0:
                node_outputs.mkdir(parents=True, exist_ok=True)
                
                # Copy all output files
                for item in output_dir.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(output_dir)
                        dest_path = node_outputs / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_path)
                
                print(f"\n5. Results saved to: {node_outputs}")
                
                # List output files
                print("\n6. Output files:")
                for file in sorted(node_outputs.rglob("*")):
                    if file.is_file():
                        rel_path = file.relative_to(node_outputs)
                        size = file.stat().st_size
                        print(f"   - {rel_path} ({size:,} bytes)")
                
                # Show figures
                figures_path = node_outputs / "figures"
                if figures_path.exists():
                    print(f"\n7. Generated figures:")
                    for fig in sorted(figures_path.glob("*.png")):
                        print(f"   - {fig.name}")
                        
        except subprocess.TimeoutExpired:
            print("ERROR: Docker execution timed out after 5 minutes")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()