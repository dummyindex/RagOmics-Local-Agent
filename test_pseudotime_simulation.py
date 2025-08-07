#!/usr/bin/env python3
"""Simulate pseudotime analysis using available R packages."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

def create_pseudotime_script():
    """Create R script for pseudotime-like analysis using available packages."""
    return '''
# Pseudotime-like analysis using available packages
# This simulates what Monocle3 would do with basic R packages

library(SeuratObject)
library(Matrix)

# Paths
input_file <- "/workspace/input/pbmc3k_seurat_object.rds"
output_dir <- "/workspace/output"
figures_dir <- file.path(output_dir, "figures")

message("Loading Seurat object for pseudotime analysis...")
seurat_obj <- readRDS(input_file)

# Extract data
counts <- seurat_obj@assays$RNA@counts
metadata <- seurat_obj@meta.data

message(paste("Analyzing", ncol(counts), "cells and", nrow(counts), "genes"))

# Get dimensional reduction if available
if ("umap" %in% names(seurat_obj@reductions)) {
    umap_coords <- seurat_obj@reductions$umap@cell.embeddings
    message("Using existing UMAP coordinates")
} else if ("pca" %in% names(seurat_obj@reductions)) {
    pca_coords <- seurat_obj@reductions$pca@cell.embeddings[, 1:2]
    umap_coords <- pca_coords
    colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
    message("Using first 2 PCA components as coordinates")
} else {
    message("No dimensional reduction found, creating simple coordinates")
    # Simple 2D projection based on total counts and gene counts
    total_counts <- Matrix::colSums(counts)
    gene_counts <- Matrix::colSums(counts > 0)
    umap_coords <- cbind(total_counts, gene_counts)
    colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
}

# Simulate pseudotime calculation
# In real Monocle3, this would be based on the learned trajectory
# Here we'll use a simple heuristic based on position in reduced space

message("Calculating simulated pseudotime...")

# Find the "root" cell (cell with minimum first coordinate)
root_cell <- which.min(umap_coords[, 1])
root_pos <- umap_coords[root_cell, ]

# Calculate distance from root as pseudotime
pseudotime <- sqrt(rowSums((umap_coords - matrix(root_pos, nrow = nrow(umap_coords), ncol = 2, byrow = TRUE))^2))

# Normalize pseudotime to 0-100 scale
pseudotime <- (pseudotime - min(pseudotime)) / (max(pseudotime) - min(pseudotime)) * 100

# Add cluster information
if ("seurat_clusters" %in% colnames(metadata)) {
    clusters <- as.numeric(as.character(metadata$seurat_clusters))
    message(paste("Found", length(unique(clusters)), "clusters"))
} else {
    # Create simple clusters based on coordinates
    message("No seurat_clusters found, creating simple clusters based on position")
    # Simple k-means like clustering based on coordinates
    clusters <- as.numeric(cut(umap_coords[, 1], breaks = 5, labels = 1:5))
}

# Create visualizations
message("Creating pseudotime visualizations...")

# Plot 1: Cells colored by cluster
png(file.path(figures_dir, "pseudotime_clusters.png"), width = 1000, height = 800)
plot(umap_coords, 
     col = rainbow(max(clusters, na.rm = TRUE))[clusters],
     pch = 16, cex = 0.6,
     main = "Cells Colored by Cluster",
     xlab = "Dimension 1", ylab = "Dimension 2")
legend("topright", 
       legend = paste("Cluster", sort(unique(clusters))),
       col = rainbow(max(clusters, na.rm = TRUE))[sort(unique(clusters))],
       pch = 16, cex = 0.8)
# Mark root cell
points(umap_coords[root_cell, 1], umap_coords[root_cell, 2], 
       col = "black", pch = 8, cex = 2)
text(umap_coords[root_cell, 1], umap_coords[root_cell, 2], 
     "Root", pos = 4, cex = 1.2)
dev.off()

# Plot 2: Cells colored by pseudotime
png(file.path(figures_dir, "pseudotime_trajectory.png"), width = 1000, height = 800)
# Create color scale for pseudotime
colors <- colorRampPalette(c("blue", "green", "yellow", "red"))(100)
pseudotime_colors <- colors[pmax(1, pmin(100, round(pseudotime) + 1))]

plot(umap_coords, 
     col = pseudotime_colors,
     pch = 16, cex = 0.6,
     main = "Simulated Pseudotime Trajectory",
     xlab = "Dimension 1", ylab = "Dimension 2")

# Add color scale legend
legend_vals <- seq(0, 100, 25)
legend("topright", 
       legend = paste("Pseudotime", legend_vals),
       col = colors[legend_vals + 1],
       pch = 16, cex = 0.8,
       title = "Pseudotime")
# Mark root cell
points(umap_coords[root_cell, 1], umap_coords[root_cell, 2], 
       col = "black", pch = 8, cex = 2)
dev.off()

# Plot 3: Pseudotime distribution by cluster
png(file.path(figures_dir, "pseudotime_by_cluster.png"), width = 1000, height = 600)
boxplot(pseudotime ~ clusters,
        main = "Pseudotime Distribution by Cluster",
        xlab = "Cluster", 
        ylab = "Pseudotime",
        col = rainbow(max(clusters, na.rm = TRUE)))
dev.off()

# Create pseudotime progression analysis
# Find genes that change along pseudotime
message("Analyzing gene expression changes along pseudotime...")

# Select top variable genes for efficiency
# Calculate row variances manually since Matrix::rowVars may not be available
message("Calculating gene variances...")
gene_vars <- sapply(1:nrow(counts), function(i) {
    var(as.numeric(counts[i, ]))
})
names(gene_vars) <- rownames(counts)
top_genes <- names(sort(gene_vars, decreasing = TRUE)[1:100])

# Calculate correlation with pseudotime for top genes
gene_correlations <- sapply(top_genes, function(gene) {
    gene_expr <- as.numeric(counts[gene, ])
    cor(gene_expr, pseudotime, use = "complete.obs")
})

# Get top positively and negatively correlated genes
top_pos_genes <- names(sort(gene_correlations, decreasing = TRUE)[1:10])
top_neg_genes <- names(sort(gene_correlations, decreasing = FALSE)[1:10])

# Plot expression of top changing genes
png(file.path(figures_dir, "pseudotime_gene_expression.png"), width = 1200, height = 800)
par(mfrow = c(2, 5), mar = c(4, 4, 3, 1))

for (i in 1:5) {
    gene <- top_pos_genes[i]
    expr <- as.numeric(counts[gene, ])
    plot(pseudotime, expr, 
         main = paste(gene, "(+)"),
         xlab = "Pseudotime", ylab = "Expression",
         pch = 16, cex = 0.3, col = "red")
    # Add trend line
    abline(lm(expr ~ pseudotime), col = "darkred", lwd = 2)
}

for (i in 1:5) {
    gene <- top_neg_genes[i]
    expr <- as.numeric(counts[gene, ])
    plot(pseudotime, expr, 
         main = paste(gene, "(-)"),
         xlab = "Pseudotime", ylab = "Expression",
         pch = 16, cex = 0.3, col = "blue")
    # Add trend line
    abline(lm(expr ~ pseudotime), col = "darkblue", lwd = 2)
}
dev.off()

# Save results
message("Saving results...")

# Create pseudotime data frame
pseudotime_df <- data.frame(
    cell_id = colnames(counts),
    pseudotime = pseudotime,
    cluster = clusters,
    UMAP_1 = umap_coords[, 1],
    UMAP_2 = umap_coords[, 2]
)

write.csv(pseudotime_df, 
          file.path(output_dir, "pseudotime_values.csv"), 
          row.names = FALSE)

# Save gene correlation results
gene_results <- data.frame(
    gene = names(gene_correlations),
    pseudotime_correlation = gene_correlations
)
gene_results <- gene_results[order(abs(gene_results$pseudotime_correlation), decreasing = TRUE), ]

write.csv(gene_results, 
          file.path(output_dir, "gene_pseudotime_correlations.csv"), 
          row.names = FALSE)

# Save pseudotime results separately (avoid modifying Seurat object)
# Create a simple results object
results_obj <- list(
    pseudotime = pseudotime,
    clusters = clusters,
    coordinates = umap_coords,
    metadata = metadata
)
saveRDS(results_obj, file.path(output_dir, "_node_seuratObject.rds"))

# Create analysis summary
summary_info <- list(
    n_cells = ncol(counts),
    n_genes = nrow(counts),
    n_clusters = length(unique(clusters)),
    pseudotime_range = range(pseudotime),
    root_cell_id = colnames(counts)[root_cell],
    top_positive_genes = top_pos_genes,
    top_negative_genes = top_neg_genes,
    max_positive_correlation = max(gene_correlations, na.rm = TRUE),
    max_negative_correlation = min(gene_correlations, na.rm = TRUE),
    analysis_type = "Simulated pseudotime analysis (Monocle3-like)",
    timestamp = Sys.time()
)

jsonlite::write_json(summary_info, 
                    file.path(output_dir, "analysis_summary.json"), 
                    pretty = TRUE)

message("Pseudotime analysis completed!")
message(paste("Root cell:", colnames(counts)[root_cell]))
message(paste("Pseudotime range:", round(min(pseudotime), 2), "-", round(max(pseudotime), 2)))
message(paste("Top positive gene:", top_pos_genes[1], 
              "correlation:", round(gene_correlations[top_pos_genes[1]], 3)))
message(paste("Top negative gene:", top_neg_genes[1], 
              "correlation:", round(gene_correlations[top_neg_genes[1]], 3)))
'''

def main():
    """Run simulated pseudotime analysis."""
    print("=== Running Simulated Pseudotime Analysis (Monocle3-like) ===")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"pseudotime_sim_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_pseudotime"
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
        r_script_path = temp_path / "pseudotime_analysis.R"
        with open(r_script_path, "w") as f:
            f.write(create_pseudotime_script())
        
        print(f"\n1. Created pseudotime analysis script")
        print(f"2. Using test data: {test_data}")
        
        # Run Docker container
        print(f"\n3. Running Docker container with ragomics-r:minimal")
        
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{temp_path}:/workspace",
            "ragomics-r:minimal",
            "Rscript", "/workspace/pseudotime_analysis.R"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            print(f"\n4. Execution completed with exit code: {result.returncode}")
            
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
                
            if result.stderr:
                print("\nWarnings:")
                print(result.stderr)
            
            # Copy outputs regardless of exit code if files exist
            if (output_dir / "analysis_summary.json").exists():
                node_outputs.mkdir(parents=True, exist_ok=True)
                
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
                
                # Read and display summary
                summary_file = node_outputs / "analysis_summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                    print("\n7. Analysis Summary:")
                    for key, value in summary.items():
                        print(f"   - {key}: {value}")
                        
        except subprocess.TimeoutExpired:
            print("ERROR: Docker execution timed out")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()