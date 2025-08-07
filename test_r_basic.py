#!/usr/bin/env python3
"""Basic R test with minimal dependencies."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

def create_basic_r_script():
    """Create basic R script that uses only installed packages."""
    return '''
# Basic analysis using only installed packages
library(Matrix)
library(jsonlite)

# Set up paths
input_dir <- "/workspace/input"
output_dir <- "/workspace/output"
figures_dir <- file.path(output_dir, "figures")

message("Starting basic analysis...")

# Load input data
input_file <- file.path(input_dir, "pbmc3k_seurat_object.rds")
message(paste("Attempting to load:", input_file))

# Try to load the data
tryCatch({
    # Note: This might fail if Seurat is not installed
    obj <- readRDS(input_file)
    message(paste("Loaded object of class:", class(obj)))
    
    # Try to extract basic info
    if ("RNA" %in% names(obj@assays)) {
        counts <- obj@assays$RNA@counts
        message(paste("Found counts matrix:", nrow(counts), "genes x", ncol(counts), "cells"))
        
        # Calculate basic statistics
        cells_per_gene <- Matrix::rowSums(counts > 0)
        genes_per_cell <- Matrix::colSums(counts > 0)
        
        # Create basic plots
        png(file.path(figures_dir, "cells_per_gene.png"), width = 800, height = 600)
        hist(cells_per_gene, breaks = 50, 
             main = "Distribution of Cells per Gene",
             xlab = "Number of Cells", 
             ylab = "Number of Genes",
             col = "lightblue")
        dev.off()
        
        png(file.path(figures_dir, "genes_per_cell.png"), width = 800, height = 600)
        hist(genes_per_cell, breaks = 50,
             main = "Distribution of Genes per Cell",
             xlab = "Number of Genes",
             ylab = "Number of Cells", 
             col = "lightgreen")
        dev.off()
        
        # Save summary
        summary_info <- list(
            n_cells = ncol(counts),
            n_genes = nrow(counts),
            mean_genes_per_cell = mean(genes_per_cell),
            median_genes_per_cell = median(genes_per_cell),
            mean_cells_per_gene = mean(cells_per_gene),
            median_cells_per_gene = median(cells_per_gene)
        )
        
        # Save as JSON
        write_json(summary_info, file.path(output_dir, "analysis_summary.json"), pretty = TRUE)
        
        # Save a subset of the data as a simple matrix
        if (ncol(counts) > 100) {
            subset_counts <- counts[1:100, 1:100]
        } else {
            subset_counts <- counts
        }
        saveRDS(as.matrix(subset_counts), file.path(output_dir, "_node_subset_matrix.rds"))
        
        message("Analysis completed successfully!")
    } else {
        message("Could not find RNA assay in object")
    }
    
}, error = function(e) {
    message(paste("Error loading data:", e$message))
    
    # Create a dummy analysis
    message("Creating dummy analysis...")
    
    # Generate random data
    set.seed(42)
    n_genes <- 1000
    n_cells <- 500
    
    # Create sparse matrix
    counts <- Matrix::rsparsematrix(n_genes, n_cells, density = 0.1)
    rownames(counts) <- paste0("Gene", 1:n_genes)
    colnames(counts) <- paste0("Cell", 1:n_cells)
    
    # Create plots
    png(file.path(figures_dir, "dummy_data_density.png"), width = 800, height = 600)
    plot(density(counts@x), 
         main = "Distribution of Non-zero Values (Dummy Data)",
         xlab = "Expression Value",
         col = "red", lwd = 2)
    dev.off()
    
    # Save summary
    summary_info <- list(
        n_cells = n_cells,
        n_genes = n_genes,
        data_type = "dummy_data",
        density = Matrix::nnzero(counts) / (n_genes * n_cells)
    )
    
    write_json(summary_info, file.path(output_dir, "analysis_summary.json"), pretty = TRUE)
    saveRDS(counts, file.path(output_dir, "_node_seuratObject.rds"))
    
    message("Dummy analysis completed!")
})
'''

def main():
    """Run basic R analysis."""
    print("=== Running Basic R Analysis ===")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path("test_outputs") / f"r_basic_{timestamp}"
    node_dir = test_output_dir / "nodes" / "node_test_basic"
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
        if test_data.exists():
            shutil.copy2(test_data, input_dir / "pbmc3k_seurat_object.rds")
        
        # Write R script
        r_script_path = temp_path / "analysis.R"
        with open(r_script_path, "w") as f:
            f.write(create_basic_r_script())
        
        print(f"\n1. Created temporary directory: {temp_path}")
        
        # Run Docker container
        print("\n2. Running Docker container...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{temp_path}:/workspace",
            "ragomics-r:minimal",
            "Rscript", "/workspace/analysis.R"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            print(f"\n3. Docker execution completed with exit code: {result.returncode}")
            
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
                
            if result.stderr:
                print("\nWarnings/Info:")
                print(result.stderr)
                
            # Copy outputs
            if result.returncode == 0 or (output_dir / "analysis_summary.json").exists():
                node_outputs.mkdir(parents=True, exist_ok=True)
                
                # Copy all output files
                for item in output_dir.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(output_dir)
                        dest_path = node_outputs / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_path)
                
                print(f"\n4. Results saved to: {node_outputs}")
                
                # List output files
                print("\n5. Output files:")
                for file in sorted(node_outputs.rglob("*")):
                    if file.is_file():
                        rel_path = file.relative_to(node_outputs)
                        size = file.stat().st_size
                        print(f"   - {rel_path} ({size:,} bytes)")
                
                # Read and display summary
                summary_file = node_outputs / "analysis_summary.json"
                if summary_file.exists():
                    import json
                    with open(summary_file) as f:
                        summary = json.load(f)
                    print("\n6. Analysis Summary:")
                    for key, value in summary.items():
                        print(f"   - {key}: {value}")
                        
        except subprocess.TimeoutExpired:
            print("ERROR: Docker execution timed out")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()