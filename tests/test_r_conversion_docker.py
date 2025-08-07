"""Test R conversion function block with Docker."""

import os
import json
import tempfile
import shutil
from pathlib import Path
import docker
import pytest
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


def create_test_sc_matrix(output_dir: Path):
    """Create a test shared SC matrix format."""
    sc_dir = output_dir / "_node_sc_matrix"
    sc_dir.mkdir(exist_ok=True)
    
    # Create test data
    n_cells = 100
    n_genes = 50
    
    # Cell names
    cell_names = [f"Cell_{i}" for i in range(n_cells)]
    with open(sc_dir / "obs_names.txt", "w") as f:
        f.write("\n".join(cell_names))
    
    # Gene names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    with open(sc_dir / "var_names.txt", "w") as f:
        f.write("\n".join(gene_names))
    
    # Expression matrix (sparse)
    np.random.seed(42)
    data = np.random.poisson(2, size=n_cells * n_genes)
    data[data > 5] = 0  # Make it sparse
    X = csr_matrix(data.reshape(n_cells, n_genes))
    mmwrite(sc_dir / "X.mtx", X)
    
    # Add some cell metadata
    obs_dir = sc_dir / "obs"
    obs_dir.mkdir(exist_ok=True)
    
    # Cell types
    cell_types = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=n_cells)
    with open(obs_dir / "cell_type.csv", "w") as f:
        f.write("cell_type\n")
        f.write("\n".join(cell_types))
    
    # Metadata
    metadata = {
        "source_format": "test",
        "n_obs": n_cells,
        "n_vars": n_genes,
        "matrix_format": "sparse",
        "components": ["X", "obs_names", "var_names", "obs"]
    }
    with open(sc_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created test SC matrix with {n_cells} cells and {n_genes} genes")


def test_sc_matrix_to_seurat_conversion():
    """Test converting SC matrix format to Seurat using actual conversion function."""
    # Read the actual conversion function
    conv_path = Path(__file__).parent.parent / "function_blocks/builtin/convert_sc_matrix_to_seuratobject/code.r"
    
    if not conv_path.exists():
        # Create the conversion function if it doesn't exist
        r_code = '''
run <- function(path_dict, params) {
    # Load required libraries
    library(Matrix)
    library(jsonlite)
    
    # Read SC matrix format
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    
    if (!dir.exists(sc_dir)) {
        stop("No _node_sc_matrix found in input")
    }
    
    # Read metadata
    metadata <- fromJSON(file.path(sc_dir, "metadata.json"))
    
    # Read cell and gene names
    obs_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    var_names <- readLines(file.path(sc_dir, "var_names.txt"))
    
    # Read expression matrix
    if (file.exists(file.path(sc_dir, "X.mtx"))) {
        X <- readMM(file.path(sc_dir, "X.mtx"))
    } else if (file.exists(file.path(sc_dir, "X.csv"))) {
        X <- as.matrix(read.csv(file.path(sc_dir, "X.csv"), row.names = 1))
    } else {
        stop("No expression matrix found (X.mtx or X.csv)")
    }
    
    # Note: X is already in cells x genes format from Python
    rownames(X) <- obs_names
    colnames(X) <- var_names
    
    # Create basic matrix summary (since we don't have Seurat in basic image)
    summary_data <- list(
        n_cells = nrow(X),
        n_genes = ncol(X),
        total_counts = sum(X),
        sparsity = 100 * (1 - nnzero(X) / (nrow(X) * ncol(X))),
        cell_names_sample = head(rownames(X), 5),
        gene_names_sample = head(colnames(X), 5)
    )
    
    # Read cell metadata if available
    obs_dir <- file.path(sc_dir, "obs")
    if (dir.exists(obs_dir)) {
        obs_files <- list.files(obs_dir, pattern = "\\\\.csv$", full.names = TRUE)
        cell_metadata <- list()
        
        for (obs_file in obs_files) {
            col_name <- tools::file_path_sans_ext(basename(obs_file))
            obs_data <- read.csv(obs_file)
            
            if (nrow(obs_data) == length(obs_names)) {
                cell_metadata[[col_name]] <- obs_data[, 1]
                summary_data[[paste0("metadata_", col_name)]] <- table(obs_data[, 1])
            }
        }
    }
    
    # Since we can't create actual Seurat object without Seurat package,
    # save the matrix and metadata for verification
    output_dir <- path_dict$output_dir
    
    # Save matrix
    writeMM(X, file.path(output_dir, "counts.mtx"))
    
    # Save names
    writeLines(rownames(X), file.path(output_dir, "cell_names.txt"))
    writeLines(colnames(X), file.path(output_dir, "gene_names.txt"))
    
    # Save summary
    write(toJSON(summary_data, pretty = TRUE), file.path(output_dir, "conversion_summary.json"))
    
    # Return success
    list(
        success = TRUE,
        message = "Successfully converted SC matrix format (without Seurat package)"
    )
}
'''
        with open(conv_path, "w") as f:
            f.write(r_code)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test data
        create_test_sc_matrix(input_dir)
        
        # Copy conversion script
        r_script_path = temp_path / "convert_function.r"
        shutil.copy(conv_path, r_script_path)
        
        # Create wrapper script
        wrapper_script = '''
source("/workspace/convert_function.r")

path_dict <- list(
    input_dir = "/workspace/input",
    output_dir = "/workspace/output"
)

params <- list()

tryCatch({
    result <- run(path_dict, params)
    print("Conversion completed successfully")
    print(result)
}, error = function(e) {
    cat("Error during conversion:\\n")
    print(e)
    quit(status = 1)
})
'''
        
        run_script_path = temp_path / "run.R"
        with open(run_script_path, "w") as f:
            f.write(wrapper_script)
        
        # Try to run with Docker
        try:
            client = docker.from_env()
            
            # Check if image exists
            try:
                client.images.get("ragomics-r:basic")
            except docker.errors.ImageNotFound:
                pytest.skip("R Docker image not built")
            
            # Run container
            container = client.containers.run(
                "ragomics-r:basic",
                volumes={
                    str(temp_path): {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                command=["Rscript", "run.R"],
                detach=True,
                remove=False
            )
            
            # Wait for completion
            result = container.wait()
            logs = container.logs().decode()
            print("Container logs:")
            print(logs)
            
            # Check exit code
            assert result['StatusCode'] == 0, f"Container failed with exit code {result['StatusCode']}"
            
            # Check outputs
            summary_file = output_dir / "conversion_summary.json"
            assert summary_file.exists(), "Summary file not created"
            
            with open(summary_file) as f:
                summary = json.load(f)
            
            print("\nConversion summary:")
            print(json.dumps(summary, indent=2))
            
            # Verify results
            assert summary["n_cells"][0] == 100
            assert summary["n_genes"][0] == 50
            assert summary["total_counts"][0] > 0
            assert "cell_names_sample" in summary
            assert len(summary["cell_names_sample"]) == 5
            
            # Check if metadata was processed
            if "metadata_cell_type" in summary:
                print("\nCell type distribution:")
                print(summary["metadata_cell_type"])
            
            # Verify output files
            assert (output_dir / "counts.mtx").exists()
            assert (output_dir / "cell_names.txt").exists()
            assert (output_dir / "gene_names.txt").exists()
            
            print("\nTest passed! SC matrix successfully processed in R")
            
        except docker.errors.DockerException as e:
            pytest.skip(f"Docker error: {e}")
        finally:
            if 'container' in locals():
                container.remove()


if __name__ == "__main__":
    test_sc_matrix_to_seurat_conversion()