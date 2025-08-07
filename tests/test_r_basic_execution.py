"""Test basic R execution without Seurat dependencies."""

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
    
    # Metadata
    metadata = {
        "source_format": "test",
        "n_obs": n_cells,
        "n_vars": n_genes,
        "matrix_format": "sparse",
        "components": ["X", "obs_names", "var_names"]
    }
    with open(sc_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created test SC matrix with {n_cells} cells and {n_genes} genes")


def test_basic_r_matrix_reading():
    """Test basic R execution with Matrix package only."""
    # Create test function block
    r_code = '''
run <- function(path_dict, params) {
    library(Matrix)
    library(jsonlite)
    
    # Read SC matrix format
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    
    if (!dir.exists(sc_dir)) {
        stop("No _node_sc_matrix found in input")
    }
    
    # Read components
    obs_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    var_names <- readLines(file.path(sc_dir, "var_names.txt"))
    X <- readMM(file.path(sc_dir, "X.mtx"))
    metadata <- fromJSON(file.path(sc_dir, "metadata.json"))
    
    # Print summary
    cat("Successfully read SC matrix\\n")
    cat(sprintf("Cells: %d\\n", length(obs_names)))
    cat(sprintf("Genes: %d\\n", length(var_names)))
    cat(sprintf("Matrix dimensions: %d x %d\\n", nrow(X), ncol(X)))
    cat(sprintf("Sparsity: %.2f%%\\n", 100 * (1 - nnzero(X) / (nrow(X) * ncol(X)))))
    
    # Create a simple output
    output_file <- file.path(path_dict$output_dir, "matrix_summary.json")
    summary_data <- list(
        n_cells = length(obs_names),
        n_genes = length(var_names),
        total_counts = sum(X),
        mean_counts_per_cell = rowSums(X),
        mean_counts_per_gene = colSums(X)
    )
    
    write(toJSON(summary_data, pretty = TRUE), output_file)
    
    # Return success
    list(
        success = TRUE,
        message = "Successfully processed SC matrix"
    )
}
'''
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test data
        create_test_sc_matrix(input_dir)
        
        # Write R script
        r_script_path = temp_path / "test_function.r"
        with open(r_script_path, "w") as f:
            f.write(r_code)
        
        # Create wrapper script with container paths
        wrapper_script = '''
source("/workspace/test_function.r")

path_dict <- list(
    input_dir = "/workspace/input",
    output_dir = "/workspace/output"
)

params <- list()

result <- run(path_dict, params)
print(result)
'''
        
        run_script_path = temp_path / "run.R"
        with open(run_script_path, "w") as f:
            f.write(wrapper_script)
        
        # Try to run with Docker if available
        try:
            client = docker.from_env()
            
            # Check if image exists
            try:
                client.images.get("ragomics-r:basic")
            except docker.errors.ImageNotFound:
                pytest.skip("R Docker image not built. Run: cd docker && docker build -f Dockerfile.r.basic -t ragomics-r:basic .")
            
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
            
            # Check output
            output_file = output_dir / "matrix_summary.json"
            assert output_file.exists(), "Output file not created"
            
            # Debug: print the file content
            print(f"Output file content:")
            with open(output_file) as f:
                content = f.read()
                print(content)
            
            with open(output_file) as f:
                summary = json.load(f)
            
            # jsonlite returns arrays, so check first element
            assert summary["n_cells"][0] == 100
            assert summary["n_genes"][0] == 50
            assert "total_counts" in summary
            assert summary["total_counts"][0] > 0
            
            print("Test passed!")
            
        except docker.errors.DockerException as e:
            pytest.skip(f"Docker not available: {e}")
        finally:
            # Cleanup
            if 'container' in locals():
                container.remove()


if __name__ == "__main__":
    test_basic_r_matrix_reading()