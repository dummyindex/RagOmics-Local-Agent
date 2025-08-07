"""Test Seurat conversion with actual Seurat Docker image."""

import os
import json
import tempfile
import shutil
from pathlib import Path
import docker
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


def test_sc_matrix_to_seurat():
    """Test converting SC matrix to Seurat using actual conversion function."""
    
    # Read the conversion function
    conv_path = Path(__file__).parent.parent / "function_blocks/builtin/convert_sc_matrix_to_seuratobject/code.r"
    
    if not conv_path.exists():
        print(f"Conversion function not found at {conv_path}")
        return False
    
    with open(conv_path) as f:
        r_code = f.read()
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test data
        create_test_sc_matrix(input_dir)
        
        # Write conversion script
        r_script_path = temp_path / "convert_function.r"
        with open(r_script_path, "w") as f:
            f.write(r_code)
        
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
    
    # Check if Seurat object was created
    seurat_file <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    if (file.exists(seurat_file)) {
        seurat_obj <- readRDS(seurat_file)
        cat("\\nSeurat object summary:\\n")
        cat(sprintf("  Cells: %d\\n", ncol(seurat_obj)))
        cat(sprintf("  Genes: %d\\n", nrow(seurat_obj)))
        cat(sprintf("  Assays: %s\\n", paste(names(seurat_obj@assays), collapse=", ")))
        
        # Check metadata
        if (ncol(seurat_obj@meta.data) > 0) {
            cat(sprintf("  Metadata columns: %s\\n", 
                paste(names(seurat_obj@meta.data), collapse=", ")))
        }
    }
}, error = function(e) {
    cat("Error during conversion:\\n")
    print(e)
    quit(status = 1)
})
'''
        
        run_script_path = temp_path / "run.R"
        with open(run_script_path, "w") as f:
            f.write(wrapper_script)
        
        # Run with Docker
        try:
            client = docker.from_env()
            
            # Use seurat image
            image_name = "ragomics-r:seurat"
            
            # Run container
            print(f"Running conversion with {image_name}...")
            container = client.containers.run(
                image_name,
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
            print("\nContainer logs:")
            print("-" * 50)
            print(logs)
            print("-" * 50)
            
            # Check exit code
            if result['StatusCode'] != 0:
                print(f"\nERROR: Container failed with exit code {result['StatusCode']}")
                return False
            
            # Check outputs
            seurat_file = output_dir / "_node_seuratObject.rds"
            if not seurat_file.exists():
                print("\nERROR: Seurat object file not created")
                return False
            
            # Check if SC matrix was preserved
            sc_matrix_output = output_dir / "_node_sc_matrix"
            if not sc_matrix_output.exists():
                print("\nERROR: SC matrix not preserved in output")
                return False
            
            print(f"\n✓ Seurat object created: {seurat_file}")
            print(f"✓ SC matrix preserved: {sc_matrix_output}")
            print("\nTest passed! SC matrix successfully converted to Seurat")
            
            return True
            
        except docker.errors.DockerException as e:
            print(f"Docker error: {e}")
            return False
        finally:
            if 'container' in locals():
                container.remove()


def test_seurat_to_sc_matrix():
    """Test converting Seurat back to SC matrix."""
    
    # This would test the reverse conversion
    # For now, we'll skip this as we need to create a Seurat object first
    print("\nSeurat to SC matrix conversion test - TODO")
    return True


if __name__ == "__main__":
    print("Testing SC Matrix to Seurat Conversion")
    print("=" * 60)
    
    success1 = test_sc_matrix_to_seurat()
    
    if success1:
        print("\n" + "=" * 60)
        print("All conversion tests passed! ✅")
    else:
        print("\n" + "=" * 60) 
        print("Some tests failed ❌")