"""Test complete Python-R-Python workflow with actual execution."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import anndata
import docker

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import conversion functions directly
from function_blocks.builtin.convert_anndata_to_sc_matrix.code import run as convert_to_sc
from function_blocks.builtin.convert_sc_matrix_to_anndata.code import run as convert_to_anndata


def create_test_anndata(output_dir: Path):
    """Create test AnnData and save it."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50
    
    # Create sparse count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    counts[counts > 20] = 0
    X = csr_matrix(counts)
    
    # Create AnnData
    adata = anndata.AnnData(X=X)
    adata.obs['cell_type'] = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=n_obs)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_obs)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    adata.var['highly_variable'] = np.random.choice([True, False], size=n_vars)
    
    # Save
    adata.write_h5ad(output_dir / "_node_anndata.h5ad")
    
    print(f"Created test AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def run_r_processing(input_dir: Path, output_dir: Path):
    """Run R processing on SC matrix format."""
    
    # R code that works with SC matrix
    r_code = '''
run <- function(path_dict, params) {
    library(Matrix)
    library(jsonlite)
    library(SeuratObject)
    
    # Read SC matrix
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    if (!dir.exists(sc_dir)) {
        stop("SC matrix not found")
    }
    
    # Load data
    X <- readMM(file.path(sc_dir, "X.mtx"))
    obs_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    var_names <- readLines(file.path(sc_dir, "var_names.txt"))
    
    # X is cells x genes from Python, transpose for R
    X <- t(X)
    rownames(X) <- var_names
    colnames(X) <- obs_names
    
    # Create SeuratObject
    seurat_obj <- CreateSeuratObject(counts = X, project = "test")
    
    # Load cell metadata
    obs_dir <- file.path(sc_dir, "obs")
    if (dir.exists(obs_dir)) {
        cell_type_file <- file.path(obs_dir, "cell_type.csv")
        if (file.exists(cell_type_file)) {
            cell_types <- read.csv(cell_type_file)
            seurat_obj[["cell_type"]] <- cell_types[, 1]
        }
    }
    
    # Simple processing - calculate per-cell statistics
    seurat_obj[["percent.mt"]] <- 0  # No mitochondrial genes in test data
    seurat_obj[["nCount_RNA"]] <- colSums(X)
    seurat_obj[["nFeature_RNA"]] <- colSums(X > 0)
    
    # Calculate some statistics
    stats <- list(
        n_cells = ncol(seurat_obj),
        n_genes = nrow(seurat_obj),
        mean_counts_per_cell = mean(seurat_obj$nCount_RNA),
        mean_genes_per_cell = mean(seurat_obj$nFeature_RNA),
        cell_type_counts = as.list(table(seurat_obj$cell_type))
    )
    
    # Save results
    write(toJSON(stats, pretty = TRUE), 
          file.path(path_dict$output_dir, "r_analysis_stats.json"))
    
    # Save SeuratObject
    saveRDS(seurat_obj, file.path(path_dict$output_dir, "_node_seuratObject.rds"))
    
    # Copy SC matrix to output for next step
    file.copy(sc_dir, path_dict$output_dir, recursive = TRUE)
    
    cat("R processing complete\\n")
    return(list(success = TRUE))
}
'''
    
    # Create temp directory for R execution
    with tempfile.TemporaryDirectory() as r_temp:
        r_temp_path = Path(r_temp)
        
        # Copy input to temp
        temp_input = r_temp_path / "input"
        temp_output = r_temp_path / "output"
        shutil.copytree(input_dir, temp_input)
        temp_output.mkdir()
        
        # Write R script
        with open(r_temp_path / "process.r", "w") as f:
            f.write(r_code)
        
        # Write runner
        runner = '''
source("/workspace/process.r")
result <- run(
    list(input_dir = "/workspace/input", output_dir = "/workspace/output"),
    list()
)
print(result)
'''
        
        with open(r_temp_path / "run.R", "w") as f:
            f.write(runner)
        
        # Run with Docker
        client = docker.from_env()
        container = client.containers.run(
            "ragomics-r:seurat",
            volumes={str(r_temp_path): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            command=["Rscript", "run.R"],
            detach=True,
            remove=False
        )
        
        result = container.wait()
        logs = container.logs().decode()
        print("R processing logs:")
        print(logs)
        
        container.remove()
        
        if result['StatusCode'] != 0:
            raise RuntimeError("R processing failed")
        
        # Copy results back
        for item in temp_output.iterdir():
            if item.is_file():
                shutil.copy2(item, output_dir)
            else:
                shutil.copytree(item, output_dir / item.name)


def test_complete_workflow():
    """Test complete Python → R → Python workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Complete Python-R-Python Workflow Test")
        print("=" * 60)
        
        # Step 1: Create test data in Python
        print("\n1. Creating test data in Python...")
        step1_input = temp_path / "step1_input"
        step1_output = temp_path / "step1_output"
        step1_input.mkdir()
        step1_output.mkdir()
        
        original_adata = create_test_anndata(step1_input)
        
        # Step 2: Convert to SC matrix
        print("\n2. Converting AnnData to SC matrix...")
        result = convert_to_sc(
            {'input_dir': str(step1_input), 'output_dir': str(step1_output)},
            {}
        )
        
        # Verify SC matrix exists
        assert (step1_output / "_node_sc_matrix").exists()
        print("✓ SC matrix created")
        
        # Step 3: Process in R
        print("\n3. Processing in R...")
        step2_input = temp_path / "step2_input"
        step2_output = temp_path / "step2_output"
        step2_input.mkdir()
        step2_output.mkdir()
        
        # Copy SC matrix to R input
        shutil.copytree(step1_output / "_node_sc_matrix", step2_input / "_node_sc_matrix")
        
        # Run R processing
        run_r_processing(step2_input, step2_output)
        
        # Check R results
        stats_file = step2_output / "r_analysis_stats.json"
        assert stats_file.exists()
        
        with open(stats_file) as f:
            r_stats = json.load(f)
        
        # Handle R's jsonlite arrays
        n_cells = r_stats['n_cells'][0] if isinstance(r_stats['n_cells'], list) else r_stats['n_cells']
        n_genes = r_stats['n_genes'][0] if isinstance(r_stats['n_genes'], list) else r_stats['n_genes']
        mean_counts = r_stats['mean_counts_per_cell'][0] if isinstance(r_stats['mean_counts_per_cell'], list) else r_stats['mean_counts_per_cell']
        mean_genes = r_stats['mean_genes_per_cell'][0] if isinstance(r_stats['mean_genes_per_cell'], list) else r_stats['mean_genes_per_cell']
        
        print(f"✓ R analysis complete: {n_cells} cells, {n_genes} genes")
        print(f"  Mean counts/cell: {mean_counts:.1f}")
        print(f"  Mean genes/cell: {mean_genes:.1f}")
        
        # Step 4: Convert back to Python
        print("\n4. Converting back to AnnData...")
        step3_input = temp_path / "step3_input"
        step3_output = temp_path / "step3_output"
        step3_input.mkdir()
        step3_output.mkdir()
        
        # We have both SeuratObject and SC matrix, test with SC matrix
        shutil.copytree(step2_output / "_node_sc_matrix", step3_input / "_node_sc_matrix")
        
        result = convert_to_anndata(
            {'input_dir': str(step3_input), 'output_dir': str(step3_output)},
            {}
        )
        
        # Step 5: Verify round-trip
        print("\n5. Verifying round-trip conversion...")
        final_adata = anndata.read_h5ad(step3_output / "_node_anndata.h5ad")
        
        # Check dimensions
        assert final_adata.shape == original_adata.shape
        print(f"✓ Dimensions preserved: {final_adata.shape}")
        
        # Check data preservation
        if hasattr(original_adata.X, 'toarray'):
            X_orig = original_adata.X.toarray()
        else:
            X_orig = original_adata.X
            
        if hasattr(final_adata.X, 'toarray'):
            X_final = final_adata.X.toarray()
        else:
            X_final = final_adata.X
            
        assert np.allclose(X_orig, X_final)
        print("✓ Expression data preserved")
        
        # Check metadata
        assert 'cell_type' in final_adata.obs.columns
        assert all(final_adata.obs['cell_type'] == original_adata.obs['cell_type'])
        print("✓ Cell metadata preserved")
        
        print("\n" + "=" * 60)
        print("Complete workflow test PASSED! ✅")
        print("\nWorkflow summary:")
        print("  Python (AnnData) → SC Matrix → R (SeuratObject) → SC Matrix → Python (AnnData)")
        print("  All data and metadata preserved through conversion")
        
        return True


if __name__ == "__main__":
    try:
        test_complete_workflow()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)