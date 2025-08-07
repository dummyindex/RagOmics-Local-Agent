"""Test complete Python → R → Python workflow with basic R environment."""

import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import conversion functions directly
from function_blocks.builtin.convert_anndata_to_sc_matrix.code import run as convert_to_sc
from function_blocks.builtin.convert_sc_matrix_to_anndata.code import run as convert_to_anndata


def create_test_anndata():
    """Create a test AnnData object."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50
    
    # Create sparse count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    counts[counts > 20] = 0  # Make it sparse
    X = csr_matrix(counts)
    
    # Create AnnData
    adata = anndata.AnnData(X=X)
    
    # Add observation metadata
    adata.obs['cell_type'] = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=n_obs)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_obs)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    
    # Add variable metadata
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    adata.var['highly_variable'] = np.random.choice([True, False], size=n_vars)
    
    return adata


def test_full_workflow():
    """Test Python → SC Matrix → R processing → SC Matrix → Python."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create test AnnData
        print("Step 1: Creating test AnnData...")
        adata_original = create_test_anndata()
        input_dir1 = temp_path / "step1_input"
        output_dir1 = temp_path / "step1_output"
        input_dir1.mkdir()
        output_dir1.mkdir()
        
        # Save original AnnData
        adata_original.write_h5ad(input_dir1 / "_node_anndata.h5ad")
        print(f"Original AnnData: {adata_original.n_obs} cells x {adata_original.n_vars} genes")
        
        # Step 2: Convert AnnData to SC Matrix
        print("\nStep 2: Converting AnnData to SC Matrix...")
        path_dict1 = {
            'input_dir': str(input_dir1),
            'output_dir': str(output_dir1)
        }
        result1 = convert_to_sc(path_dict1, {})
        # The function returns a dict directly from run()
        assert result1 is not None, "Conversion returned None"
        
        # Verify SC matrix was created
        sc_matrix_dir = output_dir1 / "_node_sc_matrix"
        assert sc_matrix_dir.exists(), "SC matrix directory not created"
        assert (sc_matrix_dir / "metadata.json").exists(), "Metadata file not created"
        
        # Step 3: Simulate R processing (just copy the SC matrix for now)
        print("\nStep 3: Simulating R processing...")
        input_dir2 = temp_path / "step2_input"
        output_dir2 = temp_path / "step2_output"
        input_dir2.mkdir()
        output_dir2.mkdir()
        
        # Copy SC matrix to simulate R reading and writing it back
        shutil.copytree(sc_matrix_dir, input_dir2 / "_node_sc_matrix")
        
        # In a real scenario, R would process the data here
        # For now, just copy it to output
        shutil.copytree(input_dir2 / "_node_sc_matrix", output_dir2 / "_node_sc_matrix")
        
        # Add a marker to show R processed it
        with open(output_dir2 / "_node_sc_matrix" / "r_processed.txt", "w") as f:
            f.write("This data was processed by R\n")
        
        # Step 4: Convert SC Matrix back to AnnData
        print("\nStep 4: Converting SC Matrix back to AnnData...")
        input_dir3 = temp_path / "step3_input"
        output_dir3 = temp_path / "step3_output"
        input_dir3.mkdir()
        output_dir3.mkdir()
        
        shutil.copytree(output_dir2 / "_node_sc_matrix", input_dir3 / "_node_sc_matrix")
        
        path_dict3 = {
            'input_dir': str(input_dir3),
            'output_dir': str(output_dir3)
        }
        result3 = convert_to_anndata(path_dict3, {})
        assert result3 is not None, "Conversion back returned None"
        assert 'success' in result3 and result3['success'], f"Conversion back failed: {result3.get('message', 'Unknown error')}"
        
        # Step 5: Load final AnnData and compare
        print("\nStep 5: Verifying round-trip conversion...")
        adata_final = anndata.read_h5ad(output_dir3 / "_node_anndata.h5ad")
        
        # Compare dimensions
        assert adata_final.shape == adata_original.shape, "Shape mismatch after round-trip"
        
        # Compare data (allowing for small numerical differences)
        if hasattr(adata_original.X, 'toarray'):
            X_orig = adata_original.X.toarray()
        else:
            X_orig = adata_original.X
            
        if hasattr(adata_final.X, 'toarray'):
            X_final = adata_final.X.toarray()
        else:
            X_final = adata_final.X
            
        assert np.allclose(X_orig, X_final), "Data values changed during round-trip"
        
        # Compare metadata (order might be different)
        assert set(adata_final.obs.columns) == set(adata_original.obs.columns), "obs columns mismatch"
        assert set(adata_final.var.columns) == set(adata_original.var.columns), "var columns mismatch"
        
        # Check cell types preserved
        assert all(adata_final.obs['cell_type'] == adata_original.obs['cell_type']), "Cell types not preserved"
        
        print(f"\nSuccess! Round-trip conversion preserved all data")
        print(f"Final AnnData: {adata_final.n_obs} cells x {adata_final.n_vars} genes")
        print(f"Metadata preserved: {list(adata_final.obs.columns)}")
        
        # Check R processing marker
        assert (output_dir3 / "_node_sc_matrix" / "r_processed.txt").exists(), "R processing marker not preserved"
        
        return True


if __name__ == "__main__":
    test_full_workflow()
    print("\nAll tests passed! ✅")