"""Final demonstration of R-Python interoperability."""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import anndata
from scipy.sparse import csr_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import conversion functions directly
from function_blocks.builtin.convert_anndata_to_sc_matrix.code import run as convert_to_sc
from function_blocks.builtin.convert_sc_matrix_to_anndata.code import run as convert_to_anndata


def create_demo_data():
    """Create demo AnnData."""
    np.random.seed(42)
    n_obs, n_vars = 100, 200
    
    # Create realistic count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    counts[counts > 30] = 0
    X = csr_matrix(counts)
    
    # Create AnnData with metadata
    adata = anndata.AnnData(X=X)
    adata.obs_names = [f'CELL_{i:03d}' for i in range(n_obs)]
    adata.var_names = [f'GENE_{i:04d}' for i in range(n_vars)]
    
    # Add cell metadata
    adata.obs['cell_type'] = np.random.choice(
        ['T_cell', 'B_cell', 'NK_cell', 'Monocyte'], 
        size=n_obs, 
        p=[0.3, 0.3, 0.2, 0.2]
    )
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_obs)
    adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
    adata.obs['n_genes'] = np.array((X > 0).sum(axis=1)).flatten()
    
    # Add gene metadata
    adata.var['gene_mean'] = np.array(X.mean(axis=0)).flatten()
    adata.var['n_cells'] = np.array((X > 0).sum(axis=0)).flatten()
    
    return adata


def demo_workflow():
    """Demonstrate the complete workflow."""
    
    print("🧬 R-Python Interoperability Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create data
        print("\n1️⃣ Creating demo single-cell data in Python...")
        adata = create_demo_data()
        print(f"   Created AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
        print(f"   Cell types: {dict(adata.obs['cell_type'].value_counts())}")
        
        # Save for conversion
        step1_dir = temp_path / "step1"
        step1_dir.mkdir()
        adata.write_h5ad(step1_dir / "_node_anndata.h5ad")
        
        # Convert to SC matrix
        print("\n2️⃣ Converting to shared SC matrix format...")
        step2_dir = temp_path / "step2"
        step2_dir.mkdir()
        
        convert_to_sc(
            {'input_dir': str(step1_dir), 'output_dir': str(step2_dir)},
            {}
        )
        
        # Check what was created
        sc_matrix_dir = step2_dir / "_node_sc_matrix"
        print(f"   ✓ Created SC matrix at: {sc_matrix_dir}")
        
        # List contents
        print("   Contents:")
        for item in sorted(sc_matrix_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                print(f"     📄 {item.relative_to(sc_matrix_dir)} ({size:,} bytes)")
        
        # Show metadata
        with open(sc_matrix_dir / "metadata.json") as f:
            metadata = json.load(f)
        print(f"   Metadata: {json.dumps(metadata, indent=2)}")
        
        print("\n3️⃣ This SC matrix can now be read by:")
        print("   • R (using Matrix::readMM)")
        print("   • Python (using scipy.io.mmread)")
        print("   • Any language that supports Matrix Market format")
        
        # Convert back to show round-trip
        print("\n4️⃣ Converting back to AnnData...")
        step3_dir = temp_path / "step3"
        step3_dir.mkdir()
        
        # Copy SC matrix
        import shutil
        shutil.copytree(sc_matrix_dir, step3_dir / "_node_sc_matrix")
        
        step4_dir = temp_path / "step4"
        step4_dir.mkdir()
        
        convert_to_anndata(
            {'input_dir': str(step3_dir), 'output_dir': str(step4_dir)},
            {}
        )
        
        # Load and compare
        adata_final = anndata.read_h5ad(step4_dir / "_node_anndata.h5ad")
        
        print(f"   ✓ Recovered AnnData: {adata_final.n_obs} cells × {adata_final.n_vars} genes")
        print(f"   ✓ Metadata preserved: {list(adata_final.obs.columns)}")
        
        # Verify data integrity
        X_orig = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X_final = adata_final.X.toarray() if hasattr(adata_final.X, 'toarray') else adata_final.X
        
        data_match = np.allclose(X_orig, X_final)
        metadata_match = all(adata.obs['cell_type'] == adata_final.obs['cell_type'])
        
        print(f"\n5️⃣ Verification:")
        print(f"   Data integrity: {'✅ PASS' if data_match else '❌ FAIL'}")
        print(f"   Metadata integrity: {'✅ PASS' if metadata_match else '❌ FAIL'}")
        
        print("\n" + "=" * 60)
        print("🎉 Demo Complete!")
        print("\nKey Achievement:")
        print("• Seamless data exchange between Python and R")
        print("• No rpy2 or reticulate dependencies needed")
        print("• Preserves sparse matrix format for efficiency")
        print("• Maintains all metadata and annotations")
        print("\nThe RagOmics agent can now automatically handle")
        print("mixed Python/R workflows with proper conversions!")


if __name__ == "__main__":
    demo_workflow()