#!/usr/bin/env python3
"""Direct test of SC matrix conversion function blocks."""

import os
import sys
import tempfile
import shutil
import json
import subprocess
from pathlib import Path

# Test Python conversion
def test_python_conversion():
    """Test Python AnnData to SC matrix conversion."""
    print("\n=== Testing Python AnnData → SC Matrix ===")
    
    test_data = Path(__file__).parent.parent / "test_data" / "zebrafish.h5ad"
    if not test_data.exists():
        print(f"Test data not found: {test_data}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Copy test data
        shutil.copy2(test_data, input_dir / "_node_anndata.h5ad")
        
        # Create path_dict
        path_dict = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir)
        }
        
        # Import and run the function
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from ragomics_agent_local.function_blocks.builtin.convert_anndata_to_sc_matrix import run
        
        try:
            result = run(path_dict, {})
            print(f"Result: {result}")
            
            # Verify output
            sc_matrix_dir = output_dir / "_node_sc_matrix"
            if sc_matrix_dir.exists():
                print(f"✓ SC matrix directory created")
                
                # List contents
                print(f"Contents of {sc_matrix_dir}:")
                for item in sorted(sc_matrix_dir.rglob("*")):
                    if item.is_file():
                        print(f"  {item.relative_to(sc_matrix_dir)}")
                
                # Check metadata
                metadata_path = sc_matrix_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    print(f"\nMetadata:")
                    print(f"  Source: {metadata['source']}")
                    print(f"  Shape: {metadata['shape']}")
                    print(f"  Components: {list(metadata['components'].keys())}")
                
                return True
            else:
                print("✗ SC matrix directory not created")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_r_conversion():
    """Test R Seurat to SC matrix conversion."""
    print("\n=== Testing R Seurat → SC Matrix ===")
    
    test_data = Path(__file__).parent.parent / "test_data" / "pbmc3k_seurat_object.rds"
    if not test_data.exists():
        print(f"Test data not found: {test_data}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Copy test data
        shutil.copy2(test_data, input_dir / "_node_seuratObject.rds")
        
        # Create R script to run the conversion
        r_script = temp_path / "run_conversion.R"
        r_code = f'''
# Load the conversion function
source("{Path(__file__).parent.parent / "src/ragomics_agent_local/function_blocks/builtin/convert_seurat_to_sc_matrix.r"}")

# Create path_dict
path_dict <- list(
    input_dir = "{input_dir}",
    output_dir = "{output_dir}"
)

# Run conversion
result <- run(path_dict, list())
print(result)
'''
        
        with open(r_script, 'w') as f:
            f.write(r_code)
        
        # Run R script
        try:
            result = subprocess.run(
                ["Rscript", str(r_script)],
                capture_output=True,
                text=True
            )
            
            print("R Output:")
            print(result.stdout)
            
            if result.stderr:
                print("R Errors:")
                print(result.stderr)
            
            # Verify output
            sc_matrix_dir = output_dir / "_node_sc_matrix"
            if sc_matrix_dir.exists():
                print(f"\n✓ SC matrix directory created")
                
                # List contents
                print(f"Contents of {sc_matrix_dir}:")
                for item in sorted(sc_matrix_dir.rglob("*")):
                    if item.is_file():
                        print(f"  {item.relative_to(sc_matrix_dir)}")
                
                # Check metadata
                metadata_path = sc_matrix_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    print(f"\nMetadata:")
                    print(f"  Source: {metadata['source']}")
                    print(f"  Shape: {metadata['shape']}")
                    print(f"  Components: {list(metadata['components'].keys())}")
                
                return result.returncode == 0
            else:
                print("✗ SC matrix directory not created")
                return False
                
        except Exception as e:
            print(f"Error running R script: {e}")
            return False


def main():
    """Run all tests."""
    print("Testing SC Matrix Conversion Function Blocks")
    print("=" * 50)
    
    # Run Python conversion test
    python_result = test_python_conversion()
    
    # Run R conversion test
    r_result = test_r_conversion()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Python conversion: {'✓ Passed' if python_result else '✗ Failed'}")
    print(f"R conversion: {'✓ Passed' if r_result else '✗ Failed'}")
    
    return 0 if (python_result and r_result) else 1


if __name__ == "__main__":
    sys.exit(main())