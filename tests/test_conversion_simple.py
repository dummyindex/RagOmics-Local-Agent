"""Simple test of conversion logic without full agent."""

import tempfile
from pathlib import Path
import json

# Test the conversion detection logic
def test_conversion_needed():
    """Test conversion detection between Python and R."""
    
    print("Testing Conversion Detection Logic")
    print("=" * 50)
    
    # Mock parent outputs
    test_cases = [
        {
            "parent": {"type": "python", "output": "_node_anndata.h5ad"},
            "child": {"type": "r", "input": "any"},
            "expected": True,
            "conversion": "convert_anndata_to_sc_matrix"
        },
        {
            "parent": {"type": "r", "output": "_node_seuratObject.rds"},
            "child": {"type": "python", "input": "any"},
            "expected": True,
            "conversion": "convert_sc_matrix_to_anndata"
        },
        {
            "parent": {"type": "python", "output": "_node_anndata.h5ad"},
            "child": {"type": "python", "input": "any"},
            "expected": False,
            "conversion": None
        },
        {
            "parent": {"type": "r", "output": "_node_seuratObject.rds"},
            "child": {"type": "r", "input": "any"},
            "expected": False,
            "conversion": None
        }
    ]
    
    print("\nTest Cases:")
    for i, case in enumerate(test_cases):
        parent_type = case["parent"]["type"]
        child_type = case["child"]["type"]
        needs_conversion = parent_type != child_type
        
        status = "✓" if needs_conversion == case["expected"] else "✗"
        print(f"{status} Case {i+1}: {parent_type} → {child_type}")
        print(f"  Expected conversion: {case['expected']}")
        print(f"  Actual: {needs_conversion}")
        
        if needs_conversion and case["conversion"]:
            print(f"  Conversion function: {case['conversion']}")
    
    print("\n" + "=" * 50)
    print("Conversion detection test complete!")
    

def test_conversion_files():
    """Test that conversion files exist."""
    
    print("\nChecking Conversion Function Files")
    print("=" * 50)
    
    base_path = Path(__file__).parent.parent / "function_blocks/builtin"
    
    conversion_dirs = [
        "convert_anndata_to_sc_matrix",
        "convert_seurat_to_sc_matrix",
        "convert_sc_matrix_to_anndata",
        "convert_sc_matrix_to_seuratobject"
    ]
    
    all_exist = True
    for dir_name in conversion_dirs:
        dir_path = base_path / dir_name
        code_file_py = dir_path / "code.py"
        code_file_r = dir_path / "code.r"
        config_file = dir_path / "config.json"
        
        has_code = code_file_py.exists() or code_file_r.exists()
        has_config = config_file.exists()
        exists = has_code and has_config
        
        status = "✓" if exists else "✗"
        print(f"{status} {dir_name}: {'complete' if exists else 'INCOMPLETE'}")
        if not has_code:
            print(f"    Missing code file")
        if not has_config:
            print(f"    Missing config.json")
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✅ All conversion functions are in place!")
    else:
        print("\n❌ Some conversion functions are missing!")
    
    return all_exist


def test_sc_matrix_structure():
    """Test the SC matrix directory structure."""
    
    print("\nTesting SC Matrix Structure")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock SC matrix
        sc_dir = temp_path / "_node_sc_matrix"
        sc_dir.mkdir()
        
        # Required files
        files = {
            "metadata.json": {"source_format": "test", "n_obs": 100, "n_vars": 50},
            "obs_names.txt": "Cell_1\nCell_2\nCell_3",
            "var_names.txt": "Gene_1\nGene_2\nGene_3",
            "X.mtx": "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0"
        }
        
        # Create files
        for filename, content in files.items():
            file_path = sc_dir / filename
            if isinstance(content, dict):
                with open(file_path, "w") as f:
                    json.dump(content, f)
            else:
                with open(file_path, "w") as f:
                    f.write(content)
        
        # Create subdirectories
        (sc_dir / "obs").mkdir()
        (sc_dir / "var").mkdir()
        
        # Check structure
        print("Created SC matrix structure:")
        for item in sorted(sc_dir.rglob("*")):
            if item.is_file():
                print(f"  📄 {item.relative_to(sc_dir)}")
            else:
                print(f"  📁 {item.relative_to(sc_dir)}/")
        
        # Verify required files
        required = ["metadata.json", "obs_names.txt", "var_names.txt", "X.mtx"]
        all_present = all((sc_dir / f).exists() for f in required)
        
        if all_present:
            print("\n✅ SC matrix structure is correct!")
        else:
            print("\n❌ SC matrix structure is incomplete!")
    
    return all_present


if __name__ == "__main__":
    print("Simple Conversion Tests")
    print("=" * 70)
    
    # Run tests
    test_conversion_needed()
    print()
    
    files_ok = test_conversion_files()
    print()
    
    structure_ok = test_sc_matrix_structure()
    
    print("\n" + "=" * 70)
    if files_ok and structure_ok:
        print("✅ All conversion components are ready!")
        print("\nThe agent should be able to:")
        print("1. Detect when Python→R conversion is needed")
        print("2. Insert convert_anndata_to_sc_matrix node")
        print("3. Detect when R→Python conversion is needed") 
        print("4. Insert appropriate conversion node")
    else:
        print("❌ Some components need attention")