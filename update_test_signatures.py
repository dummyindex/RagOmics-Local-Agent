#!/usr/bin/env python3
"""Script to update all test files to use the new function signature."""

import os
import re
from pathlib import Path

def update_python_run_functions(content):
    """Update Python run function signatures to use path_dict, params."""
    
    # Pattern 1: def run(adata, **kwargs) or similar
    pattern1 = r'def run\s*\(\s*adata\s*(?:=\s*None)?\s*,\s*\*\*(?:parameters|kwargs|params)\s*\)\s*:'
    replacement1 = 'def run(path_dict, params):'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: def run(adata=None, specific_param=value, **kwargs)
    pattern2 = r'def run\s*\(\s*adata\s*(?:=\s*None)?\s*,\s*([^)]+),\s*\*\*(?:parameters|kwargs|params)\s*\)\s*:'
    def replace_with_params(match):
        # Extract parameter definitions
        params_str = match.group(1)
        # Parse parameters to suggest param usage
        return 'def run(path_dict, params):'
    content = re.sub(pattern2, replace_with_params, content)
    
    # Pattern 3: def run(adata, specific_params without **kwargs)
    pattern3 = r'def run\s*\(\s*adata\s*(?:=\s*None)?\s*,\s*([^*][^)]+)\)\s*:'
    content = re.sub(pattern3, r'def run(path_dict, params):', content)
    
    # Pattern 4: def run(adata) without any other params
    pattern4 = r'def run\s*\(\s*adata\s*(?:=\s*None)?\s*\)\s*:'
    content = re.sub(pattern4, r'def run(path_dict, params):', content)
    
    # Now add data loading logic if it's missing
    if 'def run(path_dict, params):' in content and 'path_dict["input_dir"]' not in content:
        # Find the function and add loading logic after imports
        lines = content.split('\n')
        new_lines = []
        in_run_function = False
        imports_done = False
        data_loading_added = False
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            if 'def run(path_dict, params):' in line:
                in_run_function = True
                imports_done = False
                data_loading_added = False
                continue
                
            if in_run_function and not data_loading_added:
                # Check if we're past imports
                if line.strip() and not line.strip().startswith('import') and not line.strip().startswith('from') and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                    if not imports_done:
                        imports_done = True
                        # Add data loading code
                        indent = '    '
                        new_lines.insert(-1, f'{indent}# Load data from path_dict')
                        new_lines.insert(-1, f'{indent}input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")')
                        new_lines.insert(-1, f'{indent}if not os.path.exists(input_path):')
                        new_lines.insert(-1, f'{indent}    h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]')
                        new_lines.insert(-1, f'{indent}    if h5ad_files:')
                        new_lines.insert(-1, f'{indent}        input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])')
                        new_lines.insert(-1, f'{indent}adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None')
                        new_lines.insert(-1, '')
                        data_loading_added = True
                        
            # Check if we're out of the function
            if in_run_function and line and not line.startswith(' ') and not line.startswith('\t'):
                in_run_function = False
                
        content = '\n'.join(new_lines)
    
    # Update save paths
    content = re.sub(
        r"adata\.write\s*\(\s*['\"]\/workspace\/output\/_node_anndata\.h5ad['\"]",
        'adata.write(os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")',
        content
    )
    
    # Update any direct workspace paths
    content = re.sub(
        r"['\"]\/workspace\/input\/['\"]",
        'path_dict["input_dir"]',
        content
    )
    content = re.sub(
        r"['\"]\/workspace\/output\/['\"]", 
        'path_dict["output_dir"]',
        content
    )
    
    return content

def update_test_file(filepath):
    """Update a single test file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update the content
        content = update_python_run_functions(content)
        
        # Only write if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    """Main function to update all test files."""
    # Get all Python test files
    test_dir = Path("/Users/random/Ragomics-workspace-all/agent_cc/ragomics_agent_local/tests")
    test_files = list(test_dir.glob("**/*.py"))
    
    updated_count = 0
    error_count = 0
    
    for test_file in test_files:
        if update_test_file(test_file):
            updated_count += 1
        else:
            # Check if it was an error or just no changes needed
            try:
                with open(test_file, 'r') as f:
                    if 'def run(adata' in f.read():
                        error_count += 1
            except:
                pass
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {len(test_files)}")
    print(f"Files updated: {updated_count}")
    print(f"Files with potential errors: {error_count}")

if __name__ == "__main__":
    main()