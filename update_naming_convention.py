#!/usr/bin/env python3
"""Script to update all files to use new naming convention: _node_anndata.h5ad and _node_seuratObject.rds"""

import os
import re
from pathlib import Path

def update_file(file_path):
    """Update a single file with new naming conventions."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace output_data.h5ad with _node_anndata.h5ad
    content = re.sub(r'/workspace/output/output_data\.h5ad', '/workspace/output/_node_anndata.h5ad', content)
    content = re.sub(r'output_data\.h5ad', '_node_anndata.h5ad', content)
    
    # Replace input patterns
    content = re.sub(r'/workspace/input/adata\.h5ad', '/workspace/input/_node_anndata.h5ad', content)
    
    # Update specific patterns in test files
    content = re.sub(r"adata\.write\('/workspace/output/output_data\.h5ad'\)", 
                     "adata.write('/workspace/output/_node_anndata.h5ad')", content)
    content = re.sub(r"sc\.read_h5ad\('/workspace/input/adata\.h5ad'\)",
                     "sc.read_h5ad('/workspace/input/_node_anndata.h5ad')", content)
    
    # Update path references in tests
    content = re.sub(r'parent\'s output_data\.h5ad', "parent's _node_anndata.h5ad", content)
    content = re.sub(r'"output_data\.h5ad"', '"_node_anndata.h5ad"', content)
    content = re.sub(r"'output_data\.h5ad'", "'_node_anndata.h5ad'", content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Update all Python and markdown files in the project."""
    
    # Directories to update
    dirs_to_update = [
        'tests',
        'docs',
        'agents',
        'analysis_tree_management',
        'job_executors',
        'llm_service'
    ]
    
    files_updated = []
    
    for dir_name in dirs_to_update:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"Directory {dir_name} not found, skipping...")
            continue
            
        # Find all Python and Markdown files
        for ext in ['*.py', '*.md']:
            for file_path in dir_path.rglob(ext):
                if update_file(file_path):
                    files_updated.append(file_path)
                    print(f"Updated: {file_path}")
    
    print(f"\nTotal files updated: {len(files_updated)}")
    
    # Special handling for specific test files that need more complex updates
    special_files = [
        'tests/test_manual_tree.py',
        'tests/test_file_passing.py',
        'tests/test_file_passing_simple.py',
        'tests/test_parallel_tree_execution.py',
        'tests/test_parallel_tree_with_output.py'
    ]
    
    print("\nApplying special updates to key test files...")
    for file_name in special_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  Checking {file_name}...")
            # These files are already updated by the general pass
    
    print("\nUpdate complete!")
    print("\nNOTE: The NodeExecutor still needs to be updated to copy ALL files from parent outputs.")
    print("This requires manual code changes to ensure proper file passing behavior.")

if __name__ == "__main__":
    main()