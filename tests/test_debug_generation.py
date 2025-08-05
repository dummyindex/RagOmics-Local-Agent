#!/usr/bin/env python
"""Debug test to inspect generated function blocks."""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
import scanpy as sc
import numpy as np
from ragomics_agent_local.agents import MainAgent
from ragomics_agent_local.config import Config

def create_test_data(n_obs=100, n_vars=500, output_path=None):
    """Create small test data."""
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    adata = sc.AnnData(X=X.astype(np.float32))
    adata.obs['n_genes'] = (X > 0).sum(axis=1)
    adata.var['gene_name'] = [f'Gene_{i:04d}' for i in range(n_vars)]
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write(output_path)
    
    return adata

def debug_generation():
    """Debug function block generation."""
    print("=" * 80)
    print("DEBUG FUNCTION BLOCK GENERATION")
    print("=" * 80)
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("test_outputs/debug") / f"run_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data
    input_file = base_dir / "input_data.h5ad"
    adata = create_test_data(100, 500, input_file)
    print(f"Created test data: {adata.shape}")
    print(f"Output directory: {base_dir}\n")
    
    # Use real OpenAI API
    config = Config()
    api_key = config.openai_api_key
    
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    print(f"✅ Using OpenAI API with model: {config.openai_model}")
    agent = MainAgent(openai_api_key=api_key)
    
    # Simple request
    request = "Apply quality control filtering (remove cells with less than 200 genes)"
    
    print("Request:", request)
    print("-" * 80)
    
    try:
        # Run with just 1 node to inspect
        result = agent.run_analysis(
            input_data_path=str(input_file),
            user_request=request,
            output_dir=str(base_dir),
            max_nodes=1,  # Just create one node
            max_iterations=1,  # Just one iteration
            verbose=True
        )
        
        print("\n" + "=" * 80)
        print("INSPECTING GENERATED CODE")
        print("=" * 80)
        
        # Find and inspect the generated files
        tree_id = result.get('tree_id')
        if tree_id:
            tree_dir = base_dir / tree_id
            nodes_dir = tree_dir / "nodes"
            
            if nodes_dir.exists():
                for node_dir in nodes_dir.glob("node_*"):
                    print(f"\n📁 Node: {node_dir.name}")
                    
                    # Check requirements.txt
                    req_file = node_dir / "function_block" / "requirements.txt"
                    if req_file.exists():
                        print("\n📄 requirements.txt:")
                        print(req_file.read_text())
                    
                    # Check code.py
                    code_file = node_dir / "function_block" / "code.py"
                    if code_file.exists():
                        print("\n📄 code.py:")
                        code = code_file.read_text()
                        print(code[:1000])  # First 1000 chars
                        
                        # Check for common issues
                        if "import os" in code and "os" in req_file.read_text() if req_file.exists() else "":
                            print("\n⚠️  WARNING: 'os' found in requirements.txt but it's a built-in module!")
                        
                        if "def run(adata" in code:
                            print("\n⚠️  WARNING: Using old signature 'def run(adata, ...)' instead of 'def run(path_dict, params)'!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Results in: {base_dir}")
    print("=" * 80)

if __name__ == "__main__":
    debug_generation()