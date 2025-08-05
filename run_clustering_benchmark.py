#!/usr/bin/env python
"""Run clustering benchmark with very specific instructions."""

import os
import sys
from pathlib import Path
import time
import json
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.agents.main_agent import MainAgent

def run_specific_clustering():
    """Run clustering with very specific request."""
    
    # Clean old outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ragomics_agent_local/test_outputs/clustering_specific_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found")
        return False
    
    agent = MainAgent(openai_api_key=api_key)
    
    # Find test data
    test_data = None
    for path in [
        Path("ragomics_agent_local/test_data/zebrafish.h5ad"),
        Path("ragomics_agent_local/test_outputs/clustering_openai_20250804_010410/test_data/zebrafish.h5ad"),
    ]:
        if path.exists():
            test_data = path
            break
    
    if not test_data:
        print("❌ No test data found")
        return False
    
    print("\n" + "="*80)
    print("SPECIFIC CLUSTERING BENCHMARK TEST")
    print("="*80)
    print(f"Test data: {test_data}")
    print(f"Output dir: {output_dir}\n")
    
    # Very specific clustering request
    request = """Process this zebrafish dataset with the following exact steps:
    
1. Quality control: Filter cells with less than 200 genes, filter genes expressed in less than 3 cells
2. Normalization: Apply normalization and log transformation  
3. PCA: Compute PCA with 50 components
4. UMAP: Compute UMAP embedding
5. Clustering with multiple resolutions: Apply Leiden clustering with THREE different resolutions (0.3, 0.5, 1.0) and save each result in adata.obs with keys 'leiden_0.3', 'leiden_0.5', 'leiden_1.0'
6. Calculate metrics: For each clustering result, calculate ARI score comparing to 'Cell_type' ground truth column and save all results to a CSV file
    
Make sure to create visualizations at each step and save all intermediate results."""
    
    print(f"Request: {request}\n")
    
    start_time = time.time()
    
    # Run with Docker execution
    result = agent.run_analysis(
        input_data_path=test_data,
        user_request=request,
        output_dir=str(output_dir),
        max_nodes=10,
        max_children=2,
        max_iterations=10,
        max_debug_trials=3,
        generation_mode="only_new",
        llm_model="gpt-4o-mini",
        verbose=True
    )
    
    duration = time.time() - start_time
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Duration: {duration:.1f}s")
    print(f"Total nodes: {result['total_nodes']}")
    print(f"Completed: {result['completed_nodes']}")
    print(f"Failed: {result['failed_nodes']}")
    
    # Check for multiple clustering results
    tree_id = result['tree_id']
    nodes_dir = output_dir / tree_id / "nodes"
    
    clustering_found = False
    multiple_resolutions = False
    
    if nodes_dir.exists():
        for node_dir in nodes_dir.iterdir():
            if node_dir.is_dir():
                outputs_dir = node_dir / "outputs"
                
                # Check for clustering metrics file
                metrics_files = list(outputs_dir.glob("*metrics*.csv"))
                if metrics_files:
                    clustering_found = True
                    # Check if it has multiple resolutions
                    for mf in metrics_files:
                        content = mf.read_text()
                        if "0.3" in content or "0.5" in content or "1.0" in content:
                            multiple_resolutions = True
                            print(f"✅ Found clustering metrics with multiple resolutions in {node_dir.name}")
                            print(f"   File: {mf.name}")
                            # Show first few lines
                            lines = content.split('\n')[:5]
                            for line in lines:
                                print(f"   {line}")
    
    # Final verdict
    print("\n" + "="*80)
    if clustering_found and multiple_resolutions:
        print("✅ SUCCESS: Found clustering with multiple resolutions and metrics")
    elif clustering_found:
        print("⚠️ PARTIAL: Found clustering but not with multiple resolutions")
    else:
        print("❌ FAILED: No clustering metrics found")
    
    return result

if __name__ == "__main__":
    run_specific_clustering()