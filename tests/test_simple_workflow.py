#!/usr/bin/env python3
"""Simple test to debug workflow execution."""

import os
import sys
import json
from pathlib import Path
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config


def test_simple_workflow():
    """Test a simple preprocessing -> analysis workflow."""
    
    print("\n" + "="*80)
    print("SIMPLE WORKFLOW TEST")
    print("="*80 + "\n")
    
    # Setup
    output_dir = Path("test_outputs/simple_workflow")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    input_dir = output_dir / "input"
    input_dir.mkdir()
    shutil.copy("test_data/zebrafish.h5ad", input_dir / "zebrafish.h5ad")
    
    # Simple request
    user_request = """
Process the zebrafish dataset:
1. First, do basic preprocessing: filter cells with min_genes=200
2. Then normalize the data using scanpy normalize_total and log1p
3. Finally, compute PCA with 50 components
"""
    
    # Configure
    config.function_block_timeout = 300
    
    # Create and run
    main_agent = MainAgent(openai_api_key=config.openai_api_key)
    
    print("Running workflow...")
    result = main_agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir / "results"),
        max_nodes=10,
        max_children=1,
        max_debug_trials=2,
        max_iterations=10,
        generation_mode="mixed",
        verbose=True
    )
    
    print("\n" + "="*40)
    print("RESULTS:")
    print("="*40)
    print(f"Total nodes: {result.get('total_nodes', 0)}")
    print(f"Completed: {result.get('completed_nodes', 0)}")
    print(f"Failed: {result.get('failed_nodes', 0)}")
    
    # Check tree file
    tree_file = Path(result.get('tree_file', ''))
    if tree_file.exists():
        with open(tree_file) as f:
            tree_data = json.load(f)
            
        print(f"\nNodes in tree: {len(tree_data.get('nodes', {}))}")
        for node_id, node in tree_data.get('nodes', {}).items():
            print(f"  - {node['function_block']['name']}: {node['state']}")
            
        # Check tree counters
        print(f"\nTree counters:")
        print(f"  total_nodes: {tree_data.get('total_nodes', 0)}")
        print(f"  completed_nodes: {tree_data.get('completed_nodes', 0)}")
        print(f"  failed_nodes: {tree_data.get('failed_nodes', 0)}")
    
    # List outputs
    results_dir = output_dir / "results"
    if results_dir.exists():
        print("\nOutput files:")
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith(('.h5ad', '.png', '.json')):
                    rel_path = os.path.relpath(os.path.join(root, file), results_dir)
                    print(f"  - {rel_path}")
                    
    return result


if __name__ == "__main__":
    test_simple_workflow()