#!/usr/bin/env python3
"""Debug workflow execution to understand why it stops."""

import os
import sys
import json
import logging
from pathlib import Path
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug logging for specific modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('ragomics_agent_local.agents.main_agent').setLevel(logging.DEBUG)
logging.getLogger('ragomics_agent_local.agents.function_creator_agent').setLevel(logging.DEBUG)

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config


def test_workflow_continuation():
    """Test why workflow stops after first node."""
    
    print("\n" + "="*80)
    print("WORKFLOW CONTINUATION DEBUG TEST")
    print("="*80 + "\n")
    
    # Setup
    output_dir = Path("test_outputs/workflow_debug")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    input_dir = output_dir / "input"
    input_dir.mkdir()
    shutil.copy("test_data/zebrafish.h5ad", input_dir / "zebrafish.h5ad")
    
    # Simple multi-step request
    user_request = """
Process the zebrafish dataset with these specific steps:
1. First step: filter cells with min_genes=200 
2. Second step: normalize using scanpy normalize_total
3. Third step: apply log1p transformation
"""
    
    # Configure
    config.function_block_timeout = 300
    
    # Create agent with debug enabled
    # Use API key from config to enable orchestrator
    main_agent = MainAgent(openai_api_key=config.openai_api_key)
    
    print("Starting workflow execution with debugging...\n")
    
    result = main_agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir / "results"),
        max_nodes=5,
        max_children=1,
        max_debug_trials=1,
        max_iterations=5,  # Allow enough iterations
        generation_mode="mixed",
        verbose=True
    )
    
    print("\n" + "="*40)
    print("FINAL RESULTS:")
    print("="*40)
    print(f"Total nodes: {result.get('total_nodes', 0)}")
    print(f"Completed: {result.get('completed_nodes', 0)}")
    print(f"Failed: {result.get('failed_nodes', 0)}")
    
    # Analyze tree
    tree_file = Path(result.get('tree_file', ''))
    if tree_file.exists():
        with open(tree_file) as f:
            tree_data = json.load(f)
            
        print(f"\nNodes created: {len(tree_data.get('nodes', {}))}")
        for node_id, node in tree_data.get('nodes', {}).items():
            print(f"  - {node['function_block']['name']}: {node['state']}")
            if node.get('children'):
                print(f"    Children: {node['children']}")
                
    # Check orchestrator logs
    orchestrator_dir = output_dir / "results" / "main_20*" / "orchestrator_tasks"
    for orch_dir in output_dir.glob("results/main_*/orchestrator_tasks"):
        print(f"\nOrchestrator logs in: {orch_dir}")
        for iter_dir in sorted(orch_dir.glob("iteration_*")):
            print(f"\n  {iter_dir.name}:")
            
            # Check request
            req_file = iter_dir / "request.json"
            if req_file.exists():
                with open(req_file) as f:
                    req_data = json.load(f)
                print(f"    Satisfied: {req_data.get('satisfied', 'N/A')}")
                print(f"    Function blocks: {len(req_data.get('function_blocks', []))}")
                
            # Check response
            resp_file = iter_dir / "response.json"
            if resp_file.exists():
                with open(resp_file) as f:
                    resp_data = json.load(f)
                print(f"    Response satisfied: {resp_data.get('satisfied', 'N/A')}")
                
    return result


if __name__ == "__main__":
    test_workflow_continuation()