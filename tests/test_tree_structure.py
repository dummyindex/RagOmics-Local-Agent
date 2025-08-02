#!/usr/bin/env python3
"""Test the new analysis tree structure with nodes folder."""

import sys
import json
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.models import (
    NodeState, GenerationMode, FunctionBlockType, 
    NewFunctionBlock, StaticConfig, Arg
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def create_simple_block():
    """Create a simple test function block."""
    code = '''
def run(adata, **parameters):
    """Simple test function."""
    import scanpy as sc
    print(f"Processing data with shape: {adata.shape}")
    
    # Simple operation
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"After filtering: {adata.shape}")
    return adata
'''
    
    static_config = StaticConfig(
        args=[],
        description="Simple test function",
        tag="test"
    )
    
    return NewFunctionBlock(
        name="simple_test",
        type=FunctionBlockType.PYTHON,
        description="Simple test function block",
        static_config=static_config,
        code=code,
        requirements="scanpy",
        parameters={}
    )


def verify_structure(output_dir: Path, tree_id: str, node_id: str):
    """Verify the created structure matches specification."""
    
    print("\n" + "="*60)
    print("Verifying Tree Structure")
    print("="*60)
    
    # Check tree directory
    tree_dir = output_dir / f"tree_{tree_id}"
    assert tree_dir.exists(), f"Tree directory not found: {tree_dir}"
    print(f"✓ Tree directory exists: {tree_dir.name}")
    
    # Check tree files
    assert (tree_dir / "analysis_tree.json").exists(), "analysis_tree.json not found"
    print("  ✓ analysis_tree.json")
    
    assert (tree_dir / "tree_metadata.json").exists(), "tree_metadata.json not found"
    print("  ✓ tree_metadata.json")
    
    # Check nodes directory
    nodes_dir = tree_dir / "nodes"
    assert nodes_dir.exists(), "nodes directory not found"
    print("  ✓ nodes/ directory")
    
    # Check node directory
    node_dir = nodes_dir / f"node_{node_id}"
    assert node_dir.exists(), f"Node directory not found: node_{node_id}"
    print(f"    ✓ node_{node_id}/")
    
    # Check node structure
    assert (node_dir / "node_info.json").exists(), "node_info.json not found"
    print("      ✓ node_info.json")
    
    # Check function_block directory
    fb_dir = node_dir / "function_block"
    assert fb_dir.exists(), "function_block directory not found"
    print("      ✓ function_block/")
    
    assert (fb_dir / "code.py").exists(), "code.py not found"
    print("        ✓ code.py")
    
    assert (fb_dir / "config.json").exists(), "config.json not found"
    print("        ✓ config.json")
    
    # Check jobs directory
    jobs_dir = node_dir / "jobs"
    assert jobs_dir.exists(), "jobs directory not found"
    print("      ✓ jobs/")
    
    # Check for job folders
    job_folders = list(jobs_dir.glob("job_*"))
    if job_folders:
        print(f"        ✓ {len(job_folders)} job(s) found")
        
        # Check latest symlink if jobs exist
        if (jobs_dir / "latest").exists():
            print("        ✓ latest symlink")
    
    # Check outputs directory
    outputs_dir = node_dir / "outputs"
    assert outputs_dir.exists(), "outputs directory not found"
    print("      ✓ outputs/")
    
    # Check agent_tasks directory
    agent_tasks_dir = node_dir / "agent_tasks"
    assert agent_tasks_dir.exists(), "agent_tasks directory not found"
    print("      ✓ agent_tasks/")
    
    # Check tree-level agent_tasks
    tree_agent_tasks = tree_dir / "agent_tasks"
    assert tree_agent_tasks.exists(), "tree-level agent_tasks not found"
    print("  ✓ agent_tasks/ (tree-level)")
    
    print("\n✓ All structure checks passed!")
    return True


def main():
    """Test the new tree structure."""
    
    # Test data
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    output_dir = Path("test_outputs") / "tree_structure"
    
    print("="*60)
    print("Testing New Analysis Tree Structure")
    print("="*60)
    print(f"Input: {input_data}")
    print(f"Output: {output_dir}")
    
    # Create components
    tree_manager = AnalysisTreeManager()
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create tree
    print("\n1. Creating analysis tree...")
    tree = tree_manager.create_tree(
        user_request="Test tree structure",
        input_data_path=str(input_data),
        max_nodes=1,
        generation_mode=GenerationMode.ONLY_NEW
    )
    print(f"   Tree ID: {tree.id}")
    
    # Add a simple node
    print("\n2. Adding test node...")
    block = create_simple_block()
    node = tree_manager.add_root_node(block)
    print(f"   Node ID: {node.id}")
    
    # Create tree directory and save tree files
    tree_dir = output_dir / f"tree_{tree.id}"
    tree_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tree metadata
    tree_metadata = {
        "id": tree.id,
        "user_request": tree.user_request,
        "input_data_path": tree.input_data_path,
        "created_at": tree.created_at.isoformat() if tree.created_at else None,
        "generation_mode": tree.generation_mode.value if hasattr(tree.generation_mode, 'value') else str(tree.generation_mode),
        "max_nodes": tree.max_nodes
    }
    with open(tree_dir / "tree_metadata.json", 'w') as f:
        json.dump(tree_metadata, f, indent=2)
    
    # Save analysis tree
    tree_manager.save_tree(tree_dir / "analysis_tree.json")
    
    # Create tree-level agent_tasks
    (tree_dir / "agent_tasks").mkdir(exist_ok=True)
    
    # Execute node (this creates the structure)
    print("\n3. Executing node...")
    state, output_path = node_executor.execute_node(
        node=node,
        tree=tree,
        input_path=input_data,
        output_base_dir=output_dir
    )
    
    if state == NodeState.COMPLETED:
        print(f"   ✓ Node executed successfully")
        print(f"   Output: {output_path}")
    else:
        print(f"   ✗ Node execution failed")
    
    # Verify structure
    success = verify_structure(output_dir, tree.id, node.id)
    
    if success:
        print("\n" + "="*60)
        print("✓ TEST PASSED: Structure created correctly!")
        print("="*60)
        return 0
    else:
        print("\n✗ TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())